"""
Training script for the conditional order-book generators.

Models (--model):
    rcgan    the RCGAN-style baseline (modules.RCGANGenerator /
             RCGANDiscriminator), trained on adversarial (BCE) loss only,
             following Esteban, Hyland & Raetsch (2017).
    timegan  TimeGAN (Yoon, Jarrett & van der Schaar 2019), trained in three
             phases. Current stage: phase 1 only (embedder + recovery
             autoencoder); phases 2 and 3 will be added next and will share
             the RCGAN loop's stabilisation tricks, logging and checkpointing,
             so the comparison is fair.

Runs on GPU when available and falls back to CPU (--device auto), so the same
command works locally for smoke tests and on Rangpur for full runs.

Each run writes to --out-dir/<run-name>/ (gitignored under runs/):
    config.json        every argument, plus device, parameter counts and resources
    losses.csv         every loss at every iteration
    loss_curves.png    loss plot
    checkpoint.pt      model weights, config and scaling statistics

Usage:
    python train.py --data-dir data --iterations 2000
    python train.py --data-dir data --iterations 50 --device cpu   # quick smoke test
    python train.py --data-dir data --model timegan --ae-iterations 10000
"""

import argparse
import csv
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from dataset import TICKERS, build_datasets, make_dataloader
from modules import (
    Embedder,
    RCGANDiscriminator,
    RCGANGenerator,
    Recovery,
    count_parameters,
    sample_noise,
)


def parse_args():
    """Command-line arguments: data, model size, optimisation and output options."""
    p = argparse.ArgumentParser(description="Train the conditional RCGAN baseline or TimeGAN.")
    p.add_argument("--model", default="rcgan", choices=("rcgan", "timegan"))
    # Data
    p.add_argument("--data-dir", required=True, help="folder holding the LOBSTER CSVs")
    p.add_argument("--tickers", nargs="+", default=list(TICKERS))
    p.add_argument(
        "--representation", default="unconstrained", choices=("unconstrained", "constrained")
    )
    p.add_argument("--step", type=float, default=1.0, help="grid step in seconds")
    p.add_argument("--scope", default="global", choices=("global", "per_ticker"))
    p.add_argument("--quantile", type=float, default=0.005, help="robust min-max bound percentile")
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    # Model
    p.add_argument("--hidden-dim", type=int, default=24)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--cond-dim", type=int, default=8, help="ticker embedding size (0 = unconditional)")
    p.add_argument(
        "--latent-dim", type=int, default=None, help="TimeGAN latent size (default: hidden size)"
    )
    p.add_argument(
        "--z-dim", type=int, default=None, help="noise size (default: number of features)"
    )
    p.add_argument("--noise", default="uniform", choices=("uniform", "normal"))
    # Optimisation
    p.add_argument("--iterations", type=int, default=2000, help="generator update rounds")
    p.add_argument(
        "--ae-iterations",
        type=int,
        default=None,
        help="TimeGAN phase 1 (autoencoder) iterations (default: --iterations, as in the reference code)",
    )
    p.add_argument(
        "--val-windows",
        type=int,
        default=2048,
        help="validation windows used to track reconstruction loss",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--g-steps", type=int, default=2, help="generator steps per discriminator step")
    p.add_argument(
        "--d-threshold",
        type=float,
        default=0.15,
        help="only update D when its loss exceeds this (TimeGAN reference code)",
    )
    p.add_argument("--balanced", action="store_true", help="ticker-balanced sampling")
    # Run
    p.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:1' or 'cpu'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None, help="default: <model>_<representation>_seed<seed>")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def resolve_device(name):
    """
    Pick the torch device.

    Args:
        name (str): "auto" (CUDA if available, else CPU) or an explicit device.

    Returns:
        torch.device.

    Raises:
        RuntimeError: if CUDA is requested but not available.
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def set_seed(seed):
    """Seed Python, NumPy and PyTorch (CPU and all GPUs) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infinite_batches(loader):
    """Yield batches forever, starting a new (reshuffled) epoch when one ends."""
    while True:
        yield from loader


def train_rcgan(args, datasets, device):
    """
    Adversarial training of the conditional RCGAN baseline.

    Each iteration:
        1. args.g_steps generator updates with the non-saturating loss
           BCE(D(G(z, c), c), 1), on fresh real batches (for the ticker ids).
        2. One discriminator update with BCE(D(x, c), 1) + BCE(D(G(z, c), c), 0),
           applied only if that loss exceeds args.d_threshold.

    Why the threshold and the 2:1 ratio: the discriminator (bidirectional,
    ~3x the generator's parameters) easily becomes near-perfect, at which
    point its gradients to G vanish and training stalls. Skipping D updates
    while it is already winning, and giving G extra steps, keeps the two
    players balanced. Both tricks come from the TimeGAN reference code and
    will be used identically for TimeGAN so neither model gets extra help.

    Generated sequences use the ticker ids of the real batch, so each batch
    compares real and synthetic data for the same mix of stocks.

    Args:
        args (argparse.Namespace): see parse_args.
        datasets (dict): "train" / "val" / "test" LOBWindowDataset.
        device (torch.device): where to train.

    Returns:
        tuple: (generator, discriminator, list of per-iteration loss dicts,
        resources dict).
    """
    train_set = datasets["train"]
    feature_dim = train_set.windows.shape[-1]
    z_dim = args.z_dim or feature_dim
    n_tickers = len(TICKERS)  # ids follow TICKERS order whatever subset is loaded

    G = RCGANGenerator(
        z_dim, feature_dim, n_tickers, args.cond_dim, args.hidden_dim, args.num_layers
    ).to(device)
    D = RCGANDiscriminator(
        feature_dim, n_tickers, args.cond_dim, args.hidden_dim, args.num_layers
    ).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()  # mean over batch and time steps

    loader = make_dataloader(
        train_set, args.batch_size, shuffle=True, balanced=args.balanced, seed=args.seed
    )
    batches = infinite_batches(loader)

    def next_batch():
        x, ids = next(batches)
        return x.to(device, non_blocking=True), ids.to(device, non_blocking=True)

    def fake_like(x, ids):
        z = sample_noise(x.size(0), x.size(1), z_dim, args.noise, device=device)
        return G(z, ids)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    start = time.perf_counter()

    for it in range(1, args.iterations + 1):
        # 1. Generator steps: make D call the fakes real.
        for _ in range(args.g_steps):
            x, ids = next_batch()
            logits_fake = D(fake_like(x, ids), ids)
            g_loss = bce(logits_fake, torch.ones_like(logits_fake))
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

        # 2. Discriminator step (fakes detached so G is not updated here).
        x, ids = next_batch()
        logits_real = D(x, ids)
        logits_fake = D(fake_like(x, ids).detach(), ids)
        d_loss = bce(logits_real, torch.ones_like(logits_real)) + bce(
            logits_fake, torch.zeros_like(logits_fake)
        )
        d_updated = d_loss.item() > args.d_threshold
        if d_updated:
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

        history.append(
            {
                "iteration": it,
                "g_loss": g_loss.item(),
                "d_loss": d_loss.item(),
                "d_real_acc": (logits_real > 0).float().mean().item(),
                "d_fake_acc": (logits_fake < 0).float().mean().item(),
                "d_updated": int(d_updated),
            }
        )
        if it % args.log_every == 0 or it == 1:
            h = history[-1]
            print(
                f"[{it:>6}/{args.iterations}] g_loss {h['g_loss']:.4f}  d_loss {h['d_loss']:.4f}  "
                f"D acc real/fake {h['d_real_acc']:.2f}/{h['d_fake_acc']:.2f}  "
                f"{time.perf_counter() - start:.1f} s"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    resources = {
        "train_seconds": seconds,
        "seconds_per_iteration": seconds / args.iterations,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "params_generator": count_parameters(G),
        "params_discriminator": count_parameters(D),
        "d_update_rate": float(np.mean([h["d_updated"] for h in history])),
    }
    return G, D, history, resources


def train_embedding(args, datasets, device):
    """
    TimeGAN phase 1: train the embedder E and recovery R as an autoencoder.

    Loss, as in the reference code (timegan.py, E_loss0):
        L_R = 10 * sqrt(MSE(x, R(E(x, c), c)))


    Reconstruction MSE is also tracked on a fixed, evenly spread subset of the
    validation windows (every ticker is represented, since the validation set
    is stored ticker by ticker), and compared with the MSE of predicting each
    feature's validation mean. Reconstruction close to that baseline means the
    latent space is losing most of the information in the books. At the end,
    the same comparison is made per ticker, against each ticker's own mean, so
    a stock the latent space serves badly is not hidden by the pooled average.

    Args:
        args (argparse.Namespace): see parse_args.
        datasets (dict): "train" / "val" / "test" LOBWindowDataset.
        device (torch.device): where to train.

    Returns:
        tuple: (embedder, recovery, list of per-iteration loss dicts,
        resources dict).
    """
    train_set, val_set = datasets["train"], datasets["val"]
    feature_dim = train_set.windows.shape[-1]
    n_tickers = len(TICKERS)  # ids follow TICKERS order whatever subset is loaded
    iterations = args.ae_iterations or args.iterations

    E = Embedder(
        feature_dim, n_tickers, args.latent_dim, args.cond_dim, args.hidden_dim, args.num_layers
    ).to(device)
    R = Recovery(
        feature_dim, n_tickers, args.latent_dim, args.cond_dim, args.hidden_dim, args.num_layers
    ).to(device)
    opt = torch.optim.Adam(list(E.parameters()) + list(R.parameters()), lr=args.lr)
    mse = nn.MSELoss()  # mean over batch, time steps and features

    loader = make_dataloader(
        train_set, args.batch_size, shuffle=True, balanced=args.balanced, seed=args.seed
    )
    batches = infinite_batches(loader)

    # Fixed validation subset and its predict-the-mean baseline.
    pick = torch.linspace(0, len(val_set) - 1, min(args.val_windows, len(val_set))).long()
    val_x = val_set.windows[pick].to(device)
    val_ids = val_set.ticker_ids[pick].to(device)
    val_baseline = ((val_x - val_x.mean(dim=(0, 1))) ** 2).mean().item()

    def val_mse():  # no dropout or batch norm, so no train/eval switch is needed
        with torch.no_grad():
            return mse(R(E(val_x, val_ids), val_ids), val_x).item()

    print(f"phase 1: {iterations} iterations; val predict-the-mean MSE {val_baseline:.5f}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    start = time.perf_counter()

    for it in range(1, iterations + 1):
        x, ids = next(batches)
        x, ids = x.to(device, non_blocking=True), ids.to(device, non_blocking=True)
        recon_mse = mse(R(E(x, ids), ids), x)
        loss = 10 * torch.sqrt(recon_mse)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        logging = it % args.log_every == 0 or it == 1
        history.append(
            {
                "iteration": it,
                "recon_loss": loss.item(),
                "recon_mse": recon_mse.item(),
                # Validation MSE only on logging iterations (blank in losses.csv otherwise).
                "val_recon_mse": val_mse() if logging else None,
            }
        )
        if logging:
            h = history[-1]
            print(
                f"[{it:>6}/{iterations}] recon_mse {h['recon_mse']:.5f}  "
                f"val {h['val_recon_mse']:.5f}  {time.perf_counter() - start:.1f} s"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start

    # Per-ticker reconstruction MSE vs predicting that ticker's own mean.
    by_ticker = {}
    with torch.no_grad():
        val_err = (R(E(val_x, val_ids), val_ids) - val_x) ** 2  # (N, L, F)
    for tid in val_ids.unique().tolist():
        rows = val_ids == tid
        x_t = val_x[rows]
        by_ticker[TICKERS[tid]] = {
            "recon_mse": val_err[rows].mean().item(),
            "predict_mean_mse": ((x_t - x_t.mean(dim=(0, 1))) ** 2).mean().item(),
        }
        print(
            f"  {TICKERS[tid]}: val recon MSE {by_ticker[TICKERS[tid]]['recon_mse']:.5f}  "
            f"(own-mean baseline {by_ticker[TICKERS[tid]]['predict_mean_mse']:.5f})"
        )

    resources = {
        "phase1_seconds": seconds,
        "phase1_seconds_per_iteration": seconds / iterations,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "params_embedder": count_parameters(E),
        "params_recovery": count_parameters(R),
        "val_predict_mean_mse": val_baseline,
        "final_val_recon_mse": val_mse(),
        "final_val_by_ticker": by_ticker,
    }
    return E, R, history, resources


def save_losses(history, run_dir):
    """Write every logged value per iteration to losses.csv."""
    with open(os.path.join(run_dir, "losses.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_losses(history, run_dir, series, title, ylabel, smooth=25, baseline=None):
    """
    Plot losses: dense series raw (faint) with a moving average; sparse series
    (logged only on some iterations, e.g. validation) as connected markers.

    Args:
        history (list[dict]): per-iteration loss dicts from a training function.
        run_dir (str): output folder.
        series (sequence[tuple[str, str]]): (history key, legend label) pairs.
        title (str): plot title.
        ylabel (str): y-axis label.
        smooth (int): moving-average window in iterations.
        baseline (tuple[float, str] | None): optional (value, label) for a
            dashed horizontal reference line.
    """
    colours = ("#2a78d6", "#e8762b", "#1f9e74")
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for (key, label), colour in zip(series, colours):
        rows = [h for h in history if h[key] is not None]
        it = np.array([h["iteration"] for h in rows])
        y = np.array([h[key] for h in rows])
        if len(rows) < len(history):  # sparse series
            ax.plot(it, y, "o-", color=colour, markersize=3, linewidth=1.5, label=label)
            continue
        ax.plot(it, y, color=colour, alpha=0.25, linewidth=0.8)
        if len(y) >= smooth:
            ax.plot(
                it[smooth - 1 :],
                np.convolve(y, np.ones(smooth) / smooth, "valid"),
                color=colour,
                linewidth=2,
                label=f"{label} ({smooth}-it mean)",
            )
    if baseline is not None:
        ax.axhline(baseline[0], color="#52514e", linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(
            0.01,
            baseline[0],
            baseline[1],
            transform=ax.get_yaxis_transform(),
            color="#52514e",
            fontsize=8,
            va="bottom",
        )
    ax.set_xlabel("Iteration", color="#52514e")
    ax.set_ylabel(ylabel, color="#52514e")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    """Build the data, train the chosen model, and save config, losses, plot and checkpoint."""
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_name = args.run_name or f"{args.model}_{args.representation}_seed{args.seed}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(
        f"device: {device}"
        + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
    )

    datasets, stats = build_datasets(
        args.data_dir,
        args.tickers,
        args.representation,
        args.step,
        args.scope,
        args.quantile,
        args.seq_len,
        args.stride,
    )
    print("windows: " + ", ".join(f"{n} {len(d)}" for n, d in datasets.items()))

    if args.model == "rcgan":
        G, D, history, resources = train_rcgan(args, datasets, device)
        weights = {"generator": G.state_dict(), "discriminator": D.state_dict()}
        plot_losses(
            history,
            run_dir,
            [("g_loss", "generator"), ("d_loss", "discriminator")],
            "RCGAN adversarial losses",
            "BCE loss",
        )
    else:
        E, R, history, resources = train_embedding(args, datasets, device)
        weights = {"embedder": E.state_dict(), "recovery": R.state_dict()}
        plot_losses(
            history,
            run_dir,
            [("recon_mse", "train MSE"), ("val_recon_mse", "validation MSE")],
            "TimeGAN phase 1: reconstruction",
            "MSE (scaled features)",
            baseline=(resources["val_predict_mean_mse"], "val predict-the-mean"),
        )
    print("resources: " + json.dumps(resources))

    save_losses(history, run_dir)
    config = {
        **vars(args),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "resources": resources,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    torch.save(
        {
            **weights,
            "config": config,
            # Scaling statistics as plain lists, so predict.py can invert generator output.
            "minmax": {t: [lo.tolist(), hi.tolist()] for t, (lo, hi) in stats.items()},
        },
        os.path.join(run_dir, "checkpoint.pt"),
    )
    print(f"saved run to {run_dir}")


if __name__ == "__main__":
    main()
