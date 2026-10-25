"""
Training script for the conditional order-book generators.

Current stage: the RCGAN-style baseline (modules.RCGANGenerator /
RCGANDiscriminator), trained on adversarial (BCE) loss only, following
Esteban, Hyland & Raetsch (2017). TimeGAN's three-phase schedule will be added
here later and will share the same stabilisation tricks, logging and
checkpointing, so the comparison is fair.

Runs on GPU when available and falls back to CPU (--device auto), so the same
command works locally for smoke tests and on Rangpur for full runs.

Each run writes to --out-dir/<run-name>/ (gitignored under runs/):
    config.json        every argument, plus device, parameter counts and resources
    losses.csv         every loss at every iteration
    loss_curves.png    loss plot
    checkpoint.pt      model weights, optimiser states and scaling statistics

Usage:
    python train.py --data-dir data --iterations 2000
    python train.py --data-dir data --iterations 50 --device cpu   # quick smoke test
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
from modules import RCGANDiscriminator, RCGANGenerator, count_parameters, sample_noise


def parse_args():
    """Command-line arguments: data, model size, optimisation and output options."""
    p = argparse.ArgumentParser(description="Train the conditional RCGAN baseline.")
    # Data
    p.add_argument("--data-dir", required=True, help="folder holding the LOBSTER CSVs")
    p.add_argument("--tickers", nargs="+", default=list(TICKERS))
    p.add_argument("--representation", default="unconstrained", choices=("unconstrained", "constrained"))
    p.add_argument("--step", type=float, default=1.0, help="grid step in seconds")
    p.add_argument("--scope", default="global", choices=("global", "per_ticker"))
    p.add_argument("--quantile", type=float, default=0.005, help="robust min-max bound percentile")
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    # Model
    p.add_argument("--hidden-dim", type=int, default=24)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--cond-dim", type=int, default=8, help="ticker embedding size")
    p.add_argument("--z-dim", type=int, default=None, help="noise size (default: number of features)")
    p.add_argument("--noise", default="uniform", choices=("uniform", "normal"))
    # Optimisation
    p.add_argument("--iterations", type=int, default=2000, help="generator update rounds")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--g-steps", type=int, default=2, help="generator steps per discriminator step")
    p.add_argument("--d-threshold", type=float, default=0.15,
                   help="only update D when its loss exceeds this (TimeGAN reference code)")
    p.add_argument("--balanced", action="store_true", help="ticker-balanced sampling")
    # Run
    p.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:1' or 'cpu'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None, help="default: rcgan_<representation>_seed<seed>")
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

    G = RCGANGenerator(z_dim, feature_dim, n_tickers, args.cond_dim, args.hidden_dim, args.num_layers).to(device)
    D = RCGANDiscriminator(feature_dim, n_tickers, args.cond_dim, args.hidden_dim, args.num_layers).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()  # mean over batch and time steps

    loader = make_dataloader(train_set, args.batch_size, shuffle=True, balanced=args.balanced, seed=args.seed)
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
        d_loss = bce(logits_real, torch.ones_like(logits_real)) + bce(logits_fake, torch.zeros_like(logits_fake))
        d_updated = d_loss.item() > args.d_threshold
        if d_updated:
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

        history.append({
            "iteration": it,
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
            "d_real_acc": (logits_real > 0).float().mean().item(),
            "d_fake_acc": (logits_fake < 0).float().mean().item(),
            "d_updated": int(d_updated),
        })
        if it % args.log_every == 0 or it == 1:
            h = history[-1]
            print(f"[{it:>6}/{args.iterations}] g_loss {h['g_loss']:.4f}  d_loss {h['d_loss']:.4f}  "
                  f"D acc real/fake {h['d_real_acc']:.2f}/{h['d_fake_acc']:.2f}  "
                  f"{time.perf_counter() - start:.1f} s")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    resources = {
        "train_seconds": seconds,
        "seconds_per_iteration": seconds / args.iterations,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "params_generator": count_parameters(G),
        "params_discriminator": count_parameters(D),
        "d_update_rate": float(np.mean([h["d_updated"] for h in history])),
    }
    return G, D, history, resources


def save_losses(history, run_dir):
    """Write every logged value per iteration to losses.csv."""
    with open(os.path.join(run_dir, "losses.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_losses(history, run_dir, smooth=25):
    """
    Plot generator and discriminator losses (raw, faint) with a moving average.

    Args:
        history (list[dict]): per-iteration losses from train_rcgan.
        run_dir (str): output folder.
        smooth (int): moving-average window in iterations.
    """
    it = np.array([h["iteration"] for h in history])
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for key, colour, label in (("g_loss", "#2a78d6", "generator"), ("d_loss", "#e8762b", "discriminator")):
        y = np.array([h[key] for h in history])
        ax.plot(it, y, color=colour, alpha=0.25, linewidth=0.8)
        if len(y) >= smooth:
            ax.plot(it[smooth - 1:], np.convolve(y, np.ones(smooth) / smooth, "valid"),
                    color=colour, linewidth=2, label=f"{label} ({smooth}-it mean)")
    ax.set_xlabel("Iteration", color="#52514e")
    ax.set_ylabel("BCE loss", color="#52514e")
    ax.set_title("RCGAN adversarial losses", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    """Build the data, train the baseline, and save config, losses, plot and checkpoint."""
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_name = args.run_name or f"rcgan_{args.representation}_seed{args.seed}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    datasets, stats = build_datasets(
        args.data_dir, args.tickers, args.representation, args.step,
        args.scope, args.quantile, args.seq_len, args.stride,
    )
    print("windows: " + ", ".join(f"{n} {len(d)}" for n, d in datasets.items()))

    G, D, history, resources = train_rcgan(args, datasets, device)
    print("resources: " + json.dumps(resources))

    save_losses(history, run_dir)
    plot_losses(history, run_dir)
    config = {**vars(args), "model": "rcgan", "device": str(device),
              "torch_version": str(torch.__version__), "resources": resources}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    torch.save({
        "generator": G.state_dict(),
        "discriminator": D.state_dict(),
        "config": config,
        # Scaling statistics as plain lists, so predict.py can invert generator output.
        "minmax": {t: [lo.tolist(), hi.tolist()] for t, (lo, hi) in stats.items()},
    }, os.path.join(run_dir, "checkpoint.pt"))
    print(f"saved run to {run_dir}")


if __name__ == "__main__":
    main()
