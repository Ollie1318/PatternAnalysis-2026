"""
Training script for the conditional order-book generators.

Models (--model):
    rcgan    the RCGAN-style baseline (modules.RCGANGenerator /
             RCGANDiscriminator), trained on adversarial (BCE) loss only,
             following Esteban, Hyland & Raetsch (2017).
    timegan  TimeGAN (Yoon, Jarrett & van der Schaar 2019), trained in three
             phases: (1) embedder + recovery autoencoder, (2) supervisor on
             real embeddings, (3) joint adversarial training of all five
             networks. Phase 3 uses the same 2:1 generator/discriminator
             schedule and D-loss threshold as the RCGAN loop, so the
             comparison is fair.

Runs on GPU when available and falls back to CPU (--device auto), so the same
command works locally for smoke tests and on Rangpur for full runs.

Each run writes to --out-dir/<run-name>/ (gitignored under runs/):
    config.json        every argument, plus device, parameter counts and resources
    losses.csv         every loss at every iteration (RCGAN)
    loss_curves.png    loss plot (RCGAN)
    losses_phase<k>.csv, loss_curves_phase<k>.png
                       the same, one per TimeGAN phase
    checkpoint.pt      model weights, config and scaling statistics

Usage:
    python train.py --data-dir data --iterations 2000
    python train.py --data-dir data --iterations 50 --device cpu   # quick smoke test
    python train.py --data-dir data --model timegan --ae-iterations 10000 \
        --sup-iterations 10000 --iterations 10000
    # phase 3 only, reusing phases 1 and 2 from an earlier run with the same settings
    python train.py --data-dir data --model timegan --init-from runs/<earlier-run> --iterations 10000
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
    Discriminator,
    Embedder,
    Generator,
    RCGANDiscriminator,
    RCGANGenerator,
    Recovery,
    Supervisor,
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
    p.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="adversarial iterations (RCGAN, or TimeGAN phase 3)",
    )
    p.add_argument(
        "--ae-iterations",
        type=int,
        default=None,
        help="TimeGAN phase 1 (autoencoder) iterations (default: --iterations, as in the reference code)",
    )
    p.add_argument(
        "--sup-iterations",
        type=int,
        default=None,
        help="TimeGAN phase 2 (supervisor) iterations (default: --iterations, as in the reference code)",
    )
    p.add_argument(
        "--init-from",
        default=None,
        help="TimeGAN only: run folder whose checkpoint provides trained embedder, recovery "
        "and supervisor; phases 1 and 2 are skipped and training starts at phase 3",
    )
    p.add_argument(
        "--val-windows",
        type=int,
        default=2048,
        help="validation windows used to track the phase 1 and 2 losses",
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
    # TimeGAN phase 3 loss weights (defaults: reference code timegan.py; see train_joint)
    p.add_argument(
        "--gamma", type=float, default=1.0, help="weight of the loss on unsupervised fakes D(G(z))"
    )
    p.add_argument(
        "--sup-weight", type=float, default=100.0, help="weight of sqrt(supervised loss) in G's loss"
    )
    p.add_argument(
        "--moment-weight", type=float, default=100.0, help="weight of the moment loss in G's loss"
    )
    p.add_argument(
        "--e-sup-weight",
        type=float,
        default=0.1,
        help="weight of the supervised loss in the embedder's phase 3 loss",
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


def val_subset(val_set, n_windows, device):
    """
    Fixed, evenly spread subset of the validation windows for loss tracking.

    The validation set is stored ticker by ticker, so taking the first windows
    would cover only the first ticker; evenly spaced indices cover them all.

    Args:
        val_set (LOBWindowDataset): validation windows.
        n_windows (int): how many to keep (capped at the set size).
        device (torch.device): where to put them.

    Returns:
        tuple: (windows (N, L, F), ticker_ids (N,)) on device.
    """
    pick = torch.linspace(0, len(val_set) - 1, min(n_windows, len(val_set))).long()
    return val_set.windows[pick].to(device), val_set.ticker_ids[pick].to(device)


def peak_vram_mb(device):
    """Peak allocated GPU memory since the last reset, in MB (None on CPU)."""
    return torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None


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
        "peak_vram_mb": peak_vram_mb(device),
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
    val_x, val_ids = val_subset(val_set, args.val_windows, device)
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
        "phase1_peak_vram_mb": peak_vram_mb(device),
        "params_embedder": count_parameters(E),
        "params_recovery": count_parameters(R),
        "val_predict_mean_mse": val_baseline,
        "final_val_recon_mse": val_mse(),
        "final_val_by_ticker": by_ticker,
    }
    return E, R, history, resources


def train_supervisor(args, datasets, device, E):
    """
    TimeGAN phase 2: train the supervisor S on real embeddings.

    Loss, as in the reference code (timegan.py, G_loss_S):
        L_S = MSE(h[:, 1:], S(h, c)[:, :-1]),   h = E(x, c)
    i.e. S's output at step t is scored against the real latent at t + 1, so
    S learns the step-to-step transitions of real books (TimeGAN paper,
    Section 4.2). The embedder from phase 1 is frozen here.

    The reference code hands both G's and S's parameters to this phase's
    optimiser, but the loss only involves real embeddings, so G receives no
    gradient and is unchanged; only S is trained here. G is first trained in
    phase 3 (adversarially), where S keeps training on this loss and every
    generated sequence is passed through S.

    Validation loss is compared with a persistence baseline, predicting
    h[:, t + 1] = h[:, t]. Books change little from one second to the next, so
    this "nothing moves" guess is already strong; S has only learned real
    dynamics if it goes clearly below it. At the end both are reported per
    ticker.

    Args:
        args (argparse.Namespace): see parse_args.
        datasets (dict): "train" / "val" / "test" LOBWindowDataset.
        device (torch.device): where to train.
        E (modules.Embedder): trained embedder from phase 1 (not updated).

    Returns:
        tuple: (supervisor, list of per-iteration loss dicts, resources dict).
    """
    n_tickers = len(TICKERS)  # ids follow TICKERS order whatever subset is loaded
    iterations = args.sup_iterations or args.iterations

    # One GRU layer fewer than the other networks, as in the reference code.
    S = Supervisor(
        n_tickers, E.latent_dim, args.cond_dim, args.hidden_dim, max(1, args.num_layers - 1)
    ).to(device)
    opt = torch.optim.Adam(S.parameters(), lr=args.lr)
    mse = nn.MSELoss()  # mean over batch, time steps and latent dimensions
    E.requires_grad_(False)  # frozen: phase 2 only trains S (phase 3 must re-enable it)

    loader = make_dataloader(
        datasets["train"], args.batch_size, shuffle=True, balanced=args.balanced, seed=args.seed
    )
    batches = infinite_batches(loader)

    # Fixed validation subset, embedded once (E no longer changes).
    val_x, val_ids = val_subset(datasets["val"], args.val_windows, device)
    with torch.no_grad():
        val_h = E(val_x, val_ids)  # (N, L, K)
    persistence = mse(val_h[:, :-1], val_h[:, 1:]).item()

    def val_mse():
        with torch.no_grad():
            return mse(S(val_h, val_ids)[:, :-1], val_h[:, 1:]).item()

    print(f"phase 2: {iterations} iterations; val persistence baseline MSE {persistence:.6f}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    start = time.perf_counter()

    for it in range(1, iterations + 1):
        x, ids = next(batches)
        x, ids = x.to(device, non_blocking=True), ids.to(device, non_blocking=True)
        with torch.no_grad():
            h = E(x, ids)  # (B, L, K) real latents
        loss = mse(S(h, ids)[:, :-1], h[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        logging = it % args.log_every == 0 or it == 1
        history.append(
            {
                "iteration": it,
                "sup_mse": loss.item(),
                # Validation MSE only on logging iterations (blank in the CSV otherwise).
                "val_sup_mse": val_mse() if logging else None,
            }
        )
        if logging:
            row = history[-1]  # not "h", which is the latent batch here
            print(
                f"[{it:>6}/{iterations}] sup_mse {row['sup_mse']:.6f}  "
                f"val {row['val_sup_mse']:.6f}  {time.perf_counter() - start:.1f} s"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start

    # Per-ticker supervised MSE vs that ticker's persistence baseline.
    by_ticker = {}
    with torch.no_grad():
        sup_err = (S(val_h, val_ids)[:, :-1] - val_h[:, 1:]) ** 2  # (N, L-1, K)
    pers_err = (val_h[:, :-1] - val_h[:, 1:]) ** 2
    for tid in val_ids.unique().tolist():
        rows = val_ids == tid
        by_ticker[TICKERS[tid]] = {
            "sup_mse": sup_err[rows].mean().item(),
            "persistence_mse": pers_err[rows].mean().item(),
        }
        print(
            f"  {TICKERS[tid]}: val sup MSE {by_ticker[TICKERS[tid]]['sup_mse']:.6f}  "
            f"(persistence {by_ticker[TICKERS[tid]]['persistence_mse']:.6f})"
        )

    resources = {
        "phase2_seconds": seconds,
        "phase2_seconds_per_iteration": seconds / iterations,
        "phase2_peak_vram_mb": peak_vram_mb(device),
        "params_supervisor": count_parameters(S),
        "val_persistence_mse": persistence,
        "final_val_sup_mse": val_mse(),
        "final_val_sup_by_ticker": by_ticker,
    }
    return S, history, resources


def load_phases_1_2(run_dir, args, stats, feature_dim, device):
    """
    Load a trained embedder, recovery and supervisor from an earlier run, so
    phase 3 can start without repeating phases 1 and 2 (--init-from).

    The earlier run must match this one in everything the loaded weights
    depend on: the network sizes (otherwise the weights do not fit) and the
    data preparation (otherwise the networks would see differently scaled
    features without any error being raised). The min-max statistics are
    compared directly as a final check.

    Args:
        run_dir (str): folder containing checkpoint.pt from a TimeGAN run that
            completed at least phase 2.
        args (argparse.Namespace): this run's arguments.
        stats (dict): this run's min-max statistics from build_datasets.
        feature_dim (int): F.
        device (torch.device): where to put the networks.

    Returns:
        tuple: (E, R, S, the earlier run's config dict).

    Raises:
        ValueError: if the checkpoint lacks phase 1/2 weights or does not match
            this run's settings or scaling.
    """
    ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location=device, weights_only=True)
    missing = [k for k in ("embedder", "recovery", "supervisor") if k not in ckpt]
    if missing:
        raise ValueError(f"{run_dir} has no {missing} weights; it must have completed phase 2")

    src = ckpt["config"]
    keys = (
        "hidden_dim", "num_layers", "cond_dim",  # network sizes
        "tickers", "representation", "step", "scope", "quantile",  # data preparation
    )
    mismatched = {k: (src.get(k), getattr(args, k)) for k in keys if src.get(k) != getattr(args, k)}
    # Latent size: None means "same as hidden", so compare the effective sizes.
    src_latent = src.get("latent_dim") or src.get("hidden_dim")
    latent = args.latent_dim or args.hidden_dim
    if src_latent != latent:
        mismatched["latent_dim (effective)"] = (src_latent, latent)
    if mismatched:
        raise ValueError(f"--init-from run differs (earlier, now): {mismatched}")
    for ticker, (lo, hi) in stats.items():
        src_lo, src_hi = (np.asarray(v) for v in ckpt["minmax"][ticker])
        if not (np.allclose(src_lo, lo) and np.allclose(src_hi, hi)):
            raise ValueError(f"min-max statistics for {ticker} differ from the --init-from run")

    n_tickers = len(TICKERS)
    E = Embedder(
        feature_dim, n_tickers, args.latent_dim, args.cond_dim, args.hidden_dim, args.num_layers
    ).to(device)
    R = Recovery(
        feature_dim, n_tickers, args.latent_dim, args.cond_dim, args.hidden_dim, args.num_layers
    ).to(device)
    S = Supervisor(
        n_tickers, E.latent_dim, args.cond_dim, args.hidden_dim, max(1, args.num_layers - 1)
    ).to(device)
    E.load_state_dict(ckpt["embedder"])
    R.load_state_dict(ckpt["recovery"])
    S.load_state_dict(ckpt["supervisor"])
    return E, R, S, src


def moment_loss(x_hat, x):
    """
    TimeGAN moment-matching loss (reference code, G_loss_V = V1 + V2).

    Compares the mean and standard deviation of each (time step, feature)
    across the batch, so the generator matches the first two moments of the
    real feature distribution:
        V1 = mean |std(x_hat) - std(x)|,   V2 = mean |mean(x_hat) - mean(x)|
    with 1e-6 added to the variances before the square root, as in the
    reference code, to keep gradients finite.

    Args:
        x_hat (torch.Tensor): (B, T, F) synthetic features.
        x (torch.Tensor): (B, T, F) real features.

    Returns:
        torch.Tensor, scalar.
    """
    v1 = (
        torch.sqrt(x_hat.var(dim=0, unbiased=False) + 1e-6)
        - torch.sqrt(x.var(dim=0, unbiased=False) + 1e-6)
    ).abs().mean()
    v2 = (x_hat.mean(dim=0) - x.mean(dim=0)).abs().mean()
    return v1 + v2


def train_joint(args, datasets, device, E, R, S):
    """
    TimeGAN phase 3: joint training of all five networks.

    Notation: x real features, c ticker, h = E(x, c) real latents,
    e_hat = G(z, c) raw generated latents, h_hat = S(e_hat, c) supervised
    generated latents, x_hat = R(h_hat, c) generated features.

    Each iteration (reference code, timegan.py, joint training):
        1. args.g_steps times, on a fresh real batch:
           a. generator + supervisor step on
                L_G = BCE(D(h_hat), 1) + gamma * BCE(D(e_hat), 1)
                      + sup_weight * sqrt(L_S) + moment_weight * L_V
              with L_S = MSE(h[:, 1:], S(h)[:, :-1]) the phase 2 loss on
              real latents and L_V = moment_loss(x_hat, x);
           b. embedder + recovery step on the same batch, on
                L_E = 10 * sqrt(MSE(x, R(E(x)))) + e_sup_weight * L_S
              so the latent space keeps reconstructing books while staying
              predictable for the supervisor.
        2. One discriminator step on a fresh batch, on
                L_D = BCE(D(h), 1) + BCE(D(h_hat), 0) + gamma * BCE(D(e_hat), 0)
           applied only if L_D exceeds args.d_threshold.

    Why each term: the adversarial terms make generated latents look real;
    gamma * BCE on e_hat also trains G directly, without going through S;
    the moment term is a cheap check that feature means and spreads match.
    The supervised term L_S is computed on *real* latents h, exactly as in the
    reference code, so it keeps training S (and, in step b, E) to model real
    step-to-step transitions, but sends no gradient to G. G benefits from it
    indirectly: its output always passes through S (h_hat = S(e_hat)) before
    D and R see it, so S's learned dynamics shape every generated sequence. The 2:1 schedule and the
    D threshold stop the discriminator from overpowering the generator, and
    are identical to the RCGAN baseline so neither model gets extra help.

    Loss weights: defaults follow the reference code (gamma 1, 100 * sqrt(L_S),
    100 * L_V, 0.1 * L_S for E). The paper instead writes lambda = 1 and
    eta = 10 as weights on L_S (Section 4.3); all four are arguments.

    Each optimiser only updates its own networks, as in the reference code
    (G_solver: G + S, E_solver: E + R, D_solver: D). Gradients from one loss
    that reach other networks (e.g. L_G into D) are discarded by zeroing every
    network's gradients before each backward pass.

    Args:
        args (argparse.Namespace): see parse_args.
        datasets (dict): "train" / "val" / "test" LOBWindowDataset.
        device (torch.device): where to train.
        E, R (modules.Embedder, modules.Recovery): trained in phase 1.
        S (modules.Supervisor): trained in phase 2.

    Returns:
        tuple: (generator, discriminator, list of per-iteration loss dicts,
        resources dict). E, R and S are updated in place.
    """
    train_set = datasets["train"]
    feature_dim = train_set.windows.shape[-1]
    z_dim = args.z_dim or feature_dim  # reference code: noise size = feature count
    n_tickers = len(TICKERS)  # ids follow TICKERS order whatever subset is loaded

    G = Generator(
        z_dim, n_tickers, E.latent_dim, args.cond_dim, args.hidden_dim, args.num_layers
    ).to(device)
    D = Discriminator(
        n_tickers, E.latent_dim, args.cond_dim, args.hidden_dim, args.num_layers
    ).to(device)
    E.requires_grad_(True)  # frozen during phase 2; trained again here
    nets = (E, R, S, G, D)

    opt_gs = torch.optim.Adam(list(G.parameters()) + list(S.parameters()), lr=args.lr)
    opt_er = torch.optim.Adam(list(E.parameters()) + list(R.parameters()), lr=args.lr)
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()  # mean over batch and time steps
    mse = nn.MSELoss()

    loader = make_dataloader(
        train_set, args.batch_size, shuffle=True, balanced=args.balanced, seed=args.seed
    )
    batches = infinite_batches(loader)

    def next_batch():
        x, ids = next(batches)
        return x.to(device, non_blocking=True), ids.to(device, non_blocking=True)

    def zero_all():
        for net in nets:
            net.zero_grad(set_to_none=True)

    def generate(x, ids):
        """Raw and supervised generated latents for a batch shaped like x."""
        z = sample_noise(x.size(0), x.size(1), z_dim, args.noise, device=device)
        e_hat = G(z, ids)  # (B, T, K)
        return e_hat, S(e_hat, ids)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    start = time.perf_counter()

    for it in range(1, args.iterations + 1):
        # 1. Generator/supervisor and embedder/recovery steps.
        for _ in range(args.g_steps):
            x, ids = next_batch()

            # 1a. G + S.
            h = E(x, ids)
            loss_s = mse(S(h, ids)[:, :-1], h[:, 1:])
            e_hat, h_hat = generate(x, ids)
            x_hat = R(h_hat, ids)
            logits_fake = D(h_hat, ids)
            logits_fake_e = D(e_hat, ids)
            loss_u = bce(logits_fake, torch.ones_like(logits_fake))
            loss_u_e = bce(logits_fake_e, torch.ones_like(logits_fake_e))
            loss_v = moment_loss(x_hat, x)
            g_loss = (
                loss_u
                + args.gamma * loss_u_e
                + args.sup_weight * torch.sqrt(loss_s)
                + args.moment_weight * loss_v
            )
            zero_all()
            g_loss.backward()
            opt_gs.step()

            # 1b. E + R, recomputed after the G/S update (as in the reference code).
            h = E(x, ids)
            recon_mse = mse(R(h, ids), x)
            e_loss = 10 * torch.sqrt(recon_mse) + args.e_sup_weight * mse(
                S(h, ids)[:, :-1], h[:, 1:]
            )
            zero_all()
            e_loss.backward()
            opt_er.step()

        # 2. Discriminator step on a fresh batch; generator outputs detached.
        x, ids = next_batch()
        with torch.no_grad():
            h = E(x, ids)
            e_hat, h_hat = generate(x, ids)
        logits_real = D(h, ids)
        logits_fake = D(h_hat, ids)
        logits_fake_e = D(e_hat, ids)
        d_loss = (
            bce(logits_real, torch.ones_like(logits_real))
            + bce(logits_fake, torch.zeros_like(logits_fake))
            + args.gamma * bce(logits_fake_e, torch.zeros_like(logits_fake_e))
        )
        d_updated = d_loss.item() > args.d_threshold
        if d_updated:
            zero_all()
            d_loss.backward()
            opt_d.step()

        history.append(
            {
                "iteration": it,
                "g_loss": g_loss.item(),
                "g_loss_u": loss_u.item(),
                "g_loss_u_e": loss_u_e.item(),
                "g_loss_s": loss_s.item(),
                "g_loss_v": loss_v.item(),
                "e_recon_mse": recon_mse.item(),
                "d_loss": d_loss.item(),
                "d_real_acc": (logits_real > 0).float().mean().item(),
                "d_fake_acc": (logits_fake < 0).float().mean().item(),
                "d_updated": int(d_updated),
            }
        )
        if it % args.log_every == 0 or it == 1:
            row = history[-1]
            print(
                f"[{it:>6}/{args.iterations}] d {row['d_loss']:.4f}  g_u {row['g_loss_u']:.4f}  "
                f"g_s {row['g_loss_s']:.6f}  g_v {row['g_loss_v']:.4f}  "
                f"recon {row['e_recon_mse']:.5f}  "
                f"D acc real/fake {row['d_real_acc']:.2f}/{row['d_fake_acc']:.2f}  "
                f"{time.perf_counter() - start:.1f} s"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    resources = {
        "phase3_seconds": seconds,
        "phase3_seconds_per_iteration": seconds / args.iterations,
        "phase3_peak_vram_mb": peak_vram_mb(device),
        "params_generator": count_parameters(G),
        "params_discriminator": count_parameters(D),
        "d_update_rate": float(np.mean([row["d_updated"] for row in history])),
    }
    return G, D, history, resources


def save_losses(history, run_dir, filename="losses.csv"):
    """Write every logged value per iteration to run_dir/filename."""
    with open(os.path.join(run_dir, filename), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_losses(
    history, run_dir, series, title, ylabel, smooth=25, baseline=None, filename="loss_curves.png"
):
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
        filename (str): image file name inside run_dir.
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
    fig.savefig(os.path.join(run_dir, filename), dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    """Build the data, train the chosen model, and save config, losses, plot and checkpoint."""
    args = parse_args()
    if args.init_from and args.model != "timegan":
        raise SystemExit("--init-from only applies to --model timegan")
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
        save_losses(history, run_dir)
        plot_losses(
            history,
            run_dir,
            [("g_loss", "generator"), ("d_loss", "discriminator")],
            "RCGAN adversarial losses",
            "BCE loss",
        )
    elif args.init_from:
        # Phases 1 and 2 loaded from an earlier run; their resources are kept
        # under "init_resources" so total training cost can still be reported.
        feature_dim = datasets["train"].windows.shape[-1]
        E, R, S, src = load_phases_1_2(args.init_from, args, stats, feature_dim, device)
        print(f"phases 1 and 2 loaded from {args.init_from}")
        resources = {"init_from": args.init_from, "init_resources": src["resources"]}
    else:
        # Phase 1: embedder + recovery autoencoder.
        E, R, history1, resources = train_embedding(args, datasets, device)
        save_losses(history1, run_dir, "losses_phase1.csv")
        plot_losses(
            history1,
            run_dir,
            [("recon_mse", "train MSE"), ("val_recon_mse", "validation MSE")],
            "TimeGAN phase 1: reconstruction",
            "MSE (scaled features)",
            baseline=(resources["val_predict_mean_mse"], "val predict-the-mean"),
            filename="loss_curves_phase1.png",
        )
        # Phase 2: supervisor on the (frozen) phase 1 embeddings.
        S, history2, resources2 = train_supervisor(args, datasets, device, E)
        resources.update(resources2)
        save_losses(history2, run_dir, "losses_phase2.csv")
        plot_losses(
            history2,
            run_dir,
            [("sup_mse", "train MSE"), ("val_sup_mse", "validation MSE")],
            "TimeGAN phase 2: next-step latent prediction",
            "MSE (latent)",
            baseline=(resources["val_persistence_mse"], "val persistence (next = current)"),
            filename="loss_curves_phase2.png",
        )

    if args.model == "timegan":
        # Phase 3: joint training; E, R and S are updated in place.
        G, D, history3, resources3 = train_joint(args, datasets, device, E, R, S)
        resources.update(resources3)
        save_losses(history3, run_dir, "losses_phase3.csv")
        plot_losses(
            history3,
            run_dir,
            [("g_loss_u", "generator (adversarial)"), ("d_loss", "discriminator")],
            "TimeGAN phase 3: adversarial losses",
            "BCE loss",
            filename="loss_curves_phase3_adversarial.png",
        )
        plot_losses(
            history3,
            run_dir,
            [
                ("g_loss_s", "supervised (G/S)"),
                ("e_recon_mse", "reconstruction (E/R)"),
                ("g_loss_v", "moment (G)"),
            ],
            "TimeGAN phase 3: supervised, reconstruction and moment losses",
            "loss",
            filename="loss_curves_phase3_other.png",
        )
        weights = {
            "embedder": E.state_dict(),
            "recovery": R.state_dict(),
            "supervisor": S.state_dict(),
            "generator": G.state_dict(),
            "discriminator": D.state_dict(),
        }
    print("resources: " + json.dumps(resources))

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
