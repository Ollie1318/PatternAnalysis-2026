"""
Benchmark a trained generator (TimeGAN or the RCGAN baseline) against real books.

For each ticker, the generator produces --n-samples sequences of --seq-len
steps (default 500, longer than the training windows: the GRUs are
length-agnostic). Each sequence is mapped back to raw features with the run's
own min-max statistics and decoded into a full 10-level book
(utils.decode_features), anchored on a real mid price from the evaluation
hour. All metrics are then computed on decoded books with the same utils
functions used in the data audit.

Every metric is reported next to its real-vs-real reference: the same metric
between the real validation hour and the real test hour, cut into the same
number of non-overlapping --seq-len segments. A generator cannot be expected
to beat that reference (see the data audit).

Metrics per ticker (CLAUDE.md "Evaluation"):
    violations        % crossed/locked, ladder, negative-size and any-violation steps
    spread            mean spread in ticks, share of one-tick steps,
                      distance of prices from the tick grid
    returns           std, excess kurtosis, share of zero mid changes
    ACF               of returns (lag 1) and |returns| (lags 1, 10, 60),
                      averaged over sequences
    VWAP              book VWAP mean, within-sequence variance, VWAP - mid in ticks
    KL                KL(real test || generated) on tick-aligned spread and
                      half-tick mid-change bins, on the same number of steps as
                      the real-vs-real reference
    SSIM              depth heatmaps, generated sequence i vs real test segment i
    horizon           violation rate and return std in the first vs last 100 steps
    resources         generation time, sequences per second, peak VRAM

Outputs (in --out-dir, default <run-dir>/eval):
    metrics.json, metrics_summary.md        all metrics; README-ready table
    depth_heatmaps.png                      real vs generated depth surfaces
    acf_abs_returns.png                     volatility clustering
    spread_distribution.png                 spread in ticks
    vwap_paths.png                          VWAP and mid paths, same starting price
    vwap_stats.png                          per-ticker VWAP variance and VWAP - mid
    gifs/depth_<TICKER>.gif                 animated depth chart, real vs generated
                                            (skip with --no-gifs)

Usage:
    python predict.py --run-dir runs/<run-name> --data-dir data
    python predict.py --run-dir runs/<run-name> --data-dir data --n-samples 64 --seq-len 500
    python predict.py --run-dir runs/<run-name> --data-dir data --no-gifs   # faster
"""

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

import utils
from dataset import (
    TICKERS,
    build_datasets,
    feature_columns,
    invert_minmax,
    load_ticker,
    resample_to_grid,
    split_by_time,
)
from modules import Generator, RCGANGenerator, Recovery, Supervisor, sample_noise

# Plot styling, shared with the other project figures.
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
REAL, SYNTH, REF = "#2a78d6", "#e8762b", "#9a9893"
BID, ASK, VWAP_LINE = "#2a78d6", "#e8762b", "#1f9e74"
ACF_LAGS = 60
HORIZON = 100  # steps compared at the start vs the end of each sequence


def parse_args():
    """Command-line arguments."""
    p = argparse.ArgumentParser(description="Generate order books and benchmark them.")
    p.add_argument("--run-dir", required=True, help="training run folder with checkpoint.pt")
    p.add_argument("--data-dir", required=True, help="folder holding the LOBSTER CSVs")
    p.add_argument("--split", default="test", choices=("val", "test"), help="hour to score against")
    p.add_argument("--n-samples", type=int, default=32, help="generated sequences per ticker")
    p.add_argument("--seq-len", type=int, default=500, help="steps per generated sequence")
    p.add_argument("--device", default="auto", help="'auto', 'cuda' or 'cpu'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None, help="default: <run-dir>/eval")
    p.add_argument("--no-gifs", action="store_true", help="skip the animated depth GIFs")
    p.add_argument("--gif-every", type=int, default=5, help="steps between GIF frames")
    return p.parse_args()


def load_generator(run_dir, feature_dim, device):
    """
    Rebuild the trained generator from a run's checkpoint.

    Args:
        run_dir (str): training run folder.
        feature_dim (int): F.
        device (torch.device): where to put the networks.

    Returns:
        tuple: (generate function (z, ticker_ids) -> (B, T, F) scaled
        features in (0, 1), z_dim, the run's config, the checkpoint's
        min-max statistics as ticker -> (min, max) arrays).

    Raises:
        ValueError: if the checkpoint has no trained generator.
    """
    ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location=device, weights_only=True)
    cfg = ckpt["config"]
    if "generator" not in ckpt:
        raise ValueError(f"{run_dir} has no trained generator (TimeGAN runs need phase 3)")
    n = len(TICKERS)
    z_dim = cfg.get("z_dim") or feature_dim
    h, layers, c = cfg["hidden_dim"], cfg["num_layers"], cfg["cond_dim"]

    if cfg["model"] == "rcgan":
        G = RCGANGenerator(z_dim, feature_dim, n, c, h, layers).to(device)
        G.load_state_dict(ckpt["generator"])
        G.eval()
        generate = G
    else:
        latent = cfg.get("latent_dim") or h
        G = Generator(z_dim, n, latent, c, h, layers).to(device)
        S = Supervisor(n, latent, c, h, max(1, layers - 1)).to(device)
        R = Recovery(feature_dim, n, latent, c, h, layers).to(device)
        for net, key in ((G, "generator"), (S, "supervisor"), (R, "recovery")):
            net.load_state_dict(ckpt[key])
            net.eval()

        def generate(z, ids):  # TimeGAN generation path: noise -> G -> S -> R
            return R(S(G(z, ids), ids), ids)

    minmax = {t: (np.asarray(lo), np.asarray(hi)) for t, (lo, hi) in ckpt["minmax"].items()}
    return generate, z_dim, cfg, minmax


def real_segments(grid, seq_len):
    """
    Cut one real hour into non-overlapping segments shaped like generated books.

    Segment i covers grid rows 1 + i*L .. (i+1)*L, and its anchor is the mid at
    row i*L, the step before it, exactly as generated sequences are anchored.

    Args:
        grid (pd.DataFrame): one ticker's grid rows for one split.
        seq_len (int): L.

    Returns:
        tuple: (list of (ask, bid, ask_size, bid_size) tuples, each (L, n),
        np.ndarray of anchor mids).
    """
    ask, bid, va, vb = utils.book_arrays(grid)
    mid = utils.mid_price(ask, bid)
    n_seg = (len(grid) - 1) // seq_len
    segs, anchors = [], []
    for i in range(n_seg):
        rows = slice(1 + i * seq_len, 1 + (i + 1) * seq_len)
        segs.append((ask[rows], bid[rows], va[rows], vb[rows]))
        anchors.append(mid[i * seq_len])
    return segs, np.array(anchors)


def book_stats(books):
    """
    Per-sequence statistics, pooled or averaged over a list of decoded books.

    Args:
        books (list): (ask, bid, ask_size, bid_size) tuples, each (L, n).

    Returns:
        tuple: (stats dict of floats, pooled arrays dict with "spread" and
        "half_ticks" for KL and plots, "acf_abs" mean curve).
    """
    viol, spreads, half_ticks, rets, acf_r, acf_abs = [], [], [], [], [], []
    vwap_mean, vwap_var, vwap_mid, misalign = [], [], [], []
    early, late = [], []
    for ask, bid, va, vb in books:
        v = utils.violation_rates(ask, bid, va, vb)
        viol.append(v)
        mid = utils.mid_price(ask, bid)
        r = utils.log_returns(mid)
        rets.append(r)
        spreads.append(utils.spread_ticks(ask, bid))
        half_ticks.append(np.rint(np.diff(mid) / (utils.TICK / 2)).astype(int))
        acf_r.append(utils.acf(r, ACF_LAGS)[0])
        acf_abs.append(utils.acf(np.abs(r), ACF_LAGS))
        w = utils.vwap(ask, bid, va, vb)["book"]
        vwap_mean.append(w.mean())
        vwap_var.append(w.var())
        vwap_mid.append(((w - mid) / utils.TICK).mean())
        misalign.append(utils.tick_misalignment(np.concatenate([ask, bid], axis=1)))
        early.append((utils.violation_rates(ask[:HORIZON], bid[:HORIZON], va[:HORIZON], vb[:HORIZON])["any"],
                      r[: HORIZON - 1].std()))
        late.append((utils.violation_rates(ask[-HORIZON:], bid[-HORIZON:], va[-HORIZON:], vb[-HORIZON:])["any"],
                     r[-(HORIZON - 1):].std()))

    r_all, sp_all, ht_all = np.concatenate(rets), np.concatenate(spreads), np.concatenate(half_ticks)
    acf_abs_mean = np.mean(acf_abs, axis=0)
    stats = {
        "violations": {k: float(np.mean([v[k] for v in viol])) for k in viol[0]},
        "spread_ticks_mean": float(sp_all.mean()),
        "share_one_tick": float((sp_all == 1).mean()),
        "tick_misalignment": float(np.mean(misalign)),
        "return_std": float(r_all.std()),
        "excess_kurtosis": utils.excess_kurtosis(r_all),
        "share_zero_mid_change": float((ht_all == 0).mean()),
        "acf_r_lag1": float(np.mean(acf_r)),
        "acf_abs_r": {str(k): float(acf_abs_mean[k - 1]) for k in (1, 10, 60)},
        "vwap_mean": float(np.mean(vwap_mean)),
        "vwap_var_within_seq": float(np.mean(vwap_var)),
        "vwap_minus_mid_ticks": float(np.mean(vwap_mid)),
        "horizon": {
            f"violations_first_{HORIZON}": float(np.mean([e[0] for e in early])),
            f"violations_last_{HORIZON}": float(np.mean([e[0] for e in late])),
            f"return_std_first_{HORIZON}": float(np.mean([e[1] for e in early])),
            f"return_std_last_{HORIZON}": float(np.mean([e[1] for e in late])),
        },
    }
    return stats, {"spread": sp_all, "half_ticks": ht_all, "acf_abs": acf_abs_mean}


def kl_pair(p_arrays, q_arrays):
    """KL(P || Q) on tick-aligned spread and half-tick mid-change bins."""
    return {
        "kl_spread": utils.kl_divergence(
            p_arrays["spread"], q_arrays["spread"],
            utils.integer_bins(p_arrays["spread"], q_arrays["spread"]),
        ),
        "kl_mid_change": utils.kl_divergence(
            p_arrays["half_ticks"], q_arrays["half_ticks"],
            utils.integer_bins(p_arrays["half_ticks"], q_arrays["half_ticks"]),
        ),
    }


def mean_ssim(books_a, books_b):
    """Mean SSIM between depth heatmaps of paired books (pairs by index)."""
    values = []
    for (_, _, va_a, vb_a), (_, _, va_b, vb_b) in zip(books_a, books_b):
        ha, hb = utils.depth_heatmap(va_a, vb_a), utils.depth_heatmap(va_b, vb_b)
        rng = max(ha.max(), hb.max()) - min(ha.min(), hb.min())
        values.append(utils.ssim(ha, hb, data_range=rng))
    return float(np.mean(values))


def style(ax, title=None):
    """Shared recessive axis style."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=TEXT, fontweight="bold")


def plot_heatmaps(examples, path):
    """Real test segment vs generated sequence depth heatmaps, one row per ticker."""
    fig, axes = plt.subplots(len(examples), 2, figsize=(12, 2.2 * len(examples)), facecolor=SURFACE,
                             squeeze=False)
    for row, (t, (real, synth)) in enumerate(examples.items()):
        imgs = [utils.depth_heatmap(b[2], b[3]) for b in (real, synth)]
        vmin, vmax = min(i.min() for i in imgs), max(i.max() for i in imgs)
        for col, (img, name) in enumerate(zip(imgs, ("real", "generated"))):
            ax = axes[row, col]
            ax.imshow(img, aspect="auto", origin="lower", cmap="Blues", vmin=vmin, vmax=vmax)
            ax.axhline(9.5, color=MUTED, linewidth=0.6)
            ax.set_title(f"{t}  {name}", loc="left", fontsize=9, color=TEXT)
            ax.set_yticks([0, 9, 10, 19], ["bid 10", "bid 1", "ask 1", "ask 10"])
            ax.tick_params(colors=MUTED, labelsize=7, length=0)
    for ax in axes[-1]:
        ax.set_xlabel("Step (s)", color=MUTED, fontsize=8)
    fig.suptitle("Depth heatmaps, log(1 + size) by level: real vs generated", x=0.01, ha="left", color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=80, facecolor=SURFACE)
    plt.close(fig)


def plot_acf(curves, path):
    """ACF of |returns|, real vs generated, one panel per ticker."""
    fig, axes = plt.subplots(1, len(curves), figsize=(3 * len(curves), 3), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    lags = np.arange(1, ACF_LAGS + 1)
    for ax, (t, (real, synth)) in zip(axes[0], curves.items()):
        style(ax, t)
        ax.plot(lags, real, color=REAL, linewidth=1.5, label="real")
        ax.plot(lags, synth, color=SYNTH, linewidth=1.5, label="generated")
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.set_xlabel("Lag (s)", color=MUTED, fontsize=8)
    axes[0, 0].set_ylabel("ACF of |returns|", color=MUTED, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.suptitle("Volatility clustering: ACF of |returns|, mean over sequences", x=0.01, ha="left",
                 color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_spreads(spreads, path):
    """Spread-in-ticks distributions, real vs generated, one panel per ticker."""
    fig, axes = plt.subplots(1, len(spreads), figsize=(3 * len(spreads), 3), facecolor=SURFACE,
                             squeeze=False)
    for ax, (t, (real, synth)) in zip(axes[0], spreads.items()):
        style(ax, t)
        # Include crossed books (spread <= 0) but cap both tails so outliers cannot hide the shape.
        lo, hi = max(min(real.min(), synth.min(), 0), -20), min(max(real.max(), synth.max()), 60)
        bins = np.arange(lo, hi + 2) - 0.5
        for vals, colour, label in ((real, REAL, "real"), (synth, SYNTH, "generated")):
            ax.hist(vals, bins=bins, density=True, histtype="step", color=colour, linewidth=1.5,
                    label=label)
        ax.set_xlabel("Spread (ticks)", color=MUTED, fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.suptitle("Spread distributions (tick-aligned)", x=0.01, ha="left", color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_vwap_paths(paths, path):
    """
    Book VWAP and mid paths: one real segment vs generated sequences that
    start from the same real mid price, one panel per ticker.

    Args:
        paths (dict): ticker -> (real book, list of generated books), books as
            (ask, bid, ask_size, bid_size) tuples.
        path (str): output file.
    """
    fig, axes = plt.subplots(len(paths), 1, figsize=(10, 2.3 * len(paths)), facecolor=SURFACE,
                             squeeze=False)
    for ax, (t, (real, synths)) in zip(axes[:, 0], paths.items()):
        style(ax, t)
        steps = np.arange(1, len(real[0]) + 1)
        for book, colour, label, alpha in [(real, REAL, "real", 1.0)] + [
            (b, SYNTH, "generated" if i == 0 else None, 0.9 if i == 0 else 0.35)
            for i, b in enumerate(synths)
        ]:
            ax.plot(steps, utils.vwap(*book)["book"], color=colour, linewidth=1.4, alpha=alpha,
                    label=None if label is None else f"{label} VWAP")
            ax.plot(steps, utils.mid_price(book[0], book[1]), color=colour, linewidth=0.8,
                    linestyle=(0, (2, 2)), alpha=alpha, label=None if label is None else f"{label} mid")
        ax.set_ylabel("Price ($)", color=MUTED, fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=TEXT, ncol=4, loc="upper left")
    axes[-1, 0].set_xlabel("Step (s)", color=MUTED, fontsize=8)
    fig.suptitle("Book VWAP (solid) and mid (dashed): real vs generated from the same starting mid",
                 x=0.01, ha="left", color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def plot_vwap_stats(results, path):
    """
    Per-ticker VWAP statistics (staff metric 3): within-sequence VWAP variance
    and mean VWAP - mid distance, for the real test hour, the other real hour
    (reference) and the generated books. Both are level-free checks that
    tickers stay distinct rather than collapsing to one value.
    """
    tickers = list(results["tickers"])
    groups = (("real", "real (scored hour)", REAL), ("real_other_hour", "real (other hour)", REF),
              ("generated", "generated", SYNTH))
    x = np.arange(len(tickers))
    width = 0.27
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.4), facecolor=SURFACE)
    style(ax1, "Within-sequence VWAP variance ($², log scale)")
    style(ax2, "Mean VWAP - mid (ticks)")
    for j, (key, label, colour) in enumerate(groups):
        var = [results["tickers"][t][key]["vwap_var_within_seq"] for t in tickers]
        dist = [results["tickers"][t][key]["vwap_minus_mid_ticks"] for t in tickers]
        ax1.bar(x + (j - 1) * width, var, width, color=colour, label=label)
        ax2.bar(x + (j - 1) * width, dist, width, color=colour, label=label)
    ax1.set_yscale("log")
    ax2.axhline(0, color=MUTED, linewidth=0.6)
    for ax in (ax1, ax2):
        ax.set_xticks(x, tickers)
    ax1.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def make_depth_gif(real, synth, ticker, path, every=5):
    """
    Animated depth chart, real vs generated side by side (CLAUDE.md
    "Visualisation"): at each frame, bars of bid and ask volume at each
    price level, with vertical lines for the mid and the book VWAP.

    Both panels start from the same real mid. Axis limits are fixed for the
    whole animation (prices: 1st-99th percentile of each book's levels;
    volume: 99th percentile of both), so movement in the chart is movement in
    the book, not in the axes.

    Args:
        real, synth (tuple): (ask, bid, ask_size, bid_size) books, each (L, n).
        ticker (str): for the title.
        path (str): output .gif file.
        every (int): steps between frames (500 steps / 5 = 100 frames).
    """
    books = (("real", real), ("generated", synth))
    frames = range(0, len(real[0]), every)
    vmax = np.percentile(np.concatenate([np.concatenate([b[2], b[3]]) for _, b in books]), 99) * 1.1
    xlims = []
    for _, (ask, bid, _, _) in books:
        lo, hi = np.percentile(bid, 1), np.percentile(ask, 99)
        pad = 0.05 * (hi - lo) + 1e-9
        xlims.append((lo - pad, hi + pad))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3), facecolor=SURFACE)

    def draw(k):
        for ax, (name, (ask, bid, va, vb)), xlim in zip(axes, books, xlims):
            ax.clear()
            style(ax, f"{ticker} {name}  step {k + 1}")
            width = (xlim[1] - xlim[0]) / 70
            ax.bar(bid[k], vb[k], width=width, color=BID, label="bid volume")
            ax.bar(ask[k], va[k], width=width, color=ASK, label="ask volume")
            ax.axvline(utils.mid_price(ask[k : k + 1], bid[k : k + 1])[0], color=TEXT, linewidth=1,
                       label="mid")
            ax.axvline(utils.vwap(ask[k : k + 1], bid[k : k + 1], va[k : k + 1], vb[k : k + 1])["book"][0],
                       color=VWAP_LINE, linewidth=1, linestyle=(0, (3, 2)), label="book VWAP")
            ax.set_xlim(xlim)
            ax.set_ylim(0, vmax)
            ax.set_xlabel("Price ($)", color=MUTED, fontsize=8)
        axes[0].set_ylabel("Volume (shares)", color=MUTED, fontsize=8)
        axes[1].legend(frameon=False, fontsize=7, labelcolor=TEXT, loc="upper right")

    anim = FuncAnimation(fig, draw, frames=frames)
    anim.save(path, writer=PillowWriter(fps=10), dpi=60, savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)


def summary_markdown(results, cfg, args):
    """Key metrics as a GitHub-markdown table: generated vs real-vs-real reference."""
    lines = [
        f"# Benchmark: {cfg['model']} ({os.path.basename(os.path.normpath(args.run_dir))})",
        "",
        f"{args.n_samples} generated sequences x {args.seq_len} steps per ticker, scored against the "
        f"real {args.split} hour. Reference = real validation hour vs real test hour.",
        "",
        "| Ticker | Violations (any) | Spread ticks gen / real | One-tick share gen / real | "
        "Zero mid change gen / real | ACF \\|r\\| lag 1 gen / real | VWAP-mid ticks gen / real | "
        "KL spread gen (ref) | KL mid change gen (ref) | SSIM gen (ref) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t, r in results["tickers"].items():
        g, re, ref = r["generated"], r["real"], r["reference"]
        lines.append(
            f"| {t} | {g['violations']['any']:.1%} | {g['spread_ticks_mean']:.2f} / {re['spread_ticks_mean']:.2f} | "
            f"{g['share_one_tick']:.1%} / {re['share_one_tick']:.1%} | "
            f"{g['share_zero_mid_change']:.1%} / {re['share_zero_mid_change']:.1%} | "
            f"{g['acf_abs_r']['1']:.3f} / {re['acf_abs_r']['1']:.3f} | "
            f"{g['vwap_minus_mid_ticks']:+.2f} / {re['vwap_minus_mid_ticks']:+.2f} | "
            f"{r['kl']['kl_spread']:.3f} ({ref['kl_spread']:.3f}) | "
            f"{r['kl']['kl_mid_change']:.3f} ({ref['kl_mid_change']:.3f}) | "
            f"{r['ssim']:.3f} ({ref['ssim']:.3f}) |"
        )
    res = results["resources"]
    vram = "n/a (CPU)" if res["peak_vram_mb"] is None else f"{res['peak_vram_mb']:.0f} MB"
    lines += ["", f"Generation on {res['device']}: {res['seconds']:.2f} s for all tickers, "
              f"{res['sequences_per_second']:.1f} sequences/s, peak VRAM {vram}."]
    return "\n".join(lines) + "\n"


def main():
    """Generate books for every ticker, score them, and save metrics and figures."""
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    out_dir = args.out_dir or os.path.join(args.run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    cfg = torch.load(os.path.join(args.run_dir, "checkpoint.pt"), map_location="cpu",
                     weights_only=True)["config"]
    representation = cfg["representation"]
    feature_dim = len(feature_columns(representation))
    generate, z_dim, cfg, minmax = load_generator(args.run_dir, feature_dim, device)

    # The checkpoint's scaling must match the data it is scored against.
    _, stats = build_datasets(args.data_dir, cfg["tickers"], representation, cfg["step"],
                              cfg["scope"], cfg["quantile"], cfg["seq_len"], cfg["stride"])
    for t, (lo, hi) in stats.items():
        if not (np.allclose(lo, minmax[t][0]) and np.allclose(hi, minmax[t][1])):
            raise ValueError(f"min-max statistics for {t} differ from the checkpoint's")

    results = {"run_dir": args.run_dir, "model": cfg["model"], "split": args.split,
               "n_samples": args.n_samples, "seq_len": args.seq_len, "tickers": {}}
    examples, acf_curves, spread_dists, vwap_paths = {}, {}, {}, {}
    gen_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for ticker in cfg["tickers"]:
        tid = TICKERS.index(ticker)
        splits = split_by_time(resample_to_grid(load_ticker(args.data_dir, ticker), step=cfg["step"]))
        real, anchors = real_segments(splits[args.split], args.seq_len)
        ref_other, _ = real_segments(splits["val" if args.split == "test" else "test"], args.seq_len)
        k = min(len(real), len(ref_other))  # segments in one hour (7 for 500 s)

        # Generate in one batch; anchor sequence i on real segment i's starting mid.
        start = time.perf_counter()
        with torch.no_grad():
            z = sample_noise(args.n_samples, args.seq_len, z_dim, cfg["noise"], device=device)
            ids = torch.full((args.n_samples,), tid, dtype=torch.long, device=device)
            scaled = generate(z, ids).cpu().numpy()  # (N, L, F) in (0, 1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gen_seconds += time.perf_counter() - start
        raw = invert_minmax(scaled, *minmax[ticker])
        synth = [utils.decode_features(raw[i], anchors[i % len(anchors)], representation)
                 for i in range(args.n_samples)]

        g_stats, g_arr = book_stats(synth)
        r_stats, r_arr = book_stats(real)
        o_stats, o_arr = book_stats(ref_other)
        _, g_arr_k = book_stats(synth[:k])  # same number of steps as the reference, for KL
        ref = kl_pair(r_arr, o_arr)
        ref["ssim"] = mean_ssim(ref_other[:k], real[:k])
        results["tickers"][ticker] = {
            "generated": g_stats,
            "real": r_stats,
            "real_other_hour": o_stats,
            "kl": kl_pair(r_arr, g_arr_k),
            "ssim": mean_ssim(synth[:k], real[:k]),
            "reference": ref,
            "reference_segments": k,
        }
        examples[ticker] = (real[0], synth[0])  # same starting mid (anchor 0)
        # Up to three generated sequences that start from real segment 0's mid.
        vwap_paths[ticker] = (real[0], synth[0 : args.n_samples : len(anchors)][:3])
        acf_curves[ticker] = (r_arr["acf_abs"], g_arr["acf_abs"])
        spread_dists[ticker] = (r_arr["spread"], g_arr["spread"])
        print(f"{ticker}: violations {g_stats['violations']['any']:.1%}  "
              f"spread {g_stats['spread_ticks_mean']:.2f} (real {r_stats['spread_ticks_mean']:.2f})  "
              f"KL spread {results['tickers'][ticker]['kl']['kl_spread']:.3f} (ref {ref['kl_spread']:.3f})")

    n_seq = args.n_samples * len(cfg["tickers"])
    results["resources"] = {
        "device": str(device),
        "seconds": gen_seconds,
        "sequences_per_second": n_seq / gen_seconds,
        "steps_per_second": n_seq * args.seq_len / gen_seconds,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    summary = summary_markdown(results, cfg, args)
    with open(os.path.join(out_dir, "metrics_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    plot_heatmaps(examples, os.path.join(out_dir, "depth_heatmaps.png"))
    plot_acf(acf_curves, os.path.join(out_dir, "acf_abs_returns.png"))
    plot_spreads(spread_dists, os.path.join(out_dir, "spread_distribution.png"))
    plot_vwap_paths(vwap_paths, os.path.join(out_dir, "vwap_paths.png"))
    plot_vwap_stats(results, os.path.join(out_dir, "vwap_stats.png"))
    if not args.no_gifs:
        gif_dir = os.path.join(out_dir, "gifs")
        os.makedirs(gif_dir, exist_ok=True)
        for ticker, (real_book, synth_book) in examples.items():
            make_depth_gif(real_book, synth_book, ticker, os.path.join(gif_dir, f"depth_{ticker}.gif"),
                           args.gif_every)
            print(f"saved depth GIF for {ticker}")
    print(summary)
    print(f"saved metrics and figures to {out_dir}")


if __name__ == "__main__":
    main()
