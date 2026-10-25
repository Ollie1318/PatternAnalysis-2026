"""
Data audit of the real LOBSTER books (feasibility evidence, CLAUDE.md Phase 1).

For each ticker on the regular clock-time grid, computes and saves:
    - spread in ticks, return volatility, kurtosis and share of zero returns
    - ACF of returns and |returns| (heteroskedasticity / volatility clustering)
    - VWAP mean and variance per split (per ticker, staff metric 3)
    - order-book violation rates (expected 0% on real data)
    - real-vs-real references, validation hour vs test hour:
        KL on spread and mid returns, with bin-count and sample-length
        sensitivity; SSIM on depth heatmaps
    - grid-step comparison (1, 2, 5 s): zero returns and |return| ACF

Outputs (in --out-dir): audit_results.json, audit_summary.md and PNG figures.

Usage:
    python data_audit.py --data-dir data [--out-dir figures/audit]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import utils
from dataset import TICKERS, load_ticker, resample_to_grid, split_by_time

# Plot styling, shared with the other project figures.
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#e8762b", "#1f9e74", "#c2417f", "#7a5bd3"]  # one per ticker

ACF_LAGS = 300
KL_BIN_COUNTS = (10, 25, 50, 100, 200)
KL_LENGTHS = (250, 500, 1000, 2000, 3600)
SSIM_SEGMENT = 500
STEPS = (1.0, 2.0, 5.0)


def style(ax, title=None):
    """Apply the shared recessive axis style; optionally set a left title."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=TEXT, fontweight="bold")


def audit_ticker(grid):
    """
    Compute every audit statistic for one ticker's gridded book.

    Args:
        grid (pd.DataFrame): output of resample_to_grid for one ticker.

    Returns:
        tuple: (results dict of JSON-serialisable values, arrays dict kept for plotting).
    """
    ask, bid, ask_size, bid_size = utils.book_arrays(grid)
    mid = utils.mid_price(ask, bid)
    r = utils.log_returns(mid)
    spread = utils.spread_ticks(ask, bid)
    half_ticks = np.rint(np.diff(mid) / (utils.TICK / 2)).astype(int)  # mid change, exact
    acf_r, acf_abs = utils.acf(r, ACF_LAGS), utils.acf(np.abs(r), ACF_LAGS)

    res = {
        "steps": len(grid),
        "spread_ticks": {
            "mean": float(spread.mean()),
            "median": float(np.median(spread)),
            "share_1_tick": float((spread == 1).mean()),
        },
        "returns": {
            "std": float(r.std()),
            "excess_kurtosis": utils.excess_kurtosis(r),
            "share_zero": float((r == 0).mean()),
            "acf_r_lag1": float(acf_r[0]),
            "acf_abs_r": {str(k): float(acf_abs[k - 1]) for k in (1, 10, 60, 300)},
        },
        "violations": utils.violation_rates(ask, bid, ask_size, bid_size),
    }

    # Per-split VWAP statistics and arrays for the real-vs-real references.
    splits = split_by_time(grid)
    res["vwap"], parts = {}, {}
    for name, part in splits.items():
        a, b, va, vb = utils.book_arrays(part)
        v = utils.vwap(a, b, va, vb)["book"]
        m = utils.mid_price(a, b)
        res["vwap"][name] = {
            "mean": float(v.mean()),
            "var": float(v.var()),
            "mean_minus_mid_ticks": float(((v - m) / utils.TICK).mean()),
        }
        parts[name] = {
            "spread": utils.spread_ticks(a, b),
            "half_ticks": np.rint(np.diff(m) / (utils.TICK / 2)).astype(int),
            "r": utils.log_returns(m),
            "heatmap": utils.depth_heatmap(va, vb),
        }

    res["real_vs_real"] = real_vs_real(parts["test"], parts["val"])
    arrays = {"acf_r": acf_r, "acf_abs": acf_abs, "spread": spread, "half_ticks": half_ticks, "parts": parts}
    return res, arrays


def real_vs_real(p, q):
    """
    Reference metric values between two real hours (P = test, Q = validation).

    A generator cannot be expected to beat these: they measure how much the
    real market itself differs from one hour to the next.

    Args:
        p, q (dict): per-split arrays built in audit_ticker.

    Returns:
        dict of KL values (tick-aligned, bin-count and length sensitivity) and SSIM.
    """
    out = {
        "kl_spread_ticks": utils.kl_divergence(p["spread"], q["spread"], utils.integer_bins(p["spread"], q["spread"])),
        "kl_mid_change_half_ticks": utils.kl_divergence(
            p["half_ticks"], q["half_ticks"], utils.integer_bins(p["half_ticks"], q["half_ticks"])
        ),
    }

    # Log-return KL on equal-width bins over the pooled range: bin sensitivity.
    lo, hi = min(p["r"].min(), q["r"].min()), max(p["r"].max(), q["r"].max())
    out["kl_returns_by_bins"] = {
        str(n): utils.kl_divergence(p["r"], q["r"], np.linspace(lo, hi, n + 1)) for n in KL_BIN_COUNTS
    }
    # Length sensitivity: first n steps of the test hour vs the full validation hour.
    bins = utils.integer_bins(p["half_ticks"], q["half_ticks"])
    out["kl_mid_change_by_length"] = {
        str(n): utils.kl_divergence(p["half_ticks"][:n], q["half_ticks"], bins) for n in KL_LENGTHS
    }

    # SSIM: whole hour, and aligned 500-step segments (~staff generation length).
    ha, hb = p["heatmap"], q["heatmap"]
    rng = max(ha.max(), hb.max()) - min(ha.min(), hb.min())
    segs = [
        utils.ssim(ha[:, s : s + SSIM_SEGMENT], hb[:, s : s + SSIM_SEGMENT], data_range=rng)
        for s in range(0, ha.shape[1] - SSIM_SEGMENT + 1, SSIM_SEGMENT)
    ]
    out["ssim_depth_full_hour"] = utils.ssim(ha, hb, data_range=rng)
    out["ssim_depth_500_step_mean"] = float(np.mean(segs))
    return out


def step_comparison(events):
    """
    Share of zero mid returns and |return| ACF at several grid steps.

    The |return| ACF is read at lag 1 and at the lag closest to 60 s, so the
    steps are compared over the same clock time.

    Args:
        events (pd.DataFrame): output of load_ticker.

    Returns:
        dict step -> {"share_zero", "acf_abs_lag1", "acf_abs_60s"}.
    """
    out = {}
    for step in STEPS:
        ask, bid, *_ = utils.book_arrays(resample_to_grid(events, step=step))
        r = utils.log_returns(utils.mid_price(ask, bid))
        a = utils.acf(np.abs(r), int(60 / step))
        out[f"{step:g}s"] = {
            "share_zero": float((r == 0).mean()),
            "acf_abs_lag1": float(a[0]),
            "acf_abs_60s": float(a[-1]),
        }
    return out


def plot_acf(arrays, path):
    """Small multiples: ACF of returns and |returns| per ticker."""
    fig, axes = plt.subplots(1, len(arrays), figsize=(15, 3), sharey=True, facecolor=SURFACE)
    lags = np.arange(1, ACF_LAGS + 1)
    for ax, (t, arr) in zip(axes, arrays.items()):
        style(ax, t)
        ax.plot(lags, arr["acf_abs"], color=SERIES[0], linewidth=1.5, label="|returns|")
        ax.plot(lags, arr["acf_r"], color=SERIES[1], linewidth=1.5, label="returns")
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.set_xlabel("Lag (s)", color=MUTED, fontsize=8)
    axes[0].set_ylabel("Autocorrelation", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.suptitle("ACF of 1 s mid log-returns vs |returns|", x=0.01, ha="left", color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_distributions(arrays, path):
    """Two rows of small multiples: spread in ticks, and mid changes in half-ticks (log y)."""
    fig, axes = plt.subplots(2, len(arrays), figsize=(15, 5.5), facecolor=SURFACE)
    for col, (t, arr) in enumerate(arrays.items()):
        for row, (key, label) in enumerate([("spread", "Spread (ticks)"), ("half_ticks", "Mid change (half-ticks)")]):
            ax = axes[row, col]
            style(ax, t if row == 0 else None)
            vals, counts = np.unique(arr[key], return_counts=True)
            keep = (vals >= -20) & (vals <= 40) if key == "spread" else np.abs(vals) <= 10
            ax.bar(vals[keep], counts[keep] / counts.sum(), width=0.8, color=SERIES[0])
            ax.set_xlabel(label, color=MUTED, fontsize=8)
            if key == "half_ticks":
                ax.set_yscale("log")
    axes[0, 0].set_ylabel("Share of steps", color=MUTED, fontsize=9)
    axes[1, 0].set_ylabel("Share of steps (log)", color=MUTED, fontsize=9)
    fig.suptitle("Spread and mid-price change distributions (1 s grid)", x=0.01, ha="left", color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_kl_sensitivity(results, path):
    """KL (test vs val) against bin count and against test sample length, one line per ticker."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
    style(ax1, "Log-return KL vs number of bins")
    style(ax2, "Mid-change KL vs test steps used")
    for colour, (t, res) in zip(SERIES, results.items()):
        rr = res["real_vs_real"]
        ax1.plot(KL_BIN_COUNTS, list(rr["kl_returns_by_bins"].values()), "o-", color=colour, label=t, markersize=5, linewidth=2)
        ax2.plot(KL_LENGTHS, list(rr["kl_mid_change_by_length"].values()), "o-", color=colour, label=t, markersize=5, linewidth=2)
    for ax, xl, ticks in ((ax1, "Bins", KL_BIN_COUNTS), (ax2, "Test-hour steps", KL_LENGTHS)):
        ax.set_xscale("log")
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.axhline(0.1, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(0.01, 0.1, "spec target 0.1", transform=ax.get_yaxis_transform(),
                color=MUTED, fontsize=8, va="bottom")
        ax.set_xlabel(xl, color=MUTED, fontsize=8)
    ax1.set_ylabel("KL(test || val), nats", color=MUTED, fontsize=9)
    ax2.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.suptitle("Real-vs-real KL sensitivity (validation hour vs test hour)", x=0.01, ha="left", color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_heatmaps(arrays, results, path):
    """Depth heatmaps of the first 500 s of the validation and test hours, per ticker."""
    fig, axes = plt.subplots(len(arrays), 2, figsize=(12, 2.2 * len(arrays)), facecolor=SURFACE)
    for row, (t, arr) in enumerate(arrays.items()):
        imgs = [arr["parts"][s]["heatmap"][:, :SSIM_SEGMENT] for s in ("val", "test")]
        vmin, vmax = min(i.min() for i in imgs), max(i.max() for i in imgs)
        for col, (img, name) in enumerate(zip(imgs, ("validation 14:00", "test 15:00"))):
            ax = axes[row, col]
            ax.imshow(img, aspect="auto", origin="lower", cmap="Blues", vmin=vmin, vmax=vmax)
            ax.axhline(9.5, color=MUTED, linewidth=0.6)
            ax.set_title(f"{t}  {name}", loc="left", fontsize=9, color=TEXT)
            ax.set_yticks([0, 9, 10, 19], ["bid 10", "bid 1", "ask 1", "ask 10"])
            ax.tick_params(colors=MUTED, labelsize=7, length=0)
        ssim_seg = results[t]["real_vs_real"]["ssim_depth_500_step_mean"]
        axes[row, 1].text(1.01, 0.5, f"SSIM\n{ssim_seg:.2f}", transform=axes[row, 1].transAxes, color=TEXT, fontsize=9, va="center")
    axes[-1, 0].set_xlabel("Seconds into hour", color=MUTED, fontsize=8)
    axes[-1, 1].set_xlabel("Seconds into hour", color=MUTED, fontsize=8)
    fig.suptitle("Depth heatmaps, log(1 + size) by level: real vs real", x=0.01, ha="left", color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=80, facecolor=SURFACE)
    plt.close(fig)


def summary_markdown(results, steps):
    """Render the key numbers as GitHub-markdown tables for the README."""
    lines = ["# Data audit summary (1 s grid, 2012-06-21)", "", "## Per-ticker statistics", ""]
    lines += [
        "| Ticker | Spread mean (ticks) | Spread = 1 tick | Return std | Excess kurtosis | Zero returns | ACF r lag 1 | ACF \\|r\\| lag 1 / 60 / 300 | Violations |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t, r in results.items():
        a = r["returns"]["acf_abs_r"]
        lines.append(
            f"| {t} | {r['spread_ticks']['mean']:.2f} | {r['spread_ticks']['share_1_tick']:.1%} | "
            f"{r['returns']['std']:.2e} | {r['returns']['excess_kurtosis']:.1f} | {r['returns']['share_zero']:.1%} | "
            f"{r['returns']['acf_r_lag1']:+.3f} | {a['1']:.3f} / {a['60']:.3f} / {a['300']:.3f} | {r['violations']['any']:.1%} |"
        )
    lines += ["", "## Book VWAP per split", "", "| Ticker | Split | Mean ($) | Variance ($^2) | VWAP - mid (ticks) |", "|---|---|---|---|---|"]
    for t, r in results.items():
        for s, v in r["vwap"].items():
            lines.append(f"| {t} | {s} | {v['mean']:.3f} | {v['var']:.4f} | {v['mean_minus_mid_ticks']:+.2f} |")
    lines += [
        "", "## Real-vs-real references (KL(test || val), SSIM test vs val)", "",
        "| Ticker | KL spread | KL mid change | KL returns, bins " + " / ".join(map(str, KL_BIN_COUNTS))
        + " | SSIM full hour | SSIM 500-step mean |",
        "|---|---|---|---|---|---|",
    ]
    for t, r in results.items():
        rr = r["real_vs_real"]
        kb = " / ".join(f"{v:.3f}" for v in rr["kl_returns_by_bins"].values())
        lines.append(
            f"| {t} | {rr['kl_spread_ticks']:.3f} | {rr['kl_mid_change_half_ticks']:.3f} | {kb} | "
            f"{rr['ssim_depth_full_hour']:.3f} | {rr['ssim_depth_500_step_mean']:.3f} |"
        )
    lines += ["", "## Grid step comparison", "", "| Ticker | Step | Zero returns | ACF \\|r\\| lag 1 | ACF \\|r\\| at 60 s |", "|---|---|---|---|---|"]
    for t, s in steps.items():
        for step, v in s.items():
            lines.append(f"| {t} | {step} | {v['share_zero']:.1%} | {v['acf_abs_lag1']:.3f} | {v['acf_abs_60s']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    """Run the audit for every ticker and save results and figures."""
    parser = argparse.ArgumentParser(description="Audit the real LOBSTER books.")
    parser.add_argument("--data-dir", required=True, help="folder holding the LOBSTER CSVs")
    parser.add_argument("--out-dir", default=os.path.join("figures", "audit"), help="where to save outputs")
    parser.add_argument("--step", type=float, default=1.0, help="grid step in seconds")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results, arrays, steps = {}, {}, {}
    for ticker in TICKERS:
        events = load_ticker(args.data_dir, ticker)
        results[ticker], arrays[ticker] = audit_ticker(resample_to_grid(events, step=args.step))
        steps[ticker] = step_comparison(events)
        print(f"audited {ticker}")

    with open(os.path.join(args.out_dir, "audit_results.json"), "w") as f:
        json.dump({"step": args.step, "tickers": results, "step_comparison": steps}, f, indent=2)
    summary = summary_markdown(results, steps)
    with open(os.path.join(args.out_dir, "audit_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)

    plot_acf(arrays, os.path.join(args.out_dir, "acf_returns.png"))
    plot_distributions(arrays, os.path.join(args.out_dir, "spread_and_returns.png"))
    plot_kl_sensitivity(results, os.path.join(args.out_dir, "kl_sensitivity.png"))
    plot_heatmaps(arrays, results, os.path.join(args.out_dir, "depth_heatmaps.png"))
    print(summary)
    print(f"saved results and figures to {args.out_dir}")


if __name__ == "__main__":
    main()
