"""
Evaluation metrics for real and synthetic limit order books.

All metrics take decoded books as plain NumPy arrays, so exactly the same code
scores the real data (data audit, real-vs-real references) and the generated
books (model evaluation):

    ask, bid            (T, n_levels) prices in dollars, level 1 first
    ask_size, bid_size  (T, n_levels) sizes in shares

Metrics follow CLAUDE.md:
    staff-recommended   order-book violations, VWAP, heteroskedasticity (ACF)
    spec                KL divergence (spread, mid returns), SSIM on depth heatmaps

NumPy is allowed here; modules.py stays pure PyTorch.
"""

import numpy as np

TICK = 0.01  # NASDAQ tick size in dollars for these stocks


def book_arrays(grid, n_levels=10):
    """
    Pull price and size matrices out of a gridded book DataFrame.

    Args:
        grid (pd.DataFrame): output of dataset.resample_to_grid (or any frame
            with ask_price_i / bid_price_i / ask_size_i / bid_size_i columns).
        n_levels (int): book depth.

    Returns:
        tuple of four (T, n_levels) float arrays: ask, bid, ask_size, bid_size.
    """
    def cols(name):
        return grid[[f"{name}_{i}" for i in range(1, n_levels + 1)]].to_numpy(dtype=float)

    return cols("ask_price"), cols("bid_price"), cols("ask_size"), cols("bid_size")


def decode_features(features, anchor_mid, representation, n_levels=10):
    """
    Decode one sequence of (unscaled) features back into a full order book.

    Inverse of dataset.compute_features. The mid path is rebuilt from the
    log-returns, starting from a real mid price (the anchor), because the
    features only hold returns, not the price level:
        m_k = anchor_mid * exp(r_1 + ... + r_k)
    Prices are then rebuilt relative to m_k and sizes are exp(log size).

        unconstrained  a_i = m_k (1 + ask_offset_i),  b_i = m_k (1 + bid_offset_i)
                       (nothing forces a valid book: the model must learn it)
        constrained    a_1 = m_k (1 + s / 2), b_1 = m_k (1 - s / 2), s = exp(log_spread);
                       a_{i+1} = a_i + m_k exp(ask_log_gap_i),
                       b_{i+1} = b_i - m_k exp(bid_log_gap_i)
                       (valid by construction)

    Args:
        features (np.ndarray): (T, F) raw features in dataset.feature_columns
            order (after invert_minmax for model output).
        anchor_mid (float): real mid price in dollars at the step before the
            first row.
        representation (str): "unconstrained" or "constrained".
        n_levels (int): book depth.

    Returns:
        tuple of four (T, n_levels) arrays: ask, bid, ask_size, bid_size
        (prices in dollars, sizes in shares), the format every metric uses.

    Raises:
        ValueError: for an unknown representation or a wrong feature count.
    """
    f = np.asarray(features, dtype=float)
    n = n_levels
    expected = {"unconstrained": 1 + 4 * n, "constrained": 4 * n}
    if representation not in expected:
        raise ValueError(f"unknown representation {representation!r}")
    if f.shape[1] != expected[representation]:
        raise ValueError(f"expected {expected[representation]} features, got {f.shape[1]}")

    m = anchor_mid * np.exp(np.cumsum(f[:, 0]))[:, None]  # (T, 1) mid path
    if representation == "unconstrained":
        ask = m * (1 + f[:, 1 : 1 + n])
        bid = m * (1 + f[:, 1 + n : 1 + 2 * n])
        sizes = f[:, 1 + 2 * n :]
    else:
        half_spread = m * np.exp(f[:, 1:2]) / 2  # (T, 1)
        ask_gaps = m * np.exp(f[:, 2 : 1 + n])  # (T, n - 1)
        bid_gaps = m * np.exp(f[:, 1 + n : 2 * n])  # (T, n - 1)
        ask = (m + half_spread) + np.concatenate([np.zeros_like(m), np.cumsum(ask_gaps, 1)], 1)
        bid = (m - half_spread) - np.concatenate([np.zeros_like(m), np.cumsum(bid_gaps, 1)], 1)
        sizes = f[:, 2 * n :]
    return ask, bid, np.exp(sizes[:, :n]), np.exp(sizes[:, n:])


def tick_misalignment(prices, tick=TICK):
    """
    Mean distance of prices from the nearest tick, in ticks (0 to 0.5).

    Real prices always lie exactly on the $0.01 grid (0). Decoded synthetic
    prices are continuous, so this shows how far they sit from valid prices;
    uniformly random prices would give about 0.25.

    Args:
        prices (np.ndarray): any shape, in dollars.
        tick (float): tick size in dollars.

    Returns:
        float.
    """
    x = np.asarray(prices) / tick
    return float(np.abs(x - np.rint(x)).mean())


def mid_price(ask, bid):
    """Mid price (a_1 + b_1) / 2. Args: (T, n) arrays. Returns: (T,)."""
    return (ask[:, 0] + bid[:, 0]) / 2


def spread_ticks(ask, bid, tick=TICK):
    """
    Best-level spread in ticks, rounded to the nearest integer tick.

    Args:
        ask, bid (np.ndarray): (T, n) prices in dollars.
        tick (float): tick size in dollars.

    Returns:
        np.ndarray (T,) of ints. Real books have spread >= 1.
    """
    return np.rint((ask[:, 0] - bid[:, 0]) / tick).astype(int)


def violation_rates(ask, bid, ask_size, bid_size):
    """
    Fraction of snapshots breaking each order-book invariant (staff metric 1).

    Args:
        ask, bid (np.ndarray): (T, n) prices.
        ask_size, bid_size (np.ndarray): (T, n) sizes.

    Returns:
        dict[str, float]:
            crossed       share of steps with bid_1 >= ask_1 (crossed or locked)
            ladder        share of steps with any non-monotonic price ladder
            negative_size share of steps with any size < 0
            any           share of steps with at least one of the above
    """
    crossed = bid[:, 0] >= ask[:, 0]
    ladder = (np.diff(ask, axis=1) <= 0).any(axis=1) | (np.diff(bid, axis=1) >= 0).any(axis=1)
    negative = (ask_size < 0).any(axis=1) | (bid_size < 0).any(axis=1)
    return {
        "crossed": float(crossed.mean()),
        "ladder": float(ladder.mean()),
        "negative_size": float(negative.mean()),
        "any": float((crossed | ladder | negative).mean()),
    }


def vwap(ask, bid, ask_size, bid_size):
    """
    Volume-weighted average price per step, VWAP = sum(P_i V_i) / sum(V_i)
    (staff metrics 2 and 3).

    Args:
        ask, bid (np.ndarray): (T, n) prices.
        ask_size, bid_size (np.ndarray): (T, n) sizes.

    Returns:
        dict[str, np.ndarray]: "ask", "bid" and "book" (both sides pooled),
        each of shape (T,).
    """
    ask_v = (ask * ask_size).sum(1) / ask_size.sum(1)
    bid_v = (bid * bid_size).sum(1) / bid_size.sum(1)
    book_v = ((ask * ask_size).sum(1) + (bid * bid_size).sum(1)) / (
        ask_size.sum(1) + bid_size.sum(1)
    )
    return {"ask": ask_v, "bid": bid_v, "book": book_v}


def log_returns(prices):
    """Step log-returns log(p_k / p_{k-1}). Args: (T,). Returns: (T - 1,)."""
    return np.diff(np.log(prices))


def excess_kurtosis(x):
    """
    Sample excess kurtosis E[(x - mu)^4] / sigma^4 - 3 (0 for a Gaussian).

    Args:
        x (np.ndarray): (T,) values.

    Returns:
        float; large positive values mean heavy tails.
    """
    z = x - x.mean()
    return float((z**4).mean() / (z**2).mean() ** 2 - 3)


def acf(x, max_lag):
    """
    Sample autocorrelation function for lags 1..max_lag (staff metric 4).

    Returns should have ACF ~ 0 (no linear predictability); |returns| should
    have positive, slowly decaying ACF (volatility clustering).

    Args:
        x (np.ndarray): (T,) series.
        max_lag (int): largest lag, < T.

    Returns:
        np.ndarray (max_lag,): rho(1), ..., rho(max_lag).
    """
    z = x - x.mean()
    denom = (z * z).sum()
    return np.array([(z[:-k] * z[k:]).sum() / denom for k in range(1, max_lag + 1)])


def kl_divergence(p_samples, q_samples, bins, eps=1e-6):
    """
    KL(P || Q) between two empirical distributions on shared bins (spec metric).

    Both histograms use the same bin edges; eps is added to every bin and the
    histograms renormalised, so empty bins in Q do not give infinite KL. The
    result depends on the bins and on the sample sizes (staff warning), so
    report its sensitivity to both.

    Args:
        p_samples, q_samples (np.ndarray): 1-D samples ("reference" P, "other" Q).
        bins (np.ndarray): shared bin edges.
        eps (float): additive smoothing per bin.

    Returns:
        float, KL in nats (0 when the histograms match).
    """
    p, _ = np.histogram(p_samples, bins=bins)
    q, _ = np.histogram(q_samples, bins=bins)
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    return float((p * np.log(p / q)).sum())


def integer_bins(*samples):
    """
    Unit-width bin edges centred on integers, covering all given samples.

    Used for tick-aligned KL: spreads in ticks and mid changes in half-ticks
    are integers, so each bin holds exactly one possible value.

    Returns:
        np.ndarray of edges k - 0.5 for k from the minimum to maximum + 1.
    """
    lo = int(min(s.min() for s in samples))
    hi = int(max(s.max() for s in samples))
    return np.arange(lo, hi + 2) - 0.5


def depth_heatmap(ask_size, bid_size):
    """
    Depth heatmap as an image of (price level x time), the spec's "2D depth
    snapshot".

    Rows run from bid level 10 (bottom of the book) up to ask level 10, so the
    mid sits between rows n-1 and n; values are log(1 + size).

    Args:
        ask_size, bid_size (np.ndarray): (T, n) sizes.

    Returns:
        np.ndarray (2n, T) image.
    """
    return np.log1p(np.concatenate([bid_size[:, ::-1], ask_size], axis=1)).T


def ssim(img_a, img_b, window=7, data_range=None):
    """
    Mean structural similarity (Wang et al., 2004) between two equal-size images.

    Uses a uniform window x window filter and the standard constants
    C1 = (0.01 L)^2, C2 = (0.03 L)^2, with L the data range shared by both
    images. 1 means identical; values near 0 mean no structural agreement.

    Args:
        img_a, img_b (np.ndarray): (H, W) images, H and W >= window.
        window (int): side of the square averaging window.
        data_range (float | None): L; defaults to the joint max - min.

    Returns:
        float, mean SSIM over all valid window positions.
    """
    a = img_a.astype(float)
    b = img_b.astype(float)
    if data_range is None:
        data_range = max(a.max(), b.max()) - min(a.min(), b.min())
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    def box(x):  # mean over every window x window patch, via 2-D cumulative sums
        s = np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        w = window
        return (s[w:, w:] - s[:-w, w:] - s[w:, :-w] + s[:-w, :-w]) / (w * w)

    mu_a, mu_b = box(a), box(b)
    var_a = box(a * a) - mu_a**2
    var_b = box(b * b) - mu_b**2
    cov = box(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return float(s.mean())
