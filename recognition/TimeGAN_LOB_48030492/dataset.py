"""
Data loading and preprocessing for LOBSTER Level 10 limit order books.

This module turns the raw LOBSTER sample files for the five tickers (AMZN, AAPL,
GOOG, INTC, MSFT; 2012-06-21) into training windows for the conditional TimeGAN.

Current stages:
    1. load_ticker:       raw event-time book for one ticker, prices in dollars.
    2. resample_to_grid:  regular clock-time grid (default 1 s), last book state
                          at or before each grid time (no look-ahead).
    3. compute_features:  per-step feature vectors for the "unconstrained" or
                          "constrained" representation (see its docstring).
    4. split_by_time:     chronological train / val / test split per ticker.
    5. fit_minmax / apply_minmax / invert_minmax:
                          [0, 1] scaling fitted on the training split only.
    6. make_windows:      fixed-length sliding windows within each split.
    7. build_datasets / LOBWindowDataset / make_dataloader:
                          the full pipeline for all tickers, giving
                          (window, ticker_id) samples with tickers mixed.

Each LOBSTER sample consists of two row-aligned CSV files without headers
(see LOBSTER_SampleFiles_ReadMe.txt):

    *_message_10.csv    (N, 6):  time, event_type, order_id, size, price, direction
    *_orderbook_10.csv  (N, 40): ask_price_i, ask_size_i, bid_price_i, bid_size_i
                                 for levels i = 1..10

Row k of the orderbook file is the state of the book *after* message k.
Prices are stored as dollars x 10,000. Unoccupied levels are filled with the
dummy prices +9999999999 (ask) / -9999999999 (bid) and size 0.

NumPy and pandas are used here; the model code in modules.py stays pure PyTorch.

Data source: LOBSTER sample files, https://lobsterdata.com/info/DataSamples.php
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

TICKERS = ("AMZN", "AAPL", "GOOG", "INTC", "MSFT")
N_LEVELS = 10

# LOBSTER stores prices as integer dollars x 10,000 (e.g. $91.14 -> 911400).
PRICE_SCALE = 10_000

# Placeholder prices LOBSTER writes when fewer than N_LEVELS levels are occupied.
DUMMY_ASK_PRICE = 9_999_999_999
DUMMY_BID_PRICE = -9_999_999_999

MESSAGE_COLUMNS = [
    "time",
    "event_type",
    "order_id",
    "size",
    "price",
    "direction",
]

# Regular NASDAQ session in seconds after midnight: 09:30:00 and 16:00:00.
MARKET_OPEN = 34_200.0
MARKET_CLOSE = 57_600.0

# Chronological splits as (lower, upper] in seconds after midnight:
# train 09:30-14:00, validation 14:00-15:00, test 15:00-16:00.
SPLIT_BOUNDARIES = {
    "train": (MARKET_OPEN, 50_400.0),
    "val": (50_400.0, 54_000.0),
    "test": (54_000.0, MARKET_CLOSE),
}

# Feature encodings of the book (the penalty-loss variant reuses "unconstrained").
REPRESENTATIONS = ("unconstrained", "constrained")


def orderbook_columns(n_levels=N_LEVELS):
    """
    Column names of a LOBSTER orderbook file, in file order.

    Args:
        n_levels (int): number of book levels in the file.

    Returns:
        list[str]: 4 * n_levels names, repeating
            ask_price_i, ask_size_i, bid_price_i, bid_size_i for i = 1..n_levels.
    """
    columns = []
    for i in range(1, n_levels + 1):
        columns += [
            f"ask_price_{i}",
            f"ask_size_{i}",
            f"bid_price_{i}",
            f"bid_size_{i}",
        ]
    return columns


def find_lobster_files(data_dir, ticker, n_levels=N_LEVELS):
    """
    Locate the message and orderbook CSVs for one ticker.

    Expects LOBSTER's naming scheme:
        {TICKER}_{date}_{start}_{end}_message_{LEVEL}.csv
        {TICKER}_{date}_{start}_{end}_orderbook_{LEVEL}.csv

    Args:
        data_dir (str): folder holding the LOBSTER CSVs.
        ticker (str): e.g. "AMZN".
        n_levels (int): book depth in the file name (10 for Level 10 data).

    Returns:
        tuple[str, str]: (message_path, orderbook_path).

    Raises:
        FileNotFoundError: if either file is missing or is ambiguous (more than
            one match, e.g. several trading days for the same ticker).
    """
    paths = []
    for kind in ("message", "orderbook"):
        pattern = os.path.join(data_dir, f"{ticker}_*_{kind}_{n_levels}.csv")
        matches = sorted(glob.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one file matching {pattern}, found {len(matches)}"
            )
        paths.append(matches[0])
    return paths[0], paths[1]


def load_ticker(data_dir, ticker, n_levels=N_LEVELS):
    """
    Load one ticker's raw event-time order book, with prices in dollars.

    Steps:
        1. Read the message file (only time and event_type are kept) and the
           orderbook file, and check they have the same number of rows.
        2. Mark dummy price levels (+/-9999999999) as NaN so they can never be
           mistaken for real prices; their sizes stay 0, as LOBSTER writes them.
        3. Convert prices from dollars x 10,000 to dollars (float64).

    No resampling or filtering happens here: the result is still in event time,
    one row per LOBSTER message.

    Args:
        data_dir (str): folder holding the LOBSTER CSVs.
        ticker (str): e.g. "AMZN".
        n_levels (int): book depth of the files.

    Returns:
        pd.DataFrame of shape (N, 2 + 4 * n_levels), where N is the number of
        messages. Columns:
            time        float64  seconds after midnight (non-decreasing)
            event_type  int64    LOBSTER event type (1-5, 7 = trading halt)
            ask_price_i, bid_price_i  float64  dollars, NaN for empty levels
            ask_size_i,  bid_size_i   int64    shares, 0 for empty levels

    Raises:
        ValueError: if the two files are misaligned, timestamps go backwards,
            or a dummy level has a non-zero size.
    """
    message_path, orderbook_path = find_lobster_files(
        data_dir, ticker, n_levels
    )

    messages = pd.read_csv(
        message_path,
        header=None,
        names=MESSAGE_COLUMNS,
        usecols=["time", "event_type"],
    )
    book = pd.read_csv(
        orderbook_path,
        header=None,
        names=orderbook_columns(n_levels),
        dtype=np.int64,
    )

    if len(messages) != len(book):
        raise ValueError(
            f"{ticker}: message file has {len(messages)} rows but orderbook has {len(book)}"
        )
    if not messages["time"].is_monotonic_increasing:
        raise ValueError(
            f"{ticker}: message timestamps are not in chronological order"
        )

    ask_price_cols = [f"ask_price_{i}" for i in range(1, n_levels + 1)]
    bid_price_cols = [f"bid_price_{i}" for i in range(1, n_levels + 1)]
    ask_size_cols = [f"ask_size_{i}" for i in range(1, n_levels + 1)]
    bid_size_cols = [f"bid_size_{i}" for i in range(1, n_levels + 1)]

    # Find dummy levels on the raw integers, before any float conversion.
    ask_dummy = (
        book[ask_price_cols].to_numpy() == DUMMY_ASK_PRICE
    )  # (N, n_levels)
    bid_dummy = (
        book[bid_price_cols].to_numpy() == DUMMY_BID_PRICE
    )  # (N, n_levels)

    # LOBSTER documents that dummy levels carry size 0; fail loudly if not.
    if (book[ask_size_cols].to_numpy()[ask_dummy] != 0).any() or (
        book[bid_size_cols].to_numpy()[bid_dummy] != 0
    ).any():
        raise ValueError(
            f"{ticker}: found a dummy price level with non-zero size"
        )

    price_cols = ask_price_cols + bid_price_cols
    book[price_cols] = book[price_cols].astype(np.float64) / PRICE_SCALE
    book[ask_price_cols] = book[ask_price_cols].mask(ask_dummy)
    book[bid_price_cols] = book[bid_price_cols].mask(bid_dummy)

    return pd.concat([messages, book], axis=1)


def resample_to_grid(events, step=1.0, start=MARKET_OPEN, end=MARKET_CLOSE):
    """
    Sample an event-time book on a regular clock-time grid.

    Grid times are t_k = start + k * step for k = 1..K, with K = (end - start) / step,
    so the grid covers (start, end] and the first point is one step after start.
    The book at t_k is the state after the last message with time <= t_k
    (searchsorted with side="right", so ties take the last message at that
    timestamp). Only past and present messages are used: no look-ahead.

    If no message arrives in (t_{k-1}, t_k], the previous state is carried
    forward; n_events records how often this happens. Trading halts (none in the
    2012-06-21 samples) would also be carried forward, as LOBSTER duplicates the
    book on halt messages.

    Trimming the auction minutes is done by passing a later start / earlier end.

    Args:
        events (pd.DataFrame): output of load_ticker, shape (N, 2 + 4 * n_levels),
            sorted by time.
        step (float): grid spacing in seconds (e.g. 1, 2 or 5).
        start (float): grid origin in seconds after midnight (excluded).
        end (float): last grid time in seconds after midnight (included);
            end - start must be a whole number of steps.

    Returns:
        pd.DataFrame of shape (K, 2 + 4 * n_levels). Columns:
            time      float64  grid time t_k, seconds after midnight
            n_events  int64    number of messages in (t_{k-1}, t_k]
            ask_price_i, ask_size_i, bid_price_i, bid_size_i  as in load_ticker

    Raises:
        ValueError: if step is not positive, end - start is not a multiple of
            step, or the first grid time comes before the first message (the
            book would be undefined there).
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    n_steps = int(round((end - start) / step))
    if n_steps < 1 or not np.isclose(start + n_steps * step, end):
        raise ValueError(
            f"end - start = {end - start} s is not a whole number of {step} s steps"
        )

    grid = start + step * np.arange(1, n_steps + 1)  # (K,)
    times = events["time"].to_numpy()  # (N,)

    # Index of the last message at or before each grid time; -1 means none yet.
    last_idx = np.searchsorted(times, grid, side="right") - 1  # (K,)
    if last_idx[0] < 0:
        raise ValueError(
            f"first grid time {grid[0]:.3f} s is before the first message at {times[0]:.3f} s"
        )

    # Messages in each interval (t_{k-1}, t_k], with t_0 = start.
    edges = np.concatenate([[start], grid])  # (K + 1,)
    n_events = np.diff(np.searchsorted(times, edges, side="right"))  # (K,)

    book_cols = [c for c in events.columns if c.startswith(("ask_", "bid_"))]
    grid_book = events[book_cols].iloc[last_idx].reset_index(drop=True)
    grid_book.insert(0, "n_events", n_events)
    grid_book.insert(0, "time", grid)
    return grid_book


def level_columns(name, n_levels=N_LEVELS):
    """
    Per-level column names, e.g. level_columns("ask_price", 3)
    -> ["ask_price_1", "ask_price_2", "ask_price_3"].

    Args:
        name (str): column stem.
        n_levels (int): number of levels.

    Returns:
        list[str] of length n_levels.
    """
    return [f"{name}_{i}" for i in range(1, n_levels + 1)]


def feature_columns(representation, n_levels=N_LEVELS):
    """
    Names of the feature columns for a representation, in model-input order.

    unconstrained (1 + 4 * n_levels = 41 features):
        mid_log_return, ask_offset_1..n, bid_offset_1..n,
        ask_log_size_1..n, bid_log_size_1..n
    constrained (4 * n_levels = 40 features):
        mid_log_return, log_spread, ask_log_gap_1..n-1, bid_log_gap_1..n-1,
        ask_log_size_1..n, bid_log_size_1..n

    Args:
        representation (str): one of REPRESENTATIONS.
        n_levels (int): book depth.

    Returns:
        list[str] of feature names.

    Raises:
        ValueError: for an unknown representation.
    """
    if representation == "unconstrained":
        price_cols = level_columns("ask_offset", n_levels) + level_columns(
            "bid_offset", n_levels
        )
    elif representation == "constrained":
        price_cols = (
            ["log_spread"]
            + level_columns("ask_log_gap", n_levels - 1)
            + level_columns("bid_log_gap", n_levels - 1)
        )
    else:
        raise ValueError(
            f"unknown representation {representation!r}, expected one of {REPRESENTATIONS}"
        )
    return (
        ["mid_log_return"]
        + price_cols
        + level_columns("ask_log_size", n_levels)
        + level_columns("bid_log_size", n_levels)
    )


def check_book(grid, n_levels=N_LEVELS):
    """
    Verify a gridded real book is complete and satisfies the order-book invariants.

    Real LOBSTER books should always pass; this guards the feature encoding,
    which takes logs of sizes, spreads and level gaps and so needs them positive.

    Checks:
        - no empty (dummy, NaN) price levels: there is no fill policy yet
        - every size >= 1 share
        - positive spread:       bid_price_1 < ask_price_1
        - monotonic ladders:     ask prices strictly increase with level,
                                 bid prices strictly decrease with level

    Args:
        grid (pd.DataFrame): output of resample_to_grid, shape (K, 2 + 4 * n_levels).
        n_levels (int): book depth.

    Raises:
        ValueError: naming the first check that fails.
    """
    ask = grid[level_columns("ask_price", n_levels)].to_numpy()  # (K, n_levels)
    bid = grid[level_columns("bid_price", n_levels)].to_numpy()  # (K, n_levels)
    sizes = grid[
        level_columns("ask_size", n_levels)
        + level_columns("bid_size", n_levels)
    ].to_numpy()  # (K, 2 * n_levels)

    if np.isnan(ask).any() or np.isnan(bid).any():
        raise ValueError(
            "book has empty (dummy) price levels; no fill policy is defined"
        )
    if (sizes < 1).any():
        raise ValueError("book has a level with size < 1")
    if (ask[:, 0] <= bid[:, 0]).any():
        raise ValueError(
            "book is crossed or locked (bid_price_1 >= ask_price_1)"
        )
    if (np.diff(ask, axis=1) <= 0).any() or (np.diff(bid, axis=1) >= 0).any():
        raise ValueError("book has a non-monotonic price ladder")


def compute_features(grid, representation="unconstrained", n_levels=N_LEVELS):
    """
    Encode a gridded book as per-step feature vectors.

    Notation at grid step k: ask prices a_i, bid prices b_i, sizes va_i, vb_i
    (level i = 1..n), mid m_k = (a_1 + b_1) / 2.

    Shared features (both representations):
        mid_log_return   r_k = log(m_k / m_{k-1})
                         The spec's log-return transform; scale-free, so
                         returns are comparable across tickers.
        *_log_size_i     log(v_i). Sizes are >= 1 share (check_book), so the log
                         is finite; it compresses the heavy-tailed size
                         distribution. Decoding with exp() can never give a
                         negative size, in either representation.

    unconstrained, price features relative to the mid (dimensionless):
        ask_offset_i     (a_i - m_k) / m_k      > 0 for real books
        bid_offset_i     (b_i - m_k) / m_k      < 0 for real books
        Decoding: a_i = m_k (1 + ask_offset_i). Nothing forces the decoded
        ladder to be monotonic or the spread to be positive, so a model must
        *learn* the invariants (the dilemma being tested).

    constrained, all price features are logs of positive quantities:
        log_spread       log((a_1 - b_1) / m_k)
        ask_log_gap_i    log((a_{i+1} - a_i) / m_k),  i = 1..n-1
        bid_log_gap_i    log((b_i - b_{i+1}) / m_k),  i = 1..n-1
        Decoding: a_1 = m_k (1 + s/2), b_1 = m_k (1 - s/2) with s = exp(log_spread),
        then a_{i+1} = a_i + m_k exp(ask_log_gap_i) and
        b_{i+1} = b_i - m_k exp(bid_log_gap_i). Every exp() is positive, so
        any feature vector decodes to a positive spread and strictly monotonic
        ladders: the invariants hold by construction.

    Prices are divided by the mid so that a $27 stock (INTC) and a $580 stock
    (GOOG) give features on comparable scales; the price level itself only
    enters through the anchor mid used when decoding.

    The first grid step is dropped because r_k needs m_{k-1}. To decode a
    sequence starting at row j, anchor on the real mid of the step before it.

    Args:
        grid (pd.DataFrame): output of resample_to_grid, shape (K, 2 + 4 * n_levels).
        representation (str): one of REPRESENTATIONS.
        n_levels (int): book depth.

    Returns:
        pd.DataFrame of shape (K - 1, 1 + F): a "time" column (grid time of
        step k, seconds after midnight) followed by the F float64 features
        named by feature_columns(representation, n_levels)
        (F = 41 unconstrained, 40 constrained).

    Raises:
        ValueError: for an unknown representation, or if check_book fails.
    """
    columns = feature_columns(representation, n_levels)
    check_book(grid, n_levels)

    ask = grid[level_columns("ask_price", n_levels)].to_numpy()  # (K, n_levels)
    bid = grid[level_columns("bid_price", n_levels)].to_numpy()  # (K, n_levels)
    ask_size = grid[
        level_columns("ask_size", n_levels)
    ].to_numpy()  # (K, n_levels)
    bid_size = grid[
        level_columns("bid_size", n_levels)
    ].to_numpy()  # (K, n_levels)
    mid = (ask[:, 0] + bid[:, 0]) / 2  # (K,)

    mid_log_return = np.diff(np.log(mid))  # (K - 1,), r_k for k = 1..K-1

    # Drop step 0 so every remaining row has a return.
    ask, bid, ask_size, bid_size, mid = (
        x[1:] for x in (ask, bid, ask_size, bid_size, mid)
    )
    m = mid[:, None]  # (K - 1, 1), broadcasts across levels

    if representation == "unconstrained":
        price_features = [(ask - m) / m, (bid - m) / m]  # 2 x (K - 1, n_levels)
    else:
        spread = (ask[:, :1] - bid[:, :1]) / m  # (K - 1, 1)
        ask_gap = np.diff(ask, axis=1) / m  # (K - 1, n_levels - 1)
        bid_gap = -np.diff(bid, axis=1) / m  # (K - 1, n_levels - 1)
        price_features = [np.log(spread), np.log(ask_gap), np.log(bid_gap)]

    values = np.column_stack(
        [mid_log_return, *price_features, np.log(ask_size), np.log(bid_size)]
    )  # (K - 1, F)
    features = pd.DataFrame(values, columns=columns)
    features.insert(0, "time", grid["time"].to_numpy()[1:])
    return features


def split_by_time(features, boundaries=SPLIT_BOUNDARIES):
    """
    Split one ticker's feature rows chronologically into train / val / test.

    A row with grid time t belongs to the split whose interval (lower, upper]
    contains t. With the default boundaries:
        train  (09:30, 14:00]
        val    (14:00, 15:00]
        test   (15:00, 16:00]

    No look-ahead: each row only uses book states at or before its own time
    (the first val row's return uses the 14:00:00 mid, which is in the past).
    Splitting happens *before* windowing and normalisation, so windows cannot
    straddle a boundary and normalisation can be fitted on train only.

    Args:
        features (pd.DataFrame): output of compute_features, with a "time"
            column in seconds after midnight, shape (K - 1, 1 + F).
        boundaries (dict[str, tuple[float, float]]): split name ->
            (lower, upper] in seconds after midnight; intervals must not overlap.

    Returns:
        dict[str, pd.DataFrame]: split name -> rows in that interval, each
        with a fresh 0-based index and the same columns as features.

    Raises:
        ValueError: if the intervals overlap, or a row falls in no split.
    """
    intervals = sorted(boundaries.values())
    for (_, upper), (lower, _) in zip(intervals, intervals[1:]):
        if lower < upper:
            raise ValueError(f"split intervals overlap: {boundaries}")

    time = features["time"]
    splits = {}
    assigned = np.zeros(len(features), dtype=bool)
    for name, (lower, upper) in boundaries.items():
        in_split = ((time > lower) & (time <= upper)).to_numpy()
        splits[name] = features[in_split].reset_index(drop=True)
        assigned |= in_split
    if not assigned.all():
        raise ValueError(
            f"{(~assigned).sum()} rows fall outside every split interval"
        )
    return splits


def fit_minmax(train_features, scope="global", quantile=0.005):
    """
    Fit per-feature min-max scaling statistics on the TRAINING split only.

    Min-max to [0, 1] follows the TimeGAN paper and reference code, whose
    networks end in a sigmoid. Val/test rows are scaled with the same
    statistics and may fall slightly outside [0, 1]; no information from
    those hours leaks into the scaling.

    Robust bounds: instead of the true min/max, the bounds are the
    quantile and 1 - quantile percentiles of the training data (default
    0.5% / 99.5%). With true min/max, a few extreme returns set the range and
    squash typical returns into ~10% of [0, 1] (measured: 98% of scaled AAPL
    returns fell in [0.59, 0.69]). Values beyond the bounds land outside
    [0, 1]; apply_minmax(clip=True) then winsorises them (replaces each with
    the nearest bound, rows are never removed). quantile=0 recovers plain
    min-max.

    Scopes (the README must justify the choice, see CLAUDE.md "Normalisation"):
        global      Option A: one min/max per feature, pooled over all tickers.
                    Stocks keep their real differences (e.g. INTC's larger
                    relative spread), so the model must learn ticker identity
                    from the conditioning.
        per_ticker  Option B: separate min/max per ticker. Every stock fills
                    [0, 1], so conditioning must then be judged on
                    level-free statistics after decoding.

    Args:
        train_features (dict[str, pd.DataFrame]): ticker -> training-split
            output of split_by_time (a "time" column plus F features).
        scope (str): "global" or "per_ticker".
        quantile (float): lower percentile for the bounds, in [0, 0.5).

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: ticker -> (feature_min,
        feature_max), each of shape (F,). Under "global" every ticker maps to
        the same pair.

    Raises:
        ValueError: for an unknown scope or a quantile outside [0, 0.5).
    """
    if not 0 <= quantile < 0.5:
        raise ValueError(f"quantile must be in [0, 0.5), got {quantile}")

    def bounds(v):  # (T, F) -> two (F,) arrays
        return np.quantile(v, quantile, axis=0), np.quantile(
            v, 1 - quantile, axis=0
        )

    values = {
        t: df.drop(columns="time").to_numpy()
        for t, df in train_features.items()
    }
    if scope == "global":
        pooled = np.concatenate(list(values.values()), axis=0)  # (sum T, F)
        stats = bounds(pooled)
        return {t: stats for t in values}
    if scope == "per_ticker":
        return {t: bounds(v) for t, v in values.items()}
    raise ValueError(
        f"unknown scope {scope!r}, expected 'global' or 'per_ticker'"
    )


def apply_minmax(values, feature_min, feature_max, clip=False, eps=1e-12):
    """
    Scale features to [0, 1] using fitted training statistics.

    Args:
        values (np.ndarray): (..., F) raw features.
        feature_min, feature_max (np.ndarray): (F,) from fit_minmax.
        clip (bool): winsorise results to [0, 1] (use for model inputs, which
            a sigmoid-output generator could never reproduce anyway). Metrics
            are computed on decoded real books, not on these clipped values.
        eps (float): guards constant features (max == min) against division by 0.

    Returns:
        np.ndarray of the same shape, (values - min) / (max - min), clipped to
        [0, 1] if clip is True.
    """
    scaled = (values - feature_min) / (feature_max - feature_min + eps)
    return np.clip(scaled, 0.0, 1.0) if clip else scaled


def invert_minmax(scaled, feature_min, feature_max, eps=1e-12):
    """
    Undo apply_minmax: map scaled features back to raw feature units.

    Args:
        scaled (np.ndarray): (..., F) scaled features (e.g. generator output).
        feature_min, feature_max (np.ndarray): (F,) from fit_minmax.
        eps (float): must match the value used in apply_minmax.

    Returns:
        np.ndarray of the same shape in raw feature units.
    """
    return scaled * (feature_max - feature_min + eps) + feature_min


def make_windows(values, seq_len, stride=1):
    """
    Cut one split of one ticker into overlapping fixed-length windows.

    Called separately on each split, so no window can cross a split boundary.
    Stride 1 matches the sliding windows of the TimeGAN reference code
    (data_loading.py), but unlike that code, windows are NOT shuffled across
    periods before splitting: the split is chronological first.

    Args:
        values (np.ndarray): (T, F) scaled features of one split, in time order.
        seq_len (int): window length L (e.g. 100-128 for training).
        stride (int): step between window starts.

    Returns:
        np.ndarray of shape (n_windows, seq_len, F), with
        n_windows = (T - seq_len) // stride + 1. Window j covers rows
        j * stride .. j * stride + seq_len - 1.

    Raises:
        ValueError: if seq_len or stride is not positive, or T < seq_len.
    """
    if seq_len < 1 or stride < 1:
        raise ValueError(
            f"seq_len and stride must be positive, got {seq_len}, {stride}"
        )
    if len(values) < seq_len:
        raise ValueError(
            f"split has {len(values)} rows, fewer than seq_len={seq_len}"
        )
    starts = window_starts(len(values), seq_len, stride)  # (n_windows,)
    return np.stack([values[s : s + seq_len] for s in starts])


def window_starts(n_rows, seq_len, stride=1):
    """
    Row index at which each window of make_windows begins.

    Args:
        n_rows (int): rows T in the split.
        seq_len (int): window length L.
        stride (int): step between window starts.

    Returns:
        np.ndarray of shape (n_windows,): 0, stride, 2 * stride, ... up to T - L.
    """
    return np.arange(0, n_rows - seq_len + 1, stride)


class LOBWindowDataset(Dataset):
    """
    Windows of scaled order-book features, each paired with its ticker id.

    __getitem__ returns (window, ticker_id) only, which is all training needs.
    The per-window metadata below stays on the dataset (same order as the
    windows) for decoding real windows back to books and for analysis.

    Attributes:
        windows (torch.FloatTensor): (n_windows, L, F) features scaled to [0, 1].
        ticker_ids (torch.LongTensor): (n_windows,) index into TICKERS.
        anchor_mids (np.ndarray): (n_windows,) real mid price in dollars at the
            grid step just before each window. Decoding needs it, because the
            features only hold mid log-returns, not the price level.
        start_times (np.ndarray): (n_windows,) grid time of each window's first
            step, seconds after midnight.
    """

    def __init__(self, windows, ticker_ids, anchor_mids, start_times):
        """
        Args:
            windows (np.ndarray): (n_windows, L, F) scaled features.
            ticker_ids (np.ndarray): (n_windows,) integer ticker ids.
            anchor_mids (np.ndarray): (n_windows,) anchor mids in dollars.
            start_times (np.ndarray): (n_windows,) window start times.
        """
        self.windows = torch.as_tensor(windows, dtype=torch.float32)
        self.ticker_ids = torch.as_tensor(ticker_ids, dtype=torch.long)
        self.anchor_mids = np.asarray(anchor_mids, dtype=np.float64)
        self.start_times = np.asarray(start_times, dtype=np.float64)

    def __len__(self):
        """Number of windows."""
        return len(self.windows)

    def __getitem__(self, i):
        """
        Returns:
            tuple: (window (L, F) float32 tensor, ticker_id () long tensor).
        """
        return self.windows[i], self.ticker_ids[i]


def build_datasets(
    data_dir,
    tickers=TICKERS,
    representation="unconstrained",
    step=1.0,
    scope="global",
    quantile=0.005,
    seq_len=24,
    stride=1,
):
    """
    Run the full pipeline (stages 1-6) and return one dataset per split.

    Per ticker: load -> 1 s grid -> features -> chronological split. Then the
    scaling is fitted on the training splits only, every split is scaled with
    clip=True and windowed separately, and all tickers' windows are pooled
    into one dataset per split.

    Args:
        data_dir (str): folder holding the LOBSTER CSVs.
        tickers (sequence[str]): tickers to include; ids follow TICKERS order,
            so a ticker keeps the same id whichever subset is loaded.
        representation (str): one of REPRESENTATIONS.
        step (float): grid step in seconds.
        scope (str): "global" or "per_ticker" min-max scaling.
        quantile (float): robust bound percentile for fit_minmax.
        seq_len (int): window length L.
        stride (int): window stride.

    Returns:
        tuple:
            datasets (dict[str, LOBWindowDataset]): "train" / "val" / "test".
            stats (dict[str, tuple[np.ndarray, np.ndarray]]): ticker ->
                (feature_min, feature_max), needed to invert generator output.
    """
    splits, mids = {}, {}
    for ticker in tickers:
        grid = resample_to_grid(load_ticker(data_dir, ticker), step=step)
        mids[ticker] = (
            grid["time"].to_numpy(),
            ((grid["ask_price_1"] + grid["bid_price_1"]) / 2).to_numpy(),
        )
        splits[ticker] = split_by_time(compute_features(grid, representation))

    stats = fit_minmax(
        {t: s["train"] for t, s in splits.items()},
        scope=scope,
        quantile=quantile,
    )

    datasets = {}
    for name in SPLIT_BOUNDARIES:
        windows, ids, anchors, starts = [], [], [], []
        for ticker in tickers:
            part = splits[ticker][name]
            scaled = apply_minmax(
                part.drop(columns="time").to_numpy(), *stats[ticker], clip=True
            )
            start_rows = window_starts(len(part), seq_len, stride)
            start_times = part["time"].to_numpy()[start_rows]  # (n_windows,)

            # Anchor = mid at the grid step before each window's first step.
            # searchsorted(t - step/2) finds the row of t itself without exact
            # float equality on grid times; the anchor is the row before it.
            grid_time, grid_mid = mids[ticker]
            start_grid_rows = np.searchsorted(grid_time, start_times - step / 2)
            anchors.append(grid_mid[start_grid_rows - 1])

            windows.append(make_windows(scaled, seq_len, stride))
            ids.append(np.full(len(start_rows), TICKERS.index(ticker)))
            starts.append(start_times)
        datasets[name] = LOBWindowDataset(
            np.concatenate(windows),
            np.concatenate(ids),
            np.concatenate(anchors),
            np.concatenate(starts),
        )
    return datasets, stats


def make_dataloader(
    dataset, batch_size=128, shuffle=True, balanced=False, seed=0
):
    """
    Wrap a LOBWindowDataset in a DataLoader that mixes tickers in each batch.

    Args:
        dataset (LOBWindowDataset): windows to serve.
        batch_size (int): windows per batch (128 in the TimeGAN reference code).
        shuffle (bool): shuffle windows each epoch (use for training only).
        balanced (bool): sample tickers equally often, with replacement, via
            WeightedRandomSampler (weight 1 / windows-of-that-ticker). With all
            five tickers on the same grid the counts are equal, so this matters
            only if tickers with different window counts are mixed.
        seed (int): seeds the shuffling / sampling for reproducibility.

    Returns:
        torch.utils.data.DataLoader yielding (windows (B, L, F), ticker_ids (B,)).
    """
    generator = torch.Generator().manual_seed(seed)
    if balanced:
        counts = torch.bincount(dataset.ticker_ids)
        weights = 1.0 / counts[dataset.ticker_ids].double()
        sampler = WeightedRandomSampler(
            weights, len(dataset), replacement=True, generator=generator
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )


def main():
    """Print a short summary of each ticker before and after resampling, as a quick check."""
    parser = argparse.ArgumentParser(
        description="Load and resample LOBSTER Level 10 sample files."
    )
    parser.add_argument(
        "--data-dir", required=True, help="folder holding the LOBSTER CSVs"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(TICKERS),
        help="tickers to load (default: all five)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=1.0,
        help="grid step in seconds (default: 1)",
    )
    parser.add_argument(
        "--representation",
        choices=REPRESENTATIONS,
        default="unconstrained",
        help="feature encoding (default: unconstrained)",
    )
    parser.add_argument(
        "--scope",
        choices=("global", "per_ticker"),
        default="global",
        help="min-max normalisation scope (default: global)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=128, help="window length"
    )
    parser.add_argument("--stride", type=int, default=1, help="window stride")
    args = parser.parse_args()

    all_splits = {}
    for ticker in args.tickers:
        df = load_ticker(args.data_dir, ticker)
        price_cols = [c for c in df.columns if "price" in c]
        n_dummy = int(df[price_cols].isna().to_numpy().sum())
        n_halts = int((df["event_type"] == 7).sum())
        first = df.iloc[0]
        print(
            f"{ticker}: {len(df):>7} events, "
            f"time {df['time'].iloc[0]:.3f}-{df['time'].iloc[-1]:.3f} s, "
            f"dummy price cells {n_dummy}, halts {n_halts}, "
            f"first best bid/ask ${first['bid_price_1']:.2f}/${first['ask_price_1']:.2f}"
        )

        grid = resample_to_grid(df, step=args.step)
        mid = (grid["ask_price_1"] + grid["bid_price_1"]) / 2
        print(
            f"      {args.step:g} s grid: {len(grid)} steps, "
            f"steps with no messages {(grid['n_events'] == 0).mean():.1%}, "
            f"zero mid-price changes {(mid.diff(periods=1).iloc[1:] == 0).mean():.1%}"
        )

        features = compute_features(grid, representation=args.representation)
        values = features.drop(columns="time").to_numpy()
        print(
            f"      {args.representation} features: {values.shape[0]} steps x "
            f"{values.shape[1]} features, all finite {np.isfinite(values).all()}, "
            f"range [{values.min():.3g}, {values.max():.3g}]"
        )

        splits = split_by_time(features)
        print(
            "      splits: "
            + ", ".join(
                f"{name} {len(part)} ({part['time'].iloc[0]:.0f}-{part['time'].iloc[-1]:.0f} s)"
                for name, part in splits.items()
            )
        )
        all_splits[ticker] = splits

    stats = fit_minmax(
        {t: s["train"] for t, s in all_splits.items()}, scope=args.scope
    )
    print(
        f"\n{args.scope} min-max fitted on train; windows of {args.seq_len}, stride {args.stride}:"
    )
    for ticker, splits in all_splits.items():
        parts = []
        for name, part in splits.items():
            raw = part.drop(columns="time").to_numpy()
            scaled = apply_minmax(raw, *stats[ticker])
            clipped = np.mean((scaled < 0) | (scaled > 1))
            windows = make_windows(
                apply_minmax(raw, *stats[ticker], clip=True),
                args.seq_len,
                args.stride,
            )
            parts.append(
                f"{name} {windows.shape} ({clipped:.1%} values clipped)"
            )
        print(f"  {ticker}: " + ", ".join(parts))

    datasets, _ = build_datasets(
        args.data_dir,
        args.tickers,
        args.representation,
        args.step,
        args.scope,
        seq_len=args.seq_len,
        stride=args.stride,
    )
    print(
        "\ndatasets: " + ", ".join(f"{n} {len(d)}" for n, d in datasets.items())
    )
    windows, ticker_ids = next(iter(make_dataloader(datasets["train"])))
    print(
        f"first train batch: windows {tuple(windows.shape)} {windows.dtype}, "
        f"ticker counts {torch.bincount(ticker_ids, minlength=len(TICKERS)).tolist()}"
    )


if __name__ == "__main__":
    main()
