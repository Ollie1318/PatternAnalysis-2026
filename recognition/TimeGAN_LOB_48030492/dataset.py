"""
Data loading and preprocessing for LOBSTER Level 10 limit order books.

This module turns the raw LOBSTER sample files for the five tickers (AMZN, AAPL,
GOOG, INTC, MSFT; 2012-06-21) into training windows for the conditional TimeGAN.

Current stages:
    1. load_ticker:       raw event-time book for one ticker, prices in dollars.
    2. resample_to_grid:  regular clock-time grid (default 1 s), last book state
                          at or before each grid time (no look-ahead).

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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
