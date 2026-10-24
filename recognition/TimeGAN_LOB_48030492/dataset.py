"""
Data loading and preprocessing for LOBSTER Level 10 limit order books.

This module turns the raw LOBSTER sample files for the five tickers (AMZN, AAPL,
GOOG, INTC, MSFT; 2012-06-21) into training windows for the conditional TimeGAN.

Current stage: loading a single ticker. Each LOBSTER sample consists of two
row-aligned CSV files without headers (see LOBSTER_SampleFiles_ReadMe.txt):

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

MESSAGE_COLUMNS = ["time", "event_type", "order_id", "size", "price", "direction"]


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
        columns += [f"ask_price_{i}", f"ask_size_{i}", f"bid_price_{i}", f"bid_size_{i}"]
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
    message_path, orderbook_path = find_lobster_files(data_dir, ticker, n_levels)

    messages = pd.read_csv(
        message_path, header=None, names=MESSAGE_COLUMNS, usecols=["time", "event_type"]
    )
    book = pd.read_csv(
        orderbook_path, header=None, names=orderbook_columns(n_levels), dtype=np.int64
    )

    if len(messages) != len(book):
        raise ValueError(
            f"{ticker}: message file has {len(messages)} rows but orderbook has {len(book)}"
        )
    if not messages["time"].is_monotonic_increasing:
        raise ValueError(f"{ticker}: message timestamps are not in chronological order")

    ask_price_cols = [f"ask_price_{i}" for i in range(1, n_levels + 1)]
    bid_price_cols = [f"bid_price_{i}" for i in range(1, n_levels + 1)]
    ask_size_cols = [f"ask_size_{i}" for i in range(1, n_levels + 1)]
    bid_size_cols = [f"bid_size_{i}" for i in range(1, n_levels + 1)]

    # Find dummy levels on the raw integers, before any float conversion.
    ask_dummy = book[ask_price_cols].to_numpy() == DUMMY_ASK_PRICE  # (N, n_levels)
    bid_dummy = book[bid_price_cols].to_numpy() == DUMMY_BID_PRICE  # (N, n_levels)

    # LOBSTER documents that dummy levels carry size 0; fail loudly if not.
    if (book[ask_size_cols].to_numpy()[ask_dummy] != 0).any() or (
        book[bid_size_cols].to_numpy()[bid_dummy] != 0
    ).any():
        raise ValueError(f"{ticker}: found a dummy price level with non-zero size")

    price_cols = ask_price_cols + bid_price_cols
    book[price_cols] = book[price_cols].astype(np.float64) / PRICE_SCALE
    book[ask_price_cols] = book[ask_price_cols].mask(ask_dummy)
    book[bid_price_cols] = book[bid_price_cols].mask(bid_dummy)

    return pd.concat([messages, book], axis=1)


def main():
    """Print a short summary of each loaded ticker, as a quick loader check."""
    parser = argparse.ArgumentParser(description="Load LOBSTER Level 10 sample files.")
    parser.add_argument("--data-dir", required=True, help="folder holding the LOBSTER CSVs")
    parser.add_argument(
        "--tickers", nargs="+", default=list(TICKERS), help="tickers to load (default: all five)"
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


if __name__ == "__main__":
    main()
