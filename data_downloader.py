"""Download and preprocess stock data from Yahoo Finance."""

import os
import pandas as pd
import yfinance as yf
from config import DOW_30_TICKERS, TRAIN_START_DATE, TEST_END_DATE


def download_data(tickers=DOW_30_TICKERS, start=TRAIN_START_DATE, end=TEST_END_DATE):
    """Download adjusted closing prices and volume for given tickers."""
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")
    data_frames = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                print(f"  Warning: No data for {ticker}, skipping.")
                continue
            df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
            df.columns = ["open", "high", "low", "close", "adjclose", "volume"]
            df["tic"] = ticker
            df["date"] = df.index
            df = df.reset_index(drop=True)
            data_frames.append(df)
        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")
    data = pd.concat(data_frames, ignore_index=True)
    data = data.sort_values(["date", "tic"]).reset_index(drop=True)
    print(f"Downloaded {len(data)} rows for {data['tic'].nunique()} tickers.")
    return data


def clean_data(df):
    """Handle missing values and ensure all tickers have complete date coverage."""
    dates = df["date"].unique()
    tickers = df["tic"].unique()
    # Create full date x ticker grid
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "tic"])
    full = pd.DataFrame(index=idx).reset_index()
    merged = full.merge(df, on=["date", "tic"], how="left")
    # Forward fill then backward fill within each ticker
    merged = merged.sort_values(["tic", "date"])
    merged = merged.groupby("tic", group_keys=False).apply(
        lambda g: g.fillna(method="ffill").fillna(method="bfill")
    )
    merged = merged.sort_values(["date", "tic"]).reset_index(drop=True)
    # Drop any dates where we still have missing data
    merged = merged.dropna()
    print(f"Cleaned data: {len(merged)} rows, {merged['tic'].nunique()} tickers, "
          f"{merged['date'].nunique()} trading days.")
    return merged


def save_data(df, filepath="datasets/stock_data.csv"):
    """Save data to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def load_data(filepath="datasets/stock_data.csv"):
    """Load data from CSV."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    return df


if __name__ == "__main__":
    raw = download_data()
    clean = clean_data(raw)
    save_data(clean)
