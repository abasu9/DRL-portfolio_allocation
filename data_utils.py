"""Data download, feature engineering, and sliding window preparation."""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from config import (
    SECTOR_ETFS, MARKET_INDICATORS, ALL_TICKERS,
    DATA_START_DATE, DATA_END_DATE, LOOKBACK_WINDOW,
)


def download_data(tickers=ALL_TICKERS, start=DATA_START_DATE, end=DATA_END_DATE,
                  save_path="datasets/raw_data.csv"):
    """Download adjusted closing prices for sector ETFs and market indicators."""
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    frames = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                continue
            series = df["Adj Close"].rename(ticker)
            frames.append(series)
        except Exception:
            pass

    if not frames:
        print("  Yahoo Finance unavailable. Generating synthetic market data...")
        return _generate_synthetic_data(tickers, start, end, save_path)

    prices = pd.concat(frames, axis=1)
    prices.index.name = "date"
    prices = prices.dropna()
    prices.to_csv(save_path)
    print(f"Downloaded {len(prices)} trading days. Saved to {save_path}")
    return prices


def _generate_synthetic_data(tickers, start, end, save_path):
    """Generate synthetic price data when Yahoo Finance is unavailable.

    Uses geometric Brownian motion calibrated to realistic sector ETF parameters.
    """
    dates = pd.bdate_range(start=start, end=end)

    # Realistic annual return / vol for each ticker
    params = {
        "XLB": (0.07, 0.20), "XLI": (0.09, 0.18), "XLY": (0.11, 0.19),
        "XLP": (0.06, 0.12), "XLV": (0.08, 0.15), "XLF": (0.05, 0.25),
        "XLK": (0.13, 0.20), "XLU": (0.05, 0.14), "XLE": (0.03, 0.28),
        "^GSPC": (0.09, 0.16), "^VIX": (0.0, 0.80),
    }
    start_prices = {
        "XLB": 30, "XLI": 35, "XLY": 40, "XLP": 25, "XLV": 35,
        "XLF": 28, "XLK": 22, "XLU": 30, "XLE": 55,
        "^GSPC": 1200, "^VIX": 15,
    }

    rng = np.random.RandomState(42)
    n_days = len(dates)
    data = {}

    for ticker in tickers:
        mu, sigma = params.get(ticker, (0.07, 0.18))
        p0 = start_prices.get(ticker, 50)
        daily_mu = mu / 252
        daily_sigma = sigma / np.sqrt(252)

        if ticker == "^VIX":
            # Mean-reverting process for VIX
            vix = np.zeros(n_days)
            vix[0] = p0
            for t in range(1, n_days):
                shock = rng.normal(0, 3)
                vix[t] = vix[t - 1] + 0.02 * (18 - vix[t - 1]) + shock
                vix[t] = max(vix[t], 9)
            data[ticker] = vix
        else:
            log_returns = rng.normal(daily_mu, daily_sigma, n_days)
            prices_arr = p0 * np.exp(np.cumsum(log_returns))
            data[ticker] = prices_arr

    prices = pd.DataFrame(data, index=dates)
    prices.index.name = "date"
    prices.to_csv(save_path)
    print(f"  Generated {len(prices)} trading days of synthetic data. Saved to {save_path}")
    return prices


def load_data(path="datasets/raw_data.csv"):
    """Load price data from CSV."""
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


def compute_log_returns(prices):
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_market_features(prices, log_returns):
    """Compute market regime indicators (expanding-window z-scored).

    Features:
    - vol20_norm: 20-day realized volatility of S&P 500, z-scored
    - vol_ratio_norm: ratio of 20-day to 60-day vol, z-scored
    - vix_norm: VIX level, z-scored

    All z-scores use expanding window to avoid look-ahead bias.
    """
    sp500_returns = log_returns["^GSPC"]
    vix = prices["^VIX"]

    # 20-day and 60-day realized volatility of S&P 500
    vol20 = sp500_returns.rolling(20).std() * np.sqrt(252)
    vol60 = sp500_returns.rolling(60).std() * np.sqrt(252)
    vol_ratio = vol20 / vol60

    # Expanding window z-score (no look-ahead bias)
    def expanding_zscore(series):
        mean = series.expanding().mean()
        std = series.expanding().std()
        return (series - mean) / std.replace(0, 1)

    features = pd.DataFrame({
        "vol20_norm": expanding_zscore(vol20),
        "vol_ratio_norm": expanding_zscore(vol_ratio),
        "vix_norm": expanding_zscore(vix),
    }, index=prices.index)

    return features.dropna()


def build_state_data(prices):
    """Build all data needed for state construction.

    Returns:
        etf_prices: DataFrame of ETF adjusted prices
        etf_log_returns: DataFrame of ETF log returns
        market_features: DataFrame with 3 market regime indicators
    """
    log_returns = compute_log_returns(prices)
    etf_prices = prices[SECTOR_ETFS]
    etf_log_returns = log_returns[SECTOR_ETFS]
    market_features = compute_market_features(prices, log_returns)

    # Align all to common dates
    common_dates = etf_log_returns.index.intersection(market_features.index)
    etf_prices = etf_prices.loc[common_dates]
    etf_log_returns = etf_log_returns.loc[common_dates]
    market_features = market_features.loc[common_dates]

    return etf_prices, etf_log_returns, market_features


def get_window_data(etf_log_returns, market_features, start_date, end_date, lookback=LOOKBACK_WINDOW):
    """Extract data for a specific date range, including lookback for warm-up.

    Returns dates within [start_date, end_date) that have enough lookback history.
    """
    all_dates = etf_log_returns.index
    mask = (all_dates >= pd.Timestamp(start_date)) & (all_dates < pd.Timestamp(end_date))
    target_dates = all_dates[mask]

    # Filter to dates with enough lookback
    valid_dates = []
    for date in target_dates:
        loc = all_dates.get_loc(date)
        if loc >= lookback:
            valid_dates.append(date)

    return valid_dates


def construct_state(date, etf_log_returns, market_features, weights, lookback=LOOKBACK_WINDOW):
    """Construct the state matrix for a given date.

    State shape: (n_assets + 1, lookback)
    - Rows 0..n_assets-1: [w_i, r_{i,t-1}, ..., r_{i,t-59}]
    - Row n_assets (cash): [w_cash, vol20_norm, vol_ratio_norm, vix_norm, 0, ..., 0]

    Returns flattened state vector.
    """
    n_assets = len(SECTOR_ETFS)
    all_dates = etf_log_returns.index
    loc = all_dates.get_loc(date)

    # Lagged returns: t-1 to t-59 (59 values)
    returns_window = etf_log_returns.iloc[loc - lookback + 1:loc].values  # (59, n_assets)

    state = np.zeros((n_assets + 1, lookback), dtype=np.float32)

    # Asset rows: weight + lagged returns
    for i in range(n_assets):
        state[i, 0] = weights[i]
        state[i, 1:] = returns_window[::-1, i]  # most recent first

    # Cash row: cash weight + market indicators + zeros
    cash_weight = 1.0 - np.sum(weights)
    state[n_assets, 0] = max(cash_weight, 0.0)

    mf = market_features.loc[date]
    state[n_assets, 1] = mf["vol20_norm"]
    state[n_assets, 2] = mf["vol_ratio_norm"]
    state[n_assets, 3] = mf["vix_norm"]
    # remaining positions are 0

    return state.flatten()
