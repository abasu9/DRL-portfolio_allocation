"""Feature engineering: technical indicators and covariance matrices."""

import numpy as np
import pandas as pd
import ta
from config import LOOKBACK_WINDOW


def add_technical_indicators(df):
    """Add technical indicators for each ticker."""
    print("Adding technical indicators...")
    result = []
    for tic, group in df.groupby("tic"):
        g = group.sort_values("date").copy()
        close = g["close"]
        high = g["high"]
        low = g["low"]
        volume = g["volume"]

        # MACD
        macd_indicator = ta.trend.MACD(close)
        g["macd"] = macd_indicator.macd()

        # RSI
        g["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

        # CCI
        g["cci_14"] = ta.trend.CCIIndicator(high, low, close, window=14).cci()

        # ADX
        g["adx_14"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()

        # SMA 20
        g["sma_20"] = ta.trend.SMAIndicator(close, window=20).sma_indicator()

        # EMA 20
        g["ema_20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20)
        g["bbands_upper"] = bb.bollinger_hband()
        g["bbands_lower"] = bb.bollinger_lband()

        # ATR
        g["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

        # OBV
        g["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

        result.append(g)

    df_out = pd.concat(result, ignore_index=True)
    df_out = df_out.sort_values(["date", "tic"]).reset_index(drop=True)
    # Drop initial rows with NaN from indicator calculation
    df_out = df_out.dropna().reset_index(drop=True)
    print(f"Technical indicators added. Shape: {df_out.shape}")
    return df_out


def add_covariance_matrices(df, lookback=LOOKBACK_WINDOW):
    """Compute rolling covariance matrices of returns with given lookback."""
    print(f"Computing covariance matrices (lookback={lookback})...")
    # Pivot to get returns
    price_pivot = df.pivot_table(index="date", columns="tic", values="close")
    returns = price_pivot.pct_change().dropna()

    tickers = sorted(df["tic"].unique())
    dates = sorted(df["date"].unique())

    cov_dict = {}
    for i, date in enumerate(dates):
        if date not in returns.index:
            continue
        loc = returns.index.get_loc(date)
        if loc < lookback:
            continue
        window_returns = returns.iloc[loc - lookback:loc]
        cov_matrix = window_returns.cov().values
        cov_dict[date] = cov_matrix

    # Store covariance as flattened array in each row
    cov_list = []
    n_assets = len(tickers)
    for _, row in df.iterrows():
        date = row["date"]
        if date in cov_dict:
            cov_flat = cov_dict[date].flatten()
        else:
            cov_flat = np.zeros(n_assets * n_assets)
        cov_list.append(cov_flat)

    cov_columns = [f"cov_{i}" for i in range(n_assets * n_assets)]
    cov_df = pd.DataFrame(cov_list, columns=cov_columns, index=df.index)
    df_out = pd.concat([df, cov_df], axis=1)

    # Only keep dates with valid covariance
    valid_dates = set(cov_dict.keys())
    df_out = df_out[df_out["date"].isin(valid_dates)].reset_index(drop=True)
    print(f"Covariance matrices added. Shape: {df_out.shape}")
    return df_out


def prepare_features(df):
    """Full feature engineering pipeline."""
    df = add_technical_indicators(df)
    df = add_covariance_matrices(df)
    return df
