"""Backtesting and performance evaluation metrics (all 13 metrics from the paper)."""

import numpy as np
import pandas as pd
from scipy import stats
from config import RISK_FREE_RATE


def compute_metrics(portfolio_df, name="Strategy"):
    """Compute all 13 performance metrics from the paper.

    Metrics: Annual Return, Annual Volatility, Sharpe Ratio, Calmar Ratio,
    Stability, Max Drawdown, Omega Ratio, Sortino Ratio, Skew, Kurtosis,
    Tail Ratio, Daily VaR, Cumulative Returns.
    """
    returns = np.array(portfolio_df["daily_return"].values[1:], dtype=float)
    values = np.array(portfolio_df["portfolio_value"].values, dtype=float)
    trading_days = 252

    # Cumulative return
    cumulative_return = (values[-1] / values[0]) - 1

    # Annualized return
    n_days = len(returns)
    total_return = values[-1] / values[0]
    annualized_return = total_return ** (trading_days / max(n_days, 1)) - 1

    # Annualized volatility
    annualized_vol = np.std(returns, ddof=1) * np.sqrt(trading_days)

    # Sharpe ratio
    daily_rf = RISK_FREE_RATE / trading_days
    excess = returns - daily_rf
    sharpe = np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(trading_days) if np.std(excess, ddof=1) > 0 else 0

    # Maximum drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    max_drawdown = np.min(drawdown)

    # Calmar ratio
    calmar = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 1e-10 else 0

    # Stability: R^2 of log cumulative returns vs. linear time
    log_cum = np.log(values / values[0])
    if len(log_cum) > 1:
        x = np.arange(len(log_cum))
        slope, intercept, r_value, _, _ = stats.linregress(x, log_cum)
        stability = r_value ** 2
    else:
        stability = 0

    # Omega ratio: sum(gains) / sum(|losses|) with threshold 0
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    omega = np.sum(gains) / abs(np.sum(losses)) if abs(np.sum(losses)) > 1e-10 else 0

    # Sortino ratio
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) * np.sqrt(trading_days) if len(downside) > 1 else 1e-6
    sortino = (annualized_return - RISK_FREE_RATE) / downside_std

    # Skewness
    skew = stats.skew(returns)

    # Excess kurtosis
    kurtosis = stats.kurtosis(returns)

    # Tail ratio: P95 / |P5|
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)
    tail_ratio = abs(p95 / p5) if abs(p5) > 1e-10 else 0

    # Daily Value at Risk (5th percentile)
    daily_var = np.percentile(returns, 5)

    return {
        "Strategy": name,
        "Cumulative Return": round(cumulative_return, 4),
        "Annual Return": round(annualized_return, 4),
        "Annual Volatility": round(annualized_vol, 4),
        "Sharpe Ratio": round(sharpe, 4),
        "Calmar Ratio": round(calmar, 4),
        "Stability": round(stability, 4),
        "Max Drawdown": round(max_drawdown, 4),
        "Omega Ratio": round(omega, 4),
        "Sortino Ratio": round(sortino, 4),
        "Skew": round(skew, 4),
        "Kurtosis": round(kurtosis, 4),
        "Tail Ratio": round(tail_ratio, 4),
        "Daily VaR (5%)": round(daily_var, 6),
    }


def compare_strategies(results_dict):
    """Compare multiple strategies and return a summary DataFrame."""
    all_metrics = []
    for name, df in results_dict.items():
        metrics = compute_metrics(df, name=name)
        all_metrics.append(metrics)
    summary = pd.DataFrame(all_metrics).set_index("Strategy")
    return summary


def print_summary(summary_df):
    """Pretty-print the comparison summary."""
    print("\n" + "=" * 100)
    print("PORTFOLIO PERFORMANCE COMPARISON")
    print("=" * 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(summary_df.to_string())
    print("=" * 100 + "\n")
