"""Backtesting and performance evaluation metrics."""

import numpy as np
import pandas as pd
from config import RISK_FREE_RATE


def compute_metrics(portfolio_df, name="Strategy", risk_free_rate=RISK_FREE_RATE):
    """Compute key performance metrics for a portfolio."""
    returns = np.array(portfolio_df["daily_return"].values[1:], dtype=float)
    values = portfolio_df["portfolio_value"].values

    # Annualization factor
    trading_days = 252

    # Cumulative return
    cumulative_return = (values[-1] / values[0]) - 1

    # Annualized return
    n_days = len(returns)
    annualized_return = (1 + cumulative_return) ** (trading_days / max(n_days, 1)) - 1

    # Annualized volatility
    annualized_vol = np.std(returns) * np.sqrt(trading_days)

    # Sharpe ratio
    excess_daily = returns - risk_free_rate / trading_days
    sharpe = np.mean(excess_daily) / np.std(excess_daily) * np.sqrt(trading_days) if np.std(excess_daily) > 0 else 0

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(trading_days) if len(downside_returns) > 0 else 1e-6
    sortino = (annualized_return - risk_free_rate) / downside_std

    # Maximum drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_drawdown = np.max(drawdown)

    # Calmar ratio
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0

    metrics = {
        "Strategy": name,
        "Cumulative Return (%)": round(cumulative_return * 100, 2),
        "Annualized Return (%)": round(annualized_return * 100, 2),
        "Annualized Volatility (%)": round(annualized_vol * 100, 2),
        "Sharpe Ratio": round(sharpe, 4),
        "Sortino Ratio": round(sortino, 4),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
        "Calmar Ratio": round(calmar, 4),
    }
    return metrics


def compare_strategies(results_dict):
    """Compare multiple strategies and return a summary DataFrame.

    Args:
        results_dict: dict of {strategy_name: portfolio_df}
    """
    all_metrics = []
    for name, df in results_dict.items():
        metrics = compute_metrics(df, name=name)
        all_metrics.append(metrics)
    summary = pd.DataFrame(all_metrics)
    summary = summary.set_index("Strategy")
    return summary


def print_summary(summary_df):
    """Pretty-print the comparison summary."""
    print("\n" + "=" * 80)
    print("PORTFOLIO PERFORMANCE COMPARISON")
    print("=" * 80)
    print(summary_df.to_string())
    print("=" * 80 + "\n")
