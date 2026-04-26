"""Mean-Variance Optimization (MVO) baseline strategy.

Daily rebalanced, 60-day lookback, Ledoit-Wolf shrinkage covariance,
maximum Sharpe ratio optimization via PyPortfolioOpt.
"""

import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from config import SECTOR_ETFS, MVO_LOOKBACK, RISK_FREE_RATE, INITIAL_AMOUNT


def run_mvo_backtest(etf_prices, trading_dates, lookback=MVO_LOOKBACK,
                     initial_amount=INITIAL_AMOUNT):
    """Run daily-rebalanced MVO backtest on the given trading dates.

    At each trading day t:
    1. Take the lookback-day price window ending at t-1
    2. Compute annualized mean returns
    3. Compute Ledoit-Wolf shrinkage covariance
    4. Solve max-Sharpe portfolio (long-only, fully invested)
    5. Compute portfolio return for day t
    """
    print(f"Running MVO backtest on {len(trading_dates)} trading days...")
    all_dates = etf_prices.index
    tickers = SECTOR_ETFS

    portfolio_values = [1.0]
    daily_returns = [0.0]
    date_list = [trading_dates[0]]
    weight_history = []

    for i in range(len(trading_dates) - 1):
        current_date = trading_dates[i]
        next_date = trading_dates[i + 1]

        # Get lookback window of prices ending at current_date
        loc = all_dates.get_loc(current_date)
        if loc < lookback:
            # Not enough history, use equal weights
            weights = np.ones(len(tickers)) / len(tickers)
        else:
            price_window = etf_prices.iloc[loc - lookback:loc + 1]
            weights = _optimize_max_sharpe(price_window, tickers)

        weight_history.append(weights)

        # Compute portfolio return
        current_prices = etf_prices.loc[current_date].values
        next_prices = etf_prices.loc[next_date].values
        simple_returns = (next_prices - current_prices) / current_prices
        port_return = np.dot(weights, simple_returns)

        portfolio_values.append(portfolio_values[-1] * (1 + port_return))
        daily_returns.append(port_return)
        date_list.append(next_date)

    results = pd.DataFrame({
        "date": date_list,
        "portfolio_value": portfolio_values,
        "daily_return": daily_returns,
    })

    # Scale to initial amount
    results["portfolio_value"] = results["portfolio_value"] * initial_amount

    print(f"MVO backtest complete. Final value: ${results['portfolio_value'].iloc[-1]:,.2f}")
    return results, weight_history


def _optimize_max_sharpe(price_window, tickers):
    """Find max-Sharpe portfolio weights using Ledoit-Wolf shrinkage."""
    try:
        # Annualized mean returns
        mu = mean_historical_return(price_window)

        # Ledoit-Wolf shrinkage covariance
        S = CovarianceShrinkage(price_window).ledoit_wolf()

        # PSD correction: zero out negative eigenvalues
        S_array = S.values if hasattr(S, 'values') else np.array(S)
        eigenvalues, eigenvectors = np.linalg.eigh(S_array)
        eigenvalues = np.maximum(eigenvalues, 0)
        S_corrected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        S_df = pd.DataFrame(S_corrected, index=S.index, columns=S.columns)

        # Optimize
        ef = EfficientFrontier(mu, S_df)
        ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        cleaned = ef.clean_weights()

        weights = np.array([cleaned.get(t, 0.0) for t in tickers])

        # Ensure weights sum to 1
        if np.sum(weights) < 0.01:
            weights = np.ones(len(tickers)) / len(tickers)
        else:
            weights = weights / np.sum(weights)

        return weights

    except Exception as e:
        # Fallback to equal weights
        return np.ones(len(tickers)) / len(tickers)
