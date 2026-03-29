"""Benchmark strategies: Equal-Weight Portfolio and Mean-Variance Optimization."""

import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from config import INITIAL_AMOUNT, TRANSACTION_COST_PCT, RISK_FREE_RATE


def equal_weight_portfolio(df, initial_amount=INITIAL_AMOUNT, transaction_cost_pct=TRANSACTION_COST_PCT):
    """Compute equal-weight portfolio performance.

    Equal allocation across all assets, rebalanced daily.
    """
    print("Computing Equal-Weight Portfolio benchmark...")
    dates = sorted(df["date"].unique())
    tickers = sorted(df["tic"].unique())
    n_assets = len(tickers)
    weights = np.array([1.0 / n_assets] * n_assets)

    portfolio_values = [initial_amount]
    daily_returns = [0.0]
    date_list = [dates[0]]

    for i in range(len(dates) - 1):
        current_data = df[df["date"] == dates[i]].sort_values("tic")
        next_data = df[df["date"] == dates[i + 1]].sort_values("tic")

        current_prices = current_data["close"].values
        next_prices = next_data["close"].values

        returns = (next_prices - current_prices) / current_prices
        port_return = np.dot(weights, returns)

        new_value = portfolio_values[-1] * (1 + port_return)
        portfolio_values.append(new_value)
        daily_returns.append(port_return)
        date_list.append(dates[i + 1])

    return pd.DataFrame({
        "date": date_list,
        "portfolio_value": portfolio_values,
        "daily_return": daily_returns,
    })


def max_sharpe_portfolio(df_train, df_test, initial_amount=INITIAL_AMOUNT,
                         transaction_cost_pct=TRANSACTION_COST_PCT,
                         rebalance_freq=63):
    """Compute Maximum Sharpe Ratio (MVO) portfolio performance.

    Uses PyPortfolioOpt to find optimal weights that maximize the Sharpe ratio.
    Rebalances periodically using rolling historical data.
    """
    print("Computing Maximum Sharpe Ratio (MVO) Portfolio benchmark...")
    tickers = sorted(df_test["tic"].unique())
    dates = sorted(df_test["date"].unique())

    # Build training price history for initial optimization
    train_prices = df_train.pivot_table(index="date", columns="tic", values="close")
    train_prices = train_prices[tickers]

    # Get initial optimal weights
    weights = _optimize_sharpe(train_prices)

    portfolio_values = [initial_amount]
    daily_returns = [0.0]
    date_list = [dates[0]]
    weight_history = [weights.copy()]

    # Build full price history for rolling optimization
    full_df = pd.concat([df_train, df_test]).sort_values(["date", "tic"])
    all_prices = full_df.pivot_table(index="date", columns="tic", values="close")
    all_prices = all_prices[tickers]

    for i in range(len(dates) - 1):
        # Rebalance periodically
        if i > 0 and i % rebalance_freq == 0:
            # Use all data up to current date
            current_date = dates[i]
            hist_prices = all_prices.loc[:current_date]
            if len(hist_prices) > 60:
                new_weights = _optimize_sharpe(hist_prices)
                # Transaction cost
                weight_diff = np.sum(np.abs(
                    np.array([new_weights.get(t, 0) for t in tickers]) -
                    np.array([weights.get(t, 0) for t in tickers])
                ))
                tc = weight_diff * transaction_cost_pct
                portfolio_values[-1] *= (1 - tc)
                weights = new_weights
                weight_history.append(weights.copy())

        current_data = df_test[df_test["date"] == dates[i]].sort_values("tic")
        next_data = df_test[df_test["date"] == dates[i + 1]].sort_values("tic")

        current_prices = current_data["close"].values
        next_prices = next_data["close"].values

        returns = (next_prices - current_prices) / current_prices
        w_array = np.array([weights.get(t, 0) for t in tickers])
        port_return = np.dot(w_array, returns)

        new_value = portfolio_values[-1] * (1 + port_return)
        portfolio_values.append(new_value)
        daily_returns.append(port_return)
        date_list.append(dates[i + 1])

    return pd.DataFrame({
        "date": date_list,
        "portfolio_value": portfolio_values,
        "daily_return": daily_returns,
    })


def _optimize_sharpe(price_df):
    """Find maximum Sharpe ratio weights using PyPortfolioOpt."""
    try:
        mu = mean_historical_return(price_df)
        S = CovarianceShrinkage(price_df).ledoit_wolf()
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        weights = ef.clean_weights()
        return dict(weights)
    except Exception as e:
        print(f"  MVO optimization failed: {e}. Using equal weights.")
        tickers = price_df.columns.tolist()
        return {t: 1.0 / len(tickers) for t in tickers}
