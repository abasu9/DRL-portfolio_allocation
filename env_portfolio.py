"""Custom Gymnasium environment for portfolio allocation."""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from config import (
    INITIAL_AMOUNT,
    TRANSACTION_COST_PCT,
    REWARD_SCALING,
    TECHNICAL_INDICATORS,
)


class PortfolioAllocationEnv(gym.Env):
    """Portfolio allocation environment for DRL agents.

    State: [cash_balance, stock_prices, holdings, technical_indicators, covariance_features]
    Action: portfolio weights (softmax normalized to sum to 1)
    Reward: change in portfolio value (log return)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df, stock_dim, state_space, action_space,
                 tech_indicator_list=TECHNICAL_INDICATORS,
                 initial_amount=INITIAL_AMOUNT,
                 transaction_cost_pct=TRANSACTION_COST_PCT,
                 reward_scaling=REWARD_SCALING):
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list

        # Covariance columns
        self.cov_columns = [c for c in df.columns if c.startswith("cov_")]
        self.n_cov = len(self.cov_columns)

        # State: portfolio weights + price changes + technical indicators + covariance
        self.state_space = state_space
        self.action_space_dim = action_space

        # Gym spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32
        )

        # Get unique dates
        self.dates = sorted(df["date"].unique())
        self.terminal = False

        # Data organized by date
        self.data_by_date = {}
        for date in self.dates:
            self.data_by_date[date] = df[df["date"] == date].sort_values("tic").reset_index(drop=True)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.date_memory = [self.dates[0]]

        # Initial equal-weight allocation
        self.weights = np.array([1.0 / self.stock_dim] * self.stock_dim)

        state = self._get_state()
        return state, {}

    def _get_state(self):
        """Construct observation from current date's data."""
        current_data = self.data_by_date[self.dates[self.day]]

        # Normalized prices (relative to first day's prices)
        prices = current_data["close"].values

        # Technical indicators
        tech_features = []
        for indicator in self.tech_indicator_list:
            if indicator in current_data.columns:
                tech_features.extend(current_data[indicator].values)

        # Covariance features (flattened, same for all tickers on same date)
        if self.cov_columns:
            cov_values = current_data[self.cov_columns].iloc[0].values
        else:
            cov_values = np.array([])

        # State: weights + prices + tech indicators + covariance
        state = np.concatenate([
            self.weights,
            prices,
            np.array(tech_features),
            cov_values,
        ]).astype(np.float32)

        return state

    def step(self, action):
        self.terminal = self.day >= len(self.dates) - 2

        if self.terminal:
            state = self._get_state()
            return state, 0.0, True, False, {}

        # Normalize actions to valid portfolio weights (softmax)
        weights = self._softmax_normalization(action)

        # Get current and next day prices
        current_data = self.data_by_date[self.dates[self.day]]
        next_data = self.data_by_date[self.dates[self.day + 1]]

        current_prices = current_data["close"].values
        next_prices = next_data["close"].values

        # Portfolio return
        price_returns = (next_prices - current_prices) / current_prices

        # Transaction costs from rebalancing
        weight_diff = np.abs(weights - self.weights)
        transaction_cost = np.sum(weight_diff) * self.transaction_cost_pct

        # Portfolio return after costs
        portfolio_return = np.dot(weights, price_returns) - transaction_cost
        new_portfolio_value = self.portfolio_value * (1 + portfolio_return)

        # Reward: log return scaled
        reward = np.log(new_portfolio_value / self.portfolio_value) * self.reward_scaling

        # Update state
        self.weights = weights
        self.portfolio_value = new_portfolio_value
        self.day += 1

        self.asset_memory.append(new_portfolio_value)
        self.portfolio_return_memory.append(portfolio_return)
        self.date_memory.append(self.dates[self.day])

        state = self._get_state()
        return state, reward, False, False, {}

    def _softmax_normalization(self, actions):
        """Normalize actions to portfolio weights using softmax."""
        exp_actions = np.exp(actions - np.max(actions))
        weights = exp_actions / np.sum(exp_actions)
        return weights

    def get_portfolio_stats(self):
        """Return portfolio value history as DataFrame."""
        return pd.DataFrame({
            "date": self.date_memory,
            "portfolio_value": self.asset_memory,
            "daily_return": self.portfolio_return_memory,
        })

    def render(self, mode="human"):
        print(f"Day {self.day}, Portfolio Value: ${self.portfolio_value:,.2f}")
