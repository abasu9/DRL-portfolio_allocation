"""Custom Gymnasium environment for portfolio allocation (Sood et al., 2023).

State: (n_assets+1) x lookback matrix, flattened
Action: logits in R^(n_assets), softmax -> portfolio weights
Reward: Differential Sharpe Ratio (Moody et al., 1998)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from config import N_ASSETS, LOOKBACK_WINDOW, DSR_ETA, DSR_CLIP


class PortfolioEnv(gym.Env):
    """Portfolio allocation environment with Differential Sharpe Ratio reward."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, etf_log_returns, etf_prices, market_features,
                 trading_dates, lookback=LOOKBACK_WINDOW):
        super().__init__()
        self.etf_log_returns = etf_log_returns
        self.etf_prices = etf_prices
        self.market_features = market_features
        self.trading_dates = trading_dates
        self.lookback = lookback
        self.n_assets = N_ASSETS

        # State: (n_assets + 1) * lookback
        state_dim = (self.n_assets + 1) * self.lookback
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # Action: logits for n_assets (softmax applied internally)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        self.all_dates = etf_log_returns.index

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        # Equal weight initial allocation
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

        # DSR running statistics
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns

        # Tracking
        self.portfolio_values = [1.0]  # normalized
        self.daily_returns = [0.0]
        self.weight_history = [self.weights.copy()]
        self.date_history = [self.trading_dates[0]]

        state = self._get_state()
        return state, {}

    def _get_state(self):
        """Construct state from current date's data."""
        date = self.trading_dates[self.step_idx]
        loc = self.all_dates.get_loc(date)

        # Lagged returns: t-1 to t-(lookback-1)
        returns_window = self.etf_log_returns.iloc[loc - self.lookback + 1:loc].values

        state = np.zeros((self.n_assets + 1, self.lookback), dtype=np.float32)

        # Asset rows
        for i in range(self.n_assets):
            state[i, 0] = self.weights[i]
            state[i, 1:] = returns_window[::-1, i]

        # Cash row
        cash_weight = max(1.0 - np.sum(self.weights), 0.0)
        state[self.n_assets, 0] = cash_weight

        mf = self.market_features.loc[date]
        state[self.n_assets, 1] = mf["vol20_norm"]
        state[self.n_assets, 2] = mf["vol_ratio_norm"]
        state[self.n_assets, 3] = mf["vix_norm"]

        return state.flatten()

    def step(self, action):
        # Apply softmax to get portfolio weights
        new_weights = self._softmax(action)

        current_date = self.trading_dates[self.step_idx]
        next_idx = self.step_idx + 1
        terminated = next_idx >= len(self.trading_dates)

        if terminated:
            state = self._get_state()
            return state, 0.0, True, False, {}

        next_date = self.trading_dates[next_idx]

        # Compute portfolio return using simple returns
        current_prices = self.etf_prices.loc[current_date].values
        next_prices = self.etf_prices.loc[next_date].values
        simple_returns = (next_prices - current_prices) / current_prices
        portfolio_return = np.dot(new_weights, simple_returns)

        # Differential Sharpe Ratio reward
        reward = self._differential_sharpe_ratio(portfolio_return)

        # Update state
        self.weights = new_weights
        self.step_idx = next_idx
        self.portfolio_values.append(self.portfolio_values[-1] * (1 + portfolio_return))
        self.daily_returns.append(portfolio_return)
        self.weight_history.append(new_weights.copy())
        self.date_history.append(next_date)

        state = self._get_state()
        return state, reward, False, False, {}

    def _softmax(self, logits):
        """Convert logits to portfolio weights via softmax."""
        exp_logits = np.exp(logits - np.max(logits))
        return (exp_logits / np.sum(exp_logits)).astype(np.float32)

    def _differential_sharpe_ratio(self, R_t):
        """Compute DSR reward (Moody et al., 1998).

        A_t = A_{t-1} + eta * (R_t - A_{t-1})
        B_t = B_{t-1} + eta * (R_t^2 - B_{t-1})
        D_t = (B_{t-1} * dA - 0.5 * A_{t-1} * dB) / (B_{t-1} - A_{t-1}^2)^{3/2}
        """
        dA = R_t - self.A
        dB = R_t ** 2 - self.B

        denominator = self.B - self.A ** 2
        if denominator > 1e-12:
            D_t = (self.B * dA - 0.5 * self.A * dB) / (denominator ** 1.5)
        else:
            D_t = 0.0

        # Clip reward
        D_t = np.clip(D_t, -DSR_CLIP, DSR_CLIP)

        # Update running statistics
        self.A = self.A + DSR_ETA * dA
        self.B = self.B + DSR_ETA * dB

        return float(D_t)

    def get_results(self):
        """Return backtest results as dict."""
        import pandas as pd
        return pd.DataFrame({
            "date": self.date_history,
            "portfolio_value": self.portfolio_values,
            "daily_return": self.daily_returns,
        })

    def render(self, mode="human"):
        print(f"Step {self.step_idx}, Value: {self.portfolio_values[-1]:.4f}")
