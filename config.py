"""Configuration for DRL Portfolio Allocation."""

# Dow Jones 30 tickers
DOW_30_TICKERS = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
]

# Date ranges
TRAIN_START_DATE = "2009-01-01"
TRAIN_END_DATE = "2020-07-01"
TEST_START_DATE = "2020-07-01"
TEST_END_DATE = "2020-12-31"

# Technical indicators
TECHNICAL_INDICATORS = [
    "macd", "rsi_14", "cci_14", "adx_14",
    "sma_20", "ema_20", "bbands_upper", "bbands_lower",
    "atr_14", "obv",
]

# Training parameters
A2C_PARAMS = {
    "n_steps": 5,
    "ent_coef": 0.005,
    "learning_rate": 0.0007,
}

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.0003,
    "batch_size": 64,
}

DDPG_PARAMS = {
    "batch_size": 128,
    "buffer_size": 50000,
    "learning_rate": 0.001,
}

TIMESTEPS = {
    "a2c": 100000,
    "ppo": 100000,
    "ddpg": 50000,
}

# Portfolio parameters
INITIAL_AMOUNT = 1000000
TRANSACTION_COST_PCT = 0.001
REWARD_SCALING = 1e-4
LOOKBACK_WINDOW = 252  # 1 year for covariance matrix

# Risk-free rate for Sharpe ratio
RISK_FREE_RATE = 0.02
