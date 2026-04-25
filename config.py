"""Configuration for DRL Portfolio Allocation (Sood et al., 2023)."""

# 9 S&P 500 Sector ETFs (XLC and XLRE excluded - launched after 2006)
SECTOR_ETFS = ["XLB", "XLI", "XLY", "XLP", "XLV", "XLF", "XLK", "XLU", "XLE"]

# Market indicators (not investable, used as state features)
MARKET_INDICATORS = ["^GSPC", "^VIX"]

# All tickers to download
ALL_TICKERS = SECTOR_ETFS + MARKET_INDICATORS

# Date range for data download (extra early data for lookback warm-up)
DATA_START_DATE = "2005-07-01"
DATA_END_DATE = "2022-01-01"

# Lookback window for state construction (trading days)
LOOKBACK_WINDOW = 60

# Number of assets (sector ETFs only)
N_ASSETS = len(SECTOR_ETFS)

# Sliding window configuration
# 10 windows, each shifted by 1 year
WINDOWS = []
for i in range(10):
    train_start = 2006 + i
    train_end = 2011 + i
    val_year = 2011 + i
    test_year = 2012 + i
    WINDOWS.append({
        "train_start": f"{train_start}-01-01",
        "train_end": f"{train_end}-01-01",
        "val_start": f"{val_year}-01-01",
        "val_end": f"{val_year + 1}-01-01",
        "test_start": f"{test_year}-01-01",
        "test_end": f"{test_year + 1}-01-01",
    })

# PPO hyperparameters (Table 1 of the paper)
PPO_PARAMS = {
    "n_steps": 756,            # 252 * 3
    "batch_size": 1260,        # 252 * 5
    "n_epochs": 16,
    "gamma": 0.9,
    "gae_lambda": 0.9,
    "clip_range": 0.25,
    "learning_rate": 3e-4,     # linearly annealed to 1e-5
    "policy_kwargs": {
        "net_arch": [64, 64],
        "activation_fn": "Tanh",
        "log_std_init": -1.0,
    },
}

TOTAL_TIMESTEPS = 7_500_000
N_ENVS = 10                   # parallel environments
N_SEEDS = 5                   # random seeds per window

# Differential Sharpe Ratio parameters
DSR_ETA = 1.0 / 252
DSR_CLIP = 10.0

# Portfolio parameters
INITIAL_AMOUNT = 1_000_000
RISK_FREE_RATE = 0.0          # paper uses 0 for max_sharpe

# MVO parameters
MVO_LOOKBACK = 60             # same as DRL lookback
