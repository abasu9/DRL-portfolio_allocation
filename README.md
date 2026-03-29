# Deep Reinforcement Learning for Optimal Portfolio Allocation
## A Comparative Study with Mean-Variance Optimization

This project implements and compares Deep Reinforcement Learning (DRL) algorithms against traditional Mean-Variance Optimization (MVO) for portfolio allocation using Dow Jones 30 stocks.

## Overview

The study evaluates three DRL algorithms -- A2C, PPO, and DDPG -- against two benchmark strategies: Equal-Weight Portfolio and Maximum Sharpe Ratio (Markowitz MVO). All strategies are trained on historical data (2009-2020) and tested on an out-of-sample period (July-December 2020).

## Project Structure

```
.
├── main.py                  # Main pipeline (run this)
├── config.py                # Configuration and hyperparameters
├── data_downloader.py       # Download stock data from Yahoo Finance
├── feature_engineering.py   # Technical indicators and covariance matrices
├── env_portfolio.py         # Custom Gymnasium environment for portfolio allocation
├── models.py                # DRL model training (A2C, PPO, DDPG)
├── benchmarks.py            # Equal-weight and MVO benchmark strategies
├── backtest.py              # Performance metrics computation
├── visualize.py             # Plotting and visualization
├── notebook.ipynb           # Interactive Jupyter notebook
├── requirements.txt         # Python dependencies
├── datasets/                # Downloaded and processed data (generated)
├── trained_models/          # Saved DRL models (generated)
├── results/                 # Performance plots and summary (generated)
└── tensorboard_log/         # Training logs (generated)
```

## Methodology

### Data
- **Assets**: 30 Dow Jones Industrial Average stocks
- **Period**: January 2009 -- December 2020
- **Train/Test Split**: Train on 2009-01-01 to 2020-07-01, test on 2020-07-01 to 2020-12-31
- **Source**: Yahoo Finance via `yfinance`

### Feature Engineering
- **10 Technical Indicators**: MACD, RSI, CCI, ADX, SMA, EMA, Bollinger Bands (upper/lower), ATR, OBV
- **Covariance Matrices**: Rolling 252-day (1 year) lookback covariance of asset returns

### DRL Environment
- **State Space**: Current portfolio weights + stock prices + technical indicators + covariance features
- **Action Space**: Continuous portfolio weight allocation (softmax-normalized)
- **Reward**: Log return of portfolio value (scaled)
- **Transaction Costs**: 0.1% of rebalanced weight

### Algorithms
| Algorithm | Type | Key Feature |
|-----------|------|-------------|
| **A2C** | Actor-Critic | Synchronous advantage estimation |
| **PPO** | Policy Gradient | Clipped surrogate objective for stable updates |
| **DDPG** | Actor-Critic | Continuous action space with deterministic policy |

### Benchmarks
- **Equal Weight**: Naive 1/N allocation across all assets
- **Max Sharpe (MVO)**: Markowitz mean-variance optimization maximizing Sharpe ratio (via PyPortfolioOpt), rebalanced quarterly

### Evaluation Metrics
- Cumulative Return
- Annualized Return & Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the full pipeline
```bash
python main.py
```

### Or use the Jupyter notebook
```bash
jupyter notebook notebook.ipynb
```

### View training logs
```bash
tensorboard --logdir tensorboard_log
```

## Key Libraries
- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/) -- DRL algorithms
- [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/) -- Mean-variance optimization
- [yfinance](https://github.com/ranaroussi/yfinance) -- Financial data
- [ta](https://technical-analysis-library-in-python.readthedocs.io/) -- Technical indicators
- [Gymnasium](https://gymnasium.farama.org/) -- RL environment interface

## License
MIT
