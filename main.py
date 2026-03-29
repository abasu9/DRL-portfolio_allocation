"""
Deep Reinforcement Learning for Optimal Portfolio Allocation:
A Comparative Study with Mean-Variance Optimization

Main pipeline: data download -> feature engineering -> training -> evaluation -> visualization
"""

import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from config import (
    DOW_30_TICKERS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TECHNICAL_INDICATORS,
    INITIAL_AMOUNT,
)
from data_downloader import download_data, clean_data, save_data, load_data
from feature_engineering import prepare_features
from env_portfolio import PortfolioAllocationEnv
from models import get_model, train_model, save_model, load_model, predict_with_model
from benchmarks import equal_weight_portfolio, max_sharpe_portfolio
from backtest import compare_strategies, print_summary
from visualize import generate_all_plots


def build_env(df, tickers):
    """Build portfolio allocation environment from processed DataFrame."""
    stock_dim = len(tickers)
    n_indicators = len(TECHNICAL_INDICATORS)
    cov_columns = [c for c in df.columns if c.startswith("cov_")]
    n_cov = len(cov_columns)

    # State space: weights + prices + tech_indicators*stocks + covariance
    state_space = stock_dim + stock_dim + n_indicators * stock_dim + n_cov
    action_space = stock_dim

    env = PortfolioAllocationEnv(
        df=df,
        stock_dim=stock_dim,
        state_space=state_space,
        action_space=action_space,
    )
    return env


def main():
    print("=" * 80)
    print("Deep Reinforcement Learning for Optimal Portfolio Allocation")
    print("A Comparative Study with Mean-Variance Optimization")
    print("=" * 80)

    # ---- Step 1: Data Download ----
    print("\n[STEP 1] Downloading and cleaning stock data...")
    data_path = "datasets/stock_data.csv"
    if os.path.exists(data_path):
        print(f"Loading cached data from {data_path}")
        raw_data = load_data(data_path)
    else:
        raw_data = download_data()
        raw_data = clean_data(raw_data)
        save_data(raw_data)

    # ---- Step 2: Feature Engineering ----
    print("\n[STEP 2] Feature engineering...")
    features_path = "datasets/features_data.csv"
    if os.path.exists(features_path):
        print(f"Loading cached features from {features_path}")
        data = pd.read_csv(features_path, parse_dates=["date"])
    else:
        data = prepare_features(raw_data)
        os.makedirs("datasets", exist_ok=True)
        data.to_csv(features_path, index=False)
        print(f"Features saved to {features_path}")

    # ---- Step 3: Train/Test Split ----
    print("\n[STEP 3] Splitting data into train and test sets...")
    train_data = data[data["date"] < TEST_START_DATE].reset_index(drop=True)
    test_data = data[data["date"] >= TEST_START_DATE].reset_index(drop=True)
    tickers = sorted(data["tic"].unique())
    print(f"  Train: {train_data['date'].min()} to {train_data['date'].max()} "
          f"({train_data['date'].nunique()} days)")
    print(f"  Test:  {test_data['date'].min()} to {test_data['date'].max()} "
          f"({test_data['date'].nunique()} days)")
    print(f"  Tickers: {len(tickers)}")

    # ---- Step 4: Train DRL Models ----
    print("\n[STEP 4] Training DRL models...")
    train_env = build_env(train_data, tickers)
    algorithms = ["a2c", "ppo", "ddpg"]
    trained_models = {}

    for algo in algorithms:
        model_path = f"trained_models/{algo}.zip"
        if os.path.exists(model_path):
            print(f"  Loading pre-trained {algo.upper()} model...")
            model = load_model(algo, train_env)
        else:
            model = get_model(algo, train_env)
            model = train_model(model, algo)
            save_model(model, algo)
        trained_models[algo] = model

    # ---- Step 5: Evaluate on Test Set ----
    print("\n[STEP 5] Evaluating models on test set...")
    results = {}

    # DRL models
    for algo in algorithms:
        test_env = build_env(test_data, tickers)
        model = trained_models[algo]
        # Re-load model with test environment
        model.set_env(test_env)
        portfolio_df = predict_with_model(model, test_env)
        results[algo.upper()] = portfolio_df
        print(f"  {algo.upper()}: Final value = ${portfolio_df['portfolio_value'].iloc[-1]:,.2f}")

    # Benchmark: Equal Weight
    print("\n[STEP 5b] Computing benchmark portfolios...")
    ew_df = equal_weight_portfolio(test_data)
    results["Equal Weight"] = ew_df
    print(f"  Equal Weight: Final value = ${ew_df['portfolio_value'].iloc[-1]:,.2f}")

    # Benchmark: Max Sharpe (MVO)
    mvo_df = max_sharpe_portfolio(train_data, test_data)
    results["Max Sharpe (MVO)"] = mvo_df
    print(f"  Max Sharpe (MVO): Final value = ${mvo_df['portfolio_value'].iloc[-1]:,.2f}")

    # ---- Step 6: Performance Comparison ----
    print("\n[STEP 6] Performance comparison...")
    summary = compare_strategies(results)
    print_summary(summary)

    # Save summary
    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/performance_summary.csv")
    print("Summary saved to results/performance_summary.csv")

    # ---- Step 7: Visualization ----
    print("\n[STEP 7] Generating visualizations...")
    generate_all_plots(results, summary)

    print("\n" + "=" * 80)
    print("STUDY COMPLETE")
    print("=" * 80)
    print("\nOutputs:")
    print("  - trained_models/     : Saved DRL models (A2C, PPO, DDPG)")
    print("  - results/            : Performance plots and summary CSV")
    print("  - tensorboard_log/    : Training logs (view with: tensorboard --logdir tensorboard_log)")
    print("  - datasets/           : Downloaded and processed data")


if __name__ == "__main__":
    main()
