"""Visualization: portfolio performance plots and comparison charts."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"


def setup_plot_style():
    """Set consistent plot style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("deep")
    plt.rcParams.update({
        "figure.figsize": (14, 7),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_cumulative_returns(results_dict, save=True):
    """Plot cumulative portfolio values for all strategies."""
    setup_plot_style()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    for (name, df), color in zip(results_dict.items(), colors):
        dates = pd.to_datetime(df["date"])
        values = df["portfolio_value"].values
        # Normalize to start at 1
        normalized = values / values[0]
        ax.plot(dates, normalized, label=name, linewidth=2, color=color)

    ax.set_title("Cumulative Portfolio Returns Comparison", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio Value")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, "cumulative_returns.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


def plot_drawdowns(results_dict, save=True):
    """Plot drawdown curves for all strategies."""
    setup_plot_style()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    for (name, df), color in zip(results_dict.items(), colors):
        dates = pd.to_datetime(df["date"])
        values = df["portfolio_value"].values
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak * 100
        ax.fill_between(dates, 0, -drawdown, alpha=0.3, label=name, color=color)
        ax.plot(dates, -drawdown, linewidth=1, color=color)

    ax.set_title("Portfolio Drawdown Comparison", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, "drawdowns.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


def plot_metrics_comparison(summary_df, save=True):
    """Bar chart comparing key metrics across strategies."""
    setup_plot_style()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metrics_to_plot = [
        "Cumulative Return (%)",
        "Annualized Return (%)",
        "Sharpe Ratio",
        "Max Drawdown (%)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = summary_df[metric]
        bars = ax.bar(values.index, values.values, color=plt.cm.tab10(np.arange(len(values))))
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
        # Add value labels on bars
        for bar, val in zip(bars, values.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Strategy Performance Metrics Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, "metrics_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


def plot_daily_returns_distribution(results_dict, save=True):
    """Plot distribution of daily returns for each strategy."""
    setup_plot_style()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, results_dict.items()):
        returns = df["daily_return"].values[1:]
        ax.hist(returns, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(x=np.mean(returns), color="red", linestyle="--",
                   label=f"Mean: {np.mean(returns):.4f}")
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("Daily Return")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Frequency")
    plt.suptitle("Daily Returns Distribution", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, "returns_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


def plot_portfolio_weights(weights_history, tickers, dates, strategy_name, save=True):
    """Plot portfolio weight allocation over time."""
    setup_plot_style()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    weights_df = pd.DataFrame(weights_history, columns=tickers, index=pd.to_datetime(dates))
    weights_df.plot.area(ax=ax, alpha=0.8)
    ax.set_title(f"Portfolio Weight Allocation - {strategy_name}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, f"weights_{strategy_name.lower().replace(' ', '_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


def generate_all_plots(results_dict, summary_df):
    """Generate all visualization plots."""
    print("\nGenerating plots...")
    plot_cumulative_returns(results_dict)
    plot_drawdowns(results_dict)
    plot_metrics_comparison(summary_df)
    plot_daily_returns_distribution(results_dict)
    print("All plots saved to results/ directory.")
