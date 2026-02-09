# Precog Quant Task 2026: Algorithmic Trading Pipeline

This repository contains a complete, end-to-end algorithmic trading pipeline developed for the Precog Quant Task. The project transforms raw, anonymized stock price data into a sophisticated, optimized trading system designed to maximize risk-adjusted returns.

## Project Overview

The core objective of this project was to design and implement a robust quantitative trading strategy. The pipeline is structured into four distinct but interconnected stages:

1.  Feature Engineering- Cleaning raw data and extracting a rich set of predictive features.
2.  Model Training- Building a machine learning model to generate alpha signals from the features.
3.  Backtesting & Optimization- Simulating the strategy's performance and iteratively improving it.
4.  Statistical Arbitrage- Identifying and analyzing relative-value opportunities.

The final output is a highly optimized trading model that demonstrates strong performance on out-of-sample data, validated through a rigorous backtesting and analysis framework.

## Directory Structure

The repository is organized to separate logic, research, and artifacts, ensuring clarity and reproducibility.

```
.
├── anonymized_data/
│   └── Asset_*.csv         # Raw input data
├── model_pipeline/
│   ├── artifacts/          # Output files (features, models, plots)
│   ├── notebooks/
│   │   ├── 01_feature_engineering.ipynb
│   │   ├── 02_modeling_strategy.ipynb
│   │   ├── 03_baseline_backtest_and_stat_arb.ipynb
│   │   └── 04_optimization_and_analysis.ipynb
│   └── src/                # Core Python source code
│       ├── __init__.py
│       ├── backtest.py
│       ├── config.py
│       ├── data_loader.py
│       ├── features.py
│       ├── modeling.py
│       └── stat_arb.py
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

-   **`anonymized_data/`**: Contains the raw OHLCV data for all assets.
-   **`model_pipeline/artifacts/`**: Stores all generated files, such as the feature matrix, trained model etc.
-   **`model_pipeline/notebooks/`**: Contains the Jupyter notebooks(With outputs of all cells).
-   **`model_pipeline/src/`**: Contains all the core logic as modular, reusable Python scripts.
-   **`requirements.txt`**: Lists all Python dependencies required to run the project.

## Setup and Execution

To replicate the results, you can follow these steps:

**1. Setup Environment:**

First, create and activate a virtual environment, then install the required dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Execution Order:**

The notebooks are designed to be run in a specific sequence to ensure data dependencies are met.

1.  **`01_feature_engineering.ipynb`**: This notebook loads the raw data, cleans it, and generates the primary `feature_matrix.parquet` file.
2.  **`02_modeling_strategy.ipynb`**: This notebook trains the gradient boosting model on the feature matrix and generates the `alpha_signals.parquet` and `alpha_signals_smoothed.parquet` files.
3.  **`03_baseline_backtest_and_stat_arb.ipynb`**: This notebook runs the initial, unoptimized backtest and performs the statistical arbitrage analysis. It serves as the baseline for our improvements.
4.  **`04_optimization_and_analysis.ipynb`**: This is the final analysis notebook. It runs multiple backtests to compare the baseline strategy against the optimized versions (weekly rebalancing, signal smoothing, and enhanced features) and presents the final, superior results.

## Methodology and Approach

### Part 1: Feature Engineering

-   **Logic**: `model_pipeline/src/features.py`
-   **Notebook**: `01_feature_engineering.ipynb`
-   A comprehensive feature set was engineered to capture various market dynamics, including:
    -   **Technical Oscillators**: RSI and MACD (including its signal line and histogram).
    -   **Volatility**: Historical volatility over multiple windows (e.g., 21 and 63 days).
    -   **Volume Patterns**: A z-score to identify unusual trading activity relative to a 20-day rolling average.
    -   **Momentum & Mean-Reversion**: Returns over various windows (from 1 day to 252 days) to capture short-term reversals and long-term trends.
    -   **Positional Features**: Indicators like `price_distance_20` (distance from 20-day MA) and `close_location` (close relative to high-low range).
    -   **Cross-Sectional Ranks**: Features were created to rank each asset's daily return and volatility against its peers, providing crucial relative context.

### Part 2: Model Training & Strategy Formulation

-   **Logic**: `model_pipeline/src/modeling.py`
-   **Notebook**: `02_modeling_strategy.ipynb`
-   **Model Choice**: A `HistGradientBoostingRegressor` was chosen for its speed and ability to model complex, non-linear relationships in noisy financial data. The model was trained to predict 1-day forward returns.
-   **Signal Generation**: The model's predictions are used as the core `alpha_signal`.

### Part 3: Backtesting & Performance Analysis

-   **Logic**: `model_pipeline/src/backtest.py`
-   **Notebooks**: `03_baseline_backtest_and_stat_arb.ipynb`, `04_optimization_and_analysis.ipynb`
-   A sophisticated backtester was built to simulate a realistic long-short portfolio.
-   **Key Optimizations**:
    1.  **Rebalancing Frequency**: We demonstrated that switching from a daily to a **weekly rebalance** significantly reduced transaction costs (`avg_turnover`) and improved the Sharpe Ratio.
    2.  **Signal Smoothing**: We applied a 5-day rolling average to the raw alpha signal, which filtered out noise and led to more stable, higher-quality trading decisions.
-   The final, optimized strategy is a **weekly rebalanced, signal-smoothed model built on an enhanced feature set**, which proved to be the most profitable and robust configuration.

### Part 4: Statistical Arbitrage Overlay

-   **Logic**: `model_pipeline/src/stat_arb.py`
-   **Notebook**: `03_baseline_backtest_and_stat_arb.ipynb`
-   A two-stage process was used to identify high-potential pairs:
    1.  **Correlation Scan**: A fast scan to identify highly correlated assets.
    2.  **Cointegration Test**: The Engle-Granger two-step method was applied to the candidates to find pairs with a statistically significant, long-run mean-reverting relationship.
-   An implementation idea was proposed to integrate these pair trades as a market-neutral overlay, allocating a portion of capital based on the half-life of the pair's spread.
