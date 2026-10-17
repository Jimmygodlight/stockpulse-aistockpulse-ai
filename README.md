# StockPulse AI

**Multi-Agent Quantitative Stock Research & Machine Learning System**

A Google Cloud-based financial research prototype integrating **Gemini, BigQuery, BigQuery ML, Python and Google Agent Development Kit (ADK)**.

## Overview

StockPulse AI investigates historical equity-price behaviour, evaluates systematic trading strategies, and examines whether machine-learning models can predict next-day stock-price direction.

The system combines database-driven quantitative analysis with specialised AI research agents.

## Multi-Agent Architecture

Four specialist agents are coordinated by the main StockPulse AI agent.

| Agent | Responsibility |
|---|---|
| Market Data Analyst | Historical prices, SQL, returns, volatility and indicators |
| Strategy Researcher | Trading strategy development and backtesting analysis |
| Risk Auditor | Research validation, financial risks and methodological weaknesses |
| ML Forecasting Researcher | Evaluation of machine-learning stock-direction models |

## Technology Stack

- **Cloud:** Google Cloud, BigQuery, BigQuery ML
- **AI:** Gemini, Google ADK, Agent Studio
- **Programming:** Python, SQL
- **Libraries:** Pandas, NumPy, yfinance
- **Tools:** BigQuery MCP, Google Search

## Features

- Historical OHLCV price ingestion for AAPL, NVDA, MSFT, GOOGL and SPY
- SQL-based financial analysis
- Daily returns and moving-average indicators
- Historical volatility analysis
- 20-day/50-day moving-average strategy backtesting
- Buy-and-hold benchmarking
- Strategy risk assessment
- ML-based next-day direction classification
- Chronological training, validation and testing

## Quantitative Strategy Backtesting

Implemented a historical moving-average trading strategy with:

- 20-day and 50-day moving averages
- Delayed signal execution assumptions
- Illustrative 0.05% transaction cost per position change
- Strategy and buy-and-hold return comparisons
- Sharpe ratio and maximum drawdown calculations

Results are stored in Google BigQuery.

**Status:** Backtest result table created; independent numerical verification remains outstanding.

## Machine Learning Experiment

### Objective

Predict whether the next trading day's adjusted closing price will be higher than the current day's close.

### Models

1. Logistic Regression
2. Boosted Tree Classifier

### Input Features

- One-day return
- Five-day momentum
- Twenty-day momentum
- Twenty-day rolling volatility
- Moving-average ratio
- Relative trading volume

### Validation Results

1,155 validation observations.

| Metric | Logistic Regression | Boosted Trees | Baseline |
|---|---:|---:|---:|
| Accuracy | 56.19% | 52.21% | 57.40% |
| ROC-AUC | 0.5051 | 0.4867 | — |
| Log Loss | 0.6907 | 0.6954 | 0.6857 |
| F1 Score | 0.7082 | 0.6402 | — |

### Final Holdout Test — Logistic Regression

| Metric | Result |
|---|---:|
| Accuracy | 52.32% |
| Precision | 52.68% |
| Recall | 95.44% |
| F1 Score | 0.6789 |
| ROC-AUC | 0.5104 |
| Log Loss | 0.6921 |

### Findings

Neither machine-learning model outperformed the majority-class baseline on validation accuracy or log loss.

The logistic regression model achieved slightly better validation performance than boosted trees but demonstrated weak directional discrimination on the final holdout set.

**Conclusion:** The experiment demonstrates a reproducible financial machine-learning workflow, not a reliable stock-price forecasting advantage.

## Repository Files

- `agent.py` — Multi-agent Google ADK configuration
- `load_prices.py` — Historical market-data ingestion
- `backtest.py` — Moving-average strategy backtesting
- `features.sql` — Financial feature engineering
- `logistic.sql` — Logistic regression training
- `boosted.sql` — Boosted-tree training
- `evaluation.sql` — ML model evaluation
- `test_baseline.sql` — Test-period baseline calculation
- `requirements.txt` — Python dependencies

## Setup

1. Create a Google Cloud project and BigQuery dataset named `stockpulse`.
2. Configure Google Cloud authentication and the `GOOGLE_CLOUD_PROJECT` environment variable.
3. Install Python dependencies using `pip install -r requirements.txt`.
4. Execute the price-ingestion script.
5. Run the backtesting and ML SQL pipelines in the required sequence.
6. Configure the BigQuery MCP connection with appropriate IAM permissions.
7. Test the exported ADK agent configuration in a supported Google Cloud environment.

The Python agent requires Google Cloud authentication and access to the configured MCP server. Successful execution in a new environment has not yet been verified.

## Evaluation and Limitations

- BigQuery read-only tool execution: verified
- Agent delegation: partially verified
- Multi-agent end-to-end response:
