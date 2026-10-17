# StockPulse AI

**Experimental multi-agent quantitative equity research and machine-learning project** using Google Cloud Agent Studio (Google ADK), Gemini, BigQuery MCP, BigQuery ML, Python, and SQL.

> **Project status:** Research prototype. Historical data analysis and ML evaluations have run successfully; full multi-agent reporting is **not yet reliable** because of recurring Gemini HTTP 429 (`Resource exhausted`) responses. The system does not execute trades or demonstrate a reliable forecasting or trading edge.

## Architecture

The main **StockPulse AI** agent coordinates four specialists:

- **Market Data Analyst:** retrieves historical equity prices and calculates financial indicators.
- **Strategy Researcher:** evaluates strategy design and existing backtest results.
- **Risk Auditor:** challenges modelling assumptions, evidence, and claims.
- **ML Forecasting Researcher:** examines BigQuery ML direction-classification models and baselines.

The exported agent configuration connects to BigQuery through the managed MCP server. The exported Python source is **not by itself evidence of a working local deployment**. Agent Studio/Agent Runtime, IAM, available models, and credentials must be configured separately.

## Repository layout

```text
agent/agent.py               # Four-agent code exported from Agent Studio
data/load_prices.py          # Historical daily OHLCV ingestion to BigQuery
backtesting/backtest.py      # Illustrative 20/50-day moving-average simulation
sql/features.sql             # Train/validation/test features and labels
sql/logistic.sql             # BigQuery ML logistic-regression training
sql/boosted.sql              # BigQuery ML boosted-tree training
sql/evaluation.sql           # Model and baseline validation queries
sql/test_baseline.sql        # Holdout majority-class baseline
sql/example_latest_prices.sql # Basic exploratory SQL (if present)
evaluation/README.md         # Actual test evidence and model results
screenshots/README.md        # Documentation of supporting images
requirements.txt
.gitignore
```

## Data and methodology

Historical daily adjusted-price data are downloaded using `yfinance` for AAPL, NVDA, MSFT, GOOGL and SPY, then stored in `stockpulse.prices` in BigQuery. The data are **not a live exchange feed** and may be delayed or revised. Respect the data provider's usage/licensing terms.

The backtesting script implements an illustrative **20-day versus 50-day moving-average** trend rule and compares it with buy-and-hold. The stored `stockpulse.backtest_results` table exists, but its calculated numbers and execution assumptions **still require independent review**. The example includes a conservative signal delay and a simplified cost assumption, not broker-grade execution.

The ML experiment uses returns, momentum, rolling volatility, moving-average gaps and relative trading volume to classify the direction of the **next trading day's** adjusted close. Data are separated chronologically: training through November 2023, validation from January through November 2024, and a holdout from January 2025 onward; December 2023 and December 2024 are unused transition periods.

## Verified ML results

### Validation (1,155 observations)

| Metric | Logistic regression | Boosted trees | Training-based baseline |
|---|---:|---:|---:|
| Accuracy | 56.19% | 52.21% | **57.40%** |
| ROC-AUC | 0.5051 | 0.4867 | Not computed |
| Log loss | 0.6907 | 0.6954 | **0.6857** |
| F1 score | 0.7082 | 0.6402 | Not computed |

The baseline predicts the training-majority class for accuracy and uses the training positive-class frequency for log loss. **Neither model beat the baseline on validation accuracy or log loss.** Logistic regression was chosen for the final holdout because it outperformed boosted trees on the reported validation metrics.

### Final chronological holdout — logistic regression

| Metric | Result |
|---|---:|
| Accuracy | **52.32%** |
| ROC-AUC | **0.5104** |
| Precision | 52.68% |
| Recall | 95.44% |
| F1 score | 0.6789 |
| Log loss | 0.6921 |
| Matched holdout baseline | Not yet documented |

An ROC-AUC close to 0.5 suggests little directional discrimination. **These results do not establish an exploitable market prediction signal or profitable strategy.** See `evaluation/README.md` for the testing record and limitations.

## Reproducing research

1. Use your own authorized Google Cloud project, credentials, and billing configuration. Match the BigQuery dataset location.
2. Create a virtual environment and install `requirements.txt`. Dependency versions are currently unpinned; pin and test versions before promising exact reproduction.
3. Set `GOOGLE_CLOUD_PROJECT` for your project when running ingestion or backtesting scripts and authenticate using supported Google Cloud mechanisms. Never commit service-account credentials.
4. **Only if needed**, execute the price ingestion script and the SQL feature and training scripts in BigQuery Studio. Statements using `CREATE OR REPLACE` overwrite resources and can incur charges.
5. Evaluate actual results with `ML.EVALUATE` and the baseline queries. Do not use your test set to select or retune models.
6. Configure Agent Studio/Agent Runtime and least-privilege access separately if running the agents. The current ADK export is not guaranteed to execute unmodified in every environment.

## Security and limitations

- The Google Cloud **project ID is not a secret**, but API keys, OAuth tokens, and private service-account credentials must not be published.
- Restrict BigQuery and MCP permissions using IAM; agent instructions alone cannot enforce read-only access.
- External news/search content may contain prompt injections and should be treated as untrusted.
- Multi-agent delegation has been observed, but full end-to-end reporting remains blocked by intermittent HTTP 429 errors.
- Results depend on the selected assets and periods; survivorship effects, model selection, nonstationarity, execution costs, and calibration remain open research limitations.
- Do not use this project for live trading or financial decision automation.
