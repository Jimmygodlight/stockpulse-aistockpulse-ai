# StockPulse AI

**Multi-agent quantitative equity research prototype** built with Google Cloud Agent Studio, Gemini, the BigQuery MCP server, Python, and SQL.

> Status: development. Historical-price ingestion and backtesting should be verified independently. An Agent Studio export is not automatically a locally tested Python application. Do not claim reliable trading predictions or production readiness without evidence.

## Architecture

- **StockPulse AI**: coordinates stock research.
- **Market Data Analyst**: queries historical prices through BigQuery.
- **Strategy Researcher**: evaluates strategy rules and saved backtest results.
- **Risk Auditor**: challenges unsupported financial and methodological claims.

## Repository structure

- `agent/agent.py`: paste the **complete** output from Agent Studio → **Get code** here. Not included in this template because only the original Studio export contains your actual agent configuration.
- `data/load_prices.py`: your existing historical market-data ingestion script.
- `backtesting/backtest.py`: your existing backtesting script.
- `sql/`: reproducible BigQuery queries.
- `evaluation/`: anonymized test cases, results, and limitations.
- `screenshots/`: non-sensitive evidence of the flow and tool invocations.

## Setup (documentation-only starter)

1. Export the full Python code from Agent Studio to `agent/agent.py`.
2. Move your existing `load_prices.py` to `data/` and `backtest.py` to `backtesting/`.
3. Replace all placeholder project references before executing queries.
4. Install packages in a virtual environment: `pip install -r requirements.txt`.
5. Authenticate to Google Cloud with an authorized account and select the project. Never commit service-account keys or `.env` secrets.
6. Confirm your BigQuery datasets and tables exist and match the names in your code.
7. Independently verify analysis and backtest numbers before publishing results.

## Research design

The initial moving-average experiment compares a 20/50-day trend signal against buy-and-hold on historical adjusted prices. A positive historical return alone does not establish out-of-sample performance. Relevant risks include look-ahead bias, survivorship bias, unrealistic costs, and parameter selection bias.

## Planned extension

Build a **next-day direction classifier** with lagged features, chronological training/holdout splits, and evaluation against a majority-class baseline. Use walk-forward validation and avoid any features calculated after the prediction timestamp.

## Limitations

- This project is research-only, not trading advice or an automated brokerage agent.
- Historical stock data may be delayed or revised.
- Google model/tool responses can fail due to quota/capacity.
- Publicly shared examples must not expose credentials or private cloud resources.
