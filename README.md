# ETF Portfolio Ranker (Python)

Given a list of ETFs, this project:

1. Pulls each ETF's top holdings (components)
2. Computes component annualized return and volatility (risk)
3. Aggregates using holding weights
4. Ranks ETFs by a combined score:

`combined_score = weighted_return - weighted_volatility`

Then it shows the **best 5** (configurable).

## Setup

```bash
cd etf-portfolio-ranker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/main.py --etfs SPY QQQ IWM XLE XLK VTI --period 1y --top-components 10 --best 5
```

Optional CSV export:

```bash
python src/main.py --etfs SPY QQQ IWM XLE XLK VTI --csv examples/output.csv
```

## Notes

- Data source: Yahoo Finance via `yfinance`
- Some ETFs may not expose holdings reliably; those are skipped.
- This is research tooling, **not investment advice**.
