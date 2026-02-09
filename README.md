# ETF Portfolio Ranker (Python)

Given a list of ETFs, this project:

1. Pulls top ETF holdings (components)
2. Computes component annualized return + volatility (risk)
3. Computes a Sharpe-like metric
4. Adds a sector concentration penalty (HHI)
5. Ranks ETFs and shows best N (default 5)

## Scoring

Two score modes:

- `sharpe`: `(weighted_return - risk_free) / weighted_volatility - sector_penalty * sector_hhi`
- `return_risk`: `(weighted_return - weighted_volatility) - sector_penalty * sector_hhi`

Where sector HHI is the Herfindahl concentration of component sector weights (higher = less diversified).

## Setup

```bash
cd etf-portfolio-ranker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI Usage

```bash
python src/main.py \
  --etfs SPY QQQ IWM XLE XLK VTI \
  --period 1y \
  --top-components 10 \
  --best 5 \
  --score-mode sharpe \
  --risk-free 0.02 \
  --sector-penalty 0.20
```

Optional CSV export:

```bash
python src/main.py --etfs SPY QQQ IWM XLE XLK VTI --csv examples/output.csv
```

## Portfolio Optimizer (new)

Add `--optimize` to optimize ETF weights over the top-ranked ETFs (long-only, sum to 1, max weight cap):

```bash
python src/main.py \
  --etfs SPY QQQ IWM XLE XLK VTI \
  --score-mode sharpe \
  --best 5 \
  --optimize \
  --max-weight 0.50 \
  --n-portfolios 5000
```

## Streamlit App

```bash
streamlit run src/app.py
```

The UI now supports both ranking and optimized ETF-weight recommendations.

## Notes

- Data source: Yahoo Finance via `yfinance`
- Some ETFs/components may be skipped if source data is unavailable.
- For research/education only — not investment advice.
