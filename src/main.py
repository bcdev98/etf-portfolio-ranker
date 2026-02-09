#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate

TRADING_DAYS = 252


@dataclass
class ComponentMetric:
    symbol: str
    weight: float
    annual_return: float
    annual_volatility: float


@dataclass
class EtfScore:
    etf: str
    component_count: int
    weighted_return: float
    weighted_volatility: float
    score: float


def _get_etf_holdings(etf: str, top_n: int) -> pd.DataFrame:
    ticker = yf.Ticker(etf)

    holdings = None
    if hasattr(ticker, "funds_data") and ticker.funds_data is not None:
        holdings = getattr(ticker.funds_data, "top_holdings", None)

    if holdings is None or len(holdings) == 0:
        raise ValueError(f"No holdings found for ETF {etf}.")

    df = holdings.copy()

    if "Symbol" in df.columns:
        symbol_series = df["Symbol"]
    elif "symbol" in df.columns:
        symbol_series = df["symbol"]
    elif str(df.index.name).lower() == "symbol":
        symbol_series = pd.Series(df.index, index=df.index)
    else:
        symbol_series = df.iloc[:, 0]

    if "Holding Percent" in df.columns:
        weight_col = "Holding Percent"
    elif "holdingPercent" in df.columns:
        weight_col = "holdingPercent"
    elif "Weight" in df.columns:
        weight_col = "Weight"
    elif "weight" in df.columns:
        weight_col = "weight"
    else:
        # fallback: equal weights if unavailable
        df["__weight"] = 1.0
        weight_col = "__weight"

    out = pd.DataFrame(
        {
            "symbol": symbol_series.astype(str).str.upper().str.strip(),
            "weight": pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0),
        }
    )
    out = out[(out["symbol"] != "") & (out["symbol"] != "NAN")]
    out = out.drop_duplicates(subset=["symbol"])

    if out["weight"].sum() <= 0:
        out["weight"] = 1.0 / max(len(out), 1)
    else:
        out["weight"] = out["weight"] / out["weight"].sum()

    return out.head(top_n)


def _compute_component_metrics(holdings: pd.DataFrame, period: str) -> List[ComponentMetric]:
    symbols = holdings["symbol"].tolist()
    prices = yf.download(symbols, period=period, interval="1d", auto_adjust=True, progress=False)["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=symbols[0])

    daily_returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")

    metrics: List[ComponentMetric] = []
    for _, row in holdings.iterrows():
        sym = row["symbol"]
        wt = float(row["weight"])

        if sym not in daily_returns.columns:
            continue

        r = daily_returns[sym].dropna()
        if len(r) < 20:
            continue

        ann_return = float(r.mean() * TRADING_DAYS)
        ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))
        metrics.append(ComponentMetric(sym, wt, ann_return, ann_vol))

    return metrics


def score_etf(etf: str, period: str, top_n: int) -> Optional[EtfScore]:
    try:
        holdings = _get_etf_holdings(etf, top_n=top_n)
        metrics = _compute_component_metrics(holdings, period=period)
        if not metrics:
            return None

        w = np.array([m.weight for m in metrics], dtype=float)
        w = w / w.sum()
        ret = np.array([m.annual_return for m in metrics], dtype=float)
        vol = np.array([m.annual_volatility for m in metrics], dtype=float)

        weighted_ret = float(np.dot(w, ret))
        weighted_vol = float(np.dot(w, vol))
        score = weighted_ret - weighted_vol

        return EtfScore(
            etf=etf.upper(),
            component_count=len(metrics),
            weighted_return=weighted_ret,
            weighted_volatility=weighted_vol,
            score=score,
        )
    except Exception:
        return None


def rank_etfs(etfs: List[str], period: str, top_n: int, best_n: int) -> pd.DataFrame:
    rows = []
    for etf in etfs:
        s = score_etf(etf, period=period, top_n=top_n)
        if s is None:
            continue
        rows.append(
            {
                "ETF": s.etf,
                "Components Used": s.component_count,
                "Annual Return": s.weighted_return,
                "Annual Volatility": s.weighted_volatility,
                "Combined Score (Return-Vol)": s.score,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("Combined Score (Return-Vol)", ascending=False)
    return df.head(best_n).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank ETFs by weighted component returns and risks, then show best N by combined score."
    )
    p.add_argument("--etfs", nargs="+", required=True, help="ETF tickers, e.g. SPY QQQ IWM XLE")
    p.add_argument("--period", default="1y", help="Price history period for component stats (default: 1y)")
    p.add_argument("--top-components", type=int, default=10, help="Top holdings per ETF to analyze (default: 10)")
    p.add_argument("--best", type=int, default=5, help="How many ETFs to return (default: 5)")
    p.add_argument("--csv", default="", help="Optional CSV output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = rank_etfs(args.etfs, period=args.period, top_n=args.top_components, best_n=args.best)

    if df.empty:
        print("No ETF scores produced. Check ETF symbols and data availability.")
        return

    display_df = df.copy()
    for c in ["Annual Return", "Annual Volatility", "Combined Score (Return-Vol)"]:
        display_df[c] = (display_df[c] * 100).map(lambda x: f"{x:.2f}%")

    print(tabulate(display_df, headers="keys", tablefmt="github", showindex=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved raw numeric results to: {args.csv}")


if __name__ == "__main__":
    main()
