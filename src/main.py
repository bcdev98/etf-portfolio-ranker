#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    sector: str


@dataclass
class EtfScore:
    etf: str
    component_count: int
    weighted_return: float
    weighted_volatility: float
    sharpe_like: float
    sector_hhi: float
    score: float


def _get_etf_holdings(etf: str, top_n: int) -> pd.DataFrame:
    ticker = yf.Ticker(etf)
    holdings = getattr(getattr(ticker, "funds_data", None), "top_holdings", None)
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
        df["__weight"] = 1.0
        weight_col = "__weight"

    out = pd.DataFrame(
        {
            "symbol": symbol_series.astype(str).str.upper().str.strip(),
            "weight": pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0),
        }
    )
    out = out[(out["symbol"] != "") & (out["symbol"] != "NAN")].drop_duplicates(subset=["symbol"])
    out["weight"] = out["weight"] / out["weight"].sum() if out["weight"].sum() > 0 else 1.0 / max(len(out), 1)
    return out.head(top_n)


def _get_sector(symbol: str) -> str:
    try:
        info = yf.Ticker(symbol).info
        return str(info.get("sector") or "Unknown")
    except Exception:
        return "Unknown"


def _compute_component_metrics(holdings: pd.DataFrame, period: str) -> List[ComponentMetric]:
    symbols = holdings["symbol"].tolist()
    prices = yf.download(symbols, period=period, interval="1d", auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=symbols[0])

    daily_returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    metrics: List[ComponentMetric] = []
    for _, row in holdings.iterrows():
        sym, wt = row["symbol"], float(row["weight"])
        if sym not in daily_returns.columns:
            continue
        r = daily_returns[sym].dropna()
        if len(r) < 20:
            continue
        metrics.append(
            ComponentMetric(
                symbol=sym,
                weight=wt,
                annual_return=float(r.mean() * TRADING_DAYS),
                annual_volatility=float(r.std() * np.sqrt(TRADING_DAYS)),
                sector=_get_sector(sym),
            )
        )
    return metrics


def _sector_hhi(metrics: List[ComponentMetric]) -> float:
    by_sector: Dict[str, float] = {}
    for m in metrics:
        by_sector[m.sector] = by_sector.get(m.sector, 0.0) + m.weight
    return float(sum(w * w for w in by_sector.values()))


def score_etf(etf: str, period: str, top_n: int, score_mode: str, risk_free: float, sector_penalty: float) -> Optional[EtfScore]:
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
        sharpe_like = float((weighted_ret - risk_free) / max(weighted_vol, 1e-9))
        hhi = _sector_hhi(metrics)

        base = sharpe_like if score_mode == "sharpe" else (weighted_ret - weighted_vol)
        score = float(base - sector_penalty * hhi)

        return EtfScore(
            etf=etf.upper(),
            component_count=len(metrics),
            weighted_return=weighted_ret,
            weighted_volatility=weighted_vol,
            sharpe_like=sharpe_like,
            sector_hhi=hhi,
            score=score,
        )
    except Exception:
        return None


def rank_etfs(
    etfs: List[str], period: str, top_n: int, best_n: int, score_mode: str, risk_free: float, sector_penalty: float
) -> pd.DataFrame:
    rows = []
    for etf in etfs:
        s = score_etf(etf, period, top_n, score_mode, risk_free, sector_penalty)
        if s is None:
            continue
        rows.append(
            {
                "ETF": s.etf,
                "Components Used": s.component_count,
                "Annual Return": s.weighted_return,
                "Annual Volatility": s.weighted_volatility,
                "Sharpe-like": s.sharpe_like,
                "Sector HHI": s.sector_hhi,
                "Score": s.score,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Score", ascending=False).head(best_n).reset_index(drop=True)


def optimize_portfolio(
    ranked_df: pd.DataFrame,
    period: str,
    risk_free: float,
    sector_penalty: float,
    max_weight: float,
    n_portfolios: int,
) -> pd.DataFrame:
    tickers = ranked_df["ETF"].tolist()
    if len(tickers) < 2:
        return pd.DataFrame()

    prices = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    rets = prices.pct_change().dropna(how="all")
    if rets.empty:
        return pd.DataFrame()

    mu = rets.mean().reindex(tickers).fillna(0.0).values * TRADING_DAYS
    cov = rets.cov().reindex(index=tickers, columns=tickers).fillna(0.0).values * TRADING_DAYS
    hhi_vec = ranked_df.set_index("ETF").reindex(tickers)["Sector HHI"].fillna(1.0).values

    best_w = None
    best_obj = -1e18
    rng = np.random.default_rng(42)
    for _ in range(n_portfolios):
        w = rng.random(len(tickers))
        w = w / w.sum()
        if np.max(w) > max_weight:
            continue

        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
        sharpe = (port_ret - risk_free) / max(port_vol, 1e-9)
        penalty = sector_penalty * float(w @ hhi_vec)
        obj = sharpe - penalty
        if obj > best_obj:
            best_obj = obj
            best_w = w

    if best_w is None:
        return pd.DataFrame()

    w = best_w
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
    sharpe = (port_ret - risk_free) / max(port_vol, 1e-9)

    out = pd.DataFrame({
        "ETF": tickers,
        "Weight": w,
        "ETF Return": mu,
        "ETF Sector HHI": hhi_vec,
    }).sort_values("Weight", ascending=False).reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "Portfolio Annual Return": port_ret,
                "Portfolio Annual Volatility": port_vol,
                "Portfolio Sharpe-like": sharpe,
                "Objective (Sharpe-Penalty)": best_obj,
            }
        ]
    )
    return out, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank ETFs by return/risk with optional Sharpe and sector diversification.")
    p.add_argument("--etfs", nargs="+", required=True, help="ETF tickers, e.g. SPY QQQ IWM XLE")
    p.add_argument("--period", default="1y")
    p.add_argument("--top-components", type=int, default=10)
    p.add_argument("--best", type=int, default=5)
    p.add_argument("--score-mode", choices=["return_risk", "sharpe"], default="sharpe")
    p.add_argument("--risk-free", type=float, default=0.02, help="Annual risk-free rate, decimal (default 0.02)")
    p.add_argument("--sector-penalty", type=float, default=0.20, help="Penalty multiplier on sector concentration HHI")
    p.add_argument("--optimize", action="store_true", help="Run portfolio optimizer on top-ranked ETFs")
    p.add_argument("--max-weight", type=float, default=0.5, help="Max single ETF weight in optimizer")
    p.add_argument("--n-portfolios", type=int, default=5000, help="Random portfolios sampled in optimizer")
    p.add_argument("--csv", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = rank_etfs(
        args.etfs,
        period=args.period,
        top_n=args.top_components,
        best_n=args.best,
        score_mode=args.score_mode,
        risk_free=args.risk_free,
        sector_penalty=args.sector_penalty,
    )
    if df.empty:
        print("No ETF scores produced. Check ETF symbols and data availability.")
        return

    out = df.copy()
    for c in ["Annual Return", "Annual Volatility"]:
        out[c] = (out[c] * 100).map(lambda x: f"{x:.2f}%")
    out["Sector HHI"] = out["Sector HHI"].map(lambda x: f"{x:.3f}")
    out["Sharpe-like"] = out["Sharpe-like"].map(lambda x: f"{x:.3f}")
    out["Score"] = out["Score"].map(lambda x: f"{x:.3f}")

    print("\nETF Ranking")
    print(tabulate(out, headers="keys", tablefmt="github", showindex=False))

    if args.optimize:
        opt_weights, opt_summary = optimize_portfolio(
            df, args.period, args.risk_free, args.sector_penalty, args.max_weight, args.n_portfolios
        )
        if isinstance(opt_weights, pd.DataFrame) and not opt_weights.empty:
            wshow = opt_weights.copy()
            wshow["Weight"] = (wshow["Weight"] * 100).map(lambda x: f"{x:.2f}%")
            wshow["ETF Return"] = (wshow["ETF Return"] * 100).map(lambda x: f"{x:.2f}%")
            wshow["ETF Sector HHI"] = wshow["ETF Sector HHI"].map(lambda x: f"{x:.3f}")
            print("\nOptimized Weights (top-ranked ETFs)")
            print(tabulate(wshow, headers="keys", tablefmt="github", showindex=False))

            sshow = opt_summary.copy()
            sshow["Portfolio Annual Return"] = (sshow["Portfolio Annual Return"] * 100).map(lambda x: f"{x:.2f}%")
            sshow["Portfolio Annual Volatility"] = (sshow["Portfolio Annual Volatility"] * 100).map(lambda x: f"{x:.2f}%")
            sshow["Portfolio Sharpe-like"] = sshow["Portfolio Sharpe-like"].map(lambda x: f"{x:.3f}")
            sshow["Objective (Sharpe-Penalty)"] = sshow["Objective (Sharpe-Penalty)"].map(lambda x: f"{x:.3f}")
            print("\nOptimized Portfolio Summary")
            print(tabulate(sshow, headers="keys", tablefmt="github", showindex=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved raw numeric ETF ranking results to: {args.csv}")


if __name__ == "__main__":
    main()
