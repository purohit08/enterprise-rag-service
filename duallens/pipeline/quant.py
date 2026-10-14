"""§2 Quantitative lens — stock performance & financials (Yahoo Finance).

Mirrors notebook cells 2a/2b: 3-year price momentum and a financial-metrics
table. Returns plain data (dicts / DataFrame) so the orchestrator or a UI can
render it however it likes.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd
import yfinance as yf

from duallens.pipeline import config


def price_performance(period: str = "3y") -> Dict[str, Dict]:
    """Start/end price and total return % over `period`, per company (normalised
    so companies are comparable regardless of share price)."""
    out: Dict[str, Dict] = {}
    for ticker, name in config.COMPANIES.items():
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            out[name] = {"error": "no price data"}
            continue
        start, end = float(hist["Close"].iloc[0]), float(hist["Close"].iloc[-1])
        out[name] = {
            "start_price": round(start, 2),
            "end_price": round(end, 2),
            "total_return_pct": round((end / start - 1) * 100, 1),
        }
    return out


# Metric label -> yfinance `.info` key (notebook cell 2b)
METRICS = {
    "Market Cap": "marketCap",
    "P/E Ratio": "trailingPE",
    "Dividend Yield": "dividendYield",
    "Beta": "beta",
    "Total Revenue": "totalRevenue",
}


def financial_metrics() -> pd.DataFrame:
    """A companies × metrics DataFrame (Market Cap, P/E, Dividend Yield, Beta, Revenue)."""
    rows = {}
    for ticker, name in config.COMPANIES.items():
        info = yf.Ticker(ticker).info
        rows[name] = {label: info.get(key) for label, key in METRICS.items()}
    return pd.DataFrame(rows).T


def metrics_table_str() -> str:
    """The metrics table as text — what the ranking prompt consumes."""
    return financial_metrics().to_string()
