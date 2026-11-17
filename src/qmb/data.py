"""Loaders for the two raw datasets, returning tidy daily panels.

ETF panel (Yahoo): yfinance `history(auto_adjust=False)` gives split-adjusted OHLC, a split- and
dividend-adjusted close ("Adj Close"), and the dividend/split actions. The day's total-return factor
f_t = AdjClose_t / Close_t is applied to the open as well, so open-to-open returns include dividends.

Industry panel (Ken French): the value-weighted section of the 49-industry daily file, in percent;
-99.99 and -999 mark missing values. The Fama-French daily file supplies RF and Mkt-RF (percent).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"


def load_etf_raw(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / "yahoo" / f"{ticker}.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def etf_panel(tickers: list[str]) -> pd.DataFrame:
    """Long panel: date, ticker, open_adj, close_adj, close_raw, volume, dollar_volume."""
    rows = []
    for t in tickers:
        d = load_etf_raw(t)
        f = d["Adj Close"] / d["Close"]
        rows.append(pd.DataFrame({
            "date": d.index, "ticker": t,
            "open_adj": (d["Open"] * f).to_numpy(), "close_adj": d["Adj Close"].to_numpy(),
            "close_raw": d["Close"].to_numpy(), "volume": d["Volume"].to_numpy(),
            "dollar_volume": (d["Close"] * d["Volume"]).to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def _french_section(text: str, title: str) -> pd.DataFrame:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if title in l)
    header = lines[start + 1].split(",")
    data = []
    for l in lines[start + 2:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) != len(header) or not parts[0].isdigit():
            break
        data.append(parts)
    df = pd.DataFrame(data, columns=["date"] + [h.strip() for h in header[1:]])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float)
    return df.where(~df.isin([-99.99, -999.0]))


def french_industries_vw() -> pd.DataFrame:
    """Wide daily table (date x 49 industries) of value-weighted returns in decimal."""
    z = zipfile.ZipFile(RAW / "french" / "49_Industry_Portfolios_daily_CSV.zip")
    text = z.read(z.namelist()[0]).decode("latin-1")
    return _french_section(text, "Average Value Weighted Returns -- Daily") / 100.0


def french_factors() -> pd.DataFrame:
    """Daily Mkt-RF, SMB, HML, RF in decimal."""
    z = zipfile.ZipFile(RAW / "french" / "F-F_Research_Data_Factors_daily_CSV.zip")
    text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(",Mkt-RF"))
    rows = []
    for l in lines[start + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) != 5 or not parts[0].isdigit():
            break
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date", "mkt_rf", "smb", "hml", "rf"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").astype(float) / 100.0


def french_table(zip_name: str, title: str, date_format: str) -> pd.DataFrame:
    """One titled section of a French industry file as a wide table; missing-value codes become NaN.

    date_format is "%Y%m%d" (daily), "%Y%m" (monthly) or "%Y" (annual); monthly and annual tables are
    indexed by period."""
    z = zipfile.ZipFile(RAW / "french" / zip_name)
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == title)
    header = [h.strip() for h in lines[start + 1].split(",")]
    data = []
    for l in lines[start + 2:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) != len(header) or not parts[0].isdigit():
            break
        data.append(parts)
    df = pd.DataFrame(data, columns=["date"] + header[1:])
    dates = pd.to_datetime(df.pop("date"), format=date_format)
    df = df.astype(float)
    df.index = dates if date_format == "%Y%m%d" else pd.PeriodIndex(dates, freq="M" if date_format == "%Y%m" else "Y")
    return df.where(~df.isin([-99.99, -999.0]))
