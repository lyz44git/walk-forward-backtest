"""Download the two public datasets (run once; files are git-ignored and must not be redistributed).

1. Yahoo Finance via yfinance: daily OHLCV, adjusted close, dividends and splits for 30 liquid U.S.-listed ETFs.
2. Kenneth French data library (Dartmouth): daily returns of 49 industry portfolios, and the daily
   Fama-French factors (for the risk-free rate and the market factor).
"""
import io, sys, time, zipfile, urllib.request
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
YDIR, FDIR = ROOT / "data/raw/yahoo", ROOT / "data/raw/french"
TICKERS = ["SPY", "QQQ", "IWM", "DIA", "MDY",                    # U.S. equity indices
           "EFA", "EEM", "VGK", "EWJ", "FXI",                    # international equity
           "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IYR",   # U.S. sectors + REITs
           "TLT", "IEF", "SHY", "LQD", "HYG", "TIP",             # bonds
           "GLD", "SLV", "DBC", "USO"]                           # commodities
FRENCH = {"49_Industry_Portfolios_daily_CSV.zip": "49 industry portfolios, daily",
          "F-F_Research_Data_Factors_daily_CSV.zip": "Fama-French 3 factors + RF, daily"}
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
YDIR.mkdir(parents=True, exist_ok=True); FDIR.mkdir(parents=True, exist_ok=True)

for t in TICKERS:
    dst = YDIR / f"{t}.csv"
    if dst.exists():
        continue
    df = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=False, actions=True)
    if df.empty:
        sys.exit(f"no data for {t}")
    df.index = df.index.tz_localize(None).normalize()
    df.to_csv(dst)
    print(f"{t}: {len(df):,} rows {df.index.min().date()}..{df.index.max().date()}")
    time.sleep(0.5)

for name, desc in FRENCH.items():
    dst = FDIR / name
    if dst.exists():
        continue
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "research-download"})
    data = urllib.request.urlopen(req, timeout=60).read()
    zipfile.ZipFile(io.BytesIO(data)).testzip()
    dst.write_bytes(data)
    print(f"{name}: {len(data)/1e6:.1f} MB ({desc})")
print("done")
