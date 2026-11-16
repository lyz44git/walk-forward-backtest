"""Data-quality checks on both raw datasets; writes clean parquet panels and reports/data_quality.json."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from qmb import data
TICKERS = ["SPY", "QQQ", "IWM", "DIA", "MDY", "EFA", "EEM", "VGK", "EWJ", "FXI", "XLB", "XLE", "XLF", "XLI", "XLK",
           "XLP", "XLU", "XLV", "XLY", "IYR", "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "GLD", "SLV", "DBC", "USO"]
data.CLEAN.mkdir(parents=True, exist_ok=True)
rep = {"etf": {}, "french": {}}

# ---------------------------------------------------------------- ETFs
spy_days = data.load_etf_raw("SPY").index
issues = []
for t in TICKERS:
    d = data.load_etf_raw(t)
    f = d["Adj Close"] / d["Close"]
    r = d["Adj Close"].pct_change()
    split_days = d.index[d["Stock Splits"] != 0]
    cal = spy_days[(spy_days >= d.index.min()) & (spy_days <= d.index.max())]
    jump_at_split = [float(r.loc[s]) for s in split_days if s in r.index]
    gap = d["Open"] * f / (d["Adj Close"].shift(1)) - 1
    rec = {"first": str(d.index.min().date()), "last": str(d.index.max().date()), "rows": len(d),
           "missing_vs_spy_calendar": int(len(cal.difference(d.index))),
           "nonpositive_prices": int((d[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any(axis=1).sum()),
           "nan_rows": int(d[["Open", "Close", "Adj Close"]].isna().any(axis=1).sum()),
           "zero_volume_days": int((d["Volume"] == 0).sum()),
           "adj_factor_decreases": int((f.diff() < -1e-5).sum()),          # factor rises over time; |changes| ~1e-6 are float rounding
           "dividend_days": int((d["Dividends"] > 0).sum()), "split_days": [str(s.date()) for s in split_days],
           "abs_return_at_split_days": jump_at_split,
           "days_abs_return_gt_20pct": int((r.abs() > 0.20).sum()),
           "max_abs_daily_return": float(r.abs().max()), "max_abs_overnight_gap": float(gap.abs().max())}
    rep["etf"][t] = rec
    if rec["nonpositive_prices"] or rec["nan_rows"] or rec["adj_factor_decreases"] > 5 or rec["missing_vs_spy_calendar"] > 20:
        issues.append(t)
P = data.etf_panel(TICKERS)
P.to_parquet(data.CLEAN / "etf_daily.parquet", index=False)
common_start = max(pd.Timestamp(v["first"]) for v in rep["etf"].values())
rep["etf_summary"] = {"tickers": len(TICKERS), "common_start": str(common_start.date()),
                      "last": str(P.date.max().date()), "flagged": issues}

# ---------------------------------------------------------------- Ken French
I = data.french_industries_vw(); F = data.french_factors()
I.to_parquet(data.CLEAN / "french_vw_daily.parquet"); F.to_parquet(data.CLEAN / "ff_daily.parquet")
first_valid = I.apply(lambda s: s.first_valid_index())
rep["french"] = {"rows": len(I), "first": str(I.index.min().date()), "last": str(I.index.max().date()),
                 "industries": I.shape[1], "industries_complete_from_1970": int(I.loc["1970":].notna().all().sum()),
                 "latest_industry_start": str(first_valid.max().date()), "latest_industry": str(first_valid.idxmax()),
                 "days_abs_return_gt_20pct": int((I.abs() > 0.20).sum().sum()),
                 "missing_after_1970": int(I.loc["1970":].isna().sum().sum()),
                 "factors_first": str(F.index.min().date()), "factors_last": str(F.index.max().date()),
                 "date_mismatch_industries_vs_factors": int(len(I.index.symmetric_difference(F.index)))}
json.dump(rep, open(ROOT / "reports" / "data_quality.json", "w"), indent=1)
E = pd.DataFrame(rep["etf"]).T
pd.set_option("display.width", 220)
print(E[["first", "rows", "missing_vs_spy_calendar", "zero_volume_days", "adj_factor_decreases", "dividend_days",
         "split_days", "days_abs_return_gt_20pct", "max_abs_daily_return", "max_abs_overnight_gap"]].to_string())
print(json.dumps(rep["etf_summary"], indent=1)); print(json.dumps(rep["french"], indent=1))
