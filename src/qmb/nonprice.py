"""Features that are not functions of an asset's own past prices (design/nonprice_features_plan.md).

Every function returns a wide daily table (date x asset) whose row for date d uses only information
available on d under the timing rules of the plan; the engine ranks it on the signal dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qmb import data as qdata

MONTHLY_ZIP, DAILY_ZIP = "49_Industry_Portfolios_CSV.zip", "49_Industry_Portfolios_daily_CSV.zip"
INDUSTRY = ("log_bm", "log_bm_timely", "log_size", "firm_growth_12m", "breadth_63d")
ETF = ("carry_12m", "carry_12m_in_class")
SIGN = {"log_bm": 1, "log_bm_timely": 1, "log_size": -1, "firm_growth_12m": -1, "breadth_63d": 1,
        "carry_12m": 1, "carry_12m_in_class": 1}                      # published direction, for the single sorts
ETF_CLASS = {**dict.fromkeys(["SPY", "QQQ", "IWM", "DIA", "MDY"], "us_equity"),
             **dict.fromkeys(["EFA", "EEM", "VGK", "EWJ", "FXI"], "international_equity"),
             **dict.fromkeys(["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IYR"], "sector"),
             **dict.fromkeys(["TLT", "IEF", "SHY", "LQD", "HYG", "TIP"], "bond"),
             **dict.fromkeys(["GLD", "SLV", "DBC", "USO"], "commodity")}


def _log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.where(x > 0))


def annual_to_daily(table: pd.DataFrame, dates: pd.DatetimeIndex, extra_years: int = 0) -> pd.DataFrame:
    """Row for year t is used from 1 July of t to 30 June of t+1 (plus `extra_years`)."""
    year = np.where(dates.month >= 7, dates.year, dates.year - 1) - extra_years
    out = table.reindex(pd.PeriodIndex(year, freq="Y"))
    out.index = dates
    return out


def monthly_to_daily(table: pd.DataFrame, dates: pd.DatetimeIndex, extra_months: int = 0) -> pd.DataFrame:
    """Row for month m is used on dates after the end of month m+1 (plus `extra_months`)."""
    out = table.reindex(dates.to_period("M") - 2 - extra_months)
    out.index = dates
    return out


def industry_features(extra_years: int = 0, extra_months: int = 0) -> dict[str, pd.DataFrame]:
    vw = pd.read_parquet(qdata.CLEAN / "french_vw_daily.parquet")
    ew = qdata.french_table(DAILY_ZIP, "Average Equal Weighted Returns -- Daily", "%Y%m%d").reindex(vw.index) / 100.0
    dates = vw.index
    bm = _log(qdata.french_table(MONTHLY_ZIP, "Sum of BE / Sum of ME", "%Y"))
    firms = _log(qdata.french_table(MONTHLY_ZIP, "Number of Firms in Portfolios", "%Y%m"))
    size = _log(qdata.french_table(MONTHLY_ZIP, "Average Firm Size", "%Y%m"))

    log_bm = annual_to_daily(bm, dates, extra_years)
    level = np.log1p(vw).fillna(0.0).cumsum()
    year_end = level.groupby(dates.year).last()                      # log level on the last trading day of a year
    base_year = np.where(dates.month >= 7, dates.year, dates.year - 1) - extra_years - 1
    base = year_end.reindex(base_year)
    base.index = dates
    growth = firms - firms.shift(12)
    compound = lambda r: np.expm1(np.log1p(r).rolling(63, min_periods=63).sum())
    return {"log_bm": log_bm, "log_bm_timely": log_bm - (level - base),
            "log_size": monthly_to_daily(size, dates, extra_months),
            "firm_growth_12m": monthly_to_daily(growth, dates, extra_months),
            "breadth_63d": compound(ew) - compound(vw)}


def etf_features(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Trailing 252-day cash distributions (by ex-date; capital-gains distributions excluded) over the close."""
    carry = {}
    for t in tickers:
        d = qdata.load_etf_raw(t)
        carry[t] = d["Dividends"].rolling(252, min_periods=252).sum() / d["Close"]
    carry = pd.DataFrame(carry).sort_index()
    classes = pd.Series(ETF_CLASS).reindex(carry.columns)
    class_mean = carry.T.groupby(classes).transform("mean").T
    return {"carry_12m": carry, "carry_12m_in_class": carry - class_mean}
