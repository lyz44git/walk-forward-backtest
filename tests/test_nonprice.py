"""Non-price features: availability dates, neutral filling of missing values, and the refit log."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qmb import data as qdata, engine, nonprice  # noqa: E402

HAVE_DATA = (qdata.RAW / "french" / nonprice.MONTHLY_ZIP).exists() and (qdata.CLEAN / "french_vw_daily.parquet").exists()


def setup(seed=0, n_assets=20, n_days=1600):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    r = pd.DataFrame(rng.normal(0, 0.01, (n_days, n_assets)), index=dates, columns=[f"A{i}" for i in range(n_assets)])
    tri = (1 + r).cumprod()
    return tri, engine.weekly_schedule(dates)


def test_annual_value_is_used_from_july_of_its_year():
    table = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=pd.period_range("2010", "2012", freq="Y"))
    dates = pd.DatetimeIndex(["2011-06-30", "2011-07-01", "2012-06-29", "2012-07-02"])
    assert nonprice.annual_to_daily(table, dates)["x"].tolist() == [1.0, 2.0, 2.0, 3.0]
    assert nonprice.annual_to_daily(table, dates, extra_years=1)["x"].tolist()[1:] == [1.0, 1.0, 2.0]


def test_monthly_value_is_used_only_after_the_end_of_the_following_month():
    table = pd.DataFrame({"x": np.arange(1.0, 7.0)}, index=pd.period_range("2011-01", "2011-06", freq="M"))
    dates = pd.DatetimeIndex(["2011-02-28", "2011-03-01", "2011-03-31", "2011-04-01"])
    out = nonprice.monthly_to_daily(table, dates)["x"]
    assert np.isnan(out.iloc[0]) and out.tolist()[1:] == [1.0, 1.0, 2.0]          # January first used in March
    assert nonprice.monthly_to_daily(table, dates + pd.DateOffset(months=3), extra_months=3)["x"].tolist()[1:] == [1.0, 1.0, 2.0]


def test_changing_a_later_annual_value_leaves_earlier_dates_unchanged():
    table = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=pd.period_range("2010", "2012", freq="Y"))
    dates = pd.bdate_range("2010-07-01", "2012-12-31")
    changed = table.copy()
    changed.loc[pd.Period("2012", freq="Y"), "x"] = 99.0
    a, b = nonprice.annual_to_daily(table, dates), nonprice.annual_to_daily(changed, dates)
    before = dates < "2012-07-01"
    pd.testing.assert_frame_equal(a[before], b[before])
    assert (b[~before]["x"] == 99.0).all()


def test_missing_new_feature_is_neutral_and_keeps_the_part_1_rows():
    tri, sch = setup()
    base = engine.daily_features(tri)
    extra = pd.DataFrame(np.random.default_rng(1).normal(size=tri.shape), index=tri.index, columns=tri.columns)
    extra.iloc[:, :3] = np.nan                                       # three assets never have the new feature
    old = engine.build_panel(tri, tri, base, sch)
    new = engine.build_panel(tri, tri, {**base, "extra": extra}, sch, neutral=("extra",))
    assert new.X.index.equals(old.X.index)
    pd.testing.assert_frame_equal(new.X[list(base)], old.X)
    assert (new.X["extra"].unstack("asset").iloc[:, :3] == 0.0).all().all()
    strict = engine.build_panel(tri, tri, {**base, "extra": extra}, sch)
    assert len(strict.X) < len(old.X)                                # without the option the rows are dropped


def test_refit_log_has_one_entry_per_refit_and_does_not_change_predictions():
    tri, sch = setup(n_days=900)
    panel = engine.build_panel(tri, tri, engine.daily_features(tri), sch)
    log = []
    with_log = engine.walk_forward(panel, "ridge", 60, fit_log=log)
    pd.testing.assert_series_equal(with_log, engine.walk_forward(panel, "ridge", 60))
    weeks = with_log.index.get_level_values("week").unique()
    assert [k for k, _ in log] == list(weeks[::4])
    assert all(len(v) == panel.X.shape[1] for _, v in log)


@pytest.mark.skipif(not HAVE_DATA, reason="needs the downloaded French files")
def test_timely_book_to_market_equals_the_annual_value_minus_the_return_since_december():
    f = nonprice.industry_features()
    vw = pd.read_parquet(qdata.CLEAN / "french_vw_daily.parquet")
    day, col = pd.Timestamp("2015-09-30"), "Banks"
    since = np.log1p(vw.loc["2015-01-01":day, col]).sum()            # 2015 value: December 2014 market equity
    assert f["log_bm_timely"].loc[day, col] == pytest.approx(f["log_bm"].loc[day, col] - since)
    bm = qdata.french_table(nonprice.MONTHLY_ZIP, "Sum of BE / Sum of ME", "%Y")
    assert f["log_bm"].loc[day, col] == pytest.approx(np.log(bm.loc[pd.Period("2015", freq="Y"), col]))
    assert f["log_bm"].loc["2015-06-30", col] == pytest.approx(np.log(bm.loc[pd.Period("2014", freq="Y"), col]))


@pytest.mark.skipif(not (qdata.RAW / "yahoo" / "SPY.csv").exists(), reason="needs the downloaded Yahoo files")
def test_carry_is_trailing_distributions_over_price_and_is_centred_within_class():
    tickers = sorted(nonprice.ETF_CLASS)
    f = nonprice.etf_features(tickers)
    raw = qdata.load_etf_raw("SPY")
    day = pd.Timestamp("2020-06-30")
    window = raw.loc[:day].iloc[-252:]
    assert f["carry_12m"].loc[day, "SPY"] == pytest.approx(window["Dividends"].sum() / raw.loc[day, "Close"])
    classes = pd.Series(nonprice.ETF_CLASS)
    sums = f["carry_12m_in_class"].loc[day].groupby(classes).sum()
    assert np.abs(sums).max() < 1e-12
