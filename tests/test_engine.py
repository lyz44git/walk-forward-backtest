"""Engine tests on synthetic data (test fixtures only; no reported result uses them)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qmb import engine, stats  # noqa: E402


def synthetic(n_assets=20, n_days=1600, seed=0, planted=0.0):
    """Random-walk total-return indices. If planted > 0, each asset's return over the NEXT holding week is
    partly predictable from a value observable at the signal date (stored as an extra 'hint' feature)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    r = pd.DataFrame(rng.normal(0, 0.01, (n_days, n_assets)), index=dates, columns=[f"A{i}" for i in range(n_assets)])
    return dates, r


def test_schedule_entry_after_signal_and_exit_is_next_entry():
    dates = pd.bdate_range("2020-01-01", periods=60)
    s = engine.weekly_schedule(dates)
    assert (s.entry > s.signal).all()
    assert (s.exit.iloc[:-1].to_numpy() == s.entry.iloc[1:].to_numpy()).all()
    assert (pd.DatetimeIndex(s.signal).dayofweek == 4).all()          # Fridays in a business-day calendar


def test_features_do_not_use_future_prices():
    dates, r = synthetic()
    tri = (1 + r).cumprod()
    f1 = engine.daily_features(tri)
    cut = dates[1000]
    tri2 = tri.copy(); tri2.loc[tri2.index > cut] *= 5.0               # change only the future
    f2 = engine.daily_features(tri2)
    for k in f1:
        pd.testing.assert_frame_equal(f1[k].loc[:cut], f2[k].loc[:cut])


def test_target_is_return_from_entry_to_exit():
    dates, r = synthetic()
    tri = (1 + r).cumprod()
    sch = engine.weekly_schedule(dates)
    p = engine.build_panel(tri, tri, engine.daily_features(tri), sch)
    w, a = p.ret.index[100]
    expected = tri.loc[sch.exit[w], a] / tri.loc[sch.entry[w], a] - 1
    assert p.ret.loc[(w, a)] == pytest.approx(expected)


def test_purge_training_weeks_end_before_refit_signal():
    dates, r = synthetic()
    tri = (1 + r).cumprod()
    sch = engine.weekly_schedule(dates)
    p = engine.build_panel(tri, tri, engine.daily_features(tri), sch)
    pred = engine.walk_forward(p, "ridge", initial_weeks=60)          # asserts purge internally
    assert pred.index.get_level_values("week").min() >= p.X.index.get_level_values("week").min() + 60


def test_noise_gives_no_out_of_sample_skill():
    dates, r = synthetic(seed=3)
    tri = (1 + r).cumprod()
    sch = engine.weekly_schedule(dates)
    p = engine.build_panel(tri, tri, engine.daily_features(tri), sch)
    for sig in ["ridge", "lgbm", "mom_12_1"]:
        ic = stats.weekly_ic(engine.walk_forward(p, sig, initial_weeks=60), p.ret)
        assert abs(ic.mean()) < 0.03, (sig, ic.mean())


def test_planted_lagged_signal_is_found_and_future_leak_is_not_available():
    """Make next-week returns depend on ret_5d at the signal date (a legitimate, observable signal)."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2010-01-01", periods=1600)
    n = 20
    r = pd.DataFrame(rng.normal(0, 0.01, (len(dates), n)), index=dates, columns=[f"A{i}" for i in range(n)])
    tri = (1 + r).cumprod()
    sch = engine.weekly_schedule(dates)
    # plant: during each holding week, add drift proportional to minus the trailing 5-day return at the signal date
    f5 = tri / tri.shift(5) - 1
    for _, row in sch.iterrows():
        d = f5.loc[row.signal]
        if d.isna().any():
            continue
        span = (r.index > row.entry) & (r.index <= row.exit)
        r.loc[span] += (-0.5 * d.to_numpy() / span.sum())
    tri = (1 + r).cumprod()
    p = engine.build_panel(tri, tri, engine.daily_features(tri), sch)
    ic = stats.weekly_ic(engine.walk_forward(p, "ridge", initial_weeks=60), p.ret)
    assert ic.mean() > 0.2


def test_long_short_accounting():
    idx = pd.MultiIndex.from_product([[0, 1], ["a", "b", "c", "d", "e"]], names=["week", "asset"])
    pred = pd.Series([5, 4, 3, 2, 1, 5, 4, 3, 2, 1], index=idx, dtype=float)
    ret = pd.Series([0.10, 0, 0, 0, -0.10, 0, 0, 0, 0, 0], index=idx)
    port = engine.portfolio(pred, ret, quantile=0.2)
    assert port.loc[0, "ret_gross"] == pytest.approx(0.20)        # long a (+10%), short e (-10%)
    assert port.loc[0, "turnover"] == pytest.approx(2.0)
    # week 1: same names; drifted weights a: 1.1/1.2, e: -0.9/1.2 -> turnover |1-1.1/1.2| + |-1+0.9/1.2|
    assert port.loc[1, "turnover"] == pytest.approx(abs(1 - 1.1 / 1.2) + abs(-1 + 0.9 / 1.2))
    net = engine.net_returns(port, cost_bps=10, borrow_bps_per_year=52)
    assert net.loc[0] == pytest.approx(0.20 - 2.0 * 0.001 - 1.0 * 0.0001)


def test_newey_west_matches_iid_t_with_zero_lags():
    x = np.random.default_rng(1).normal(0.1, 1, 500)
    t0 = stats.newey_west_t(x, lags=0)
    assert t0 == pytest.approx(x.mean() / (x.std(ddof=0) / np.sqrt(len(x))), rel=1e-9)
