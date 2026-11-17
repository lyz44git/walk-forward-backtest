"""Rebalancing rules: accounting, causality and neutral-parameter identities on synthetic data.

The last two tests use the cached targets of the real backtest and are skipped when the cache is absent
(for example in continuous integration, where no market data is downloaded).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmb.rebalance import cases, env, rules  # noqa: E402
from qmb.rebalance.splits import split_indices  # noqa: E402

CACHED = sorted(cases.TMP.glob("case_*.npz"))


def synthetic(T=120, N=25, seed=0, missing=True):
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((T, N))
    M = np.ones((T, N), bool)
    if missing:
        M[:40, :3] = False                      # three assets list late
    S = np.where(M, S, 0.0)
    R = np.where(M, 0.02 * rng.standard_normal((T, N)), 0.0)
    return S, M, R


class FakeMarket:
    def __init__(self, S, M, R):
        self.S, self.M, self.R = S, M, R
        self.W = rules.quintile_targets(S, M)


def test_accounting_with_a_forced_close():
    h, net, gross, turnover, short = env.step_accounting(
        d=np.array([0.4, -0.4, 0.2]), W=np.array([1.0, -1.0, 0.0]), M=np.array([True, True, False]),
        r=np.array([0.1, 0.03, 0.2]), eta=0.5)
    np.testing.assert_allclose(h, [0.7, -0.7, 0.0])
    assert turnover == pytest.approx(0.8) and short == pytest.approx(0.7) and gross == pytest.approx(0.049)
    assert net == pytest.approx(0.049 - 0.8 * 0.0005 - 0.7 * 0.005 / 52)


def test_neutral_parameters_return_the_quintile_targets():
    S, M, _ = synthetic()
    W = rules.quintile_targets(S, M)
    np.testing.assert_array_equal(rules.buffered_targets(S, M, 1.0), W)
    np.testing.assert_array_equal(rules.quintile_targets(rules.smoothed_scores(S, M, 0), M), W)


@pytest.mark.parametrize("multiple", [1.0, 1.5, 2.0])
def test_buffer_keeps_both_legs_full_and_disjoint(multiple):
    S, M, _ = synthetic()
    W = rules.buffered_targets(S, M, multiple)
    k = np.maximum(1, np.rint(0.2 * M.sum(1)))
    np.testing.assert_allclose(np.where(W > 0, W, 0).sum(1), 1.0)
    np.testing.assert_allclose(np.where(W < 0, -W, 0).sum(1), 1.0)
    np.testing.assert_array_equal((W > 0).sum(1), k)
    np.testing.assert_array_equal((W < 0).sum(1), k)
    assert not (W[~M] != 0).any()               # nothing held in an asset that is not available


def test_wider_buffer_and_longer_half_life_trade_less():
    S, M, R = synthetic(T=300, seed=1)
    mkt = FakeMarket(S, M, R)
    def turnover(targets):
        return env.simulate(mkt, 60, 300, eta_const=1.0, targets=targets)["turnover"].mean()
    full = turnover(mkt.W)
    assert turnover(rules.buffered_targets(S, M, 1.5)) < full
    assert turnover(rules.buffered_targets(S, M, 2.0)) < turnover(rules.buffered_targets(S, M, 1.5))
    assert turnover(rules.quintile_targets(rules.smoothed_scores(S, M, 4), M)) < full


@pytest.mark.parametrize("transform", [
    lambda S, M: rules.buffered_targets(S, M, 1.75),
    lambda S, M: rules.quintile_targets(rules.smoothed_scores(S, M, 4), M),
])
def test_targets_do_not_use_later_predictions(transform):
    S, M, _ = synthetic()
    changed = S.copy()
    changed[80:] = -changed[80:] + 3.0          # rewrite every prediction from week 80 on
    np.testing.assert_array_equal(transform(S, M)[:80], transform(changed, M)[:80])
    assert not np.array_equal(transform(S, M)[80:], transform(changed, M)[80:])


def test_zero_cost_net_equals_gross_and_full_rule_matches_eta_one():
    S, M, R = synthetic(seed=2)
    mkt = FakeMarket(S, M, R)
    free = env.simulate(mkt, 50, 120, eta_const=1.0, cost_bps=0.0, borrow_bps=0.0)
    np.testing.assert_allclose(free["net"], free["gross"], atol=1e-15)
    paid = env.simulate(mkt, 50, 120, eta_const=1.0)
    np.testing.assert_allclose(paid["gross"], free["gross"])
    np.testing.assert_allclose(free["gross"] - paid["net"],
                               paid["turnover"] * 5e-4 + paid["short"] * 0.005 / 52, atol=1e-15)
    band0 = env.simulate(mkt, 50, 120, band=0.0)
    np.testing.assert_allclose(band0["net"], paid["net"], atol=1e-15)


def test_a_band_as_wide_as_a_position_lets_the_book_run_off():
    """The degenerate case that the exposure constraint excludes (design/rebalancing_v2_plan.md): the band
    never opens a position and closes one when its target flips sign, so exposure decays towards zero."""
    S, M, R = synthetic(T=200, seed=3, missing=False)
    mkt = FakeMarket(S, M, R)
    wide = env.simulate(mkt, 50, 200, band=1.0 / 5)       # 25 assets, 5 names per leg
    assert wide["short"][-50:].mean() < 0.1
    assert env.simulate(mkt, 50, 200, eta_const=1.0)["short"].mean() == pytest.approx(1.0)


@pytest.mark.skipif(not CACHED, reason="no cached targets (scripts/07_build_targets.py)")
@pytest.mark.parametrize("path", CACHED, ids=[p.stem for p in CACHED])
def test_full_rebalancing_reproduces_the_backtest(path):
    case = cases.load_case(path)
    ref = pd.read_csv(ROOT / "reports" / "ls_weekly.csv", parse_dates=["date"])
    ref = ref[(ref.dataset == case["dataset"]) & (ref.signal == case["signal"])].sort_values("date")
    np.testing.assert_array_equal(case["signal_date"], ref.date.to_numpy(dtype="datetime64[ns]"))
    np.testing.assert_allclose(env.full_rebalance(case)[:, :3], ref[["net", "gross", "turnover"]],
                               atol=1e-12, rtol=0)
    np.testing.assert_array_equal(rules.quintile_targets(case["S"], case["M"]), case["W"])


@pytest.mark.skipif(not CACHED, reason="no cached targets (scripts/07_build_targets.py)")
def test_policy_state_uses_only_completed_weeks():
    import copy
    case = cases.load_case(cases.TMP / "case_etf_ridge.npz")
    lo, hi, test, _ = split_indices(case)
    done = env.last_completed(case)[test]
    assert case["exit"][done] <= case["signal_date"][test] < case["exit"][done + 1]
    changed = copy.deepcopy(case)
    changed["R"][done + 1:] += 0.25; changed["IC"][done + 1:] = -0.9
    changed["W"][test + 1:] *= -1; changed["S"][test + 1:] *= -1; changed["DISP"][test + 1:] *= 2
    a, b = env.Market(case, np.arange(lo, hi)), env.Market(changed, np.arange(lo, hi))
    h = case["W"][test - 1].copy()
    np.testing.assert_array_equal(a.obs(test, h), b.obs(test, h))
