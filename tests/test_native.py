"""C++ kernels against the Python reference implementations.

The bootstrap kernel must reproduce qmb.stats.stationary_bootstrap_sharpe and the confidence intervals
published in reports/main_results.csv; the accounting kernel must reproduce qmb.rebalance.env.simulate.
Tests are skipped when the library cannot be built, unless QMB_REQUIRE_NATIVE=1 (set in continuous
integration, where a missing compiler must fail the run).
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmb import native, stats  # noqa: E402
from qmb.rebalance import env, rules  # noqa: E402

TOLERANCE = 1e-12


@pytest.fixture(scope="module", autouse=True)
def kernels():
    try:
        return native.load()
    except native.NativeUnavailable as error:
        if os.environ.get("QMB_REQUIRE_NATIVE") == "1":
            raise
        pytest.skip(str(error))


def reference_indices(n, block, draws, seed=0):
    """The index walk of scripts/09_rebalancing_rules.py, written out in Python."""
    rng = np.random.default_rng(seed)
    idx = np.empty((draws, n), dtype=np.int64)
    for path in idx:
        path[0] = rng.integers(n)
        restart = rng.random(n) < 1.0 / block
        jump = rng.integers(n, size=n)
        for i in range(1, n):
            path[i] = jump[i] if restart[i] else (path[i - 1] + 1) % n
    return idx


def synthetic_market(T=160, N=25, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((T, N))
    M = np.ones((T, N), bool)
    M[:40, :3] = False                          # three assets list late
    M[90:110, 7] = False                        # one asset leaves and returns
    S = np.where(M, S, 0.0)
    R = np.where(M, 0.02 * rng.standard_normal((T, N)), 0.0)
    return SimpleNamespace(S=S, M=M, R=R, W=rules.quintile_targets(S, M))


@pytest.mark.parametrize("block", [1, 4, 8, 16])
def test_index_paths_equal_the_reference_walk(block):
    n, draws = 137, 40
    np.testing.assert_array_equal(native.bootstrap_indices(n, block, draws), reference_indices(n, block, draws))


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_interval_matches_python_on_synthetic_returns(seed):
    r = pd.Series(np.random.default_rng(100 + seed).normal(0.001, 0.02, 400))
    expected = stats.stationary_bootstrap_sharpe(r, draws=300, seed=seed)
    np.testing.assert_allclose(native.stationary_bootstrap_sharpe(r, draws=300, seed=seed), expected,
                               rtol=0, atol=TOLERANCE)


def test_missing_weeks_are_dropped_like_the_reference():
    r = pd.Series(np.random.default_rng(5).normal(0.0, 0.01, 200))
    r.iloc[[3, 50, 51]] = np.nan
    np.testing.assert_allclose(native.stationary_bootstrap_sharpe(r, draws=100),
                               stats.stationary_bootstrap_sharpe(r, draws=100), rtol=0, atol=TOLERANCE)


def test_resampled_sharpes_equal_a_direct_calculation():
    x = np.random.default_rng(2).normal(0.002, 0.03, 150)
    idx = reference_indices(len(x), 8, 60)
    paths = x[idx]
    expected = paths.mean(axis=1) / paths.std(axis=1, ddof=1) * np.sqrt(stats.PER_YEAR)
    np.testing.assert_allclose(native.bootstrap_sharpes(x, mean_block=8, draws=60), expected, rtol=0, atol=TOLERANCE)


def test_a_constant_series_has_no_sharpe_ratio():
    assert np.isnan(native.bootstrap_sharpes(np.full(50, 0.01), draws=10)).all()


def test_published_intervals_are_reproduced():
    """All eight dataset-signal pairs of Part 1: 2,000 draws on the weekly net returns in reports/."""
    weekly = pd.read_csv(ROOT / "reports" / "ls_weekly.csv")
    published = pd.read_csv(ROOT / "reports" / "main_results.csv").set_index(["dataset", "signal"])
    assert len(published) == 8
    for key, group in weekly.groupby(["dataset", "signal"]):
        low, high = native.stationary_bootstrap_sharpe(group["net"])
        row = published.loc[key]
        assert low == pytest.approx(row.ls_net_sharpe_ci_lo, abs=1e-10)
        assert high == pytest.approx(row.ls_net_sharpe_ci_hi, abs=1e-10)


@pytest.mark.parametrize("cost", [0.0, 5.0, 10.0])
@pytest.mark.parametrize("eta", [1.0, 0.35, 0.05])
def test_partial_adjustment_matches_the_python_simulator(eta, cost):
    m = synthetic_market()
    expected = env.simulate(m, 41, 160, eta_const=eta, cost_bps=cost)
    got = native.simulate(m.W, m.R, m.M, 41, 160, eta_const=eta, cost_bps=cost)
    for name in ("net", "gross", "turnover", "short"):
        np.testing.assert_allclose(got[name], expected[name], rtol=0, atol=TOLERANCE, err_msg=name)


@pytest.mark.parametrize("width", [0.0, 0.02, 0.2])
def test_no_trade_band_matches_the_python_simulator(width):
    m = synthetic_market(seed=3)
    expected = env.simulate(m, 1, 160, band=width)
    got = native.simulate(m.W, m.R, m.M, 1, 160, band=width)
    for name in ("net", "gross", "turnover", "short"):
        np.testing.assert_allclose(got[name], expected[name], rtol=0, atol=TOLERANCE, err_msg=name)


def test_a_band_that_varies_by_week_matches_the_python_simulator():
    m = synthetic_market(seed=4)
    names = np.maximum(1, np.rint(rules.QUANTILE * m.M.sum(axis=1)))          # as in scripts/09
    band = 0.5 / names[20:120]
    expected = env.simulate(m, 20, 120, band=band)
    got = native.simulate(m.W, m.R, m.M, 20, 120, band=band)
    for name in ("net", "gross", "turnover", "short"):
        np.testing.assert_allclose(got[name], expected[name], rtol=0, atol=TOLERANCE, err_msg=name)


def test_rank_buffer_targets_can_be_simulated():
    m = synthetic_market(seed=6)
    targets = rules.buffered_targets(m.S, m.M, 1.5)
    expected = env.simulate(m, 41, 160, eta_const=1.0, targets=targets)
    got = native.simulate(targets, m.R, m.M, 41, 160, eta_const=1.0)
    np.testing.assert_allclose(got["net"], expected["net"], rtol=0, atol=TOLERANCE)


def test_full_rebalancing_ends_on_the_target_and_absent_assets_are_closed():
    m = synthetic_market()
    got = native.simulate(m.W, m.R, m.M, 95, 100, eta_const=1.0)
    np.testing.assert_allclose(got["holdings"], m.W[99], rtol=0, atol=TOLERANCE)
    assert got["holdings"][7] == 0.0 and not m.M[99, 7]


def test_invalid_spans_and_arguments_are_rejected():
    m = synthetic_market()
    with pytest.raises(ValueError):
        native.simulate(m.W, m.R, m.M, 0, 10, eta_const=1.0)            # week t0 - 1 does not exist
    with pytest.raises(ValueError):
        native.simulate(m.W, m.R, m.M, 10, 161, eta_const=1.0)          # past the last week
    with pytest.raises(ValueError):
        native.simulate(m.W, m.R, m.M, 10, 20)                          # neither eta nor band
    with pytest.raises(ValueError):
        native.simulate(m.W, m.R[:, :5], m.M, 10, 20, eta_const=1.0)    # shapes differ
