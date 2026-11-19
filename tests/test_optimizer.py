"""Part 5 (design/portfolio_optimizer_plan.md): the risk model, the weekly problem and the selection rule."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmb.rebalance import env, optimizer as opt, rules  # noqa: E402


def synthetic_case(T=120, N=12, seed=0):
    rng = np.random.default_rng(seed)
    signal = np.datetime64("2010-01-04") + 7 * np.arange(T).astype("timedelta64[D]")
    S = rng.standard_normal((T, N))
    M = np.ones((T, N), bool)
    R = 0.02 * rng.standard_normal((T, N)) + 0.01 * rng.standard_normal((T, 1))
    return {"W": rules.quintile_targets(S, M), "R": R, "M": M, "S": S, "signal_date": signal,
            "entry": signal + np.timedelta64(1, "D"), "exit": signal + np.timedelta64(7, "D")}


def inputs(n=12, seed=1):
    rng = np.random.default_rng(seed)
    cov, _ = opt.ledoit_wolf(0.03 * rng.standard_normal((52, n)) + 0.02 * rng.standard_normal((52, 1)))
    values, vectors = np.linalg.eigh(cov)
    root = np.sqrt(values)[:, None] * vectors.T
    alpha = 0.04 * np.sqrt(np.diag(cov)) * rng.standard_normal(n)
    return cov, root, alpha


def test_ledoit_wolf_matches_scikit_learn():
    sklearn_covariance = pytest.importorskip("sklearn.covariance")
    rng = np.random.default_rng(3)
    for T, N in ((52, 49), (52, 30), (300, 8)):
        X = 0.03 * rng.standard_normal((T, N)) + 0.02 * rng.standard_normal((T, 1))
        cov, shrinkage = opt.ledoit_wolf(X)
        expected, expected_shrinkage = sklearn_covariance.ledoit_wolf(X)
        np.testing.assert_allclose(cov, expected, rtol=0, atol=1e-15)
        assert shrinkage == pytest.approx(expected_shrinkage, abs=1e-12)


def test_solution_satisfies_the_constraints():
    cov, root, alpha = inputs()
    present = np.ones(12, bool)
    present[4] = False
    problem = opt.WeeklyProblem(12)
    d = np.linspace(-0.2, 0.2, 12)
    for gamma in opt.GAMMAS:
        for kappa in (0.0, 5e-4, 2e-3):
            h, status = problem.solve(alpha, root, gamma, kappa, d, present)
            assert status == "optimal"
            assert abs(h.sum()) < 1e-7 and np.abs(h).sum() <= opt.GROSS + 1e-6 and h[4] == 0.0


def test_without_risk_or_cost_the_book_is_the_best_and_worst_name():
    _, root, alpha = inputs()
    h, _ = opt.WeeklyProblem(12).solve(alpha, root, 1e-9, 0.0, np.zeros(12), np.ones(12, bool))
    expected = np.zeros(12)
    expected[np.argmax(alpha)], expected[np.argmin(alpha)] = 1.0, -1.0
    np.testing.assert_allclose(h, expected, atol=1e-4)


def test_with_high_risk_aversion_the_book_is_the_closed_form_mean_variance_portfolio():
    cov, root, alpha = inputs()
    gamma = 1e4
    inv = np.linalg.inv(cov)
    ones = np.ones(12)
    lam = ones @ inv @ alpha / (ones @ inv @ ones)
    expected = inv @ (alpha - lam) / gamma
    assert np.abs(expected).sum() < opt.GROSS                       # the gross cap does not bind
    h, _ = opt.WeeklyProblem(12).solve(alpha, root, gamma, 0.0, np.zeros(12), np.ones(12, bool))
    np.testing.assert_allclose(h, expected, rtol=1e-5, atol=1e-9)


def test_a_higher_turnover_penalty_trades_less():
    case = synthetic_case()
    weeks = range(60, 120)
    alphas, roots, _ = opt.risk_inputs(case, weeks, ic=0.05)
    problem = opt.WeeklyProblem(12)
    turnover = [opt.simulate(case, alphas, roots, 60, 120, 3.0, m, problem)["turnover"].sum() for m in (0, 1, 4)]
    assert turnover[0] > turnover[1] > turnover[2]


def test_risk_inputs_use_only_weeks_completed_by_the_signal():
    case = synthetic_case()
    t = 80
    last = env.last_completed(case)[t]
    alphas, roots, _ = opt.risk_inputs(case, [t], ic=0.05)
    later = {**case, "R": case["R"].copy()}
    later["R"][last + 1:] += 1.0
    alphas2, roots2, _ = opt.risk_inputs(later, [t], ic=0.05)
    np.testing.assert_array_equal(alphas[t], alphas2[t])
    np.testing.assert_array_equal(roots[t], roots2[t])
    earlier = {**case, "R": case["R"].copy()}
    earlier["R"][last] += 0.05
    assert not np.array_equal(opt.risk_inputs(earlier, [t], ic=0.05)[0][t], alphas[t])


def test_accounting_follows_the_part_2_formulas():
    case = synthetic_case()
    alphas, roots, _ = opt.risk_inputs(case, range(60, 70), ic=0.05)
    rec = opt.simulate(case, alphas, roots, 60, 70, 10.0, 1, opt.WeeklyProblem(12), cost_bps=5.0, borrow_bps=50.0)
    np.testing.assert_allclose(rec["net"], rec["gross"] - rec["turnover"] * 5e-4 - rec["short"] * 0.005 / 52,
                               rtol=0, atol=1e-15)
    assert rec["not_optimal"] == 0 and rec["violation"].max() < 1e-6


def test_selection_follows_the_plan():
    train = {(1, 0): (0.30, 1.00), (3, 0): (0.30, 0.95), (1, 1): (0.50, 0.70), (3, 1): (0.40, 0.92)}
    assert opt.select(train) == ((3, 1), True)                     # (1, 1) has the best Sharpe but is not admissible
    assert opt.select(train, penalties=(0,)) == ((1, 0), True)     # a tie goes to the earlier point
    assert opt.select({(1, 0): (0.3, 0.5), (3, 0): (0.4, 0.8)}) == ((3, 0), False)
    assert opt.grid(0.0) == [(g, 0) for g in opt.GAMMAS]
    assert opt.grid(5.0)[:5] == [(g, 0) for g in opt.GAMMAS] and len(opt.grid(5.0)) == 20
