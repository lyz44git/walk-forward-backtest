"""Mean-variance portfolio construction with a turnover penalty (design/portfolio_optimizer_plan.md).

Each week, from the drifted holdings d:
    maximize   alpha'h - (gamma / 2) h' Sigma h - kappa ||h - d||_1
    subject to sum(h) = 0, ||h||_1 <= 2
Sigma: Ledoit-Wolf shrinkage of the last 52 completed weekly returns; alpha = IC * sigma * z.
Accounting, costs and borrow fees are those of qmb.rebalance.env.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from . import env
from .cases import BORROW_BPS, COST_BPS

WINDOW = 52
GAMMAS = (1, 3, 10, 30, 100)
PENALTIES = (0, 1, 2, 4)
GROSS = 2.0
EXPOSURE = (0.90, 1.10)


def ledoit_wolf(X: np.ndarray):
    """Ledoit and Wolf (2004) shrinkage of the sample covariance of the rows of X towards mu * I.

    The estimator of sklearn.covariance.ledoit_wolf (assume_centered=False). Returns (covariance, shrinkage).
    """
    X = np.asarray(X, float)
    T, N = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    mu = np.trace(S) / N
    delta = ((S - mu * np.eye(N)) ** 2).sum() / N
    X2 = Xc ** 2
    beta = ((X2.T @ X2) / T - S ** 2).sum() / (N * T)
    shrinkage = 0.0 if delta == 0 else min(beta, delta) / delta
    cov = (1.0 - shrinkage) * S
    cov[np.diag_indices(N)] += shrinkage * mu
    return cov, shrinkage


def risk_inputs(case: dict, weeks, ic: float):
    """For each week t in `weeks`: alpha_t and R_t with R_t' R_t = Sigma_t, from the 52 weeks completed by its signal.

    Returns dicts keyed by week, and the shrinkage intensities.
    """
    R, S = case["R"], case["S"]
    last = env.last_completed(case)
    alphas, roots, shrink = {}, {}, {}
    for t in weeks:
        j = int(last[t])
        if j < WINDOW - 1:
            raise ValueError(f"week {t}: only {j + 1} completed weeks before its signal")
        cov, shrink[t] = ledoit_wolf(R[j - WINDOW + 1:j + 1])
        values, vectors = np.linalg.eigh(cov)
        roots[t] = np.sqrt(np.clip(values, 0.0, None))[:, None] * vectors.T
        alphas[t] = ic * np.sqrt(np.diag(cov)) * S[t]
    return alphas, roots, shrink


class WeeklyProblem:
    """The weekly problem for n assets, compiled once (CVXPY parameters) and re-solved with new inputs."""

    def __init__(self, n: int):
        self.h = cp.Variable(n)
        self.alpha = cp.Parameter(n)
        self.root = cp.Parameter((n, n))              # sqrt(gamma / 2) * R, so ||root h||^2 = gamma / 2 h'Sigma h
        self.kappa = cp.Parameter(nonneg=True)
        self.kd = cp.Parameter(n)                     # kappa * d
        self.closed = cp.Parameter(n, nonneg=True)    # 1 for assets absent this week
        objective = cp.Maximize(self.alpha @ self.h - cp.sum_squares(self.root @ self.h)
                                - cp.norm1(self.kappa * self.h - self.kd))
        constraints = [cp.sum(self.h) == 0, cp.norm1(self.h) <= GROSS, cp.multiply(self.closed, self.h) == 0]
        self.problem = cp.Problem(objective, constraints)

    def solve(self, alpha, root, gamma, kappa, d, present):
        self.alpha.value = alpha
        self.root.value = np.sqrt(gamma / 2.0) * root
        self.kappa.value = kappa
        self.kd.value = kappa * d
        self.closed.value = (~present).astype(float)
        self.problem.solve(solver=cp.CLARABEL)
        if self.h.value is None:
            raise RuntimeError(f"solver returned {self.problem.status}")
        return np.where(present, self.h.value, 0.0), self.problem.status


def simulate(case: dict, alphas, roots, t0: int, t1: int, gamma: float, m: float, problem: WeeklyProblem,
             cost_bps: float = COST_BPS, borrow_bps: float = BORROW_BPS) -> dict:
    """Weeks t0..t1-1 from the full quintile target of week t0-1, as qmb.rebalance.env.simulate."""
    W, R, M = case["W"], case["R"], case["M"]
    h = W[t0 - 1].copy()
    r_prev = R[t0 - 1]
    kappa = m * cost_bps / 1e4
    rec = {k: [] for k in ("net", "gross", "turnover", "short", "violation")}
    not_optimal = 0
    for t in range(t0, t1):
        d = env.drift(h, r_prev)
        h, status = problem.solve(alphas[t], roots[t], gamma, kappa, d, M[t])
        not_optimal += status != cp.OPTIMAL
        turnover = np.abs(h - d).sum()
        gross = (h * R[t]).sum()
        short = -h[h < 0].sum()
        net = gross - turnover * cost_bps / 1e4 - short * borrow_bps / 1e4 / 52.0
        violation = max(abs(h.sum()), np.abs(h).sum() - GROSS, 0.0)
        for k, v in zip(rec, (net, gross, turnover, short, violation)):
            rec[k].append(float(v))
        r_prev = R[t]
    out = {k: np.array(v) for k, v in rec.items()}
    out["not_optimal"] = int(not_optimal)
    out["holdings"] = h
    return out


def grid(cost_bps: float):
    """(gamma, m) in tie-break order: smaller m first, then smaller gamma. At zero cost m has no effect."""
    penalties = (0,) if cost_bps == 0 else PENALTIES
    return [(g, m) for m in penalties for g in GAMMAS]


def select(train: dict, penalties=None):
    """train: {(gamma, m): (net Sharpe, mean short)}. Best admissible point; else the one closest to full exposure."""
    points = [p for p in train if penalties is None or p[1] in penalties]
    admissible = [p for p in points if EXPOSURE[0] <= train[p][1] <= EXPOSURE[1] and np.isfinite(train[p][0])]
    if admissible:
        best = admissible[0]
        for p in admissible[1:]:
            if train[p][0] > train[best][0] + 1e-12:
                best = p
        return best, True
    return min(points, key=lambda p: abs(train[p][1] - 1.0)), False
