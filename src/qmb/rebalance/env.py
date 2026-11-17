"""Weekly partial-rebalancing MDP on top of the project's target long-short portfolios.

Accounting (identical to qmb.engine.portfolio / net_returns when eta = 1):
  d_t   = h_{t-1} (1 + r_{t-1}) / (1 + h_{t-1} . r_{t-1})        drifted weights at execution of week t
  h_t   = M_t * (d_t + eta_t (W_t - d_t))                        assets absent in week t are closed
  turnover_t = |h_t - d_t|_1 ; gross_t = h_t . r_t ; short_t = -sum(h_t[h_t < 0])
  net_t = gross_t - turnover_t * 5e-4 - short_t * 0.005 / 52
State (all known at the signal date s_t): see FEATURES. h_{t-1} was set at entry_{t-1} <= s_t; returns and ICs
are used only for weeks whose holding period ended on or before s_t.
"""
from __future__ import annotations

import numpy as np

from .cases import COST_BPS, BORROW_BPS

FEATURES = ["req_turnover", "gross_prev", "signal_gap", "target_change", "dispersion_21d",
            "vol4_full", "vol26_full", "ic13", "ic52"]
EXO = FEATURES[3:]          # do not depend on the policy


def drift(h, r):
    g = 1.0 + (h * r).sum(-1, keepdims=True)
    return h * (1.0 + r) / g


def step_accounting(d, W, M, r, eta, cost_bps=COST_BPS, borrow_bps=BORROW_BPS):
    """d, W, M, r: (..., N); eta broadcastable (..., 1). Returns h, net, gross, turnover, short."""
    h = np.where(M, d + eta * (W - d), 0.0)
    to = np.abs(h - d).sum(-1)
    gross = (h * r).sum(-1)
    short = -np.where(h < 0, h, 0.0).sum(-1)
    net = gross - to * cost_bps / 1e4 - short * borrow_bps / 1e4 / 52.0
    return h, net, gross, to, short


def band_holdings(d, W, M, b):
    """No-trade band per asset: trade only to the edge of [W - b, W + b]."""
    return np.where(M, W + np.clip(d - W, -b, b), 0.0)


def last_completed(case) -> np.ndarray:
    """For each OOS week t: index of the latest OOS week whose holding period ended on/before signal_t (-1: none)."""
    ex = case["exit"].astype("datetime64[ns]"); sg = case["signal_date"].astype("datetime64[ns]")
    return np.searchsorted(ex, sg, side="right") - 1


def full_rebalance(case):
    """eta = 1 from zero holdings at the first OOS week (reproduces the project's ls_weekly)."""
    W, R, M = case["W"], case["R"], case["M"]
    T, N = W.shape
    h = np.zeros(N); r_prev = np.zeros(N)
    out = np.zeros((T, 4))
    for t in range(T):
        d = drift(h, r_prev) if t > 0 else np.zeros(N)
        h, net, gross, to, short = step_accounting(d, W[t], M[t], R[t], 1.0)
        out[t] = net, gross, to, short; r_prev = R[t]
    return out


def exogenous_features(case, gross_full):
    """(T, len(EXO)) features that do not depend on the policy; NaN where not yet defined."""
    W, IC = case["W"], case["IC"]
    T = len(W)
    lc = last_completed(case)
    X = np.full((T, len(EXO)), np.nan)
    for t in range(T):
        X[t, 0] = np.abs(W[t] - W[t - 1]).sum() if t > 0 else np.nan
        X[t, 1] = case["DISP"][t]
        j = lc[t]
        if j >= 3:
            X[t, 2] = gross_full[j - 3:j + 1].std(ddof=1)
        if j >= 25:
            X[t, 3] = gross_full[j - 25:j + 1].std(ddof=1)
        if j >= 12:
            X[t, 4] = np.nanmean(IC[j - 12:j + 1])
        if j >= 51:
            X[t, 5] = np.nanmean(IC[j - 51:j + 1])
    return X


def endogenous_features(h_prev, W, S):
    """Features from the holdings set at the previous entry and this week's target/signal."""
    gap = W - h_prev
    req = np.abs(gap).sum(-1)
    gross_prev = np.abs(h_prev).sum(-1)
    tgt_exp = (W * S).sum(-1)
    sig_gap = (gap * S).sum(-1) / np.where(np.abs(tgt_exp) > 1e-12, tgt_exp, 1.0)
    return np.stack([req / 2.0, gross_prev - 2.0, np.clip(sig_gap, -2, 2)], -1)


class Market:
    """Precomputed arrays for one (dataset, signal) case plus exogenous feature normalisation."""

    def __init__(self, case, train_idx):
        self.c = case
        self.W, self.R, self.M, self.S = case["W"], case["R"], case["M"], case["S"]
        self.full = full_rebalance(case)                    # columns: net, gross, turnover, short
        exo = exogenous_features(case, self.full[:, 1])
        tr = exo[train_idx]
        self.mu, self.sd = np.nanmean(tr, 0), np.nanstd(tr, 0) + 1e-12   # train span only
        self.exo_raw = exo
        self.exo = np.nan_to_num((exo - self.mu) / self.sd)
        self.first_valid = int(np.argmax(~np.isnan(exo).any(1)))

    def obs(self, t, h_prev):
        return np.concatenate([endogenous_features(h_prev, self.W[t], self.S[t]), self.exo[t]], -1)


def simulate(mkt: Market, t0: int, t1: int, policy=None, eta_const=None, band=None):
    """Run weeks t0..t1-1 starting from h_{t0-1} = W_{t0-1} (full target the week before; same for every rule).
    Exactly one of policy (callable obs->eta), eta_const, band is used. Returns dict of arrays."""
    W, R, M = mkt.W, mkt.R, mkt.M
    h = W[t0 - 1].copy(); r_prev = R[t0 - 1]
    rec = {k: [] for k in ("net", "gross", "turnover", "short", "eta")}
    for t in range(t0, t1):
        d = drift(h, r_prev)
        if band is not None:
            width = band[t - t0] if np.ndim(band) else band
            hn = band_holdings(d, W[t], M[t], width)
            to = np.abs(hn - d).sum(); gross = (hn * R[t]).sum(); short = -hn[hn < 0].sum()
            net = gross - to * COST_BPS / 1e4 - short * BORROW_BPS / 1e4 / 52.0
            eta = np.nan
            h = hn
        else:
            eta = float(policy(mkt.obs(t, h))) if policy is not None else float(eta_const)
            h, net, gross, to, short = step_accounting(d, W[t], M[t], R[t], eta)
        for k, v in zip(rec, (net, gross, to, short, eta)):
            rec[k].append(float(v))
        r_prev = R[t]
    return {k: np.array(v) for k, v in rec.items()}


class BatchEnv:
    """B synchronous episodes of length L on the training span, for PPO rollouts."""

    def __init__(self, mkt: Market, lo: int, hi: int, L: int, B: int, rng, risk_aversion: float):
        self.m, self.lo, self.hi, self.L, self.B, self.rng, self.ga = mkt, lo, hi, L, B, rng, risk_aversion

    def reset(self):
        self.t = self.rng.integers(self.lo, self.hi - self.L + 1, size=self.B)    # all weeks t..t+L-1 in [lo, hi)
        self.k = 0
        self.h = self.m.W[self.t - 1].copy(); self.r_prev = self.m.R[self.t - 1]
        return self._obs()

    def _obs(self):
        t = self.t + self.k
        return np.concatenate([endogenous_features(self.h, self.m.W[t], self.m.S[t]), self.m.exo[t]], -1)

    def step(self, eta):
        t = self.t + self.k
        d = drift(self.h, self.r_prev)
        self.h, net, *_ = step_accounting(d, self.m.W[t], self.m.M[t], self.m.R[t], eta[:, None])
        nf = self.m.full[t, 0]
        # reward: mean-variance utility of the policy minus that of full rebalancing (an action-independent
        # baseline that removes the common market noise), in percent
        rew = 100.0 * ((net - nf) - 0.5 * self.ga * (net ** 2 - nf ** 2))
        self.r_prev = self.m.R[t]; self.k += 1
        done = self.k >= self.L
        return self._obs(), rew, done          # after the last step: the bootstrap state (week t+L, < len(W))
