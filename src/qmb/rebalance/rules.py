"""Target transforms that trade less by changing the target instead of the trade size.

Both transforms are causal: week t uses the predictions up to week t and, for the rank buffer, the
names held after week t-1. With the neutral parameter (multiple 1, half-life 0) each returns the
backtest's own top and bottom quintile targets.
"""
from __future__ import annotations

import numpy as np

QUANTILE = 0.2


def _order(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Valid asset indices from lowest to highest score, stable for ties (the backtest's ordering)."""
    idx = np.flatnonzero(valid)
    return idx[np.argsort(score[idx], kind="mergesort")]


def _leg_size(n: int) -> int:
    return max(1, int(round(QUANTILE * n)))


def quintile_targets(S: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Long the top 20% and short the bottom 20% of valid assets each week, equal weight."""
    W = np.zeros(S.shape)
    for t in range(len(S)):
        order = _order(S[t], M[t]); k = _leg_size(len(order))
        W[t, order[-k:]] = 1.0 / k; W[t, order[:k]] = -1.0 / k
    return W


def smoothed_scores(S: np.ndarray, M: np.ndarray, half_life: float) -> np.ndarray:
    """Exponentially weighted mean of past standardized predictions, per asset.

    An asset absent in week t keeps its last value; an asset seen for the first time starts at its
    current score. Half-life 0 returns S.
    """
    if half_life <= 0:
        return S.copy()
    a = 0.5 ** (1.0 / half_life)
    E = np.zeros(S.shape); state = np.zeros(S.shape[1]); seen = np.zeros(S.shape[1], bool)
    for t in range(len(S)):
        m = M[t]
        state = np.where(m & seen, a * state + (1 - a) * S[t], np.where(m, S[t], state))
        seen |= m
        E[t] = state
    return E


def buffered_targets(S: np.ndarray, M: np.ndarray, multiple: float) -> np.ndarray:
    """Rank buffer (hysteresis): a held name stays while it ranks inside the top (bottom)
    multiple x 20%; vacancies are filled with the best-ranked names. Each leg holds k names at 1/k."""
    T, N = S.shape
    W = np.zeros((T, N)); long_held = np.zeros(N, bool); short_held = np.zeros(N, bool)
    for t in range(T):
        order = _order(S[t], M[t]); n = len(order); k = _leg_size(n)
        k_out = min(max(k, int(round(multiple * k))), n // 2)
        legs = []
        for held, ranked in ((long_held, order[::-1]), (short_held, order)):
            keep = [i for i in ranked[:k_out] if held[i]][:k]          # best-ranked first
            kept = set(keep)
            keep += [i for i in ranked if i not in kept][:k - len(keep)]
            legs.append(np.array(keep))
        long_held[:] = False; short_held[:] = False
        long_held[legs[0]] = True; short_held[legs[1]] = True
        W[t, legs[0]] = 1.0 / k; W[t, legs[1]] = -1.0 / k
    return W
