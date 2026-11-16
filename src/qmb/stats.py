"""Evaluation statistics for weekly series."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PER_YEAR = 52


def weekly_ic(pred: pd.Series, ret: pd.Series) -> pd.Series:
    df = pd.DataFrame({"p": pred, "r": ret.reindex(pred.index)}).dropna()
    return df.groupby(level="week").apply(lambda g: stats.spearmanr(g.p, g.r).statistic if len(g) > 2 else np.nan).dropna()


def newey_west_t(x: pd.Series | np.ndarray, lags: int = 4) -> float:
    """t-statistic of the mean with a Newey-West (Bartlett) long-run variance."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    e = x - x.mean()
    lrv = e @ e / n
    for L in range(1, lags + 1):
        lrv += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / np.sqrt(lrv / n))


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(PER_YEAR)) if r.std() > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    w = (1 + r.fillna(0)).cumprod()
    return float((w / w.cummax() - 1).min())


def stationary_bootstrap_sharpe(r: pd.Series, mean_block: int = 8, draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the annualized Sharpe ratio, Politis-Romano stationary bootstrap."""
    x = r.dropna().to_numpy(); n = len(x); rng = np.random.default_rng(seed); p = 1.0 / mean_block
    out = np.empty(draws)
    for d in range(draws):
        idx = np.empty(n, dtype=int); idx[0] = rng.integers(n)
        new = rng.random(n) < p; jumps = rng.integers(n, size=n)
        for i in range(1, n):
            idx[i] = jumps[i] if new[i] else (idx[i - 1] + 1) % n
        s = x[idx]; out[d] = s.mean() / s.std(ddof=1) * np.sqrt(PER_YEAR)
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def deflated_sharpe(r: pd.Series, trial_sharpes_annual: list[float]) -> float:
    """Probability that the true (per-period) Sharpe exceeds the expected maximum of N null trials
    (Bailey and Lopez de Prado 2014). Inputs are annualized; computed on the per-week scale."""
    x = r.dropna().to_numpy(); n = len(x)
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    trials = np.asarray(trial_sharpes_annual, float) / np.sqrt(PER_YEAR)
    N = len(trials); var_sr = trials.var(ddof=1) if N > 1 else 0.0
    gamma = 0.5772156649
    sr0 = np.sqrt(var_sr) * ((1 - gamma) * stats.norm.ppf(1 - 1 / N) + gamma * stats.norm.ppf(1 - 1 / (N * np.e))) if N > 1 else 0.0
    z = (sr - sr0) * np.sqrt(n - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf(z))


def alpha_beta(r: pd.Series, bench: pd.Series, lags: int = 4) -> dict:
    df = pd.concat([r, bench], axis=1, keys=["r", "b"]).dropna()
    X = np.column_stack([np.ones(len(df)), df.b]); y = df.r.to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ beta; n = len(y)
    S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for L in range(1, lags + 1):
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None]) / n
        S += (1 - L / (lags + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n); V = Q @ S @ Q / n
    return {"alpha_annual": float(beta[0] * PER_YEAR), "alpha_t": float(beta[0] / np.sqrt(V[0, 0])),
            "beta": float(beta[1]), "beta_t": float(beta[1] / np.sqrt(V[1, 1]))}


def summarize(r: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = r.dropna()
    return {"weeks": int(len(r)), "ann_return": float(r.mean() * PER_YEAR), "ann_vol": float(r.std(ddof=1) * np.sqrt(PER_YEAR)),
            "sharpe": sharpe(r), "max_drawdown": max_drawdown(r),
            "annual_turnover": float(turnover.mean() * PER_YEAR) if turnover is not None else np.nan}
