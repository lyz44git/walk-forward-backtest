"""Factor regressions of weekly long-short returns (used by scripts 11 and 13)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qmb import data as qdata, stats


def weekly_factors(entry, exit_, dataset: str) -> pd.DataFrame:
    """Market, size and value factor returns compounded over each holding period.

    Industries trade close to close: trading days in (entry, exit]. ETFs trade open to open: trading days in
    [entry, exit), i.e. the close-to-close days nearest the open-to-open window. Days after the last factor
    date are dropped."""
    ff = pd.read_parquet(qdata.CLEAN / "ff_daily.parquet")[["mkt_rf", "smb", "hml"]]
    g = np.log1p(ff).cumsum()
    entry, exit_ = pd.DatetimeIndex(entry), pd.DatetimeIndex(exit_)
    if dataset == "etf":
        prev = g.shift(1)
        a, b = prev.reindex(entry).to_numpy(), prev.reindex(exit_).to_numpy()
    else:
        a, b = g.reindex(entry).to_numpy(), g.reindex(exit_).to_numpy()
    return pd.DataFrame(np.expm1(b - a), columns=ff.columns)


def ols_nw(y, X, names, lags=4):
    """OLS with Newey-West (Bartlett) standard errors; the first column of X is the constant."""
    ok = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    y, X = y[ok], X[ok]
    n = len(y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for L in range(1, lags + 1):
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None]) / n
        S += (1 - L / (lags + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    se = np.sqrt(np.diag(Q @ S @ Q / n))
    out = {"weeks": n, "alpha_annual": beta[0] * stats.PER_YEAR, "alpha_t": beta[0] / se[0],
           "r_squared": 1 - e.var() / y.var()}
    for name, b, s in zip(names, beta[1:], se[1:]):
        out[f"beta_{name}"], out[f"t_{name}"] = b, b / s
    return out
