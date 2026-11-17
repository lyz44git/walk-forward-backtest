"""Walk-forward engine: weekly schedule, features, targets, purged expanding-window training, portfolios.

Timing (all from design/preregistration.json):
- signal date s_k = last trading day of calendar week k; features use prices up to and including s_k.
- ETF ("open" mode): enter at the open of the first trading day after s_k, exit at the open of the first trading
  day after s_{k+1}. Industries / robustness ("close" mode): enter at the close of the first trading day after
  s_k, exit at the close of the first trading day after s_{k+1}.
- a week j may be used for training at refit week k only if exit_j < s_k - embargo (purge + embargo).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_FEATURES = ["ret_5d", "ret_21d", "ret_63d", "ret_126d", "mom_12_1", "vol_21d", "vol_63d", "gap_ma200", "dd_63d"]
LGB_PARAMS = dict(objective="regression", learning_rate=0.02, num_leaves=7, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  verbose=-1, seed=0, num_threads=16, deterministic=True, force_row_wise=True)
LGB_ROUNDS = 300


# ---------------------------------------------------------------- schedule
def weekly_schedule(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per week: signal date, entry date and exit date (exit_k = entry_{k+1})."""
    dates = pd.DatetimeIndex(sorted(dates))
    s = pd.Series(dates, index=dates)
    sig = s.groupby(dates.to_period("W-FRI")).max().to_numpy()
    pos = np.searchsorted(dates.to_numpy(), sig)                 # index of the signal day
    nxt = pos + 1
    ok = nxt < len(dates)
    sig, nxt = sig[ok], nxt[ok]
    entry = dates[nxt]
    sch = pd.DataFrame({"signal": sig, "entry": entry})
    sch["exit"] = sch["entry"].shift(-1)
    return sch.dropna().reset_index(drop=True)


# ---------------------------------------------------------------- features
def daily_features(tri: pd.DataFrame, dollar_volume: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Wide (date x asset) feature tables computed only from data up to each date."""
    r = tri.pct_change(fill_method=None)
    f = {
        "ret_5d": tri / tri.shift(5) - 1,
        "ret_21d": tri / tri.shift(21) - 1,
        "ret_63d": tri / tri.shift(63) - 1,
        "ret_126d": tri / tri.shift(126) - 1,
        "mom_12_1": tri.shift(21) / tri.shift(252) - 1,
        "vol_21d": r.rolling(21, min_periods=21).std(),
        "vol_63d": r.rolling(63, min_periods=63).std(),
        "gap_ma200": tri / tri.rolling(200, min_periods=200).mean() - 1,
        "dd_63d": tri / tri.rolling(63, min_periods=63).max() - 1,
    }
    if dollar_volume is not None:
        dv = dollar_volume.where(dollar_volume > 0)
        f["dvol_surge"] = np.log(dv.rolling(5, min_periods=5).mean()) - np.log(dv.rolling(63, min_periods=63).mean())
    return f


def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank per row scaled to [-0.5, 0.5]; NaN stays NaN."""
    n = x.notna().sum(axis=1)
    return (x.rank(axis=1) - 0.5).div(n, axis=0) - 0.5


@dataclass
class Panel:
    """Long panel of (week, asset) rows with ranked features, realized period return and its rank."""
    sched: pd.DataFrame
    X: pd.DataFrame                 # index (week, asset); ranked features
    ret: pd.Series                  # realized holding-period return
    y: pd.Series                    # cross-sectional rank of ret, fitting target
    features: list[str] = field(default_factory=list)


def build_panel(tri: pd.DataFrame, exec_px: pd.DataFrame, features: dict[str, pd.DataFrame],
                sched: pd.DataFrame, feature_lag_weeks: int = 0) -> Panel:
    names = list(features)
    sig = pd.DatetimeIndex(sched["signal"])
    # features observed at the signal date (optionally at an earlier signal date, for the lag robustness test)
    src_sig = sig if feature_lag_weeks == 0 else pd.DatetimeIndex(sched["signal"].shift(feature_lag_weeks))
    fx = {}
    for nm in names:
        t = features[nm].reindex(src_sig)
        t.index = range(len(sched))
        fx[nm] = cs_rank(t)
    px_in = exec_px.reindex(pd.DatetimeIndex(sched["entry"]))
    px_out = exec_px.reindex(pd.DatetimeIndex(sched["exit"]))
    ret = pd.DataFrame(px_out.to_numpy() / px_in.to_numpy() - 1, columns=exec_px.columns)
    X = pd.concat({nm: fx[nm].stack(future_stack=True) for nm in names}, axis=1)
    R = ret.stack(future_stack=True)
    R.index = X.index
    ok = X.notna().all(axis=1) & R.notna()
    X, R = X[ok], R[ok]
    X.index.names = R.index.names = ["week", "asset"]
    y = R.groupby(level="week").transform(lambda v: (v.rank() - 0.5) / len(v) - 0.5)
    return Panel(sched=sched, X=X, ret=R, y=y, features=names)


# ---------------------------------------------------------------- walk-forward
def horizon_target(panel: Panel, exec_px: pd.DataFrame, horizon: int) -> pd.Series:
    """Cross-sectional rank of the return from the entry of week k to the exit of week k+horizon-1,
    aligned to the panel rows (NaN where that exit is not in the sample). horizon = 1 returns panel.y."""
    sch = panel.sched
    px_in = exec_px.reindex(pd.DatetimeIndex(sch["entry"]))
    px_out = exec_px.reindex(pd.DatetimeIndex(sch["exit"].shift(-(horizon - 1))))
    ret = pd.DataFrame(px_out.to_numpy() / px_in.to_numpy() - 1, columns=exec_px.columns)
    r = ret.stack(future_stack=True)
    r.index.names = ["week", "asset"]
    r = r.reindex(panel.X.index)
    return r.groupby(level="week").transform(lambda v: (v.rank() - 0.5) / v.notna().sum() - 0.5)


def walk_forward(panel: Panel, signal: str, initial_weeks: int, refit_every: int = 4, embargo: int = 1,
                 placebo: bool = False, seed: int = 0, target: pd.Series | None = None,
                 label_weeks: int = 1) -> pd.Series:
    """Out-of-sample predictions for every week after the initial training span. Returns Series(week, asset).

    The first test week is chosen so that the first refit trains on exactly `initial_weeks` weeks: the purge
    drops the week whose holding period has not ended (k-1, k-2 in a regular calendar) and the embargo one more.
    Placebo: training targets are permuted across assets within each week, with a FRESH permutation at every
    refit, so the fitted directions are independent across refits (a single permutation reused over nested
    expanding windows produces a persistent random direction, not a null).
    `target` replaces the one-week rank target (see horizon_target); `label_weeks` is the number of weeks it
    spans, so that a week is used for training only after the exit of its last label week."""
    sch = panel.sched
    weeks = np.array(sorted(panel.X.index.get_level_values("week").unique()))
    test_weeks = weeks[weeks >= weeks[0] + initial_weeks + 1 + embargo]
    if signal == "mom_12_1":
        return panel.X["mom_12_1"].loc[test_weeks]
    if signal == "rev_1w":
        return -panel.X["ret_5d"].loc[test_weeks]
    exit_ = pd.DatetimeIndex(sch["exit"]); sig = pd.DatetimeIndex(sch["signal"])
    wk = panel.X.index.get_level_values("week").to_numpy()
    y = panel.y if target is None else target.reindex(panel.X.index)
    Xv, y0 = panel.X.to_numpy(np.float64), y.to_numpy(np.float64)
    known = ~np.isnan(y0)
    label_end = weeks + label_weeks - 1                       # week whose exit closes the label of each week
    in_sample = label_end < len(sch)
    if placebo:
        rng = np.random.default_rng(seed)
        order = np.argsort(wk, kind="stable")
        _, starts, counts = np.unique(wk[order], return_index=True, return_counts=True)
    out = []
    for start in range(0, len(test_weeks), refit_every):
        block = test_weeks[start:start + refit_every]
        k = block[0]
        cutoff = sig[k] - pd.Timedelta(weeks=embargo)
        train_weeks = weeks[in_sample][exit_[label_end[in_sample]] < cutoff]
        assert exit_[train_weeks + label_weeks - 1].max() < sig[k], "purge violated"
        tr = np.isin(wk, train_weeks) & known; te = np.isin(wk, block)
        yv = y0
        if placebo:                                # fresh within-week permutation for this refit
            yv = y0.copy()
            for a, c in zip(starts, counts):
                m = order[a:a + c]; yv[m] = y0[rng.permutation(m)]
        if signal == "ridge":
            A = np.column_stack([np.ones(tr.sum()), Xv[tr]])
            pen = np.eye(A.shape[1]); pen[0, 0] = 0.0
            beta = np.linalg.solve(A.T @ A + 1.0 * pen, A.T @ yv[tr])
            p = np.column_stack([np.ones(te.sum()), Xv[te]]) @ beta
        elif signal == "lgbm":
            m = lgb.train(LGB_PARAMS, lgb.Dataset(Xv[tr], yv[tr]), LGB_ROUNDS)
            p = m.predict(Xv[te])
        else:
            raise ValueError(signal)
        out.append(pd.Series(p, index=panel.X.index[te]))
    return pd.concat(out).sort_index()


# ---------------------------------------------------------------- portfolios
def portfolio(pred: pd.Series, ret: pd.Series, quantile: float = 0.2, kind: str = "long_short") -> pd.DataFrame:
    """Weekly portfolio: gross return, turnover (vs drifted weights), short notional, n assets per leg."""
    rows, prev_w, prev_r = [], None, None
    for w_, p in pred.groupby(level="week"):
        p = p.droplevel("week"); r = ret.loc[w_].reindex(p.index)
        n = len(p); k = max(1, int(round(quantile * n)))
        order = p.sort_values(kind="mergesort")
        w = pd.Series(0.0, index=p.index)
        if kind == "long_short":
            w[order.index[-k:]] = 1.0 / k; w[order.index[:k]] = -1.0 / k
        elif kind == "long_only":
            w[order.index[-k:]] = 1.0 / k
        elif kind == "equal_weight":
            w[:] = 1.0 / n
        else:
            raise ValueError(kind)
        if prev_w is None:
            turnover = w.abs().sum()
        else:
            gross = 1.0 + (prev_w * prev_r).sum()
            drifted = (prev_w * (1.0 + prev_r)) / gross
            drifted = drifted.reindex(w.index.union(drifted.index), fill_value=0.0)
            turnover = (w.reindex(drifted.index, fill_value=0.0) - drifted).abs().sum()
        rows.append({"week": w_, "ret_gross": float((w * r).sum()), "turnover": float(turnover),
                     "short_notional": float(-w[w < 0].sum()), "n_leg": k, "n_assets": n})
        prev_w, prev_r = w, r
    return pd.DataFrame(rows).set_index("week")


def net_returns(port: pd.DataFrame, cost_bps: float, borrow_bps_per_year: float = 50.0) -> pd.Series:
    return port["ret_gross"] - port["turnover"] * cost_bps / 1e4 - port["short_notional"] * borrow_bps_per_year / 1e4 / 52.0
