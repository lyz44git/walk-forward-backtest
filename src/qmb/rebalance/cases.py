"""Per-week target portfolios for the rebalancing study, built with the backtest engine.

The loaders, schedule, features, walk-forward and portfolio rule are the ones used by
scripts/03_backtest.py (main run: default execution, no extra lag), so full rebalancing of these
targets reproduces reports/ls_weekly.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qmb import data as qdata, engine, stats

ROOT = qdata.ROOT
TMP = ROOT / "tmp" / "rebalance"
COST_BPS, BORROW_BPS = 5.0, 50.0          # main cost model (bps one-way; bps per year on shorts)
SPLIT = {"etf": "2019-01-01", "french": "2001-01-01"}   # first test signal date is the first one on/after this


def load(name: str) -> dict:
    """Identical to load() in scripts/03_backtest.py."""
    if name == "etf":
        P = pd.read_parquet(qdata.CLEAN / "etf_daily.parquet")
        P = P[P.date >= "2007-04-11"]
        piv = lambda c: P.pivot(index="date", columns="ticker", values=c).sort_index()
        return {"tri": piv("close_adj"), "open": piv("open_adj"), "dv": piv("dollar_volume"), "initial": 156}
    I = pd.read_parquet(qdata.CLEAN / "french_vw_daily.parquet").loc["1969-07-01":]
    tri = (1 + I).cumprod()
    tri = tri.where(I.notna())
    return {"tri": tri, "open": None, "dv": None, "initial": 260}


def target_weights(pred_w: pd.Series, quantile: float = 0.2) -> pd.Series:
    """Exactly the long-short rule of qmb.engine.portfolio for one week (pred indexed by asset)."""
    n = len(pred_w); k = max(1, int(round(quantile * n)))
    order = pred_w.sort_values(kind="mergesort")
    w = pd.Series(0.0, index=pred_w.index)
    w[order.index[-k:]] = 1.0 / k; w[order.index[:k]] = -1.0 / k
    return w


def build_cases(name: str, signals: list[str]) -> dict:
    """Return {signal: case-dict of dense arrays over the OOS weeks}."""
    d = load(name)
    tri = d["tri"]
    exec_px = d["open"] if d["open"] is not None else tri
    sch = engine.weekly_schedule(tri.index)
    feats = engine.daily_features(tri, d["dv"])
    panel = engine.build_panel(tri, exec_px, feats, sch)
    raw21 = feats["ret_21d"].reindex(pd.DatetimeIndex(sch["signal"]))       # raw (unranked), at signal dates
    raw21.index = range(len(sch))
    assets = list(tri.columns)
    out = {}
    for s in signals:
        pred = engine.walk_forward(panel, s, d["initial"])
        weeks = np.array(sorted(pred.index.get_level_values("week").unique()))
        T, N = len(weeks), len(assets)
        W = np.zeros((T, N)); R = np.zeros((T, N)); M = np.zeros((T, N), bool); S = np.zeros((T, N))
        aidx = {a: i for i, a in enumerate(assets)}
        for t, (wk, p) in enumerate(pred.groupby(level="week")):
            assert wk == weeks[t]
            p = p.droplevel("week")
            cols = [aidx[a] for a in p.index]
            w = target_weights(p)
            r = panel.ret.loc[wk].reindex(p.index)
            W[t, cols] = w.to_numpy(); R[t, cols] = r.to_numpy(); M[t, cols] = True
            z = (p - p.mean()) / (p.std(ddof=0) if p.std(ddof=0) > 0 else 1.0)
            S[t, cols] = z.to_numpy()
        ic = stats.weekly_ic(pred, panel.ret).reindex(weeks).to_numpy()
        disp = raw21.loc[weeks].std(axis=1, ddof=0).to_numpy()
        out[s] = {"dataset": name, "signal": s, "weeks": weeks, "assets": np.array(assets),
                  "signal_date": sch["signal"].iloc[weeks].to_numpy(), "entry": sch["entry"].iloc[weeks].to_numpy(),
                  "exit": sch["exit"].iloc[weeks].to_numpy(), "W": W, "R": R, "M": M, "S": S, "IC": ic, "DISP": disp}
    return out


def save_case(c: dict, path: Path) -> None:
    np.savez_compressed(path, **{k: (v.astype("datetime64[ns]") if k in ("signal_date", "entry", "exit") else v)
                                 for k, v in c.items()})


def load_case(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    c = {k: z[k] for k in z.files}
    c["dataset"] = str(c["dataset"]); c["signal"] = str(c["signal"])
    return c
