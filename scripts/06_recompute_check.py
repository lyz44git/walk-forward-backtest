"""Recompute the headline metrics from the saved weekly series with formulas written here (nothing is imported
from qmb), and check how sensitive the 'net Sharpe CI excludes 0' leg of the pass rule is to the bootstrap seed.

Reads:  reports/{ic_weekly.csv, ls_weekly.csv, main_results.csv}
Writes: reports/recompute_check.csv, reports/bootstrap_seed_sensitivity.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SRC = OUT = Path(__file__).resolve().parents[1] / "reports"

ic = pd.read_csv(SRC / "ic_weekly.csv", parse_dates=["date"])
ls = pd.read_csv(SRC / "ls_weekly.csv", parse_dates=["date"])
M = pd.read_csv(SRC / "main_results.csv")


def nw_t(x, lags=4):
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2.0 * (1.0 - L / (lags + 1.0)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(v / n)


def sharpe(r):
    r = np.asarray(r, float)
    return r.mean() / r.std(ddof=1) * np.sqrt(52.0)


def stat_boot_ci(x, seed, draws=2000, mean_block=8):
    """Politis-Romano stationary bootstrap, vectorised over draws (independent implementation;
    random stream differs from the repo's, so only the seed-to-seed spread is informative)."""
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    idx = np.empty((draws, n), dtype=np.int64)
    idx[:, 0] = rng.integers(n, size=draws)
    new = rng.random((draws, n)) < p
    jumps = rng.integers(n, size=(draws, n))
    for i in range(1, n):
        idx[:, i] = np.where(new[:, i], jumps[:, i], (idx[:, i - 1] + 1) % n)
    s = x[idx]
    sh = s.mean(axis=1) / s.std(axis=1, ddof=1) * np.sqrt(52.0)
    return np.quantile(sh, 0.025), np.quantile(sh, 0.975)


rows = []
for (ds, sig), g in ic.groupby(["dataset", "signal"]):
    g = g.sort_values("date")
    l = ls[(ls.dataset == ds) & (ls.signal == sig)].sort_values("date")
    rep = M[(M.dataset == ds) & (M.signal == sig)].iloc[0]
    rec = {
        "dataset": ds, "signal": sig,
        "weeks_ic": len(g), "weeks_ls": len(l),
        "first_date": str(g.date.min().date()), "last_date": str(g.date.max().date()),
        "ic_mean": g.ic.mean(), "ic_nw_t": nw_t(g.ic.to_numpy()), "ic_hit": (g.ic > 0).mean(),
        "gross_sharpe": sharpe(l.gross), "net_sharpe": sharpe(l.net),
        "net_ann_return": l.net.mean() * 52, "annual_turnover": l.turnover.mean() * 52,
    }
    diffs = {
        "d_ic_mean": rec["ic_mean"] - rep.ic_mean, "d_ic_nw_t": rec["ic_nw_t"] - rep.ic_nw_t,
        "d_ic_hit": rec["ic_hit"] - rep.ic_hit, "d_gross_sharpe": rec["gross_sharpe"] - rep.ls_gross_sharpe,
        "d_net_sharpe": rec["net_sharpe"] - rep.ls_net_sharpe, "d_net_ann_return": rec["net_ann_return"] - rep.ls_net_ann_return,
        "d_turnover": rec["annual_turnover"] - rep.ls_annual_turnover,
    }
    rec.update(diffs)
    rec["max_abs_diff"] = max(abs(v) for v in diffs.values())
    rows.append(rec)
R = pd.DataFrame(rows)
R.to_csv(OUT / "recompute_check.csv", index=False)

# bootstrap-seed sensitivity of the CI lower bound (the pass-rule leg)
brow = []
for (ds, sig), l in ls.groupby(["dataset", "signal"]):
    if sig == "rev_1w":
        continue
    x = l.sort_values("date").net.to_numpy()
    rep = M[(M.dataset == ds) & (M.signal == sig)].iloc[0]
    los, his = [], []
    for seed in range(1, 21):
        lo, hi = stat_boot_ci(x, seed)
        los.append(lo); his.append(hi)
    brow.append({"dataset": ds, "signal": sig, "reported_ci_lo": rep.ls_net_sharpe_ci_lo, "reported_ci_hi": rep.ls_net_sharpe_ci_hi,
                 "seeds": 20, "ci_lo_min": min(los), "ci_lo_median": float(np.median(los)), "ci_lo_max": max(los),
                 "share_seeds_ci_lo_gt_0": float(np.mean(np.array(los) > 0))})
B = pd.DataFrame(brow)
B.to_csv(OUT / "bootstrap_seed_sensitivity.csv", index=False)

pd.set_option("display.width", 250)
print(R[["dataset", "signal", "weeks_ic", "first_date", "last_date", "ic_mean", "ic_nw_t", "net_sharpe", "annual_turnover", "max_abs_diff"]].to_string(index=False))
print(B.round(4).to_string(index=False))
print("largest absolute difference from the reported metrics:", R.max_abs_diff.max())
