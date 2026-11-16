"""Placebo distribution (deviation from the pre-registration; see design/deviations.md).

For each dataset and fitted model, re-run the walk-forward with training targets permuted across assets within
each week (a fresh permutation at every refit), for many seeds. Report the placebo distribution of the mean OOS
IC and where the real model's IC falls in it (empirical p-value = share of placebo seeds with IC >= real IC).
Ridge: 100 seeds; LightGBM: 10 seeds (slower). Writes reports/placebo_seeds.csv and reports/placebo_summary.csv.
"""
import sys, os, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd

SEEDS = {"ridge": list(range(1, 101)), "lgbm": list(range(1, 11))}
_PANEL = {}


def _init(threads):
    from qmb import engine
    engine.LGB_PARAMS["num_threads"] = threads


def _panel(name):
    if name not in _PANEL:
        from qmb import engine
        d = _load(name)
        sch = engine.weekly_schedule(d["tri"].index)
        exec_px = d["open"] if d["open"] is not None else d["tri"]
        _PANEL[name] = (engine.build_panel(d["tri"], exec_px, engine.daily_features(d["tri"], d["dv"]), sch), d["initial"])
    return _PANEL[name]


def _load(name):
    from qmb import data
    if name == "etf":
        P = pd.read_parquet(data.CLEAN / "etf_daily.parquet"); P = P[P.date >= "2007-04-11"]
        piv = lambda c: P.pivot(index="date", columns="ticker", values=c).sort_index()
        return {"tri": piv("close_adj"), "open": piv("open_adj"), "dv": piv("dollar_volume"), "initial": 156}
    I = pd.read_parquet(data.CLEAN / "french_vw_daily.parquet").loc["1969-07-01":]
    return {"tri": (1 + I).cumprod().where(I.notna()), "open": None, "dv": None, "initial": 260}


def _job(args):
    name, signal, seed = args
    from qmb import engine, stats
    panel, initial = _panel(name)
    p = engine.walk_forward(panel, signal, initial, placebo=True, seed=seed)
    ic = stats.weekly_ic(p, panel.ret)
    return {"dataset": name, "signal": signal, "seed": seed, "ic_mean": float(ic.mean()), "ic_nw_t": stats.newey_west_t(ic, 4)}


def summarize():
    S = pd.read_csv(ROOT / "reports" / "placebo_seeds.csv")
    M = pd.read_csv(ROOT / "reports" / "main_results.csv")
    rows = []
    for (n, s), g in S.groupby(["dataset", "signal"]):
        real = M[(M.dataset == n) & (M.signal == s)].iloc[0]
        rows.append({"dataset": n, "signal": s, "seeds": len(g), "placebo_ic_mean": g.ic_mean.mean(), "placebo_ic_sd": g.ic_mean.std(),
                     "placebo_ic_p2_5": g.ic_mean.quantile(0.025), "placebo_ic_p97_5": g.ic_mean.quantile(0.975),
                     "share_placebo_abs_t_gt_2": float((g.ic_nw_t.abs() > 2).mean()),
                     "real_ic_mean": real.ic_mean, "real_ic_nw_t": real.ic_nw_t,
                     "empirical_p_value": float(((g.ic_mean >= real.ic_mean).sum() + 1) / (len(g) + 1))})
    out = pd.DataFrame(rows); out.to_csv(ROOT / "reports" / "placebo_summary.csv", index=False)
    pd.set_option("display.width", 220); print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summarize(); sys.exit(0)
    t0 = time.time()
    jobs = [(n, s, k) for n in ["etf", "french"] for s in ["lgbm", "ridge"] for k in SEEDS[s]]
    workers = int(os.environ.get("PLACEBO_WORKERS", "8"))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(2,)) as ex:
        res = []
        for i, r in enumerate(ex.map(_job, jobs, chunksize=1)):
            res.append(r)
            if (i + 1) % 20 == 0:
                print(f"[{time.time()-t0:6.0f}s] {i+1}/{len(jobs)}", flush=True)
    pd.DataFrame(res).to_csv(ROOT / "reports" / "placebo_seeds.csv", index=False)
    print(f"done {time.time()-t0:.0f}s")
