"""Part 4: features that are not functions of the assets' own past prices (design/nonprice_features_plan.md).

Coverage:   python scripts/13_nonprice_features.py coverage
            (writes reports/nonprice/coverage.csv; run before any fit)
Fit:        python scripts/13_nonprice_features.py fit --dataset french --signal lgbm --set PN
            [--variant main|lagged|placebo] [--seed 0]
            (writes tmp/nonprice/pred_*.parquet; set P reproduces Part 1)
Summarize:  python scripts/13_nonprice_features.py summarize
            (writes reports/nonprice/: results.csv, weekly.csv, pooled.csv, placebo.csv, attribution.csv,
             feature_weights.csv)
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmb import engine, nonprice, stats  # noqa: E402
from qmb.attribution import ols_nw, weekly_factors  # noqa: E402
from qmb.rebalance import cases  # noqa: E402

TMP, OUT = ROOT / "tmp" / "nonprice", ROOT / "reports" / "nonprice"
MODELS, SETS, COSTS, BASE_COST, BORROW_BPS = ("ridge", "lgbm"), ("P", "PN", "N"), (0.0, 2.0, 5.0, 10.0), 5.0, 50.0
PLACEBO_SEEDS = {"ridge": range(20), "lgbm": range(3)}
LAG_YEARS, LAG_MONTHS = 1, 3                     # timing check: further lag on annual and monthly fields
DRAWS, BLOCKS, MAIN_BLOCK = 2000, (4, 8, 16), 8


def build(name, lagged=False):
    """Panel with the Part 1 features and the new ones; its rows are exactly the Part 1 rows."""
    d = cases.load(name)
    tri = d["tri"]
    exec_px = d["open"] if d["open"] is not None else tri
    sch = engine.weekly_schedule(tri.index)
    base = engine.daily_features(tri, d["dv"])
    if name == "french":
        new = nonprice.industry_features(LAG_YEARS, LAG_MONTHS) if lagged else nonprice.industry_features()
    else:
        new = nonprice.etf_features(list(tri.columns))
    panel = engine.build_panel(tri, exec_px, {**base, **new}, sch, neutral=tuple(new))
    return d, panel, list(base), list(new), new


def columns(fset, base, new):
    return {"P": base, "PN": base + new, "N": new}[fset]


def stem(name, signal, fset, variant, seed):
    return f"pred_{name}_{signal}_{fset}_{variant}" + (f"_s{seed}" if variant == "placebo" else "")


def coverage():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("etf", "french"):
        _, panel, _, new, raw = build(name)
        sig = pd.DatetimeIndex(panel.sched["signal"])
        year = pd.Series(sig.year[panel.X.index.get_level_values("week")], index=panel.X.index)
        for nm in new:
            t = raw[nm].reindex(sig)
            t.index = range(len(sig))
            missing = t.stack(future_stack=True).reindex(panel.X.index).isna()
            by_year = missing.groupby(year).mean()
            rows.append({"dataset": name, "feature": nm, "panel_rows": len(missing), "missing_share": missing.mean(),
                         "worst_year": int(by_year.idxmax()), "worst_year_missing_share": by_year.max(),
                         "kept": bool(missing.mean() <= 0.20)})
    pd.DataFrame(rows).to_csv(OUT / "coverage.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def fit(name, signal, fset, variant, seed):
    d, panel, base, new, _ = build(name, lagged=(variant == "lagged"))
    X = panel.X
    if variant == "placebo":                     # the new features of each week are reassigned across assets
        rng = np.random.default_rng(seed)
        X = X.copy()
        block = X[new].to_numpy()
        week = X.index.get_level_values("week").to_numpy()
        starts = np.flatnonzero(np.r_[True, week[1:] != week[:-1], True])
        for a, b in zip(starts[:-1], starts[1:]):
            block[a:b] = block[a:b][rng.permutation(b - a)]
        X[new] = block
    cols = columns(fset, base, new)
    log = []
    pred = engine.walk_forward(dataclasses.replace(panel, X=X[cols], features=cols), signal, d["initial"], fit_log=log)
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / stem(name, signal, fset, variant, seed)
    pred.rename("pred").to_frame().to_parquet(out.with_suffix(".parquet"))
    if variant == "main":
        pd.DataFrame([v for _, v in log], index=[k for k, _ in log], columns=cols).to_parquet(
            TMP / f"weights_{name}_{signal}_{fset}.parquet")
    print(f"{out.name}: {pred.index.get_level_values('week').nunique()} weeks", flush=True)


def bootstrap_indices(n, block):
    rng = np.random.default_rng(0)
    idx = np.empty((DRAWS, n), dtype=int)
    for path in idx:
        path[0] = rng.integers(n)
        restart = rng.random(n) < 1.0 / block
        jump = rng.integers(n, size=n)
        for i in range(1, n):
            path[i] = jump[i] if restart[i] else (path[i - 1] + 1) % n
    return idx


def resampled_sharpe(net, idx):
    paths = net[idx]
    return np.sqrt(stats.PER_YEAR) * paths.mean(axis=1) / paths.std(axis=1, ddof=1)


def summarize():
    OUT.mkdir(parents=True, exist_ok=True)
    ls = pd.read_csv(ROOT / "reports" / "ls_weekly.csv", parse_dates=["date"])
    rows, weekly, pooled, placebo, attr, weights = [], [], [], [], [], []
    for name in ("etf", "french"):
        d, panel, base, new, _ = build(name)
        sig_dates = pd.DatetimeIndex(panel.sched["signal"])
        read = lambda s: pd.read_parquet(TMP / f"{s}.parquet")["pred"]
        preds = {(m, f): read(stem(name, m, f, "main", 0)) for m in MODELS for f in SETS}
        if name == "french":
            preds.update({(m, "PN_lagged"): read(stem(name, m, "PN", "lagged", 0)) for m in MODELS})
        test_weeks = preds["ridge", "P"].index.get_level_values("week").unique()
        for nm in new:
            preds["sort", nm] = nonprice.SIGN[nm] * panel.X[nm].loc[test_weeks]
        ports = {k: engine.portfolio(p, panel.ret) for k, p in preds.items()}
        dates = sig_dates[ports["ridge", "P"].index]
        mom = ls[(ls.dataset == name) & (ls.signal == "mom_12_1")].sort_values("date")
        assert (mom.date.to_numpy() == dates.to_numpy()).all(), "weeks differ from Part 1"
        for m in MODELS:                         # set P must be the published Part 1 portfolio
            part1 = ls[(ls.dataset == name) & (ls.signal == m)].sort_values("date").gross.to_numpy()
            assert np.abs(part1 - ports[m, "P"].ret_gross.to_numpy()).max() < 1e-10, f"{name} {m}: P differs from Part 1"
        mom_net = mom.net.to_numpy()
        factors = weekly_factors(panel.sched["entry"].loc[ports["ridge", "P"].index],
                                 panel.sched["exit"].loc[ports["ridge", "P"].index], name)
        base_x = np.column_stack([np.ones(len(dates)), factors.to_numpy()])
        borrow = BORROW_BPS / 1e4 / stats.PER_YEAR

        for (signal, fset), port in ports.items():
            pred = preds[signal, fset]
            ic = stats.weekly_ic(pred, panel.ret)
            rank = pred.unstack("asset").rank(axis=1)
            autocorr = rank.corrwith(rank.shift(1), axis=1).mean()
            net5 = engine.net_returns(port, BASE_COST)
            lo, hi = stats.stationary_bootstrap_sharpe(net5)
            for cost in COSTS:
                net = engine.net_returns(port, cost)
                rows.append({"dataset": name, "signal": signal, "features": fset, "cost_bps": cost, "weeks": len(net),
                             "ic_mean": ic.mean(), "ic_nw_t": stats.newey_west_t(ic), "rank_autocorr_1w": autocorr,
                             "sharpe": stats.sharpe(net), "sharpe_ci_low_5bps": lo, "sharpe_ci_high_5bps": hi,
                             "ann_gross": port.ret_gross.mean() * stats.PER_YEAR, "ann_net": net.mean() * stats.PER_YEAR,
                             "annual_turnover": port.turnover.mean() * stats.PER_YEAR,
                             "breakeven_cost_bps": (port.ret_gross.mean() - borrow) / port.turnover.mean() * 1e4,
                             "corr_with_momentum": np.corrcoef(net5.to_numpy(), mom_net)[0, 1]})
            weekly.append(pd.DataFrame({"date": dates, "dataset": name, "signal": signal, "features": fset,
                                        "gross": port.ret_gross.to_numpy(), "turnover": port.turnover.to_numpy(),
                                        "net": net5.to_numpy()}))
            y = net5.to_numpy()
            key = {"dataset": name, "signal": signal, "features": fset}
            attr.append(dict(key, model="market, size, value", **ols_nw(y, base_x, list(factors.columns))))
            attr.append(dict(key, model="market, size, value, same-universe momentum",
                             **ols_nw(y, np.column_stack([base_x, mom_net]), list(factors.columns) + ["momentum"])))

        n = len(dates)
        half = np.arange(n) < n // 2
        spans = {"full": np.ones(n, bool), "first_half": half, "second_half": ~half,
                 "part2_test": np.asarray(dates >= cases.SPLIT[name])}
        others = [f for f in ("PN", "N", "PN_lagged") if ("ridge", f) in ports]
        for span, keep in spans.items():
            for block in (BLOCKS if span == "full" else (MAIN_BLOCK,)):
                idx = bootstrap_indices(int(keep.sum()), block)
                for cost in (COSTS if (block == MAIN_BLOCK and span == "full") else (BASE_COST,)):
                    net = {k: engine.net_returns(p, cost).to_numpy()[keep] for k, p in ports.items() if k[0] in MODELS}
                    boot = {k: resampled_sharpe(v, idx) for k, v in net.items()}
                    point = {k: stats.sharpe(pd.Series(v)) for k, v in net.items()}
                    for f in others:
                        for group, members in (("mean", MODELS),) + tuple((m, (m,)) for m in MODELS):
                            delta = np.mean([boot[m, f] - boot[m, "P"] for m in members], axis=0)
                            pooled.append({"dataset": name, "span": span, "cost_bps": cost, "block": block,
                                           "features": f, "versus": "P", "model": group,
                                           "sharpe_difference": np.mean([point[m, f] - point[m, "P"] for m in members]),
                                           "ci_low": np.quantile(delta, 0.025), "ci_high": np.quantile(delta, 0.975),
                                           "share_above_zero": float((delta > 0).mean())})

        for m in MODELS:
            p_sharpe = stats.sharpe(engine.net_returns(ports[m, "P"], BASE_COST))
            real = stats.sharpe(engine.net_returns(ports[m, "PN"], BASE_COST)) - p_sharpe
            diffs = np.array([stats.sharpe(engine.net_returns(engine.portfolio(
                read(stem(name, m, "PN", "placebo", s)), panel.ret), BASE_COST)) - p_sharpe for s in PLACEBO_SEEDS[m]])
            placebo.append({"dataset": name, "signal": m, "seeds": len(diffs), "real_sharpe_difference": real,
                            "placebo_mean": diffs.mean(), "placebo_min": diffs.min(), "placebo_max": diffs.max(),
                            "share_placebo_at_least_real": float((diffs >= real).mean())})
            w = pd.read_parquet(TMP / f"weights_{name}_{m}_PN.parquet")
            for nm in w.columns:
                weights.append({"dataset": name, "signal": m, "feature": nm, "new": nm in new, "refits": len(w),
                                "quantity": "ridge slope" if m == "ridge" else "share of gain",
                                "mean": w[nm].mean(), "p10": w[nm].quantile(0.1), "p90": w[nm].quantile(0.9),
                                "share_of_refits_positive": float((w[nm] > 0).mean())})
        print(f"summarized {name}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    pd.concat(weekly, ignore_index=True).to_csv(OUT / "weekly.csv", index=False)
    pd.DataFrame(pooled).to_csv(OUT / "pooled.csv", index=False)
    pd.DataFrame(placebo).to_csv(OUT / "placebo.csv", index=False)
    pd.DataFrame(attr).to_csv(OUT / "attribution.csv", index=False)
    pd.DataFrame(weights).to_csv(OUT / "feature_weights.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("coverage", "fit", "summarize"))
    ap.add_argument("--dataset", choices=("etf", "french"))
    ap.add_argument("--signal", choices=MODELS)
    ap.add_argument("--set", choices=SETS, dest="fset")
    ap.add_argument("--variant", choices=("main", "lagged", "placebo"), default="main")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"coverage": coverage, "summarize": summarize}.get(
        a.command, lambda: fit(a.dataset, a.signal, a.fset, a.variant, a.seed))()
