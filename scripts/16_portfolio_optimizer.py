"""Part 5 (design/portfolio_optimizer_plan.md): mean-variance optimizer against full rebalancing.

Inputs:  tmp/rebalance/case_*.npz (scripts/07_build_targets.py); reports/rebalancing/rules_weekly.csv (Part 2).
Outputs: reports/optimizer/selection.csv  training net Sharpe and exposure of every grid point, case and cost
         reports/optimizer/results.csv    one row per case x cost x variant on the test span
         reports/optimizer/weekly.csv     weekly test-span series at 5 bps
         reports/optimizer/paired.csv     per-case Sharpe differences with paired intervals (5 bps)
         reports/optimizer/pooled.csv     mean difference over the three signals: primary and secondary
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmb import stats  # noqa: E402
from qmb.rebalance import cases, env, optimizer as opt  # noqa: E402
from qmb.rebalance.splits import split_indices  # noqa: E402

OUT = ROOT / "reports" / "optimizer"
COSTS = (0.0, 2.0, 5.0, 10.0)
BASE_COST = 5.0
VARIANTS = {"optimizer": None, "optimizer_no_penalty": (0,)}
DRAWS, BLOCKS, MAIN_BLOCK = 2000, (4, 8, 16), 8
PART2_RULES = ("full", "partial_adjustment", "no_trade_band", "rank_buffer", "signal_smoothing", "ppo")


def sharpe(x):
    return stats.sharpe(pd.Series(x))


def run_case(path):
    case = cases.load_case(path)
    ds, sig = case["dataset"], case["signal"]
    lo, hi, t0, t1 = split_indices(case)
    ic = float(np.nanmean(case["IC"][lo:hi]))
    alphas, roots, shrink = opt.risk_inputs(case, range(lo, t1), ic)
    problem = opt.WeeklyProblem(case["W"].shape[1])
    market = SimpleNamespace(W=case["W"], R=case["R"], M=case["M"])
    selection, results, weekly, series = [], [], [], {}
    for cost in COSTS:
        train = {}
        for gamma, m in opt.grid(cost):
            rec = opt.simulate(case, alphas, roots, lo, hi, gamma, m, problem, cost_bps=cost)
            train[gamma, m] = (sharpe(rec["net"]), float(rec["short"].mean()))
            selection.append({"dataset": ds, "signal": sig, "cost_bps": cost, "gamma": gamma, "m": m,
                              "train_sharpe": train[gamma, m][0], "train_mean_short": train[gamma, m][1],
                              "not_optimal": rec["not_optimal"], "max_violation": rec["violation"].max()})
        runs = {"full": env.simulate(market, t0, t1, eta_const=1.0, cost_bps=cost)}
        params = {"full": (np.nan, np.nan, np.nan, True)}
        for name, penalties in VARIANTS.items():
            (gamma, m), admissible = opt.select(train, penalties)
            runs[name] = opt.simulate(case, alphas, roots, t0, t1, gamma, m, problem, cost_bps=cost)
            params[name] = (gamma, m, train[gamma, m][0], admissible)
        for name, rec in runs.items():
            gamma, m, train_sharpe, admissible = params[name]
            net = rec["net"]
            results.append({
                "dataset": ds, "signal": sig, "cost_bps": cost, "variant": name, "gamma": gamma, "m": m,
                "admissible": admissible, "ic_train": ic, "train_sharpe": train_sharpe, "weeks": len(net),
                "sharpe": sharpe(net), "ann_net": net.mean() * stats.PER_YEAR,
                "ann_gross": rec["gross"].mean() * stats.PER_YEAR,
                "ann_cost": (rec["gross"] - net).mean() * stats.PER_YEAR,
                "ann_vol": net.std(ddof=1) * np.sqrt(stats.PER_YEAR),
                "annual_turnover": rec["turnover"].mean() * stats.PER_YEAR, "mean_short": rec["short"].mean(),
                "max_drawdown": stats.max_drawdown(pd.Series(net)),
                "not_optimal": rec.get("not_optimal", 0),
                "max_violation": float(rec["violation"].max()) if "violation" in rec else 0.0,
                "mean_shrinkage": float(np.mean([shrink[t] for t in range(t0, t1)]))})
            series[name, cost] = net
            if cost == BASE_COST:
                weekly.append(pd.DataFrame({"date": case["signal_date"][t0:t1], "dataset": ds, "signal": sig,
                                            "variant": name, "net": net, "gross": rec["gross"],
                                            "turnover": rec["turnover"], "short": rec["short"]}))
    print(f"done {ds} {sig}", flush=True)
    return ds, sig, selection, results, pd.concat(weekly, ignore_index=True), series


def bootstrap_indices(n, block):
    """Part 2's index paths (seed 0); the C++ kernel when available, else the same walk in Python."""
    try:
        from qmb import native
        return native.bootstrap_indices(n, block, DRAWS, seed=0)
    except Exception:   # noqa: BLE001  (no compiler: fall back to the reference walk)
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
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(stats.PER_YEAR) * paths.mean(axis=1) / paths.std(axis=1, ddof=1)
    return np.where(np.isfinite(out), out, np.nan)


def interval(delta):
    ok = delta[np.isfinite(delta)]
    lo, hi = np.quantile(ok, [0.025, 0.975])
    return {"ci_low": lo, "ci_high": hi, "valid_draws": len(ok), "share_above_zero": float((ok > 0).mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(cases.TMP.glob("case_*.npz"))
    if not paths:
        raise FileNotFoundError("No case caches; run scripts/07_build_targets.py first")
    with ProcessPoolExecutor(max_workers=len(paths)) as pool:
        done = list(pool.map(run_case, paths))
    selection = pd.DataFrame([row for d in done for row in d[2]])
    results = pd.DataFrame([row for d in done for row in d[3]])
    weekly = pd.concat([d[4] for d in done], ignore_index=True)
    series = {(d[0], d[1], name, cost): v for d in done for (name, cost), v in d[5].items()}
    selection.to_csv(OUT / "selection.csv", index=False)
    results.to_csv(OUT / "results.csv", index=False)
    weekly.to_csv(OUT / "weekly.csv", index=False)

    part2 = pd.read_csv(ROOT / "reports" / "rebalancing" / "rules_weekly.csv")
    for (ds, sig), group in part2.groupby(["dataset", "signal"]):
        for rule, g in group.groupby("rule"):
            series[ds, sig, f"part2_{rule}", BASE_COST] = g["net"].to_numpy()
        gap = np.abs(series[ds, sig, "part2_full", BASE_COST] - series[ds, sig, "full", BASE_COST]).max()
        if gap > 1e-12:
            raise AssertionError(f"full rebalancing differs from Part 2 for {ds} {sig}: {gap}")

    paired, pooled = [], []
    for ds in sorted({k[0] for k in series}):
        signals = sorted({k[1] for k in series if k[0] == ds})
        n = len(series[ds, signals[0], "full", BASE_COST])
        for block in BLOCKS:
            idx = bootstrap_indices(n, block)
            comparisons = [("optimizer", "full", c) for c in (COSTS if block == MAIN_BLOCK else (BASE_COST,))]
            if block == MAIN_BLOCK:
                comparisons += [("optimizer_no_penalty", "full", c) for c in COSTS]
                comparisons += [("optimizer", f"part2_{r}", BASE_COST) for r in PART2_RULES[1:]]
            for a, b, cost in comparisons:
                boot = {s: resampled_sharpe(series[ds, s, a, cost], idx) - resampled_sharpe(series[ds, s, b, cost], idx)
                        for s in signals}
                point = {s: sharpe(series[ds, s, a, cost]) - sharpe(series[ds, s, b, cost]) for s in signals}
                pooled.append({"dataset": ds, "cost_bps": cost, "block": block, "variant": a, "versus": b,
                               "primary": a == "optimizer" and b == "full" and cost == BASE_COST and block == MAIN_BLOCK,
                               "mean_sharpe_difference": float(np.mean(list(point.values()))),
                               **interval(np.mean([boot[s] for s in signals], axis=0))})
                if cost == BASE_COST and block == MAIN_BLOCK:
                    for s in signals:
                        paired.append({"dataset": ds, "signal": s, "variant": a, "versus": b,
                                       "sharpe_difference": point[s], **interval(boot[s])})
    pd.DataFrame(paired).to_csv(OUT / "paired.csv", index=False)
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(OUT / "pooled.csv", index=False)
    print(pooled[pooled.primary].to_string(index=False))


if __name__ == "__main__":
    main()
