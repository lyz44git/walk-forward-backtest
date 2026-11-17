"""Evaluate the rebalancing rules of design/rebalancing_v2_plan.md on the frozen test spans.

Inputs:  tmp/rebalance/case_*.npz (scripts/07_build_targets.py) and tmp/rebalance/policies/main (08_train_ppo.py).
Outputs: reports/rebalancing/rules_results.csv   one row per case x cost level x rule
         reports/rebalancing/rules_weekly.csv    weekly series at the base cost
         reports/rebalancing/rules_paired.csv    per-case Sharpe differences with paired intervals
         reports/rebalancing/rules_pooled.csv    primary estimand: mean difference over the three signals
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmb import stats  # noqa: E402
from qmb.rebalance import cases, env, ppo, rules  # noqa: E402
from qmb.rebalance.splits import split_indices  # noqa: E402

OUT = ROOT / "reports" / "rebalancing"
COSTS = (0.0, 2.0, 5.0, 10.0)
BASE_COST = 5.0
GRIDS = {                                   # ordered from full rebalancing outwards; first best wins a tie
    "partial_adjustment": tuple(np.round(np.arange(20, 0, -1) / 20, 2)),
    "no_trade_band": (0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2),
    "rank_buffer": (1, 1.25, 1.5, 1.75, 2),
    "signal_smoothing": (0, 1, 2, 4, 8),
}
SIMPLE = tuple(GRIDS)
RULES = ("full",) + SIMPLE + ("ppo",)
EXPOSURE = (0.90, 1.10)
SEEDS = range(5)
DRAWS, BLOCKS, MAIN_BLOCK = 2000, (4, 8, 16), 8


def run(market, name, param, t0, t1, cost, cache):
    if name == "full":
        return env.simulate(market, t0, t1, eta_const=1.0, cost_bps=cost)
    if name == "partial_adjustment":
        return env.simulate(market, t0, t1, eta_const=param, cost_bps=cost)
    if name == "no_trade_band":
        k = np.maximum(1, np.rint(rules.QUANTILE * market.M.sum(axis=1)))
        return env.simulate(market, t0, t1, band=param / k[t0:t1], cost_bps=cost)
    key = (name, param)
    if key not in cache:
        if name == "rank_buffer":
            cache[key] = rules.buffered_targets(market.S, market.M, param)
        else:
            cache[key] = rules.quintile_targets(rules.smoothed_scores(market.S, market.M, param), market.M)
    return env.simulate(market, t0, t1, eta_const=1.0, targets=cache[key], cost_bps=cost)


def select(market, name, lo, hi, cost, cache):
    """Best admissible grid value by training net Sharpe; the first value (full rebalancing) is the fallback."""
    best, best_sharpe = GRIDS[name][0], -np.inf
    for param in GRIDS[name]:
        rec = run(market, name, param, lo, hi, cost, cache)
        sharpe = stats.sharpe(pd.Series(rec["net"]))
        if EXPOSURE[0] <= rec["short"].mean() <= EXPOSURE[1] and sharpe > best_sharpe + 1e-12:
            best, best_sharpe = param, sharpe
    return float(best), float(best_sharpe)


def load_policies(case):
    models = []
    for seed in SEEDS:
        stem = cases.TMP / "policies" / "main" / f"{case['dataset']}_{case['signal']}_s{seed}"
        meta = json.loads(stem.with_suffix(".json").read_text())
        model = ppo.ActorCritic(len(env.FEATURES), meta["hp"]["hidden"]).double()
        model.load_state_dict(torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=True))
        models.append(model.eval())
    return models


def bootstrap_indices(n, block):
    """Stationary bootstrap (Politis and Romano); the same weeks are used for every series compared."""
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
    torch.set_num_threads(1)
    paths = sorted(cases.TMP.glob("case_*.npz"))
    if not paths:
        raise FileNotFoundError("No case caches; run scripts/07_build_targets.py first")
    rows, weekly, series = [], [], {}
    for path in paths:
        case = cases.load_case(path)
        ds, sig = case["dataset"], case["signal"]
        lo, hi, t0, t1 = split_indices(case)
        market = env.Market(case, np.arange(lo, hi))
        policies, cache = load_policies(case), {}
        for cost in COSTS:
            for name in RULES:
                param, train_sharpe = np.nan, np.nan
                if name == "ppo":
                    recs = [env.simulate(market, t0, t1, policy=m.mean_action, cost_bps=cost) for m in policies]
                    rec = {k: np.mean([r[k] for r in recs], axis=0) for k in recs[0]}
                else:
                    if name != "full":
                        param, train_sharpe = select(market, name, lo, hi, cost, cache)
                    rec = run(market, name, param, t0, t1, cost, cache)
                net = pd.Series(rec["net"])
                rows.append({
                    "dataset": ds, "signal": sig, "cost_bps": cost, "rule": name, "param": param,
                    "train_sharpe": train_sharpe, "weeks": len(net), "sharpe": stats.sharpe(net),
                    "ann_net": net.mean() * stats.PER_YEAR, "ann_gross": rec["gross"].mean() * stats.PER_YEAR,
                    "ann_cost": (rec["gross"] - rec["net"]).mean() * stats.PER_YEAR,
                    "ann_vol": net.std(ddof=1) * np.sqrt(stats.PER_YEAR),
                    "annual_turnover": rec["turnover"].mean() * stats.PER_YEAR,
                    "mean_short": rec["short"].mean(), "max_drawdown": stats.max_drawdown(net),
                })
                series[ds, sig, cost, name] = rec["net"]
                if cost == BASE_COST:
                    weekly.append(pd.DataFrame({
                        "date": case["signal_date"][t0:t1], "dataset": ds, "signal": sig, "rule": name,
                        "net": rec["net"], "gross": rec["gross"], "turnover": rec["turnover"],
                        "short": rec["short"]}))
        print(f"evaluated {ds} {sig}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "rules_results.csv", index=False)
    pd.concat(weekly, ignore_index=True).to_csv(OUT / "rules_weekly.csv", index=False)

    paired, pooled = [], []
    datasets = sorted({k[0] for k in series})
    for ds in datasets:
        signals = sorted({k[1] for k in series if k[0] == ds})
        n = len(series[ds, signals[0], BASE_COST, "full"])
        for block in BLOCKS:
            idx = bootstrap_indices(n, block)
            for cost in (COSTS if block == MAIN_BLOCK else (BASE_COST,)):
                boot = {(s, r): resampled_sharpe(series[ds, s, cost, r], idx) for s in signals for r in RULES}
                point = {(s, r): stats.sharpe(pd.Series(series[ds, s, cost, r])) for s in signals for r in RULES}
                comparisons = [(r, "full") for r in RULES[1:]] + [("ppo", r) for r in SIMPLE]
                for a, b in comparisons:
                    deltas = np.stack([boot[s, a] - boot[s, b] for s in signals])
                    pooled.append({"dataset": ds, "cost_bps": cost, "block": block, "rule": a, "versus": b,
                                   "mean_sharpe_difference": np.mean([point[s, a] - point[s, b] for s in signals]),
                                   **interval(deltas.mean(axis=0))})
                    if cost == BASE_COST:
                        for s, d in zip(signals, deltas):
                            paired.append({"dataset": ds, "signal": s, "block": block, "rule": a, "versus": b,
                                           "sharpe_difference": point[s, a] - point[s, b], **interval(d)})
    pd.DataFrame(paired).to_csv(OUT / "rules_paired.csv", index=False)
    pd.DataFrame(pooled).to_csv(OUT / "rules_pooled.csv", index=False)


if __name__ == "__main__":
    main()
