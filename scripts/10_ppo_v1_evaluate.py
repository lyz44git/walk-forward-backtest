"""First-design evaluation (design/rebalancing_v1_preregistration.md), kept so that reports/rebalancing/v1 can be
regenerated, including the one-process versus eight-process parity check and timing."""

from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmb.rebalance import cases as data, env, ppo
from qmb.rebalance.splits import split_indices

REPORTS = ROOT / "reports" / "rebalancing" / "v1"
SEEDS = range(5)
FIXED_ETA = np.arange(1, 21) / 20
BAND_MULTIPLIER = (0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2)
BASELINES = ("full", "fixed_eta", "no_trade_band")
DRAW_COUNT = 2000
BLOCK_LENGTH = 8


def pick_baselines(market, train_start, train_stop):
    """Select by training net Sharpe, with the lower grid value winning a tie."""
    n_per_leg = np.maximum(1, np.rint(0.2 * market.M.sum(axis=1)))
    widths = 1 / n_per_leg

    def select(grid, simulate):
        return max(grid, key=lambda param: data.stats.sharpe(pd.Series(simulate(param)["net"])))

    eta = select(FIXED_ETA, lambda param: env.simulate(market, train_start, train_stop, eta_const=param))
    band = select(BAND_MULTIPLIER, lambda param: env.simulate(
        market, train_start, train_stop, band=param * widths[train_start:train_stop]))
    return float(eta), float(band), widths


def load_policy(case_path, tag, seed, expected_world):
    stem = data.TMP / "policies" / tag / f"{case_path.stem[5:]}_s{seed}"
    metadata = json.loads(stem.with_suffix(".json").read_text())
    if (metadata["world_size"] != expected_world or metadata["hp"] != ppo.HP or
            len(metadata["history"]) != ppo.HP["updates"]):
        raise ValueError(f"Incomplete or mismatched training metadata: {stem}")
    params = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=True)
    model = ppo.ActorCritic(len(env.FEATURES), metadata["hp"]["hidden"]).double()
    model.load_state_dict(params)
    return model.eval(), metadata


def summarize(record, indices):
    net = pd.Series(record["net"])
    with np.errstate(invalid="ignore", divide="ignore"):
        interval = data.stats.stationary_bootstrap_sharpe(
            net, mean_block=BLOCK_LENGTH, draws=DRAW_COUNT, seed=0)
    eta = record["eta"]
    supplemental = {}
    if not np.isfinite(interval).all():
        supplemental = finite_interval(resampled_sharpe(net.to_numpy(), indices))
    return dict(data.stats.summarize(net, pd.Series(record["turnover"])),
                sharpe_ci_low=interval[0], sharpe_ci_high=interval[1],
                mean_eta=eta.mean() if np.isfinite(eta).all() else np.nan, **supplemental)


def stationary_indices(n):
    """Same stationary resampling scheme and RNG sequence as qmb.stats."""
    rng = np.random.default_rng(0)
    idx = np.empty((DRAW_COUNT, n), dtype=int)
    for path in idx:
        path[0] = rng.integers(n)
        restart = rng.random(n) < 1 / BLOCK_LENGTH
        jump = rng.integers(n, size=n)
        for i in range(1, n):
            path[i] = jump[i] if restart[i] else (path[i - 1] + 1) % n
    return idx


def resampled_sharpe(net, indices):
    paths = net[indices]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(data.stats.PER_YEAR) * paths.mean(axis=1) / paths.std(axis=1, ddof=1)


def finite_interval(draws):
    finite = draws[np.isfinite(draws)]
    lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
    return {"finite_draws": len(finite), "finite_draw_ci_low": lo, "finite_draw_ci_high": hi}


def launcher_time(label):
    text = (data.TMP / f"ppo_{label}.log").read_text()
    match = re.search(r"(?m)^real\s+(\d+)m([\d.]+)s$", text)
    if match is None:
        raise ValueError(f"The launcher wall time was not recorded in ppo_{label}.log")
    return 60 * int(match[1]) + float(match[2])


@torch.no_grad()
def compare_topologies(case_path, market, start, stop):
    single, single_meta = load_policy(case_path, "matched_single", 0, expected_world=1)
    multi, multi_meta = load_policy(case_path, "main", 0, expected_world=8)
    a = env.simulate(market, start, stop, policy=single.mean_action)
    b = env.simulate(market, start, stop, policy=multi.mean_action)
    qmb = data.stats
    comparisons = {
        "max_parameter_abs_diff": max(float((p - q).abs().max())
                                      for p, q in zip(single.parameters(), multi.parameters())),
        "max_weekly_eta_abs_diff": float(np.abs(a["eta"] - b["eta"]).max()),
        "max_weekly_net_abs_diff": float(np.abs(a["net"] - b["net"]).max()),
        "test_sharpe_abs_diff": abs(qmb.sharpe(pd.Series(a["net"])) - qmb.sharpe(pd.Series(b["net"]))),
        "annual_turnover_abs_diff": float(qmb.PER_YEAR * abs(a["turnover"].mean() - b["turnover"].mean())),
        "max_drawdown_abs_diff": abs(qmb.max_drawdown(pd.Series(a["net"])) -
                                     qmb.max_drawdown(pd.Series(b["net"]))),
    }
    tolerances = {key: 1e-8 if key == "max_weekly_net_abs_diff" else 1e-6 for key in comparisons}
    timing = pd.DataFrame([{
        "dataset": "etf", "signal": "ridge", "seed": 0,
        "processes": meta["world_size"], "wall_seconds": meta["wall_seconds"],
        "launcher_wall_seconds": launcher_time(label),
        "updates": len(meta["history"]), "logical_partitions": ppo.ROLLOUT_PARTITIONS,
    } for meta, label in ((single_meta, "single"), (multi_meta, "matched"))])
    timing.to_csv(REPORTS / "timing.csv", index=False)
    pd.DataFrame([{"check": key, "absolute_difference": value, "tolerance": tolerances[key],
                   "passed": value <= tolerances[key]} for key, value in comparisons.items()]).to_csv(
                       REPORTS / "distributed_parity.csv", index=False)
    return all(value <= tolerances[key] for key, value in comparisons.items())


def evaluate_case(case_path):
    case = data.load_case(case_path)
    info = {"dataset": case["dataset"], "signal": case["signal"]}
    lo, hi, start, stop = split_indices(case)
    market = env.Market(case, np.arange(lo, hi))
    fixed_eta, band_multiplier, widths = pick_baselines(market, lo, hi)
    records = {
        ("full", ""): env.simulate(market, start, stop, eta_const=1),
        ("fixed_eta", ""): env.simulate(market, start, stop, eta_const=fixed_eta),
        ("no_trade_band", ""): env.simulate(market, start, stop, band=band_multiplier * widths[start:stop]),
    }
    tuned_values = {"full": 1.0, "fixed_eta": fixed_eta, "no_trade_band": band_multiplier}
    np.testing.assert_allclose(records["full", ""]["net"], market.full[start:stop, 0], atol=1e-12, rtol=0)
    for seed in SEEDS:
        model, _ = load_policy(case_path, "main", seed, expected_world=8)
        records["ppo", seed] = env.simulate(market, start, stop, policy=model.mean_action)
    records["ppo_seed_mean", ""] = {
        k: np.mean([records["ppo", seed][k] for seed in SEEDS], axis=0) for k in records["ppo", 0]
    }

    indices = stationary_indices(stop - start)
    rows, week_frames = [], []
    for (strategy, seed), rec in records.items():
        identity = dict(info, strategy=strategy, seed=seed)
        rows.append(dict(identity, train_start=case["signal_date"][lo], train_end=case["signal_date"][hi-1],
                         test_start=case["signal_date"][start], test_end=case["signal_date"][stop-1],
                         tuned_value=tuned_values.get(strategy, np.nan), **summarize(rec, indices)))
        week_frames.append(pd.DataFrame(dict(identity, date=case["signal_date"][start:stop], **rec)))

    indexed = {(row["strategy"], row["seed"]): row for row in rows}
    ppo_resampled = resampled_sharpe(records["ppo_seed_mean", ""]["net"], indices)
    average_sharpe = indexed["ppo_seed_mean", ""]["sharpe"]
    differences = []
    for benchmark in BASELINES:
        baseline_sharpe = indexed[benchmark, ""]["sharpe"]
        delta = ppo_resampled - resampled_sharpe(records[benchmark, ""]["net"], indices)
        interval_defined = np.isfinite(delta).all()
        lower, upper = np.quantile(delta, [0.025, 0.975]) if interval_defined else (np.nan, np.nan)
        supplemental = {} if interval_defined else finite_interval(delta)
        n_higher = sum(indexed["ppo", seed]["sharpe"] > baseline_sharpe for seed in SEEDS)
        differences.append(dict(info, baseline=benchmark, sharpe_difference=average_sharpe - baseline_sharpe,
                                paired_ci_low=lower, paired_ci_high=upper, seeds_higher=n_higher,
                                meets_rule=bool(n_higher == len(SEEDS) and lower > 0), **supplemental))
    return rows, week_frames, differences, (market, start, stop)


def main():
    torch.set_num_threads(1)
    cases = sorted(data.TMP.glob("case_*.npz"))
    if not cases:
        raise FileNotFoundError("No case caches")
    rows, frames, comparisons = [], [], []
    matched = None
    for path in cases:
        print(f"Evaluating {path.name}", flush=True)
        results, week_frames, differences, market_slice = evaluate_case(path)
        rows.extend(results)
        frames.extend(week_frames)
        comparisons.extend(differences)
        if path.stem == "case_etf_ridge":
            matched = (path, *market_slice)
    pd.DataFrame(rows).to_csv(REPORTS / "results.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(REPORTS / "weekly_results.csv", index=False)
    pd.DataFrame(comparisons).to_csv(REPORTS / "paired_comparisons.csv", index=False)
    if not compare_topologies(*matched):
        raise AssertionError("Distributed comparison exceeded preregistered tolerances")


if __name__ == "__main__":
    main()
