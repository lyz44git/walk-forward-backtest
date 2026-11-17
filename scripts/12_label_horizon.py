"""Part 3, step B: walk-forward predictions with longer label horizons (design/label_horizon_plan.md).

Fit:        python scripts/12_label_horizon.py fit --dataset etf --signal lgbm --horizon 4
            (writes tmp/horizon/pred_{dataset}_{signal}_h{H}.parquet; H = 1 reproduces Part 1)
Summarize:  python scripts/12_label_horizon.py summarize
            (writes reports/horizon/horizon_results.csv, horizon_weekly.csv, horizon_pooled.csv)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmb import engine, stats  # noqa: E402
from qmb.rebalance import cases  # noqa: E402

TMP, OUT = ROOT / "tmp" / "horizon", ROOT / "reports" / "horizon"
MODELS, HORIZONS, COSTS, BASE_COST = ("ridge", "lgbm"), (1, 4, 13), (0.0, 2.0, 5.0, 10.0), 5.0
SPLIT = cases.SPLIT
DRAWS, BLOCKS, MAIN_BLOCK = 2000, (4, 8, 16), 8


def build(name):
    d = cases.load(name)
    tri = d["tri"]
    exec_px = d["open"] if d["open"] is not None else tri
    sch = engine.weekly_schedule(tri.index)
    panel = engine.build_panel(tri, exec_px, engine.daily_features(tri, d["dv"]), sch)
    return d, panel, exec_px


def fit(name, signal, horizon):
    d, panel, exec_px = build(name)
    target = engine.horizon_target(panel, exec_px, horizon)
    pred = engine.walk_forward(panel, signal, d["initial"], target=target, label_weeks=horizon)
    TMP.mkdir(parents=True, exist_ok=True)
    pred.rename("pred").to_frame().to_parquet(TMP / f"pred_{name}_{signal}_h{horizon}.parquet")
    print(f"{name} {signal} H={horizon}: {pred.index.get_level_values('week').nunique()} weeks", flush=True)


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
    rows, weekly, pooled = [], [], []
    for name in ("etf", "french"):
        _, panel, _ = build(name)
        sig_dates = pd.DatetimeIndex(panel.sched["signal"])
        ports = {}
        for signal in MODELS:
            for h in HORIZONS:
                pred = pd.read_parquet(TMP / f"pred_{name}_{signal}_h{h}.parquet")["pred"]
                port = engine.portfolio(pred, panel.ret)
                ic = stats.weekly_ic(pred, panel.ret)
                rank = pred.unstack("asset").rank(axis=1)
                autocorr = rank.corrwith(rank.shift(1), axis=1).mean()
                ports[signal, h] = port
                for cost in COSTS:
                    net = engine.net_returns(port, cost)
                    rows.append({"dataset": name, "signal": signal, "horizon": h, "cost_bps": cost,
                                 "weeks": len(net), "ic_mean": ic.mean(), "ic_nw_t": stats.newey_west_t(ic),
                                 "rank_autocorr_1w": autocorr, "sharpe": stats.sharpe(net),
                                 "ann_gross": port.ret_gross.mean() * stats.PER_YEAR,
                                 "ann_net": net.mean() * stats.PER_YEAR,
                                 "annual_turnover": port.turnover.mean() * stats.PER_YEAR})
                weekly.append(pd.DataFrame({"date": sig_dates[port.index], "dataset": name, "signal": signal,
                                            "horizon": h, "gross": port.ret_gross.to_numpy(),
                                            "turnover": port.turnover.to_numpy(),
                                            "net": engine.net_returns(port, BASE_COST).to_numpy()}))
        dates = sig_dates[ports[MODELS[0], 1].index]
        spans = {"full": np.ones(len(dates), bool), "part2_test": np.asarray(dates >= SPLIT[name])}
        for span, keep in spans.items():
            for block in BLOCKS:
                idx = bootstrap_indices(int(keep.sum()), block)
                for cost in (COSTS if (block == MAIN_BLOCK and span == "full") else (BASE_COST,)):
                    net = {k: engine.net_returns(p, cost).to_numpy()[keep] for k, p in ports.items()}
                    boot = {k: resampled_sharpe(v, idx) for k, v in net.items()}
                    point = {k: stats.sharpe(pd.Series(v)) for k, v in net.items()}
                    for h in HORIZONS[1:]:
                        for group, members in (("mean", MODELS),) + tuple((m, (m,)) for m in MODELS):
                            delta = np.mean([boot[m, h] - boot[m, 1] for m in members], axis=0)
                            pooled.append({"dataset": name, "span": span, "cost_bps": cost, "block": block,
                                           "horizon": h, "model": group,
                                           "sharpe_difference": np.mean([point[m, h] - point[m, 1] for m in members]),
                                           "ci_low": np.quantile(delta, 0.025), "ci_high": np.quantile(delta, 0.975)})
    pd.DataFrame(rows).to_csv(OUT / "horizon_results.csv", index=False)
    pd.concat(weekly, ignore_index=True).to_csv(OUT / "horizon_weekly.csv", index=False)
    pd.DataFrame(pooled).to_csv(OUT / "horizon_pooled.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("fit", "summarize"))
    ap.add_argument("--dataset", choices=("etf", "french"))
    ap.add_argument("--signal", choices=MODELS)
    ap.add_argument("--horizon", type=int)
    a = ap.parse_args()
    fit(a.dataset, a.signal, a.horizon) if a.command == "fit" else summarize()
