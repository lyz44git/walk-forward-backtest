"""Parity and timing of the C++ kernels (native/qmb_kernels.cpp) against the Python reference.

Parity: the Sharpe-ratio confidence interval of every Part 1 series, recomputed in Python and in C++ from
reports/ls_weekly.csv and compared with reports/main_results.csv; and the weekly accounting of full
rebalancing, partial adjustment and the no-trade band on the cached test spans of Part 2 (synthetic
markets when tmp/rebalance is absent).
Timing: median wall-clock time of each implementation on this machine; the numbers depend on the machine.

Writes: reports/native/parity.csv, reports/native/timing.csv
"""
from pathlib import Path
from types import SimpleNamespace
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmb import native, stats  # noqa: E402
from qmb.rebalance import cases, env, rules  # noqa: E402
from qmb.rebalance.splits import split_indices  # noqa: E402

OUT = ROOT / "reports" / "native"
COSTS = (0.0, 2.0, 5.0, 10.0)
FIELDS = ("net", "gross", "turnover", "short")


def median_seconds(function, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def markets():
    """(label, market, first test week, end) for each cached case, or two synthetic markets."""
    paths = sorted(cases.TMP.glob("case_*.npz"))
    if paths:
        for path in paths:
            case = cases.load_case(path)
            _, _, t0, t1 = split_indices(case)
            yield f"{case['dataset']}_{case['signal']}", SimpleNamespace(W=case["W"], R=case["R"], M=case["M"]), t0, t1
        return
    for seed, (T, N) in enumerate(((810, 30), (2672, 49))):
        rng = np.random.default_rng(seed)
        S = rng.standard_normal((T, N)); M = np.ones((T, N), bool)
        R = 0.02 * rng.standard_normal((T, N))
        yield f"synthetic_{T}x{N}", SimpleNamespace(W=rules.quintile_targets(S, M), R=R, M=M), 1, T


def rule_runs(market, t0, t1):
    names = np.maximum(1, np.rint(rules.QUANTILE * market.M.sum(axis=1)))
    yield "full", {"eta_const": 1.0}
    yield "partial_adjustment_0.5", {"eta_const": 0.5}
    yield "no_trade_band_0.5", {"band": 0.5 / names[t0:t1]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    native.load()
    weekly = pd.read_csv(ROOT / "reports" / "ls_weekly.csv")
    published = pd.read_csv(ROOT / "reports" / "main_results.csv").set_index(["dataset", "signal"])

    parity = []
    for (dataset, signal), group in weekly.groupby(["dataset", "signal"]):
        python = stats.stationary_bootstrap_sharpe(group["net"])
        cpp = native.stationary_bootstrap_sharpe(group["net"])
        row = published.loc[(dataset, signal)]
        parity.append({
            "kernel": "stationary_bootstrap_sharpe", "case": f"{dataset}_{signal}", "weeks": len(group),
            "values_compared": 2,
            "max_abs_diff_vs_python": float(np.abs(np.subtract(cpp, python)).max()),
            "max_abs_diff_vs_published": float(max(abs(cpp[0] - row.ls_net_sharpe_ci_lo),
                                                   abs(cpp[1] - row.ls_net_sharpe_ci_hi)))})
        print(f"bootstrap {dataset} {signal}: C++ {cpp[0]:+.4f} {cpp[1]:+.4f}", flush=True)

    for label, market, t0, t1 in markets():
        for rule, arguments in rule_runs(market, t0, t1):
            worst, compared = 0.0, 0
            for cost in COSTS:
                python = env.simulate(market, t0, t1, cost_bps=cost, **arguments)
                cpp = native.simulate(market.W, market.R, market.M, t0, t1, cost_bps=cost, **arguments)
                for field in FIELDS:
                    worst = max(worst, float(np.abs(cpp[field] - python[field]).max()))
                    compared += len(python[field])
            parity.append({"kernel": f"simulate_{rule}", "case": label, "weeks": t1 - t0,
                           "values_compared": compared, "max_abs_diff_vs_python": worst,
                           "max_abs_diff_vs_published": np.nan})
    parity = pd.DataFrame(parity)
    parity.to_csv(OUT / "parity.csv", index=False)

    longest = max((group["net"] for _, group in weekly.groupby(["dataset", "signal"])), key=len)
    label, market, t0, t1 = max(markets(), key=lambda item: item[3] - item[2])
    grid = np.round(np.arange(20, 0, -1) / 20, 2)                      # the partial-adjustment grid of Part 2
    tasks = [
        (f"Sharpe CI, {len(longest)} weeks x 2000 draws",
         lambda: stats.stationary_bootstrap_sharpe(longest), lambda: native.stationary_bootstrap_sharpe(longest)),
        (f"Index paths, {len(longest)} weeks x 2000 draws",
         lambda: native_reference_indices(len(longest)), lambda: native.bootstrap_indices(len(longest), 8)),
        (f"Partial-adjustment grid, 20 values x {t1 - t0} weeks x {market.W.shape[1]} assets ({label})",
         lambda: [env.simulate(market, t0, t1, eta_const=e) for e in grid],
         lambda: [native.simulate(market.W, market.R, market.M, t0, t1, eta_const=e) for e in grid]),
    ]
    compiler = subprocess.run([native._builder().compiler(), "--version"], capture_output=True, text=True).stdout
    timing = []
    for name, python, cpp in tasks:
        seconds_python, seconds_cpp = median_seconds(python, 3), median_seconds(cpp, 9)
        timing.append({"task": name, "python_seconds": seconds_python, "cpp_seconds": seconds_cpp,
                       "speedup": seconds_python / seconds_cpp, "machine": platform.machine(),
                       "system": platform.system(), "compiler": compiler.splitlines()[0],
                       "python": platform.python_version(), "numpy": np.__version__})
        print(f"{name}: Python {seconds_python:.3f}s, C++ {seconds_cpp:.4f}s", flush=True)
    pd.DataFrame(timing).to_csv(OUT / "timing.csv", index=False)
    print(parity.groupby("kernel")[["max_abs_diff_vs_python", "max_abs_diff_vs_published"]].max())


def native_reference_indices(n, block=8, draws=2000):
    """The interpreted index walk of scripts/09_rebalancing_rules.py."""
    rng = np.random.default_rng(0)
    idx = np.empty((draws, n), dtype=int)
    for path in idx:
        path[0] = rng.integers(n)
        restart = rng.random(n) < 1.0 / block
        jump = rng.integers(n, size=n)
        for i in range(1, n):
            path[i] = jump[i] if restart[i] else (path[i - 1] + 1) % n
    return idx


if __name__ == "__main__":
    main()
