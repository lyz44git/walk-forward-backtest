"""What limits the Part 1 signals: trading cost and breadth. Descriptive; uses existing results, no refitting.

Added 2026-11-17 after every result of Parts 1-4 was known. Nothing here is a pre-specified test.

Inputs:  reports/cost_grid.csv, reports/main_results.csv, tmp/rebalance/case_*.npz
Outputs: reports/limits/cost_crossover.csv   one-way cost at which each model's net Sharpe equals momentum's
         reports/limits/breadth.csv          Sharpe before costs on random subsets of the assets
         reports/limits/breadth_slope.csv    log-log slope of that Sharpe on the number of assets
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmb import stats  # noqa: E402
from qmb.rebalance import cases  # noqa: E402

OUT = ROOT / "reports" / "limits"
SIZES = {"french": (8, 12, 16, 24, 32, 40, 49), "etf": (8, 12, 16, 20, 24, 30)}
DRAWS, QUANTILE = 200, 0.2


def crossover(grid):
    """Linear interpolation, between adjacent grid costs, of the cost at which model minus momentum is zero."""
    rows = []
    for ds, g in grid.groupby("dataset"):
        wide = g.pivot(index="cost_bps", columns="signal", values="net_sharpe").sort_index()
        for model in ("ridge", "lgbm"):
            diff = wide[model] - wide["mom_12_1"]
            cost = np.nan
            for (c0, d0), (c1, d1) in zip(list(diff.items())[:-1], list(diff.items())[1:]):
                if d0 > 0 >= d1:
                    cost = c0 + (c1 - c0) * d0 / (d0 - d1)
                    break
            rows.append({"dataset": ds, "signal": model, "difference_at_zero_cost": diff.iloc[0],
                         "difference_at_5bps": diff.loc[5.0], "crossover_cost_bps": cost})
    return pd.DataFrame(rows)


def gross_sharpe(S, R, M):
    """Sharpe ratio before costs of the equal-weight top-minus-bottom 20% portfolio, formed each week."""
    score = np.where(M, S, np.nan)
    rank = pd.DataFrame(score).rank(axis=1).to_numpy()
    count = M.sum(axis=1)[:, None]
    k = np.maximum(1, np.rint(QUANTILE * count))
    ret = np.where(M, R, np.nan)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        spread = np.nanmean(np.where(rank > count - k, ret, np.nan), axis=1) \
            - np.nanmean(np.where(rank <= k, ret, np.nan), axis=1)
    return stats.sharpe(pd.Series(spread[np.isfinite(spread)]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    crossover(pd.read_csv(ROOT / "reports" / "cost_grid.csv")).to_csv(OUT / "cost_crossover.csv", index=False)

    main_results = pd.read_csv(ROOT / "reports" / "main_results.csv").set_index(["dataset", "signal"])
    rng = np.random.default_rng(0)
    rows, slopes = [], []
    for path in sorted(cases.TMP.glob("case_*.npz")):
        case = cases.load_case(path)
        ds, sig = case["dataset"], case["signal"]
        S, R, M = case["S"], case["R"], case["M"]
        n_all = S.shape[1]
        full = gross_sharpe(S, R, M)
        assert abs(full - main_results.loc[(ds, sig), "ls_gross_sharpe"]) < 5e-3, (ds, sig, full)
        med = {}
        for n in SIZES[ds]:
            if n == n_all:
                draws = np.array([full])
            else:
                draws = np.array([gross_sharpe(S[:, c], R[:, c], M[:, c])
                                  for c in (rng.choice(n_all, n, replace=False) for _ in range(DRAWS))])
            med[n] = np.median(draws)
            rows.append({"dataset": ds, "signal": sig, "assets": n, "subsets": len(draws), "median_sharpe": med[n],
                         "p10": np.quantile(draws, 0.1), "p90": np.quantile(draws, 0.9)})
        slope = np.polyfit(np.log(list(med)), np.log(list(med.values())), 1)[0] if min(med.values()) > 0 else np.nan
        slopes.append({"dataset": ds, "signal": sig, "loglog_slope": slope})
        print(f"done {ds} {sig}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "breadth.csv", index=False)
    pd.DataFrame(slopes).to_csv(OUT / "breadth_slope.csv", index=False)


if __name__ == "__main__":
    main()
