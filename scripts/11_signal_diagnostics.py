"""Part 3, step A: diagnostics of the Part 1 predictions, with no refitting (design/label_horizon_plan.md).

Inputs:  tmp/rebalance/case_*.npz, reports/ls_weekly.csv, reports/rebalancing/rules_results.csv, data/clean.
Outputs (reports/diagnostics/): ic_decay.csv, rank_autocorrelation.csv, quintile_returns.csv,
         breakeven_cost.csv, factor_attribution.csv
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmb import data as qdata, stats  # noqa: E402
from qmb.rebalance import cases  # noqa: E402

OUT = ROOT / "reports" / "diagnostics"
LAGS, AUTO_LAGS = range(13), (1, 2, 4, 8, 13)
BORROW_BPS, BASE_COST = 50.0, 5.0


def rank_corr(a, b, valid):
    """Spearman correlation across the assets valid in both rows; NaN if fewer than three."""
    if valid.sum() < 3:
        return np.nan
    return sps.spearmanr(a[valid], b[valid]).statistic


def decay(case):
    S, R, M = case["S"], case["R"], case["M"]
    T = len(S)
    rows = []
    for j in LAGS:
        ic = np.array([rank_corr(S[t], R[t + j], M[t] & M[t + j]) for t in range(T - j)])
        ic = ic[~np.isnan(ic)]
        rows.append({"lag_weeks": j, "ic_mean": ic.mean(), "ic_nw_t": stats.newey_west_t(ic), "weeks": len(ic)})
    return rows


def autocorrelation(case):
    S, M = case["S"], case["M"]
    T = len(S)
    return [{"lag_weeks": j,
             "rank_autocorr": np.nanmean([rank_corr(S[t], S[t + j], M[t] & M[t + j]) for t in range(T - j)])}
            for j in AUTO_LAGS]


def quintiles(case):
    """Mean one-week return of each prediction quintile (1 = lowest prediction), equal weight."""
    S, R, M = case["S"], case["R"], case["M"]
    out = np.full((len(S), 5), np.nan)
    for t in range(len(S)):
        idx = np.flatnonzero(M[t])
        order = idx[np.argsort(S[t, idx], kind="mergesort")]
        for q, part in enumerate(np.array_split(order, 5)):
            out[t, q] = R[t, part].mean()
    mean = np.nanmean(out, axis=0) * stats.PER_YEAR
    spread = out[:, 4] - out[:, 0]
    return {**{f"q{q + 1}_ann_return": mean[q] for q in range(5)},
            "ordered_adjacent_pairs": int((np.diff(mean) > 0).sum()),
            "q5_minus_q1_ann": np.nanmean(spread) * stats.PER_YEAR, "q5_minus_q1_nw_t": stats.newey_west_t(spread)}


def weekly_factors(case):
    """Market, size and value factor returns compounded over each holding period.

    Industries trade close to close: trading days in (entry, exit]. ETFs trade open to open: trading days in
    [entry, exit), i.e. the close-to-close days nearest the open-to-open window. Days after the last factor
    date are dropped."""
    ff = pd.read_parquet(qdata.CLEAN / "ff_daily.parquet")[["mkt_rf", "smb", "hml"]]
    g = np.log1p(ff).cumsum()
    entry, exit_ = pd.DatetimeIndex(case["entry"]), pd.DatetimeIndex(case["exit"])
    if case["dataset"] == "etf":
        prev = g.shift(1)
        a, b = prev.reindex(entry).to_numpy(), prev.reindex(exit_).to_numpy()
    else:
        a, b = g.reindex(entry).to_numpy(), g.reindex(exit_).to_numpy()
    return pd.DataFrame(np.expm1(b - a), columns=ff.columns)


def ols_nw(y, X, names, lags=4):
    """OLS with Newey-West (Bartlett) standard errors; the first column of X is the constant."""
    ok = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    y, X = y[ok], X[ok]
    n = len(y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for L in range(1, lags + 1):
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None]) / n
        S += (1 - L / (lags + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    se = np.sqrt(np.diag(Q @ S @ Q / n))
    out = {"weeks": n, "alpha_annual": beta[0] * stats.PER_YEAR, "alpha_t": beta[0] / se[0],
           "r_squared": 1 - e.var() / y.var()}
    for name, b, s in zip(names, beta[1:], se[1:]):
        out[f"beta_{name}"], out[f"t_{name}"] = b, b / s
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ls = pd.read_csv(ROOT / "reports" / "ls_weekly.csv", parse_dates=["date"])
    dec, auto, quint, brk, attr = [], [], [], [], []
    for path in sorted(cases.TMP.glob("case_*.npz")):
        case = cases.load_case(path)
        key = {"dataset": case["dataset"], "signal": case["signal"]}
        dec += [dict(key, **r) for r in decay(case)]
        auto += [dict(key, **r) for r in autocorrelation(case)]
        quint.append(dict(key, **quintiles(case)))

        own = ls[(ls.dataset == key["dataset"]) & (ls.signal == key["signal"])].sort_values("date")
        gross, turnover = own.gross.mean(), own.turnover.mean()
        borrow = BORROW_BPS / 1e4 / stats.PER_YEAR                      # short notional is 1 under full rebalancing
        brk.append(dict(key, span="part1_full_sample", rule="full", annual_turnover=turnover * stats.PER_YEAR,
                        ann_gross=gross * stats.PER_YEAR, breakeven_cost_bps=(gross - borrow) / turnover * 1e4))

        factors = weekly_factors(case)
        y = own.net.to_numpy()
        base = np.column_stack([np.ones(len(y)), factors.to_numpy()])
        attr.append(dict(key, model="market, size, value", **ols_nw(y, base, list(factors.columns))))
        if key["signal"] != "mom_12_1":
            mom = ls[(ls.dataset == key["dataset"]) & (ls.signal == "mom_12_1")].sort_values("date").net.to_numpy()
            attr.append(dict(key, model="market, size, value, same-universe momentum",
                             **ols_nw(y, np.column_stack([base, mom]), list(factors.columns) + ["momentum"])))
        print(f"done {key['dataset']} {key['signal']}", flush=True)

    rules = pd.read_csv(ROOT / "reports" / "rebalancing" / "rules_results.csv")
    for _, r in rules[rules.cost_bps == BASE_COST].iterrows():
        borrow = r.ann_cost - r.annual_turnover * BASE_COST / 1e4
        brk.append({"dataset": r.dataset, "signal": r.signal, "span": "part2_test", "rule": r.rule,
                    "annual_turnover": r.annual_turnover, "ann_gross": r.ann_gross,
                    "breakeven_cost_bps": (r.ann_gross - borrow) / r.annual_turnover * 1e4})

    pd.DataFrame(dec).to_csv(OUT / "ic_decay.csv", index=False)
    pd.DataFrame(auto).to_csv(OUT / "rank_autocorrelation.csv", index=False)
    pd.DataFrame(quint).to_csv(OUT / "quintile_returns.csv", index=False)
    pd.DataFrame(brk).to_csv(OUT / "breakeven_cost.csv", index=False)
    pd.DataFrame(attr).to_csv(OUT / "factor_attribution.csv", index=False)


if __name__ == "__main__":
    main()
