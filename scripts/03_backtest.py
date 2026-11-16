"""Main walk-forward backtests and pre-registered robustness checks on both datasets.

Outputs (reports/): main_results.csv, robustness.csv, halves.csv, cost_grid.csv, ic_weekly.csv, ls_weekly.csv,
run_meta.json. Design: design/preregistration.json.
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from qmb import engine, stats, data

OUT = ROOT / "reports"
SIGNALS = ["mom_12_1", "rev_1w", "ridge", "lgbm"]
MAIN_COST, BORROW, COST_GRID = 5.0, 50.0, [0.0, 2.0, 5.0, 10.0]
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)


def load(name):
    if name == "etf":
        P = pd.read_parquet(data.CLEAN / "etf_daily.parquet")
        P = P[P.date >= "2007-04-11"]
        piv = lambda c: P.pivot(index="date", columns="ticker", values=c).sort_index()
        return {"tri": piv("close_adj"), "open": piv("open_adj"), "dv": piv("dollar_volume"), "initial": 156}
    I = pd.read_parquet(data.CLEAN / "french_vw_daily.parquet").loc["1969-07-01":]
    tri = (1 + I).cumprod()
    tri = tri.where(I.notna())                       # a missing return leaves that day's level missing
    return {"tri": tri, "open": None, "dv": None, "initial": 260}


RF_DAILY = pd.read_parquet(data.CLEAN / "ff_daily.parquet")["rf"]


def weekly_rf(sched, calendar):
    """Risk-free return compounded over each holding week's trading days (entry, exit]. Days after the last
    Fama-French date (2026-09-30) carry the last available daily rate forward."""
    rf = RF_DAILY.reindex(calendar).ffill()
    g = np.log1p(rf).cumsum()
    a = g.reindex(pd.DatetimeIndex(sched["entry"])).to_numpy(); b = g.reindex(pd.DatetimeIndex(sched["exit"])).to_numpy()
    return pd.Series(np.expm1(b - a), index=range(len(sched)))


def run(d, execution="default", lag=0, placebo=False, signals=SIGNALS):
    tri = d["tri"]
    exec_px = d["open"] if (execution == "default" and d["open"] is not None) else tri
    sch = engine.weekly_schedule(tri.index)
    feats = engine.daily_features(tri, d["dv"])
    panel = engine.build_panel(tri, exec_px, feats, sch, feature_lag_weeks=lag)
    preds = {s: engine.walk_forward(panel, s, d["initial"], placebo=placebo) for s in signals}
    panel.rf_week = weekly_rf(sch, tri.index)
    return panel, preds


def evaluate(panel, preds, cost=MAIN_COST, trials=None):
    rows, ic_series, ls_series = [], {}, {}
    for s, p in preds.items():
        ic = stats.weekly_ic(p, panel.ret)
        ls = engine.portfolio(p, panel.ret, 0.2, "long_short")
        lo = engine.portfolio(p, panel.ret, 0.2, "long_only")
        ew = engine.portfolio(p, panel.ret, 0.2, "equal_weight")
        net = engine.net_returns(ls, cost, BORROW)
        lo_net, ew_net = engine.net_returns(lo, cost, 0.0), engine.net_returns(ew, cost, 0.0)
        ci = stats.stationary_bootstrap_sharpe(net)
        ab = stats.alpha_beta(net, ew_net - panel.rf_week.reindex(ew_net.index))   # LS is already an excess return
        sg = stats.summarize(ls.ret_gross); sn = stats.summarize(net, ls.turnover)
        rows.append({"signal": s, "weeks": int(len(ic)), "assets_median": float(ls.n_assets.median()), "leg_size_median": float(ls.n_leg.median()),
                     "ic_mean": float(ic.mean()), "ic_ir_annual": float(ic.mean() / ic.std() * np.sqrt(52)),
                     "ic_nw_t": stats.newey_west_t(ic, 4), "ic_hit": float((ic > 0).mean()),
                     "ls_gross_ann_return": sg["ann_return"], "ls_gross_sharpe": sg["sharpe"],
                     "ls_net_ann_return": sn["ann_return"], "ls_net_ann_vol": sn["ann_vol"], "ls_net_sharpe": sn["sharpe"],
                     "ls_net_sharpe_ci_lo": ci[0], "ls_net_sharpe_ci_hi": ci[1], "ls_net_max_drawdown": sn["max_drawdown"],
                     "ls_annual_turnover": sn["annual_turnover"],
                     "ls_alpha_vs_ew_annual": ab["alpha_annual"], "ls_alpha_t": ab["alpha_t"], "ls_beta_vs_ew": ab["beta"],
                     "lo_net_ann_return": float(lo_net.mean() * 52), "ew_net_ann_return": float(ew_net.mean() * 52),
                     "lo_minus_ew_ann": float((lo_net - ew_net).mean() * 52),
                     "lo_minus_ew_sharpe": stats.sharpe(lo_net - ew_net)})
        ic_series[s], ls_series[s] = ic, net
        ls_series[s + "|turnover"] = ls.turnover; ls_series[s + "|gross"] = ls.ret_gross; ls_series[s + "|short"] = ls.short_notional
    R = pd.DataFrame(rows)
    trials = R.ls_net_sharpe.tolist() if trials is None else trials
    R["deflated_sharpe_prob"] = [stats.deflated_sharpe(ls_series[s], trials) for s in R.signal]
    return R, ic_series, ls_series


def week_dates(panel, idx):
    return pd.DatetimeIndex(panel.sched.signal.iloc[np.asarray(idx)])


main, rob, halves, grid, ic_all, ls_all, meta = [], [], [], [], [], [], {}
for name in ["etf", "french"]:
    d = load(name)
    log(name, "loaded", d["tri"].shape)
    panel, preds = run(d)
    R, ics, lss = evaluate(panel, preds)
    R.insert(0, "dataset", name); main.append(R)
    oos_weeks = preds["ridge"].index.get_level_values("week").unique()
    meta[name] = {"panel_rows": int(len(panel.X)), "features": panel.features,
                  "first_signal_oos": str(week_dates(panel, [oos_weeks.min()])[0].date()),
                  "last_signal_oos": str(week_dates(panel, [oos_weeks.max()])[0].date()),
                  "oos_weeks": int(len(oos_weeks)), "first_feature_week": str(week_dates(panel, [panel.X.index.get_level_values('week').min()])[0].date())}
    for s in SIGNALS:
        ic = ics[s]; ic_all.append(pd.DataFrame({"dataset": name, "signal": s, "date": week_dates(panel, ic.index), "ic": ic.values}))
        ls_all.append(pd.DataFrame({"dataset": name, "signal": s, "date": week_dates(panel, lss[s].index), "net": lss[s].values,
                                    "gross": lss[s + "|gross"].values, "turnover": lss[s + "|turnover"].values}))
        # cost grid
        for c in COST_GRID:
            net_c = lss[s + "|gross"] - lss[s + "|turnover"] * c / 1e4 - lss[s + "|short"] * BORROW / 1e4 / 52
            grid.append({"dataset": name, "signal": s, "cost_bps": c, "net_sharpe": stats.sharpe(net_c), "net_ann_return": float(net_c.mean() * 52)})
        # halves of the OOS period
        wk = np.array(sorted(ic.index)); mid = wk[len(wk) // 2]
        for lab, m in [("first half", ic.index < mid), ("second half", ic.index >= mid)]:
            net = lss[s][(lss[s].index < mid) if lab == "first half" else (lss[s].index >= mid)]
            dd = week_dates(panel, ic.index[m])
            halves.append({"dataset": name, "signal": s, "half": lab, "from": str(dd.min().date()), "to": str(dd.max().date()),
                           "ic_mean": float(ic[m].mean()), "ic_nw_t": stats.newey_west_t(ic[m], 4), "net_sharpe": stats.sharpe(net)})
    log(name, "main done")
    # robustness: placebo, extra lag, next-close execution (ETF)
    variants = [("features lagged 1 extra week", dict(lag=1), SIGNALS)]      # placebo: scripts/04_placebo.py
    if name == "etf":
        variants.append(("enter at next close", dict(execution="close"), SIGNALS))
    for lab, kw, sigs in variants:
        p2, pr2 = run(d, signals=sigs, **kw)
        R2, _, _ = evaluate(p2, pr2, trials=R.ls_net_sharpe.tolist())
        R2.insert(0, "variant", lab); R2.insert(0, "dataset", name); rob.append(R2)
        log(name, lab, "done")

M_ = pd.concat(main)
M_["passes_dataset"] = (M_.ic_mean > 0) & (M_.ic_nw_t > 2) & (M_.ls_net_sharpe_ci_lo > 0)
dec = M_.pivot(index="signal", columns="dataset", values="passes_dataset")
dec["passes_preregistered_rule"] = dec.all(axis=1)
dec.to_csv(OUT / "decision.csv")
M_.to_csv(OUT / "main_results.csv", index=False)
pd.concat(rob).to_csv(OUT / "robustness.csv", index=False)
pd.DataFrame(halves).to_csv(OUT / "halves.csv", index=False)
pd.DataFrame(grid).to_csv(OUT / "cost_grid.csv", index=False)
pd.concat(ic_all).to_csv(OUT / "ic_weekly.csv", index=False)
pd.concat(ls_all).to_csv(OUT / "ls_weekly.csv", index=False)
json.dump(meta, open(OUT / "run_meta.json", "w"), indent=1)
pd.set_option("display.width", 250)
M = pd.concat(main)
print(M[["dataset", "signal", "weeks", "ic_mean", "ic_nw_t", "ic_hit", "ls_gross_sharpe", "ls_net_sharpe", "ls_net_sharpe_ci_lo",
         "ls_net_sharpe_ci_hi", "ls_net_ann_return", "ls_annual_turnover", "ls_beta_vs_ew", "deflated_sharpe_prob", "lo_minus_ew_ann"]].round(3).to_string(index=False))
print(pd.concat(rob)[["dataset", "variant", "signal", "ic_mean", "ic_nw_t", "ls_net_sharpe"]].round(3).to_string(index=False))
print(dec.to_string()); print(json.dumps(meta, indent=1)); log("all done")
