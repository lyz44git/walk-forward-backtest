"""Overview figure at the top of the README: the baseline through time, and every change tested against it.

Usage: python figures/make_overview.py   (writes figures/overview.png)

Panel a: rolling five-year net Sharpe of the Part 1 long-short portfolios (reports/ls_weekly.csv).
Panel b: change in net Sharpe at 5 bps from each modification in Parts 2-4, with the paired 95% bootstrap
         interval (block 8) from rules_pooled.csv, horizon_pooled.csv and nonprice/pooled.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INK, AXIS, GREY, SHADE = "#111111", "#333333", "#6e6e6e", "#f8f5f1"
SIG = {"mom_12_1": ("12-1 momentum", "#009E73"), "lgbm": ("LightGBM", "#0072B2"), "ridge": ("Ridge", "#D55E00")}
DS = {"etf": ("30 ETFs", "#56B4E9"), "french": ("49 industries", "#0072B2")}
PER_YEAR, WINDOW = 52, 260
RULES = {"partial_adjustment": "Partial adjustment", "no_trade_band": "No-trade band", "rank_buffer": "Rank buffer",
         "signal_smoothing": "Signal smoothing", "ppo": "PPO policy"}


def rolling_sharpe(net):
    roll = net.rolling(WINDOW)
    return np.sqrt(PER_YEAR) * roll.mean() / roll.std(ddof=1)


def rows():
    """(group, label, {dataset: (point, low, high)}) for every modification, in the order of the README."""
    rep = ROOT / "reports"
    r2 = pd.read_csv(rep / "rebalancing" / "rules_pooled.csv")
    r2 = r2[(r2.cost_bps == 5.0) & (r2.block == 8) & (r2.versus == "full")]
    r3 = pd.read_csv(rep / "horizon" / "horizon_pooled.csv")
    r3 = r3[(r3.span == "full") & (r3.cost_bps == 5.0) & (r3.block == 8) & (r3.model == "mean")]
    r4 = pd.read_csv(rep / "nonprice" / "pooled.csv")
    r4 = r4[(r4.span == "full") & (r4.cost_bps == 5.0) & (r4.block == 8) & (r4.model == "mean") & (r4.versus == "P")]

    def pick(frame, mask, col):
        out = {}
        for ds in DS:
            r = frame[mask & (frame.dataset == ds)]
            assert len(r) == 1, (ds, len(r))
            out[ds] = (r[col].iloc[0], r.ci_low.iloc[0], r.ci_high.iloc[0])
        return out

    out = [("Part 2: rebalancing rules", lab, pick(r2, r2.rule == key, "mean_sharpe_difference"))
           for key, lab in RULES.items()]
    out += [("Part 3: longer label horizons", f"{h}-week label", pick(r3, r3.horizon == h, "sharpe_difference"))
            for h in (4, 13)]
    out += [("Part 4: non-price features", lab, pick(r4, r4.features == key, "sharpe_difference"))
            for key, lab in (("PN", "Price + non-price"), ("N", "Non-price only"))]
    return out


def main():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica Neue", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "savefig.facecolor": "white"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.62})

    ls = pd.read_csv(ROOT / "reports" / "ls_weekly.csv", parse_dates=["date"])
    ls = ls[ls.dataset == "french"]
    ends = {}
    for key, (lab, col) in SIG.items():
        s = ls[ls.signal == key].sort_values("date").set_index("date").net
        r = rolling_sharpe(s).dropna()
        a.plot(r.index, r.values, color=col, lw=0.75, solid_capstyle="round")
        ends[key] = (r.index[-1], r.iloc[-1])
    a.plot([pd.Timestamp("1979-01-01"), pd.Timestamp("2027-01-01")], [0, 0], color=AXIS, lw=0.5)
    a.fill_between([pd.Timestamp("1979-01-01"), pd.Timestamp("2027-01-01")], -1.2, 0, color=SHADE, lw=0, zorder=0)
    a.set_ylim(-1.2, 2.2)
    order = sorted(ends, key=lambda k: ends[k][1])
    ys = [ends[k][1] for k in order]
    for i in range(1, len(ys)):                                   # keep end labels at least 0.2 apart
        ys[i] = max(ys[i], ys[i - 1] + 0.2)
    for k, y in zip(order, ys):
        a.annotate(SIG[k][0], xy=ends[k], xytext=(ends[k][0] + pd.Timedelta(days=500), y), va="center",
                   fontsize=6.5, arrowprops={"arrowstyle": "-", "lw": 0.4, "color": GREY, "shrinkA": 0, "shrinkB": 1})
    a.set_xlim(pd.Timestamp("1979-01-01"), pd.Timestamp("2040-01-01"))
    a.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(1980, 2021, 10)])
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a.spines["bottom"].set_bounds(mdates.date2num(pd.Timestamp("1979-01-01")), mdates.date2num(pd.Timestamp("2027-01-01")))
    a.set_ylabel("Net Sharpe over the previous five years")
    a.text(-0.14, 1.03, "a", transform=a.transAxes, fontsize=8, fontweight="bold", va="baseline")
    a.text(-0.09, 1.03, "49 industry portfolios: the baseline after costs", transform=a.transAxes, fontsize=7.5,
           va="baseline", ha="left")

    data, y, ticks, labels, group_y = rows(), 0.0, [], [], []
    last = None
    for group, lab, vals in data:
        if group != last:
            y += 0.9 if last else 0.0
            group_y.append((y, group)); y += 1.0; last = group
        for off, ds in ((-0.17, "etf"), (0.17, "french")):
            p, lo, hi = vals[ds]
            col = DS[ds][1]
            b.plot([lo, hi], [y + off, y + off], color=col, lw=1.3, solid_capstyle="round")
            b.plot(p, y + off, "o", ms=4, color=col, mfc="white" if ds == "etf" else col, mew=1.0)
        ticks.append(y); labels.append(lab); y += 1.0
    b.axvspan(-0.8, 0, color=SHADE, lw=0, zorder=0)
    b.axvline(0, color=AXIS, lw=0.5)
    for gy, group in group_y:
        b.text(-0.8, gy, group, fontsize=6.3, color=GREY, va="center", ha="left")
    b.set_yticks(ticks); b.set_yticklabels(labels)
    b.tick_params(axis="y", length=0); b.spines["left"].set_visible(False)
    b.set_ylim(y + 1.3, -0.7); b.set_xlim(-0.8, 0.5)
    b.set_xlabel("Change in net Sharpe at 5 bps, 95% interval")
    for ds, (lab, col) in DS.items():
        b.plot([], [], "o", ms=4, color=col, mfc="white" if ds == "etf" else col, mew=1.0, label=lab)
    b.legend(frameon=False, fontsize=6.3, loc="lower right", handletextpad=0.2, borderaxespad=0.3)
    b.text(-0.42, 1.03, "b", transform=b.transAxes, fontsize=8, fontweight="bold", va="baseline")
    b.text(-0.37, 1.03, "Every change tested, against its own baseline", transform=b.transAxes, fontsize=7.5,
           va="baseline", ha="left")

    fig.savefig(ROOT / "figures" / "overview.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
