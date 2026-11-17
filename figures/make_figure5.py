"""Figure 5 of the README: the Part 1 long-short portfolios year by year, and their drawdowns.

Usage: python figures/make_figure5.py   (writes figures/figure5.png)

Input: reports/ls_weekly.csv (weekly net returns at 5 bps). A year's return is the compounded weekly net return
of the weeks whose signal date falls in that calendar year; the first and last years of each dataset are partial.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INK, AXIS, GREY = "#111111", "#333333", "#6e6e6e"
SIG = {"mom_12_1": ("12-1 momentum", "#009E73"), "ridge": ("Ridge", "#D55E00"), "lgbm": ("LightGBM", "#0072B2"),
       "rev_1w": ("1-week reversal", "#999999")}
DS = {"french": "49 industries", "etf": "30 ETFs"}
CMAP = LinearSegmentedColormap.from_list("div", ["#B2182B", "#EF8A62", "#ECECEC", "#67A9CF", "#2166AC"])
LIMIT = 30.0                                                    # colour scale saturates at +/-30% a year
X0, X1 = 1975, 2027


def yearly(ls, dataset):
    """Signals x calendar years table of compounded net return, in percent."""
    d = ls[ls.dataset == dataset]
    out = d.groupby(["signal", d.date.dt.year]).net.apply(lambda r: (1 + r).prod() - 1).unstack() * 100
    return out.loc[list(SIG)]


def drawdown(net):
    wealth = (1 + net).cumprod()
    return (wealth / wealth.cummax() - 1) * 100


def main():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica Neue", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "savefig.facecolor": "white"})
    ls = pd.read_csv(ROOT / "reports" / "ls_weekly.csv", parse_dates=["date"])
    fig = plt.figure(figsize=(7.2, 4.6))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2.3], width_ratios=[1, 0.012], hspace=0.5, wspace=0.02)
    norm = TwoSlopeNorm(vmin=-LIMIT, vcenter=0, vmax=LIMIT)

    for row, (ds, title) in enumerate(DS.items()):
        ax = fig.add_subplot(grid[row, 0])
        table = yearly(ls, ds)
        years = table.columns.to_numpy()
        mesh = ax.pcolormesh(np.append(years, years[-1] + 1), np.arange(len(table) + 1), table.to_numpy(),
                             cmap=CMAP, norm=norm, edgecolors="white", linewidth=0.6)
        ax.set_xlim(X0, X1); ax.set_ylim(len(table), 0)
        ax.set_yticks(np.arange(len(table)) + 0.5); ax.set_yticklabels([SIG[k][0] for k in table.index])
        ax.tick_params(axis="y", length=0); ax.set_xticks(range(1980, 2021, 10)); ax.tick_params(axis="x", labelbottom=row == 1, length=0, pad=1)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        if row == 0:
            ax.text(-0.13, 1.08, "a", transform=ax.transAxes, fontsize=8, fontweight="bold", va="baseline")
        ax.text(-0.115, 1.08, f"{title}: net return of each long-short portfolio by calendar year (%)",
                transform=ax.transAxes, fontsize=7.5, va="baseline", ha="left")
    cax = fig.add_subplot(grid[0:2, 1])
    bar = fig.colorbar(mesh, cax=cax, extend="both", ticks=[-30, -15, 0, 15, 30])
    bar.outline.set_visible(False); bar.ax.tick_params(length=0, labelsize=6.3)
    bar.set_label("% a year", fontsize=6.5)

    ax = fig.add_subplot(grid[2, 0])
    french = ls[ls.dataset == "french"]
    ends = {}
    for key in ("mom_12_1", "ridge", "lgbm"):
        s = french[french.signal == key].sort_values("date")
        dd = drawdown(s.net.reset_index(drop=True))
        x = s.date.dt.year + (s.date.dt.dayofyear - 1) / 365.25
        ax.plot(x.to_numpy(), dd.to_numpy(), color=SIG[key][1], lw=0.75, solid_capstyle="round")
        ends[key] = (x.iloc[-1], dd.iloc[-1], dd.min())
    ax.axhline(0, color=AXIS, lw=0.5)
    ax.set_xlim(X0, X1); ax.set_ylim(-62, 2)
    ax.set_xticks(range(1980, 2021, 10))
    ax.set_ylabel("Drawdown from previous peak (%)", fontsize=6.5)
    for i, key in enumerate(("mom_12_1", "ridge", "lgbm")):
        ax.plot([], [], color=SIG[key][1], lw=1.2, label=f"{SIG[key][0]} (deepest \u2212{-ends[key][2]:.0f}%)")
    ax.legend(frameon=False, fontsize=6.3, loc="lower left", handlelength=1.4, borderaxespad=0.2)
    ax.text(-0.13, 1.06, "b", transform=ax.transAxes, fontsize=8, fontweight="bold", va="baseline")
    ax.text(-0.115, 1.06, "49 industries: drawdown of the net long-short portfolios", transform=ax.transAxes,
            fontsize=7.5, va="baseline", ha="left")

    fig.savefig(ROOT / "figures" / "figure5.png", dpi=300, bbox_inches="tight")
    for key, (_, _, low) in ends.items():
        print(key, round(low, 1))


if __name__ == "__main__":
    main()
