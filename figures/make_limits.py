"""Figure at the top of the README: what limits the signals (trading cost, breadth).

Usage: python figures/make_limits.py   (writes figures/limits.png)
Inputs: reports/cost_grid.csv, reports/limits/cost_crossover.csv, reports/limits/breadth.csv,
        reports/limits/breadth_slope.csv (scripts/14_limits.py)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INK, AXIS, GREY, SHADE = "#111111", "#333333", "#6e6e6e", "#f8f5f1"
SIG = {"lgbm": ("LightGBM", "#0072B2"), "ridge": ("Ridge", "#D55E00"), "mom_12_1": ("12-1 momentum", "#009E73")}


def main():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica Neue", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "savefig.facecolor": "white"})
    grid = pd.read_csv(ROOT / "reports" / "cost_grid.csv")
    grid = grid[grid.dataset == "french"]
    cross = pd.read_csv(ROOT / "reports" / "limits" / "cost_crossover.csv")
    cross = cross[cross.dataset == "french"].set_index("signal").crossover_cost_bps
    breadth = pd.read_csv(ROOT / "reports" / "limits" / "breadth.csv")
    breadth = breadth[breadth.dataset == "french"]
    slope = pd.read_csv(ROOT / "reports" / "limits" / "breadth_slope.csv")
    slope = slope[slope.dataset == "french"].set_index("signal").loglog_slope

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.42})

    lo, hi = cross.min(), cross.max()
    a.axvspan(lo, hi, color=SHADE, lw=0, zorder=0)
    for key, (lab, col) in SIG.items():
        g = grid[grid.signal == key].sort_values("cost_bps")
        a.plot(g.cost_bps, g.net_sharpe, "-o", color=col, lw=1.2, ms=3.5, mfc="white", mew=1.0)
        a.text(10.25, g.net_sharpe.iloc[-1], lab, va="center", fontsize=6.5)
    a.axhline(0, color=AXIS, lw=0.5)
    a.axvline(5, color=GREY, lw=0.6, ls=(0, (2, 2)))
    a.text(5.15, 0.685, "assumed cost", fontsize=6.3, color=GREY, va="top")
    a.text(4.75, 0.02, f"models match momentum\nat {lo:.1f}–{hi:.1f} bps", fontsize=6.3, color=GREY,
           ha="right", va="bottom")
    a.set_xticks([0, 2, 5, 10]); a.set_xlim(-0.4, 13.2); a.set_ylim(-0.02, 0.7)
    a.spines["bottom"].set_bounds(0, 10)
    a.set_xlabel("One-way trading cost (bps)", x=0.38); a.set_ylabel("Net Sharpe ratio, 49 industries")
    a.text(-0.16, 1.05, "a", transform=a.transAxes, fontsize=8, fontweight="bold", va="baseline")
    a.text(-0.10, 1.05, "Cost: the models lead only when trading is cheap", transform=a.transAxes, fontsize=7.5,
           va="baseline")

    for key, (lab, col) in SIG.items():
        g = breadth[breadth.signal == key].sort_values("assets")
        b.fill_between(g.assets, g.p10, g.p90, color=col, alpha=0.10, lw=0)
        b.plot(g.assets, g.median_sharpe, "-o", color=col, lw=1.2, ms=3.5, mfc="white", mew=1.0)
        b.text(51, g.median_sharpe.iloc[-1], f"{lab} ({slope[key]:.2f})", va="center", fontsize=6.5)
    n = np.array([8, 49])
    ref = breadth[(breadth.signal == "lgbm") & (breadth.assets == 8)].median_sharpe.iloc[0] * np.sqrt(n / 8)
    b.plot(n, ref, color=GREY, lw=0.7, ls=(0, (2, 2)))
    b.text(30, ref[0] * np.sqrt(30 / 8) + 0.03, "independent bets (0.50)", fontsize=6.3, color=GREY, rotation=31,
           ha="center", va="bottom")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xticks([8, 12, 16, 24, 32, 49]); b.set_xticklabels([8, 12, 16, 24, 32, 49]); b.minorticks_off()
    b.set_yticks([0.3, 0.4, 0.5, 0.7, 1.0]); b.set_yticklabels(["0.3", "0.4", "0.5", "0.7", "1.0"])
    b.set_xlim(7.3, 95); b.set_ylim(0.2, 1.1)
    b.spines["bottom"].set_bounds(8, 49)
    b.set_xlabel("Industries in the cross-section (log scale)", x=0.36)
    b.set_ylabel("Sharpe ratio before costs (log scale)")
    b.text(-0.16, 1.05, "b", transform=b.transAxes, fontsize=8, fontweight="bold", va="baseline")
    b.text(-0.10, 1.05, "Breadth: more correlated assets add little", transform=b.transAxes, fontsize=7.5,
           va="baseline")

    fig.savefig(ROOT / "figures" / "limits.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
