"""Figure 3 of the README: how fast the predictions decay, and what a longer label does.

Usage: python figures/make_figure3.py   (writes figures/figure3.png)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, AXIS = "#111111", "#5a5a5a", "#333333"
COL = {"mom_12_1": "#0072B2", "ridge": "#009E73", "lgbm": "#D55E00"}
LAB = {"mom_12_1": "12-1 momentum", "ridge": "Ridge", "lgbm": "LightGBM"}
DS = {"etf": "30 ETFs, 2011–2026", "french": "49 industry portfolios, 1975–2026"}


def main():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica Neue", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "savefig.facecolor": "white"})
    decay = pd.read_csv(ROOT / "reports" / "diagnostics" / "ic_decay.csv")
    hor = pd.read_csv(ROOT / "reports" / "horizon" / "horizon_results.csv")
    hor = hor[hor.cost_bps == 5.0]
    main_res = pd.read_csv(ROOT / "reports" / "main_results.csv").set_index(["dataset", "signal"])

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    for row, ds in enumerate(["etf", "french"]):
        ax = axes[row, 0]
        for s in COL:
            d = decay[(decay.dataset == ds) & (decay.signal == s)]
            ax.plot(d.lag_weeks, d.ic_mean, marker="o", ms=2.4, lw=1.0, color=COL[s], label=LAB[s])
        ax.set_ylim(0, 0.045); ax.set_xticks([0, 4, 8, 12]); ax.set_ylabel("Mean rank IC with the\none-week return j weeks later")
        ax.set_title(DS[ds], fontsize=7.5, loc="left", fontweight="bold", pad=8)
        if row == 0:
            ax.legend(frameon=False, fontsize=6.2, loc="lower left")
        for col, (key, label) in enumerate((("annual_turnover", "Turnover / yr (×)"), ("sharpe", "Net Sharpe at 5 bps")), 1):
            ax = axes[row, col]
            for s in ("ridge", "lgbm"):
                d = hor[(hor.dataset == ds) & (hor.signal == s)].sort_values("horizon")
                ax.plot(range(3), d[key], marker="o", ms=3, lw=1.0, color=COL[s])
            ref = main_res.loc[(ds, "mom_12_1"), "ls_annual_turnover" if key == "annual_turnover" else "ls_net_sharpe"]
            ax.axhline(ref, color=COL["mom_12_1"], lw=1.0, ls="--")
            if key == "sharpe":
                ax.axhline(0, color=AXIS, lw=0.5)
            else:
                ax.set_ylim(0, None)
            ax.set_xticks(range(3)); ax.set_xticklabels(["1", "4", "13"]); ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Label horizon (weeks)")
        if row == 1:
            axes[row, 0].set_xlabel("Lag j (weeks)")
    axes[0, 1].text(0.98, 0.30, "dashed: 12-1 momentum", transform=axes[0, 1].transAxes, ha="right", va="bottom",
                    fontsize=6.2, color=COL["mom_12_1"])
    fig.subplots_adjust(left=0.085, right=0.99, top=0.93, bottom=0.10, wspace=0.42, hspace=0.42)
    fig.savefig(ROOT / "figures" / "figure3.png", dpi=300)


if __name__ == "__main__":
    main()
