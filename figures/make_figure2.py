"""Figure 2 of the README: what trading less is worth, drawn from reports/rebalancing.

Usage: python figures/make_figure2.py   (writes figures/figure2.png)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "rebalancing"
INK, INK2, AXIS = "#111111", "#5a5a5a", "#333333"
COL = {"full": "#9a9a9a", "partial_adjustment": "#0072B2", "no_trade_band": "#009E73",
       "rank_buffer": "#D55E00", "signal_smoothing": "#CC79A7", "ppo": "#111111"}
LAB = {"full": "Full rebalancing", "partial_adjustment": "Partial adjustment", "no_trade_band": "No-trade band",
       "rank_buffer": "Rank buffer", "signal_smoothing": "Signal smoothing", "ppo": "PPO policy"}
ORDER = ["partial_adjustment", "no_trade_band", "rank_buffer", "signal_smoothing", "ppo"]
DS = {"etf": "30 ETFs, test 2019–2026 (408 weeks)", "french": "49 industry portfolios, test 2001–2026 (1,342 weeks)"}


def main():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica Neue", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "savefig.facecolor": "white"})
    res = pd.read_csv(REP / "rules_results.csv")
    pooled = pd.read_csv(REP / "rules_pooled.csv")
    pooled = pooled[(pooled.block == 8) & (pooled.versus == "full")]
    mean = res.groupby(["dataset", "cost_bps", "rule"])[["sharpe", "annual_turnover"]].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    for row, ds in enumerate(["etf", "french"]):
        ax = axes[row, 0]
        full_to = mean[(mean.dataset == ds) & (mean.cost_bps == 5) & (mean.rule == "full")].annual_turnover.iloc[0]
        for i, rule in enumerate(ORDER):
            y = len(ORDER) - 1 - i
            for cost, dy, filled in ((5.0, 0.14, True), (10.0, -0.14, False)):
                p = pooled[(pooled.dataset == ds) & (pooled.cost_bps == cost) & (pooled.rule == rule)].iloc[0]
                ax.plot([p.ci_low, p.ci_high], [y + dy] * 2, color=COL[rule], lw=1.1, solid_capstyle="butt")
                ax.plot(p.mean_sharpe_difference, y + dy, "o", ms=4, mec=COL[rule], mew=0.9,
                        mfc=COL[rule] if filled else "white")
            to = mean[(mean.dataset == ds) & (mean.cost_bps == 5) & (mean.rule == rule)].annual_turnover.iloc[0]
            ax.text(1.02, y, f"{to:.0f}×  ({to / full_to - 1:+.0%})", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=6.5, color=INK2)
        ax.axvline(0, color=AXIS, lw=0.5)
        ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([LAB[r] for r in ORDER[::-1]])
        ax.set_ylim(-0.6, len(ORDER) - 0.4); ax.set_xlim(-0.45, 0.4)
        ax.set_title(DS[ds], fontsize=7.5, loc="left", pad=10, fontweight="bold")
        ax.text(1.02, 1.02, f"Turnover / yr\n(full: {full_to:.0f}×)", transform=ax.transAxes, fontsize=6.5,
                color=INK2, va="bottom", ha="left")
        if row == 1:
            ax.set_xlabel("Net Sharpe minus full rebalancing, mean over three signals (95% interval)")

        ax = axes[row, 1]
        for rule in ["full"] + ORDER:
            d = mean[(mean.dataset == ds) & (mean.rule == rule)].sort_values("cost_bps")
            ax.plot(d.cost_bps, d.sharpe, marker="o", color=COL[rule], lw=1.6 if rule == "full" else 1.0, ms=2.6,
                    ls="--" if rule == "full" else "-", label=LAB[rule])
        ax.axhline(0, color=AXIS, lw=0.5); ax.set_xticks([0, 2, 5, 10])
        ax.set_ylabel("Net Sharpe, mean over signals")
        if row == 1:
            ax.set_xlabel("One-way trading cost (bps)")
        if row == 0:
            ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="lower left", handlelength=1.6, columnspacing=1.0)
    axes[1, 0].plot([], [], "o", ms=4, mec=INK2, mfc=INK2, label="5 bps (base cost)")
    axes[1, 0].plot([], [], "o", ms=4, mec=INK2, mfc="white", label="10 bps")
    axes[1, 0].legend(frameon=False, fontsize=6.2, loc="lower right", handletextpad=0.2)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.93, bottom=0.10, wspace=0.52, hspace=0.42)
    fig.savefig(ROOT / "figures" / "figure2.png", dpi=300)


if __name__ == "__main__":
    main()
