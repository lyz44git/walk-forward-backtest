"""Figure 4 of the README: what features beyond the assets' own past prices add to the price-based models.

Usage: python figures/make_figure4.py   (writes figures/figure4.png)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INK, AXIS, GREY = "#111111", "#333333", "#8a8a8a"
COL = {"mean": "#111111", "ridge": "#009E73", "lgbm": "#D55E00"}
LAB = {"mean": "Mean of both", "ridge": "Ridge", "lgbm": "LightGBM"}
DS = {"etf": "30 ETFs", "french": "49 industries"}


def main():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica Neue", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "savefig.facecolor": "white"})
    out = ROOT / "reports" / "nonprice"
    pooled = pd.read_csv(out / "pooled.csv")
    pooled = pooled[(pooled.span == "full") & (pooled.block == 8) & (pooled.cost_bps == 5.0) & (pooled.features == "PN")]
    attr = pd.read_csv(out / "attribution.csv")
    attr = attr[attr.model.str.contains("momentum") & attr.signal.isin(["ridge", "lgbm"]) & attr.features.isin(["P", "PN"])]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for i, ds in enumerate(DS):
        for j, m in enumerate(COL):
            r = pooled[(pooled.dataset == ds) & (pooled.model == m)].iloc[0]
            x = i + (j - 1) * 0.2
            a.plot([x, x], [r.ci_low, r.ci_high], color=COL[m], lw=1.0)
            a.plot(x, r.sharpe_difference, "o", ms=3.5, color=COL[m], label=LAB[m] if i == 0 else None)
    a.axhline(0, color=AXIS, lw=0.5)
    a.set_xticks(range(2)); a.set_xticklabels(DS.values()); a.set_xlim(-0.5, 1.5)
    a.set_ylabel("Net Sharpe at 5 bps: with new features\nminus price features only (95% CI)")
    a.legend(frameon=False, fontsize=6.2, loc="upper right")
    a.set_title("a", fontsize=8, loc="left", fontweight="bold")

    ticks, labels = [], []
    for i, ds in enumerate(DS):
        for j, m in enumerate(("ridge", "lgbm")):
            for k, f in enumerate(("P", "PN")):
                r = attr[(attr.dataset == ds) & (attr.signal == m) & (attr.features == f)].iloc[0]
                x = i * 2.6 + j * 1.1 + k * 0.35
                half = 1.96 * abs(r.alpha_annual / r.alpha_t) * 100
                b.plot([x, x], [r.alpha_annual * 100 - half, r.alpha_annual * 100 + half], color=COL[m], lw=1.0)
                b.plot(x, r.alpha_annual * 100, "o", ms=3.5, color=COL[m], mfc=COL[m] if f == "PN" else "white", mew=0.8)
            ticks.append(i * 2.6 + j * 1.1 + 0.175); labels.append(f"{LAB[m]}\n{DS[ds]}")
    b.axhline(0, color=AXIS, lw=0.5)
    b.set_xticks(ticks); b.set_xticklabels(labels, fontsize=6)
    b.set_ylabel("Alpha after market, size, value and\nsame-universe momentum (% / yr, 95% CI)")
    b.text(0.99, 0.03, "open: price features only\nfilled: with new features", transform=b.transAxes, ha="right",
           va="bottom", fontsize=6.2, color=GREY)
    b.set_title("b", fontsize=8, loc="left", fontweight="bold")
    fig.subplots_adjust(left=0.095, right=0.99, top=0.91, bottom=0.14, wspace=0.32)
    fig.savefig(ROOT / "figures" / "figure4.png", dpi=300)


if __name__ == "__main__":
    main()
