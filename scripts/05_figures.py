"""Figures from the saved reports (no recomputation of results)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"mom_12_1": "#2a78d6", "lgbm": "#eb6834", "ridge": "#1baf7a", "rev_1w": "#eda100"}
LAB = {"mom_12_1": "12-1 momentum", "rev_1w": "1-week reversal", "ridge": "Ridge", "lgbm": "LightGBM"}
DS = {"etf": "30 ETFs, 2011–2026 (enter next open)", "french": "49 industry portfolios, 1975–2026 (enter next close)"}
mpl.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
                     "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.labelsize": 9.5,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
                     "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5, "savefig.dpi": 160, "savefig.bbox": "tight"})
note = lambda fig, t: fig.text(0.0, -0.02, t, ha="left", va="top", fontsize=7.8, color=INK2, wrap=True)
OUT = ROOT / "reports"
M = pd.read_csv(OUT / "main_results.csv"); L = pd.read_csv(OUT / "ls_weekly.csv", parse_dates=["date"])
P = pd.read_csv(OUT / "placebo_seeds.csv"); PS = pd.read_csv(OUT / "placebo_summary.csv")

# ---------------------------------------------------------------- fig 1: cumulative net long-short
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, ds in zip(axes, ["etf", "french"]):
    for s in ["mom_12_1", "lgbm", "ridge", "rev_1w"]:
        d = L[(L.dataset == ds) & (L.signal == s)].sort_values("date")
        cum = np.log1p(d.net).cumsum()
        sh = M[(M.dataset == ds) & (M.signal == s)].ls_net_sharpe.iloc[0]
        ax.plot(d.date, cum, color=COL[s], lw=1.6, label=f"{LAB[s]} (net Sharpe {sh:.2f})")
    ax.axhline(0, color=INK2, lw=0.8); ax.set_title(DS[ds]); ax.set_ylabel("Cumulative net log return")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
fig.suptitle("Weekly long-short, top vs bottom 20%, after 5 bps one-way costs and 50 bps/yr borrow", x=0.0, ha="left",
             fontsize=11.5, fontweight="semibold", color=INK, y=1.03)
fig.text(0.0, -0.08, "Out-of-sample walk-forward (expanding window, refit every 4 weeks, purged with a 1-week embargo). "
          "Models and costs fixed in advance (design/preregistration.json). ETF universe = 30 funds alive today (survivorship).", ha="left", va="top", fontsize=7.8, color=INK2, wrap=True)
fig.tight_layout(); fig.savefig(OUT / "fig1_cumulative_net.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 2: placebo distributions
fig, axes = plt.subplots(2, 2, figsize=(10, 5.6))
for ax, (ds, s) in zip(axes.ravel(), [("etf", "ridge"), ("etf", "lgbm"), ("french", "ridge"), ("french", "lgbm")]):
    g = P[(P.dataset == ds) & (P.signal == s)].ic_mean
    row = PS[(PS.dataset == ds) & (PS.signal == s)].iloc[0]
    ax.hist(g, bins=20 if len(g) > 20 else 8, color="#b9b8b2", edgecolor=SURFACE, linewidth=1.0)
    ax.axvline(row.real_ic_mean, color=COL[s], lw=2.2)
    ax.text(row.real_ic_mean, ax.get_ylim()[1] * 0.92, f" real IC {row.real_ic_mean:.3f}", color=INK, fontsize=8.5, va="top")
    ax.set_title(f"{'ETFs' if ds == 'etf' else 'Industries'}: {LAB[s]} ({int(row.seeds)} placebo seeds, p = {row.empirical_p_value:.3f})", fontsize=10)
    ax.set_xlabel("Mean out-of-sample rank IC"); ax.set_ylabel("Seeds")
fig.suptitle("Placebo: training targets shuffled across assets within each week (fresh shuffle at every refit)",
             x=0.0, ha="left", fontsize=11.5, fontweight="semibold", color=INK, y=1.02)
note(fig, "Grey: placebo distribution of the mean OOS IC. Line: the real model. p = (seeds with IC at or above the real IC + 1) / (seeds + 1); "
          "with 10 seeds the smallest possible p is 0.091.")
fig.tight_layout(); fig.savefig(OUT / "fig2_placebo.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 3: gross vs net Sharpe
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
order = ["mom_12_1", "ridge", "lgbm", "rev_1w"]
for ax, ds in zip(axes, ["etf", "french"]):
    d = M[M.dataset == ds].set_index("signal").loc[order]
    x = np.arange(len(order)); w = 0.36
    ax.bar(x - w / 2 - 0.01, d.ls_gross_sharpe, w, color="#9ec5f0", label="Before costs")
    ax.bar(x + w / 2 + 0.01, d.ls_net_sharpe, w, color="#2a78d6", label="After costs (95% CI)")
    ax.errorbar(x + w / 2 + 0.01, d.ls_net_sharpe, yerr=[d.ls_net_sharpe - d.ls_net_sharpe_ci_lo, d.ls_net_sharpe_ci_hi - d.ls_net_sharpe],
                fmt="none", ecolor=INK2, elinewidth=1.0, capsize=3)
    for xi, (tv, sh) in enumerate(zip(d.ls_annual_turnover, d.ls_net_sharpe)):
        ax.text(xi, -0.95, f"turnover\n{tv:.0f}x/yr", ha="center", va="bottom", fontsize=7.6, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([LAB[s] for s in order]); ax.set_title(DS[ds], fontsize=10)
    ax.set_ylim(-1.0, 1.0)
axes[0].set_ylabel("Annualized Sharpe ratio"); axes[0].legend(loc="upper right")
fig.suptitle("Long-short Sharpe before and after trading costs", x=0.0, ha="left", fontsize=11.5, fontweight="semibold", color=INK, y=1.03)
note(fig, "Costs: 5 bps one-way per unit traded plus 50 bps/yr borrow on the short side. CI: stationary block bootstrap (mean block 8 weeks, 2,000 draws).")
fig.tight_layout(); fig.savefig(OUT / "fig3_gross_vs_net_sharpe.png"); plt.close(fig)
print("figures saved")
