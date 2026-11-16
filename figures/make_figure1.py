"""Figure 1 of the README: the four headline results, drawn from the committed report tables.

Usage: python figures/make_figure1.py   (writes figures/figure1.png)
"""
import datetime as dt
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
T = dict(bg="#ffffff", ink="#111111", ink2="#555555", axis="#333333", grid="#e6e6e6", band="#e9eff5",
         blue="#0072B2", red="#D55E00", green="#009E73", grey="#8c8c8c")
FIGSIZE = (7.2, 5.6)
X0, W, H = (0.115, 0.625), 0.355, 0.30          # two equal columns
Y0 = (0.575, 0.085)                             # two equal rows


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def style(t):
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica Neue", "DejaVu Sans"], "font.size": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "text.color": t["ink"], "axes.labelcolor": t["ink"], "axes.edgecolor": t["axis"],
        "xtick.color": t["axis"], "ytick.color": t["axis"], "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2, "legend.frameon": False})


def grid(fig):
    """Four axes of equal size on a 2 x 2 grid."""
    return [fig.add_axes([x, y, W, H]) for y in Y0 for x in X0]


def panel(fig, ax, letter, title, legend_rows=1):
    """Panel letter and bold title above the legend strip; the strip holds `legend_rows` rows in this figure row."""
    box = ax.get_position()
    y = box.y1 + (7 + 10.5 * legend_rows) / 72 / fig.get_figheight()
    fig.text(box.x0 - 0.105, y, letter, fontsize=11.5, fontweight="bold", va="bottom")
    fig.text(box.x0, y, title, fontsize=8.5, fontweight="bold", va="bottom")


def legend(ax, ncol, handles=None, labels=None):
    """Legend in a strip between the title and the axes, clear of the data."""
    args = (handles, labels) if handles is not None else ()
    return ax.legend(*args, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, borderaxespad=0.35, borderpad=0,
                     handlelength=1.3, handletextpad=0.5, columnspacing=1.4, labelspacing=0.35)


SIG = {"lgbm": "LightGBM", "ridge": "Ridge", "mom_12_1": "12-1 momentum", "rev_1w": "1-week reversal"}


def draw(t):
    R = ROOT / "reports"
    col = {"lgbm": t["blue"], "ridge": t["red"], "mom_12_1": t["green"], "rev_1w": t["grey"]}
    fig = plt.figure(figsize=FIGSIZE, dpi=300)
    a, b, c, d = grid(fig)

    # a, b: cumulative long-short return after costs
    weekly = {}
    for r in rows(R / "ls_weekly.csv"):
        weekly.setdefault((r["dataset"], r["signal"]), []).append((dt.date.fromisoformat(r["date"]), float(r["net"])))
    M = {(r["dataset"], r["signal"]): r for r in rows(R / "main_results.csv")}
    for ax, ds in ((a, "etf"), (b, "french")):
        ax.axhline(1, color=t["axis"], lw=0.6)
        for sig in ("lgbm", "ridge", "mom_12_1", "rev_1w"):
            s, v, ys = sorted(weekly[(ds, sig)]), 1.0, []
            for _, x in s:
                v *= 1 + x
                ys.append(v)
            ax.plot([p[0] for p in s], ys, color=col[sig], lw=0.9, label=SIG[sig],
                    zorder=2 if sig == "rev_1w" else 3)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("Value of 1 invested, net of costs")
        legend(ax, 2)
    a.set_yticks([0.5, 1, 2]); a.set_ylim(0.45, 2.2); b.set_yticks([0.1, 1, 10]); b.set_ylim(0.04, 20)
    panel(fig, a, "a", f"30 ETFs, {int(M[('etf', 'lgbm')]['weeks']):,} weeks", 2)
    panel(fig, b, "b", f"49 industry portfolios, {int(M[('french', 'lgbm')]['weeks']):,} weeks", 2)

    # c: real rank IC against shuffled-target placebos (ridge, 100 seeds per dataset)
    P = rows(R / "placebo_seeds.csv")
    S = {(r["dataset"], r["signal"]): r for r in rows(R / "placebo_summary.csv")}
    bins = [-0.03 + 0.002 * i for i in range(31)]
    for ds, name, colr, side in (("etf", "ETFs", t["blue"], "right"), ("french", "Industries", t["red"], "left")):
        vals = [float(r["ic_mean"]) for r in P if r["dataset"] == ds and r["signal"] == "ridge"]
        c.hist(vals, bins=bins, color=colr, alpha=0.55, lw=0, label=f"{name} placebos")
        real = float(S[(ds, "ridge")]["real_ic_mean"])
        c.axvline(real, color=colr, lw=1.3)
        c.text(real + (0.0012 if side == "left" else -0.0012), 23.5, f"{real:.3f}" if side == "left" else f"real IC  {real:.3f}",
               ha=side, va="top")
    c.set_xlim(-0.03, 0.05); c.set_ylim(0, 25)
    c.set_xlabel("Mean weekly rank IC (ridge)"); c.set_ylabel("Placebo seeds")
    legend(c, 2)
    panel(fig, c, "c", "Signal against shuffled-target placebos")

    # d: net Sharpe with bootstrap 95% intervals
    ticks, labels = [], []
    d.axvline(0, color=t["axis"], lw=0.6)
    for g, (ds, name) in enumerate((("etf", "ETFs"), ("french", "Industry portfolios"))):
        top = 9.4 - 5.0 * g
        d.text(-0.73, top, name, fontweight="bold", va="center", fontsize=7)
        for i, sig in enumerate(("mom_12_1", "ridge", "lgbm", "rev_1w")):
            r, y = M[(ds, sig)], top - 1 - i
            v, lo, hi = (float(r[k]) for k in ("ls_net_sharpe", "ls_net_sharpe_ci_lo", "ls_net_sharpe_ci_hi"))
            d.errorbar(v, y, xerr=[[v - lo], [hi - v]], color=t["grey"] if lo < 0 < hi else t["blue"], marker="o",
                       ms=4, elinewidth=1.1, capsize=0)
            ticks.append(y); labels.append(SIG[sig])
    d.set_yticks(ticks, labels); d.tick_params(axis="y", length=0); d.set_ylim(-0.2, 10.0); d.set_xlim(-0.75, 0.75)
    d.set_xlabel("Net Sharpe ratio, 95% CI")
    panel(fig, d, "d", "Net Sharpe ratio against zero")
    return fig


if __name__ == "__main__":
    style(T)
    fig = draw(T)
    fig.savefig(Path(__file__).parent / "figure1.png", dpi=300)
