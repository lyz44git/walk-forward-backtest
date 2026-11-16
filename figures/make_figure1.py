"""Figure 1 of the README: cumulative net returns with three supporting panels, drawn from the report tables.

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
INK, INK2, AXIS = "#111111", "#5a5a5a", "#333333"
BLUE, RED, GREEN, GREY, SKY = "#0072B2", "#D55E00", "#009E73", "#9a9a9a", "#56B4E9"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f6f1e4", "#8fd0c0", "#2b83ba", "#0b2a66"])
HERO = [0.075, 0.095, 0.385, 0.815]             # large left panel
X0, W, H, ROWS = 0.60, 0.385, 0.20, (0.735, 0.41, 0.085)   # three stacked panels on the right


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def style():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica Neue", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.minor.size": 1.2, "ytick.minor.size": 1.2, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "savefig.facecolor": "white"})


def detach(ax):
    """Axes drawn slightly away from the data, as in journal figures."""
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 3))


def letter(fig, ax, s, title, x=0.497):
    box = ax.get_position()
    fig.text(x, box.y1 + 0.022, s, fontsize=10, fontweight="bold", va="bottom")
    fig.text(x + 0.022, box.y1 + 0.0235, title, fontsize=7.5, va="bottom")


def leader(ax, xy, xytext, text, **kw):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=6.5, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=AXIS, shrinkA=1.5, shrinkB=2), **kw)


SIG = {"lgbm": "LightGBM", "ridge": "Ridge", "mom_12_1": "12-1 momentum", "rev_1w": "1-week reversal"}
COL = {"lgbm": BLUE, "ridge": RED, "mom_12_1": GREEN, "rev_1w": GREY}
R = ROOT / "reports"


def curves(ax, weekly, ds, lw):
    """Value of 1 invested in each long-short signal after costs; returns the final values."""
    end = {}
    for sig in ("rev_1w", "mom_12_1", "ridge", "lgbm"):
        s, v, ys = sorted(weekly[(ds, sig)]), 1.0, []
        for _, x in s:
            v *= 1 + x
            ys.append(v)
        ax.plot([p[0] for p in s], ys, color=COL[sig], lw=lw)
        end[sig] = (s[-1][0], v)
    ax.axhline(1, color=AXIS, lw=0.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    return end


def draw():
    fig = plt.figure(figsize=(7.2, 5.0), dpi=300)
    weekly = {}
    for r in rows(R / "ls_weekly.csv"):
        weekly.setdefault((r["dataset"], r["signal"]), []).append((dt.date.fromisoformat(r["date"]), float(r["net"])))
    M = {(r["dataset"], r["signal"]): r for r in rows(R / "main_results.csv")}
    Hv = {(r["dataset"], r["signal"], r["half"]): r for r in rows(R / "halves.csv")}

    # ---- a: industry portfolios, value of 1 invested after costs
    a = fig.add_axes(HERO)
    end = curves(a, weekly, "french", 0.9)
    split = dt.date.fromisoformat(Hv[("french", "lgbm", "second half")]["from"])
    a.axvline(split, color=INK2, lw=0.6, ls=(0, (3, 3)))
    for half, x, ha in (("first half", dt.date(1987, 6, 1), "center"), ("second half", dt.date(2011, 6, 1), "center")):
        h = Hv[("french", "lgbm", half)]
        a.text(x, 0.042, f'{h["from"][:4]}–{h["to"][:4]}\nSharpe {float(h["net_sharpe"]):.2f}', ha=ha,
               va="bottom", fontsize=6.5, linespacing=1.2)
    last = max(d for d, _ in end.values())
    off = {"mom_12_1": 1.0, "lgbm": 1.3, "ridge": 0.75, "rev_1w": 1.0}
    for sig, (d0, v) in end.items():
        a.text(last + dt.timedelta(days=330), v * off[sig], SIG[sig], color=INK, fontsize=6.5, va="center")
        a.plot([last + dt.timedelta(days=60), last + dt.timedelta(days=270)], [v, v * off[sig]], color=AXIS, lw=0.5)
    a.set_xlim(dt.date(1974, 6, 1), dt.date(2042, 6, 1)); a.set_ylim(0.035, 70); a.set_yticks([0.1, 1, 10])
    a.set_xticks([dt.date(y, 1, 1) for y in (1980, 1990, 2000, 2010, 2020)], ["1980", "1990", "2000", "2010", "2020"])
    a.set_ylabel("Value of 1 invested, long-short, net of costs (log scale)")
    detach(a)
    fig.text(0.012, 0.957, "a", fontsize=10, fontweight="bold", va="bottom")
    fig.text(0.034, 0.9585, f'49 industry portfolios, {int(M[("french", "lgbm")]["weeks"]):,} weeks: LightGBM net Sharpe by half',
             fontsize=7.5, va="bottom")

    # inset: the same on tradable ETFs
    ins = a.inset_axes([0.07, 0.72, 0.29, 0.21])
    curves(ins, weekly, "etf", 0.6)
    ins.set_ylim(0.45, 2.1); ins.set_yticks([0.5, 1, 2])
    ins.set_xticks([dt.date(y, 1, 1) for y in (2012, 2019, 2026)], ["2012", "2019", "2026"])
    ins.tick_params(labelsize=6, length=1.8, pad=1.5); ins.set_facecolor("none")
    ins.set_title(f'30 ETFs, {int(M[("etf", "lgbm")]["weeks"])} weeks', fontsize=6.5, loc="left", pad=2)

    b, c, d = (fig.add_axes([X0, yy, W, H]) for yy in ROWS)

    # ---- b: real rank IC against shuffled-target placebos (ridge, 100 seeds per dataset)
    P = rows(R / "placebo_seeds.csv")
    S = {(r["dataset"], r["signal"]): r for r in rows(R / "placebo_summary.csv")}
    bins = [-0.03 + 0.002 * i for i in range(31)]
    for ds, name, col in (("etf", "ETFs", SKY), ("french", "industries", BLUE)):
        vals = [float(r["ic_mean"]) for r in P if r["dataset"] == ds and r["signal"] == "ridge"]
        b.hist(vals, bins=bins, color=col, alpha=0.75, lw=0.4, edgecolor="white")
        real = float(S[(ds, "ridge")]["real_ic_mean"])
        b.plot([real, real], [0, 17], color=col, lw=1.4)
        b.text(real + (-0.001 if ds == "etf" else 0.001), 12, f"{real:.3f}", ha="right" if ds == "etf" else "left",
               va="center", fontsize=6.5, fontweight="bold")
    leader(b, (-0.016, 4.5), (-0.027, 13), "ETF placebos", ha="left")
    leader(b, (0.0045, 14), (0.009, 22), "industry placebos", ha="left")
    b.text(0.036, 19.5, "real IC", ha="center", fontsize=6.5, color=INK2)
    b.set_xlim(-0.03, 0.047); b.set_ylim(0, 27); b.set_yticks([0, 10, 20]); b.set_xticks([-0.02, 0, 0.02, 0.04])
    b.set_xlabel("Mean weekly rank IC, ridge", labelpad=2); b.set_ylabel("Placebo seeds")
    detach(b); letter(fig, b, "b", "The ranking signal beats all 100 shuffled-target placebos")

    # ---- c: net Sharpe with bootstrap 95% intervals
    c.axvspan(-0.9, 0, color=RED, alpha=0.05, lw=0)
    c.axvline(0, color=AXIS, lw=0.5)
    order = ("mom_12_1", "ridge", "lgbm", "rev_1w")
    ticks = []
    for i, sig in enumerate(order):
        y = len(order) - 1 - i
        ticks.append(y)
        for ds, dy, filled in (("etf", 0.17, False), ("french", -0.17, True)):
            r = M[(ds, sig)]
            v, lo, hi = (float(r[k]) for k in ("ls_net_sharpe", "ls_net_sharpe_ci_lo", "ls_net_sharpe_ci_hi"))
            col = SKY if ds == "etf" else BLUE
            c.plot([lo, hi], [y + dy, y + dy], color=col, lw=1.3, solid_capstyle="round")
            c.plot(v, y + dy, marker="o", ms=4, mfc=col if filled else "white", mec=col, mew=1.0)
    e = M[("etf", "mom_12_1")]; f = M[("french", "mom_12_1")]
    c.plot(0.02, 3.95, marker="o", ms=4, mfc="white", mec=SKY, mew=1.0); c.text(0.06, 3.95, "ETFs", va="center", fontsize=6.5)
    c.plot(0.26, 3.95, marker="o", ms=4, color=BLUE); c.text(0.30, 3.95, "industry portfolios", va="center", fontsize=6.5)
    c.set_yticks(ticks, [SIG[s] for s in order]); c.tick_params(axis="y", length=0)
    c.set_xlim(-0.75, 0.75); c.set_ylim(-0.6, 4.3); c.set_xticks([-0.5, 0, 0.5])
    c.set_xlabel("Net Sharpe ratio, 95% interval", labelpad=2)
    c.spines["left"].set_visible(False); c.spines["bottom"].set_position(("outward", 3))
    letter(fig, c, "c", "After costs, every ETF interval spans zero")

    # ---- d: net Sharpe on ETFs against the assumed one-way cost
    G = {}
    for r in rows(R / "cost_grid.csv"):
        if r["dataset"] == "etf":
            G.setdefault(r["signal"], []).append((float(r["cost_bps"]), float(r["net_sharpe"])))
    d.axhline(0, color=AXIS, lw=0.5)
    d.axvline(5, color=INK2, lw=0.6, ls=(0, (3, 3)))
    d.text(5.15, 0.62, "assumed cost", fontsize=6.3, color=INK2, va="center")
    for sig in ("rev_1w", "mom_12_1", "ridge", "lgbm"):
        pts = sorted(G[sig])
        d.plot([p[0] for p in pts], [p[1] for p in pts], color=COL[sig], lw=1.2, marker="o", ms=3, mfc="white", mew=1.0)
    ends = {sig: sorted(G[sig])[-1][1] for sig in G}
    place = {"mom_12_1": ends["mom_12_1"] + 0.03, "ridge": ends["ridge"] + 0.07, "lgbm": ends["lgbm"] - 0.09,
             "rev_1w": ends["rev_1w"]}
    for sig, yy in place.items():
        d.text(10.35, yy, SIG[sig], fontsize=6.5, va="center")
    d.set_xlim(-0.4, 13.6); d.set_xticks([0, 2, 5, 10]); d.set_ylim(-0.75, 0.75); d.set_yticks([-0.5, 0, 0.5])
    d.set_xlabel("One-way trading cost (bps)", labelpad=2); d.set_ylabel("Net Sharpe,\nETFs")
    detach(d); letter(fig, d, "d", "Turnover uses up the models' edge as costs rise")
    return fig


if __name__ == "__main__":
    style()
    draw().savefig(Path(__file__).parent / "figure1.png", dpi=300)
