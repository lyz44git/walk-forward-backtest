"""Part 6 (design/tick_costs_plan.md): trading costs of the 30 ETFs from NASDAQ TotalView-ITCH 5.0.

Each day is decompressed and passed to the C++ parser (native/itch_book.cpp), from tmp/tick/raw when the file is
there and otherwise streamed from NASDAQ without being stored. The parser's per-day outputs stay in tmp/tick
(git-ignored); only per-ETF aggregates are written to reports/tick/.

Usage: python3 scripts/17_tick_costs.py [--day 2025-12-08] [--reparse]
Outputs: reports/tick/etf_costs.csv   one row per ETF and day
         reports/tick/summary.csv     the primary estimand and its spread across ETFs, per day
         reports/tick/checks.csv      the parser's checks, per day
"""
from pathlib import Path
import argparse
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
import build as native_build  # noqa: E402

DAYS = {"2025-12-08": "S120825-v50.txt.gz", "2025-12-09": "S120925-v50.txt.gz"}
SOURCE = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"
ETFS = ("DBC DIA EEM EFA EWJ FXI GLD HYG IEF IWM IYR LQD MDY QQQ SHY SLV SPY TIP TLT USO VGK XLB XLE XLF XLI XLK "
        "XLP XLU XLV XLY").split()
TMP, OUT = ROOT / "tmp" / "tick", ROOT / "reports" / "tick"
NS = 1_000_000_000
OPEN, CLOSE = 34_200 * NS, 57_600 * NS
HORIZONS = {"1min": 60, "5min": 300}


def parse(day: str, reparse: bool) -> Path:
    prefix = TMP / day
    if (TMP / f"{day}_summary.csv").exists() and not reparse:
        return prefix
    binary = native_build.build_itch()
    local = TMP / "raw" / DAYS[day]
    reader = f"gzip -dc '{local}'" if local.exists() else f"curl -sS --fail -A research-download '{SOURCE}{DAYS[day]}' | gzip -dc"
    command = f"set -o pipefail; {reader} | '{binary}' --symbols {','.join(ETFS)} --out '{prefix}'"
    print(f"{day}: {'local file' if local.exists() else 'streaming from NASDAQ'}", flush=True)
    result = subprocess.run(["bash", "-c", command])
    if result.returncode:
        raise SystemExit(f"{day}: parser or download failed (exit {result.returncode}); see {prefix}_checks.csv")
    return prefix


def signed_costs(trades: pd.DataFrame, mids: pd.DataFrame) -> pd.DataFrame:
    """Per-execution effective half-spread and, where defined, impact and realized half-spread at each horizon (bps)."""
    t = trades[(trades.ts_ns >= OPEN) & (trades.ts_ns < CLOSE)].copy()
    ok = (t.bid > 0) & (t.ask > t.bid)
    t = t[ok].copy()
    t["mid"] = (t.bid + t.ask) / 2
    quote_sign = np.sign(t.price - t["mid"]).astype(int)
    t["sign_used"] = np.where(t.kind == "P", quote_sign, t.sign)
    t = t[t.sign_used != 0]
    t["effective_bps"] = t.sign_used * (t.price - t["mid"]) / t["mid"] * 1e4
    grid = mids.pivot(index="second", columns="symbol", values="mid")
    for name, seconds in HORIZONS.items():
        later = t.ts_ns + seconds * NS
        second = np.ceil((later - OPEN) / NS).astype(int)
        usable = later <= CLOSE
        values = np.full(len(t), np.nan)
        for symbol, rows in t[usable].groupby("symbol").groups.items():
            if symbol in grid:
                idx = second.loc[rows].to_numpy()
                values[t.index.get_indexer(rows)] = grid[symbol].reindex(idx).to_numpy()
        t[f"impact_{name}_bps"] = t.sign_used * (values - t["mid"]) / t["mid"] * 1e4
        t[f"realized_{name}_bps"] = t["effective_bps"] - t[f"impact_{name}_bps"]
    return t


def weighted(frame, column):
    ok = frame[column].notna()
    w = frame.loc[ok, "shares"]
    return float((frame.loc[ok, column] * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def measure(day: str, prefix: Path) -> pd.DataFrame:
    summary = pd.read_csv(f"{prefix}_summary.csv").set_index("symbol")
    trades = pd.read_csv(f"{prefix}_trades.csv")
    mids = pd.read_csv(f"{prefix}_mids.csv")
    costs = signed_costs(trades, mids)
    rows = []
    for symbol in ETFS:
        s = summary.loc[symbol]
        displayed = costs[(costs.symbol == symbol) & (costs.kind != "P")]
        hidden = costs[(costs.symbol == symbol) & (costs.kind == "P")]
        total = s.displayed_shares + s.hidden_shares
        rows.append({
            "day": day, "etf": symbol, "quoted_half_spread_bps": s.quoted_half_spread_bps,
            "open5_quoted_half_spread_bps": s.open5_quoted_half_spread_bps,
            "effective_half_spread_bps": weighted(displayed, "effective_bps"),
            **{f"impact_{h}_bps": weighted(displayed, f"impact_{h}_bps") for h in HORIZONS},
            **{f"realized_{h}_bps": weighted(displayed.dropna(subset=[f"impact_{h}_bps"]), f"realized_{h}_bps")
               for h in HORIZONS},
            "hidden_effective_half_spread_bps": weighted(hidden, "effective_bps"),
            "hidden_share_of_volume": s.hidden_shares / total if total else np.nan,
            "bid_depth_usd": s.bid_depth_usd, "ask_depth_usd": s.ask_depth_usd,
            "displayed_execs": int(s.displayed_execs), "displayed_shares": int(s.displayed_shares),
            "hidden_execs": int(s.hidden_execs), "hidden_shares": int(s.hidden_shares),
            "quoted_seconds": s.valid_seconds, "crossed_or_locked_seconds": s.crossed_or_locked_seconds,
            "open_cross_price": s.open_cross_price, "open_cross_shares": int(s.open_cross_shares),
        })
    return pd.DataFrame(rows)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--day", choices=sorted(DAYS))
    args.add_argument("--reparse", action="store_true")
    a = args.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for day in ([a.day] if a.day else sorted(DAYS)):
        parse(day, a.reparse)
    done = [d for d in sorted(DAYS) if (TMP / f"{d}_summary.csv").exists()]
    table = pd.concat([measure(d, TMP / d) for d in done], ignore_index=True)
    table.to_csv(OUT / "etf_costs.csv", index=False)
    checks = pd.concat([pd.read_csv(TMP / f"{d}_checks.csv").assign(day=d) for d in done], ignore_index=True)
    checks.to_csv(OUT / "checks.csv", index=False)
    rows = []
    for d, g in table.groupby("day"):
        e = g["effective_half_spread_bps"]
        rows.append({"day": d, "etfs": int(e.notna().sum()), "primary_mean_effective_half_spread_bps": e.mean(),
                     "median": e.median(), "min": e.min(), "max": e.max(),
                     "mean_quoted_half_spread_bps": g.quoted_half_spread_bps.mean(),
                     "mean_open5_quoted_half_spread_bps": g.open5_quoted_half_spread_bps.mean(),
                     "mean_impact_1min_bps": g.impact_1min_bps.mean(), "mean_impact_5min_bps": g.impact_5min_bps.mean(),
                     "mean_realized_5min_bps": g.realized_5min_bps.mean(),
                     "mean_hidden_share_of_volume": g.hidden_share_of_volume.mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
