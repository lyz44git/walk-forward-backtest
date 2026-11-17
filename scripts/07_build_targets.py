"""Build the weekly target portfolios of the main backtest (same walk-forward, timing and purge) and cache them.

Usage: python3 scripts/07_build_targets.py [signal ...] [--dataset etf|french]
Defaults to both datasets and the momentum, ridge and LightGBM signals.
Output: tmp/rebalance/case_{dataset}_{signal}.npz
"""
import argparse
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from qmb.rebalance import cases as data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("signals", nargs="*", default=["mom_12_1", "ridge", "lgbm"])
parser.add_argument("--dataset", choices=("etf", "french"), action="append")
args = parser.parse_args()

t0 = time.perf_counter()
for name in args.dataset or ("etf", "french"):
    cases = data.build_cases(name, args.signals)
    for s, c in cases.items():
        data.save_case(c, data.TMP / f"case_{name}_{s}.npz")
        print(f"[{time.perf_counter()-t0:5.0f}s] {name} {s}: {c['W'].shape[0]} weeks, {c['W'].shape[1]} assets", flush=True)
