"""Part 6 (design/tick_costs_plan.md): the ITCH 5.0 parser and order book on message streams built from the
specification. Skipped when no C++ compiler is available, unless QMB_REQUIRE_NATIVE=1."""
import importlib.util
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NS = 1_000_000_000
OPEN = 34_200 * NS


@pytest.fixture(scope="module")
def binary():
    spec = importlib.util.spec_from_file_location("qmb_native_build", ROOT / "native" / "build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    try:
        return build.build_itch()
    except (OSError, subprocess.CalledProcessError) as error:
        if os.environ.get("QMB_REQUIRE_NATIVE") == "1":
            raise
        pytest.skip(f"no C++ compiler: {error}")


def header(kind, locate, ts):
    return kind.encode() + struct.pack(">HH", locate, 0) + ts.to_bytes(6, "big")


def stock(name):
    return name.ljust(8).encode()


def directory(locate, name):
    return header("R", locate, 0) + stock(name) + b"N" + b"N" + struct.pack(">I", 100) + b"N" + b"C" + b"  " \
        + b"P" + b"N" + b"N" + b"1" + b"Y" + struct.pack(">I", 0) + b"N"


def add(locate, ts, ref, side, shares, name, price):
    return header("A", locate, ts) + struct.pack(">Q", ref) + side.encode() + struct.pack(">I", shares) \
        + stock(name) + struct.pack(">I", round(price * 1e4))


def execute(locate, ts, ref, shares):
    return header("E", locate, ts) + struct.pack(">QIQ", ref, shares, 1)


def execute_price(locate, ts, ref, shares, printable, price):
    return header("C", locate, ts) + struct.pack(">QIQ", ref, shares, 2) + printable.encode() \
        + struct.pack(">I", round(price * 1e4))


def cancel(locate, ts, ref, shares):
    return header("X", locate, ts) + struct.pack(">QI", ref, shares)


def delete(locate, ts, ref):
    return header("D", locate, ts) + struct.pack(">Q", ref)


def replace(locate, ts, old, new, shares, price):
    return header("U", locate, ts) + struct.pack(">QQII", old, new, shares, round(price * 1e4))


def hidden_trade(locate, ts, shares, name, price):
    return header("P", locate, ts) + struct.pack(">Q", 0) + b"B" + struct.pack(">I", shares) + stock(name) \
        + struct.pack(">IQ", round(price * 1e4), 3)


def cross(locate, ts, shares, name, price, kind):
    return header("Q", locate, ts) + struct.pack(">Q", shares) + stock(name) + struct.pack(">IQ", round(price * 1e4), 4) \
        + kind.encode()


def run(binary, messages, tmp_path, symbols="SPY"):
    stream = b"".join(struct.pack(">H", len(m)) + m for m in messages)
    out = tmp_path / "day"
    result = subprocess.run([str(binary), "--symbols", symbols, "--out", str(out)], input=stream, capture_output=True)
    read = lambda name: pd.read_csv(f"{out}_{name}.csv")  # noqa: E731
    checks = read("checks").set_index("item")["value"]
    return result.returncode, read("summary"), read("trades"), read("mids"), checks


def day(s):
    return OPEN + int(s * NS)


def normal_day():
    return [
        directory(1, "SPY"), directory(2, "AAA"),
        add(1, day(-1), 1, "B", 300, "SPY", 100.00),
        add(1, day(-1), 2, "S", 200, "SPY", 100.02),
        add(2, day(0), 50, "B", 999, "AAA", 5.00),                      # another stock: ignored
        b"Z" + bytes(20),                                               # a type the parser does not read: skipped
        execute(1, day(10), 2, 50),                                     # incoming buy at the ask
        execute_price(1, day(12), 1, 10, "N", 100.00),                  # not printable: book only
        replace(1, day(20), 1, 3, 100, 100.01),                         # bid moves up to 100.01
        hidden_trade(1, day(25), 40, "SPY", 100.015),
        cancel(1, day(30), 3, 50),
        delete(1, day(40), 3),                                          # bid side now empty
        cross(1, day(0), 1234, "SPY", 100.01, "O"),
    ]


def test_book_trades_and_time_weighted_spread(binary, tmp_path):
    code, summary, trades, mids, checks = run(binary, normal_day(), tmp_path)
    assert code == 0
    assert checks["bad_length_total"] == 0 and checks["unknown_reference"] == 0 and checks["negative_level"] == 0
    row = summary.iloc[0]
    first = 0.01 / 100.01 * 1e4                                         # half-spread in bps, seconds 0 to 20
    second = 0.005 / 100.015 * 1e4                                      # seconds 20 to 40
    assert row.valid_seconds == pytest.approx(40.0)
    assert row.quoted_half_spread_bps == pytest.approx((first + second) / 2, rel=1e-9)
    assert row.open5_quoted_half_spread_bps == pytest.approx((first + second) / 2, rel=1e-9)
    assert row.displayed_execs == 1 and row.displayed_shares == 50
    assert row.hidden_execs == 1 and row.hidden_shares == 40
    assert row.open_cross_price == pytest.approx(100.01) and row.open_cross_shares == 1234
    buy = trades[trades.kind == "E"].iloc[0]
    assert (buy.sign, buy.price, buy.shares, buy.bid, buy.ask) == (1, 100.02, 50, 100.00, 100.02)
    assert list(trades.kind) == ["E", "P"] and trades[trades.kind == "P"].sign.iloc[0] == 0


def test_midpoint_grid_uses_the_state_at_each_whole_second(binary, tmp_path):
    _, _, _, mids, _ = run(binary, normal_day(), tmp_path)
    grid = mids.set_index("second")["mid"]
    assert grid.loc[0] == pytest.approx(100.01) and grid.loc[19] == pytest.approx(100.01)
    assert grid.loc[20] == pytest.approx(100.015) and grid.loc[39] == pytest.approx(100.015)
    assert 40 not in grid.index and grid.index.max() == 39


def test_depth_at_the_best_quotes(binary, tmp_path):
    _, summary, _, _, _ = run(binary, normal_day(), tmp_path)
    row = summary.iloc[0]
    bid = (300 * 100.00 * 12 + 290 * 100.00 * 8 + 100 * 100.01 * 10 + 50 * 100.01 * 10) / 40  # shares x price x s
    ask = (200 * 100.02 * 10 + 150 * 100.02 * 30) / 40
    assert row.bid_depth_usd == pytest.approx(bid, abs=0.01) and row.ask_depth_usd == pytest.approx(ask, abs=0.01)


@pytest.mark.parametrize("bad, item", [
    (delete(1, day(5), 999), "unknown_reference"),
    (execute(1, day(5), 2, 500), "negative_level"),
    (add(1, day(5), 7, "B", 10, "SPY", 99.0)[:-1], "bad_length_total"),
])
def test_failed_checks_stop_the_run(binary, tmp_path, bad, item):
    messages = normal_day()[:4] + [bad]
    code, _, _, _, checks = run(binary, messages, tmp_path)
    assert code == 3 and checks[item] > 0


def test_a_locked_or_crossed_book_is_not_averaged(binary, tmp_path):
    messages = [directory(1, "SPY"), add(1, day(-1), 1, "B", 100, "SPY", 100.00),
                add(1, day(-1), 2, "S", 100, "SPY", 100.02), add(1, day(10), 3, "B", 100, "SPY", 100.02),
                delete(1, day(20), 3)]
    _, summary, _, mids, checks = run(binary, messages, tmp_path)
    row = summary.iloc[0]
    assert row.crossed_or_locked_seconds == pytest.approx(10.0)
    assert row.valid_seconds == pytest.approx(23_400 - 10.0)
    assert checks["crossed_after_message"] == 1
    assert 10 not in set(mids.second) and np.isclose(mids.set_index("second").mid.loc[20], 100.01)


def test_effective_spread_impact_and_quote_rule():
    spec = importlib.util.spec_from_file_location("tick_costs", ROOT / "scripts" / "17_tick_costs.py")
    tick = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tick)
    trades = pd.DataFrame({
        "symbol": ["SPY", "SPY", "SPY", "SPY"], "ts_ns": [day(10), day(20), day(30), day(23_399)],
        "kind": ["E", "E", "P", "E"], "price": [100.02, 100.00, 100.015, 100.02], "shares": [100, 300, 50, 10],
        "sign": [1, -1, 0, 1], "bid": [100.00, 100.00, 100.00, 100.00], "ask": [100.02, 100.02, 100.02, 100.02]})
    mids = pd.DataFrame({"symbol": "SPY", "second": [70, 80, 90, 310, 320], "mid": [100.03, 99.99, 100.01, 100.05, 100.00]})
    out = tick.signed_costs(trades, mids).set_index("ts_ns")
    half = 0.01 / 100.01 * 1e4
    assert out.loc[day(10), "effective_bps"] == pytest.approx(half)
    assert out.loc[day(20), "effective_bps"] == pytest.approx(half)                 # a sell at the bid
    assert out.loc[day(30), "sign_used"] == 1                                         # hidden trade above the mid
    assert out.loc[day(10), "impact_1min_bps"] == pytest.approx(0.02 / 100.01 * 1e4)  # mid at second 70
    assert out.loc[day(20), "impact_1min_bps"] == pytest.approx(0.02 / 100.01 * 1e4)  # sell, mid fell to 99.99
    assert out.loc[day(10), "realized_5min_bps"] == pytest.approx(half - 0.04 / 100.01 * 1e4)
    assert np.isnan(out.loc[day(23_399), "impact_1min_bps"])                          # less than a minute to the close
