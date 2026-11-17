"""Longer label horizons: the target spans the right weeks and the purge covers the whole label."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qmb import engine  # noqa: E402


def setup(seed=0, n_assets=20, n_days=1600):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    r = pd.DataFrame(rng.normal(0, 0.01, (n_days, n_assets)), index=dates, columns=[f"A{i}" for i in range(n_assets)])
    tri = (1 + r).cumprod()
    sch = engine.weekly_schedule(dates)
    return tri, sch, engine.build_panel(tri, tri, engine.daily_features(tri), sch)


def test_one_week_horizon_is_the_part_1_target():
    tri, _, panel = setup()
    pd.testing.assert_series_equal(engine.horizon_target(panel, tri, 1), panel.y, check_names=False)


@pytest.mark.parametrize("horizon", [4, 13])
def test_target_spans_entry_to_the_exit_of_the_last_label_week(horizon):
    tri, sch, panel = setup()
    k = 150
    target = engine.horizon_target(panel, tri, horizon)
    realized = tri.loc[sch.exit[k + horizon - 1]] / tri.loc[sch.entry[k]] - 1
    expected = (realized.rank() - 0.5) / len(realized) - 0.5
    np.testing.assert_allclose(target.loc[k].to_numpy(), expected.reindex(target.loc[k].index).to_numpy())
    last = panel.X.index.get_level_values("week").max()
    assert target.loc[last - horizon + 2:].isna().all()        # labels that would end after the sample
    assert target.loc[:last - horizon + 1].notna().all()


@pytest.mark.parametrize("horizon", [1, 4, 13])
def test_predictions_do_not_depend_on_prices_after_the_signal_date(horizon):
    """Rewrite every price after the signal date of week k; predictions up to week k must not move."""
    tri, sch, panel = setup(seed=1)
    k = 200
    changed = tri.copy()
    after = changed.index > sch.signal[k]
    rng = np.random.default_rng(9)
    changed.loc[after] = changed.loc[after] * np.exp(np.cumsum(rng.normal(0, 0.05, (after.sum(), tri.shape[1])), axis=0))
    panel2 = engine.build_panel(changed, changed, engine.daily_features(changed), sch)
    kwargs = dict(signal="ridge", initial_weeks=60, label_weeks=horizon)
    a = engine.walk_forward(panel, target=engine.horizon_target(panel, tri, horizon), **kwargs)
    b = engine.walk_forward(panel2, target=engine.horizon_target(panel2, changed, horizon), **kwargs)
    weeks = a.index.get_level_values("week")
    block_start = weeks.min() + ((k - weeks.min()) // 4) * 4       # refit block that contains week k
    keep = weeks <= k
    np.testing.assert_allclose(a[keep].to_numpy(), b[keep].to_numpy(), atol=1e-12)
    later = weeks >= block_start + 4 * 5
    assert not np.allclose(a[later].to_numpy(), b[later].to_numpy())


def test_a_purge_that_ignores_the_label_length_would_leak():
    """With a 13-week label, treating it as a one-week label lets training see returns after the signal date."""
    tri, sch, panel = setup(seed=2)
    k = 200
    changed = tri.copy()
    after = changed.index > sch.signal[k]
    changed.loc[after] = changed.loc[after] * 1.5
    changed.loc[after, changed.columns[:10]] *= 2.0
    panel2 = engine.build_panel(changed, changed, engine.daily_features(changed), sch)
    wrong = dict(signal="ridge", initial_weeks=60, label_weeks=1)
    a = engine.walk_forward(panel, target=engine.horizon_target(panel, tri, 13), **wrong)
    b = engine.walk_forward(panel2, target=engine.horizon_target(panel2, changed, 13), **wrong)
    weeks = a.index.get_level_values("week")
    block = (weeks <= k) & (weeks > k - 4)
    assert not np.allclose(a[block].to_numpy(), b[block].to_numpy())
