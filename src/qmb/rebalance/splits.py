"""Train/test index ranges over a case's OOS weeks (design/preregistration.md)."""
import numpy as np
from .cases import SPLIT
from . import env


def split_indices(case):
    """Returns lo, hi (training window [lo, hi): first fully-defined state week .. split-1, last train week dropped)
    and te0, te1 (test span [te0, te1))."""
    split = int(np.searchsorted(case["signal_date"].astype("datetime64[ns]"), np.datetime64(SPLIT[case["dataset"]])))
    lc = env.last_completed(case)
    lo = int(np.argmax(lc >= 51))            # ic52 (the longest look-back) defined
    return lo, split - 1, split, len(case["W"])
