"""ctypes bindings for the C++ kernels in native/qmb_kernels.cpp.

The kernels replace two interpreted loops of the project: the index walk of the stationary
bootstrap and the week-by-week portfolio accounting of the rebalancing rules. qmb.stats and
qmb.rebalance.env stay the reference implementations; tests/test_native.py requires agreement to 1e-12.
The library is optional: nothing else in the package imports this module.
"""
from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .rebalance.cases import BORROW_BPS, COST_BPS
from .stats import PER_YEAR

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 256                     # bootstrap draws handed to the kernel per call

_library = None


class NativeUnavailable(RuntimeError):
    """The shared library is missing and could not be built (for example, no C++ compiler)."""


def _builder():
    spec = importlib.util.spec_from_file_location("qmb_native_build", ROOT / "native" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load() -> ctypes.CDLL:
    """Load the kernels, compiling them first when the library is absent or older than its source."""
    global _library
    if _library is not None:
        return _library
    builder = _builder()
    stale = (not builder.LIBRARY.exists()
             or builder.LIBRARY.stat().st_mtime < builder.SOURCE.stat().st_mtime)
    if stale:
        try:
            builder.build()
        except (OSError, subprocess.CalledProcessError) as error:
            raise NativeUnavailable(f"could not build {builder.LIBRARY.name}: {error}") from error
    lib = ctypes.CDLL(str(builder.LIBRARY))

    f64 = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    i64 = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
    u8 = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
    n = ctypes.c_int64
    lib.qmb_sb_indices.argtypes = [n, n, i64, u8, i64, i64]
    lib.qmb_sb_sharpe.argtypes = [f64, n, n, i64, u8, i64, ctypes.c_double, f64]
    lib.qmb_simulate.argtypes = [n, n, f64, f64, u8, n, n, ctypes.c_int, f64,
                                 ctypes.c_double, ctypes.c_double, f64, ctypes.c_void_p]
    for function in (lib.qmb_sb_indices, lib.qmb_sb_sharpe, lib.qmb_simulate):
        function.restype = ctypes.c_int
    _library = lib
    return lib


def _check(status: int, name: str) -> None:
    if status:
        reason = {1: "invalid argument", 2: "index out of range"}.get(status, f"status {status}")
        raise ValueError(f"{name}: {reason}")


def _bootstrap_draws(n: int, mean_block: float, draws: int, seed: int):
    """Chunks of (start, restart, jump), drawn in the order of qmb.stats.stationary_bootstrap_sharpe."""
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    for first in range(0, draws, CHUNK):
        m = min(CHUNK, draws - first)
        start = np.empty(m, np.int64)
        restart = np.empty((m, n), np.uint8)
        jump = np.empty((m, n), np.int64)
        for d in range(m):
            start[d] = rng.integers(n)
            restart[d] = rng.random(n) < p
            jump[d] = rng.integers(n, size=n)
        yield first, m, start, restart, jump


def bootstrap_indices(n: int, mean_block: float, draws: int = 2000, seed: int = 0) -> np.ndarray:
    """(draws, n) resampled week indices; the same weeks can then be used for every series compared."""
    lib = load()
    out = np.empty((draws, n), np.int64)
    for first, m, start, restart, jump in _bootstrap_draws(n, mean_block, draws, seed):
        _check(lib.qmb_sb_indices(n, m, start, restart, jump, out[first:first + m]), "qmb_sb_indices")
    return out


def bootstrap_sharpes(r, mean_block: float = 8, draws: int = 2000, seed: int = 0) -> np.ndarray:
    """Annualized Sharpe ratio of each stationary-bootstrap resample of r."""
    lib = load()
    x = np.ascontiguousarray(pd.Series(r).dropna().to_numpy(float))
    out = np.empty(draws)
    for first, m, start, restart, jump in _bootstrap_draws(len(x), mean_block, draws, seed):
        _check(lib.qmb_sb_sharpe(x, len(x), m, start, restart, jump, float(PER_YEAR), out[first:first + m]),
               "qmb_sb_sharpe")
    return out


def stationary_bootstrap_sharpe(r, mean_block: float = 8, draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the annualized Sharpe ratio; same arguments and result as qmb.stats.stationary_bootstrap_sharpe."""
    out = bootstrap_sharpes(r, mean_block, draws, seed)
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def simulate(W, R, M, t0: int, t1: int, eta_const=None, band=None,
             cost_bps: float = COST_BPS, borrow_bps: float = BORROW_BPS) -> dict:
    """Weeks t0..t1-1 of qmb.rebalance.env.simulate for a constant adjustment speed or a no-trade band.

    W is the (T, N) array of target weights to trade towards (the case's targets, or the output of a
    rank buffer or of signal smoothing). Returns weekly net, gross, turnover and short, plus the
    holdings after the last trade.
    """
    if (eta_const is None) == (band is None):
        raise ValueError("exactly one of eta_const and band is required")
    lib = load()
    W = np.ascontiguousarray(W, np.float64)
    R = np.ascontiguousarray(R, np.float64)
    M = np.ascontiguousarray(M, np.uint8)
    if not (W.ndim == 2 and W.shape == R.shape == M.shape):
        raise ValueError("W, R and M must be (T, N) arrays of the same shape")
    T, N = W.shape
    steps = max(int(t1) - int(t0), 0)
    value = eta_const if band is None else band
    param = np.ascontiguousarray(np.broadcast_to(np.asarray(value, np.float64), (steps,)))
    out = np.empty((steps, 4))
    holdings = np.empty(N)
    status = lib.qmb_simulate(T, N, W, R, M, int(t0), int(t1), 0 if band is None else 1, param,
                              float(cost_bps), float(borrow_bps), out, holdings.ctypes.data_as(ctypes.c_void_p))
    _check(status, "qmb_simulate")
    return {"net": out[:, 0].copy(), "gross": out[:, 1].copy(), "turnover": out[:, 2].copy(),
            "short": out[:, 3].copy(), "holdings": holdings}
