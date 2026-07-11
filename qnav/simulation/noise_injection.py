"""Measurement-stream degradation: dropout, timestamp jitter, outliers.

All functions are deterministic given the supplied generator and never
mutate inputs — degrade copies of clean datasets to build reproducible
robustness benchmarks.
"""

from __future__ import annotations

import numpy as np

__all__ = ["dropout_mask", "apply_dropout", "jitter_timestamps", "inject_outliers"]


def dropout_mask(
    n: int, rate: float, rng: np.random.Generator, min_gap: int = 0
) -> np.ndarray:
    """Boolean keep-mask with i.i.d. drop probability ``rate``.

    ``min_gap > 0`` additionally forces at least that many kept samples
    between drops (modelling burst-free loss).
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must be in [0, 1)")
    keep = rng.random(n) >= rate
    if min_gap > 0:
        last_drop = -min_gap - 1
        for i in range(n):
            if not keep[i]:
                if i - last_drop <= min_gap:
                    keep[i] = True
                else:
                    last_drop = i
    return keep


def apply_dropout(x: np.ndarray, keep: np.ndarray, fill: str = "nan") -> np.ndarray:
    """Apply a keep-mask: dropped samples become NaN (``fill='nan'``) or hold
    the previous value (``fill='hold'``)."""
    x = np.asarray(x, dtype=float).copy()
    if fill == "nan":
        x[~keep] = np.nan
    elif fill == "hold":
        for i in range(1, x.shape[0]):
            if not keep[i]:
                x[i] = x[i - 1]
    else:
        raise ValueError("fill must be 'nan' or 'hold'")
    return x


def jitter_timestamps(
    t: np.ndarray, sigma: float, rng: np.random.Generator, keep_monotonic: bool = True
) -> np.ndarray:
    """Add Gaussian jitter to timestamps; optionally enforce strict monotonicity
    by sorting (documenting that real systems may also reorder)."""
    t = np.asarray(t, dtype=float) + sigma * rng.standard_normal(np.shape(t))
    return np.sort(t) if keep_monotonic else t


def inject_outliers(
    x: np.ndarray, rate: float, magnitude: float, rng: np.random.Generator
) -> np.ndarray:
    """Replace a fraction ``rate`` of samples with ``x + U(−mag, mag)`` spikes."""
    x = np.asarray(x, dtype=float).copy()
    n = x.shape[0]
    hit = np.asarray(rng.random(n) < rate)
    x[hit] += rng.uniform(-magnitude, magnitude, size=(int(hit.sum()),) + x.shape[1:])
    return x
