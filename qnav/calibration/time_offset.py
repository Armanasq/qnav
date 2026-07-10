"""Sensor time-offset estimation by normalized cross-correlation.

Estimates the constant clock offset between two sensors observing the same
motion (e.g. gyro rate magnitude vs. differentiated external orientation,
or two IMUs on the same rigid body). Signals are resampled to a common
uniform grid, mean-removed, cross-correlated, and the peak is refined by
parabolic interpolation — resolution is not limited to the grid spacing.

The result carries a peak-correlation quality metric; a low peak means the
signals do not share enough excitation and the offset is not trustworthy
(the estimate is still returned, the caller decides using ``reliable``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qnav._validate import ensure_finite, ensure_monotonic, ensure_positive

__all__ = ["TimeOffsetEstimate", "estimate_time_offset"]


@dataclass(frozen=True)
class TimeOffsetEstimate:
    """``t_b + offset ≈ t_a``: add ``offset`` to B's stamps to align with A."""

    offset: float
    peak_correlation: float   #: normalized, in [-1, 1]
    reliable: bool            #: peak_correlation >= min_correlation
    grid_dt: float


def estimate_time_offset(
    t_a: np.ndarray, x_a: np.ndarray,
    t_b: np.ndarray, x_b: np.ndarray,
    *,
    max_offset: float = 0.5,
    grid_dt: float | None = None,
    min_correlation: float = 0.5,
) -> TimeOffsetEstimate:
    """Estimate the clock offset of stream B relative to stream A.

    ``x_a``/``x_b`` are scalar excitation signals (e.g. ``|omega|``) sampled
    at ``t_a``/``t_b`` [s]. ``max_offset`` bounds the search; ``grid_dt``
    defaults to the finer of the two median sample intervals.
    """
    t_a = ensure_monotonic(ensure_finite(t_a, "t_a"), "t_a")
    t_b = ensure_monotonic(ensure_finite(t_b, "t_b"), "t_b")
    x_a = ensure_finite(x_a, "x_a")
    x_b = ensure_finite(x_b, "x_b")
    if t_a.shape != x_a.shape or t_b.shape != x_b.shape:
        raise ValueError("time and signal arrays must have matching shapes")
    if t_a.size < 8 or t_b.size < 8:
        raise ValueError("need at least 8 samples per stream")
    max_offset = ensure_positive(max_offset, "max_offset")

    if float(np.std(x_a)) < 1e-12 or float(np.std(x_b)) < 1e-12:
        raise ValueError("signals have (near-)zero variance: no excitation")

    if grid_dt is None:
        grid_dt = float(min(np.median(np.diff(t_a)), np.median(np.diff(t_b))))
    grid_dt = ensure_positive(grid_dt, "grid_dt")

    lo = max(t_a[0], t_b[0]) - max_offset
    hi = min(t_a[-1], t_b[-1]) + max_offset
    if hi - lo < 4 * grid_dt:
        raise ValueError("streams do not overlap enough for correlation")
    grid = np.arange(lo, hi, grid_dt)
    a = np.interp(grid, t_a, x_a, left=0.0, right=0.0)
    b = np.interp(grid, t_b, x_b, left=0.0, right=0.0)
    a = a - a.mean()
    b = b - b.mean()

    max_lag = int(np.ceil(max_offset / grid_dt))
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty(lags.size)
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom < 1e-300:
        raise ValueError("signals have (near-)zero variance: no excitation")
    for i, m in enumerate(lags):
        if m >= 0:
            corr[i] = np.sum(a[m:] * b[: b.size - m])
        else:
            corr[i] = np.sum(a[:m] * b[-m:])
    corr /= denom

    k = int(np.argmax(corr))
    peak = float(corr[k])
    # parabolic sub-sample refinement (guard the window edges)
    lag: float = float(lags[k])
    if 0 < k < lags.size - 1:
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        d = y0 - 2 * y1 + y2
        if abs(d) > 1e-15:
            lag += 0.5 * float((y0 - y2) / d)
    # positive lag means B lags A by lag*grid_dt: t_b + offset aligns to t_a
    offset = lag * grid_dt
    return TimeOffsetEstimate(
        offset=float(offset), peak_correlation=peak,
        reliable=peak >= min_correlation, grid_dt=grid_dt,
    )
