"""Gyro bias estimation from static data.

The simplest, most reliable gyro calibration: average the output while the
sensor is verifiably static. Static-interval detection uses a variance gate
on both gyro and accelerometer streams.
"""

from __future__ import annotations

import numpy as np

from qnav.errors import CalibrationError

__all__ = ["detect_static_intervals", "estimate_bias"]


def detect_static_intervals(
    gyro: np.ndarray, accel: np.ndarray, dt: float,
    window: float = 1.0,
    gyro_thresh: float = np.deg2rad(0.5), accel_thresh: float = 0.05,
    gyro_mag_thresh: float = np.deg2rad(3.0),
) -> np.ndarray:
    """Boolean static-mask per sample from rolling std **and** magnitude gates.

    A sample is static when, over the centered ``window`` [s]:

    - per-axis gyro std < ``gyro_thresh`` [rad/s]  (no shaking), and
    - accelerometer-magnitude std < ``accel_thresh`` [m/s²], and
    - mean ‖ω‖ < ``gyro_mag_thresh`` [rad/s] — a *constant-rate spin* has low
      variance but is not static; the magnitude gate rejects it. Set this
      above the worst plausible bias magnitude (default 3°/s).
    """
    g = np.asarray(gyro, dtype=float)
    a = np.linalg.norm(np.asarray(accel, dtype=float), axis=-1)
    n = g.shape[0]
    m = max(int(round(window / dt)), 2)
    half = m // 2
    static = np.zeros(n, dtype=bool)
    gn = np.linalg.norm(g, axis=-1)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        if hi - lo < 2:
            continue
        if (
            np.all(np.std(g[lo:hi], axis=0) < gyro_thresh)
            and np.std(a[lo:hi]) < accel_thresh
            and np.mean(gn[lo:hi]) < gyro_mag_thresh
        ):
            static[i] = True
    return static


def estimate_bias(
    gyro: np.ndarray, static_mask: np.ndarray | None = None, min_samples: int = 50
):
    """Mean gyro output over static samples → bias estimate and its std.

    Returns ``(bias (3,), sigma (3,))`` where sigma is the standard error of
    the mean. Raises :class:`CalibrationError` with the sample count when too
    few static samples are available.
    """
    g = np.asarray(gyro, dtype=float)
    if static_mask is not None:
        g = g[np.asarray(static_mask, dtype=bool)]
    if g.shape[0] < min_samples:
        raise CalibrationError(
            f"only {g.shape[0]} static samples (< {min_samples}); collect more data"
        )
    return g.mean(axis=0), g.std(axis=0, ddof=1) / np.sqrt(g.shape[0])
