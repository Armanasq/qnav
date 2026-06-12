"""Accelerometer bias/scale calibration from multi-orientation static data.

Model: ``f_meas = (I + S) f_true + b`` with ``‖f_true‖ = g`` when static.
Static measurements therefore lie on an ellipsoid — the same mathematics as
magnetometer calibration, reused deliberately (one validated solver, two
sensors). The returned correction maps static measurements to magnitude g.

Gauge note: as with the magnetometer, a residual rotation of (I+S) is not
observable from magnitudes alone; the correction is symmetric. Use
:mod:`qnav.calibration.frame_alignment` with an external attitude reference
to resolve the rotation if needed.
"""

from __future__ import annotations

import numpy as np

from qnav.calibration.mag_ellipsoid import fit_ellipsoid
from qnav.errors import CalibrationError

__all__ = ["calibrate_accelerometer", "AccelCalibration"]


class AccelCalibration:
    """Corrective accelerometer calibration: ``f̂ = C (f − b)``."""

    def __init__(self, bias: np.ndarray, correction: np.ndarray, rms_residual: float):
        self.bias = bias
        self.correction = correction
        self.rms_residual = rms_residual

    def correct(self, f_meas: np.ndarray) -> np.ndarray:
        return (np.asarray(f_meas, dtype=float) - self.bias) @ self.correction.T


def calibrate_accelerometer(
    f_static: np.ndarray, gravity: float = 9.80665
) -> AccelCalibration:
    """Fit bias + symmetric scale correction from static specific-force
    samples ``(N, 3)`` spanning diverse orientations (≥ 9 well-spread poses
    recommended; classic six-position data works)."""
    f = np.asarray(f_static, dtype=float)
    cal = fit_ellipsoid(f)
    C = cal.soft_iron_inv * (gravity / cal.radius)
    resid = cal.rms_residual * (gravity / cal.radius)
    if cal.rms_residual / cal.radius > 0.05:
        raise CalibrationError(
            f"residual {100 * cal.rms_residual / cal.radius:.1f}% of g — data "
            "likely non-static or poorly distributed"
        )
    return AccelCalibration(bias=cal.hard_iron, correction=C, rms_residual=resid)
