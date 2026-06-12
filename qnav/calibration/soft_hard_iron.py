"""Hard/soft-iron parameter extraction (semantic layer over the ellipsoid fit).

:func:`qnav.calibration.mag_ellipsoid.fit_ellipsoid` returns the corrective
map; this module converts between corrective and *generative* parameters
(:class:`qnav.sensors.magnetometer.MagnetometerModel`) given the known local
field intensity, and quantifies calibration quality.
"""

from __future__ import annotations

import numpy as np

from qnav.calibration.mag_ellipsoid import MagCalibration, fit_ellipsoid
from qnav.sensors.magnetometer import MagnetometerModel

__all__ = ["calibration_to_model", "calibrate", "quality_report"]


def calibration_to_model(
    cal: MagCalibration, field_intensity: float
) -> MagnetometerModel:
    """Generative model from a corrective calibration.

    With gauge ``A = A_sym`` (symmetric soft iron — the rotation part is
    unobservable): ``A = S⁻¹ / (B·radius_gauge)`` where S maps onto the unit
    sphere; concretely ``A = inv(S)/B`` normalized so that corrected
    measurements have magnitude B.
    """
    A = np.linalg.inv(cal.soft_iron_inv) / field_intensity
    return MagnetometerModel(hard_iron=cal.hard_iron.copy(), soft_iron=A)


def calibrate(m_meas: np.ndarray, field_intensity: float | None = None):
    """One-call calibration: fit + scale to physical units.

    Returns ``(cal, correct_fn)`` where ``correct_fn(m)`` yields corrected
    measurements with magnitude ``field_intensity`` (or unit magnitude if
    None — direction-only use is fully supported).
    """
    cal = fit_ellipsoid(m_meas)
    k = 1.0 if field_intensity is None else field_intensity

    def correct_fn(m: np.ndarray) -> np.ndarray:
        return cal.correct(m) * (k / cal.radius)

    return cal, correct_fn


def quality_report(cal: MagCalibration) -> dict:
    """Quality metrics: relative sphere residual and soft-iron anisotropy
    (ratio of extreme eigenvalues of the corrective map; 1 = none)."""
    w = np.linalg.eigvalsh(cal.soft_iron_inv)
    return {
        "relative_residual": cal.rms_residual / cal.radius,
        "anisotropy": float(w[-1] / w[0]),
        "hard_iron_norm": float(np.linalg.norm(cal.hard_iron)),
    }
