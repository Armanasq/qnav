"""Accelerometer lever-arm estimation from rigid-body kinematics.

Two accelerometers (or an accelerometer and the body reference point) on the
same rigid body differ by the lever-arm term

    a_s − a_ref = ω̇ × r + ω × (ω × r)  =  ([ω̇]× + [ω]×[ω]×) r

which is linear in the lever arm ``r``. Stacking samples gives an LSQ
problem whose excitation requires *angular acceleration and/or rotation
about at least two axes* — pure translation or constant single-axis spin
leaves parts of ``r`` unobservable, which the returned report exposes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qnav._validate import ensure_shape
from qnav.attitude import so3
from qnav.calibration.observability import (
    Observability,
    ObservabilityReport,
    assess_least_squares,
)
from qnav.errors import CalibrationError

__all__ = ["LeverArmEstimate", "estimate_lever_arm"]


@dataclass(frozen=True)
class LeverArmEstimate:
    lever_arm: np.ndarray          #: r [m], sensor position relative to reference
    covariance: np.ndarray         #: 3x3 [m^2]
    rms_residual: float            #: [m/s^2]
    observability: ObservabilityReport


def estimate_lever_arm(
    omega: np.ndarray,
    omega_dot: np.ndarray,
    accel_diff: np.ndarray,
) -> LeverArmEstimate:
    """Estimate the lever arm from synchronized samples.

    ``omega``/``omega_dot``: (N, 3) body angular rate [rad/s] and its
    derivative [rad/s²]; ``accel_diff``: (N, 3) measured ``a_sensor −
    a_reference`` [m/s²] in the same body frame. Raises
    :class:`CalibrationError` when the motion leaves the lever arm
    unobservable (report available on the exception's ``args``).
    """
    w = ensure_shape(omega, (-1, 3), "omega")
    wd = ensure_shape(omega_dot, (-1, 3), "omega_dot")
    ad = ensure_shape(accel_diff, (-1, 3), "accel_diff")
    if not (w.shape == wd.shape == ad.shape):
        raise ValueError("omega, omega_dot, accel_diff must share shape (N, 3)")
    n = w.shape[0]
    if n < 3:
        raise ValueError("need at least 3 samples")

    A = np.empty((3 * n, 3))
    for k in range(n):
        Wx = so3.hat(w[k])
        A[3 * k: 3 * k + 3] = so3.hat(wd[k]) + Wx @ Wx
    b = ad.reshape(-1)

    report = assess_least_squares(A)
    if report.status is Observability.UNOBSERVABLE:
        raise CalibrationError(
            "motion does not excite the lever arm (needs angular acceleration "
            f"or multi-axis rotation); weakest direction {report.weakest_direction}"
        )

    r, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = A @ r - b
    dof = max(3 * n - 3, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    return LeverArmEstimate(
        lever_arm=r, covariance=cov,
        rms_residual=float(np.sqrt(np.mean(resid**2))),
        observability=report,
    )
