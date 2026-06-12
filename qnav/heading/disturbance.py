"""Magnetic disturbance detection and heading fallback policy.

A magnetometer measurement is *trustworthy* only if it is consistent with the
expected local field. qnav tests three invariants (each with an explicit
threshold):

1. **Magnitude**: ``| ‖m‖ − B_ref | ≤ tol_B`` — ferromagnetic disturbances
   change field strength.
2. **Dip angle**: with roll/pitch from the accelerometer, the angle between
   the leveled field and the horizontal must match the reference inclination:
   ``| I_meas − I_ref | ≤ tol_I``.
3. **Innovation rate** (stateful): heading jumps faster than the gyro
   indicates are flagged by :class:`HeadingMonitor`.

References: standard practice in adaptive AHRS literature (attitude survey,
disturbance-rejection discussion).
"""

from __future__ import annotations

import numpy as np

from qnav.heading.compass import heading_difference
from qnav.heading.tilt_compensation import detilt

__all__ = ["magnitude_check", "dip_check", "is_field_trustworthy", "HeadingMonitor"]


def magnitude_check(
    m_body: np.ndarray, ref_intensity: float, tol: float
) -> np.ndarray:
    """True where ``|‖m‖ − B_ref| ≤ tol`` (units of the input field)."""
    return np.abs(np.linalg.norm(np.asarray(m_body, dtype=float), axis=-1) - ref_intensity) <= tol


def dip_check(
    m_body: np.ndarray, roll: np.ndarray, pitch: np.ndarray,
    ref_inclination: float, tol: float,
) -> np.ndarray:
    """True where the measured dip angle matches the reference within ``tol`` [rad]."""
    m_lev = detilt(m_body, roll, pitch)
    dip = np.arctan2(m_lev[..., 2], np.hypot(m_lev[..., 0], m_lev[..., 1]))
    return np.abs(dip - ref_inclination) <= tol


def is_field_trustworthy(
    m_body: np.ndarray, roll: np.ndarray, pitch: np.ndarray,
    ref_intensity: float, ref_inclination: float,
    tol_intensity: float, tol_inclination: float,
) -> np.ndarray:
    """Combined magnitude + dip gate (logical AND)."""
    return magnitude_check(m_body, ref_intensity, tol_intensity) & dip_check(
        m_body, roll, pitch, ref_inclination, tol_inclination
    )


class HeadingMonitor:
    """Stateful gyro-consistency gate with gyro-integrated fallback heading.

    Between accepted magnetic headings, heading is propagated with the
    body-z (down) component of the leveled angular rate. A new magnetic
    heading is accepted only if it agrees with the propagated heading within
    ``gate`` radians; otherwise the propagated (fallback) heading is used and
    ``last_rejected`` is set.

    This is a deliberately simple, fully deterministic policy — a building
    block, not a replacement for a proper estimator (see ``qnav.filters``).
    """

    def __init__(self, psi0: float, gate: float = np.deg2rad(15.0)) -> None:
        self.psi = float(psi0)
        self.gate = float(gate)
        self.last_rejected = False

    def update(
        self, psi_mag: float | None, yaw_rate: float, dt: float
    ) -> float:
        """Advance by ``dt`` with leveled yaw rate; fuse ``psi_mag`` if it passes.

        ``psi_mag=None`` means "no magnetic measurement this step".
        Returns the current heading estimate in [0, 2π).
        """
        self.psi = float(np.mod(self.psi + yaw_rate * dt, 2.0 * np.pi))
        self.last_rejected = False
        if psi_mag is not None:
            innov = float(heading_difference(psi_mag, self.psi))
            if abs(innov) <= self.gate:
                self.psi = float(np.mod(psi_mag, 2.0 * np.pi))
            else:
                self.last_rejected = True
        return self.psi
