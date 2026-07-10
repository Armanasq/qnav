"""Tilt-compensated compass heading.

Heading convention (``docs/conventions.md`` §7): compass heading ψ_c is the
clockwise angle from **north** to the body x-axis projection, in ``[0, 2π)``.
For the NED/FRD matched pair the ZYX yaw equals the heading
(``ψ_c = wrap(ψ_yaw)``).

Magnetic heading from a magnetometer (body frame FRD), after leveling with
roll/pitch (attitude survey eq. for the tilt-compensated compass):

    m_level = Ry(θ) Rx(φ) · m_B
    ψ_mag   = atan2(−m_level,y , m_level,x)

(The minus sign: in NED a heading rotation by +ψ moves north into the body
x–y plane as ``m_level = [cosψ·mN + …]``; equivalently
``ψ = atan2(−m_E', m_N')`` for the leveled components.)
"""

from __future__ import annotations

import warnings

import numpy as np

from qnav.types import ScalarOrArray

from qnav.errors import DegenerateGeometryWarning
from qnav.heading.tilt_compensation import detilt, roll_pitch_from_accel

__all__ = [
    "wrap_heading", "heading_difference", "magnetic_heading",
    "true_heading", "heading_from_accel_mag", "yaw_to_heading", "heading_to_yaw",
    "heading_variance",
]


def wrap_heading(psi: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to ``[0, 2π)``."""
    return np.mod(np.asarray(psi, dtype=float), 2.0 * np.pi)


def heading_difference(psi1: ScalarOrArray, psi2: ScalarOrArray) -> np.ndarray:
    """Signed smallest difference ``psi1 − psi2`` in ``(−π, π]``."""
    d = np.asarray(psi1, dtype=float) - np.asarray(psi2, dtype=float)
    return np.pi - np.mod(np.pi - d, 2.0 * np.pi)


def magnetic_heading(
    m_body: np.ndarray, roll: np.ndarray, pitch: np.ndarray
) -> np.ndarray:
    """Tilt-compensated magnetic heading in ``[0, 2π)`` (FRD body, NED nav).

    Degenerate when the leveled horizontal field vanishes (magnetic poles or
    fully disturbed field): warns and returns 0 there.
    """
    m_lev = detilt(m_body, roll, pitch)
    mx, my = m_lev[..., 0], m_lev[..., 1]
    h = np.hypot(mx, my)
    bad = h < 1e-12
    if np.any(bad):
        warnings.warn(
            "horizontal magnetic field is zero; heading undefined, returning 0",
            DegenerateGeometryWarning, stacklevel=2,
        )
    return wrap_heading(np.where(bad, 0.0, np.arctan2(-my, mx)))


def true_heading(psi_mag: ScalarOrArray, declination: ScalarOrArray) -> np.ndarray:
    """True heading = magnetic heading + declination (declination positive east)."""
    return wrap_heading(np.asarray(psi_mag, dtype=float) + np.asarray(declination, dtype=float))


def heading_from_accel_mag(
    f_body: np.ndarray, m_body: np.ndarray, declination: float = 0.0
):
    """Full accelerometer + magnetometer compass solution (FRD body, NED nav).

    Returns ``(roll, pitch, heading_true)``; pass ``declination=0`` for
    magnetic heading.
    """
    roll, pitch = roll_pitch_from_accel(f_body, frame="NED")
    psi = magnetic_heading(m_body, roll, pitch)
    return roll, pitch, true_heading(psi, declination)


def yaw_to_heading(yaw: np.ndarray) -> np.ndarray:
    """ZYX yaw (NED/FRD) → compass heading: same angle wrapped to [0, 2π)."""
    return wrap_heading(yaw)


def heading_to_yaw(heading: np.ndarray) -> np.ndarray:
    """Compass heading → ZYX yaw in (−π, π]."""
    return heading_difference(heading, 0.0)


def heading_variance(
    m_level_horizontal: np.ndarray, sigma_m: float
) -> np.ndarray:
    """First-order heading variance from isotropic leveled-field noise.

    For ``ψ = atan2(−m_y, m_x)`` with per-axis noise σ_m on the leveled
    horizontal components: ``var(ψ) ≈ σ_m² / ‖m_h‖²``. Inputs: horizontal
    leveled field ``(..., 2)`` and σ_m (same units).
    """
    mh = np.asarray(m_level_horizontal, dtype=float)
    h2 = np.sum(mh * mh, axis=-1)
    h2 = np.where(h2 < 1e-300, np.inf, h2)
    return (sigma_m**2) / h2
