"""Local magnetic-field models.

qnav deliberately does **not** bundle full WMM/IGRF coefficient tables (they
are versioned, licensed datasets with expiry dates). It provides:

- :func:`field_from_elements` — exact local field from declination D,
  inclination I, and intensity B (the quantities a WMM lookup returns);
- :func:`elements_from_field` — the inverse;
- :func:`dipole_field` — a centered tilted-dipole model adequate for
  simulation and algorithm validation.

Conventions: D positive **east**, I positive **down** (toward +z in NED).
``m_NED = B·[cos I cos D, cos I sin D, sin I]``.
"""

from __future__ import annotations

import numpy as np

from qnav.errors import ConventionError
from qnav.frames.earth import DCM_ENU_NED

__all__ = ["field_from_elements", "elements_from_field", "dipole_field", "EARTH_DIPOLE_MOMENT"]

#: Earth's magnetic dipole moment magnitude [T·m³] (≈ 7.94e15 ≈ m·µ0/4π form below).
EARTH_DIPOLE_MOMENT = 7.94e22  # [A·m²]
_MU0_4PI = 1e-7  # [T·m/A]


def field_from_elements(
    declination: np.ndarray, inclination: np.ndarray, intensity: np.ndarray = 1.0,
    frame: str = "NED",
) -> np.ndarray:
    """Local field vector from (D, I, B). Radians; intensity in caller's units."""
    D = np.asarray(declination, dtype=float)
    I = np.asarray(inclination, dtype=float)
    B = np.asarray(intensity, dtype=float)
    m_ned = np.stack(
        [B * np.cos(I) * np.cos(D), B * np.cos(I) * np.sin(D), B * np.sin(I)], axis=-1
    )
    if frame == "NED":
        return m_ned
    if frame == "ENU":
        return m_ned @ DCM_ENU_NED.T
    raise ConventionError(f"frame must be 'NED' or 'ENU', got {frame!r}")


def elements_from_field(m: np.ndarray, frame: str = "NED"):
    """(declination, inclination, intensity) from a local field vector."""
    m = np.asarray(m, dtype=float)
    if frame == "ENU":
        m = m @ DCM_ENU_NED.T
    elif frame != "NED":
        raise ConventionError(f"frame must be 'NED' or 'ENU', got {frame!r}")
    B = np.linalg.norm(m, axis=-1)
    D = np.arctan2(m[..., 1], m[..., 0])
    I = np.arctan2(m[..., 2], np.hypot(m[..., 0], m[..., 1]))
    return D, I, B


def dipole_field(lat_mag: np.ndarray, r: np.ndarray = 6371008.8) -> np.ndarray:
    """Centered-dipole field in NED at geomagnetic latitude ``lat_mag`` [rad].

    ``B_N = −(µ0 m / 4π r³)·cos λ_m``? — explicit standard result:
    horizontal (north) component ``B_h = (µ0 m/4π r³) cos λ_m`` and vertical
    (down) ``B_z = 2(µ0 m/4π r³) sin λ_m``; declination is zero in the
    geomagnetic frame. Returns ``(..., 3)`` NED vector in Tesla.
    """
    lam = np.asarray(lat_mag, dtype=float)
    r = np.asarray(r, dtype=float)
    B0 = _MU0_4PI * EARTH_DIPOLE_MOMENT / r**3
    return np.stack(
        [B0 * np.cos(lam), np.zeros_like(lam + r), 2.0 * B0 * np.sin(lam)], axis=-1
    )
