"""World Magnetic Model: spherical-harmonic geomagnetic field synthesis.

Implements the full WMM evaluation chain (degree/order 12, epoch 2025.0):

1. Geodetic → geocentric spherical conversion on the WGS-84 ellipsoid.
2. Schmidt semi-normalized associated Legendre functions ``P̄ₙᵐ(sin φ′)``
   and their derivatives by stable forward recursion.
3. Spherical-harmonic synthesis of the field components in the geocentric
   frame, with linear secular variation from the epoch.
4. Rotation back to the geodetic (NED) frame.

The output feeds directly into qnav's heading stack: ``declination`` for
:func:`qnav.heading.declination.apply_declination`, the NED field vector for
the ESKF/Mahony magnetometer reference, and ``inclination``/``intensity``
for the disturbance gates in :mod:`qnav.heading.disturbance`.

Accuracy: matches the official WMM2025 test values to < 0.1 nT in each
component (verified in ``tests/test_geomag.py``); the model itself is
specified to ~1° RMS declination error globally.

Conventions: latitude/longitude in **radians** (qnav-wide policy; the
official test tables are degrees — convert at the boundary), height in
meters above the WGS-84 ellipsoid, decimal year for time. Output field in
**Tesla** (the WMM native nT is converted; 1 nT = 1e-9 T).

Reference: NCEI/BGS, "The US/UK World Magnetic Model for 2025-2030",
NOAA Technical Report (2024). Coefficients in ``_wmm2025.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qnav.frames.earth import WGS84_A, WGS84_B
from qnav.geomag._wmm2025 import COEFFICIENTS, EPOCH

__all__ = ["MagneticFieldEstimate", "wmm_field", "wmm_elements"]

_MAX_DEGREE = 12
_RE = 6371200.0          # WMM geomagnetic reference radius [m]


def _build_coefficient_tables():
    """(g, h, gd, hd) as (13, 13) arrays indexed [n, m], nT and nT/yr."""
    g = np.zeros((_MAX_DEGREE + 1, _MAX_DEGREE + 1))
    h = np.zeros_like(g)
    gd = np.zeros_like(g)
    hd = np.zeros_like(g)
    for n, m, gnm, hnm, gtnm, htnm in COEFFICIENTS:
        g[n, m], h[n, m], gd[n, m], hd[n, m] = gnm, hnm, gtnm, htnm
    return g, h, gd, hd


_G, _H, _GD, _HD = _build_coefficient_tables()


def _schmidt_legendre(x: float, nmax: int):
    """Schmidt semi-normalized ``P̄ₙᵐ(x)`` and ``dP̄ₙᵐ/dθ`` (θ = colatitude).

    Forward column recursion (numerically stable for nmax ≤ ~50; WMM needs
    12). ``x = cos θ`` and ``s = sin θ ≥ 0``.
    """
    s = np.sqrt(max(1.0 - x * x, 0.0))
    P = np.zeros((nmax + 1, nmax + 1))
    dP = np.zeros_like(P)            # derivative w.r.t. θ
    P[0, 0] = 1.0
    for m in range(nmax + 1):
        if m > 0:
            # diagonal: P̄ₘₘ = s·√((2m−1)/(2m))·P̄ₘ₋₁,ₘ₋₁ (Schmidt norm)
            fac = np.sqrt((2.0 * m - 1.0) / (2.0 * m)) if m > 1 else 1.0
            P[m, m] = s * fac * P[m - 1, m - 1]
            dP[m, m] = x * fac * P[m - 1, m - 1] + s * fac * dP[m - 1, m - 1]
        if m + 1 <= nmax:
            P[m + 1, m] = x * np.sqrt(2.0 * m + 1.0) * P[m, m]
            dP[m + 1, m] = np.sqrt(2.0 * m + 1.0) * (x * dP[m, m] - s * P[m, m])
        for n in range(m + 2, nmax + 1):
            k1 = (2.0 * n - 1.0) / np.sqrt((n - m) * (n + m))
            k2 = np.sqrt(((n - 1.0) ** 2 - m * m) / ((n - m) * (n + m)))
            P[n, m] = k1 * x * P[n - 1, m] - k2 * P[n - 2, m]
            dP[n, m] = k1 * (x * dP[n - 1, m] - s * P[n - 1, m]) - k2 * dP[n - 2, m]
    return P, dP


@dataclass(frozen=True)
class MagneticFieldEstimate:
    """WMM output bundle. Field components in **Tesla**, angles in radians.

    ``ned``: field vector [X, Y, Z] in local NED. ``declination`` D (positive
    east), ``inclination`` I (positive down), ``horizontal`` H, ``total`` F.
    Secular variation ``ned_sv`` in T/year.
    """

    ned: np.ndarray
    declination: float
    inclination: float
    horizontal: float
    total: float
    ned_sv: np.ndarray


def wmm_field(
    lat: float, lon: float, h: float = 0.0, year: float = EPOCH
) -> MagneticFieldEstimate:
    """Geomagnetic field at a geodetic position and decimal year.

    Parameters: ``lat``/``lon`` [rad], ``h`` meters above the WGS-84
    ellipsoid, ``year`` decimal (e.g. 2026.45). Raises ``ValueError`` outside
    the model's validity window [2025.0, 2030.0).
    """
    if not EPOCH <= year < EPOCH + 5.0:
        raise ValueError(
            f"WMM2025 is valid for [{EPOCH}, {EPOCH + 5.0}); got year={year}"
        )
    dt_years = year - EPOCH
    g = _G + dt_years * _GD
    hcf = _H + dt_years * _HD

    # geodetic → geocentric spherical (lat′, r)
    sphi, cphi = np.sin(lat), np.cos(lat)
    rc = WGS84_A**2 * cphi**2 + WGS84_B**2 * sphi**2          # (a cosφ)²+(b sinφ)² scale
    p = np.sqrt(rc)
    r = np.sqrt(h * h + 2.0 * h * p + (WGS84_A**4 * cphi**2 + WGS84_B**4 * sphi**2) / rc)
    sphi_c = sphi * (h * p + WGS84_B**2) / (r * p)             # sin geocentric lat
    cphi_c = np.sqrt(max(1.0 - sphi_c * sphi_c, 0.0))
    lat_c = np.arcsin(np.clip(sphi_c, -1.0, 1.0))

    P, dP = _schmidt_legendre(sphi_c, _MAX_DEGREE)

    cml = np.cos(np.arange(_MAX_DEGREE + 1) * lon)
    sml = np.sin(np.arange(_MAX_DEGREE + 1) * lon)

    ar = _RE / r
    Bx_c = By_c = Bz_c = 0.0       # geocentric north, east, down
    Bx_sv = By_sv = Bz_sv = 0.0
    safe_c = cphi_c if cphi_c > 1e-12 else 1e-12
    arn = ar * ar                  # (a/r)^{n+2} starting at n=1
    for n in range(1, _MAX_DEGREE + 1):
        arn *= ar
        for m in range(0, n + 1):
            gc, hc = g[n, m], hcf[n, m]
            gs, hs = _GD[n, m], _HD[n, m]
            cs, sn = cml[m], sml[m]
            # north (−1/r ∂V/∂θ → +dP convention), east, down components
            Bx_c += -arn * (gc * cs + hc * sn) * (-dP[n, m])
            By_c += (arn * m * (gc * sn - hc * cs) * P[n, m]) / safe_c
            Bz_c += -(n + 1.0) * arn * (gc * cs + hc * sn) * P[n, m]
            Bx_sv += -arn * (gs * cs + hs * sn) * (-dP[n, m])
            By_sv += (arn * m * (gs * sn - hs * cs) * P[n, m]) / safe_c
            Bz_sv += -(n + 1.0) * arn * (gs * cs + hs * sn) * P[n, m]

    # rotate geocentric (north, east, down) to geodetic NED through the
    # latitude difference ψ = φ_c − φ
    psi = lat_c - lat
    cpsi, spsi = np.cos(psi), np.sin(psi)
    X = Bx_c * cpsi - Bz_c * spsi
    Z = Bx_c * spsi + Bz_c * cpsi
    Y = By_c
    Xs = Bx_sv * cpsi - Bz_sv * spsi
    Zs = Bx_sv * spsi + Bz_sv * cpsi
    Ys = By_sv

    ned = np.array([X, Y, Z]) * 1e-9            # nT → T
    ned_sv = np.array([Xs, Ys, Zs]) * 1e-9
    Hh = float(np.hypot(ned[0], ned[1]))
    return MagneticFieldEstimate(
        ned=ned,
        declination=float(np.arctan2(ned[1], ned[0])),
        inclination=float(np.arctan2(ned[2], Hh)),
        horizontal=Hh,
        total=float(np.linalg.norm(ned)),
        ned_sv=ned_sv,
    )


def wmm_elements(lat: float, lon: float, h: float = 0.0, year: float = EPOCH):
    """Convenience: ``(declination, inclination, total_intensity)`` —
    drop-in inputs for :func:`qnav.heading.magnetic_model.field_from_elements`."""
    f = wmm_field(lat, lon, h, year)
    return f.declination, f.inclination, f.total
