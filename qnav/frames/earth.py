"""Earth-fixed frames: WGS-84 geodesy, ECEF/NED/ENU rotations, gravity.

Conventions (``docs/conventions.md`` §6–7):

- Geodetic coordinates ``(lat, lon, h)`` in radians/meters, WGS-84 ellipsoid.
- ``R_NED_ECEF(lat, lon)`` maps ECEF coordinates to local NED, etc.
- Normal gravity from the Somigliana formula with a free-air height
  correction; ``g_NED = [0, 0, +γ]``.

References: Kok–Hol–Schön tutorial (navigation frames); standard WGS-84
defining parameters (NIMA TR8350.2). See ``docs/math/frames.md``.
"""

from __future__ import annotations

import numpy as np

from qnav.types import ScalarOrArray

from qnav.errors import ConventionError

__all__ = [
    "WGS84_A", "WGS84_F", "WGS84_B", "WGS84_E2", "WGS84_GM", "WGS84_OMEGA",
    "geodetic_to_ecef", "ecef_to_geodetic",
    "dcm_ecef_to_ned", "dcm_ecef_to_enu", "dcm_ned_to_ecef", "dcm_enu_to_ecef",
    "DCM_ENU_NED", "DCM_NED_ENU", "dcm_eci_to_ecef",
    "normal_gravity", "gravity_vector", "earth_rate_ned",
    "meridian_radius", "transverse_radius", "gaussian_radius", "transport_rate_ned",
]

# WGS-84 defining/derived parameters (NIMA TR8350.2)
WGS84_A = 6378137.0                    # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563          # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)    # semi-minor axis [m]
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)   # first eccentricity squared
WGS84_GM = 3.986004418e14              # gravitational constant [m³/s²]
WGS84_OMEGA = 7.292115e-5              # Earth rotation rate [rad/s]

#: Permutation NED→ENU coordinates (own inverse): v_ENU = DCM_ENU_NED @ v_NED.
DCM_ENU_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
DCM_NED_ENU = DCM_ENU_NED  # involutory


def geodetic_to_ecef(lat: ScalarOrArray, lon: ScalarOrArray, h: ScalarOrArray) -> np.ndarray:
    """ECEF position from geodetic ``(lat, lon, h)`` (radians, meters).

    ``N = a/√(1 − e² sin²lat)``;
    ``[ (N+h)cosφ cosλ, (N+h)cosφ sinλ, (N(1−e²)+h) sinφ ]``.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    h = np.asarray(h, dtype=float)
    sl, cl = np.sin(lat), np.cos(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
    return np.stack(
        [(N + h) * cl * np.cos(lon), (N + h) * cl * np.sin(lon),
         (N * (1.0 - WGS84_E2) + h) * sl],
        axis=-1,
    )


def ecef_to_geodetic(r_ecef: np.ndarray, max_iter: int = 10, tol: float = 1e-12):
    """Geodetic ``(lat, lon, h)`` from ECEF by Bowring-style fixed point.

    Converges to sub-millimeter in ≤ 5 iterations for terrestrial points.
    Returns a tuple of arrays (lat, lon, h).
    """
    r = np.asarray(r_ecef, dtype=float)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))  # initial (reduced) guess
    h = np.zeros_like(lat)
    for _ in range(max_iter):
        sl = np.sin(lat)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
        h_new = np.where(np.abs(np.cos(lat)) > 1e-10, p / np.cos(lat) - N,
                         z / np.where(np.abs(sl) < 1e-300, 1.0, sl) - N * (1.0 - WGS84_E2))
        lat_new = np.arctan2(z, p * (1.0 - WGS84_E2 * N / (N + h_new)))
        done = np.all(np.abs(lat_new - lat) < tol) and np.all(np.abs(h_new - h) < 1e-9)
        lat, h = lat_new, h_new
        if done:
            break
    return lat, lon, h


def dcm_ecef_to_ned(lat: ScalarOrArray, lon: ScalarOrArray) -> np.ndarray:
    """``R_NED_ECEF``: rows are the N, E, D unit vectors in ECEF coordinates."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    sφ, cφ = np.sin(lat), np.cos(lat)
    sλ, cλ = np.sin(lon), np.cos(lon)
    R = np.empty(np.broadcast(lat, lon).shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 0, 2] = -sφ * cλ, -sφ * sλ, cφ
    R[..., 1, 0], R[..., 1, 1], R[..., 1, 2] = -sλ, cλ, 0.0
    R[..., 2, 0], R[..., 2, 1], R[..., 2, 2] = -cφ * cλ, -cφ * sλ, -sφ
    return R


def dcm_ned_to_ecef(lat: ScalarOrArray, lon: ScalarOrArray) -> np.ndarray:
    """``R_ECEF_NED = R_NED_ECEFᵀ``."""
    return np.swapaxes(dcm_ecef_to_ned(lat, lon), -1, -2)


def dcm_ecef_to_enu(lat: ScalarOrArray, lon: ScalarOrArray) -> np.ndarray:
    """``R_ENU_ECEF = DCM_ENU_NED @ R_NED_ECEF``."""
    return DCM_ENU_NED @ dcm_ecef_to_ned(lat, lon)


def dcm_enu_to_ecef(lat: ScalarOrArray, lon: ScalarOrArray) -> np.ndarray:
    """``R_ECEF_ENU = R_ENU_ECEFᵀ``."""
    return np.swapaxes(dcm_ecef_to_enu(lat, lon), -1, -2)


def dcm_eci_to_ecef(gst: np.ndarray) -> np.ndarray:
    """``R_ECEF_ECI`` for Greenwich sidereal angle ``gst`` (rad): a z-rotation.

    qnav treats ECI kinematically (no precession/nutation model):
    ``v_ECEF = Rz(gst)ᵀ?`` — explicitly: ECEF axes lead ECI by gst about +z,
    so ``v_ECEF = Rz(−gst)·v_ECI`` is implemented here.
    """
    g = np.asarray(gst, dtype=float)
    c, s = np.cos(g), np.sin(g)
    R = np.zeros(np.shape(g) + (3, 3))
    R[..., 0, 0], R[..., 0, 1] = c, s
    R[..., 1, 0], R[..., 1, 1] = -s, c
    R[..., 2, 2] = 1.0
    return R


def normal_gravity(lat: ScalarOrArray, h: ScalarOrArray = 0.0) -> np.ndarray:
    """Somigliana normal gravity magnitude [m/s²] with free-air correction.

    ``γ = γ_e (1 + k sin²φ)/√(1 − e² sin²φ)`` with WGS-84 constants
    (γ_e = 9.7803253359, k = 0.00193185265241), then
    ``γ_h = γ (1 − 2h/a (1 + f + m − 2f sin²φ) + 3h²/a²)``.
    """
    lat = np.asarray(lat, dtype=float)
    h = np.asarray(h, dtype=float)
    gamma_e = 9.7803253359
    k = 0.00193185265241
    s2 = np.sin(lat) ** 2
    gamma = gamma_e * (1.0 + k * s2) / np.sqrt(1.0 - WGS84_E2 * s2)
    m = WGS84_OMEGA**2 * WGS84_A**2 * WGS84_B / WGS84_GM
    return gamma * (
        1.0 - 2.0 * h / WGS84_A * (1.0 + WGS84_F + m - 2.0 * WGS84_F * s2)
        + 3.0 * h**2 / WGS84_A**2
    )


def gravity_vector(lat: ScalarOrArray, h: ScalarOrArray = 0.0, frame: str = "NED") -> np.ndarray:
    """Gravity vector in the local tangent frame: ``[0,0,+γ]`` NED, ``[0,0,−γ]`` ENU."""
    g = normal_gravity(lat, h)
    z = np.zeros_like(g)
    if frame == "NED":
        return np.stack([z, z, g], axis=-1)
    if frame == "ENU":
        return np.stack([z, z, -g], axis=-1)
    raise ConventionError(f"frame must be 'NED' or 'ENU', got {frame!r}")


def earth_rate_ned(lat: ScalarOrArray) -> np.ndarray:
    """Earth rotation rate expressed in NED: ``ω_ie = Ω[cosφ, 0, −sinφ]``."""
    lat = np.asarray(lat, dtype=float)
    return WGS84_OMEGA * np.stack(
        [np.cos(lat), np.zeros_like(lat), -np.sin(lat)], axis=-1
    )


def meridian_radius(lat: ScalarOrArray) -> np.ndarray:
    """Meridian (north-south) radius of curvature ``M = a(1−e²)/(1−e²sin²φ)^{3/2}``.

    The radius governing latitude rate: ``φ̇ = v_N / (M + h)``.
    """
    lat = np.asarray(lat, dtype=float)
    s2 = np.sin(lat) ** 2
    return WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s2) ** 1.5


def transverse_radius(lat: ScalarOrArray) -> np.ndarray:
    """Transverse (east-west / prime-vertical) radius ``N = a/√(1−e²sin²φ)``.

    Governs longitude rate: ``λ̇ = v_E / ((N + h) cos φ)``.
    """
    lat = np.asarray(lat, dtype=float)
    return WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)


def gaussian_radius(lat: ScalarOrArray) -> np.ndarray:
    """Gaussian mean radius of curvature ``√(M·N)`` — the single-radius
    spherical approximation that matches the ellipsoid locally."""
    return np.sqrt(meridian_radius(lat) * transverse_radius(lat))


def transport_rate_ned(lat: ScalarOrArray, v_ned: np.ndarray, h: ScalarOrArray = 0.0) -> np.ndarray:
    """Transport rate ``ω_EN`` in NED: rotation of the local-level frame due
    to vehicle motion over the curved Earth.

    ``ω_EN = [v_E/(N+h), −v_N/(M+h), −v_E·tanφ/(N+h)]`` — required (with the
    Earth rate) for inertial navigation above MEMS grade; omitting it causes
    a velocity-proportional attitude drift (~v/R rad/s ≈ 0.5°/hr at 250 m/s).
    """
    lat = np.asarray(lat, dtype=float)
    v = np.asarray(v_ned, dtype=float)
    h = np.asarray(h, dtype=float)
    M = meridian_radius(lat)
    N = transverse_radius(lat)
    return np.stack(
        [
            v[..., 1] / (N + h),
            -v[..., 0] / (M + h),
            -v[..., 1] * np.tan(lat) / (N + h),
        ],
        axis=-1,
    )
