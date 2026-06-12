"""Roll/pitch from accelerometer and tilt compensation of body vectors.

Sensor model (rest or low dynamics): the accelerometer measures specific
force ``f_B = R_BN (a_N − g_N)``; at rest in NED (``g_N = [0,0,+g]``) this is
``f_B = −R_BN g_N``, i.e. **+1 g pointing up** in body coordinates when level.

With intrinsic ZYX attitude ``R_NB = Rz(ψ)Ry(θ)Rx(φ)``:

    f_B = g·[sinθ, −cosθ sinφ, −cosθ cosφ]

giving the standard tilt equations (attitude survey, ``__data/Efficient
Attitude Estimators .../attitudesurvey.tex``):

    φ = atan2(−f_y, −f_z)
    θ = atan2( f_x, √(f_y² + f_z²) )

Roll is defined for all attitudes except free fall; pitch is in [−π/2, π/2].
"""

from __future__ import annotations

import warnings

import numpy as np

from qnav.attitude import dcm as _dcm
from qnav.errors import DegenerateGeometryWarning

__all__ = ["roll_pitch_from_accel", "detilt", "tilt_dcm"]


def roll_pitch_from_accel(f_body: np.ndarray, frame: str = "NED") -> tuple:
    """Roll φ and pitch θ (radians) from specific force ``f_body`` (FRD body).

    ``frame`` declares the navigation frame the angles refer to ("NED" with
    FRD body, or "ENU" with FLU body — the equations coincide for matched
    pairs). Near free fall (‖f‖ ≈ 0) a :class:`DegenerateGeometryWarning` is
    issued and angles default to zero.
    """
    f = np.asarray(f_body, dtype=float)
    if frame not in ("NED", "ENU"):
        raise ValueError(f"frame must be 'NED' or 'ENU', got {frame!r}")
    if frame == "ENU":
        # FLU body with ENU nav: f_B = g[−sinθ?]; matched-pair equations are
        # identical after axis relabeling done by the caller's convention;
        # qnav requires FRD components here, so convert FLU → FRD first.
        from qnav.frames.conventions import flu_to_frd
        f = flu_to_frd(f)
    n = np.linalg.norm(f, axis=-1)
    bad = n < 1e-6
    if np.any(bad):
        warnings.warn(
            "specific force near zero (free fall); roll/pitch undefined, returning 0",
            DegenerateGeometryWarning, stacklevel=2,
        )
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    roll = np.where(bad, 0.0, np.arctan2(-fy, -fz))
    pitch = np.where(bad, 0.0, np.arctan2(fx, np.hypot(fy, fz)))
    return roll, pitch


def tilt_dcm(roll: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Partial attitude ``R_N'B = Ry(θ) Rx(φ)`` (navigation frame up to yaw)."""
    return _dcm.rot_y(pitch) @ _dcm.rot_x(roll)


def detilt(v_body: np.ndarray, roll: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Rotate a body vector into the **leveled** (yaw-free navigation) frame:
    ``v_level = Ry(θ) Rx(φ) v_B``. Used for tilt-compensated compassing."""
    v = np.asarray(v_body, dtype=float)
    return np.einsum("...ij,...j->...i", tilt_dcm(roll, pitch), v)
