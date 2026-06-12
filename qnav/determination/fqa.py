"""FQA: Factored Quaternion Algorithm (Yun, Bachmann, McGhee 2008).

Estimates attitude as a product of three single-axis quaternions —
pitch ⊗ roll from the accelerometer, then yaw from the magnetometer — using
**half-angle algebra only** (no trigonometric function calls; half-angle
sines/cosines come from square-root identities).

The factored structure has a property none of the optimal Wahba solvers
share: **magnetic disturbances affect only the yaw factor**. Roll and pitch
are computed entirely from the accelerometer, so a corrupted magnetometer
degrades heading but never tilt. This makes FQA the determination method of
choice when the magnetic environment is unreliable and tilt accuracy matters
(e.g. human-motion capture, the algorithm's original domain).

qnav conventions: input is the **specific force** ``f_body`` and the magnetic
field ``m_body``; ``m_ref`` is the NED field direction (defaults to
``[1, 0, 0]`` — only its *horizontal direction* matters, the vertical
component is discarded by construction). Output is ``q_NB``.

Singularity: at pitch = ±90° the roll factor is undefined (the gravity vector
carries no roll information). qnav warns (`DegenerateGeometryWarning`) and
sets roll to zero — same deterministic policy as the Euler gimbal-lock
handler.

Reference: Yun, Bachmann, McGhee, "A simplified quaternion-based algorithm
for orientation estimation from Earth gravity and magnetic field
measurements", IEEE Trans. Instrumentation and Measurement 57(3), 2008.
"""

from __future__ import annotations

import warnings

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.errors import DegenerateGeometryWarning

__all__ = ["fqa"]


def _half_angle(s: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sin θ/2, cos θ/2) from (sin θ, cos θ) via half-angle square roots."""
    s_half = np.sign(s) * np.sqrt(np.clip((1.0 - c) / 2.0, 0.0, None))
    c_half = np.sqrt(np.clip((1.0 + c) / 2.0, 0.0, None))
    return s_half, c_half


def fqa(
    f_body: np.ndarray, m_body: np.ndarray | None = None,
    m_ref: np.ndarray | None = None,
    *, singularity_tol: float = 1e-9,
) -> np.ndarray:
    """Factored attitude ``q_NB = q_yaw ⊗ q_pitch ⊗ q_roll`` (single sample).

    With ``m_body=None`` only the tilt factors are returned (yaw = 0). Raises
    ``ValueError`` on a zero-norm accelerometer sample; a zero-norm
    magnetometer sample degrades gracefully to tilt-only.
    """
    f = np.asarray(f_body, dtype=float)
    fn = np.linalg.norm(f)
    if fn < 1e-12:
        raise ValueError("zero-norm accelerometer sample")
    fx, fy, fz = f / fn

    # pitch factor: sin θ = f̂_x (specific force convention, NED/FRD)
    s_t = np.clip(fx, -1.0, 1.0)
    c_t = np.sqrt(1.0 - s_t * s_t)
    st2, ct2 = _half_angle(s_t, c_t)
    q_pitch = np.array([ct2, 0.0, st2, 0.0])

    # roll factor: undefined at |pitch| = 90° (c_t = 0)
    if c_t < singularity_tol:
        warnings.warn(
            "FQA singularity: |pitch| = 90 deg, roll unobservable; set to 0",
            DegenerateGeometryWarning, stacklevel=2,
        )
        q_roll = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        s_p = np.clip(-fy / c_t, -1.0, 1.0)
        c_p = np.clip(-fz / c_t, -1.0, 1.0)
        # antipodal tie-break: at c_p = −1, s_p = 0 the sign is arbitrary
        sign_sp = np.sign(s_p) if not (c_p == -1.0 and s_p == 0.0) else 1.0
        sp2 = sign_sp * np.sqrt(np.clip((1.0 - c_p) / 2.0, 0.0, None))
        cp2 = np.sqrt(np.clip((1.0 + c_p) / 2.0, 0.0, None))
        q_roll = np.array([cp2, sp2, 0.0, 0.0])

    q_tilt = quat.mul(q_pitch, q_roll)
    if m_body is None:
        return quat.canonical(q_tilt)
    m = np.asarray(m_body, dtype=float)
    mn = np.linalg.norm(m)
    if mn < 1e-12:
        return quat.canonical(q_tilt)

    # yaw factor: de-tilt the magnetic reading, align its horizontal
    # projection with the horizontal projection of the reference
    m_lvl = quat.rotate_vector(q_tilt, m / mn)        # tilt-corrected, nav axes
    h = m_lvl[:2]
    hn = np.linalg.norm(h)
    if hn < 1e-12:
        warnings.warn(
            "magnetic field is vertical after de-tilt: yaw unobservable",
            DegenerateGeometryWarning, stacklevel=2,
        )
        return quat.canonical(q_tilt)
    mx, my = h / hn

    ref = np.array([1.0, 0.0, 0.0]) if m_ref is None else np.asarray(m_ref, dtype=float)
    rn = np.linalg.norm(ref[:2])
    if rn < 1e-12:
        raise ValueError("m_ref has no horizontal component: yaw reference undefined")
    nx, ny = ref[:2] / rn

    c_y = np.clip(mx * nx + my * ny, -1.0, 1.0)
    s_y = -my * nx + mx * ny
    sy2, cy2 = _half_angle(s_y, c_y)
    q_yaw = np.array([cy2, 0.0, 0.0, sy2])
    return quat.canonical(quat.mul(q_yaw, q_tilt))
