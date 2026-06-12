"""SAAM: Super-fast Attitude from Accelerometer and Magnetometer (Wu et al. 2018).

A fully closed-form quaternion solution of the two-vector Wahba problem with
the gravity/magnetic pair, requiring no eigendecomposition, no iteration, and
no trigonometric calls — only one square root. It exploits the structure of
the NED references: gravity is purely vertical and the magnetic reference can
be written ``[m_N, 0, m_D]`` with the horizontal/vertical split recovered
*from the measurements themselves* (``m_D = ĝ_b · m̂_b`` is frame-invariant),
so no a-priori magnetic dip angle is needed.

qnav conventions: input is the **specific force** ``f_body`` (at rest
``f = −g``; any norm) and the magnetic field ``m_body`` (any norm); output is
``q_NB`` (NED-from-body) with ``v_N = R(q_NB) v_B``.

Accuracy is identical to QUEST/Davenport for the noise-free two-vector case
(it solves the same optimality conditions); under noise it weights the two
observations implicitly and is marginally less optimal but ~10× faster.

Reference: Wu, Zhou, Fourati, "Super-fast attitude determination of an
accelerometer-magnetometer combination", IEEE Sensors Letters (2018).
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat

__all__ = ["saam"]


def saam(f_body: np.ndarray, m_body: np.ndarray) -> np.ndarray:
    """Closed-form attitude ``q_NB`` from one accelerometer + magnetometer pair.

    Vectorized: ``f_body``, ``m_body`` of shape ``(..., 3)`` → ``(..., 4)``.
    Raises ``ValueError`` on zero-norm inputs.

    Derivation sketch: with ``a = −f̂`` (gravity-down direction in body axes)
    and ``m̂``, the NED references are ``r_a = [0,0,1]`` and
    ``r_m = [m_N, 0, m_D]`` where ``m_D = a·m̂`` and ``m_N = √(1−m_D²)``.
    Substituting into Davenport's eigenvalue problem and exploiting
    ``λ_max = 1`` for consistent measurements yields a linear closed form for
    the optimal quaternion.
    """
    f = np.asarray(f_body, dtype=float)
    m = np.asarray(m_body, dtype=float)
    fn = np.linalg.norm(f, axis=-1, keepdims=True)
    mn = np.linalg.norm(m, axis=-1, keepdims=True)
    if np.any(fn < 1e-12) or np.any(mn < 1e-12):
        raise ValueError("zero-norm accelerometer or magnetometer sample")
    a = -f / fn                      # gravity-down direction in body axes
    mh = m / mn
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    mx, my, mz = mh[..., 0], mh[..., 1], mh[..., 2]

    m_d = np.sum(a * mh, axis=-1)            # vertical (down) field fraction
    m_n = np.sqrt(np.clip(1.0 - m_d * m_d, 0.0, None))  # horizontal fraction

    qw = ax * my - ay * (m_n + mx)
    qx = (az - 1.0) * (m_n + mx) + ax * (m_d - mz)
    qy = (az - 1.0) * my + ay * (m_d - mz)
    qz = az * m_d - ax * m_n - mz
    q = np.stack([qw, qx, qy, qz], axis=-1)
    return quat.canonical(quat.normalize(q))
