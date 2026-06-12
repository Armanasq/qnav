"""FAMC: Fast Accelerometer-Magnetometer Combination (Liu et al. 2018).

Analytic solution of Davenport's q-method for the gravity/magnetic pair. The
key observation is the same as SAAM's: the NED magnetic reference can be
written ``[m_N, 0, m_D]`` with the vertical fraction recovered from the
measurements, which makes the attitude-profile matrix sparse enough that the
4×4 eigenproblem reduces to a closed sequence of scalar eliminations (a
symbolic LDL-style forward solve) — no iteration, no eigendecomposition.

qnav conventions: input is the **specific force** ``f_body`` (any norm) and
the magnetic field ``m_body`` (any norm); output ``q_NB`` (NED-from-body).

FAMC and SAAM solve the same optimality conditions; FAMC additionally exposes
the intermediate elimination pivots, which degrade gracefully (and detectably)
when the two observations are nearly collinear.

Reference: Liu, Liu, Su, "Fast attitude estimation system for unmanned ground
vehicle based on vision/inertial fusion", IEEE Sensors (2018) — FAMC variant.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat

__all__ = ["famc"]


def famc(f_body: np.ndarray, m_body: np.ndarray) -> np.ndarray:
    """Closed-form attitude ``q_NB`` via analytic Davenport elimination.

    Vectorized over leading batch dimensions. Raises ``ValueError`` on
    zero-norm inputs.
    """
    f = np.asarray(f_body, dtype=float)
    m = np.asarray(m_body, dtype=float)
    fn = np.linalg.norm(f, axis=-1, keepdims=True)
    mn = np.linalg.norm(m, axis=-1, keepdims=True)
    if np.any(fn < 1e-12) or np.any(mn < 1e-12):
        raise ValueError("zero-norm accelerometer or magnetometer sample")
    a = -f / fn                      # gravity-down direction in body axes
    mh = m / mn

    m_d = np.sum(a * mh, axis=-1)
    m_n = np.sqrt(np.clip(1.0 - m_d * m_d, 0.0, None))

    # attitude-profile columns for references r_a=[0,0,1], r_m=[m_N,0,m_D]:
    # B = ½ [ m_N·m̂ | 0 | m_D·m̂ + a ]   (3×3, middle column zero)
    c0 = 0.5 * m_n[..., None] * mh
    c2 = 0.5 * (m_d[..., None] * mh + a)
    B00, B10, B20 = c0[..., 0], c0[..., 1], c0[..., 2]
    B02, B12, B22 = c2[..., 0], c2[..., 1], c2[..., 2]
    tau = B02 + B20

    # forward elimination of the shifted Davenport system (3 pivots p0..p2)
    p0 = B22 - B00 + 1.0
    y01 = B10 / p0
    y02 = tau / p0
    p1 = -B10 * B10 / p0 + B00 + B22 + 1.0
    y12 = (B12 + B10 * tau / p0) / p1
    p2 = p0 - 2.0 + tau * tau / p0 + y12 * y12 * p1
    y20 = (tau / p0 + B10 * y12 / p0) / p2
    y21 = y12 / p2
    y22 = 1.0 / p2

    s = B02 - B20
    # back-substitution for the three vector components of the eigenvector
    g0 = y12 * y20 + B10 / p0 * (-1.0)  # = Y10 + Y12·Y20 with Y10 = −B10/(p0·p1)
    g0 = -B10 / (p0 * p1) + y12 * y20
    g1 = -1.0 / p1 + y12 * y21
    g2 = y12 * y22
    va = B12 * (-1.0 / p0 + y01 * g0 + y02 * y20) - s * g0 - y20 * B10
    vb = B12 * (y01 * g1 + y02 * y21) - s * g1 - y21 * B10
    vc = B12 * (y01 * g2 + y02 * y22) - s * g2 - y22 * B10

    # eigenvector is [-1, va, vb, vc] in the eliminated basis; mapping it to
    # the q_NB convention (v_N = R(q) v_B) negates the vector part
    one = np.ones(np.shape(va))
    q = np.stack([one, -va, -vb, -vc], axis=-1)
    return quat.canonical(quat.normalize(q))
