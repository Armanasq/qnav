"""Coning and sculling corrections for multi-sample IMU intervals.

High-rate strapdown systems accumulate several IMU samples into one attitude
/velocity increment before running the (slower) navigation update. Under
coning motion (oscillation about two axes with a phase shift) the naive sum
of angle increments is biased; under sculling motion the naive velocity sum
is likewise biased. The classic two-sample accumulation (Savage,
*Strapdown Analytics*) corrects both to second order:

    Δθ = Σ Δθ_i + ½ Σ α_{i-1} x Δθ_i                        (coning)
    Δv = Σ Δv_i + ½ Σ (α_{i-1} x Δv_i + ν_{i-1} x Δθ_i)     (sculling)

where α/ν are the running angle/velocity sums *before* sample ``i``. The
correction terms vanish for constant rates (any single-axis rotation),
which the tests verify along with drift reduction under true coning motion.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from qnav._validate import ensure_positive_dt, ensure_shape

__all__ = ["accumulate_increments"]


def accumulate_increments(
    omega_ib_b: np.ndarray, f_b: np.ndarray, dt_sample: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Coning/sculling-corrected (Δθ, Δv) over ``N`` uniform IMU samples.

    ``omega_ib_b``/``f_b`` are ``(N, 3)`` rate [rad/s] and specific force
    [m/s²] samples at spacing ``dt_sample``; returns the body-frame rotation
    vector increment [rad] and velocity increment [m/s] for the whole
    interval ``N * dt_sample``, corrected to second order.
    """
    w = ensure_shape(omega_ib_b, (-1, 3), "omega_ib_b")
    f = ensure_shape(f_b, (-1, 3), "f_b")
    if w.shape != f.shape:
        raise ValueError(f"omega and f must have the same shape, got {w.shape} vs {f.shape}")
    dt = ensure_positive_dt(dt_sample, "dt_sample")

    dthetas = w * dt
    dvs = f * dt

    alpha = np.zeros(3)  # running angle sum before current sample
    nu = np.zeros(3)     # running velocity sum before current sample
    coning = np.zeros(3)
    sculling = np.zeros(3)
    for i in range(dthetas.shape[0]):
        coning += 0.5 * np.cross(alpha, dthetas[i])
        sculling += 0.5 * (np.cross(alpha, dvs[i]) + np.cross(nu, dthetas[i]))
        alpha += dthetas[i]
        nu += dvs[i]

    return alpha + coning, nu + sculling
