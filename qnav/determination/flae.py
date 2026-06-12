"""FLAE: Fast Linear Attitude Estimator (Wu et al. 2018).

Solves the N-vector Wahba problem by recasting Davenport's 4×4 eigenproblem
as a **quartic characteristic polynomial** whose coefficients come directly
from the 3×3 attitude-profile matrix — the polynomial is
``λ⁴ + τ₁λ² + τ₂λ + τ₃`` (no cubic term, because tr K = 0). The optimal
quaternion is the null vector of ``W − λ_max I``.

qnav implementation notes (deliberate departures from the original paper's
"symbolic" path):

- λ_max is found from the **companion-matrix roots** of the quartic followed
  by two Newton polish steps. The closed-form (Ferrari) resolvent used in the
  original is numerically fragile when the resolvent cubic has near-equal
  roots (clean data!), producing NaN exactly in the noise-free case. The
  companion route is uniformly stable, and the Newton polish restores full
  closed-form accuracy.
- The null vector is taken from the SVD of ``W − λI`` (smallest right
  singular vector) instead of a 3×3 bordered solve — rank-safe when the
  geometry is degenerate.

Conventions: rows of ``v_ref``/``v_body`` are observation pairs with
``v_ref ≈ R(q) v_body``; returns ``q_AB`` (reference-from-body) like every
other qnav Wahba solver.

Reference: Wu, Zhou, Gao, Fourati, Liu, "Fast linear quaternion attitude
estimator using vector observations", IEEE Trans. Automation Science and
Engineering (2018).
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.determination.wahba import normalize_observations

__all__ = ["flae"]


def _w_matrix(H: np.ndarray) -> np.ndarray:
    """4×4 symmetric data matrix from the rows of ``H = Σ wᵢ b̂ᵢ r̂ᵢᵀ``.

    Each row of H enters through a fixed sparse pattern (the paper's
    P₁/P₂/P₃ operators); their sum is the matrix whose dominant eigenvector
    is the optimal quaternion.
    """
    hx, hy, hz = H[0], H[1], H[2]
    W = np.zeros((4, 4))
    # row-x pattern
    W += np.array([
        [hx[0], 0.0, -hx[2], hx[1]],
        [0.0, hx[0], hx[1], hx[2]],
        [-hx[2], hx[1], -hx[0], 0.0],
        [hx[1], hx[2], 0.0, -hx[0]],
    ])
    # row-y pattern
    W += np.array([
        [hy[1], hy[2], 0.0, -hy[0]],
        [hy[2], -hy[1], hy[0], 0.0],
        [0.0, hy[0], hy[1], hy[2]],
        [-hy[0], 0.0, hy[2], -hy[1]],
    ])
    # row-z pattern
    W += np.array([
        [hz[2], -hz[1], hz[0], 0.0],
        [-hz[1], -hz[2], 0.0, hz[0]],
        [hz[0], 0.0, -hz[2], hz[1]],
        [0.0, hz[0], hz[1], hz[2]],
    ])
    return W


def flae(
    v_ref: np.ndarray, v_body: np.ndarray, weights: np.ndarray | None = None,
    *, newton_iters: int = 2,
) -> np.ndarray:
    """Fast linear attitude ``q_ref_body`` from N weighted vector pairs.

    Same interface as :func:`qnav.determination.davenport.davenport` /
    :func:`qnav.determination.quest.quest`. ``newton_iters`` polish steps are
    applied to the companion-matrix λ_max (2 reaches machine precision).
    """
    vr, vb, w = normalize_observations(v_ref, v_body, weights)
    H = (vb * w[:, None]).T @ vr            # Σ wᵢ b̂ᵢ r̂ᵢᵀ  (3×3)
    W = _w_matrix(H)

    # characteristic polynomial λ⁴ + t1·λ² + t2·λ + t3 (tr W = 0 kills λ³)
    t1 = -2.0 * float(np.trace(H @ H.T))
    t2 = -8.0 * float(np.linalg.det(H))
    t3 = float(np.linalg.det(W))

    roots = np.roots([1.0, 0.0, t1, t2, t3])
    real = roots[np.abs(roots.imag) < 1e-9].real
    lam = float(real.max()) if real.size else 1.0
    for _ in range(newton_iters):
        fval = lam**4 + t1 * lam**2 + t2 * lam + t3
        fp = 4.0 * lam**3 + 2.0 * t1 * lam + t2
        if abs(fp) < 1e-300:
            break
        lam -= fval / fp

    # optimal quaternion: null vector of (W − λ_max I), rank-safe via SVD
    _, _, Vt = np.linalg.svd(W - lam * np.eye(4))
    return quat.canonical(Vt[-1])
