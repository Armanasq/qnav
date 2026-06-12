"""QUEST (QUaternion ESTimator, Shuster 1981) — fast optimal Wahba solution.

QUEST avoids the 4×4 eigendecomposition by solving the characteristic
equation of Davenport's K for its largest root λ_max with Newton iteration
from the excellent starting guess ``λ₀ = Σ wᵢ`` (= 1 with qnav's normalized
weights), then forms the optimal quaternion via the Gibbs-vector / adjoint
formulation:

    x = adj((λ+σ)I − S) z + ... — implemented in the standard form
    q ∝ [γ, x] with γ = det((λ+σ)I − S)

qnav implementation notes (departures from naive QUEST):

- Newton on the quartic ``ψ(λ) = λ⁴ − (a+b)λ² − cλ + (ab + cσ − d)``
  (Shuster's coefficients), with fallback to the eigendecomposition
  (Davenport) if Newton stalls or the Gibbs denominator degenerates
  (rotation near π) — the classical *sequential rotation* fix replaced by an
  exact fallback, trading a little speed for guaranteed correctness.

Reference: attitude survey; ``__data/attitude.pdf``.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.determination.davenport import davenport, davenport_matrix
from qnav.determination.wahba import attitude_profile

__all__ = ["quest"]


def quest(
    v_ref: np.ndarray, v_body: np.ndarray, weights: np.ndarray | None = None,
    *, newton_tol: float = 1e-12, max_iter: int = 20,
) -> np.ndarray:
    """Optimal quaternion ``q_ref_body`` via QUEST with exact fallback.

    Same observation conventions as :func:`qnav.determination.davenport.davenport`.
    """
    B = attitude_profile(v_ref, v_body, weights)
    S = B + B.T
    sigma = float(np.trace(B))
    z = np.array([B[2, 1] - B[1, 2], B[0, 2] - B[2, 0], B[1, 0] - B[0, 1]])

    # Shuster's characteristic-polynomial coefficients
    adjS = np.linalg.inv(S) * np.linalg.det(S) if abs(np.linalg.det(S)) > 1e-300 else _adj(S)
    kappa = float(np.trace(adjS))
    a = sigma * sigma - kappa
    b = sigma * sigma + float(z @ z)
    c = float(np.linalg.det(S) + z @ S @ z)
    d = float(z @ S @ S @ z)

    lam = 1.0  # Σ wᵢ (weights normalized in attitude_profile)
    ok = False
    for _ in range(max_iter):
        psi = lam**4 - (a + b) * lam**2 - c * lam + (a * b + c * sigma - d)
        dpsi = 4.0 * lam**3 - 2.0 * (a + b) * lam - c
        if abs(dpsi) < 1e-300:
            break
        step = psi / dpsi
        lam -= step
        if abs(step) < newton_tol:
            ok = True
            break
    if not ok:
        return davenport(v_ref, v_body, weights)

    alpha = lam**2 - sigma**2 + kappa
    beta = lam - sigma
    gamma = (lam + sigma) * alpha - float(np.linalg.det(S))
    x = (alpha * np.eye(3) + beta * S + S @ S) @ z
    nrm2 = gamma * gamma + float(x @ x)
    if nrm2 < 1e-24 or abs(gamma) < 1e-12 * (1.0 + np.linalg.norm(x)):
        # rotation near π: Gibbs formulation degenerates — exact fallback
        return davenport(v_ref, v_body, weights)
    q = np.concatenate([[gamma], x]) / np.sqrt(nrm2)
    return quat.canonical(q)


def _adj(S: np.ndarray) -> np.ndarray:
    """Adjugate of a 3×3 matrix (valid also for singular S)."""
    c = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            m = np.delete(np.delete(S, i, axis=0), j, axis=1)
            c[i, j] = ((-1) ** (i + j)) * np.linalg.det(m)
    return c.T
