"""Davenport's q-method: optimal Wahba solution by eigendecomposition.

The Wahba loss is ``L(q) = λ₀ − qᵀ K q`` with Davenport's 4×4 matrix

    K = [[σ, zᵀ], [z, S − σI]],   S = B + Bᵀ,  σ = tr B,
    z = [B₃₂−B₂₃, B₁₃−B₃₁, B₂₁−B₁₂]ᵀ  (= Σ wᵢ vᵢ_body × vᵢ_ref, i.e. vee(Bᵀ−B))

built from the attitude profile ``B = Σ wᵢ v_ref v_bodyᵀ``. (Classical texts
write B with body·refᵀ ordering, which flips the sign of z — qnav states its
convention explicitly and verifies it by test.) The optimal
``q_AB`` is the eigenvector of K with the **largest** eigenvalue. Robust
(no root-finding), at the cost of a 4×4 eigendecomposition.

Note on quaternion layout: K is written here for **scalar-first** ``[w, x, y, z]``.

Reference: attitude survey; ``__data/attitude.pdf``.
"""

from __future__ import annotations

import warnings

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.determination.wahba import attitude_profile
from qnav.errors import DegenerateGeometryWarning

__all__ = ["davenport", "davenport_matrix"]


def davenport_matrix(B: np.ndarray) -> np.ndarray:
    """Davenport K (scalar-first ordering) from the attitude profile B."""
    B = np.asarray(B, dtype=float)
    S = B + B.T
    sigma = np.trace(B)
    z = np.array([B[2, 1] - B[1, 2], B[0, 2] - B[2, 0], B[1, 0] - B[0, 1]])
    K = np.empty((4, 4))
    K[0, 0] = sigma
    K[0, 1:] = z
    K[1:, 0] = z
    K[1:, 1:] = S - sigma * np.eye(3)
    return K


def davenport(
    v_ref: np.ndarray, v_body: np.ndarray, weights: np.ndarray | None = None,
    *, eig_gap_tol: float = 1e-9,
) -> np.ndarray:
    """Optimal quaternion ``q_ref_body`` for weighted vector observations.

    Warns (:class:`DegenerateGeometryWarning`) when the top two eigenvalues
    of K are separated by less than ``eig_gap_tol`` — the optimum is then
    ill-defined (degenerate geometry).
    """
    K = davenport_matrix(attitude_profile(v_ref, v_body, weights))
    eigval, eigvec = np.linalg.eigh(K)
    if eigval[-1] - eigval[-2] < eig_gap_tol:
        warnings.warn(
            "Davenport K has (near-)repeated maximum eigenvalue: attitude is "
            "not uniquely determined by the observations",
            DegenerateGeometryWarning, stacklevel=2,
        )
    return quat.canonical(eigvec[:, -1])
