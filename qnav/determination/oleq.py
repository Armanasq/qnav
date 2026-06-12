"""OLEQ: Optimal Linear Estimator of Quaternion (Zhou et al. 2018 family).

Each unit observation pair (v_ref, v_body) gives the linear constraint
``q = W q`` where ``W = ½(M(v_ref) + I₄?)`` — concretely, with the identity

    v_ref ≈ R(q) v_body  ⇔  [0, v_ref] ⊗ q = q ⊗ [0, v_body]

each pair yields ``A q = 0`` with ``A = L([0, v_ref]) − R([0, v_body])``
(left/right quaternion matrices). OLEQ solves the weighted stack by power
iteration on the symmetric form. qnav implements the mathematically
equivalent, more transparent route: smallest-eigenvector of
``Σ wᵢ Aᵢᵀ Aᵢ`` — a direct linear least-squares quaternion estimate
(identical optimum, deterministic).

Reference: attitude survey (linear quaternion estimators section).
"""

from __future__ import annotations

import warnings

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.determination.wahba import normalize_observations
from qnav.errors import DegenerateGeometryWarning

__all__ = ["oleq"]


def oleq(
    v_ref: np.ndarray, v_body: np.ndarray, weights: np.ndarray | None = None,
    *, eig_gap_tol: float = 1e-9,
) -> np.ndarray:
    """Linear least-squares quaternion ``q_ref_body`` from vector observations.

    Same conventions as the other Wahba solvers. Warns when the two smallest
    eigenvalues of the stacked normal matrix are (near-)equal — degenerate
    observation geometry.
    """
    vr, vb, w = normalize_observations(v_ref, v_body, weights)
    H = np.zeros((4, 4))
    for i in range(vr.shape[0]):
        pr = np.concatenate([[0.0], vr[i]])
        pb = np.concatenate([[0.0], vb[i]])
        A = quat.left_matrix(pr) - quat.right_matrix(pb)
        H += w[i] * (A.T @ A)
    eigval, eigvec = np.linalg.eigh(H)
    if eigval[1] - eigval[0] < eig_gap_tol:
        warnings.warn(
            "OLEQ normal matrix has (near-)repeated minimum eigenvalue: attitude "
            "is not uniquely determined",
            DegenerateGeometryWarning, stacklevel=2,
        )
    return quat.canonical(eigvec[:, 0])
