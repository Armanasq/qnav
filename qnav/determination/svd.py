"""SVD attitude determination (Markley 1988) — most robust Wahba solver.

From the SVD of the attitude profile ``B = U Σ Vᵀ``:

    R_ref_body = U · diag(1, 1, det U · det V) · Vᵀ

The determinant factor enforces a proper rotation. The smallest singular
values diagnose observability: ``s₂ + s₃ → 0`` means rotation about an axis
is unobservable.

Reference: attitude survey; ``__data/attitude.pdf``.
"""

from __future__ import annotations

import warnings

import numpy as np

from qnav.determination.wahba import attitude_profile
from qnav.errors import DegenerateGeometryWarning

__all__ = ["svd_attitude"]


def svd_attitude(
    v_ref: np.ndarray, v_body: np.ndarray, weights: np.ndarray | None = None,
    *, degeneracy_tol: float = 1e-9,
) -> np.ndarray:
    """Optimal rotation ``R_ref_body`` (3×3) via Markley's SVD method.

    Warns when ``s₂ + s₃ < degeneracy_tol`` (attitude not uniquely
    determined). Returns the optimal rotation regardless — the warning, not
    a silent guess, is the contract.
    """
    B = attitude_profile(v_ref, v_body, weights)
    U, s, Vt = np.linalg.svd(B)
    if s[1] + s[2] < degeneracy_tol:
        warnings.warn(
            "attitude profile is rank-deficient: rotation about the dominant "
            "axis is unobservable",
            DegenerateGeometryWarning, stacklevel=2,
        )
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt
