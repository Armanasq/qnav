"""TRIAD: deterministic two-vector attitude determination.

Given one *primary* and one *secondary* observation pair, build orthonormal
triads in each frame and match them:

    t1 = v1,  t2 = (v1 × v2)/‖v1 × v2‖,  t3 = t1 × t2     (per frame)
    R_AB = [t1_A t2_A t3_A] · [t1_B t2_B t3_B]ᵀ

The primary direction is matched exactly; secondary errors are pushed into
rotation about the primary. Put the more accurate sensor first (classically:
accelerometer primary, magnetometer secondary).

Reference: attitude survey; ``__data/attitude.pdf``.
"""

from __future__ import annotations

import numpy as np

from qnav.determination.wahba import normalize_observations
from qnav.errors import DegenerateGeometryWarning

__all__ = ["triad"]


def triad(
    v1_ref: np.ndarray, v2_ref: np.ndarray,
    v1_body: np.ndarray, v2_body: np.ndarray,
) -> np.ndarray:
    """Rotation ``R_ref_body`` from a primary (v1) and secondary (v2) pair.

    Raises ``ValueError`` on (near-)collinear pairs — TRIAD is undefined
    there (use :func:`qnav.determination.wahba.check_observability` to
    pre-screen).
    """
    vr, vb, _ = normalize_observations(
        np.stack([np.asarray(v1_ref, dtype=float), np.asarray(v2_ref, dtype=float)]),
        np.stack([np.asarray(v1_body, dtype=float), np.asarray(v2_body, dtype=float)]),
    )
    (v1r, v2r), (v1b, v2b) = vr, vb

    def make_triad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        c = np.cross(a, b)
        n = np.linalg.norm(c)
        if n < 1e-10:
            raise ValueError("TRIAD is undefined for (near-)collinear observations")
        t2 = c / n
        return np.column_stack([a, t2, np.cross(a, t2)])

    Tr = make_triad(v1r, v2r)
    Tb = make_triad(v1b, v2b)
    return Tr @ Tb.T
