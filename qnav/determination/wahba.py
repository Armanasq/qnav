"""Wahba's problem: shared formulation, attitude profile matrix, degeneracy checks.

Wahba (1965): find ``R_AB ∈ SO(3)`` minimizing

    L(R) = ½ Σᵢ wᵢ ‖vᵢ_A − R vᵢ_B‖²

for unit vector pairs observed in two frames (B = body, A = reference/nav).
All qnav solvers share this convention: **rows are observations**, body
vectors map to reference vectors as ``v_A ≈ R_AB v_B``; the returned
quaternion is ``q_AB``.

The sufficient statistic is the attitude profile matrix

    Bmat = Σᵢ wᵢ vᵢ_A vᵢ_Bᵀ

(Davenport's K and QUEST both derive from it).

References: attitude survey (``__data/Efficient Attitude Estimators ...``);
``__data/attitude.pdf``. See ``docs/math/attitude_determination.md``.
"""

from __future__ import annotations

import warnings

import numpy as np

from qnav.errors import DegenerateGeometryWarning

__all__ = ["attitude_profile", "loss", "check_observability", "normalize_observations"]


def normalize_observations(
    v_ref: np.ndarray, v_body: np.ndarray, weights: np.ndarray | None = None
):
    """Validate/normalize observation sets.

    Inputs ``(N, 3)`` each; rows are unitized; weights default to uniform
    and are scaled to sum to 1. Returns ``(v_ref, v_body, w)``.
    """
    v_ref = np.atleast_2d(np.asarray(v_ref, dtype=float))
    v_body = np.atleast_2d(np.asarray(v_body, dtype=float))
    if v_ref.shape != v_body.shape or v_ref.shape[-1] != 3:
        raise ValueError("v_ref and v_body must both have shape (N, 3)")
    n_ref = np.linalg.norm(v_ref, axis=-1, keepdims=True)
    n_body = np.linalg.norm(v_body, axis=-1, keepdims=True)
    if np.any(n_ref < 1e-12) or np.any(n_body < 1e-12):
        raise ValueError("zero-norm observation vector")
    v_ref = v_ref / n_ref
    v_body = v_body / n_body
    if weights is None:
        w = np.full(v_ref.shape[0], 1.0 / v_ref.shape[0])
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (v_ref.shape[0],) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be nonnegative with positive sum")
        w = w / w.sum()
    return v_ref, v_body, w


def attitude_profile(
    v_ref: np.ndarray, v_body: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """``B = Σ wᵢ vᵢ_ref vᵢ_bodyᵀ`` (3×3), after normalization."""
    v_ref, v_body, w = normalize_observations(v_ref, v_body, weights)
    return (v_ref * w[:, None]).T @ v_body


def loss(R: np.ndarray, v_ref: np.ndarray, v_body: np.ndarray,
         weights: np.ndarray | None = None) -> float:
    """Wahba loss ``½ Σ wᵢ ‖vᵢ_ref − R vᵢ_body‖²`` (normalized weights)."""
    v_ref, v_body, w = normalize_observations(v_ref, v_body, weights)
    r = v_ref - v_body @ np.asarray(R, dtype=float).T
    return float(0.5 * np.sum(w * np.sum(r * r, axis=-1)))


def check_observability(
    v_body: np.ndarray, min_angle: float = np.deg2rad(1.0)
) -> bool:
    """True if the observation set determines attitude (≥ 2 non-collinear
    directions). Warns (:class:`DegenerateGeometryWarning`) and returns False
    when all directions are within ``min_angle`` of collinear — a single
    direction leaves rotation about it unobservable."""
    v = np.atleast_2d(np.asarray(v_body, dtype=float))
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    if v.shape[0] < 2:
        warnings.warn(
            "fewer than two vector observations: attitude not fully observable",
            DegenerateGeometryWarning, stacklevel=2,
        )
        return False
    cross_max = 0.0
    for i in range(1, v.shape[0]):
        cross_max = max(cross_max, float(np.linalg.norm(np.cross(v[0], v[i]))))
    if cross_max < np.sin(min_angle):
        warnings.warn(
            "all observation directions are (near-)collinear: rotation about the "
            "common axis is unobservable",
            DegenerateGeometryWarning, stacklevel=2,
        )
        return False
    return True
