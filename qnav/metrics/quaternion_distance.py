"""Quaternion distance functions (each with its geometry made explicit)."""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat

__all__ = ["geodesic_angle", "inner_product_distance", "chordal_quaternion_distance"]

geodesic_angle = quat.angular_distance


def inner_product_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """``1 − |⟨q1, q2⟩|`` ∈ [0, 1] — cheap, sign-invariant, monotone in the
    geodesic angle (≈ θ²/8 for small θ)."""
    d = np.abs(np.sum(np.asarray(q1, dtype=float) * np.asarray(q2, dtype=float), axis=-1))
    return 1.0 - np.clip(d, 0.0, 1.0)


def chordal_quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """``min(‖q1 − q2‖, ‖q1 + q2‖)`` — Euclidean chord on S³ with double-cover
    handling (= 2 sin(θ/4))."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    return np.minimum(
        np.linalg.norm(q1 - q2, axis=-1), np.linalg.norm(q1 + q2, axis=-1)
    )
