"""Geodesic (Riemannian) distances on SO(3) — re-exported reference metrics."""

from __future__ import annotations

import numpy as np

from qnav.attitude import so3

__all__ = ["geodesic_distance", "chordal_distance"]

geodesic_distance = so3.geodesic_distance


def chordal_distance(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Frobenius (chordal) distance ``‖R1 − R2‖_F``; relates to the geodesic
    angle θ by ``‖R1 − R2‖_F = 2√2·sin(θ/2)``."""
    d = np.asarray(R1, dtype=float) - np.asarray(R2, dtype=float)
    return np.linalg.norm(d, axis=(-2, -1))
