"""Rodrigues parameters (Gibbs vector) and modified Rodrigues parameters (MRP).

Definitions (for unit quaternion ``q = [w, u]``):

- Gibbs / Rodrigues vector: ``g = u / w = tan(θ/2)·n`` — singular at θ = π.
- MRP: ``σ = u / (1 + w) = tan(θ/4)·n`` — singular at θ = 2π; the **shadow
  set** ``σˢ = −σ/‖σ‖²`` represents the same attitude and keeps ‖σ‖ ≤ 1.

References: Hashim SO(3) survey (Rodrigues vector mapping); standard attitude
references (``__data/attitude.pdf``). See ``docs/math/so3.md``.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat

__all__ = [
    "gibbs_from_quaternion", "gibbs_to_quaternion",
    "from_quaternion", "to_quaternion", "shadow", "to_shadow_if_needed",
]

_W_EPS = 1e-12


def gibbs_from_quaternion(q: np.ndarray) -> np.ndarray:
    """Gibbs vector ``g = v/w``. Raises for θ within ~1e−6 rad of π (w → 0)."""
    q = quat.canonical(q)
    w = q[..., :1]
    if np.any(np.abs(w) < _W_EPS):
        raise ValueError("Gibbs vector is singular at θ = π (w = 0)")
    return q[..., 1:] / w


def gibbs_to_quaternion(g: np.ndarray) -> np.ndarray:
    """``q = [1, g] / √(1 + ‖g‖²)`` (always well-defined)."""
    g = np.asarray(g, dtype=float)
    one = np.ones(g.shape[:-1] + (1,))
    q = np.concatenate([one, g], axis=-1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def from_quaternion(q: np.ndarray) -> np.ndarray:
    """MRP ``σ = v / (1 + w)`` after canonicalization (w ≥ 0 ⇒ ‖σ‖ ≤ 1)."""
    q = quat.canonical(q)
    return q[..., 1:] / (1.0 + q[..., :1])


def to_quaternion(sigma: np.ndarray) -> np.ndarray:
    """Inverse MRP map: ``w = (1 − ‖σ‖²)/(1 + ‖σ‖²)``, ``v = 2σ/(1 + ‖σ‖²)``."""
    sigma = np.asarray(sigma, dtype=float)
    s2 = np.sum(sigma * sigma, axis=-1, keepdims=True)
    w = (1.0 - s2) / (1.0 + s2)
    v = 2.0 * sigma / (1.0 + s2)
    return np.concatenate([w, v], axis=-1)


def shadow(sigma: np.ndarray) -> np.ndarray:
    """Shadow set ``σˢ = −σ/‖σ‖²`` (same attitude, complementary domain)."""
    sigma = np.asarray(sigma, dtype=float)
    s2 = np.sum(sigma * sigma, axis=-1, keepdims=True)
    if np.any(s2 < _W_EPS):
        raise ValueError("shadow set is undefined at the identity (σ = 0)")
    return -sigma / s2


def to_shadow_if_needed(sigma: np.ndarray) -> np.ndarray:
    """Switch to the shadow set wherever ‖σ‖ > 1, keeping ‖σ‖ ≤ 1 (standard
    MRP switching used in attitude control/estimation)."""
    sigma = np.asarray(sigma, dtype=float)
    s2 = np.sum(sigma * sigma, axis=-1, keepdims=True)
    return np.where(s2 > 1.0, -sigma / np.where(s2 > 1.0, s2, 1.0), sigma)
