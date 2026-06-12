"""Library-wide mathematical invariants, callable from tests and benchmarks.

Each invariant returns the worst-case violation (a scalar ≥ 0) so it can be
asserted against a tolerance *and* tracked over time as a regression metric.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import dcm as _dcm
from qnav.attitude import quaternion as quat
from qnav.attitude import so3

__all__ = [
    "quaternion_norm_violation", "dcm_orthogonality_violation",
    "dcm_det_violation", "double_cover_violation",
    "conversion_roundtrip_violation", "exp_log_roundtrip_violation",
    "composition_consistency_violation",
]


def quaternion_norm_violation(q: np.ndarray) -> float:
    """max |‖q‖ − 1| over the batch."""
    return float(np.max(np.abs(quat.norm(q) - 1.0)))


def dcm_orthogonality_violation(R: np.ndarray) -> float:
    """max ‖RᵀR − I‖∞ over the batch."""
    R = np.asarray(R, dtype=float)
    E = np.swapaxes(R, -1, -2) @ R - np.eye(3)
    return float(np.max(np.abs(E)))


def dcm_det_violation(R: np.ndarray) -> float:
    """max |det R − 1| over the batch."""
    return float(np.max(np.abs(np.linalg.det(np.asarray(R, dtype=float)) - 1.0)))


def double_cover_violation(q: np.ndarray) -> float:
    """max geodesic distance between rotations of q and −q (must be 0)."""
    return float(np.max(quat.angular_distance(q, -np.asarray(q, dtype=float))))


def conversion_roundtrip_violation(q: np.ndarray) -> float:
    """max angular error of quat → DCM → quat over the batch."""
    q2 = _dcm.to_quaternion(_dcm.from_quaternion(q))
    return float(np.max(quat.angular_distance(q, q2)))


def exp_log_roundtrip_violation(phi: np.ndarray) -> float:
    """max ‖Log(Exp(φ)) − φ‖ for rotation vectors in the principal domain."""
    return float(np.max(np.linalg.norm(so3.log(so3.exp(phi)) - np.asarray(phi, dtype=float), axis=-1)))


def composition_consistency_violation(q1: np.ndarray, q2: np.ndarray) -> float:
    """max difference between quaternion and DCM composition paths:
    ``R(q1 ⊗ q2)`` vs ``R(q1) R(q2)``."""
    lhs = _dcm.from_quaternion(quat.mul(q1, q2))
    rhs = _dcm.from_quaternion(q1) @ _dcm.from_quaternion(q2)
    return float(np.max(np.abs(lhs - rhs)))
