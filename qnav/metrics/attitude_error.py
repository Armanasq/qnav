"""Attitude error metrics for estimator evaluation.

All metrics are double-cover safe (q ≡ −q) and operate on batches.
Error definition: ``e_k = Log(q̂_k* ⊗ q_k_true)`` — the **local** tangent
error consistent with ``docs/conventions.md`` §5.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat

__all__ = ["attitude_error_vector", "angle_error", "rmse_angle", "per_axis_rmse"]


def attitude_error_vector(q_est: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Local error rotation vector ``Log(q̂* ⊗ q_true)``, shape ``(..., 3)``."""
    return quat.log(quat.relative(q_est, q_true))


def angle_error(q_est: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Geodesic angle error [rad] per sample (∈ [0, π])."""
    return quat.angular_distance(q_est, q_true)


def rmse_angle(q_est: np.ndarray, q_true: np.ndarray) -> float:
    """Root-mean-square geodesic angle error over the batch [rad]."""
    e = angle_error(q_est, q_true)
    return float(np.sqrt(np.mean(e * e)))


def per_axis_rmse(q_est: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Per-axis RMSE of the local error vector [rad], shape (3,)."""
    e = attitude_error_vector(q_est, q_true)
    return np.sqrt(np.mean(e * e, axis=0))
