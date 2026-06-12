"""Heading error metrics with correct circular statistics."""

from __future__ import annotations

import numpy as np

from qnav.heading.compass import heading_difference

__all__ = ["heading_error", "heading_rmse", "circular_mean_error"]


def heading_error(psi_est: np.ndarray, psi_true: np.ndarray) -> np.ndarray:
    """Signed smallest-angle error ``psi_est − psi_true`` in (−π, π]."""
    return heading_difference(psi_est, psi_true)


def heading_rmse(psi_est: np.ndarray, psi_true: np.ndarray) -> float:
    """RMS of the wrapped heading error [rad]."""
    e = heading_error(psi_est, psi_true)
    return float(np.sqrt(np.mean(e * e)))


def circular_mean_error(psi_est: np.ndarray, psi_true: np.ndarray) -> float:
    """Circular mean of the heading error (bias), via the mean resultant:
    ``atan2(mean sin e, mean cos e)`` [rad]."""
    e = heading_error(psi_est, psi_true)
    return float(np.arctan2(np.mean(np.sin(e)), np.mean(np.cos(e))))
