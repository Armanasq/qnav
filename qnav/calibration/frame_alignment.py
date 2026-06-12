"""Sensor-to-body (or sensor-to-sensor) rotational alignment estimation.

Given paired vector observations of the same physical quantity in two frames
(e.g. body-frame references vs sensor measurements, or two gyros' rates),
the alignment rotation is a Wahba problem — solved with the SVD method for
maximal robustness and built-in degeneracy diagnostics.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import dcm as _dcm
from qnav.determination.svd import svd_attitude
from qnav.determination.wahba import check_observability, loss
from qnav.errors import CalibrationError
from qnav.sensors.alignment import SensorAlignment

__all__ = ["align_from_vector_pairs", "align_gyro_to_body"]


def align_from_vector_pairs(
    v_body: np.ndarray, v_sensor: np.ndarray,
    weights: np.ndarray | None = None, max_residual: float = 0.2,
) -> SensorAlignment:
    """Estimate ``q_body_sensor`` from pairs ``v_body ≈ R_BS v_sensor``.

    ``(N, 3)`` arrays; directions only (rows are unitized). Raises
    :class:`CalibrationError` when geometry is unobservable or the Wahba
    residual exceeds ``max_residual`` (suspect data/time misalignment).
    """
    if not check_observability(np.atleast_2d(np.asarray(v_sensor, dtype=float))):
        raise CalibrationError(
            "alignment unobservable: provide rotationally diverse vector pairs"
        )
    R = svd_attitude(v_body, v_sensor, weights)
    resid = loss(R, v_body, v_sensor, weights)
    if resid > max_residual:
        raise CalibrationError(
            f"alignment residual {resid:.3f} exceeds {max_residual}: check data"
        )
    return SensorAlignment(q_body_sensor=_dcm.to_quaternion(R))


def align_gyro_to_body(
    omega_body_ref: np.ndarray, omega_sensor: np.ndarray,
    min_rate: float = 0.05,
) -> SensorAlignment:
    """Gyro alignment from synchronized rate pairs during dynamic motion.

    Samples with ‖ω‖ < ``min_rate`` [rad/s] are discarded (direction
    meaningless at rest); remaining directions feed the Wahba solver with
    rate-magnitude weights.
    """
    wb = np.asarray(omega_body_ref, dtype=float)
    ws = np.asarray(omega_sensor, dtype=float)
    n = np.linalg.norm(ws, axis=-1)
    keep = (n > min_rate) & (np.linalg.norm(wb, axis=-1) > min_rate)
    if keep.sum() < 3:
        raise CalibrationError("too few high-rate samples for gyro alignment")
    return align_from_vector_pairs(wb[keep], ws[keep], weights=n[keep])
