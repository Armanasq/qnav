"""Shared gated Joseph-form Kalman update used by every qnav ESKF.

One implementation of the update pipeline — NIS, chi-square gating, robust
weighting, quarantine, Joseph-form covariance update, error injection —
shared by the 6-state attitude ESKF and the 15-state navigation ESKF so the
two cannot silently diverge.

The estimator must provide: ``P`` (error covariance), ``gate``
(:class:`~qnav.filters.robust.GatePolicy` or None), ``monitors`` (dict of
:class:`~qnav.filters.robust.SensorMonitor`), and ``_record_update``.
``inject`` applies an accepted error-state correction to the nominal state
(including any post-injection covariance reset on ``P``).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol

import numpy as np

from qnav.filters.contracts import UpdateResult
from qnav.filters.robust import GatePolicy, SensorMonitor

__all__ = ["gated_joseph_update"]


class _GatedEstimator(Protocol):
    P: np.ndarray
    gate: Optional[GatePolicy]
    monitors: Dict[str, SensorMonitor]

    def _record_update(self, result: UpdateResult) -> UpdateResult: ...


def gated_joseph_update(
    est: "_GatedEstimator",
    H: np.ndarray,
    R: np.ndarray,
    innov: np.ndarray,
    *,
    inject: Callable[[np.ndarray], None],
    sensor_id: str,
    timestamp: Optional[float] = None,
) -> UpdateResult:
    """Run one gated, robust, Joseph-form Kalman update; returns the result.

    On rejection (gate or quarantine) the state and covariance are untouched.
    ``UpdateResult.nis`` is always the pre-inflation NIS tested against the
    gate; ``innovation_covariance`` is the S actually used for the update.
    """
    m = innov.shape[0]
    S = H @ est.P @ H.T + R
    nis = float(innov @ np.linalg.solve(S, innov))

    gate = est.gate
    threshold: Optional[float] = None
    weight = 1.0
    rejection: Optional[str] = None
    if gate is not None:
        threshold = gate.threshold(m)
        if nis > threshold:
            if gate.on_gate == "reject":
                rejection = "nis_gate"
            else:
                weight *= threshold / nis
        if rejection is None:
            weight *= gate.robust_weight(nis, m)

    monitor = est.monitors.get(sensor_id)
    if monitor is not None:
        allowed = monitor.note_measurement(rejection is None, timestamp)
        if rejection is None and not allowed:
            rejection = "quarantine"

    if rejection is not None:
        return est._record_update(UpdateResult(
            accepted=False, innovation=innov, innovation_covariance=S,
            nis=nis, gate_threshold=threshold, robust_weight=0.0,
            rejection_reason=rejection, timestamp=timestamp, sensor_id=sensor_id,
        ))

    if weight != 1.0:
        R = R / weight
        S = H @ est.P @ H.T + R

    n = est.P.shape[0]
    K = est.P @ H.T @ np.linalg.solve(S, np.eye(m))
    dx = K @ innov
    IKH = np.eye(n) - K @ H
    est.P = IKH @ est.P @ IKH.T + K @ R @ K.T
    inject(dx)

    return est._record_update(UpdateResult(
        accepted=True, innovation=innov, innovation_covariance=S, nis=nis,
        gate_threshold=threshold, robust_weight=weight, state_correction=dx,
        timestamp=timestamp, sensor_id=sensor_id,
    ))
