"""Shared estimator contracts: measurements, update results, health.

These types unify how every qnav recursive estimator reports what it did with
a measurement and what state it is in. They are plain, immutable data
carriers — no behavior beyond validation — so estimators, gating policies
(Phase 3), and user monitoring code all speak the same language.

Conventions
-----------
- ``timestamp`` is seconds in the caller's clock; qnav never assumes an epoch.
- ``covariance`` matrices follow the owning estimator's documented error
  ordering (e.g. ``[δθ, δb]`` for :class:`qnav.filters.Eskf`).
- ``nis`` is the normalized innovation squared ``νᵀ S⁻¹ ν`` (chi-square
  distributed with ``len(ν)`` degrees of freedom for a consistent filter).
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Mapping, Optional, Tuple

import numpy as np

from qnav._validate import ensure_covariance

__all__ = [
    "EstimatorHealth",
    "EstimatorSnapshot",
    "InnovationStatistics",
    "Measurement",
    "UpdateResult",
]


class EstimatorHealth(enum.Enum):
    """Coarse estimator condition, ordered from cold start to failure."""

    INITIALIZING = "initializing"  #: no measurement accepted yet
    HEALTHY = "healthy"            #: state and covariance pass all checks
    DEGRADED = "degraded"          #: usable, but recent rejections/inflation
    UNOBSERVABLE = "unobservable"  #: geometry cannot constrain part of the state
    DIVERGING = "diverging"        #: covariance or innovations growing without bound
    INVALID = "invalid"            #: non-finite state or indefinite covariance


@dataclass(frozen=True)
class Measurement:
    """A timestamped sensor observation handed to an estimator.

    ``value`` is the raw measurement in ``frame``; ``covariance`` its noise
    covariance (matching ``value``'s dimension) or ``None`` when the consuming
    update supplies its own noise model. ``validity_interval`` bounds the
    time span over which the value is meaningful (e.g. an averaged sample).
    """

    value: np.ndarray
    timestamp: float
    frame: str = "body"
    covariance: Optional[np.ndarray] = None
    sensor_id: str = ""
    sequence_id: int = 0
    quality: Mapping[str, Any] = field(default_factory=dict)
    validity_interval: Optional[Tuple[float, float]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # defensive copy + write protection: a frozen dataclass with mutable
        # arrays is not immutable; snapshots and replay determinism rely on
        # measurements never changing after construction.
        v = np.array(self.value, dtype=float, copy=True)
        if not np.all(np.isfinite(v)):
            raise ValueError("Measurement.value contains non-finite values")
        v.setflags(write=False)
        object.__setattr__(self, "value", v)
        t = float(self.timestamp)
        if not np.isfinite(t):
            raise ValueError("Measurement.timestamp must be finite")
        object.__setattr__(self, "timestamp", t)
        if int(self.sequence_id) < 0:
            raise ValueError(f"Measurement.sequence_id must be >= 0, got {self.sequence_id}")
        if not isinstance(self.sensor_id, str) or not isinstance(self.frame, str):
            raise TypeError("Measurement.sensor_id and .frame must be strings")
        if self.covariance is not None:
            P = ensure_covariance(self.covariance, v.size, "Measurement.covariance").copy()
            P.setflags(write=False)
            object.__setattr__(self, "covariance", P)
        if self.validity_interval is not None:
            lo, hi = (float(x) for x in self.validity_interval)
            if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
                raise ValueError("Measurement.validity_interval must be finite (lo, hi), lo <= hi")
            object.__setattr__(self, "validity_interval", (lo, hi))


@dataclass(frozen=True)
class UpdateResult:
    """Everything an estimator did with one measurement.

    ``accepted`` is False when the measurement was rejected (gating,
    numerical failure); ``rejection_reason`` then names why. ``robust_weight``
    is 1.0 for a plain Kalman update and in (0, 1] under robust losses.
    ``state_correction`` follows the estimator's error ordering.
    """

    accepted: bool
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    nis: float
    gate_threshold: Optional[float] = None
    robust_weight: float = 1.0
    state_correction: Optional[np.ndarray] = None
    rejection_reason: Optional[str] = None
    timestamp: Optional[float] = None
    sensor_id: str = ""


@dataclass
class InnovationStatistics:
    """Running NIS statistics for one measurement source.

    ``mean_nis`` should approach the innovation dimension for a consistent
    filter; sustained deviation indicates mistuned noise or a faulty sensor.
    """

    count: int = 0
    accepted: int = 0
    rejected: int = 0
    consecutive_rejections: int = 0
    #: innovation dimension of the last recorded update (0 before any).
    dim: int = 0
    _nis_sum: float = 0.0
    _nis_sq_sum: float = 0.0
    #: sliding window of the most recent NIS values (divergence detection).
    recent_nis: Deque[float] = field(default_factory=lambda: deque(maxlen=20))

    def record(self, result: UpdateResult) -> None:
        self.count += 1
        self.dim = int(np.asarray(result.innovation).size)
        if result.accepted:
            self.accepted += 1
            self.consecutive_rejections = 0
        else:
            self.rejected += 1
            self.consecutive_rejections += 1
        if np.isfinite(result.nis):
            self._nis_sum += result.nis
            self._nis_sq_sum += result.nis**2
            self.recent_nis.append(float(result.nis))

    @property
    def mean_recent_nis(self) -> float:
        """Mean NIS over the sliding window (NaN when empty)."""
        if not self.recent_nis:
            return float("nan")
        return float(np.mean(self.recent_nis))

    @property
    def mean_nis(self) -> float:
        return self._nis_sum / self.count if self.count else float("nan")

    @property
    def var_nis(self) -> float:
        if self.count < 2:
            return float("nan")
        m = self.mean_nis
        return max(self._nis_sq_sum / self.count - m * m, 0.0) * self.count / (self.count - 1)


@dataclass(frozen=True)
class EstimatorSnapshot:
    """A restorable copy of an estimator's full mutable state.

    ``state`` maps attribute names to deep-copied values. Snapshots are only
    valid for the estimator type that produced them; ``estimator_type`` is
    checked on restore.
    """

    estimator_type: str
    state: Mapping[str, Any]
    timestamp: Optional[float] = None
