"""Filter foundations: explicit state, noise, and measurement declarations.

Every qnav estimator is a **stepwise** object — construct once, then call
``predict(gyro, dt)`` and ``update_*(...)`` per sample. Batch processing is a
thin loop the user controls (no hidden batch-in-constructor semantics).

Every estimator must declare:

- state definition (attributes & meaning)
- error definition (for covariance-bearing filters)
- process model and noise semantics (continuous densities vs discrete)
- per-measurement frame and noise semantics
- explicit assumptions (e.g. "accelerometer ≈ gravity only")
"""

from __future__ import annotations

import abc
import copy
from typing import Callable, Dict, Optional

import numpy as np

from qnav._validate import ensure_positive_dt, ensure_vector3
from qnav.attitude import quaternion as quat
from qnav.filters.contracts import (
    EstimatorHealth,
    EstimatorSnapshot,
    InnovationStatistics,
    UpdateResult,
)
from qnav.types import ArrayLike

__all__ = ["AttitudeFilter"]


class AttitudeFilter(abc.ABC):
    """Base class for attitude estimators producing ``q_nav_body``.

    Subclasses keep the current estimate in ``self.q`` (scalar-first
    Hamilton, ``q_nav_body`` for the declared navigation frame) and document
    their additional state.
    """

    #: navigation frame of the estimate ("NED" or "ENU"); set by subclass.
    nav_frame: str
    #: as-constructed state captured automatically after __init__ (for reset).
    _initial_state: Dict[str, object]

    def __init_subclass__(cls, **kwargs: object) -> None:
        # After the most-derived __init__ completes, record the constructed
        # state so reset() can deterministically restore it.
        super().__init_subclass__(**kwargs)
        inner = cls.__init__

        def __init__(self: "AttitudeFilter", *args: object, **kw: object) -> None:
            inner(self, *args, **kw)  # type: ignore[arg-type]
            if type(self) is cls:
                self._initial_state = self._capture_state()

        cls.__init__ = __init__  # type: ignore[method-assign]

    def __init__(self, q0: np.ndarray | None = None, nav_frame: str = "NED") -> None:
        if nav_frame not in ("NED", "ENU"):
            raise ValueError(f"nav_frame must be 'NED' or 'ENU', got {nav_frame!r}")
        self.nav_frame = nav_frame
        self.q = quat.identity() if q0 is None else quat.normalize(np.asarray(q0, dtype=float))
        #: result of the most recent measurement update, if any.
        self.last_update: Optional[UpdateResult] = None
        #: running innovation statistics keyed by sensor identifier.
        self.innovation_stats: Dict[str, InnovationStatistics] = {}

    def predict(self, omega_body: ArrayLike, dt: float) -> np.ndarray:
        """Propagate the attitude with a body-frame rate sample [rad/s]; returns q.

        Validates that ``omega_body`` is a finite 3-vector and ``dt`` a finite
        positive scalar, then dispatches to the subclass propagation.
        """
        omega = ensure_vector3(omega_body, "omega_body")
        if omega.ndim != 1:
            raise ValueError(f"omega_body must be a single 3-vector, got shape {omega.shape}")
        return self._predict(omega, ensure_positive_dt(dt))

    @abc.abstractmethod
    def _predict(self, omega_body: np.ndarray, dt: float) -> np.ndarray:
        """Subclass propagation; receives validated inputs."""

    # -- lifecycle ----------------------------------------------------------
    def _capture_state(self) -> Dict[str, object]:
        return copy.deepcopy(self.__dict__)

    def snapshot(self, timestamp: float | None = None) -> EstimatorSnapshot:
        """A restorable deep copy of the full estimator state."""
        state = self._capture_state()
        state.pop("_initial_state", None)
        return EstimatorSnapshot(
            estimator_type=type(self).__qualname__, state=state, timestamp=timestamp
        )

    def restore(self, snap: EstimatorSnapshot) -> None:
        """Restore a state captured by :meth:`snapshot` of the same type."""
        if snap.estimator_type != type(self).__qualname__:
            raise ValueError(
                f"snapshot is for {snap.estimator_type!r}, not {type(self).__qualname__!r}"
            )
        for key, value in copy.deepcopy(dict(snap.state)).items():
            setattr(self, key, value)

    def reset(self) -> None:
        """Deterministically restore the as-constructed state."""
        initial = self._initial_state
        self.__dict__.clear()
        self.__dict__.update(copy.deepcopy(initial))
        self._initial_state = initial

    def _record_update(self, result: UpdateResult) -> UpdateResult:
        """Store a measurement-update outcome for health/monitoring."""
        self.last_update = result
        self.innovation_stats.setdefault(
            result.sensor_id, InnovationStatistics()
        ).record(result)
        return result

    @property
    def health(self) -> EstimatorHealth:
        """Coarse estimator condition (see :class:`EstimatorHealth`).

        Base implementation checks state finiteness and — when the estimator
        has a covariance attribute ``P`` — covariance finiteness, symmetry,
        and positive semidefiniteness. Estimators that have not yet absorbed
        any measurement report ``INITIALIZING``; recent consecutive
        rejections report ``DEGRADED``.
        """
        if not np.all(np.isfinite(self.q)):
            return EstimatorHealth.INVALID
        P = getattr(self, "P", None)
        if P is not None:
            P = np.asarray(P, dtype=float)
            if not np.all(np.isfinite(P)):
                return EstimatorHealth.INVALID
            scale = max(float(np.abs(P).max()), 1e-300)
            if float(np.abs(P - P.T).max()) > 1e-9 * scale:
                return EstimatorHealth.INVALID
            if float(np.linalg.eigvalsh(0.5 * (P + P.T)).min()) < -1e-10 * scale:
                return EstimatorHealth.INVALID
        if not self.innovation_stats:
            return EstimatorHealth.INITIALIZING
        if any(s.consecutive_rejections >= 3 for s in self.innovation_stats.values()):
            return EstimatorHealth.DEGRADED
        return EstimatorHealth.HEALTHY

    def run(
        self,
        omegas: np.ndarray,
        dt: float,
        update_fn: Optional[Callable[["AttitudeFilter", int], object]] = None,
    ) -> np.ndarray:
        """Convenience batch loop: predict each sample, optionally calling
        ``update_fn(self, k)`` after each predict. Returns ``(N, 4)``."""
        omegas = np.asarray(omegas, dtype=float)
        out = np.empty((omegas.shape[0], 4))
        for k in range(omegas.shape[0]):
            self.predict(omegas[k], dt)
            if update_fn is not None:
                update_fn(self, k)
            out[k] = self.q
        return out
