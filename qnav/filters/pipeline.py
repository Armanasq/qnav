"""Timestamped fusion pipeline: real sensor timing on top of any attitude filter.

Real sensor streams are not synchronized, evenly sampled, or ordered.
:class:`FusionPipeline` wraps a stepwise estimator (any
:class:`~qnav.filters.base.AttitudeFilter`) and handles:

- variable time steps (dt is derived from consecutive gyro timestamps)
- multiple sensor rates (measurements arrive whenever they arrive)
- duplicate gyro samples and duplicate measurements (sensor_id, sequence_id)
- out-of-order (delayed) measurements via bounded rollback-and-replay over a
  fixed-lag snapshot history
- dropped/missing samples (gaps are propagated with the true dt and flagged)
- clock discontinuities (backward gyro time beyond a tolerance)
- per-sensor time offsets (known lag correction; estimation hooks)
- attitude interpolation at arbitrary times inside the history window (SLERP
  between snapshots — valid on SO(3); covariance is *not* interpolated)

Every call returns a :class:`ProcessReport` stating exactly what was done and
why — the pipeline never silently drops data.
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from qnav._validate import ensure_positive, ensure_vector3
from qnav.attitude import interpolation as interp
from qnav.filters.base import AttitudeFilter
from qnav.filters.contracts import EstimatorSnapshot, Measurement

__all__ = ["ClockDiscontinuityError", "FusionPipeline", "ProcessReport"]


class ClockDiscontinuityError(RuntimeError):
    """Gyro time went backwards beyond the configured tolerance."""


#: an event in the replay log: ("imu", t, omega) or ("meas", t, Measurement)
_Event = Tuple[str, float, object]


@dataclass(frozen=True)
class ProcessReport:
    """What the pipeline did with one input sample."""

    applied: bool
    action: str                    #: "propagate", "update", "replay", "drop", "init"
    reason: Optional[str] = None   #: set when not applied ("duplicate", "too_old", ...)
    dt: Optional[float] = None     #: propagation step actually used
    replayed_events: int = 0       #: events re-applied after a rollback
    gap: bool = False              #: dt exceeded max_gap (missing samples likely)


@dataclass
class _HistoryEntry:
    t: float
    snapshot: EstimatorSnapshot
    events: List[_Event] = field(default_factory=list)


class FusionPipeline:
    """Bounded fixed-lag fusion driver for one estimator.

    Parameters
    ----------
    estimator:
        Any stepwise attitude filter. Propagation uses
        ``estimator.predict(omega, dt)``; measurements are applied by the
        handler registered for their ``sensor_id``.
    max_lag:
        Seconds of state history kept for delayed-measurement replay.
        Measurements older than this (relative to the newest gyro sample)
        are rejected with reason ``"too_old"``.
    max_gap:
        Gyro gaps larger than this [s] are still propagated with the true dt
        but flagged (``ProcessReport.gap``) so callers can react.
    discontinuity_tol:
        Backward gyro-time tolerance [s]. A regression within tolerance is
        treated as a duplicate; beyond it, :class:`ClockDiscontinuityError`.
    """

    def __init__(
        self,
        estimator: AttitudeFilter,
        max_lag: float = 0.5,
        max_gap: float = 0.1,
        discontinuity_tol: float = 1e-9,
    ) -> None:
        if not isinstance(estimator, AttitudeFilter):
            raise TypeError("estimator must be an AttitudeFilter")
        self.estimator = estimator
        self.max_lag = ensure_positive(max_lag, "max_lag")
        self.max_gap = ensure_positive(max_gap, "max_gap")
        self.discontinuity_tol = ensure_positive(discontinuity_tol, "discontinuity_tol")
        self._handlers: Dict[str, Callable[[AttitudeFilter, Measurement], object]] = {}
        self._time_offsets: Dict[str, float] = {}
        self._history: Deque[_HistoryEntry] = deque()
        self._seen: Set[Tuple[str, int]] = set()
        self._seen_order: Deque[Tuple[str, int]] = deque(maxlen=4096)
        self.t: Optional[float] = None  #: newest gyro timestamp

    # -- configuration -------------------------------------------------------
    def register_handler(
        self, sensor_id: str, handler: Callable[[AttitudeFilter, Measurement], object]
    ) -> None:
        """Register how measurements from ``sensor_id`` are fused.

        ``handler(estimator, measurement)`` applies one measurement; the
        estimator's own gating and reporting (``last_update``) apply as
        configured on the estimator.
        """
        self._handlers[sensor_id] = handler

    def set_time_offset(self, sensor_id: str, offset: float) -> None:
        """Known sensor clock offset [s]: ``t_corrected = t_meas + offset``.

        This is also the hook for online time-offset estimation — an external
        estimator may update the offset between samples.
        """
        self._time_offsets[sensor_id] = float(offset)

    # -- gyro path -----------------------------------------------------------
    def process_imu(self, omega_body: np.ndarray, t: float) -> ProcessReport:
        """Propagate with one timestamped gyro sample."""
        omega = ensure_vector3(omega_body, "omega_body")
        t = float(t)
        if not np.isfinite(t):
            raise ValueError("t must be finite")

        if self.t is None:
            self.t = t
            self._push_history(t)
            return ProcessReport(applied=True, action="init", dt=None)

        if t <= self.t:
            if self.t - t <= self.discontinuity_tol:
                return ProcessReport(applied=False, action="drop", reason="duplicate")
            raise ClockDiscontinuityError(
                f"gyro time went backwards: {t} after {self.t} "
                f"(tolerance {self.discontinuity_tol})"
            )

        dt = t - self.t
        self.estimator.predict(omega, dt)
        self._history[-1].events.append(("imu", t, omega))
        self.t = t
        self._push_history(t)
        self._trim_history()
        return ProcessReport(applied=True, action="propagate", dt=dt, gap=dt > self.max_gap)

    # -- measurement path ------------------------------------------------------
    def process_measurement(self, m: Measurement) -> ProcessReport:
        """Fuse one timestamped measurement, replaying history if delayed."""
        if m.sensor_id not in self._handlers:
            raise KeyError(
                f"no handler registered for sensor_id {m.sensor_id!r}; "
                f"known: {sorted(self._handlers)}"
            )
        if self.t is None:
            return ProcessReport(applied=False, action="drop", reason="no_imu_yet")

        key = (m.sensor_id, m.sequence_id)
        if key in self._seen:
            return ProcessReport(applied=False, action="drop", reason="duplicate")

        t_meas = m.timestamp + self._time_offsets.get(m.sensor_id, 0.0)

        if t_meas < self.t - self.max_lag or (
            self._history and t_meas < self._history[0].t
        ):
            return ProcessReport(applied=False, action="drop", reason="too_old")

        self._mark_seen(key)

        if t_meas >= self.t:
            # in-order (or future-stamped within reason): apply at current state
            self._handlers[m.sensor_id](self.estimator, m)
            self._history[-1].events.append(("meas", t_meas, m))
            return ProcessReport(applied=True, action="update")

        # delayed: roll back to the last snapshot at/before t_meas and replay
        return self._replay_with(m, t_meas)

    # -- interpolation ---------------------------------------------------------
    def attitude_at(self, t: float) -> np.ndarray:
        """SLERP-interpolated attitude quaternion at time ``t`` within the
        history window. Exact at snapshot times; raises outside the window."""
        if not self._history:
            raise ValueError("no history available")
        times = [h.t for h in self._history]
        if not times[0] <= t <= times[-1]:
            raise ValueError(f"t={t} outside history window [{times[0]}, {times[-1]}]")
        i = bisect.bisect_right(times, t)
        if i == len(times):
            return np.asarray(self._history[-1].snapshot.state["q"], dtype=float)
        lo, hi = self._history[i - 1], self._history[i]
        if hi.t == lo.t:
            return np.asarray(lo.snapshot.state["q"], dtype=float)
        alpha = (t - lo.t) / (hi.t - lo.t)
        q0 = np.asarray(lo.snapshot.state["q"], dtype=float)
        q1 = np.asarray(hi.snapshot.state["q"], dtype=float)
        return interp.slerp(q0, q1, alpha)

    # -- internals ---------------------------------------------------------------
    def _push_history(self, t: float) -> None:
        self._history.append(_HistoryEntry(t=t, snapshot=self.estimator.snapshot(timestamp=t)))

    def _trim_history(self) -> None:
        assert self.t is not None
        while len(self._history) > 2 and self._history[1].t < self.t - self.max_lag:
            self._history.popleft()

    def _mark_seen(self, key: Tuple[str, int]) -> None:
        if len(self._seen_order) == self._seen_order.maxlen:
            self._seen.discard(self._seen_order[0])
        self._seen_order.append(key)
        self._seen.add(key)

    def _replay_with(self, m: Measurement, t_meas: float) -> ProcessReport:
        times = [h.t for h in self._history]
        i = bisect.bisect_right(times, t_meas) - 1
        base = self._history[i]

        # collect the events that followed the rollback point
        pending: List[_Event] = []
        for entry in list(self._history)[i:]:
            pending.extend(entry.events)

        self.estimator.restore(base.snapshot)
        # drop history after the rollback point; it will be rebuilt
        while len(self._history) > i + 1:
            self._history.pop()
        base.events = []

        # merge the delayed measurement into the pending event stream by time
        merged: List[_Event] = []
        inserted = False
        for ev in pending:
            if not inserted and ev[1] > t_meas:
                merged.append(("meas", t_meas, m))
                inserted = True
            merged.append(ev)
        if not inserted:
            merged.append(("meas", t_meas, m))

        self.t = base.t
        replayed = 0
        for kind, t_ev, payload in merged:
            if kind == "imu":
                omega = payload
                dt = t_ev - self.t
                if dt > 0:
                    self.estimator.predict(np.asarray(omega, dtype=float), dt)
                    self._history[-1].events.append(("imu", t_ev, omega))
                    self.t = t_ev
                    self._push_history(t_ev)
            else:
                meas = payload
                assert isinstance(meas, Measurement)
                self._handlers[meas.sensor_id](self.estimator, meas)
                self._history[-1].events.append(("meas", t_ev, meas))
            replayed += 1
        self._trim_history()
        return ProcessReport(applied=True, action="replay", replayed_events=replayed)
