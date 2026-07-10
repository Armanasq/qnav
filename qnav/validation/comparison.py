"""Fair estimator comparison harness.

Runs multiple attitude estimators over the *identical* dataset — same
measurements, same initialization, same noise settings, same measurement
schedule — and reports per-estimator error metrics. This is the required
methodology for any claim that one estimator outperforms another
(single-example comparisons are not evidence).

The dataset is explicit (arrays in, no hidden generation inside the loop) so
runs are reproducible and estimators cannot see different data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from qnav._validate import ensure_positive_dt, ensure_shape
from qnav.attitude import quaternion as quat
from qnav.filters.base import AttitudeFilter

__all__ = ["ComparisonResult", "compare_attitude_estimators"]


@dataclass(frozen=True)
class ComparisonResult:
    """Per-estimator error metrics over one shared dataset."""

    name: str
    rmse_deg: float             #: RMS geodesic attitude error over the run
    final_error_deg: float
    max_error_deg: float
    mean_nis: Dict[str, float]  #: per-sensor mean NIS (NaN if none recorded)
    converged: bool             #: final error below `convergence_deg`


def compare_attitude_estimators(
    estimators: Mapping[str, AttitudeFilter],
    gyro: np.ndarray,
    q_true: np.ndarray,
    dt: float,
    update_fn: Optional[Callable[[AttitudeFilter, int], object]] = None,
    convergence_deg: float = 5.0,
    settle_fraction: float = 0.5,
) -> Sequence[ComparisonResult]:
    """Run every estimator over the same gyro stream and update schedule.

    ``gyro``: (N, 3) measured rates; ``q_true``: (N, 4) ground-truth
    attitude after each step; ``update_fn(filter, k)`` applies the aiding
    measurements for step ``k`` (must draw from pre-generated data so all
    estimators see identical values). RMSE is computed over the last
    ``1 − settle_fraction`` of the run (steady state).
    """
    g = ensure_shape(gyro, (-1, 3), "gyro")
    qt = ensure_shape(q_true, (-1, 4), "q_true")
    if g.shape[0] != qt.shape[0]:
        raise ValueError("gyro and q_true must have the same length")
    dt = ensure_positive_dt(dt)
    n = g.shape[0]
    k0 = int(settle_fraction * n)

    results = []
    for name, f in estimators.items():
        errs = np.empty(n)
        for k in range(n):
            f.predict(g[k], dt)
            if update_fn is not None:
                update_fn(f, k)
            errs[k] = quat.angular_distance(f.q, qt[k])
        errs_deg = np.rad2deg(errs)
        results.append(ComparisonResult(
            name=name,
            rmse_deg=float(np.sqrt(np.mean(errs_deg[k0:] ** 2))),
            final_error_deg=float(errs_deg[-1]),
            max_error_deg=float(errs_deg.max()),
            mean_nis={sid: s.mean_nis for sid, s in f.innovation_stats.items()},
            converged=bool(errs_deg[-1] < convergence_deg),
        ))
    return results
