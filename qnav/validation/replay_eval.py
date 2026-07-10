"""Real-dataset attitude replay evaluation.

Replays an :class:`~qnav.validation.imu_datasets.AttitudeDataset` through a
stepwise attitude estimator and reports standard benchmark metrics:

- **total RMSE after heading alignment**: the optical reference frame and
  the estimator's navigation frame differ by an unknown constant rotation
  about gravity; a single optimal yaw offset (closed form) is removed before
  computing the geodesic error — the standard attitude-benchmark metric.
- **inclination (tilt) RMSE**: heading-independent error of the estimated
  gravity direction — meaningful even without a magnetometer.
- **heading RMSE** (after removing the constant offset), **final drift**,
  per-sensor **mean NIS**, **rejection rate**, and the wall-clock
  **real-time factor**.

Only ground-truth-valid samples after ``settle_s`` enter the error metrics;
tracking gaps degrade neither the replay nor the statistics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from qnav.attitude import dcm as dcm_mod
from qnav.filters.base import AttitudeFilter
from qnav.validation.imu_datasets import AttitudeDataset

__all__ = ["AttitudeReplayReport", "heading_aligned_errors", "replay_attitude"]


def _default_gravity_update(flt: AttitudeFilter, ds: "AttitudeDataset", k: int) -> None:
    flt.update_direction(  # type: ignore[attr-defined]
        np.array([0.0, 0.0, 1.0]), ds.accel[k], sigma=0.05, sensor_id="accel")


@dataclass(frozen=True)
class AttitudeReplayReport:
    dataset: str
    estimator: str
    n_samples: int
    duration_s: float
    rate_hz: float
    rmse_deg: float               #: total geodesic RMSE after heading alignment
    tilt_rmse_deg: float          #: inclination-only RMSE (heading-free)
    heading_rmse_deg: float       #: heading RMSE after constant-offset removal
    final_error_deg: float
    heading_offset_deg: float     #: the constant alignment that was removed
    mean_nis: Dict[str, float]
    rejection_rate: float
    realtime_factor: float        #: dataset seconds processed per wall second


def heading_aligned_errors(
    q_est: np.ndarray, q_ref: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Per-sample (total, tilt, heading) errors [rad] + alignment yaw [rad].

    The constant heading offset ``psi*`` maximizes
    ``sum_k trace(Rz(psi)^T R_ref_k R_est_k^T)`` (closed form via atan2).
    """
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise ValueError("no valid ground-truth samples")
    R_est = dcm_mod.from_quaternion(q_est[idx])
    R_ref = dcm_mod.from_quaternion(q_ref[idx])
    E = R_ref @ np.swapaxes(R_est, -1, -2)          # reference <- estimate error
    c = float(np.sum(E[:, 0, 0] + E[:, 1, 1]))
    s = float(np.sum(E[:, 1, 0] - E[:, 0, 1]))
    psi = float(np.arctan2(s, c))
    cz, sz = np.cos(psi), np.sin(psi)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    E_aligned = Rz.T @ E

    tr = np.clip((np.trace(E_aligned, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    total = np.arccos(tr)
    # tilt: angle between estimated and true gravity (reference z) direction
    cos_tilt = np.clip(E_aligned[:, 2, 2], -1.0, 1.0)
    tilt = np.arccos(cos_tilt)
    heading = np.abs(np.arctan2(E_aligned[:, 1, 0], E_aligned[:, 0, 0]))
    return total, tilt, heading, psi


def replay_attitude(
    dataset: AttitudeDataset,
    make_estimator: Callable[[AttitudeDataset], AttitudeFilter],
    update_fn: Optional[Callable[[AttitudeFilter, AttitudeDataset, int], object]] = None,
    settle_s: float = 5.0,
) -> AttitudeReplayReport:
    """Replay one dataset through a freshly constructed estimator.

    ``make_estimator(dataset)`` builds the filter (it may read the dataset's
    rate or first valid ground-truth sample for initialization — document
    what your factory uses). ``update_fn(filter, dataset, k)`` applies the
    aiding measurements for step ``k``; default: gravity update from the
    accelerometer every sample with sigma scaled to the local norm.
    """
    f = make_estimator(dataset)
    n = len(dataset)
    q_est = np.empty((n, 4))

    apply = update_fn if update_fn is not None else _default_gravity_update

    t0 = time.perf_counter()
    for k in range(n):
        f.predict(dataset.gyro[k], dataset.dt)
        apply(f, dataset, k)
        q_est[k] = f.q
    wall = time.perf_counter() - t0

    k0 = min(int(settle_s / dataset.dt), n - 1)
    window = np.zeros(n, dtype=bool)
    window[k0:] = True
    total, tilt, heading, psi = heading_aligned_errors(
        q_est, dataset.q_ref, dataset.valid & window)

    stats = f.innovation_stats
    rejected = sum(s.rejected for s in stats.values())
    count = sum(s.count for s in stats.values())
    return AttitudeReplayReport(
        dataset=dataset.name,
        estimator=type(f).__name__,
        n_samples=n,
        duration_s=dataset.duration_s,
        rate_hz=dataset.rate_hz,
        rmse_deg=float(np.rad2deg(np.sqrt(np.mean(total**2)))),
        tilt_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(tilt**2)))),
        heading_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(heading**2)))),
        final_error_deg=float(np.rad2deg(total[-1])),
        heading_offset_deg=float(np.rad2deg(psi)),
        mean_nis={sid: s.mean_nis for sid, s in stats.items()},
        rejection_rate=rejected / count if count else 0.0,
        realtime_factor=dataset.duration_s / wall if wall > 0 else float("inf"),
    )
