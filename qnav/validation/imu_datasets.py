"""Real IMU attitude-benchmark dataset access (BROAD-style HDF5 + NPZ).

Supported layout (as used by the BROAD attitude benchmark family — EuRoC-MAV,
TUM-VI, OxIOD, RepoIMU, Caruso-Sassari, Myon recordings): one HDF5 file per
trial with per-axis datasets ``gyr_x/y/z`` [rad/s], ``acc_x/y/z`` [m/s²],
optional ``mag_x/y/z`` [arbitrary but consistent units], uniform ``dt`` [s],
``movement_mask``, and optical ground truth ``opt_a/b/c/d``.

Verified conventions (checked empirically in ``tests/test_real_data.py``
against gyro integration and gravity direction, not assumed):

- ``opt_a..d`` is a **scalar-first Hamilton** quaternion ``q_ref_body``
  (rotates body coordinates into the reference frame),
- the reference frame has **z up** (ENU-like): at rest the accelerometer
  reads ``R(q)ᵀ [0, 0, +g]``,
- ground-truth samples with NaN components are optical-tracking gaps; the
  loader exposes them through ``valid`` instead of dropping rows (IMU data
  remains usable during gaps).

Data location: ``QNAV_DATA_DIR`` environment variable, else ``qnav/data``
next to the installed package. The datasets are **not** shipped with qnav
(hundreds of MB); tests fall back to the compact fixture in
``tests/fixtures``. h5py is an optional dependency (``qnav[datasets]``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from qnav._validate import ensure_positive_dt

__all__ = ["AttitudeDataset", "available_datasets", "data_root", "load_attitude_dataset"]


@dataclass(frozen=True)
class AttitudeDataset:
    """One attitude-benchmark trial with optical ground truth.

    ``q_ref`` rows may contain NaN where tracking dropped out; ``valid``
    flags rows with usable ground truth. ``mag`` is None when the trial has
    no magnetometer.
    """

    name: str
    dt: float                      #: uniform sample interval [s]
    gyro: np.ndarray               #: (N, 3) [rad/s], body frame
    accel: np.ndarray              #: (N, 3) [m/s²], specific force, body frame
    q_ref: np.ndarray              #: (N, 4) scalar-first q_ref_body (z-up reference)
    valid: np.ndarray              #: (N,) bool, ground-truth validity
    mag: Optional[np.ndarray] = None
    movement: Optional[np.ndarray] = None  #: (N,) bool, motion mask when present

    def __len__(self) -> int:
        return int(self.gyro.shape[0])

    @property
    def rate_hz(self) -> float:
        return 1.0 / self.dt

    @property
    def duration_s(self) -> float:
        return len(self) * self.dt


def data_root() -> Path:
    """Dataset directory: ``$QNAV_DATA_DIR`` or ``qnav/data`` in the package."""
    env = os.environ.get("QNAV_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data"


def available_datasets(root: Optional[Path] = None) -> Sequence[Path]:
    """All benchmark trial files under the data root (sorted, may be empty)."""
    base = Path(root) if root is not None else data_root()
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*") if p.suffix in (".hdf5", ".h5", ".npz"))


def load_attitude_dataset(path: os.PathLike | str) -> AttitudeDataset:
    """Load one trial (HDF5 as documented above, or an NPZ fixture with the
    keys ``dt, gyro, accel, q_ref`` and optional ``mag, movement``)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset file not found: {p}")
    if p.suffix == ".npz":
        return _load_npz(p)
    return _load_hdf5(p)


def _finalize(name: str, dt: float, gyro: np.ndarray, accel: np.ndarray,
              q_raw: np.ndarray, mag: Optional[np.ndarray],
              movement: Optional[np.ndarray]) -> AttitudeDataset:
    dt = ensure_positive_dt(dt)
    if not (gyro.shape == accel.shape and gyro.ndim == 2 and gyro.shape[1] == 3):
        raise ValueError(f"gyro/accel must be matching (N, 3), got {gyro.shape}/{accel.shape}")
    if not np.all(np.isfinite(gyro)) or not np.all(np.isfinite(accel)):
        raise ValueError(f"{name}: IMU channels contain non-finite samples")
    valid = np.all(np.isfinite(q_raw), axis=1)
    n = np.linalg.norm(np.where(valid[:, None], q_raw, 1.0), axis=1)
    bad_norm = valid & (np.abs(n - 1.0) > 1e-3)
    if np.any(bad_norm):
        raise ValueError(f"{name}: {int(bad_norm.sum())} ground-truth quaternions "
                         "deviate from unit norm by > 1e-3")
    q = q_raw.copy()
    q[valid] = q_raw[valid] / n[valid, None]
    return AttitudeDataset(
        name=name, dt=dt, gyro=np.ascontiguousarray(gyro, dtype=float),
        accel=np.ascontiguousarray(accel, dtype=float), q_ref=q.astype(float),
        valid=valid, mag=None if mag is None else np.ascontiguousarray(mag, dtype=float),
        movement=None if movement is None else movement.astype(bool),
    )


def _load_hdf5(p: Path) -> AttitudeDataset:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "loading HDF5 datasets requires h5py: pip install 'qnav[datasets]'"
        ) from exc
    with h5py.File(p, "r") as f:
        def stack(prefix: str, axes: str) -> np.ndarray:
            return np.stack([np.asarray(f[f"{prefix}_{a}"], dtype=float) for a in axes], axis=1)

        gyro = stack("gyr", "xyz")
        accel = stack("acc", "xyz")
        q_raw = stack("opt", "abcd")
        mag = stack("mag", "xyz") if "mag_x" in f else None
        movement = np.asarray(f["movement_mask"], dtype=float) > 0.5 if "movement_mask" in f else None
        dt = float(np.median(np.asarray(f["dt"], dtype=float)))
    return _finalize(p.stem, dt, gyro, accel, q_raw, mag, movement)


def _load_npz(p: Path) -> AttitudeDataset:
    with np.load(p) as z:
        mag = np.asarray(z["mag"], dtype=float) if "mag" in z else None
        movement = np.asarray(z["movement"]) if "movement" in z else None
        return _finalize(
            p.stem, float(z["dt"]), np.asarray(z["gyro"], dtype=float),
            np.asarray(z["accel"], dtype=float), np.asarray(z["q_ref"], dtype=float),
            mag, movement,
        )
