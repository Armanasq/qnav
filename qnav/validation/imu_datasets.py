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

__all__ = ["AttitudeDataset", "available_datasets", "data_root", "load_attitude_dataset", "verify_conventions"]


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


#: plausibility bounds enforced at load time (values outside these are data
#: errors, not motion: consumer/industrial IMUs and optical mocap trials)
_RATE_HZ_RANGE = (20.0, 2000.0)
_GYRO_MAX_RAD_S = 70.0           # ~4000 deg/s, beyond any benchmark motion
_ACCEL_MAX_M_S2 = 400.0          # ~40 g


def _finalize(name: str, dt: float, gyro: np.ndarray, accel: np.ndarray,
              q_raw: np.ndarray, mag: Optional[np.ndarray],
              movement: Optional[np.ndarray]) -> AttitudeDataset:
    dt = ensure_positive_dt(dt)
    if not _RATE_HZ_RANGE[0] <= 1.0 / dt <= _RATE_HZ_RANGE[1]:
        raise ValueError(f"{name}: implausible sampling rate {1.0 / dt:.1f} Hz")
    if not (gyro.shape == accel.shape and gyro.ndim == 2 and gyro.shape[1] == 3):
        raise ValueError(f"gyro/accel must be matching (N, 3), got {gyro.shape}/{accel.shape}")
    n_samples = gyro.shape[0]
    if q_raw.shape != (n_samples, 4):
        raise ValueError(f"{name}: ground truth length {q_raw.shape} != IMU length {n_samples}")
    if mag is not None and mag.shape != (n_samples, 3):
        raise ValueError(f"{name}: magnetometer length {mag.shape} != IMU length {n_samples}")
    if movement is not None and movement.shape[0] != n_samples:
        raise ValueError(f"{name}: movement mask length {movement.shape[0]} != {n_samples}")
    if not np.all(np.isfinite(gyro)) or not np.all(np.isfinite(accel)):
        raise ValueError(f"{name}: IMU channels contain non-finite samples")
    if mag is not None and not np.all(np.isfinite(mag)):
        raise ValueError(f"{name}: magnetometer contains non-finite samples")
    if float(np.abs(gyro).max()) > _GYRO_MAX_RAD_S:
        raise ValueError(f"{name}: gyro exceeds {_GYRO_MAX_RAD_S} rad/s — wrong units?")
    if float(np.abs(accel).max()) > _ACCEL_MAX_M_S2:
        raise ValueError(f"{name}: accel exceeds {_ACCEL_MAX_M_S2} m/s² — wrong units?")
    for label, arr in (("gyro", gyro), ("accel", accel)) + (
            (("mag", mag),) if mag is not None else ()):
        if np.all(np.ptp(arr, axis=0) == 0.0):
            raise ValueError(f"{name}: {label} channels are constant — corrupt export?")

    valid = np.all(np.isfinite(q_raw), axis=1)
    if not np.any(valid):
        raise ValueError(f"{name}: no valid ground-truth samples")
    n = np.linalg.norm(np.where(valid[:, None], q_raw, 1.0), axis=1)
    bad_norm = valid & (np.abs(n - 1.0) > 1e-3)
    if np.any(bad_norm):
        raise ValueError(f"{name}: {int(bad_norm.sum())} ground-truth quaternions "
                         "deviate from unit norm by > 1e-3")
    q = q_raw.copy()
    q[valid] = q_raw[valid] / n[valid, None]
    # quaternion double cover: enforce sign continuity over valid runs so
    # consumers can difference consecutive samples safely
    vidx = np.flatnonzero(valid)
    for i, j in zip(vidx[:-1], vidx[1:]):
        if float(q[i] @ q[j]) < 0.0:
            q[j] = -q[j]
    return AttitudeDataset(
        name=name, dt=dt, gyro=np.ascontiguousarray(gyro, dtype=float),
        accel=np.ascontiguousarray(accel, dtype=float), q_ref=q.astype(float),
        valid=valid, mag=None if mag is None else np.ascontiguousarray(mag, dtype=float),
        movement=None if movement is None else movement.astype(bool),
    )


def verify_conventions(ds: AttitudeDataset, n_check: int = 3000) -> dict:
    """Machine-readable convention verification for one trial.

    Checks the loader's documented conventions against physics and returns a
    JSON-serializable report; ``report["ok"]`` is the overall verdict:

    - ``gyro_consistency``: residual between the body rate reconstructed from
      consecutive ground-truth quaternions and the measured gyro, relative to
      the rate signal level (scalar-first ``q_ref_body`` convention),
    - ``gravity_residual_m_s2``: mean residual of ``accel ≈ R(q)ᵀ [0,0,g]``
      (z-up reference convention),
    - ground-truth gap statistics.
    """
    from qnav.attitude import quaternion as quat

    vidx = np.flatnonzero(ds.valid)
    step = max(1, min(n_check, vidx.size - 1) // 300)
    pairs = [(k, k + 1) for k in vidx[:n_check:step]
             if k + 1 < len(ds) and ds.valid[k + 1]]

    # hypothesis test: the documented convention (q_ref_body) must explain
    # the gyro strictly better than its conjugate — robust to noisy ground
    # truth, unlike an absolute residual threshold.
    def rate_residual(conjugated: bool) -> float:
        res = []
        for i, j in pairs:
            qi = quat.conjugate(ds.q_ref[i]) if conjugated else ds.q_ref[i]
            qj = quat.conjugate(ds.q_ref[j]) if conjugated else ds.q_ref[j]
            w_est = quat.log(quat.mul(quat.conjugate(qi), qj)) / ds.dt
            res.append(np.linalg.norm(w_est - ds.gyro[i]))
        return float(np.mean(res)) if res else float("nan")

    gyro_resid = rate_residual(conjugated=False)
    gyro_resid_alt = rate_residual(conjugated=True)
    signal = float(np.abs(ds.gyro[vidx[:n_check]]).mean())

    ks = vidx[:n_check:max(1, len(vidx) // 60)]
    g_up = np.stack([quat.rotate_frame(ds.q_ref[k], np.array([0.0, 0.0, 9.81]))
                     for k in ks])
    grav_resid = float(np.linalg.norm(g_up - ds.accel[ks], axis=1).mean())
    grav_resid_alt = float(np.linalg.norm(-g_up - ds.accel[ks], axis=1).mean())

    gaps = np.diff(np.flatnonzero(ds.valid))
    longest_gap_s = float((gaps.max() - 1) * ds.dt) if gaps.size else 0.0

    # verdicts discriminate *conventions* via competing hypotheses; they do
    # not grade ground-truth quality (noisy/interpolated GT raises both
    # hypotheses' residuals equally — RMSE metrics capture quality instead).
    gravity_confirmed = grav_resid < 0.5 * grav_resid_alt
    if gyro_resid < 0.8 * gyro_resid_alt or gyro_resid < 0.1:
        gyro_verdict = "confirmed"
    elif gyro_resid > 1.25 * gyro_resid_alt:
        gyro_verdict = "contradicted"
    else:
        gyro_verdict = "inconclusive"

    report = {
        "dataset": ds.name,
        "rate_hz": ds.rate_hz,
        "n_samples": len(ds),
        "gyro_consistency": {
            "residual_rad_s": gyro_resid,
            "residual_conjugate_rad_s": gyro_resid_alt,
            "signal_rad_s": signal,
            "verdict": gyro_verdict,
        },
        "gravity": {
            "residual_m_s2": grav_resid,
            "residual_z_down_m_s2": grav_resid_alt,
            "confirmed": bool(gravity_confirmed),
        },
        "gt_valid_fraction": float(ds.valid.mean()),
        "gt_longest_gap_s": longest_gap_s,
        "has_mag": ds.mag is not None,
    }
    report["ok"] = bool(gravity_confirmed and gyro_verdict != "contradicted")
    return report


def _load_hdf5(p: Path) -> AttitudeDataset:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "loading HDF5 datasets requires h5py: pip install 'qnav[datasets]'"
        ) from exc
    with h5py.File(p, "r") as f:
        if "gyr_x" in f:            # BROAD layout: per-axis channels
            def stack(prefix: str, axes: str) -> np.ndarray:
                return np.stack(
                    [np.asarray(f[f"{prefix}_{a}"], dtype=float) for a in axes], axis=1)

            gyro = stack("gyr", "xyz")
            accel = stack("acc", "xyz")
            q_raw = stack("opt", "abcd")
            mag = stack("mag", "xyz") if "mag_x" in f else None
            movement = (np.asarray(f["movement_mask"], dtype=float) > 0.5
                        if "movement_mask" in f else None)
            dt = float(np.median(np.asarray(f["dt"], dtype=float)))
        elif "gyr" in f and "quat" in f:   # DIODEM layout: (N,3)/(N,4) matrices
            gyro = np.asarray(f["gyr"], dtype=float)
            accel = np.asarray(f["acc"], dtype=float)
            q_raw = np.asarray(f["quat"], dtype=float)
            mag = np.asarray(f["mag"], dtype=float) if "mag" in f else None
            movement = None
            t = np.asarray(f["timestamps"], dtype=float)
            dts = np.diff(t)
            if dts.size and (dts.min() <= 0 or
                             np.median(np.abs(dts - np.median(dts))) > 0.2 * np.median(dts)):
                raise ValueError(f"{p.stem}: timestamps non-monotonic or jitter > 20% "
                                 "of the nominal interval — not a uniform-rate trial")
            dt = float(np.median(dts)) if dts.size else 0.0
            _check_sidecar(p)
        else:
            raise ValueError(
                f"{p.name}: unrecognized HDF5 layout (expected BROAD per-axis "
                "channels or DIODEM matrix datasets)")
    return _finalize(p.stem, dt, gyro, accel, q_raw, mag, movement)


def _check_sidecar(p: Path) -> None:
    """DIODEM sidecar metadata must agree with the loader's conventions."""
    import json

    sidecar = p.with_suffix(".json")
    if not sidecar.exists():
        return
    meta = json.loads(sidecar.read_text())
    expected = {"quat_order": "wxyz", "quat_dir": "body_to_world", "world_frame": "ENU"}
    for key, want in expected.items():
        got = meta.get(key)
        if got is not None and got != want:
            raise ValueError(f"{p.stem}: sidecar declares {key}={got!r}, loader "
                             f"requires {want!r}")
    n_meta = meta.get("n_samples")
    if n_meta is not None:
        import h5py

        with h5py.File(p, "r") as f:
            n_file = int(f["gyr"].shape[0])
        if int(n_meta) != n_file:
            raise ValueError(f"{p.stem}: sidecar n_samples={n_meta} != file {n_file}")


def _load_npz(p: Path) -> AttitudeDataset:
    with np.load(p) as z:
        mag = np.asarray(z["mag"], dtype=float) if "mag" in z else None
        movement = np.asarray(z["movement"]) if "movement" in z else None
        return _finalize(
            p.stem, float(z["dt"]), np.asarray(z["gyro"], dtype=float),
            np.asarray(z["accel"], dtype=float), np.asarray(z["q_ref"], dtype=float),
            mag, movement,
        )
