"""Dataset loaders: CSV (NumPy-only) and pandas DataFrames -> validated IMU arrays.

Output contract (:class:`ImuData`): monotonically increasing timestamps [s],
gyro [rad/s] and accel [m/s²] as ``(N, 3)``, optional magnetometer ``(N, 3)``
in the caller's units. Loading *rejects* non-finite rows and non-monotonic
timestamps instead of silently repairing them; pass ``drop_bad_rows=True``
to opt into dropping (the count is reported on the result).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from qnav._validate import ensure_monotonic

__all__ = ["ImuData", "imu_from_dataframe", "load_imu_csv"]


@dataclass(frozen=True)
class ImuData:
    """A validated, time-ordered IMU recording."""

    t: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    mag: Optional[np.ndarray] = None
    dropped_rows: int = 0

    def __len__(self) -> int:
        return int(self.t.shape[0])


def _assemble(t: np.ndarray, gyro: np.ndarray, accel: np.ndarray,
              mag: Optional[np.ndarray], drop_bad_rows: bool) -> ImuData:
    cols = [t[:, None], gyro, accel] + ([mag] if mag is not None else [])
    finite = np.all(np.isfinite(np.concatenate(cols, axis=1)), axis=1)
    dropped = int((~finite).sum())
    if dropped and not drop_bad_rows:
        raise ValueError(
            f"{dropped} rows contain non-finite values; pass drop_bad_rows=True "
            "to drop them explicitly"
        )
    if dropped:
        t, gyro, accel = t[finite], gyro[finite], accel[finite]
        mag = mag[finite] if mag is not None else None
    ensure_monotonic(t, "timestamps")
    return ImuData(t=t, gyro=gyro, accel=accel, mag=mag, dropped_rows=dropped)


def load_imu_csv(
    path: Union[str, Path],
    *,
    time_col: int = 0,
    gyro_cols: Sequence[int] = (1, 2, 3),
    accel_cols: Sequence[int] = (4, 5, 6),
    mag_cols: Optional[Sequence[int]] = None,
    delimiter: str = ",",
    skip_header: int = 1,
    drop_bad_rows: bool = False,
) -> ImuData:
    """Load an IMU log from CSV using NumPy only (no pandas dependency).

    Column indices are explicit — qnav never guesses which column is which.
    """
    raw = np.genfromtxt(str(path), delimiter=delimiter, skip_header=skip_header,
                        dtype=float)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.size == 0:
        raise ValueError(f"no data rows in {path}")
    t = raw[:, time_col]
    gyro = raw[:, list(gyro_cols)]
    accel = raw[:, list(accel_cols)]
    mag = raw[:, list(mag_cols)] if mag_cols is not None else None
    return _assemble(t, gyro, accel, mag, drop_bad_rows)


def imu_from_dataframe(
    df: object,
    *,
    time_col: str = "t",
    gyro_cols: Sequence[str] = ("gx", "gy", "gz"),
    accel_cols: Sequence[str] = ("ax", "ay", "az"),
    mag_cols: Optional[Sequence[str]] = None,
    drop_bad_rows: bool = False,
) -> ImuData:
    """Convert a pandas DataFrame into :class:`ImuData` (pandas optional)."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "imu_from_dataframe requires pandas: pip install 'qnav[interop]'"
        ) from exc
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected pandas DataFrame, got {type(df).__name__}")
    t = df[time_col].to_numpy(dtype=float)
    gyro = df[list(gyro_cols)].to_numpy(dtype=float)
    accel = df[list(accel_cols)].to_numpy(dtype=float)
    mag = df[list(mag_cols)].to_numpy(dtype=float) if mag_cols is not None else None
    return _assemble(t, gyro, accel, mag, drop_bad_rows)
