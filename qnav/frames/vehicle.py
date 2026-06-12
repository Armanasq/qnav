"""Generic vehicle body-frame definitions and attitude helpers.

A vehicle body frame is declared by a convention token:

- ``"FRD"`` — x forward, y right, z down (aircraft/marine; pairs with NED)
- ``"FLU"`` — x forward, y left, z up (robotics/ROS REP-103; pairs with ENU)

Attitude helpers build ``q_nav_body`` from yaw/pitch/roll for the matched
pair, raising on mismatched conventions instead of guessing.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import euler as _euler
from qnav.errors import ConventionError
from qnav.frames.core import Frame

__all__ = ["body_frame", "attitude_from_ypr", "ypr_from_attitude", "MATCHED_PAIRS"]

#: Navigation-frame / body-frame pairs for which yaw-pitch-roll is standard.
MATCHED_PAIRS = {("NED", "FRD"), ("ENU", "FLU")}


def body_frame(convention: str, name: str | None = None) -> Frame:
    """Create a body :class:`Frame` with a declared axis convention."""
    axes = {
        "FRD": "x:forward y:right z:down",
        "FLU": "x:forward y:left z:up",
    }.get(convention)
    if axes is None:
        raise ConventionError(f"body convention must be 'FRD' or 'FLU', got {convention!r}")
    return Frame(name=name or convention, axes=axes, kind="body")


def attitude_from_ypr(
    yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray,
    nav: str = "NED", body: str = "FRD",
) -> np.ndarray:
    """Quaternion ``q_nav_body`` from intrinsic Z-Y′-X″ yaw/pitch/roll (radians).

    Only the matched pairs NED/FRD and ENU/FLU are accepted: for those,
    ``R_nav_body = Rz(ψ) Ry(θ) Rx(φ)`` is the standard attitude
    parameterization (zero angles ⇒ body axes aligned with nav axes).
    """
    if (nav, body) not in MATCHED_PAIRS:
        raise ConventionError(
            f"({nav}, {body}) is not a matched pair {sorted(MATCHED_PAIRS)}; "
            "convert explicitly via qnav.frames.conventions"
        )
    angles = np.stack(
        [np.asarray(yaw, dtype=float), np.asarray(pitch, dtype=float),
         np.asarray(roll, dtype=float)], axis=-1
    )
    return _euler.to_quaternion(angles, "ZYX")


def ypr_from_attitude(
    q_nav_body: np.ndarray, nav: str = "NED", body: str = "FRD",
    gimbal_tol: float = 1e-7,
):
    """Yaw, pitch, roll (radians) from ``q_nav_body`` for a matched pair.

    Returns a tuple ``(yaw, pitch, roll)``; see
    :func:`qnav.attitude.euler.from_dcm` for the gimbal-lock policy.
    """
    if (nav, body) not in MATCHED_PAIRS:
        raise ConventionError(
            f"({nav}, {body}) is not a matched pair {sorted(MATCHED_PAIRS)}"
        )
    a = _euler.from_quaternion(q_nav_body, "ZYX", gimbal_tol=gimbal_tol)
    return a[..., 0], a[..., 1], a[..., 2]
