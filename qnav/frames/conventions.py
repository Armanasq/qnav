"""Convention bridges between common frame pairs (NED/ENU, FRD/FLU) and
attitude re-expression helpers.

These are the only sanctioned ways to move between conventions — call them
explicitly instead of hand-rolling permutation matrices at call sites.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import dcm as _dcm
from qnav.attitude import quaternion as quat
from qnav.frames.earth import DCM_ENU_NED

__all__ = [
    "DCM_FLU_FRD", "ned_to_enu", "enu_to_ned", "frd_to_flu", "flu_to_frd",
    "attitude_ned_frd_to_enu_flu", "attitude_enu_flu_to_ned_frd",
]

#: v_FLU = DCM_FLU_FRD @ v_FRD (x forward kept; y, z flipped). Involutory.
DCM_FLU_FRD = np.diag([1.0, -1.0, -1.0])


def ned_to_enu(v: np.ndarray) -> np.ndarray:
    """Re-express local-tangent vector(s): ``v_ENU = P·v_NED`` (P involutory)."""
    return np.asarray(v, dtype=float) @ DCM_ENU_NED.T


def enu_to_ned(v: np.ndarray) -> np.ndarray:
    """Inverse of :func:`ned_to_enu` (the permutation is its own inverse)."""
    return ned_to_enu(v)


def frd_to_flu(v: np.ndarray) -> np.ndarray:
    """Re-express body vector(s) from FRD to FLU axes."""
    return np.asarray(v, dtype=float) @ DCM_FLU_FRD.T


def flu_to_frd(v: np.ndarray) -> np.ndarray:
    """Inverse of :func:`frd_to_flu` (involutory)."""
    return frd_to_flu(v)


def attitude_ned_frd_to_enu_flu(q_ned_frd: np.ndarray) -> np.ndarray:
    """Convert an attitude quaternion ``q_NED_FRD`` to ``q_ENU_FLU``.

    ``R_ENU_FLU = R_ENU_NED · R_NED_FRD · R_FRD_FLU`` — both the navigation
    and the body convention change. Useful when moving between aerospace
    (NED/FRD) and ROS (ENU/FLU) stacks.
    """
    q_enu_ned = _dcm.to_quaternion(DCM_ENU_NED)
    q_frd_flu = _dcm.to_quaternion(DCM_FLU_FRD)  # involutory: same both ways
    return quat.mul(q_enu_ned, quat.mul(q_ned_frd, q_frd_flu))


def attitude_enu_flu_to_ned_frd(q_enu_flu: np.ndarray) -> np.ndarray:
    """Inverse of :func:`attitude_ned_frd_to_enu_flu`."""
    q_ned_enu = _dcm.to_quaternion(DCM_ENU_NED)
    q_flu_frd = _dcm.to_quaternion(DCM_FLU_FRD)
    return quat.mul(q_ned_enu, quat.mul(q_enu_flu, q_flu_frd))
