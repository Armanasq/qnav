"""Aerospace frame helpers: NED navigation, FRD body, wind/stability axes.

References: attitude survey (``__data/Efficient Attitude Estimators ...``);
standard flight-dynamics texts. Angles in radians.
"""

from __future__ import annotations

import numpy as np

from qnav.frames.core import Frame

__all__ = ["AIRCRAFT_BODY", "dcm_body_to_stability", "dcm_body_to_wind"]

AIRCRAFT_BODY = Frame("FRD", "x:forward(nose) y:right(starboard) z:down", "body")


def dcm_body_to_stability(alpha: np.ndarray) -> np.ndarray:
    """``R_S_B`` for angle of attack α: stability axes are body axes rotated by
    −α about body y (x_S along the projection of airspeed in the x-z plane).

    ``v_S = R_S_B v_B`` with ``R_S_B = Ry(α)`` … explicitly
    ``[[cosα, 0, sinα], [0, 1, 0], [−sinα, 0, cosα]]``.
    """
    return _stab(np.asarray(alpha, dtype=float))


def _stab(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(np.shape(a) + (3, 3))
    R[..., 0, 0], R[..., 0, 2] = c, s
    R[..., 1, 1] = 1.0
    R[..., 2, 0], R[..., 2, 2] = -s, c
    return R


def dcm_body_to_wind(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """``R_W_B = Rz(β)ᵀ?`` — explicitly: wind axes from body via α then sideslip β:
    ``R_W_B = R_W_S(β) · R_S_B(α)`` with ``R_W_S = [[cosβ, sinβ, 0],
    [−sinβ, cosβ, 0], [0, 0, 1]]``.
    """
    a = np.asarray(alpha, dtype=float)
    b = np.asarray(beta, dtype=float)
    cb, sb = np.cos(b), np.sin(b)
    RWS = np.zeros(np.shape(b) + (3, 3))
    RWS[..., 0, 0], RWS[..., 0, 1] = cb, sb
    RWS[..., 1, 0], RWS[..., 1, 1] = -sb, cb
    RWS[..., 2, 2] = 1.0
    return RWS @ _stab(a)
