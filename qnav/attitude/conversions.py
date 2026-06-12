"""Single hub for conversions between all attitude representations.

Representations and canonical shapes:

- ``quat``: unit quaternion ``[w, x, y, z]`` — ``(..., 4)``
- ``dcm``: rotation matrix ``R_AB`` — ``(..., 3, 3)``
- ``euler``: angles for an explicit sequence — ``(..., 3)`` (radians)
- ``rotvec``: rotation vector ``θ·u`` — ``(..., 3)``
- ``axis_angle``: tuple ``(unit axis (...,3), angle (...))``
- ``gibbs``: Rodrigues/Gibbs vector — ``(..., 3)``
- ``mrp``: modified Rodrigues parameters — ``(..., 3)``

Every pairwise conversion routes through the numerically safest path
(quaternion or DCM core). Round-trip, singularity, and convention tests live
in ``tests/test_conversions.py``.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import dcm as _dcm
from qnav.attitude import euler as _euler
from qnav.attitude import mrp as _mrp
from qnav.attitude import quaternion as _quat
from qnav.attitude import rotvec as _rotvec

__all__ = [
    "quat_to_dcm", "dcm_to_quat",
    "quat_to_euler", "euler_to_quat",
    "dcm_to_euler", "euler_to_dcm",
    "quat_to_rotvec", "rotvec_to_quat",
    "dcm_to_rotvec", "rotvec_to_dcm",
    "quat_to_axis_angle", "axis_angle_to_quat",
    "quat_to_gibbs", "gibbs_to_quat",
    "quat_to_mrp", "mrp_to_quat",
    "convert",
]

quat_to_dcm = _dcm.from_quaternion
dcm_to_quat = _dcm.to_quaternion
quat_to_euler = _euler.from_quaternion
euler_to_quat = _euler.to_quaternion
dcm_to_euler = _euler.from_dcm
euler_to_dcm = _euler.to_dcm
quat_to_rotvec = _quat.log
rotvec_to_quat = _quat.exp
dcm_to_rotvec = _rotvec.from_dcm
rotvec_to_dcm = _rotvec.to_dcm
quat_to_gibbs = _mrp.gibbs_from_quaternion
gibbs_to_quat = _mrp.gibbs_to_quaternion
quat_to_mrp = _mrp.from_quaternion
mrp_to_quat = _mrp.to_quaternion


def quat_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit axis and angle θ ∈ [0, π] of a unit quaternion."""
    return _quat.axis(q), _quat.angle(q)


def axis_angle_to_quat(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Quaternion from unit axis and angle (axis normalized defensively)."""
    return _quat.exp(_rotvec.from_axis_angle(axis, angle))


_TO_QUAT = {
    "quat": lambda x, **kw: np.asarray(x, dtype=float),
    "dcm": lambda x, **kw: dcm_to_quat(x),
    "euler": lambda x, seq="ZYX", **kw: euler_to_quat(x, seq),
    "rotvec": lambda x, **kw: rotvec_to_quat(x),
    "gibbs": lambda x, **kw: gibbs_to_quat(x),
    "mrp": lambda x, **kw: mrp_to_quat(x),
}

_FROM_QUAT = {
    "quat": lambda q, **kw: q,
    "dcm": lambda q, **kw: quat_to_dcm(q),
    "euler": lambda q, seq="ZYX", **kw: quat_to_euler(q, seq),
    "rotvec": lambda q, **kw: quat_to_rotvec(q),
    "gibbs": lambda q, **kw: quat_to_gibbs(q),
    "mrp": lambda q, **kw: quat_to_mrp(q),
}


def convert(x, src: str, dst: str, *, seq: str = "ZYX"):
    """Convert ``x`` between any two named representations (via the quaternion hub).

    ``seq`` applies only when ``src`` or ``dst`` is ``"euler"``.
    """
    if src not in _TO_QUAT:
        raise ValueError(f"unknown source representation {src!r}; one of {sorted(_TO_QUAT)}")
    if dst not in _FROM_QUAT:
        raise ValueError(f"unknown target representation {dst!r}; one of {sorted(_FROM_QUAT)}")
    q = _TO_QUAT[src](x, seq=seq)
    return _FROM_QUAT[dst](q, seq=seq)
