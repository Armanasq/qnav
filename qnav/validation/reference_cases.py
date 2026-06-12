"""Hand-checkable reference cases with known exact answers.

Every case is independent of qnav's own conversion code — values are written
out explicitly so a regression in the core cannot silently regenerate its own
expectations.
"""

from __future__ import annotations

import numpy as np

__all__ = ["QUATERNION_DCM_CASES", "EULER_ZYX_CASES"]

_S2 = np.sqrt(0.5)

#: (name, q [w,x,y,z], R) — exact pairs.
QUATERNION_DCM_CASES = [
    ("identity", np.array([1.0, 0.0, 0.0, 0.0]), np.eye(3)),
    ("90deg_z", np.array([_S2, 0.0, 0.0, _S2]),
     np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),
    ("90deg_x", np.array([_S2, _S2, 0.0, 0.0]),
     np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])),
    ("90deg_y", np.array([_S2, 0.0, _S2, 0.0]),
     np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])),
    ("180deg_z", np.array([0.0, 0.0, 0.0, 1.0]),
     np.diag([-1.0, -1.0, 1.0])),
    ("180deg_x", np.array([0.0, 1.0, 0.0, 0.0]),
     np.diag([1.0, -1.0, -1.0])),
    ("120deg_111", np.array([0.5, 0.5, 0.5, 0.5]),
     np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])),
]

#: (name, (yaw, pitch, roll) [rad], q [w,x,y,z]) for intrinsic ZYX.
EULER_ZYX_CASES = [
    ("zero", (0.0, 0.0, 0.0), np.array([1.0, 0.0, 0.0, 0.0])),
    ("yaw90", (np.pi / 2, 0.0, 0.0), np.array([_S2, 0.0, 0.0, _S2])),
    ("pitch90", (0.0, np.pi / 2, 0.0), np.array([_S2, 0.0, _S2, 0.0])),
    ("roll90", (0.0, 0.0, np.pi / 2), np.array([_S2, _S2, 0.0, 0.0])),
    # yaw 90 then pitch 90 (intrinsic): q = qz(90) ⊗ qy(90)
    ("yaw90_pitch90", (np.pi / 2, np.pi / 2, 0.0),
     np.array([0.5, -0.5, 0.5, 0.5])),
]
