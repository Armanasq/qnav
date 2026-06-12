"""Frame-checked rigid transforms.

A :class:`FrameTransform` ``T_AB`` maps points/vectors expressed in the
**source** frame B to the **target** frame A:

    p_A = R_AB · p_B + t_A^{AB}

where ``t_A^{AB}`` is the position of B's origin expressed in A. Composition
requires matching inner frames and is written ``T_AC = T_AB @ T_BC``.
Optional 6×6 covariance is over the local tangent ``[δθ, δt]`` (rotation
first), with ``δθ`` the right/local rotation error and ``δt`` expressed in
the target frame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from qnav.attitude import covariance as att_cov
from qnav.attitude import dcm as _dcm
from qnav.attitude import quaternion as quat
from qnav.attitude import so3
from qnav.errors import FrameMismatchError

__all__ = ["FrameTransform"]


@dataclass(frozen=True)
class FrameTransform:
    """Rigid transform from ``source`` frame to ``target`` frame (names).

    ``rotation`` is a unit quaternion ``q_target_source`` (scalar-first,
    Hamilton); ``translation`` is B's origin in A coordinates (may be zero
    for pure rotations); ``covariance`` an optional 6×6 over ``[δθ, δt]``;
    ``stamp`` an optional timestamp (seconds).
    """

    target: str
    source: str
    rotation: np.ndarray
    translation: np.ndarray = None  # type: ignore[assignment]
    covariance: Optional[np.ndarray] = None
    stamp: Optional[float] = None

    def __post_init__(self) -> None:
        q = np.asarray(self.rotation, dtype=float)
        if q.shape != (4,):
            raise ValueError("rotation must be a single quaternion of shape (4,)")
        if abs(np.linalg.norm(q) - 1.0) > 1e-6:
            raise ValueError("rotation quaternion must be unit-norm (|‖q‖−1| ≤ 1e-6)")
        object.__setattr__(self, "rotation", q.copy())
        t = self.translation
        t = np.zeros(3) if t is None else np.asarray(t, dtype=float)
        if t.shape != (3,):
            raise ValueError("translation must have shape (3,)")
        object.__setattr__(self, "translation", t.copy())
        if self.covariance is not None:
            P = np.asarray(self.covariance, dtype=float)
            if P.shape != (6, 6):
                raise ValueError("covariance must have shape (6, 6) over [δθ, δt]")
            object.__setattr__(self, "covariance", P.copy())

    # -- constructors ------------------------------------------------------
    @classmethod
    def identity(cls, frame: str) -> "FrameTransform":
        """Identity transform of a frame onto itself."""
        return cls(target=frame, source=frame, rotation=quat.identity())

    @classmethod
    def from_dcm(
        cls, target: str, source: str, R: np.ndarray, translation=None, **kw
    ) -> "FrameTransform":
        """Build from a rotation matrix ``R_target_source``."""
        return cls(target=target, source=source, rotation=_dcm.to_quaternion(R),
                   translation=translation, **kw)

    # -- core ops ----------------------------------------------------------
    @property
    def dcm(self) -> np.ndarray:
        """Rotation matrix ``R_target_source``."""
        return _dcm.from_quaternion(self.rotation)

    def inverse(self) -> "FrameTransform":
        """``T_BA`` with ``R_BA = R_ABᵀ``, ``t_B^{BA} = −R_BA t_A^{AB}``.

        Covariance transport (first order, right/local convention; verified
        by finite differences and Monte-Carlo in
        ``tests/test_frame_transforms.py``): perturbing
        ``R̃ = R·Exp(δθ)``, ``t̃ = t + δt`` gives

        - ``δθ_inv = −R_AB δθ``  (adjoint flip of the local rotation error)
        - ``δt_inv = [t_inv]× δθ − R_BA δt``

        so ``P_inv = J P Jᵀ`` with ``J = [[−R_AB, 0], [[t_inv]×, −R_BA]]``.
        """
        q_inv = quat.conjugate(self.rotation)
        R_BA = _dcm.from_quaternion(q_inv)
        t_inv = -R_BA @ self.translation
        P = None
        if self.covariance is not None:
            R_AB = self.dcm
            J = np.zeros((6, 6))
            J[:3, :3] = -R_AB
            J[3:, :3] = so3.hat(t_inv)
            J[3:, 3:] = -R_BA
            P = J @ self.covariance @ J.T
        return FrameTransform(
            target=self.source, source=self.target, rotation=q_inv,
            translation=t_inv, covariance=P, stamp=self.stamp,
        )

    def compose(self, other: "FrameTransform") -> "FrameTransform":
        """``T_AC = T_AB.compose(T_BC)`` — inner frames must match.

        ``R_AC = R_AB R_BC``, ``t_A^{AC} = t_A^{AB} + R_AB t_B^{BC}``.
        Covariances (if both present) are composed assuming independence.
        """
        if self.source != other.target:
            raise FrameMismatchError(
                f"cannot compose {self.target}<-{self.source} with "
                f"{other.target}<-{other.source}: inner frames differ "
                f"({self.source!r} != {other.target!r})"
            )
        q = quat.normalize(quat.mul(self.rotation, other.rotation))
        R_AB = self.dcm
        t = self.translation + R_AB @ other.translation
        P = None
        if self.covariance is not None and other.covariance is not None:
            R_BC = other.dcm
            # δθ_AC = R_BCᵀ δθ_AB + δθ_BC ; δt_AC = δt_AB + R_AB δt_BC
            #         − R_AB [t_B^{BC}]× δθ_AB? (rotation error of AB moves BC's lever arm)
            J1 = np.zeros((6, 6))
            J1[:3, :3] = R_BC.T
            J1[3:, 3:] = np.eye(3)
            J1[3:, :3] = -R_AB @ so3.hat(other.translation)
            J2 = np.zeros((6, 6))
            J2[:3, :3] = np.eye(3)
            J2[3:, 3:] = R_AB
            P = J1 @ self.covariance @ J1.T + J2 @ other.covariance @ J2.T
        stamp = self.stamp if self.stamp is not None else other.stamp
        return FrameTransform(
            target=self.target, source=other.source, rotation=q,
            translation=t, covariance=P, stamp=stamp,
        )

    def __matmul__(self, other: "FrameTransform") -> "FrameTransform":
        return self.compose(other)

    # -- application -------------------------------------------------------
    def apply_vector(self, v: np.ndarray) -> np.ndarray:
        """Transform free vector(s) (rotation only): ``v_A = R_AB v_B``.

        Use for velocities, magnetic fields, angular rates — quantities
        unaffected by the origin shift.
        """
        return quat.rotate_vector(self.rotation, v)

    def apply_point(self, p: np.ndarray) -> np.ndarray:
        """Transform point(s): ``p_A = R_AB p_B + t``."""
        return quat.rotate_vector(self.rotation, p) + self.translation

    def apply_covariance(self, P_v: np.ndarray) -> np.ndarray:
        """Rotate a 3×3 vector covariance into the target frame: ``R P Rᵀ``."""
        return att_cov.transform_vector_covariance(P_v, self.dcm)

    def with_stamp(self, stamp: float) -> "FrameTransform":
        """Copy with a new timestamp."""
        return replace(self, stamp=stamp)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"FrameTransform({self.target} <- {self.source}, "
            f"q={np.round(self.rotation, 6).tolist()}, "
            f"t={np.round(self.translation, 6).tolist()})"
        )
