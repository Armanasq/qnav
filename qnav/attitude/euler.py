"""Euler angles with explicit sequence, intrinsic/extrinsic interpretation, radians.

Sequence grammar
----------------
``seq`` is a 3-letter string over ``{X, Y, Z}`` (uppercase = **intrinsic**,
rotations about the moving frame's axes, applied left-to-right) or over
``{x, y, z}`` (lowercase = **extrinsic**, fixed-frame axes). Adjacent letters
must differ. Examples:

- ``"ZYX"`` (default): intrinsic yaw–pitch–roll, ``R_AB = Rz(ψ) Ry(θ) Rx(φ)``.
- ``"ZXZ"``: classical proper Euler sequence.
- ``"xyz"``: extrinsic; equals intrinsic ``"ZYX"`` with angles reversed.

Gimbal lock: for Tait–Bryan sequences the middle angle is ±π/2; for proper
sequences 0 or π. Within ``gimbal_tol`` of these, a :class:`GimbalLockWarning`
is issued and the **third angle is set to zero** (the lost degree of freedom is
assigned entirely to the first angle) — deterministic, documented behavior.

References: attitude survey (``__data/Efficient Attitude Estimators .../
attitudesurvey.tex``); Hashim SO(3) survey. See ``docs/math/euler_angles.md``.
"""

from __future__ import annotations

import warnings

import numpy as np

from qnav.attitude import dcm as _dcm
from qnav.errors import ConventionError, GimbalLockWarning

__all__ = ["to_dcm", "from_dcm", "to_quaternion", "from_quaternion", "ALLOWED_SEQUENCES"]

_AXES = {"X": 0, "Y": 1, "Z": 2}
_PRINCIPAL = {"X": _dcm.rot_x, "Y": _dcm.rot_y, "Z": _dcm.rot_z}

ALLOWED_SEQUENCES = tuple(
    a + b + c
    for a in "XYZ" for b in "XYZ" for c in "XYZ"
    if a != b and b != c
)


def _parse(seq: str) -> tuple[str, bool]:
    """Return (uppercase sequence, intrinsic?) after validation."""
    if not isinstance(seq, str) or len(seq) != 3:
        raise ConventionError(f"Euler sequence must be a 3-letter string, got {seq!r}")
    if seq.isupper():
        intrinsic = True
    elif seq.islower():
        intrinsic = False
    else:
        raise ConventionError(
            f"Euler sequence must be all-uppercase (intrinsic) or all-lowercase "
            f"(extrinsic), got {seq!r}"
        )
    up = seq.upper()
    if up not in ALLOWED_SEQUENCES:
        raise ConventionError(f"invalid Euler sequence {seq!r}: adjacent axes must differ")
    return up, intrinsic


def to_dcm(angles: np.ndarray, seq: str = "ZYX") -> np.ndarray:
    """DCM ``R_AB`` from Euler angles (radians), shape ``(..., 3)`` → ``(..., 3, 3)``.

    Intrinsic ``"ABC"`` with angles ``(α, β, γ)`` gives ``R = R_A(α) R_B(β) R_C(γ)``;
    extrinsic ``"abc"`` gives ``R = R_C(γ) R_B(β) R_A(α)`` — the standard duality
    ``intrinsic ABC(α,β,γ) ≡ extrinsic cba(γ,β,α)``.
    """
    up, intrinsic = _parse(seq)
    angles = np.asarray(angles, dtype=float)
    if angles.shape[-1] != 3:
        raise ValueError("angles must have shape (..., 3)")
    a, b, c = angles[..., 0], angles[..., 1], angles[..., 2]
    R1 = _PRINCIPAL[up[0]](a)
    R2 = _PRINCIPAL[up[1]](b)
    R3 = _PRINCIPAL[up[2]](c)
    if intrinsic:
        return R1 @ R2 @ R3
    return R3 @ R2 @ R1


def from_dcm(R: np.ndarray, seq: str = "ZYX", gimbal_tol: float = 1e-7) -> np.ndarray:
    """Euler angles (radians) of a DCM for the given sequence.

    Tait–Bryan middle angle is returned in ``[−π/2, π/2]``; proper-sequence
    middle angle in ``[0, π]``. First/third angles in ``(−π, π]``. Near gimbal
    lock the third angle is set to zero (see module docstring).
    """
    up, intrinsic = _parse(seq)
    R = np.asarray(R, dtype=float)
    if not intrinsic:
        # extrinsic abc(α,β,γ) == intrinsic CBA(γ,β,α)
        rev = from_dcm(R, up[::-1], gimbal_tol=gimbal_tol)
        return rev[..., ::-1]

    batch: tuple[int, ...] = R.shape[:-2]
    Rf = R.reshape((-1, 3, 3))
    n_flat = int(Rf.shape[0])
    i, j, k = (_AXES[ch] for ch in up)
    tait_bryan = up[0] != up[2]
    out = np.empty((n_flat, 3))
    # parity of the axis pair (i, j): +1 if (i, j) is a cyclic step
    eps = 1.0 if (j - i) % 3 == 1 else -1.0
    Rj_principal = _PRINCIPAL[up[1]]
    # cyclic successors of axis i: Ri(a)[p, p] = cos a, Ri(a)[q, p] = sin a
    p, qx = (i + 1) % 3, (i + 2) % 3

    locked_any = False
    for n in range(n_flat):
        m = Rf[n]
        if tait_bryan:
            # R = Ri(a) Rj(b) Rk(c), all axes distinct
            sb = eps * m[i, k]
            sb = np.clip(sb, -1.0, 1.0)
            b = np.arcsin(sb)
            locked = 0.5 * np.pi - abs(b) <= gimbal_tol
            if not locked:
                a = np.arctan2(-eps * m[j, k], m[k, k])
                c = np.arctan2(-eps * m[i, j], m[i, i])
        else:
            # proper sequence R = Ri(a) Rj(b) Ri(c)
            cb = np.clip(m[i, i], -1.0, 1.0)
            b = np.arccos(cb)
            ax_l = 3 - i - j  # the unused axis
            locked = min(b, np.pi - b) <= gimbal_tol
            if not locked:
                a = np.arctan2(m[j, i], -eps * m[ax_l, i])
                c = np.arctan2(m[i, j], eps * m[i, ax_l])
        if locked:
            # only (a ± c) is observable; assign it all to a by fixing c = 0,
            # then R ≈ Ri(a)·Rj(b)  ⇒  Ri(a) = R·Rj(b)ᵀ  (generic, axis-safe)
            locked_any = True
            c = 0.0
            Ria = m @ Rj_principal(b).T
            a = np.arctan2(Ria[qx, p], Ria[p, p])
        out[n] = (a, b, c)

    if locked_any:
        warnings.warn(
            f"gimbal lock detected for sequence {seq!r}; third angle set to 0",
            GimbalLockWarning, stacklevel=2,
        )
    return out.reshape(batch + (3,))


def to_quaternion(angles: np.ndarray, seq: str = "ZYX") -> np.ndarray:
    """Unit quaternion ``[w, x, y, z]`` from Euler angles (via the DCM path)."""
    return _dcm.to_quaternion(to_dcm(angles, seq))


def from_quaternion(q: np.ndarray, seq: str = "ZYX", gimbal_tol: float = 1e-7) -> np.ndarray:
    """Euler angles from a unit quaternion (via the DCM path)."""
    return from_dcm(_dcm.from_quaternion(q), seq, gimbal_tol=gimbal_tol)
