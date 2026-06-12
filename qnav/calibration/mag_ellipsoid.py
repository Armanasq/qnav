"""Magnetometer ellipsoid calibration (hard + soft iron).

Model (:mod:`qnav.sensors.magnetometer`): ``m = A m_true + b`` with
``‖m_true‖ = B`` constant ⇒ measurements lie on the ellipsoid

    (m − b)ᵀ Q (m − b) = 1,   Q = (A Aᵀ)⁻¹/B²  (SPD)

Algorithm: linear least-squares fit of the general quadric
``xᵀMx + 2nᵀx + d = 0`` with SPD validation, then recovery of center
``b = −M⁻¹n`` and shape ``A_inv = (1/√(bᵀMb − d)) · M^{1/2}`` — applying
``m̂ = A_inv (m − b)`` maps measurements to the unit sphere (scaled by B).

The absolute intensity B and a residual rotation are **not observable** from
magnitude-only data; the returned correction makes the field spherical with
``A_inv`` symmetric (the standard, explicitly documented gauge choice).

Reference: ellipsoid-fitting calibration as discussed in the indexed survey
literature; formulation derived in ``docs/math/heading.md``.
"""

from __future__ import annotations

import numpy as np

from qnav.errors import CalibrationError

__all__ = ["fit_ellipsoid", "MagCalibration"]


class MagCalibration:
    """Result of :func:`fit_ellipsoid`: apply with :meth:`correct`."""

    def __init__(self, hard_iron: np.ndarray, soft_iron_inv: np.ndarray,
                 radius: float, rms_residual: float) -> None:
        self.hard_iron = hard_iron          # b̂ (3,)
        self.soft_iron_inv = soft_iron_inv  # symmetric (3,3); m̂ = S(m − b)
        self.radius = radius                # fitted field magnitude after correction
        self.rms_residual = rms_residual    # rms of (‖m̂‖ − radius)

    def correct(self, m_meas: np.ndarray) -> np.ndarray:
        """``m̂ = S (m − b)`` — corrected, sphere-distributed measurements."""
        d = np.asarray(m_meas, dtype=float) - self.hard_iron
        return d @ self.soft_iron_inv.T


def fit_ellipsoid(m: np.ndarray, min_points: int = 12) -> MagCalibration:
    """Fit hard/soft-iron calibration to raw measurements ``(N, 3)``.

    Requires diverse orientations (the quadric fit is rank-deficient for
    planar data); raises :class:`CalibrationError` for infeasible inputs
    (too few points, non-SPD quadric ⇒ not an ellipsoid).
    """
    m = np.asarray(m, dtype=float)
    if m.ndim != 2 or m.shape[1] != 3:
        raise ValueError("expected (N, 3) measurements")
    if m.shape[0] < min_points:
        raise CalibrationError(f"need ≥ {min_points} points, got {m.shape[0]}")

    x, y, z = m[:, 0], m[:, 1], m[:, 2]
    # design for xᵀMx + 2nᵀx + d = 0 with unknowns
    # [Mxx, Myy, Mzz, 2Mxy, 2Mxz, 2Myz, 2nx, 2ny, 2nz, d]; fix scale via lstsq
    D = np.column_stack([x*x, y*y, z*z, x*y, x*z, y*z, x, y, z, np.ones_like(x)])
    # solve D p = 0: smallest right singular vector
    _, s, Vt = np.linalg.svd(D, full_matrices=False)
    cond = s[0] / max(s[-2], 1e-300)
    p = Vt[-1]
    M = np.array([
        [p[0], p[3] / 2, p[4] / 2],
        [p[3] / 2, p[1], p[5] / 2],
        [p[4] / 2, p[5] / 2, p[2]],
    ])
    n = p[6:9] / 2.0
    d = p[9]
    # normalize sign so M is positive definite
    eigval = np.linalg.eigvalsh(M)
    if np.all(eigval < 0):
        M, n, d = -M, -n, -d
        eigval = -eigval[::-1]
    if np.any(eigval <= 0):
        raise CalibrationError(
            "fitted quadric is not an ellipsoid (insufficient orientation "
            "coverage or dominant disturbances)"
        )
    b = -np.linalg.solve(M, n)
    scale = float(b @ M @ b - d)
    if scale <= 0:
        raise CalibrationError("degenerate ellipsoid fit (non-positive scale)")
    # symmetric square root of M/scale
    w, V = np.linalg.eigh(M / scale)
    S = (V * np.sqrt(w)) @ V.T  # m̂ = S(m−b) ⇒ ‖m̂‖ = 1 on the fitted surface
    corrected = (m - b) @ S.T
    r = np.linalg.norm(corrected, axis=1)
    radius = float(r.mean())
    return MagCalibration(
        hard_iron=b, soft_iron_inv=S, radius=radius,
        rms_residual=float(np.sqrt(np.mean((r - radius) ** 2))),
    )
