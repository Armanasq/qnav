"""Total-state quaternion EKF (for reference and comparison).

State: the quaternion ``q_nav_body`` itself (4 components), covariance P
4×4 over raw quaternion components. Known structural weaknesses — the unit
constraint makes P singular in the radial direction and renormalization is
extrinsic to the filter — are documented, mitigated (renormalization +
covariance projection each step), and **measured** against the ESKF in the
benchmarks. For production use prefer :class:`qnav.filters.eskf.Eskf`.

Process model: ``q_{k+1} = q_k ⊗ Exp(ω dt)`` ⇒ ``F = [Exp(ω dt)]_R`` (right-
multiplication matrix). Gyro noise maps through
``G = ∂(q ⊗ Exp(φ))/∂φ|₀ = ½ [q]_L Ξ`` with Ξ the 4×3 pure-quaternion
embedding; ``Qd = G (σ_g² dt I₃) Gᵀ`` ... explicitly implemented below.

Measurement: known direction ``v_nav``; ``h(q) = R(q)ᵀ v_nav`` with raw
quaternion Jacobian from :func:`qnav.attitude.jacobians.drotate_dq`.

Reference: Solà; EKF exposition in the attitude survey.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import jacobians as jac
from qnav.attitude import quaternion as quat
from qnav.filters.base import AttitudeFilter

__all__ = ["QuaternionEkf"]

_CONJ = np.diag([1.0, -1.0, -1.0, -1.0])
_XI = np.zeros((4, 3))
_XI[1:, :] = np.eye(3)


class QuaternionEkf(AttitudeFilter):
    """Total-state quaternion EKF with explicit renormalization policy."""

    def __init__(
        self, gyro_noise_density: float, P0: np.ndarray | None = None,
        q0=None, nav_frame: str = "NED",
    ) -> None:
        super().__init__(q0=q0, nav_frame=nav_frame)
        self.sigma_g = float(gyro_noise_density)
        self.P = np.asarray(P0, dtype=float).copy() if P0 is not None else 0.01 * np.eye(4)
        if self.P.shape != (4, 4):
            raise ValueError("P0 must be 4×4")

    def predict(self, omega_meas: np.ndarray, dt: float) -> np.ndarray:
        phi = np.asarray(omega_meas, dtype=float) * dt
        dq = quat.exp(phi)
        F = quat.right_matrix(dq)
        # noise injection: q⊗Exp(φ+δ) ≈ q_{k+1} + ½ [q_{k+1}]_L Ξ δ  (small δ)
        G = 0.5 * quat.left_matrix(quat.mul(self.q, dq)) @ _XI
        Qd = G @ ((self.sigma_g**2 * dt) * np.eye(3)) @ G.T
        self.q = quat.mul(self.q, dq)
        self.P = F @ self.P @ F.T + Qd
        self._renormalize()
        return self.q

    def update_direction(
        self, v_nav: np.ndarray, v_body_meas: np.ndarray, sigma: float
    ) -> np.ndarray:
        """Fuse a unit-direction measurement; returns the innovation."""
        vn = np.asarray(v_nav, dtype=float)
        vn = vn / np.linalg.norm(vn)
        vb = np.asarray(v_body_meas, dtype=float)
        vb = vb / np.linalg.norm(vb)
        qc = quat.conjugate(self.q)
        v_hat = quat.rotate_vector(qc, vn)
        H = jac.drotate_dq(qc, vn) @ _CONJ  # 3×4
        R = (sigma**2) * np.eye(3)
        innov = vb - v_hat
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3))
        self.q = self.q + K @ innov
        IKH = np.eye(4) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self._renormalize()
        return innov

    def _renormalize(self) -> None:
        """Renormalize q and project P onto the tangent of the unit sphere.

        Projection ``J = (I − q qᵀ)/‖q‖`` removes the unobservable radial
        component that the linearized update injects.
        """
        n = np.linalg.norm(self.q)
        self.q = self.q / n
        J = (np.eye(4) - np.outer(self.q, self.q)) / n
        self.P = J @ self.P @ J.T
