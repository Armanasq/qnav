"""IMU preintegration (Forster et al., on-manifold, right/local convention).

Accumulates gravity-free, body-frame relative motion between two times
independently of the absolute state:

    ΔR_ij = Π Exp((ω_k − b_g) δt)
    Δv_ij = Σ ΔR_ik (f_k − b_a) δt
    Δp_ij = Σ [Δv_ik δt + ½ ΔR_ik (f_k − b_a) δt²]

together with the first-order bias Jacobians (∂Δ·/∂b_g, ∂Δ·/∂b_a), the
9x9 preintegrated noise covariance over ``[δθ, δv, δp]``, and the total
interval. :meth:`ImuPreintegration.corrected` re-linearizes the deltas at a
new bias estimate without re-integrating.

Relation to the recursive mechanization (flat, non-rotating Earth):

    q_j = q_i (x) ΔR_ij
    v_j = v_i + g Δt + R_i Δv_ij
    p_j = p_i + v_i Δt + ½ g Δt² + R_i Δp_ij

The cross-consistency between this module, the naive per-sample recursion of
the same equations, and the full NED mechanization (Earth terms bounded) is
enforced by ``tests/test_preintegration.py`` — the implementations cannot
silently diverge.

Reference: Forster, Carlone, Dellaert, Scaramuzza, "On-Manifold
Preintegration for Real-Time Visual-Inertial Odometry", T-RO 2017.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from qnav._validate import ensure_nonnegative, ensure_positive_dt, ensure_vector3
from qnav.attitude import dcm as dcm_mod
from qnav.attitude import quaternion as quat
from qnav.attitude import so3

__all__ = ["ImuPreintegration", "PreintegratedImu"]


@dataclass(frozen=True)
class PreintegratedImu:
    """The immutable result of one preintegration interval.

    ``delta_rotation`` is a unit quaternion (body_i <- body_j); velocities
    and positions are in the body_i frame, gravity-free. ``covariance`` is
    9x9 over ``[δθ, δv, δp]``; the bias Jacobians linearize the deltas about
    the reference biases used during integration.
    """

    delta_rotation: np.ndarray
    delta_velocity: np.ndarray
    delta_position: np.ndarray
    J_R_bg: np.ndarray
    J_v_bg: np.ndarray
    J_v_ba: np.ndarray
    J_p_bg: np.ndarray
    J_p_ba: np.ndarray
    covariance: np.ndarray
    dt_total: float
    bg_ref: np.ndarray
    ba_ref: np.ndarray

    def corrected(self, bg: np.ndarray, ba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """First-order bias-corrected ``(ΔR, Δv, Δp)`` at new bias estimates.

        Valid while ``|b − b_ref|`` stays small (the Jacobians are a first-
        order expansion); re-integrate when the bias moves far.
        """
        dbg = ensure_vector3(bg, "bg") - self.bg_ref
        dba = ensure_vector3(ba, "ba") - self.ba_ref
        dR = quat.mul(self.delta_rotation, quat.exp(self.J_R_bg @ dbg))
        dv = self.delta_velocity + self.J_v_bg @ dbg + self.J_v_ba @ dba
        dp = self.delta_position + self.J_p_bg @ dbg + self.J_p_ba @ dba
        return quat.normalize(dR), dv, dp


class ImuPreintegration:
    """Accumulates IMU samples into a :class:`PreintegratedImu`.

    Parameters are the IMU white-noise densities [rad/s/√Hz, m/s²/√Hz] and
    the reference biases at which the deltas are linearized.
    """

    def __init__(
        self,
        gyro_noise_density: float,
        accel_noise_density: float,
        bg_ref: np.ndarray | None = None,
        ba_ref: np.ndarray | None = None,
    ) -> None:
        self.sigma_g = ensure_nonnegative(gyro_noise_density, "gyro_noise_density")
        self.sigma_a = ensure_nonnegative(accel_noise_density, "accel_noise_density")
        self.bg_ref = np.zeros(3) if bg_ref is None else ensure_vector3(bg_ref, "bg_ref").copy()
        self.ba_ref = np.zeros(3) if ba_ref is None else ensure_vector3(ba_ref, "ba_ref").copy()
        self.reset()

    def reset(self) -> None:
        """Start a fresh interval (reference biases retained)."""
        self.dq: np.ndarray = quat.identity()
        self.dv = np.zeros(3)
        self.dp = np.zeros(3)
        self.J_R_bg: np.ndarray = np.zeros((3, 3))
        self.J_v_bg: np.ndarray = np.zeros((3, 3))
        self.J_v_ba: np.ndarray = np.zeros((3, 3))
        self.J_p_bg: np.ndarray = np.zeros((3, 3))
        self.J_p_ba: np.ndarray = np.zeros((3, 3))
        self.P: np.ndarray = np.zeros((9, 9))
        self.dt_total = 0.0

    def integrate(self, omega_ib_b: np.ndarray, f_b: np.ndarray, dt: float) -> None:
        """Fold one IMU sample into the running preintegration."""
        w = ensure_vector3(omega_ib_b, "omega_ib_b") - self.bg_ref
        a = ensure_vector3(f_b, "f_b") - self.ba_ref
        dt = ensure_positive_dt(dt)

        dR_k = dcm_mod.from_quaternion(self.dq)     # ΔR_ik
        E = so3.exp(w * dt)                          # Exp(w dt)
        Jr = so3.right_jacobian(w * dt)
        A_hat = so3.hat(a)

        # noise/state transition over [δθ, δv, δp] (Forster eq. 59-63 form)
        A = np.eye(9)
        A[0:3, 0:3] = E.T
        A[3:6, 0:3] = -dR_k @ A_hat * dt
        A[6:9, 0:3] = -0.5 * dR_k @ A_hat * dt**2
        A[6:9, 3:6] = np.eye(3) * dt
        B = np.zeros((9, 6))
        B[0:3, 0:3] = Jr * dt
        B[3:6, 3:6] = dR_k * dt
        B[6:9, 3:6] = 0.5 * dR_k * dt**2
        # per-sample noise variance of the *rate* inputs: sigma^2 / dt, so
        # B (which carries one dt) yields the standard sigma^2 * dt increment
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = (self.sigma_g**2 / dt) * np.eye(3)
        Q[3:6, 3:6] = (self.sigma_a**2 / dt) * np.eye(3)
        self.P = A @ self.P @ A.T + B @ Q @ B.T

        # bias Jacobians (position first — they use the pre-update v/R terms)
        self.J_p_ba = self.J_p_ba + self.J_v_ba * dt - 0.5 * dR_k * dt**2
        self.J_p_bg = self.J_p_bg + self.J_v_bg * dt - 0.5 * dR_k @ A_hat @ self.J_R_bg * dt**2
        self.J_v_ba = self.J_v_ba - dR_k * dt
        self.J_v_bg = self.J_v_bg - dR_k @ A_hat @ self.J_R_bg * dt
        self.J_R_bg = E.T @ self.J_R_bg - Jr * dt

        # deltas (position first — it uses the pre-update dv)
        self.dp = self.dp + self.dv * dt + 0.5 * dR_k @ a * dt**2
        self.dv = self.dv + dR_k @ a * dt
        self.dq = quat.normalize(quat.mul(self.dq, quat.exp(w * dt)))
        self.dt_total += dt

    def result(self) -> PreintegratedImu:
        """Snapshot the current interval as an immutable summary."""
        return PreintegratedImu(
            delta_rotation=self.dq.copy(),
            delta_velocity=self.dv.copy(),
            delta_position=self.dp.copy(),
            J_R_bg=self.J_R_bg.copy(),
            J_v_bg=self.J_v_bg.copy(),
            J_v_ba=self.J_v_ba.copy(),
            J_p_bg=self.J_p_bg.copy(),
            J_p_ba=self.J_p_ba.copy(),
            covariance=self.P.copy(),
            dt_total=self.dt_total,
            bg_ref=self.bg_ref.copy(),
            ba_ref=self.ba_ref.copy(),
        )
