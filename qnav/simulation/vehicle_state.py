"""Full 6-DoF vehicle state container and kinematic propagation.

:class:`VehicleState` couples translation (navigation frame) with attitude
(``q_nav_body``) for ground/aerial/marine/spacecraft simulation. Propagation
is strapdown-style: body rates and navigation-frame acceleration in,
position/velocity/attitude out.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from qnav.attitude import kinematics as kin
from qnav.attitude import quaternion as quat

__all__ = ["VehicleState"]


@dataclass(frozen=True)
class VehicleState:
    """Position/velocity in the navigation frame, attitude, body rates."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))   # [m] nav
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))   # [m/s] nav
    q_nav_body: np.ndarray = field(default_factory=quat.identity)
    omega_body: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [rad/s]
    nav_frame: str = "NED"

    def propagate(
        self, accel_nav: np.ndarray, omega_body: np.ndarray, dt: float
    ) -> "VehicleState":
        """Constant-acceleration / constant-rate step (midpoint on rates).

        ``p ← p + v dt + ½ a dt²``, ``v ← v + a dt``,
        ``q ← q ⊗ Exp(ω̄ dt)`` with ω̄ the average of old and new rates.
        """
        a = np.asarray(accel_nav, dtype=float)
        w_new = np.asarray(omega_body, dtype=float)
        p = self.position + self.velocity * dt + 0.5 * a * dt * dt
        v = self.velocity + a * dt
        q = kin.integrate_midpoint(self.q_nav_body, self.omega_body, w_new, dt)
        return replace(self, position=p, velocity=v, q_nav_body=q, omega_body=w_new)

    def velocity_body(self) -> np.ndarray:
        """Velocity expressed in body axes: ``R_BN v_N``."""
        return quat.rotate_frame(self.q_nav_body, self.velocity)
