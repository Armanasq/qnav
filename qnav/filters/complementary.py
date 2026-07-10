"""Linear (interpolation-based) complementary attitude filter.

Estimate ``q_nav_body``. Per step:

1. **Predict**: exact exponential gyro propagation
   ``q ← q ⊗ Exp(ω dt)``.
2. **Update**: form an absolute attitude fix ``q_am`` from accelerometer
   (+ optional magnetometer via tilt-compensated heading), then blend on the
   geodesic: ``q ← slerp(q, q_am, α)`` with gain ``α ∈ [0, 1]``.

Assumptions (declared, not hidden): accelerometer measures −gravity at low
dynamics (FRD body, NED nav or FLU/ENU); magnetometer is calibrated and
undisturbed (gate with ``qnav.heading.disturbance`` upstream). Without a
magnetometer the yaw of the fix is taken from the current estimate (gyro
yaw), i.e. the accelerometer only levels the filter.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import euler as _euler
from qnav.attitude import interpolation as interp
from qnav.attitude import kinematics as kin
from qnav.filters.base import AttitudeFilter
from qnav.heading.compass import magnetic_heading
from qnav.heading.tilt_compensation import roll_pitch_from_accel

__all__ = ["ComplementaryFilter"]


class ComplementaryFilter(AttitudeFilter):
    """Geodesic-blend complementary filter (NED/FRD or ENU/FLU matched pair).

    Parameters
    ----------
    gain:
        Blend factor α per update (0 = gyro only, 1 = trust fix fully).
    q0, nav_frame:
        Initial attitude and navigation frame (see :class:`AttitudeFilter`).
    """

    def __init__(self, gain: float = 0.02, q0=None, nav_frame: str = "NED") -> None:
        super().__init__(q0=q0, nav_frame=nav_frame)
        if not 0.0 <= gain <= 1.0:
            raise ValueError("gain must be in [0, 1]")
        self.gain = float(gain)

    def _predict(self, omega_body: np.ndarray, dt: float) -> np.ndarray:
        self.q = kin.integrate_exponential(self.q, omega_body, dt)
        return self.q

    def update(
        self, f_body: np.ndarray, m_body: np.ndarray | None = None
    ) -> np.ndarray:
        """Blend toward the accel(+mag) fix; ``f_body`` is specific force.

        Body components must be FRD when ``nav_frame='NED'`` and FLU when
        ``'ENU'`` (matched pairs only — the tilt equations coincide there
        after the conversion done by ``roll_pitch_from_accel``).
        """
        roll, pitch = roll_pitch_from_accel(f_body, frame=self.nav_frame)
        if m_body is not None:
            m = np.asarray(m_body, dtype=float)
            if self.nav_frame == "ENU":
                from qnav.frames.conventions import flu_to_frd
                m = flu_to_frd(m)
            yaw = magnetic_heading(m, roll, pitch)
            yaw = np.arctan2(np.sin(yaw), np.cos(yaw))  # to (−π, π]
        else:
            yaw = _euler.from_quaternion(self.q, "ZYX")[..., 0]
        if self.nav_frame == "NED":
            q_fix = _euler.to_quaternion(np.stack([yaw, pitch, roll], axis=-1), "ZYX")
        else:
            # ENU/FLU matched pair: angles computed in the FRD/NED sense, then
            # re-expressed — q_ENU_FLU = q_ENU_NED ⊗ q_NED_FRD ⊗ q_FRD_FLU
            from qnav.frames.conventions import attitude_ned_frd_to_enu_flu
            q_ned_frd = _euler.to_quaternion(np.stack([yaw, pitch, roll], axis=-1), "ZYX")
            q_fix = attitude_ned_frd_to_enu_flu(q_ned_frd)
        self.q = interp.slerp(self.q, q_fix, self.gain)
        return self.q
