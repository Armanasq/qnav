"""Modular measurement models for the 15-state navigation ESKF.

Each model is an immutable object with one method::

    residual(state, value, **aux) -> (innovation, H)

- ``innovation``: measured minus predicted, in the model's documented units
- ``H``: Jacobian of the measurement w.r.t. the error state
  ``[δθ, δv, δp, δbg, δba]`` (right/local attitude error; δp in meters in
  the navigation frame — NED north/east/down or ECEF)

Models never touch the estimator: :meth:`qnav.nav.eskf.NavEskf.update_measurement`
applies any model through the shared gated Joseph-form kernel, so gating,
robust losses, quarantine, and `UpdateResult` reporting are uniform. There
are no sensor-specific branches inside the estimator.

Every model documents: frame contract, units, observability conditions, and
failure/degradation modes. All Jacobians are verified against finite
differences in ``tests/test_measurements.py``.

Common observability facts (static vehicle, no heading aiding):

- position/velocity fixes observe δp, δv, tilt (via gravity coupling) and
  the tilt-axis gyro biases; **yaw and yaw-axis gyro bias stay unobservable**
- magnetic yaw / dual-antenna heading restores yaw observability
- ZUPT/ZARU bound velocity and gyro-bias drift during standstill only
- a single UWB range observes one direction of δp per anchor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from qnav._validate import ensure_vector3
from qnav.attitude import dcm as dcm_mod
from qnav.attitude import quaternion as quat
from qnav.attitude import so3
from qnav.frames.earth import geodetic_to_ecef, meridian_radius, transverse_radius
from qnav.nav.state import NavState

__all__ = [
    "BaroAltitude",
    "DualAntennaHeading",
    "ExternalAttitude",
    "ExternalPose",
    "ExternalVelocityBody",
    "GnssPosition",
    "GnssVelocity",
    "MagYaw",
    "NonholonomicConstraint",
    "RangefinderHeight",
    "UwbRange",
    "WheelSpeed",
    "ZaruGyroBias",
    "ZuptVelocity",
]

_Residual = Tuple[np.ndarray, np.ndarray]


def _position_innov_m(state: NavState, p_meas: np.ndarray) -> np.ndarray:
    """Measured-minus-predicted position in meters in the nav frame."""
    if state.frame == "NED":
        lat, lon, h = (float(x) for x in state.p)
        M = float(meridian_radius(lat))
        N = float(transverse_radius(lat))
        return np.array([
            (p_meas[0] - lat) * (M + h),
            (p_meas[1] - lon) * (N + h) * np.cos(lat),
            -(p_meas[2] - h),
        ])
    return np.asarray(p_meas, dtype=float) - state.p


def _yaw_of(q: np.ndarray) -> float:
    """ZYX yaw of ``q_nav_body`` [rad]."""
    R = dcm_mod.from_quaternion(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class GnssPosition:
    """GNSS antenna position fix.

    Frame/units: NED states take ``[lat, lon, h]`` [rad, rad, m]; ECEF states
    take ``r_ecef`` [m]. ``lever_arm`` is the antenna position in the body
    frame [m]; the model predicts the *antenna* position ``p + R l``.

    Observability: δp fully; δθ only through the lever arm (weak unless the
    lever arm is long). Failure modes: multipath (use the estimator gate),
    datum mismatch (constant bias — not modeled here).
    """

    lever_arm: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def residual(self, state: NavState, value: np.ndarray, **aux: object) -> _Residual:
        lever = ensure_vector3(self.lever_arm, "lever_arm")
        R = dcm_mod.from_quaternion(state.q)
        innov = _position_innov_m(state, np.asarray(value, dtype=float)) - R @ lever
        H = np.zeros((3, 15))
        H[:, 0:3] = -R @ so3.hat(lever)
        H[:, 6:9] = np.eye(3)
        return innov, H


@dataclass(frozen=True)
class GnssVelocity:
    """GNSS antenna velocity in the navigation frame [m/s].

    Requires the current bias-corrected body rate ``omega_ib_b`` (pass as
    ``aux``) when a lever arm is set: ``v_ant = v + R (ω × l)``.

    Observability: δv fully; δθ weakly through the lever-arm term.
    """

    lever_arm: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def residual(self, state: NavState, value: np.ndarray, *,
                 omega_ib_b: np.ndarray | None = None, **aux: object) -> _Residual:
        v_meas = ensure_vector3(value, "value")
        lever = ensure_vector3(self.lever_arm, "lever_arm")
        R = dcm_mod.from_quaternion(state.q)
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        pred = state.v.copy()
        if np.any(lever != 0.0):
            if omega_ib_b is None:
                raise ValueError("GnssVelocity with a lever arm needs omega_ib_b")
            wl = np.cross(ensure_vector3(omega_ib_b, "omega_ib_b"), lever)
            pred = pred + R @ wl
            H[:, 0:3] = -R @ so3.hat(wl)
        return v_meas - pred, H


@dataclass(frozen=True)
class BaroAltitude:
    """Barometric altitude above the ellipsoid [m] (NED states only).

    ``bias`` is a known static offset (e.g. QNH calibration) subtracted from
    the measurement. Observability: vertical channel only — baro is the
    standard fix for the unstable INS vertical channel. Failure modes:
    weather-driven drift (slow bias — gate wide or estimate externally).
    """

    bias: float = 0.0

    def residual(self, state: NavState, value: float, **aux: object) -> _Residual:
        if state.frame != "NED":
            raise ValueError("BaroAltitude requires a NED state")
        innov = np.array([float(value) - self.bias - float(state.p[2])])
        H = np.zeros((1, 15))
        H[0, 8] = -1.0  # δp[2] is *down*; altitude is up
        return innov, H


@dataclass(frozen=True)
class RangefinderHeight:
    """Height above ground [m] from a downward rangefinder (NED states).

    ``ground_elevation`` is the terrain height above the ellipsoid at the
    current location [m]. Contract: valid only for near-level attitude; the
    model divides by the cosine of the tilt and raises beyond ``max_tilt``.
    """

    ground_elevation: float = 0.0
    max_tilt: float = np.deg2rad(30.0)

    def residual(self, state: NavState, value: float, **aux: object) -> _Residual:
        if state.frame != "NED":
            raise ValueError("RangefinderHeight requires a NED state")
        R = dcm_mod.from_quaternion(state.q)
        cos_tilt = float(R[2, 2])
        if cos_tilt < np.cos(self.max_tilt):
            raise ValueError("tilt exceeds RangefinderHeight.max_tilt; reject upstream")
        h_agl_meas = float(value) * cos_tilt
        pred = float(state.p[2]) - self.ground_elevation
        innov = np.array([h_agl_meas - pred])
        H = np.zeros((1, 15))
        # the tilt compensation R33 = e3^T R e3 depends on the attitude:
        # d(R33)/d(dtheta) = -e3^T R [e3]x, entering the *innovation* with
        # + sign, hence the predicted-measurement Jacobian gets the minus.
        e3 = np.array([0.0, 0.0, 1.0])
        H[0, 0:3] = float(value) * (R[2, :] @ so3.hat(e3))
        H[0, 8] = -1.0
        return innov, H


@dataclass(frozen=True)
class ExternalAttitude:
    """Full attitude from an external source (motion capture, other filter).

    ``value``: ``q_nav_body`` (scalar-first, unit). Innovation is the
    right/local rotation-vector error ``Log(q̂* ⊗ q_meas)`` [rad]; H = I on
    δθ (small-angle). Observability: δθ fully.
    """

    def residual(self, state: NavState, value: np.ndarray, **aux: object) -> _Residual:
        q_meas = np.asarray(value, dtype=float)
        innov = quat.log(quat.mul(quat.conjugate(state.q), q_meas))
        H = np.zeros((3, 15))
        # innov(dtheta) = Log(Exp(-dtheta) Exp(innov0)) ~= innov0 - Jl(innov0)^-1 dtheta
        # (BCH, first order), so the measurement Jacobian is Jl(innov)^-1.
        H[:, 0:3] = np.linalg.inv(so3.left_jacobian(innov))
        return innov, H


@dataclass(frozen=True)
class ExternalPose:
    """Attitude + position from an external source.

    ``value``: tuple ``(q_nav_body, position)`` with position in the state's
    own convention. Innovation is ``[δθ (rad), δp (m)]`` (6,).
    """

    def residual(self, state: NavState, value: object, **aux: object) -> _Residual:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("ExternalPose value must be (q_nav_body, position)")
        q_meas = np.asarray(value[0], dtype=float)
        p_meas = np.asarray(value[1], dtype=float)
        att_innov, att_H = ExternalAttitude().residual(state, q_meas)
        pos_innov = _position_innov_m(state, p_meas)
        innov = np.concatenate([att_innov, pos_innov])
        H = np.zeros((6, 15))
        H[0:3] = att_H
        H[3:6, 6:9] = np.eye(3)
        return innov, H


@dataclass(frozen=True)
class ExternalVelocityBody:
    """Velocity measured in the body frame [m/s] (e.g. visual/DVL odometry).

    Predicted: ``v_b = Rᵀ v_nav``. Observability: δv (rotated) and δθ via
    ``[v_b]×`` — attitude becomes observable only while moving.
    """

    def residual(self, state: NavState, value: np.ndarray, **aux: object) -> _Residual:
        v_meas = ensure_vector3(value, "value")
        R = dcm_mod.from_quaternion(state.q)
        v_b = R.T @ state.v
        H = np.zeros((3, 15))
        H[:, 0:3] = so3.hat(v_b)
        H[:, 3:6] = R.T
        return v_meas - v_b, H


@dataclass(frozen=True)
class WheelSpeed:
    """Scalar forward (body-x) speed [m/s] from wheel odometry.

    Contract: body x is the rolling direction; slip appears as outliers
    (gate) or a scale error (calibrate upstream). Observability: the
    body-x row of :class:`ExternalVelocityBody`.
    """

    def residual(self, state: NavState, value: float, **aux: object) -> _Residual:
        innov3, H3 = ExternalVelocityBody().residual(state, np.zeros(3))
        pred = -innov3  # residual of 0 minus prediction = -prediction
        innov = np.array([float(value) - float(pred[0])])
        return innov, H3[0:1]


@dataclass(frozen=True)
class NonholonomicConstraint:
    """Zero lateral and vertical body velocity (wheeled-vehicle constraint).

    Pseudo-measurement ``v_b[1:3] = 0``. Valid only without side slip or
    jumps; gate on NIS to reject during violations. Observability: bounds
    the cross-track velocity drift; with motion, also tilt/yaw errors.
    """

    def residual(self, state: NavState, value: object = None, **aux: object) -> _Residual:
        innov3, H3 = ExternalVelocityBody().residual(state, np.zeros(3))
        return innov3[1:3], H3[1:3]


@dataclass(frozen=True)
class ZuptVelocity:
    """Zero-velocity update: nav-frame velocity is zero at standstill.

    Apply only when a stance/standstill detector fires (see
    ``qnav.calibration.gyro_bias.detect_static_intervals``); applying while
    moving corrupts the state — gate accordingly.
    """

    def residual(self, state: NavState, value: object = None, **aux: object) -> _Residual:
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        return -state.v.copy(), H


@dataclass(frozen=True)
class ZaruGyroBias:
    """Zero-angular-rate update: at standstill (Earth rate compensated or
    negligible), the measured rate equals the gyro bias.

    ``value``: raw gyro sample [rad/s]. Innovation: ``omega_meas − b̂g``.
    Observability: δbg directly — the fastest gyro-bias estimator available
    during standstill.
    """

    def residual(self, state: NavState, value: np.ndarray, **aux: object) -> _Residual:
        w = ensure_vector3(value, "value")
        H = np.zeros((3, 15))
        H[:, 9:12] = np.eye(3)
        return w - state.bg, H


@dataclass(frozen=True)
class UwbRange:
    """Range [m] to a fixed UWB anchor.

    ``anchor``: anchor position in the state's own convention (NED states:
    ``[lat, lon, h]``; ECEF: meters). Observability: one direction of δp per
    anchor — full position needs ≥ 3 well-spread anchors (or motion).
    Failure modes: NLOS bias (positive outliers — gate one-sidedly upstream
    or with a tight NIS gate). Raises within ``min_range`` of the anchor
    (direction undefined).
    """

    anchor: np.ndarray = field(default_factory=lambda: np.zeros(3))
    min_range: float = 0.1

    def residual(self, state: NavState, value: float, **aux: object) -> _Residual:
        if state.frame == "NED":
            lat, lon, h = (float(x) for x in state.p)
            d_ecef = geodetic_to_ecef(*(float(x) for x in self.anchor)) - geodetic_to_ecef(lat, lon, h)
            from qnav.frames.earth import dcm_ecef_to_ned
            d = dcm_ecef_to_ned(lat, lon) @ d_ecef      # anchor - vehicle, NED meters
        else:
            d = np.asarray(self.anchor, dtype=float) - state.p
        rho = float(np.linalg.norm(d))
        if rho < self.min_range:
            raise ValueError(f"anchor within min_range ({rho:.3f} m): direction undefined")
        u = d / rho
        H = np.zeros((1, 15))
        H[0, 6:9] = -u  # d = anchor - p, so d(rho)/d(dp) = -u (FD-verified)
        return np.array([float(value) - rho]), H


@dataclass(frozen=True)
class MagYaw:
    """Heading (yaw) [rad] derived from a tilt-compensated magnetometer.

    ``value``: measured yaw, ZYX convention, radians, already declination-
    corrected (see ``qnav.heading``). Innovation is wrapped to (−π, π].
    Observability: yaw (and, with motion, the yaw-axis gyro bias) — the
    canonical fix for the unobservable yaw channel of position-aided INS.
    Failure modes: magnetic disturbance (gate; see ``qnav.heading.disturbance``).
    """

    def residual(self, state: NavState, value: float, **aux: object) -> _Residual:
        innov = np.array([_wrap_pi(float(value) - _yaw_of(state.q))])
        R = dcm_mod.from_quaternion(state.q)
        H = np.zeros((1, 15))
        # psi = atan2(R10, R00); dR/d(dtheta_j) = R [e_j]x (right/local error)
        denom = R[0, 0] ** 2 + R[1, 0] ** 2
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1.0
            dR = R @ so3.hat(e)
            H[0, j] = (R[0, 0] * dR[1, 0] - R[1, 0] * dR[0, 0]) / denom
        return innov, H


@dataclass(frozen=True)
class DualAntennaHeading:
    """Heading [rad] of a dual-antenna GNSS baseline aligned with body x.

    Same measurement equation as :class:`MagYaw` but immune to magnetic
    disturbance; accuracy scales with baseline length (contract: the
    baseline is body-x aligned; a mounting offset must be calibrated out).
    """

    def residual(self, state: NavState, value: float, **aux: object) -> _Residual:
        return MagYaw().residual(state, value)
