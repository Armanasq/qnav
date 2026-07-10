"""Strapdown inertial mechanization: NED and ECEF, single-step kernels.

Both kernels take bias-corrected IMU samples (``omega_ib_b`` [rad/s],
``f_b`` [m/s², specific force]) and advance a :class:`~qnav.nav.state.NavState`
by ``dt``. They are the **only** propagation equations in qnav's navigation
stack — the ESKF and (future) preintegration call these kernels rather than
duplicating the math.

Integration scheme (documented order): attitude uses the exact exponential
of the constant-rate assumption over the step; velocity uses first-order
integration of the specific-force/gravity/Coriolis sum evaluated at the
step start attitude; position uses the trapezoid of old/new velocity.
Errors are O(dt²) per step; validated against closed-form cases in
``tests/test_nav.py``.

NED mechanization (Groves ch. 5 / Titterton ch. 3):

    q_nb <- Exp_q(-(w_ie_n + w_en_n) dt) (x) q_nb (x) Exp_q(w_ib_b dt)
    v̇_n  = R(q_nb) f_b + g_n(lat, h) - (2 w_ie_n + w_en_n) x v_n
    laṫ  = v_N / (M + h),  loṅ = v_E / ((N + h) cos lat),  ḣ = -v_D

ECEF mechanization:

    q_eb <- Exp_q(-w_ie_e dt) (x) q_eb (x) Exp_q(w_ib_b dt)
    v̇_e  = R(q_eb) f_b + g_e(r) - 2 w_ie_e x v_e
    ṙ    = v_e

Gravity is WGS-84 Somigliana normal gravity with free-air correction
(includes the centrifugal term, as required for the resolved-frame velocity
equations above).
"""

from __future__ import annotations

import numpy as np

from qnav._validate import ensure_positive_dt, ensure_vector3
from qnav.attitude import quaternion as quat
from qnav.frames.earth import (
    WGS84_OMEGA,
    dcm_ned_to_ecef,
    earth_rate_ned,
    ecef_to_geodetic,
    gravity_vector,
    meridian_radius,
    transport_rate_ned,
    transverse_radius,
)
from qnav.nav.state import NavState

__all__ = ["gravity_ecef", "propagate_ecef", "propagate_ned", "propagate_state"]


def propagate_ned(state: NavState, omega_ib_b: np.ndarray, f_b: np.ndarray, dt: float) -> NavState:
    """One NED strapdown step with Earth rate, transport rate, and Coriolis."""
    if state.frame != "NED":
        raise ValueError(f"propagate_ned requires a NED state, got {state.frame!r}")
    w = ensure_vector3(omega_ib_b, "omega_ib_b")
    f = ensure_vector3(f_b, "f_b")
    dt = ensure_positive_dt(dt)

    lat, lon, h = (float(x) for x in state.p)
    v = state.v
    w_ie = earth_rate_ned(lat)
    w_en = transport_rate_ned(lat, v, h)

    # attitude: nav-frame rotation removed on the left, body increment on the right
    q_new = quat.normalize(quat.mul(
        quat.exp(-(w_ie + w_en) * dt), quat.mul(state.q, quat.exp(w * dt))
    ))

    # velocity: specific force resolved at the step-start attitude
    a = quat.rotate_vector(state.q, f) + gravity_vector(lat, h) - np.cross(2.0 * w_ie + w_en, v)
    v_new = v + a * dt

    # position: trapezoidal velocity
    v_mid = 0.5 * (v + v_new)
    M = float(meridian_radius(lat))
    N = float(transverse_radius(lat))
    lat_new = lat + v_mid[0] / (M + h) * dt
    lon_new = lon + v_mid[1] / ((N + h) * np.cos(lat)) * dt
    h_new = h - v_mid[2] * dt

    return state.evolve(q=q_new, v=v_new, p=np.array([lat_new, lon_new, h_new]))


def gravity_ecef(r_ecef: np.ndarray) -> np.ndarray:
    """WGS-84 normal gravity resolved in ECEF at position ``r_ecef`` [m]."""
    r = ensure_vector3(r_ecef, "r_ecef")
    lat, lon, h = ecef_to_geodetic(r)
    g_ned = gravity_vector(lat, h)
    return dcm_ned_to_ecef(lat, lon) @ g_ned


def propagate_ecef(state: NavState, omega_ib_b: np.ndarray, f_b: np.ndarray, dt: float) -> NavState:
    """One ECEF strapdown step with Earth rotation and Coriolis."""
    if state.frame != "ECEF":
        raise ValueError(f"propagate_ecef requires an ECEF state, got {state.frame!r}")
    w = ensure_vector3(omega_ib_b, "omega_ib_b")
    f = ensure_vector3(f_b, "f_b")
    dt = ensure_positive_dt(dt)

    w_ie = np.array([0.0, 0.0, WGS84_OMEGA])
    q_new = quat.normalize(quat.mul(
        quat.exp(-w_ie * dt), quat.mul(state.q, quat.exp(w * dt))
    ))

    a = quat.rotate_vector(state.q, f) + gravity_ecef(state.p) - np.cross(2.0 * w_ie, state.v)
    v_new = state.v + a * dt
    p_new = state.p + 0.5 * (state.v + v_new) * dt

    return state.evolve(q=q_new, v=v_new, p=p_new)


def propagate_state(state: NavState, omega_ib_b: np.ndarray, f_b: np.ndarray, dt: float) -> NavState:
    """Frame-dispatching strapdown step (the single propagation kernel)."""
    if state.frame == "NED":
        return propagate_ned(state, omega_ib_b, f_b, dt)
    return propagate_ecef(state, omega_ib_b, f_b, dt)
