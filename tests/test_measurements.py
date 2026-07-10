"""Measurement models (Phase 6): predictions, FD-verified Jacobians, fusion.

The Jacobian test perturbs each error-state component, injects it into the
nominal state exactly as the ESKF does, and compares the change in the
predicted measurement with ``H @ delta`` — so H is checked against the same
error definition the filter uses.
"""

import numpy as np
import pytest

from qnav.attitude import quaternion as quat
from qnav.frames.earth import meridian_radius, transverse_radius
from qnav.nav import NavEskf, NavState
from qnav.nav.measurements import (
    BaroAltitude,
    DualAntennaHeading,
    ExternalAttitude,
    ExternalPose,
    ExternalVelocityBody,
    GnssPosition,
    GnssVelocity,
    MagYaw,
    NonholonomicConstraint,
    RangefinderHeight,
    UwbRange,
    WheelSpeed,
    ZaruGyroBias,
    ZuptVelocity,
)

LAT, LON, H0 = np.deg2rad(45.0), np.deg2rad(7.0), 200.0


def _state():
    q = quat.exp(np.array([0.05, -0.1, 0.7]))
    return NavState(q=q, v=np.array([3.0, -1.0, 0.5]), p=[LAT, LON, H0],
                    bg=np.array([0.001, -0.002, 0.0005]),
                    ba=np.array([0.05, 0.02, -0.03]))


def _inject(state: NavState, dx: np.ndarray) -> NavState:
    """Mirror NavEskf._inject for a NED state."""
    lat, lon, h = (float(x) for x in state.p)
    M, N = float(meridian_radius(lat)), float(transverse_radius(lat))
    return state.evolve(
        q=quat.normalize(quat.mul(state.q, quat.exp(dx[0:3]))),
        v=state.v + dx[3:6],
        p=np.array([lat + dx[6] / (M + h), lon + dx[7] / ((N + h) * np.cos(lat)),
                    h - dx[8]]),
        bg=state.bg + dx[9:12], ba=state.ba + dx[12:15],
    )


def _fd_check(model, value, aux=None, eps=1e-6, tol=1e-4):
    """H @ dx must match the finite-difference innovation change."""
    aux = aux or {}
    s0 = _state()
    innov0, H = model.residual(s0, value, **aux)
    innov0 = np.atleast_1d(innov0)
    for j in range(15):
        dx = np.zeros(15)
        dx[j] = eps
        innov_j = np.atleast_1d(model.residual(_inject(s0, dx), value, **aux)[0])
        # innovation = z - h(x): d(innov)/dx = -dh/dx = -H_col... H is dh/dx
        fd = (innov0 - innov_j) / eps          # = ∂h/∂δx_j
        np.testing.assert_allclose(fd, H[:, j], atol=tol,
                                   err_msg=f"H column {j} mismatch for {type(model).__name__}")


class TestJacobians:
    def test_gnss_position_with_lever_arm(self):
        _fd_check(GnssPosition(lever_arm=np.array([1.0, 0.5, -0.2])),
                  np.array([LAT, LON, H0]), tol=1e-3)  # geodetic curvature

    def test_gnss_velocity_with_lever_arm(self):
        _fd_check(GnssVelocity(lever_arm=np.array([1.0, 0.5, -0.2])),
                  np.array([3.0, -1.0, 0.5]),
                  aux={"omega_ib_b": np.array([0.1, -0.2, 0.3])})

    def test_baro(self):
        _fd_check(BaroAltitude(), 201.0)

    def test_rangefinder(self):
        _fd_check(RangefinderHeight(ground_elevation=150.0), 55.0)

    def test_external_attitude(self):
        _fd_check(ExternalAttitude(), quat.exp(np.array([0.06, -0.09, 0.71])),
                  tol=5e-4)

    def test_external_pose(self):
        _fd_check(ExternalPose(),
                  (quat.exp(np.array([0.06, -0.09, 0.71])), np.array([LAT, LON, H0])),
                  tol=5e-4)

    def test_external_velocity_body(self):
        _fd_check(ExternalVelocityBody(), np.array([2.0, 0.1, -0.05]))

    def test_wheel_speed(self):
        _fd_check(WheelSpeed(), 2.5)

    def test_nonholonomic(self):
        _fd_check(NonholonomicConstraint(), None)

    def test_zupt(self):
        _fd_check(ZuptVelocity(), None)

    def test_zaru(self):
        _fd_check(ZaruGyroBias(), np.array([0.002, -0.001, 0.001]))

    def test_uwb_range(self):
        anchor = np.array([LAT + 100 / 6.4e6, LON + 50 / 4.5e6, H0 + 10.0])
        _fd_check(UwbRange(anchor=anchor), 110.0, tol=5e-3)

    def test_mag_yaw(self):
        _fd_check(MagYaw(), 0.75, tol=5e-3)

    def test_dual_antenna(self):
        _fd_check(DualAntennaHeading(), 0.75, tol=5e-3)


class TestPredictions:
    def test_baro_innovation_sign(self):
        innov, _ = BaroAltitude().residual(_state(), H0 + 3.0)
        assert innov[0] == pytest.approx(3.0)

    def test_zupt_innovation_is_minus_velocity(self):
        s = _state()
        innov, _ = ZuptVelocity().residual(s, None)
        np.testing.assert_allclose(innov, -s.v)

    def test_zaru_innovation(self):
        s = _state()
        innov, _ = ZaruGyroBias().residual(s, s.bg)
        np.testing.assert_allclose(innov, 0.0, atol=1e-15)

    def test_mag_yaw_wraps(self):
        innov, _ = MagYaw().residual(_state(), np.pi - 0.01)
        assert -np.pi <= innov[0] <= np.pi

    def test_uwb_rejects_close_anchor(self):
        s = _state()
        with pytest.raises(ValueError, match="min_range"):
            UwbRange(anchor=np.array([LAT, LON, H0])).residual(s, 0.0)

    def test_rangefinder_rejects_high_tilt(self):
        s = _state().evolve(q=quat.exp(np.array([0.0, np.deg2rad(60), 0.0])))
        with pytest.raises(ValueError, match="max_tilt"):
            RangefinderHeight().residual(s, 50.0)

    def test_gnss_velocity_lever_arm_needs_omega(self):
        with pytest.raises(ValueError, match="omega_ib_b"):
            GnssVelocity(lever_arm=np.array([1.0, 0, 0])).residual(
                _state(), np.zeros(3))

    def test_baro_requires_ned(self):
        s = NavState(q=quat.identity(), p=[6.4e6, 0.0, 0.0], frame="ECEF")
        with pytest.raises(ValueError, match="NED"):
            BaroAltitude().residual(s, 100.0)


class TestFusion:
    def _filter(self, **kw):
        s0 = NavState(q=quat.identity(), p=[LAT, LON, H0])
        return NavEskf(s0, gyro_noise_density=0.002, accel_noise_density=0.02,
                       gyro_bias_walk=1e-6, accel_bias_walk=1e-5, **kw)

    def test_update_measurement_applies_and_reports(self):
        f = self._filter()
        innov = f.update_measurement(BaroAltitude(), H0 + 2.0, sigma=0.5,
                                     timestamp=1.0)
        assert innov[0] == pytest.approx(2.0)
        r = f.last_update
        assert r.accepted and r.sensor_id == "BaroAltitude"
        assert float(f.state.p[2]) > H0    # altitude corrected upward

    def test_zaru_estimates_gyro_bias(self):
        f = self._filter()
        bg_true = np.array([0.004, -0.003, 0.002])
        from qnav.frames.earth import normal_gravity
        f_static = np.array([0.0, 0.0, -float(normal_gravity(LAT, H0))])
        for _ in range(200):
            w_meas = bg_true  # standstill, Earth rate ignored (MEMS-level)
            f.predict(w_meas, f_static, 0.01)
            f.update_measurement(ZuptVelocity(), None, sigma=0.01)
            f.update_measurement(ZaruGyroBias(), w_meas, sigma=0.001)
        np.testing.assert_allclose(f.state.bg, bg_true, atol=2e-4)

    def test_mag_yaw_fixes_heading(self):
        f = self._filter()
        f.state = f.state.evolve(q=quat.exp(np.array([0.0, 0.0, np.deg2rad(20)])))
        from qnav.frames.earth import normal_gravity
        f_static = np.array([0.0, 0.0, -float(normal_gravity(LAT, H0))])
        for _ in range(200):
            f.predict(np.zeros(3), f_static, 0.01)
            f.update_measurement(MagYaw(), 0.0, sigma=np.deg2rad(2.0))
        yaw_err = abs(float(quat.log(f.state.q)[2]))
        assert yaw_err < np.deg2rad(1.0)

    def test_gate_applies_to_models(self):
        from qnav.filters import GatePolicy
        f = self._filter(gate=GatePolicy())
        for _ in range(50):
            f.update_measurement(BaroAltitude(), H0, sigma=0.5)
        p_before = f.state.p.copy()
        f.update_measurement(BaroAltitude(), H0 + 500.0, sigma=0.5)  # outlier
        assert f.last_update.rejection_reason == "nis_gate"
        np.testing.assert_array_equal(f.state.p, p_before)

    def test_nonholonomic_bounds_lateral_velocity(self):
        f = self._filter()
        f.state = f.state.evolve(v=np.array([5.0, 0.8, 0.0]))  # spurious lateral
        for _ in range(50):
            f.update_measurement(NonholonomicConstraint(), None, sigma=0.05)
        # the constraint zeroes the *body-frame* lateral velocity; part of the
        # nav-frame residual is legitimately absorbed by a yaw correction
        v_body = quat.rotate_frame(f.state.q, f.state.v)
        assert abs(float(v_body[1])) < 0.05
        assert float(v_body[0]) > 4.5      # forward speed preserved
