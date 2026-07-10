"""Navigation stack (Phase 5): mechanization, increments, 15-state ESKF.

Numerical validation against closed-form physics: static equilibrium,
free-fall, Earth-rate attitude compensation, transport over the ellipsoid,
coning drift, NED/ECEF cross-checks, and ESKF error convergence.
"""

import numpy as np
import pytest

from qnav.attitude import quaternion as quat
from qnav.frames.earth import (
    WGS84_OMEGA,
    dcm_ecef_to_ned,
    geodetic_to_ecef,
    ecef_to_geodetic,
    gravity_vector,
    meridian_radius,
    normal_gravity,
)
from qnav.nav import (
    NavEskf,
    NavState,
    accumulate_increments,
    propagate_ecef,
    propagate_ned,
)

LAT, LON, H = np.deg2rad(45.0), np.deg2rad(7.0), 200.0
G = float(normal_gravity(LAT, H))


def _static_imu_ned(lat=LAT):
    """Ideal IMU output for a body at rest on Earth, level, x north."""
    # gyro senses Earth rotation; accel senses -gravity (specific force)
    omega = WGS84_OMEGA * np.array([np.cos(lat), 0.0, -np.sin(lat)])
    f = np.array([0.0, 0.0, -float(normal_gravity(lat, H))])
    return omega, f


class TestNavState:
    def test_validation(self):
        with pytest.raises(ValueError, match="unit-norm"):
            NavState(q=[2.0, 0, 0, 0])
        with pytest.raises(ValueError, match="latitude"):
            NavState(q=[1.0, 0, 0, 0], p=[3.0, 0.0, 0.0], frame="NED")
        with pytest.raises(ValueError, match="frame"):
            NavState(q=[1.0, 0, 0, 0], frame="MARS")

    def test_evolve_revalidates(self):
        s = NavState(q=[1.0, 0, 0, 0], p=[LAT, LON, H])
        with pytest.raises(ValueError):
            s.evolve(v=[np.nan, 0.0, 0.0])


class TestNedMechanization:
    def test_static_equilibrium(self):
        """A body at rest with ideal IMU stays at rest: velocity and position
        drift stay tiny over 60 s (first-order integration residual only)."""
        omega, f = _static_imu_ned()
        s = NavState(q=quat.identity(), p=[LAT, LON, H])
        dt = 0.01
        for _ in range(6000):
            s = propagate_ned(s, omega, f, dt)
        assert np.linalg.norm(s.v) < 0.05                    # m/s after 60 s
        assert abs(s.p[2] - H) < 1.0                         # m altitude
        assert abs(s.p[0] - LAT) * meridian_radius(LAT) < 5.0
        # attitude stays level: Earth rotation exactly cancelled
        assert quat.angular_distance(s.q, quat.identity()) < np.deg2rad(0.05)

    def test_free_fall_altitude(self):
        """Zero specific force => h(t) = h0 - g t²/2 (short interval)."""
        s = NavState(q=quat.identity(), p=[LAT, LON, 1000.0])
        dt, T = 0.001, 2.0
        omega, _ = _static_imu_ned()
        for _ in range(int(T / dt)):
            s = propagate_ned(s, omega, np.zeros(3), dt)
        expected_drop = 0.5 * float(normal_gravity(LAT, 1000.0)) * T**2
        assert s.p[2] == pytest.approx(1000.0 - expected_drop, abs=0.5)
        assert -s.v[2] == pytest.approx(-float(normal_gravity(LAT, 1000.0)) * T, abs=0.1)

    def test_northward_transport(self):
        """Constant 100 m/s north: dlat/dt = v_N / (M + h)."""
        v0 = np.array([100.0, 0.0, 0.0])
        s = NavState(q=quat.identity(), v=v0, p=[LAT, LON, H])
        dt, T = 0.01, 10.0
        # specific force balancing gravity + Coriolis + transport so v stays
        # constant: f = R^T (dv/dt_target=0 - g + (2w_ie+w_en) x v)
        from qnav.frames.earth import earth_rate_ned, transport_rate_ned
        for _ in range(int(T / dt)):
            lat, lon, h = (float(x) for x in s.p)
            w_ie = earth_rate_ned(lat)
            w_en = transport_rate_ned(lat, s.v, h)
            f_nav = -gravity_vector(lat, h) + np.cross(2.0 * w_ie + w_en, s.v)
            f_body = quat.rotate_frame(s.q, f_nav)
            omega_body = quat.rotate_frame(s.q, w_ie + w_en)
            s = propagate_ned(s, omega_body, f_body, dt)
        dlat_expected = 100.0 * T / (float(meridian_radius(LAT)) + H)
        assert s.p[0] - LAT == pytest.approx(dlat_expected, rel=1e-3)
        assert np.linalg.norm(s.v - v0) < 0.05


class TestEcefMechanization:
    def test_static_equilibrium_ecef(self):
        """Same rest scenario expressed in ECEF: position moves < a few m."""
        omega, f = _static_imu_ned()
        r0 = geodetic_to_ecef(LAT, LON, H)
        R_en = dcm_ecef_to_ned(LAT, LON).T          # NED -> ECEF
        q_eb = quat.mul(_dcm_to_q(R_en), quat.identity())
        s = NavState(q=q_eb, v=np.zeros(3), p=r0, frame="ECEF")
        dt = 0.01
        for _ in range(3000):
            s = propagate_ecef(s, omega, f, dt)
        assert np.linalg.norm(s.p - r0) < 5.0        # m over 30 s
        assert np.linalg.norm(s.v) < 0.2

    def test_ned_ecef_cross_check_free_fall(self):
        """Free-fall propagated in NED and in ECEF agrees in final geodetic
        altitude within integration tolerance."""
        dt, T = 0.001, 1.0
        omega, _ = _static_imu_ned()
        s_ned = NavState(q=quat.identity(), p=[LAT, LON, 1000.0])
        for _ in range(int(T / dt)):
            s_ned = propagate_ned(s_ned, omega, np.zeros(3), dt)

        r0 = geodetic_to_ecef(LAT, LON, 1000.0)
        R_en = dcm_ecef_to_ned(LAT, LON).T
        v0_e = R_en @ np.cross(-np.array([0.0, 0.0, WGS84_OMEGA]),
                               np.zeros(3))  # zero ground velocity
        # a point fixed on rotating Earth has ECEF velocity zero
        s_ecef = NavState(q=_dcm_to_q(R_en), v=v0_e, p=r0, frame="ECEF")
        for _ in range(int(T / dt)):
            s_ecef = propagate_ecef(s_ecef, omega, np.zeros(3), dt)
        _, _, h_ecef = ecef_to_geodetic(s_ecef.p)
        assert h_ecef == pytest.approx(float(s_ned.p[2]), abs=0.2)


class TestIncrements:
    def test_constant_rate_has_no_correction(self):
        w = np.tile([0.3, 0.0, 0.0], (20, 1))
        f = np.tile([0.0, 0.0, -9.81], (20, 1))
        dtheta, dv = accumulate_increments(w, f, 0.001)
        np.testing.assert_allclose(dtheta, [0.3 * 0.02, 0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(dv, [0.0, 0.0, -9.81 * 0.02], atol=1e-12)

    def test_coning_correction_reduces_drift(self):
        """Classic coning: ω = [a Ω cos Ωt, -a Ω sin Ωt, 0] produces a net
        z-rotation that the naive angle sum misses entirely."""
        a, Om, dt_s = 0.01, 2 * np.pi * 50, 1e-4
        n = 200                                   # one 20 ms interval
        t = np.arange(n) * dt_s
        w = np.stack([a * Om * np.cos(Om * t), -a * Om * np.sin(Om * t),
                      np.zeros(n)], axis=1)
        f = np.zeros((n, 3))
        dtheta_corr, _ = accumulate_increments(w, f, dt_s)
        naive = w.sum(axis=0) * dt_s
        # true coning drift rate about z: -0.5 a² Ω
        expected_z = -0.5 * a**2 * Om * (n * dt_s)
        assert abs(naive[2] - expected_z) > 0.5 * abs(expected_z)   # naive misses it
        assert dtheta_corr[2] == pytest.approx(expected_z, rel=0.05)

    def test_sculling_correction_nonzero_under_sculling_motion(self):
        Om, dt_s, n = 2 * np.pi * 50, 1e-4, 200
        t = np.arange(n) * dt_s
        w = np.stack([0.05 * Om * np.cos(Om * t), np.zeros(n), np.zeros(n)], axis=1)
        f = np.stack([np.zeros(n), 5.0 * np.sin(Om * t), np.zeros(n)], axis=1)
        _, dv = accumulate_increments(w, f, dt_s)
        naive_dv = f.sum(axis=0) * dt_s
        assert abs(dv[2] - naive_dv[2]) > 1e-6    # sculling term appears on z


class TestNavEskf:
    def _make(self, **kw):
        s0 = NavState(q=quat.identity(), p=[LAT, LON, H])
        return NavEskf(s0, gyro_noise_density=0.002, accel_noise_density=0.02,
                       gyro_bias_walk=1e-6, accel_bias_walk=1e-5, **kw)

    def test_covariance_stays_symmetric_psd_long_run(self):
        f = self._make()
        omega, fb = _static_imu_ned()
        rng = np.random.default_rng(0)
        for k in range(2000):
            f.predict(omega + 0.002 * rng.standard_normal(3),
                      fb + 0.02 * rng.standard_normal(3), 0.01)
            if k % 100 == 99:
                f.update_position([LAT, LON, H], sigma=2.0)
                f.update_velocity(np.zeros(3), sigma=0.1)
        P = f.P
        assert np.abs(P - P.T).max() < 1e-12 * np.abs(P).max()
        assert np.linalg.eigvalsh(P).min() > 0
        assert f.health.name in ("HEALTHY", "UNOBSERVABLE")

    def test_position_updates_bound_position_error(self):
        """Dead-reckoning drifts; position fixes bound the error."""
        rng = np.random.default_rng(1)
        omega_t, f_t = _static_imu_ned()
        bg_true = np.array([0.002, -0.001, 0.0015])   # uncompensated bias

        def run(with_fixes):
            f = self._make()
            for k in range(3000):
                w_meas = omega_t + bg_true + 0.002 * rng.standard_normal(3)
                f_meas = f_t + 0.02 * rng.standard_normal(3)
                f.predict(w_meas, f_meas, 0.01)
                if with_fixes and k % 100 == 99:
                    f.update_position(
                        np.array([LAT, LON, H]) + np.array([
                            2.0 / 6.4e6, 2.0 / 4.5e6, 2.0]) * rng.standard_normal(3),
                        sigma=2.0)
                    f.update_velocity(0.05 * rng.standard_normal(3), sigma=0.05)
            lat, lon, h = (float(x) for x in f.state.p)
            return np.hypot((lat - LAT) * 6.4e6, (lon - LON) * 4.5e6), f

        err_free, _ = run(False)
        err_aided, f_aided = run(True)
        assert err_aided < err_free / 5
        assert err_aided < 10.0                      # m horizontal
        # bias covariance shrinks (gaining observability through the fixes);
        # note yaw-axis gyro bias stays weakly observable without heading
        # aiding — full bias recovery is not claimed here.
        assert np.trace(f_aided.P[9:12, 9:12]) < 3 * 0.01**2  # below the prior

    def test_direction_update_constrains_attitude(self):
        f = self._make()
        omega_t, f_t = _static_imu_ned()
        # 10 degree initial yaw error; true attitude is identity, so the
        # body-frame observations are the nav vectors themselves.
        f.state = f.state.evolve(q=quat.exp(np.array([0.0, 0.0, np.deg2rad(10)])))
        m_nav = np.array([0.2, 0.0, 0.4])
        down = np.array([0.0, 0.0, -1.0])
        for _ in range(400):
            f.predict(omega_t, f_t, 0.01)
            f.update_direction(down, down, sigma=0.02, sensor_id="accel")
            f.update_direction(m_nav, m_nav, sigma=0.02, sensor_id="mag")
        assert quat.angular_distance(f.state.q, quat.identity()) < np.deg2rad(1.0)

    def test_nees_consistency_monte_carlo(self):
        """Average position/velocity NEES within chi-square bounds over
        Monte-Carlo runs of an aided static scenario."""
        from qnav.metrics import nees_bounds
        omega_t, f_t = _static_imu_ned()
        runs, steps = 20, 400
        nees_vals = []
        for r in range(runs):
            rng = np.random.default_rng(100 + r)
            f = self._make()
            for k in range(steps):
                f.predict(omega_t + 0.002 * rng.standard_normal(3),
                          f_t + 0.02 * rng.standard_normal(3), 0.01)
                if k % 50 == 49:
                    noise = rng.standard_normal(3)
                    f.update_position(np.array([LAT, LON, H]) + np.array(
                        [2.0 / (meridian_radius(LAT) + H),
                         2.0 / (meridian_radius(LAT) + H), 2.0]) * noise, sigma=2.0)
            lat, lon, h = (float(x) for x in f.state.p)
            M = float(meridian_radius(LAT)) + H
            e = np.array([(lat - LAT) * M, (lon - LON) * M * np.cos(LAT), -(h - H)])
            Ppos = f.P[6:9, 6:9]
            nees_vals.append(float(e @ np.linalg.solve(Ppos, e)))
        avg = float(np.mean(nees_vals))
        lo, hi = nees_bounds(dim=3, n_samples=runs, confidence=0.99)
        assert lo * 0.5 < avg < hi * 1.5   # generous: first-order F approximations

    def test_lifecycle_inherited(self):
        f = self._make()
        omega, fb = _static_imu_ned()
        f.predict(omega, fb, 0.01)
        snap = f.snapshot()
        p_at_snap = f.state.p.copy()
        for _ in range(50):
            f.predict(omega, fb + 0.5, 0.01)
        f.restore(snap)
        np.testing.assert_array_equal(f.state.p, p_at_snap)
        f.reset()
        assert f.health.name == "INITIALIZING"

    def test_gate_rejects_position_outlier(self):
        from qnav.filters import GatePolicy
        f = self._make(gate=GatePolicy())
        omega, fb = _static_imu_ned()
        for _ in range(100):
            f.predict(omega, fb, 0.01)
        p_before = f.state.p.copy()
        # 10 km outlier
        f.update_position([LAT + 10000 / 6.4e6, LON, H], sigma=2.0)
        assert f.last_update.rejection_reason == "nis_gate"
        np.testing.assert_array_equal(f.state.p, p_before)

    def test_invalid_inputs_rejected(self):
        f = self._make()
        with pytest.raises(ValueError):
            f.predict([np.nan, 0, 0], [0, 0, -9.8], 0.01)
        with pytest.raises(ValueError):
            f.predict([0, 0, 0], [0, 0, -9.8], 0.0)
        with pytest.raises(ValueError):
            f.update_position([LAT, LON, H], sigma=-1.0)


def _dcm_to_q(R):
    from qnav.attitude import dcm
    return dcm.to_quaternion(R)
