"""Tests for the one-call batch estimation API (qnav.estimate_attitude)."""

import numpy as np
import pytest

from qnav import AttitudeEstimate, estimate_attitude
from qnav.attitude import kinematics as kin
from qnav.attitude import quaternion as quat
from qnav.errors import ConventionError
from qnav.filters.contracts import EstimatorHealth

ALL_METHODS = [
    "eskf", "invariant", "ukf", "ekf", "mahony", "madgwick",
    "complementary", "aqua", "fourati", "roleq", "fkf",
]

G_NED = np.array([0.0, 0.0, 9.81])
M_NED = np.array([0.4, 0.0, 0.5])          # dipping field, NED


def make_truth(n=600, dt=0.01, omega=(0.15, -0.1, 0.35), q0=(0.2, -0.1, 0.5),
               nav_frame="NED", rng=None, gyro_noise=0.0, accel_noise=0.0,
               mag_noise=0.0):
    """Rotating rigid body with exact sensors (+ optional white noise)."""
    w = np.asarray(omega, dtype=float)
    q = np.empty((n, 4))
    q[0] = quat.exp(np.asarray(q0, dtype=float))
    for k in range(1, n):
        q[k] = kin.integrate_exponential(q[k - 1], w, dt)
    g_nav = G_NED if nav_frame == "NED" else np.array([0.0, 0.0, -9.81])
    m_nav = M_NED if nav_frame == "NED" else np.array([0.0, 0.4, -0.5])
    f = np.stack([quat.rotate_frame(qk, -g_nav) for qk in q])
    m = np.stack([quat.rotate_frame(qk, m_nav) for qk in q])
    gyro = np.tile(w, (n, 1))
    if rng is not None:
        gyro = gyro + gyro_noise * rng.standard_normal((n, 3))
        f = f + accel_noise * rng.standard_normal((n, 3))
        m = m + mag_noise * rng.standard_normal((n, 3))
    return q, gyro, f, m, dt


class TestAllMethodsConverge:
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_noiseless_tracking(self, method):
        q_true, gyro, f, m, dt = make_truth()
        est = estimate_attitude(gyro, f, m, dt=dt, method=method)
        assert isinstance(est, AttitudeEstimate)
        assert est.q.shape == (len(gyro), 4)
        err = np.rad2deg(quat.angular_distance(est.q[-1], q_true[-1]))
        assert err < 1.0, f"{method}: {err:.3f} deg final error"

    @pytest.mark.parametrize("method", ["eskf", "mahony", "madgwick"])
    def test_noisy_tracking(self, method):
        rng = np.random.default_rng(3)
        q_true, gyro, f, m, dt = make_truth(
            n=1500, rng=rng, gyro_noise=0.005, accel_noise=0.05, mag_noise=0.01)
        est = estimate_attitude(gyro, f, m, dt=dt, method=method)
        err = np.rad2deg(quat.angular_distance(est.q[-1], q_true[-1]))
        assert err < 3.0, f"{method}: {err:.3f} deg final error under noise"


class TestOutputs:
    def test_covariance_bias_health(self):
        q_true, gyro, f, m, dt = make_truth()
        est = estimate_attitude(gyro, f, m, dt=dt, method="eskf")
        assert est.attitude_std is not None and est.attitude_std.shape == (len(gyro), 3)
        assert np.all(est.attitude_std > 0)
        # uncertainty must shrink as measurements accumulate
        assert np.all(est.attitude_std[-1] < est.attitude_std[0])
        assert est.gyro_bias is not None and est.gyro_bias.shape == (3,)
        assert est.health is EstimatorHealth.HEALTHY
        assert est.n_updates_applied > 0 and est.n_updates_skipped == 0
        assert len(est) == len(gyro)

    def test_no_covariance_methods_return_none(self):
        _, gyro, f, m, dt = make_truth(n=50)
        est = estimate_attitude(gyro, f, m, dt=dt, method="madgwick")
        assert est.attitude_std is None
        assert est.gyro_bias is None

    def test_euler_dcm_heading_converters(self):
        q_true, gyro, f, m, dt = make_truth()
        est = estimate_attitude(gyro, f, m, dt=dt, method="eskf")
        e = est.euler("ZYX")
        assert e.shape == (len(gyro), 3)
        np.testing.assert_allclose(est.euler("ZYX", degrees=True), np.rad2deg(e))
        R = est.dcm()
        assert R.shape == (len(gyro), 3, 3)
        np.testing.assert_allclose(R[-1] @ R[-1].T, np.eye(3), atol=1e-12)
        h = est.heading()
        assert h.shape == (len(gyro),)
        assert np.all((h >= 0) & (h < 2 * np.pi))

    def test_heading_matches_known_yaw(self):
        # static body yawed 30 deg in NED: heading must be 30 deg (magnetic)
        n, dt = 200, 0.01
        q = quat.exp(np.array([0.0, 0.0, np.deg2rad(30)]))
        gyro = np.zeros((n, 3))
        f = np.tile(quat.rotate_frame(q, -G_NED), (n, 1))
        m = np.tile(quat.rotate_frame(q, M_NED), (n, 1))
        est = estimate_attitude(gyro, f, m, dt=dt, method="eskf",
                                mag_ref=M_NED)
        assert est.heading(degrees=True)[-1] == pytest.approx(30.0, abs=0.1)

    def test_filter_object_continues_online(self):
        _, gyro, f, m, dt = make_truth(n=50)
        est = estimate_attitude(gyro, f, m, dt=dt, method="eskf")
        q_before = est.filter.q.copy()
        est.filter.predict(np.array([0.5, 0.0, 0.0]), dt)   # keep stepping
        assert not np.allclose(est.filter.q, q_before)

    def test_to_dataframe(self):
        pd = pytest.importorskip("pandas")
        _, gyro, f, m, dt = make_truth(n=30)
        df = estimate_attitude(gyro, f, m, dt=dt, method="eskf").to_dataframe()
        assert isinstance(df, pd.DataFrame) and len(df) == 30
        for col in ("t", "q_w", "yaw", "att_std_x"):
            assert col in df.columns


class TestInitialization:
    def test_vector_init_beats_identity(self):
        # large initial attitude: FQA init must land near truth immediately
        q_true, gyro, f, m, dt = make_truth(q0=(0.8, -0.5, 1.2))
        est = estimate_attitude(gyro, f, m, dt=dt, method="eskf")
        err0 = np.rad2deg(quat.angular_distance(est.q[0], q_true[0]))
        assert err0 < 1.0, f"vector init off by {err0:.2f} deg"

    def test_explicit_q0_respected(self):
        _, gyro, f, m, dt = make_truth(n=20)
        q0 = quat.exp(np.array([0.0, 0.0, 1.0]))
        est = estimate_attitude(gyro, f, m, dt=dt, method="eskf", q0=q0)
        np.testing.assert_allclose(est.q[0], q0, atol=1e-12)

    def test_gyro_only_dead_reckoning(self):
        q_true, gyro, _, _, dt = make_truth(q0=(0.0, 0.0, 0.0))
        est = estimate_attitude(gyro, dt=dt, method="eskf")
        err = np.rad2deg(quat.angular_distance(est.q[-1], q_true[-1]))
        assert err < 0.5     # perfect gyro, identity init matches truth at k=0

    def test_enu_flu_matched_pair(self):
        q_true, gyro, f, m, dt = make_truth(nav_frame="ENU")
        est = estimate_attitude(gyro, f, m, dt=dt, method="eskf",
                                nav_frame="ENU")
        err = np.rad2deg(quat.angular_distance(est.q[-1], q_true[-1]))
        assert err < 1.0

    def test_ned_only_method_raises_on_enu(self):
        _, gyro, f, m, dt = make_truth(n=10)
        with pytest.raises(ConventionError):
            estimate_attitude(gyro, f, m, dt=dt, method="aqua", nav_frame="ENU")


class TestTiming:
    def test_timestamps_match_uniform_dt(self):
        q_true, gyro, f, m, dt = make_truth()
        t = np.arange(len(gyro)) * dt + 1234.5
        a = estimate_attitude(gyro, f, m, dt=dt, method="mahony")
        b = estimate_attitude(gyro, f, m, t=t, method="mahony")
        np.testing.assert_allclose(a.q, b.q, atol=1e-12)
        np.testing.assert_allclose(b.t, t)

    def test_variable_rate(self):
        rng = np.random.default_rng(7)
        n = 400
        dts = 0.01 + 0.004 * rng.random(n - 1)          # jittered 100 Hz
        t = np.concatenate([[0.0], np.cumsum(dts)])
        w = np.array([0.15, -0.1, 0.35])
        q = np.empty((n, 4))
        q[0] = quat.exp(np.array([0.2, -0.1, 0.5]))
        for k in range(1, n):
            q[k] = kin.integrate_exponential(q[k - 1], w, dts[k - 1])
        f = np.stack([quat.rotate_frame(qk, -G_NED) for qk in q])
        m = np.stack([quat.rotate_frame(qk, M_NED) for qk in q])
        est = estimate_attitude(np.tile(w, (n, 1)), f, m, t=t, method="eskf")
        err = np.rad2deg(quat.angular_distance(est.q[-1], q[-1]))
        assert err < 0.5


class TestDropouts:
    @pytest.mark.parametrize("method", ["eskf", "mahony", "aqua", "fkf"])
    def test_nan_rows_skipped_not_fused(self, method):
        q_true, gyro, f, m, dt = make_truth()
        f = f.copy()
        m = m.copy()
        f[100:150] = np.nan                            # accel dropout
        m[200:220] = np.nan                            # mag dropout
        est = estimate_attitude(gyro, f, m, dt=dt, method=method)
        assert np.all(np.isfinite(est.q))
        assert est.n_updates_skipped == 70
        err = np.rad2deg(quat.angular_distance(est.q[-1], q_true[-1]))
        assert err < 1.0, f"{method}: {err:.3f} deg with dropouts"

    def test_nan_gyro_raises_with_row(self):
        _, gyro, f, m, dt = make_truth(n=20)
        gyro = gyro.copy()
        gyro[7, 1] = np.nan
        with pytest.raises(ValueError, match="row 7"):
            estimate_attitude(gyro, f, m, dt=dt)

    def test_leading_nan_defers_vector_init(self):
        q_true, gyro, f, m, dt = make_truth(q0=(0.8, -0.5, 1.2))
        f = f.copy()
        f[:5] = np.nan                # init must use row 5, not identity
        est = estimate_attitude(gyro, f, m, dt=dt, method="eskf")
        err0 = np.rad2deg(quat.angular_distance(est.q[0], q_true[0]))
        assert err0 < 2.0


class TestValidation:
    def test_unknown_method(self):
        with pytest.raises(ValueError, match="unknown method"):
            estimate_attitude(np.zeros((5, 3)), dt=0.01, method="kalmanator")

    def test_dt_and_t_are_exclusive(self):
        g = np.zeros((5, 3))
        with pytest.raises(ValueError, match="exactly one"):
            estimate_attitude(g, dt=0.01, t=np.arange(5.0))
        with pytest.raises(ValueError, match="exactly one"):
            estimate_attitude(g)

    def test_nonincreasing_t(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            estimate_attitude(np.zeros((3, 3)), t=np.array([0.0, 1.0, 1.0]))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="accel"):
            estimate_attitude(np.zeros((5, 3)), np.zeros((4, 3)), dt=0.01)
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            estimate_attitude(np.zeros((5, 2)), dt=0.01)

    def test_acc_mag_required_methods(self):
        g = np.zeros((5, 3))
        f = np.tile([0.0, 0.0, -9.81], (5, 1))
        for method in ("fourati", "roleq", "fkf"):
            with pytest.raises(ValueError, match="requires both"):
                estimate_attitude(g, f, dt=0.01, method=method)

    def test_bad_mag_ref(self):
        _, gyro, f, m, dt = make_truth(n=10)
        with pytest.raises(ValueError, match="mag_ref"):
            estimate_attitude(gyro, f, m, dt=dt, mag_ref=np.zeros(3))

    def test_filter_kwargs_forwarded(self):
        _, gyro, f, m, dt = make_truth(n=20)
        est = estimate_attitude(gyro, f, m, dt=dt, method="mahony",
                                filter_kwargs={"kp": 2.5, "ki": 0.0})
        assert est.filter.kp == 2.5 and est.filter.ki == 0.0

    def test_single_sample(self):
        est = estimate_attitude(np.zeros((1, 3)),
                                np.array([[0.0, 0.0, -9.81]]), dt=0.01)
        assert est.q.shape == (1, 4)
