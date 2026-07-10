"""IMU preintegration (Phase 7): cross-checks against recursive propagation,
finite-difference bias Jacobians, and Monte-Carlo covariance consistency."""

import numpy as np
import pytest

from qnav.attitude import dcm as dcm_mod
from qnav.attitude import quaternion as quat
from qnav.nav.preintegration import ImuPreintegration

RNG = np.random.default_rng(7)
DT = 0.005
N = 200  # 1 s interval


def _trajectory_imu(n=N, seed=3):
    """Smoothly varying synthetic IMU signal (rates and specific forces)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    w = np.stack([0.4 * np.sin(2 * t), 0.3 * np.cos(3 * t), 0.2 * np.sin(t + 1)], axis=1)
    f = np.stack([1.0 * np.sin(t), -9.81 + 0.5 * np.cos(2 * t), 2.0 * np.sin(3 * t)], axis=1)
    return w + 0.0 * rng.standard_normal((n, 3)), f


def _preintegrate(w, f, bg=None, ba=None):
    p = ImuPreintegration(0.002, 0.02, bg_ref=bg, ba_ref=ba)
    for k in range(w.shape[0]):
        p.integrate(w[k], f[k], DT)
    return p.result()


def _recursive_reference(w, f, bg=None, ba=None):
    """Naive per-sample recursion of the defining preintegration equations."""
    bg = np.zeros(3) if bg is None else bg
    ba = np.zeros(3) if ba is None else ba
    dq = quat.identity()
    dv = np.zeros(3)
    dp = np.zeros(3)
    for k in range(w.shape[0]):
        R = dcm_mod.from_quaternion(dq)
        a = f[k] - ba
        dp = dp + dv * DT + 0.5 * R @ a * DT**2
        dv = dv + R @ a * DT
        dq = quat.normalize(quat.mul(dq, quat.exp((w[k] - bg) * DT)))
    return dq, dv, dp


class TestCrossCheckRecursive:
    def test_deltas_match_recursion_exactly(self):
        w, f = _trajectory_imu()
        pre = _preintegrate(w, f)
        dq_ref, dv_ref, dp_ref = _recursive_reference(w, f)
        assert quat.angular_distance(pre.delta_rotation, dq_ref) < 1e-14
        np.testing.assert_allclose(pre.delta_velocity, dv_ref, atol=1e-12)
        np.testing.assert_allclose(pre.delta_position, dp_ref, atol=1e-12)
        assert pre.dt_total == pytest.approx(N * DT)

    def test_state_advance_matches_ned_mechanization(self):
        """Applying the preintegrated deltas to a NavState matches step-by-
        step NED mechanization within Earth-rate/curvature bounds (short
        interval, moderate rates)."""
        from qnav.frames.earth import gravity_vector
        from qnav.nav import NavState, propagate_ned

        w, f = _trajectory_imu()
        lat, lon, h = np.deg2rad(45.0), 0.0, 100.0
        q0 = quat.exp(np.array([0.02, -0.05, 0.3]))
        s = NavState(q=q0, v=np.array([1.0, 2.0, -0.5]), p=[lat, lon, h])
        for k in range(N):
            s = propagate_ned(s, w[k], f[k], DT)

        pre = _preintegrate(w, f)
        T = pre.dt_total
        g = gravity_vector(lat, h)
        R0 = dcm_mod.from_quaternion(q0)
        v_pre = np.array([1.0, 2.0, -0.5]) + g * T + R0 @ pre.delta_velocity
        q_pre = quat.mul(q0, pre.delta_rotation)

        # Earth rate over 1 s ~ 7e-5 rad; curvature effects ~ mm level
        assert quat.angular_distance(s.q, q_pre) < 2e-4
        np.testing.assert_allclose(s.v, v_pre, atol=2e-3)


class TestBiasJacobians:
    @pytest.mark.parametrize("which", ["bg", "ba"])
    def test_first_order_correction_matches_reintegration(self, which):
        w, f = _trajectory_imu()
        pre = _preintegrate(w, f)
        db = np.array([1e-3, -2e-3, 1.5e-3])
        bg = db if which == "bg" else np.zeros(3)
        ba = db if which == "ba" else np.zeros(3)

        dq_c, dv_c, dp_c = pre.corrected(bg, ba)
        exact = _preintegrate(w, f, bg=bg, ba=ba)

        # first-order correction error is O(|db|^2)
        assert quat.angular_distance(dq_c, exact.delta_rotation) < 5e-6
        np.testing.assert_allclose(dv_c, exact.delta_velocity, atol=5e-5)
        np.testing.assert_allclose(dp_c, exact.delta_position, atol=5e-5)

    def test_jacobians_match_finite_differences(self):
        w, f = _trajectory_imu(n=50)
        pre = _preintegrate(w, f)
        eps = 1e-6
        for j in range(3):
            db = np.zeros(3)
            db[j] = eps
            pg = _preintegrate(w, f, bg=db)
            pa = _preintegrate(w, f, ba=db)
            # rotation Jacobian: Log(dR0^-1 dR(bg)) / eps
            dtheta = quat.log(quat.mul(quat.conjugate(pre.delta_rotation),
                                       pg.delta_rotation)) / eps
            np.testing.assert_allclose(dtheta, pre.J_R_bg[:, j], atol=1e-3)
            np.testing.assert_allclose((pg.delta_velocity - pre.delta_velocity) / eps,
                                       pre.J_v_bg[:, j], atol=1e-3)
            np.testing.assert_allclose((pg.delta_position - pre.delta_position) / eps,
                                       pre.J_p_bg[:, j], atol=1e-3)
            np.testing.assert_allclose((pa.delta_velocity - pre.delta_velocity) / eps,
                                       pre.J_v_ba[:, j], atol=1e-6)
            np.testing.assert_allclose((pa.delta_position - pre.delta_position) / eps,
                                       pre.J_p_ba[:, j], atol=1e-6)


class TestCovariance:
    def test_monte_carlo_consistency(self):
        """Empirical spread of noisy preintegrations matches the reported
        9x9 covariance (NEES within generous chi-square bounds)."""
        w, f = _trajectory_imu(n=100)
        sigma_g, sigma_a = 0.01, 0.1
        clean = _preintegrate(w, f)

        runs = 200
        errs = np.zeros((runs, 9))
        rng = np.random.default_rng(11)
        for r in range(runs):
            p = ImuPreintegration(sigma_g, sigma_a)
            for k in range(w.shape[0]):
                wn = w[k] + sigma_g / np.sqrt(DT) * rng.standard_normal(3)
                fn = f[k] + sigma_a / np.sqrt(DT) * rng.standard_normal(3)
                p.integrate(wn, fn, DT)
            res = p.result()
            errs[r, 0:3] = quat.log(quat.mul(quat.conjugate(clean.delta_rotation),
                                             res.delta_rotation))
            errs[r, 3:6] = res.delta_velocity - clean.delta_velocity
            errs[r, 6:9] = res.delta_position - clean.delta_position

        # rebuild the covariance with the test noise levels
        p_ref = ImuPreintegration(sigma_g, sigma_a)
        for k in range(w.shape[0]):
            p_ref.integrate(w[k], f[k], DT)
        P = p_ref.result().covariance
        nees = np.mean([e @ np.linalg.solve(P, e) for e in errs])
        # chi2(9): mean 9; accept 7..11.5 for 200 runs (first-order model)
        assert 7.0 < nees < 11.5

    def test_covariance_symmetric_psd_and_growing(self):
        w, f = _trajectory_imu()
        p = ImuPreintegration(0.002, 0.02)
        prev_trace = 0.0
        for k in range(w.shape[0]):
            p.integrate(w[k], f[k], DT)
            P = p.P
            assert np.abs(P - P.T).max() < 1e-15 + 1e-12 * np.abs(P).max()
            tr = float(np.trace(P))
            assert tr >= prev_trace
            prev_trace = tr
        assert np.linalg.eigvalsh(p.P).min() >= -1e-18


class TestLifecycle:
    def test_reset_clears_interval(self):
        w, f = _trajectory_imu(n=10)
        p = ImuPreintegration(0.002, 0.02)
        for k in range(10):
            p.integrate(w[k], f[k], DT)
        p.reset()
        assert p.dt_total == 0.0
        np.testing.assert_array_equal(p.dv, 0.0)
        assert quat.angular_distance(p.dq, quat.identity()) == 0.0

    def test_invalid_inputs(self):
        p = ImuPreintegration(0.002, 0.02)
        with pytest.raises(ValueError):
            p.integrate([np.nan, 0, 0], [0, 0, -9.8], DT)
        with pytest.raises(ValueError):
            p.integrate([0, 0, 0], [0, 0, -9.8], 0.0)
        with pytest.raises(ValueError):
            ImuPreintegration(-1.0, 0.02)
