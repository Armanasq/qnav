"""Filters: convergence on synthetic MARG, bias estimation, ESKF consistency."""

import numpy as np
import pytest

from qnav.attitude import quaternion as quat
from qnav.filters import (
    ComplementaryFilter, Eskf, MadgwickStyleFilter, MahonyFilter, QuaternionEkf,
)
from qnav.heading.magnetic_model import field_from_elements
from qnav.metrics import average_nees, nees_bounds, rmse_angle
from qnav.metrics.attitude_error import attitude_error_vector
from qnav.validation.datasets import marg_dataset

M_NAV = field_from_elements(0.0, np.deg2rad(60.0), 1.0)
UP_NED = np.array([0.0, 0.0, -1.0])


@pytest.fixture(scope="module")
def ds():
    return marg_dataset(duration=30.0, dt=0.01, seed=7)


def run_eskf(ds, every=5):
    f = Eskf(gyro_noise_density=0.005, gyro_bias_walk=1e-5, nav_frame="NED")
    qs = np.empty((ds.truth.n, 4))
    Ps = np.empty((ds.truth.n, 3, 3))
    for k in range(ds.truth.n):
        f.predict(ds.gyro[k], ds.truth.dt)
        if k % every == 0:
            f.update_gravity(ds.accel[k], sigma=0.02)
            f.update_magnetometer(M_NAV, ds.mag[k], sigma=0.02)
        qs[k] = f.q
        Ps[k] = f.P[:3, :3]
    return f, qs, Ps


class TestEskf:
    def test_convergence(self, ds):
        f, qs, _ = run_eskf(ds)
        n2 = ds.truth.n // 2
        assert np.rad2deg(rmse_angle(qs[n2:], ds.truth.q[n2:])) < 2.0

    def test_bias_estimation(self, ds):
        f, _, _ = run_eskf(ds)
        assert np.allclose(f.bias, [0.02, -0.01, 0.015], atol=5e-3)

    def test_covariance_psd_and_bounded(self, ds):
        f, _, Ps = run_eskf(ds)
        assert np.all(np.linalg.eigvalsh(f.P) > 0)
        assert np.all(f.attitude_std < 0.2)

    def test_consistency_order_of_magnitude(self, ds):
        # average attitude NEES within a loose factor of dim=3 (single run,
        # discretization + linearization tolerated; catches gross P bugs)
        _, qs, Ps = run_eskf(ds)
        n2 = ds.truth.n // 2
        e = attitude_error_vector(qs[n2:], ds.truth.q[n2:])
        a = average_nees(e, Ps[n2:])
        assert 0.2 < a < 20.0, f"average NEES {a}"

    def test_gyro_only_uncertainty_grows(self):
        f = Eskf(gyro_noise_density=0.01)
        s0 = f.attitude_std.copy()
        for _ in range(500):
            f.predict(np.array([0.1, 0.0, -0.05]), 0.01)
        assert np.all(f.attitude_std > s0)

    def test_zero_norm_measurement_rejected(self):
        f = Eskf(gyro_noise_density=0.01)
        with pytest.raises(ValueError):
            f.update_gravity(np.zeros(3), sigma=0.1)


class TestMahony:
    def test_convergence_and_bias(self, ds):
        f = MahonyFilter(kp=1.0, ki=0.3)
        qs = np.empty((ds.truth.n, 4))
        vn = np.stack([UP_NED, M_NAV])
        for k in range(ds.truth.n):
            vb = np.stack([ds.accel[k], ds.mag[k]])
            f.step(ds.gyro[k], ds.truth.dt, v_nav=vn, v_body=vb)
            qs[k] = f.q
        n2 = ds.truth.n // 2
        assert np.rad2deg(rmse_angle(qs[n2:], ds.truth.q[n2:])) < 3.0
        assert np.allclose(f.bias, [0.02, -0.01, 0.015], atol=8e-3)

    def test_static_equilibrium(self):
        # with perfect measurements at identity, the filter must stay put
        f = MahonyFilter(kp=2.0, ki=0.5)
        vn = np.stack([UP_NED, M_NAV])
        for _ in range(200):
            f.step(np.zeros(3), 0.01, v_nav=vn, v_body=vn)
        assert quat.angular_distance(f.q, quat.identity()) < 1e-9
        assert np.allclose(f.bias, 0.0, atol=1e-12)


class TestMadgwick:
    def test_levels_from_large_initial_error(self, ds):
        q_wrong = quat.exp(np.array([0.8, -0.5, 0.3]))
        f = MadgwickStyleFilter(beta=0.2, q0=quat.mul(ds.truth.q[0], q_wrong))
        vn = np.stack([UP_NED, M_NAV])
        for k in range(ds.truth.n):
            f.step(ds.gyro[k], ds.truth.dt, v_nav=vn,
                   v_body=np.stack([ds.accel[k], ds.mag[k]]))
        assert quat.angular_distance(f.q, ds.truth.q[-1]) < np.deg2rad(15)

    def test_gradient_matches_fd(self, rng):
        f = MadgwickStyleFilter(q0=quat.random((), rng))
        vn = rng.standard_normal((2, 3))
        vb = rng.standard_normal((2, 3))
        g = f.objective_gradient(vn, vb)
        # finite difference of F via the *sandwich product* (the analytic
        # gradient's off-manifold extension; the fast Rodrigues path differs
        # for non-unit q, which the FD perturbation produces)
        vnu = vn / np.linalg.norm(vn, axis=1, keepdims=True)
        vbu = vb / np.linalg.norm(vb, axis=1, keepdims=True)

        def rot_frame_sandwich(q, v):
            p = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
            return quat.mul(quat.mul(quat.conjugate(q), p), q)[..., 1:]

        def F(q):
            e = rot_frame_sandwich(q, vnu) - vbu
            return 0.5 * np.sum(e * e)

        eps = 1e-7
        g_fd = np.array([
            (F(f.q + eps * np.eye(4)[i]) - F(f.q - eps * np.eye(4)[i])) / (2 * eps)
            for i in range(4)
        ])
        assert np.allclose(g, g_fd, atol=1e-5)


class TestComplementary:
    def test_convergence(self, ds):
        f = ComplementaryFilter(gain=0.02)
        qs = np.empty((ds.truth.n, 4))
        for k in range(ds.truth.n):
            f.predict(ds.gyro[k], ds.truth.dt)
            f.update(ds.accel[k], ds.mag[k])
            qs[k] = f.q
        n2 = ds.truth.n // 2
        assert np.rad2deg(rmse_angle(qs[n2:], ds.truth.q[n2:])) < 3.0

    def test_gain_validation(self):
        with pytest.raises(ValueError):
            ComplementaryFilter(gain=1.5)


class TestQuaternionEkf:
    def test_stability_and_norm(self, ds):
        f = QuaternionEkf(gyro_noise_density=0.005)
        for k in range(ds.truth.n):
            f.predict(ds.gyro[k], ds.truth.dt)
            if k % 5 == 0:
                f.update_direction(UP_NED, ds.accel[k], sigma=0.05)
                f.update_direction(M_NAV, ds.mag[k], sigma=0.05)
        assert abs(np.linalg.norm(f.q) - 1.0) < 1e-12
        # no bias state -> bounded but nonzero error
        assert quat.angular_distance(f.q, ds.truth.q[-1]) < np.deg2rad(20)


class TestNeesBounds:
    def test_chi2_interval(self):
        lo, hi = nees_bounds(dim=3, n_samples=100)
        assert lo < 3.0 < hi
        lo2, hi2 = nees_bounds(dim=3, n_samples=10000)
        assert hi2 - lo2 < hi - lo  # tightens with samples
