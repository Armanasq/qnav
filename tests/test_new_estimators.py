"""SAAM, FAMC, FLAE, FQA determination + AQUA, Fourati, FKF, ROLEQ, UKF filters.

Validation strategy: synthetic truth round-trips (exact recovery on clean
data), Wahba-optimality cross-checks against QUEST, and convergence on the
canonical noisy MARG dataset.
"""

import numpy as np
import pytest

from qnav.attitude import dcm, quaternion as quat
from qnav.determination import famc_q, flae_q, fqa_q, saam_q
from qnav.determination.quest import quest
from qnav.errors import DegenerateGeometryWarning
from qnav.filters import (
    AquaFilter, FastKalmanFilter, FouratiFilter, RoleqFilter, UkfAttitude,
)
from qnav.filters.roleq import _involution
from qnav.heading.magnetic_model import field_from_elements
from qnav.metrics import rmse_angle
from qnav.validation.datasets import marg_dataset

DIP = np.deg2rad(60.0)
M_NED = np.array([np.cos(DIP), 0.0, np.sin(DIP)])
G_DOWN = np.array([0.0, 0.0, 1.0])
M_NAV = field_from_elements(0.0, DIP, 1.0)
UP_NED = np.array([0.0, 0.0, -1.0])


def synth_pair(q, rng=None, noise=0.0):
    """Body-frame specific force + magnetic field for attitude q_NB."""
    R = dcm.from_quaternion(q)
    f = R.T @ (-G_DOWN) * 9.81
    m = R.T @ M_NED * 48.0
    if noise:
        f = f + noise * 9.81 * rng.standard_normal(3)
        m = m + noise * 48.0 * rng.standard_normal(3)
    return f, m


@pytest.fixture(scope="module")
def ds():
    return marg_dataset(duration=30.0, dt=0.01, seed=7)


class TestClosedFormDetermination:
    @pytest.mark.parametrize("solver", [saam_q, famc_q])
    def test_exact_recovery_batchable(self, solver, rng):
        qs = quat.random((100,), rng)
        R = dcm.from_quaternion(qs)
        f = np.einsum("nij,j->ni", np.swapaxes(R, -1, -2), -G_DOWN) * 9.81
        m = np.einsum("nij,j->ni", np.swapaxes(R, -1, -2), M_NED) * 48.0
        qe = solver(f, m)
        assert np.rad2deg(quat.angular_distance(qe, qs).max()) < 1e-6

    @pytest.mark.parametrize("solver", [saam_q, famc_q])
    def test_zero_norm_raises(self, solver):
        with pytest.raises(ValueError):
            solver(np.zeros(3), M_NED)

    def test_fqa_exact_recovery(self, rng):
        for _ in range(50):
            q = quat.random((), rng)
            f, m = synth_pair(q)
            qe = fqa_q(f, m, m_ref=M_NED)
            assert quat.angular_distance(qe, q) < 1e-9

    def test_fqa_tilt_immune_to_mag_disturbance(self, rng):
        # corrupt the magnetometer arbitrarily: gravity direction must be
        # reproduced exactly (structural property of the factored form)
        q = quat.random((), rng)
        f, _ = synth_pair(q)
        qe = fqa_q(f, rng.standard_normal(3) * 100.0, m_ref=M_NED)
        g_est = quat.rotate_frame(qe, G_DOWN)
        assert np.allclose(g_est, -f / np.linalg.norm(f), atol=1e-9)

    def test_fqa_singularity_warns(self):
        # pitch = +90 deg: specific force along +x only
        with pytest.warns(DegenerateGeometryWarning):
            fqa_q(np.array([9.81, 0.0, 0.0]), np.array([10.0, 20.0, -30.0]))

    def test_flae_matches_quest_under_noise(self, rng):
        q = quat.random((), rng)
        R = dcm.from_quaternion(q)
        vr = rng.standard_normal((5, 3))
        vr /= np.linalg.norm(vr, axis=1, keepdims=True)
        vb = vr @ R + 0.01 * rng.standard_normal((5, 3))
        assert quat.angular_distance(flae_q(vr, vb), quest(vr, vb)) < 1e-10

    def test_flae_exact_recovery(self, rng):
        for _ in range(50):
            q = quat.random((), rng)
            R = dcm.from_quaternion(q)
            vr = rng.standard_normal((3, 3))
            vr /= np.linalg.norm(vr, axis=1, keepdims=True)
            qe = flae_q(vr, vr @ R)
            assert quat.angular_distance(qe, q) < 1e-9


class TestAqua:
    def test_initialize_exact(self, rng):
        for _ in range(50):
            q = quat.random((), rng)
            f, m = synth_pair(q)
            qe = AquaFilter().initialize(f, m)
            assert quat.angular_distance(qe, q) < 1e-9

    def test_convergence(self, ds):
        f = AquaFilter(alpha=0.02, beta=0.02)
        f.initialize(ds.accel[0], ds.mag[0])
        qs = np.empty((ds.truth.n, 4))
        for k in range(ds.truth.n):
            qs[k] = f.step(ds.gyro[k], ds.truth.dt, ds.accel[k], ds.mag[k])
        n2 = ds.truth.n // 2
        assert np.rad2deg(rmse_angle(qs[n2:], ds.truth.q[n2:])) < 4.0

    def test_mag_cannot_touch_tilt(self, rng):
        # after convergence on clean data, inject a strong magnetic
        # disturbance: the gravity alignment must be unaffected
        q = quat.random((), rng)
        f_b, m_b = synth_pair(q)
        filt = AquaFilter(alpha=0.5, beta=0.5)
        filt.initialize(f_b, m_b)
        for _ in range(50):
            filt.step(np.zeros(3), 0.01, f_b, rng.standard_normal(3) * 100.0)
        g_est = quat.rotate_frame(filt.q, G_DOWN)
        assert np.allclose(g_est, -f_b / np.linalg.norm(f_b), atol=1e-6)

    def test_adaptive_gain_gates_maneuvers(self):
        f = AquaFilter(alpha=0.1, adaptive=True)
        assert f._effective_alpha(9.80665) == pytest.approx(0.1)
        assert f._effective_alpha(9.80665 * 1.5) == 0.0


class TestFourati:
    def test_static_convergence_from_large_error(self, rng):
        q = quat.random((), rng)
        f_b, m_b = synth_pair(q)
        filt = FouratiFilter(gain=5.0, m_ref=M_NED)
        for _ in range(2000):
            filt.step(np.zeros(3), 0.01, f_b, m_b)
        assert quat.angular_distance(filt.q, q) < 1e-9

    def test_dataset_convergence(self, ds):
        filt = FouratiFilter(gain=5.0, m_ref=M_NAV)
        qs = np.empty((ds.truth.n, 4))
        for k in range(ds.truth.n):
            qs[k] = filt.step(ds.gyro[k], ds.truth.dt, ds.accel[k], ds.mag[k])
        n2 = ds.truth.n // 2
        assert np.rad2deg(rmse_angle(qs[n2:], ds.truth.q[n2:])) < 3.0


class TestRoleq:
    def test_involution_property(self, rng):
        q = quat.random((), rng)
        R = dcm.from_quaternion(q)
        vr = rng.standard_normal(3)
        vr /= np.linalg.norm(vr)
        W = _involution(vr, R.T @ vr)
        assert np.allclose(W, W.T)
        assert np.allclose(W @ W, np.eye(4))     # involution
        assert np.allclose(W @ q, q)             # true q is +1 eigenvector

    def test_dataset_convergence(self, ds):
        filt = RoleqFilter(m_ref=M_NAV)
        qs = np.empty((ds.truth.n, 4))
        for k in range(ds.truth.n):
            qs[k] = filt.step(ds.gyro[k], ds.truth.dt, ds.accel[k], ds.mag[k])
        n2 = ds.truth.n // 2
        assert np.rad2deg(rmse_angle(qs[n2:], ds.truth.q[n2:])) < 6.0


class TestFkf:
    def test_dataset_convergence(self, ds):
        filt = FastKalmanFilter(gyro_noise=0.01, accel_noise=0.02, mag_noise=0.02)
        qs = np.empty((ds.truth.n, 4))
        for k in range(ds.truth.n):
            qs[k] = filt.step(ds.gyro[k], ds.truth.dt, ds.accel[k], ds.mag[k])
        n2 = ds.truth.n // 2
        assert np.rad2deg(rmse_angle(qs[n2:], ds.truth.q[n2:])) < 12.0

    def test_covariance_stays_finite_psd(self, ds):
        filt = FastKalmanFilter()
        for k in range(500):
            filt.step(ds.gyro[k], ds.truth.dt, ds.accel[k], ds.mag[k])
        assert np.all(np.isfinite(filt.P))
        assert np.all(np.linalg.eigvalsh(filt.P) > -1e-12)


class TestUkf:
    def run(self, ds, q0=None, P0=None, n=None):
        u = UkfAttitude(gyro_noise_density=0.005, q0=q0, P0=P0)
        n = n or ds.truth.n
        qs = np.empty((n, 4))
        for k in range(n):
            u.predict(ds.gyro[k], ds.truth.dt)
            if k % 5 == 0:
                u.update_direction(UP_NED, ds.accel[k], sigma=0.02)
                u.update_direction(M_NAV, ds.mag[k], sigma=0.02)
            qs[k] = u.q
        return u, qs

    def test_convergence(self, ds):
        _, qs = self.run(ds)
        n2 = ds.truth.n // 2
        assert np.rad2deg(rmse_angle(qs[n2:], ds.truth.q[n2:])) < 4.0

    def test_large_initial_error_recovery(self, ds):
        # 115 deg initial error with matching covariance: the ESKF's
        # linearization typically fails here; the UKF must recover
        q_wrong = quat.mul(ds.truth.q[0], quat.exp(np.array([1.5, -1.0, 0.8])))
        u, _ = self.run(ds, q0=q_wrong, P0=(1.5**2) * np.eye(3), n=2000)
        assert np.rad2deg(quat.angular_distance(u.q, ds.truth.q[1999])) < 5.0

    def test_covariance_psd(self, ds):
        u, _ = self.run(ds, n=500)
        assert np.all(np.linalg.eigvalsh(u.P) > 0)
        assert np.all(u.attitude_std < 0.2)
