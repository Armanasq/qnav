"""Wahba solvers: exactness, noise optimality, degeneracy handling."""

import numpy as np
import pytest

from qnav.attitude import dcm, quaternion as quat
from qnav.determination import davenport_q, oleq_q, quest_q, svd_attitude, triad_dcm
from qnav.determination.wahba import attitude_profile, check_observability, loss
from qnav.errors import DegenerateGeometryWarning
from tests.conftest import TOL_NUM

SOLVERS_Q = {"davenport": davenport_q, "quest": quest_q, "oleq": oleq_q}


def make_obs(rng, n=5, noise=0.0):
    q = quat.random((), rng)
    R = dcm.from_quaternion(q)
    vb = rng.standard_normal((n, 3))
    vb /= np.linalg.norm(vb, axis=1, keepdims=True)
    vr = vb @ R.T
    if noise:
        vr = vr + noise * rng.standard_normal(vr.shape)
    return q, R, vr, vb


class TestExact:
    @pytest.mark.parametrize("name", list(SOLVERS_Q))
    def test_noise_free(self, name, rng):
        for _ in range(50):
            q, R, vr, vb = make_obs(rng, n=int(rng.integers(2, 8)))
            assert quat.angular_distance(SOLVERS_Q[name](vr, vb), q) < 1e-9

    def test_svd(self, rng):
        for _ in range(50):
            q, R, vr, vb = make_obs(rng)
            assert np.linalg.norm(svd_attitude(vr, vb) - R) < 1e-9

    def test_triad_exact_primary(self, rng):
        q, R, vr, vb = make_obs(rng, n=2, noise=0.0)
        Rt = triad_dcm(vr[0], vr[1], vb[0], vb[1])
        assert np.linalg.norm(Rt - R) < 1e-9

    def test_triad_primary_priority(self, rng):
        # noise on the secondary must not perturb the primary direction match
        q, R, vr, vb = make_obs(rng, n=2)
        vr_noisy = vr.copy()
        vr_noisy[1] += 0.05 * rng.standard_normal(3)
        vr_noisy[1] /= np.linalg.norm(vr_noisy[1])
        Rt = triad_dcm(vr_noisy[0], vr_noisy[1], vb[0], vb[1])
        assert np.allclose(Rt @ vb[0], vr[0], atol=1e-9)


class TestNearPi:
    @pytest.mark.parametrize("name", list(SOLVERS_Q))
    def test_solvers_at_pi(self, name):
        for axis in ([0, 0, 1.0], [1.0, 0, 0], [1, 1, 0]):
            u = np.asarray(axis) / np.linalg.norm(axis)
            q = quat.exp(np.pi * 0.99999 * u)
            R = dcm.from_quaternion(q)
            vb = np.eye(3)
            vr = vb @ R.T
            assert quat.angular_distance(SOLVERS_Q[name](vr, vb), q) < 1e-6


class TestNoisy:
    @pytest.mark.parametrize("name", list(SOLVERS_Q))
    def test_optimality(self, name, rng):
        # solver loss must match the exact Davenport optimum
        q, R, vr, vb = make_obs(rng, n=8, noise=0.02)
        l_solver = loss(dcm.from_quaternion(SOLVERS_Q[name](vr, vb)), vr, vb)
        l_opt = loss(dcm.from_quaternion(davenport_q(vr, vb)), vr, vb)
        assert l_solver <= l_opt + 1e-10

    def test_weights_pull_solution(self, rng):
        q, R, vr, vb = make_obs(rng, n=2)
        vr2 = vr.copy()
        vr2[1] = np.roll(vr2[1], 1)  # corrupt the second observation
        w_first = davenport_q(vr2, vb, weights=np.array([100.0, 1.0]))
        w_second = davenport_q(vr2, vb, weights=np.array([1.0, 100.0]))
        assert quat.angular_distance(w_first, q) < quat.angular_distance(w_second, q)


class TestDegeneracy:
    def test_collinear_warns(self, rng):
        v = np.array([0.0, 0.0, 1.0])
        vb = np.stack([v, v, v])
        with pytest.warns(DegenerateGeometryWarning):
            assert not check_observability(vb)

    def test_single_vector_warns(self):
        with pytest.warns(DegenerateGeometryWarning):
            assert not check_observability(np.array([[1.0, 0, 0]]))

    def test_davenport_degenerate_warns(self, rng):
        q, R, vr, vb = make_obs(rng, n=1)
        vb = np.vstack([vb, vb])
        vr = np.vstack([vr, vr])
        with pytest.warns(DegenerateGeometryWarning):
            davenport_q(vr, vb)

    def test_triad_collinear_raises(self):
        v = np.array([1.0, 0, 0])
        with pytest.raises(ValueError):
            triad_dcm(v, v, v, v)

    def test_zero_vector_rejected(self):
        with pytest.raises(ValueError):
            attitude_profile(np.zeros((2, 3)), np.ones((2, 3)))

    def test_bad_weights(self, rng):
        q, R, vr, vb = make_obs(rng, n=3)
        with pytest.raises(ValueError):
            davenport_q(vr, vb, weights=np.array([-1.0, 1.0, 1.0]))
