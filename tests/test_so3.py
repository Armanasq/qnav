"""SO(3): exp/log, Jacobians (vs finite differences), box operators."""

import numpy as np
import pytest

from qnav.attitude import so3
from tests.conftest import TOL_ALG, TOL_FD, TOL_NUM


class TestHatVee:
    def test_roundtrip(self, rng):
        w = rng.standard_normal((20, 3))
        assert np.allclose(so3.vee(so3.hat(w)), w, atol=TOL_ALG)

    def test_cross_product(self, rng):
        a, b = rng.standard_normal((2, 3))
        assert np.allclose(so3.hat(a) @ b, np.cross(a, b), atol=TOL_ALG)

    def test_skew(self, rng):
        W = so3.hat(rng.standard_normal(3))
        assert np.allclose(W, -W.T, atol=TOL_ALG)


class TestExpLog:
    def test_exp_is_rotation(self, rng):
        phi = rng.uniform(-np.pi, np.pi, (100, 3))
        R = so3.exp(phi)
        assert np.all(so3.is_rotation(R, tol=1e-10))

    def test_roundtrip_principal(self, rng):
        phi = rng.standard_normal((200, 3))
        n = np.linalg.norm(phi, axis=1, keepdims=True)
        phi = phi / n * (rng.uniform(0, np.pi - 1e-6, (200, 1)))
        assert np.allclose(so3.log(so3.exp(phi)), phi, atol=TOL_NUM)

    def test_small_angles(self):
        for mag in [0.0, 1e-12, 1e-8, 1e-5]:
            phi = mag * np.array([1.0, 0, 0])
            assert np.allclose(so3.log(so3.exp(phi)), phi, atol=1e-12)

    def test_near_pi(self):
        for u in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [-1, 2, 0.5]):
            u = np.asarray(u, dtype=float)
            u /= np.linalg.norm(u)
            for theta in [np.pi, np.pi - 1e-8, np.pi - 1e-11]:
                R = so3.exp(theta * u)
                back = so3.log(R)
                assert np.allclose(so3.exp(back), R, atol=1e-6)

    def test_exp_rodrigues_reference(self):
        # 90 deg about z, written out by hand
        R = so3.exp(np.array([0.0, 0.0, np.pi / 2]))
        assert np.allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=TOL_ALG)


class TestJacobians:
    def _fd_right(self, phi, eps=1e-7):
        J = np.zeros((3, 3))
        R0 = so3.exp(phi)
        for i in range(3):
            d = np.zeros(3)
            d[i] = eps
            J[:, i] = so3.log(R0.T @ so3.exp(phi + d)) / eps
        return J

    def test_right_jacobian_fd(self, rng):
        for _ in range(20):
            phi = rng.uniform(-2.5, 2.5, 3)
            assert np.allclose(so3.right_jacobian(phi), self._fd_right(phi), atol=TOL_FD)

    def test_left_right_relation(self, rng):
        phi = rng.standard_normal((10, 3))
        assert np.allclose(so3.left_jacobian(phi), so3.right_jacobian(-phi), atol=TOL_ALG)
        Jl = so3.left_jacobian(phi)
        Jr = so3.right_jacobian(phi)
        assert np.allclose(Jl, np.swapaxes(Jr, -1, -2), atol=TOL_ALG)

    def test_inverses(self, rng):
        phi = rng.uniform(-2.5, 2.5, (20, 3))
        I = np.broadcast_to(np.eye(3), (20, 3, 3))
        assert np.allclose(so3.left_jacobian(phi) @ so3.left_jacobian_inverse(phi), I, atol=1e-9)
        assert np.allclose(so3.right_jacobian(phi) @ so3.right_jacobian_inverse(phi), I, atol=1e-9)

    def test_small_angle_jacobians(self):
        phi = 1e-10 * np.array([1.0, -2.0, 0.5])
        assert np.allclose(so3.right_jacobian(phi), np.eye(3) - 0.5 * so3.hat(phi), atol=1e-12)
        assert np.allclose(so3.right_jacobian_inverse(phi), np.eye(3) + 0.5 * so3.hat(phi), atol=1e-12)


class TestBoxOps:
    def test_boxplus_boxminus(self, rng):
        for _ in range(20):
            R1 = so3.exp(rng.uniform(-2, 2, 3))
            R2 = so3.exp(rng.uniform(-2, 2, 3))
            assert np.allclose(so3.boxplus(R2, so3.boxminus(R1, R2)), R1, atol=1e-9)

    def test_geodesic(self, rng):
        R = so3.exp(rng.uniform(-1, 1, 3))
        assert so3.geodesic_distance(R, R) < TOL_NUM
        Rz = so3.exp(np.array([0, 0, 0.7]))
        assert abs(so3.geodesic_distance(np.eye(3), Rz) - 0.7) < TOL_NUM

    def test_distance_bi_invariance(self, rng):
        R1 = so3.exp(rng.uniform(-1, 1, 3))
        R2 = so3.exp(rng.uniform(-1, 1, 3))
        G = so3.exp(rng.uniform(-1, 1, 3))
        d = so3.geodesic_distance(R1, R2)
        assert abs(so3.geodesic_distance(G @ R1, G @ R2) - d) < TOL_NUM
        assert abs(so3.geodesic_distance(R1 @ G, R2 @ G) - d) < TOL_NUM


class TestProject:
    def test_projection_fixed_point(self, rng):
        R = so3.exp(rng.uniform(-1, 1, 3))
        assert np.allclose(so3.project(R), R, atol=TOL_NUM)

    def test_repairs_perturbation(self, rng):
        R = so3.exp(rng.uniform(-1, 1, 3))
        M = R + 1e-3 * rng.standard_normal((3, 3))
        Rp = so3.project(M)
        assert np.all(so3.is_rotation(Rp, tol=1e-10))
        assert so3.geodesic_distance(R, Rp) < 5e-3

    def test_reflection_corrected(self):
        M = np.diag([1.0, 1.0, -1.0])  # det = −1
        Rp = so3.project(M)
        assert np.linalg.det(Rp) > 0
