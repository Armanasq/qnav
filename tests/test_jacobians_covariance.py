"""Jacobians vs finite differences; covariance propagation vs Monte-Carlo."""

import numpy as np

from qnav.attitude import covariance as cov, dcm, jacobians as jac, quaternion as quat, so3
from tests.conftest import TOL_FD


class TestJacobians:
    def test_drotate_dtheta_fd(self, rng):
        eps = 1e-7
        for _ in range(10):
            q = quat.random((), rng)
            v = rng.standard_normal(3)
            J = jac.drotate_dtheta(q, v)
            J_fd = np.zeros((3, 3))
            for i in range(3):
                d = np.zeros(3)
                d[i] = eps
                J_fd[:, i] = (
                    quat.rotate_vector(quat.mul(q, quat.exp(d)), v)
                    - quat.rotate_vector(q, v)
                ) / eps
            assert np.allclose(J, J_fd, atol=TOL_FD)

    def test_drotate_dq_fd_sandwich(self, rng):
        def sandwich(q, v):
            p = np.concatenate([[0.0], v])
            return quat.mul(quat.mul(q, p), quat.conjugate(q))[1:]

        eps = 1e-7
        q = quat.random((), rng)
        v = rng.standard_normal(3)
        J = jac.drotate_dq(q, v)
        J_fd = np.zeros((3, 4))
        for i in range(4):
            d = np.zeros(4)
            d[i] = eps
            J_fd[:, i] = (sandwich(q + d, v) - sandwich(q, v)) / eps
        assert np.allclose(J, J_fd, atol=1e-5)

    def test_composition_jacobians_fd(self, rng):
        eps = 1e-7
        Ra = so3.exp(rng.uniform(-1, 1, 3))
        Rb = so3.exp(rng.uniform(-1, 1, 3))
        Jl = jac.dcomposition_left(Ra, Rb)
        Jr = jac.dcomposition_right(Ra, Rb)
        for i in range(3):
            d = np.zeros(3)
            d[i] = eps
            fl = so3.boxminus(so3.boxplus(Ra, d) @ Rb, Ra @ Rb) / eps
            fr = so3.boxminus(Ra @ so3.boxplus(Rb, d), Ra @ Rb) / eps
            assert np.allclose(Jl[:, i], fl, atol=TOL_FD)
            assert np.allclose(Jr[:, i], fr, atol=TOL_FD)

    def test_inverse_jacobian_fd(self, rng):
        eps = 1e-7
        R = so3.exp(rng.uniform(-1, 1, 3))
        J = jac.dinverse_dtheta(R)
        for i in range(3):
            d = np.zeros(3)
            d[i] = eps
            f = so3.boxminus(so3.boxplus(R, d).T, R.T) / eps
            assert np.allclose(J[:, i], f, atol=TOL_FD)

    def test_dlog_dR_fd(self, rng):
        eps = 1e-7
        R = so3.exp(rng.uniform(-1.5, 1.5, 3))
        J = jac.dlog_dR_local(R)
        for i in range(3):
            d = np.zeros(3)
            d[i] = eps
            f = (so3.log(so3.boxplus(R, d)) - so3.log(R)) / eps
            assert np.allclose(J[:, i], f, atol=TOL_FD)


class TestCovariance:
    def test_local_global_roundtrip(self, rng):
        q = quat.random((), rng)
        P = np.diag([0.01, 0.02, 0.03])
        assert np.allclose(
            cov.global_to_local(cov.local_to_global(P, q), q), P, atol=1e-12
        )

    def test_propagation_monte_carlo(self, rng):
        # one gyro step: analytic propagation vs sampled
        q0 = quat.random((), rng)
        P0 = np.diag([0.02, 0.01, 0.015]) ** 2
        w = np.array([0.5, -0.3, 0.8])
        dt = 0.1
        psd = 1e-4
        P1 = cov.propagate_gyro(P0, w, dt, psd)
        n = 40000
        L = np.linalg.cholesky(P0)
        d0 = rng.standard_normal((n, 3)) @ L.T
        noise = np.sqrt(psd / dt) * rng.standard_normal((n, 3))
        q_nom = quat.mul(q0, quat.exp(w * dt))
        q_samp = quat.mul(quat.mul(q0, quat.exp(d0)), quat.exp((w + noise) * dt))
        e = quat.log(quat.mul(quat.conjugate(q_nom)[None, :].repeat(n, 0), q_samp))
        P_emp = e.T @ e / n
        assert np.allclose(P_emp, P1, atol=0.1 * np.max(np.diag(P1)))

    def test_compose_covariance_monte_carlo(self, rng):
        q_ab = quat.random((), rng)
        q_bc = quat.random((), rng)
        P_ab = np.diag([0.02, 0.03, 0.01]) ** 2
        P_bc = np.diag([0.01, 0.02, 0.025]) ** 2
        R_bc = dcm.from_quaternion(q_bc)
        P_ac = cov.compose_covariance(P_ab, P_bc, R_bc)
        n = 40000
        d1 = rng.standard_normal((n, 3)) @ np.linalg.cholesky(P_ab).T
        d2 = rng.standard_normal((n, 3)) @ np.linalg.cholesky(P_bc).T
        q_nom = quat.mul(q_ab, q_bc)
        q_samp = quat.mul(quat.mul(q_ab, quat.exp(d1)), quat.mul(q_bc, quat.exp(d2)))
        e = quat.log(quat.mul(quat.conjugate(q_nom)[None, :].repeat(n, 0), q_samp))
        P_emp = e.T @ e / n
        assert np.allclose(P_emp, P_ac, atol=0.1 * np.max(np.diag(P_ac)))

    def test_sampling_matches_request(self, rng):
        q0 = quat.random((), rng)
        P = np.diag([0.05, 0.02, 0.03]) ** 2
        qs = cov.sample_attitudes(q0, P, 50000, rng)
        e = quat.log(quat.mul(quat.conjugate(q0)[None, :].repeat(qs.shape[0], 0), qs))
        P_emp = e.T @ e / qs.shape[0]
        assert np.allclose(P_emp, P, atol=0.05 * np.max(np.diag(P)))

    def test_is_psd(self):
        assert cov.is_psd(np.eye(3))
        assert not cov.is_psd(np.diag([1.0, -0.1, 1.0]))
        assert not cov.is_psd(np.array([[1.0, 2.0, 0], [0.0, 1, 0], [0, 0, 1]]))


class TestInvariantsModule:
    def test_all_invariants_near_zero(self, rng):
        from qnav.validation import invariants as inv
        q = quat.random((200,), rng)
        R = dcm.from_quaternion(q)
        phi = rng.uniform(-1, 1, (200, 3))
        assert inv.quaternion_norm_violation(q) < 1e-12
        assert inv.dcm_orthogonality_violation(R) < 1e-12
        assert inv.dcm_det_violation(R) < 1e-12
        assert inv.double_cover_violation(q) < 1e-9
        assert inv.conversion_roundtrip_violation(q) < 1e-9
        assert inv.exp_log_roundtrip_violation(phi) < 1e-9
        q2 = quat.random((200,), rng)
        assert inv.composition_consistency_violation(q, q2) < 1e-12
