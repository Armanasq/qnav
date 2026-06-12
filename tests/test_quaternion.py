"""Quaternion algebra: identities, conventions, edge cases, property tests."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qnav.attitude import dcm, quaternion as quat
from tests.conftest import TOL_ALG, TOL_NUM

unit_quats = st.builds(
    lambda a, b, c, d: np.array([a, b, c, d]) / np.linalg.norm([a, b, c, d]),
    *[st.floats(-1, 1).filter(lambda x: abs(x) > 1e-3) for _ in range(4)],
)


class TestAlgebra:
    def test_identity_neutral(self, rng):
        q = quat.random((50,), rng)
        e = quat.identity()
        assert np.allclose(quat.mul(e, q), q, atol=TOL_ALG)
        assert np.allclose(quat.mul(q, e), q, atol=TOL_ALG)

    def test_associativity(self, rng):
        a, b, c = (quat.random((20,), rng) for _ in range(3))
        assert np.allclose(
            quat.mul(quat.mul(a, b), c), quat.mul(a, quat.mul(b, c)), atol=TOL_ALG
        )

    def test_noncommutative(self):
        qx = quat.exp(np.array([0.5, 0, 0]))
        qy = quat.exp(np.array([0, 0.5, 0]))
        assert not np.allclose(quat.mul(qx, qy), quat.mul(qy, qx))

    def test_conjugate_is_inverse_for_unit(self, rng):
        q = quat.random((50,), rng)
        assert np.allclose(quat.mul(q, quat.conjugate(q)), quat.identity((50,)), atol=TOL_ALG)
        assert np.allclose(quat.conjugate(q), quat.inverse(q), atol=TOL_ALG)

    def test_general_inverse(self):
        q = np.array([2.0, 1.0, -1.0, 0.5])
        assert np.allclose(quat.mul(q, quat.inverse(q)), quat.identity(), atol=TOL_ALG)

    def test_norm_multiplicative(self, rng):
        a = rng.standard_normal((20, 4))
        b = rng.standard_normal((20, 4))
        assert np.allclose(quat.norm(quat.mul(a, b)), quat.norm(a) * quat.norm(b), atol=1e-10)

    def test_hamilton_ij_equals_k(self):
        i = np.array([0.0, 1.0, 0.0, 0.0])
        j = np.array([0.0, 0.0, 1.0, 0.0])
        k = np.array([0.0, 0.0, 0.0, 1.0])
        assert np.allclose(quat.mul(i, j), k, atol=TOL_ALG)  # Hamilton, not JPL

    def test_left_right_matrices(self, rng):
        p, q = quat.random((), rng), quat.random((), rng)
        assert np.allclose(quat.left_matrix(p) @ q, quat.mul(p, q), atol=TOL_ALG)
        assert np.allclose(quat.right_matrix(q) @ p, quat.mul(p, q), atol=TOL_ALG)


class TestRotation:
    def test_rotate_matches_dcm(self, rng):
        q = quat.random((50,), rng)
        v = rng.standard_normal((50, 3))
        R = dcm.from_quaternion(q)
        assert np.allclose(
            quat.rotate_vector(q, v), np.einsum("...ij,...j->...i", R, v), atol=TOL_ALG
        )

    def test_rotate_frame_is_inverse(self, rng):
        q = quat.random((20,), rng)
        v = rng.standard_normal((20, 3))
        assert np.allclose(quat.rotate_frame(q, quat.rotate_vector(q, v)), v, atol=TOL_ALG)

    def test_double_cover(self, rng):
        q = quat.random((20,), rng)
        v = rng.standard_normal((20, 3))
        assert np.allclose(quat.rotate_vector(q, v), quat.rotate_vector(-q, v), atol=TOL_ALG)

    def test_composition_convention(self, rng):
        # q_AC = q_AB ⊗ q_BC chains coordinate maps
        q_ab, q_bc = quat.random((), rng), quat.random((), rng)
        v_c = rng.standard_normal(3)
        v_a1 = quat.rotate_vector(quat.mul(q_ab, q_bc), v_c)
        v_a2 = quat.rotate_vector(q_ab, quat.rotate_vector(q_bc, v_c))
        assert np.allclose(v_a1, v_a2, atol=TOL_ALG)


class TestExpLog:
    def test_roundtrip(self, rng):
        phi = rng.uniform(-3.0, 3.0, (200, 3))
        phi = phi[np.linalg.norm(phi, axis=1) < np.pi]
        assert np.allclose(quat.log(quat.exp(phi)), phi, atol=TOL_NUM)

    def test_small_angle(self):
        for mag in [1e-12, 1e-9, 1e-6]:
            phi = np.array([mag, 0, 0])
            q = quat.exp(phi)
            assert abs(np.linalg.norm(q) - 1) < TOL_ALG
            assert np.allclose(quat.log(q), phi, atol=1e-15 + mag * 1e-6)

    def test_zero(self):
        assert np.allclose(quat.exp(np.zeros(3)), quat.identity(), atol=TOL_ALG)
        assert np.allclose(quat.log(quat.identity()), np.zeros(3), atol=TOL_ALG)

    def test_near_pi(self):
        u = np.array([1.0, 2.0, -1.0])
        u /= np.linalg.norm(u)
        for theta in [np.pi - 1e-9, np.pi - 1e-12, np.pi]:
            phi = theta * u
            back = quat.log(quat.exp(phi))
            # at exactly pi the axis sign is a convention; compare rotations
            assert quat.angular_distance(quat.exp(back), quat.exp(phi)) < 1e-7

    def test_power(self, rng):
        q = quat.random((), rng)
        assert np.allclose(quat.power(q, 1.0), quat.canonical(q), atol=TOL_NUM) or \
               np.allclose(quat.power(q, 1.0), -quat.canonical(q), atol=TOL_NUM)
        q_half = quat.power(q, 0.5)
        assert quat.angular_distance(quat.mul(q_half, q_half), q) < TOL_NUM


class TestConventions:
    def test_scalar_last_roundtrip(self, rng):
        q = quat.random((10,), rng)
        assert np.allclose(quat.from_scalar_last(quat.to_scalar_last(q)), q, atol=TOL_ALG)

    def test_jpl_same_rotation_matrix(self, rng):
        # JPL bridge must preserve the physical rotation
        q = quat.random((10,), rng)
        q_back = quat.from_jpl(quat.to_jpl(q))
        assert np.max(quat.angular_distance(q, q_back)) < TOL_ALG

    def test_canonical(self, rng):
        q = quat.random((100,), rng)
        qc = quat.canonical(q)
        assert np.all(qc[:, 0] >= 0)
        assert np.max(quat.angular_distance(q, qc)) < TOL_ALG


class TestMean:
    def test_mean_of_cluster(self, rng):
        q0 = quat.random((), rng)
        deltas = 0.01 * rng.standard_normal((100, 3))
        qs = quat.mul(q0, quat.exp(deltas))
        # randomize signs: mean must be sign-invariant
        signs = np.where(rng.random(100) > 0.5, 1.0, -1.0)
        qm = quat.mean(qs * signs[:, None])
        assert quat.angular_distance(qm, q0) < 0.01

    def test_weighted(self, rng):
        qa, qb = quat.identity(), quat.exp(np.array([0.2, 0, 0]))
        qm = quat.mean(np.stack([qa, qb]), weights=np.array([1.0, 0.0]))
        assert quat.angular_distance(qm, qa) < TOL_NUM


@given(unit_quats, unit_quats)
@settings(max_examples=100, deadline=None)
def test_property_distance_symmetry(q1, q2):
    d12 = quat.angular_distance(q1, q2)
    d21 = quat.angular_distance(q2, q1)
    assert abs(d12 - d21) < 1e-9
    assert 0.0 <= d12 <= np.pi + 1e-12


@given(unit_quats)
@settings(max_examples=100, deadline=None)
def test_property_exp_log(q):
    assert quat.angular_distance(quat.exp(quat.log(q)), q) < 1e-9


def test_normalize_zero_raises():
    with pytest.raises(ValueError):
        quat.normalize(np.zeros(4))


def test_normalize_warns():
    from qnav.errors import NormalizationWarning
    with pytest.warns(NormalizationWarning):
        quat.normalize(np.array([2.0, 0, 0, 0]), warn_tol=1e-3)
