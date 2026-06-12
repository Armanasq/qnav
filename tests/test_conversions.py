"""Conversion layer: round trips, reference cases, singularities, conventions."""

import numpy as np
import pytest

from qnav.attitude import conversions as conv
from qnav.attitude import dcm, euler, mrp, quaternion as quat
from qnav.attitude.euler import ALLOWED_SEQUENCES
from qnav.errors import ConventionError, GimbalLockWarning
from qnav.validation.reference_cases import EULER_ZYX_CASES, QUATERNION_DCM_CASES
from tests.conftest import TOL_ALG, TOL_NUM


class TestReferenceCases:
    @pytest.mark.parametrize("name,q,R", QUATERNION_DCM_CASES, ids=[c[0] for c in QUATERNION_DCM_CASES])
    def test_quat_to_dcm(self, name, q, R):
        assert np.allclose(dcm.from_quaternion(q), R, atol=TOL_ALG)

    @pytest.mark.parametrize("name,q,R", QUATERNION_DCM_CASES, ids=[c[0] for c in QUATERNION_DCM_CASES])
    def test_dcm_to_quat(self, name, q, R):
        assert quat.angular_distance(dcm.to_quaternion(R), q) < TOL_NUM

    @pytest.mark.parametrize("name,ypr,q", EULER_ZYX_CASES, ids=[c[0] for c in EULER_ZYX_CASES])
    def test_euler_zyx(self, name, ypr, q):
        qc = euler.to_quaternion(np.array(ypr), "ZYX")
        assert quat.angular_distance(qc, q) < TOL_NUM


class TestRoundTrips:
    def test_quat_dcm(self, rng):
        q = quat.random((500,), rng)
        q2 = dcm.to_quaternion(dcm.from_quaternion(q))
        assert np.max(quat.angular_distance(q, q2)) < TOL_NUM

    def test_dcm_to_quat_near_pi(self):
        # Shepperd branches: rotations by pi about many axes
        for u in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -1, 2]):
            u = np.asarray(u, dtype=float)
            u /= np.linalg.norm(u)
            q = quat.exp((np.pi - 1e-12) * u)
            R = dcm.from_quaternion(q)
            assert quat.angular_distance(dcm.to_quaternion(R), q) < 1e-7

    @pytest.mark.parametrize("seq", ALLOWED_SEQUENCES)
    def test_euler_all_sequences(self, seq, rng):
        q = quat.random((100,), rng)
        R = dcm.from_quaternion(q)
        a = euler.from_dcm(R, seq)
        assert np.allclose(euler.to_dcm(a, seq), R, atol=1e-9)

    @pytest.mark.parametrize("seq", ["zyx", "xyz", "zxz", "yxy"])
    def test_euler_extrinsic(self, seq, rng):
        q = quat.random((50,), rng)
        R = dcm.from_quaternion(q)
        a = euler.from_dcm(R, seq)
        assert np.allclose(euler.to_dcm(a, seq), R, atol=1e-9)

    def test_intrinsic_extrinsic_duality(self, rng):
        # intrinsic ABC(a,b,c) == extrinsic cba(c,b,a)
        ang = rng.uniform(-1.0, 1.0, 3)
        R1 = euler.to_dcm(ang, "ZYX")
        R2 = euler.to_dcm(ang[::-1], "xyz")
        assert np.allclose(R1, R2, atol=TOL_ALG)

    def test_mrp(self, rng):
        q = quat.random((100,), rng)
        s = mrp.from_quaternion(q)
        assert np.all(np.linalg.norm(s, axis=-1) <= 1.0 + 1e-12)
        assert np.max(quat.angular_distance(mrp.to_quaternion(s), q)) < TOL_NUM

    def test_mrp_shadow_same_attitude(self, rng):
        q = quat.random((20,), rng)
        s = mrp.from_quaternion(q)
        s_shadow = mrp.shadow(s)
        assert np.max(quat.angular_distance(mrp.to_quaternion(s_shadow), q)) < TOL_NUM

    def test_gibbs(self, rng):
        q = quat.random((100,), rng)
        # exclude near-pi where Gibbs is singular
        keep = quat.angle(q) < 3.0
        q = q[keep]
        g = mrp.gibbs_from_quaternion(q)
        assert np.max(quat.angular_distance(mrp.gibbs_to_quaternion(g), q)) < TOL_NUM

    def test_gibbs_singular_raises(self):
        q_pi = quat.exp(np.array([0.0, 0.0, np.pi]))
        with pytest.raises(ValueError):
            mrp.gibbs_from_quaternion(q_pi)

    def test_convert_hub(self, rng):
        q = quat.random((10,), rng)
        for rep in ("dcm", "euler", "rotvec", "mrp"):
            x = conv.convert(q, "quat", rep)
            q2 = conv.convert(x, rep, "quat")
            assert np.max(quat.angular_distance(q, q2)) < TOL_NUM

    def test_convert_unknown(self):
        with pytest.raises(ValueError):
            conv.convert(np.eye(3), "dcm", "nope")


class TestGimbalLock:
    def test_warning_and_reconstruction(self):
        # pitch = +90 deg exactly (ZYX Tait-Bryan lock)
        ang = np.array([0.4, np.pi / 2, 0.3])
        R = euler.to_dcm(ang, "ZYX")
        with pytest.warns(GimbalLockWarning):
            a = euler.from_dcm(R, "ZYX")
        # third angle zeroed, but the rotation must be preserved
        assert a[2] == 0.0
        assert np.allclose(euler.to_dcm(a, "ZYX"), R, atol=1e-9)

    def test_proper_sequence_lock(self):
        ang = np.array([0.7, 0.0, 0.2])  # ZXZ with middle angle 0
        R = euler.to_dcm(ang, "ZXZ")
        with pytest.warns(GimbalLockWarning):
            a = euler.from_dcm(R, "ZXZ")
        assert np.allclose(euler.to_dcm(a, "ZXZ"), R, atol=1e-9)

    def test_near_lock_no_blowup(self):
        for d in [1e-6, 1e-9]:
            ang = np.array([0.5, np.pi / 2 - d, -0.2])
            R = euler.to_dcm(ang, "ZYX")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = euler.from_dcm(R, "ZYX")
            assert np.allclose(euler.to_dcm(a, "ZYX"), R, atol=1e-7)


class TestSequenceValidation:
    @pytest.mark.parametrize("bad", ["XXY", "ZZ", "ZYXY", "Zyx", "ABC", ""])
    def test_invalid_sequences(self, bad):
        with pytest.raises(ConventionError):
            euler.to_dcm(np.zeros(3), bad)


class TestDcmHelpers:
    def test_principal_rotations(self):
        assert np.allclose(dcm.rot_z(np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=TOL_ALG)
        assert np.allclose(dcm.rot_x(np.pi / 2) @ [0, 1, 0], [0, 0, 1], atol=TOL_ALG)
        assert np.allclose(dcm.rot_y(np.pi / 2) @ [0, 0, 1], [1, 0, 0], atol=TOL_ALG)

    def test_orthonormalize(self, rng):
        R = dcm.from_quaternion(quat.random((), rng))
        M = R + 1e-4 * rng.standard_normal((3, 3))
        Rp = dcm.orthonormalize(M)
        assert dcm.orthogonality_error(Rp) < 1e-12
