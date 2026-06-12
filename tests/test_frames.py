"""Frames: transforms, graph composition, Earth geodesy, convention bridges."""

import numpy as np
import pytest

from qnav.attitude import dcm, quaternion as quat
from qnav.errors import FrameGraphError, FrameMismatchError
from qnav.frames import FrameGraph, FrameTransform, conventions as fconv, earth
from tests.conftest import TOL_ALG, TOL_NUM


class TestFrameTransform:
    def test_apply_and_inverse(self, rng):
        q = quat.random((), rng)
        t = rng.standard_normal(3)
        T = FrameTransform(target="A", source="B", rotation=q, translation=t)
        p_b = rng.standard_normal(3)
        p_a = T.apply_point(p_b)
        assert np.allclose(T.inverse().apply_point(p_a), p_b, atol=TOL_NUM)

    def test_compose_checks_frames(self, rng):
        T_ab = FrameTransform(target="A", source="B", rotation=quat.random((), rng))
        T_cd = FrameTransform(target="C", source="D", rotation=quat.random((), rng))
        with pytest.raises(FrameMismatchError):
            T_ab @ T_cd

    def test_compose_correctness(self, rng):
        q1, q2 = quat.random((), rng), quat.random((), rng)
        t1, t2 = rng.standard_normal((2, 3))
        T_ab = FrameTransform(target="A", source="B", rotation=q1, translation=t1)
        T_bc = FrameTransform(target="B", source="C", rotation=q2, translation=t2)
        p_c = rng.standard_normal(3)
        assert np.allclose(
            (T_ab @ T_bc).apply_point(p_c), T_ab.apply_point(T_bc.apply_point(p_c)),
            atol=TOL_NUM,
        )

    def test_non_unit_rejected(self):
        with pytest.raises(ValueError):
            FrameTransform(target="A", source="B", rotation=np.array([2.0, 0, 0, 0]))

    def test_inverse_covariance_monte_carlo(self, rng):
        # validate the analytic inverse-covariance Jacobian by sampling
        q = quat.random((), rng)
        t = rng.standard_normal(3)
        P = np.diag([0.02, 0.03, 0.01, 0.05, 0.04, 0.06]) ** 2
        T = FrameTransform(target="A", source="B", rotation=q, translation=t, covariance=P)
        Ti = T.inverse()
        n = 20000
        L = np.linalg.cholesky(P)
        d = rng.standard_normal((n, 6)) @ L.T
        # perturbed transforms, inverted, errors measured in [dtheta, dt]
        errs = np.empty((n, 6))
        R = dcm.from_quaternion(q)
        for i in range(n):
            q_p = quat.mul(q, quat.exp(d[i, :3]))
            t_p = t + d[i, 3:]
            Tp_inv = FrameTransform(target="A", source="B", rotation=q_p, translation=t_p).inverse()
            errs[i, :3] = quat.log(quat.relative(Ti.rotation, Tp_inv.rotation))
            errs[i, 3:] = Tp_inv.translation - Ti.translation
        P_emp = errs.T @ errs / n
        assert np.allclose(P_emp, Ti.covariance, atol=5e-3 * np.max(np.diag(P)) / np.min(np.diag(P)))

    def test_compose_covariance_psd(self, rng):
        P = 0.001 * np.eye(6)
        T1 = FrameTransform(target="A", source="B", rotation=quat.random((), rng),
                            translation=rng.standard_normal(3), covariance=P)
        T2 = FrameTransform(target="B", source="C", rotation=quat.random((), rng),
                            translation=rng.standard_normal(3), covariance=P)
        T = T1 @ T2
        assert np.all(np.linalg.eigvalsh(T.covariance) > 0)


class TestFrameGraph:
    def _graph(self, rng):
        g = FrameGraph()
        self.T_nb = FrameTransform(target="NED", source="body", rotation=quat.random((), rng))
        self.T_bs = FrameTransform(target="body", source="imu", rotation=quat.random((), rng),
                                   translation=np.array([0.1, 0.0, -0.05]))
        g.add(self.T_nb)
        g.add(self.T_bs)
        return g

    def test_direct_and_reverse_lookup(self, rng):
        g = self._graph(rng)
        v = rng.standard_normal(3)
        v_ned = g.transform_vector(v, target="NED", source="imu")
        expected = self.T_nb.apply_vector(self.T_bs.apply_vector(v))
        assert np.allclose(v_ned, expected, atol=TOL_NUM)
        # reverse direction
        back = g.transform_vector(v_ned, target="imu", source="NED")
        assert np.allclose(back, v, atol=TOL_NUM)

    def test_identity(self, rng):
        g = self._graph(rng)
        T = g.get("body", "body")
        assert quat.angular_distance(T.rotation, quat.identity()) < TOL_ALG

    def test_unknown_frame(self, rng):
        g = self._graph(rng)
        with pytest.raises(FrameGraphError):
            g.get("NED", "camera")

    def test_ambiguous_edge_rejected(self, rng):
        g = self._graph(rng)
        with pytest.raises(FrameGraphError):
            g.add(FrameTransform(target="NED", source="imu", rotation=quat.random((), rng)))

    def test_duplicate_edge_replace(self, rng):
        g = self._graph(rng)
        T_new = FrameTransform(target="NED", source="body", rotation=quat.identity())
        with pytest.raises(FrameGraphError):
            g.add(T_new)
        g.add(T_new, replace=True)
        assert quat.angular_distance(g.get("NED", "body").rotation, quat.identity()) < TOL_ALG


class TestEarth:
    def test_geodetic_ecef_roundtrip(self, rng):
        lat = rng.uniform(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, 50)
        lon = rng.uniform(-np.pi, np.pi, 50)
        h = rng.uniform(-100, 10000, 50)
        r = earth.geodetic_to_ecef(lat, lon, h)
        lat2, lon2, h2 = earth.ecef_to_geodetic(r)
        assert np.allclose(lat2, lat, atol=1e-9)
        assert np.allclose(lon2, lon, atol=1e-9)
        assert np.allclose(h2, h, atol=1e-4)

    def test_ecef_reference_points(self):
        # equator / prime meridian: x = a
        r = earth.geodetic_to_ecef(0.0, 0.0, 0.0)
        assert np.allclose(r, [earth.WGS84_A, 0, 0], atol=1e-6)
        # north pole: z = b
        r = earth.geodetic_to_ecef(np.pi / 2, 0.0, 0.0)
        assert np.allclose(r, [0, 0, earth.WGS84_B], atol=1e-6)

    def test_ned_rotation_properties(self, rng):
        lat, lon = 0.7, -1.2
        R = earth.dcm_ecef_to_ned(lat, lon)
        assert np.allclose(R @ R.T, np.eye(3), atol=TOL_ALG)
        assert np.isclose(np.linalg.det(R), 1.0, atol=TOL_ALG)
        # down axis points toward Earth center (opposite ellipsoid normal)
        up_ecef = earth.geodetic_to_ecef(lat, lon, 1.0) - earth.geodetic_to_ecef(lat, lon, 0.0)
        d_ned = R @ (up_ecef / np.linalg.norm(up_ecef))
        assert np.allclose(d_ned, [0, 0, -1], atol=1e-9)

    def test_enu_ned_consistency(self):
        lat, lon = 0.3, 0.9
        v_ecef = np.array([1.0, -2.0, 0.5])
        v_ned = earth.dcm_ecef_to_ned(lat, lon) @ v_ecef
        v_enu = earth.dcm_ecef_to_enu(lat, lon) @ v_ecef
        assert np.allclose(v_enu, earth.DCM_ENU_NED @ v_ned, atol=TOL_ALG)

    def test_normal_gravity_known_values(self):
        # WGS-84: equator 9.7803253359, pole 9.8321849379 (defining values)
        assert abs(earth.normal_gravity(0.0) - 9.7803253359) < 1e-8
        assert abs(earth.normal_gravity(np.pi / 2) - 9.8321849379) < 1e-6
        # free-air gradient ~ -3.086e-6 /m near surface
        dg = earth.normal_gravity(0.5, 1000.0) - earth.normal_gravity(0.5, 0.0)
        assert -3.2e-3 < dg < -2.9e-3

    def test_gravity_vector_signs(self):
        g_ned = earth.gravity_vector(0.5, frame="NED")
        g_enu = earth.gravity_vector(0.5, frame="ENU")
        assert g_ned[2] > 9.7 and g_enu[2] < -9.7

    def test_earth_rate(self):
        w = earth.earth_rate_ned(np.pi / 2)  # north pole: all in -z (down)
        assert np.allclose(w, [0, 0, -earth.WGS84_OMEGA], atol=1e-12)


class TestConventionBridges:
    def test_ned_enu_involution(self, rng):
        v = rng.standard_normal((10, 3))
        assert np.allclose(fconv.enu_to_ned(fconv.ned_to_enu(v)), v, atol=TOL_ALG)

    def test_ned_enu_axes(self):
        # north in NED -> y in ENU; down -> -z
        assert np.allclose(fconv.ned_to_enu([1.0, 0, 0]), [0, 1, 0], atol=TOL_ALG)
        assert np.allclose(fconv.ned_to_enu([0, 0, 1.0]), [0, 0, -1], atol=TOL_ALG)

    def test_attitude_bridge_consistency(self, rng):
        # transforming a vector must commute with the attitude conversion
        q_nf = quat.random((), rng)  # q_NED_FRD
        v_frd = rng.standard_normal(3)
        v_ned = quat.rotate_vector(q_nf, v_frd)
        q_ef = fconv.attitude_ned_frd_to_enu_flu(q_nf)
        v_enu = quat.rotate_vector(q_ef, fconv.frd_to_flu(v_frd))
        assert np.allclose(v_enu, fconv.ned_to_enu(v_ned), atol=TOL_NUM)
        # and the round trip
        q_back = fconv.attitude_enu_flu_to_ned_frd(q_ef)
        assert quat.angular_distance(q_back, q_nf) < TOL_NUM

    def test_level_yaw_attitude_maps_identity(self):
        # level, north-pointing FRD body in NED == level, "north"-pointing FLU in ENU:
        # FLU x must point north = ENU y, i.e. yaw 90 in ENU terms
        q_nf = quat.identity()
        q_ef = fconv.attitude_ned_frd_to_enu_flu(q_nf)
        x_enu = quat.rotate_vector(q_ef, np.array([1.0, 0, 0]))
        assert np.allclose(x_enu, [0, 1, 0], atol=TOL_NUM)
