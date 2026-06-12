"""Heading: tilt equations, compass under tilt, declination, disturbance gates."""

import numpy as np
import pytest

from qnav.attitude import euler, quaternion as quat
from qnav.errors import DegenerateGeometryWarning
from qnav.heading import compass, declination as decl, disturbance, magnetic_model as magmod
from qnav.heading.tilt_compensation import detilt, roll_pitch_from_accel
from tests.conftest import TOL_NUM

G = 9.80665


def make_truth(yaw, pitch, roll):
    """Specific force and field a NED/FRD body at (yaw, pitch, roll) measures."""
    q = euler.to_quaternion(np.array([yaw, pitch, roll]), "ZYX")  # q_NED_FRD
    g_ned = np.array([0.0, 0.0, G])
    m_ned = magmod.field_from_elements(0.0, np.deg2rad(60.0), 50.0)
    f_body = quat.rotate_frame(q, -g_ned)   # f = R_BN(−g)
    m_body = quat.rotate_frame(q, m_ned)
    return f_body, m_body


class TestTilt:
    @pytest.mark.parametrize("roll,pitch", [
        (0.0, 0.0), (0.3, 0.2), (-0.5, 0.4), (1.2, -1.0), (np.pi - 0.1, 0.0),
    ])
    def test_roll_pitch_recovery(self, roll, pitch):
        f, _ = make_truth(0.7, pitch, roll)
        r, p = roll_pitch_from_accel(f)
        assert abs(r - roll) < TOL_NUM
        assert abs(p - pitch) < TOL_NUM

    def test_yaw_invariance(self):
        # roll/pitch from accel must not depend on yaw
        for yaw in np.linspace(0, 2 * np.pi, 7):
            f, _ = make_truth(yaw, 0.25, -0.4)
            r, p = roll_pitch_from_accel(f)
            assert abs(r + 0.4) < TOL_NUM and abs(p - 0.25) < TOL_NUM

    def test_free_fall_warns(self):
        with pytest.warns(DegenerateGeometryWarning):
            r, p = roll_pitch_from_accel(np.zeros(3))
        assert r == 0.0 and p == 0.0

    def test_detilt_levels_gravity(self):
        f, _ = make_truth(0.3, 0.5, -0.7)
        r, p = roll_pitch_from_accel(f)
        f_lev = detilt(f, r, p)
        assert np.allclose(f_lev, [0, 0, -G], atol=1e-9)


class TestCompass:
    @pytest.mark.parametrize("yaw", np.linspace(0.0, 2 * np.pi, 9)[:-1])
    @pytest.mark.parametrize("tilt", [(0.0, 0.0), (0.4, -0.3), (-0.9, 0.6)])
    def test_heading_under_tilt(self, yaw, tilt):
        roll, pitch = tilt
        f, m = make_truth(yaw, pitch, roll)
        r, p = roll_pitch_from_accel(f)
        psi = compass.magnetic_heading(m, r, p)
        assert abs(compass.heading_difference(psi, yaw)) < 1e-9

    def test_full_solution_with_declination(self):
        f, m = make_truth(1.0, 0.2, -0.1)
        d = np.deg2rad(7.0)
        r, p, psi_true = compass.heading_from_accel_mag(f, m, declination=d)
        assert abs(compass.heading_difference(psi_true, 1.0 + d)) < 1e-9

    def test_zero_horizontal_field_warns(self):
        m_vertical = np.array([0.0, 0.0, 50.0])
        with pytest.warns(DegenerateGeometryWarning):
            psi = compass.magnetic_heading(m_vertical, 0.0, 0.0)
        assert psi == 0.0

    def test_wrap(self):
        assert abs(compass.wrap_heading(-0.1) - (2 * np.pi - 0.1)) < TOL_NUM
        assert compass.wrap_heading(2 * np.pi) == 0.0

    def test_difference_range(self, rng):
        a, b = rng.uniform(0, 2 * np.pi, (2, 100))
        d = compass.heading_difference(a, b)
        assert np.all(d > -np.pi) and np.all(d <= np.pi)

    def test_variance(self):
        var = compass.heading_variance(np.array([20.0, 0.0]), sigma_m=1.0)
        assert abs(var - 1.0 / 400.0) < 1e-12


class TestDeclination:
    def test_apply_remove(self, rng):
        psi = rng.uniform(0, 2 * np.pi, 20)
        d = np.deg2rad(11.5)
        assert np.allclose(
            decl.remove_declination(decl.apply_declination(psi, d), d), psi, atol=TOL_NUM
        )


class TestMagneticModel:
    def test_elements_roundtrip(self, rng):
        D, I, B = 0.2, 1.1, 48.0
        m = magmod.field_from_elements(D, I, B)
        D2, I2, B2 = magmod.elements_from_field(m)
        assert abs(D - D2) < TOL_NUM and abs(I - I2) < TOL_NUM and abs(B - B2) < TOL_NUM

    def test_ned_enu(self):
        m_ned = magmod.field_from_elements(0.1, 0.9, 1.0, frame="NED")
        m_enu = magmod.field_from_elements(0.1, 0.9, 1.0, frame="ENU")
        from qnav.frames.conventions import ned_to_enu
        assert np.allclose(m_enu, ned_to_enu(m_ned), atol=TOL_NUM)

    def test_dipole_poles_equator(self):
        eq = magmod.dipole_field(0.0)
        pole = magmod.dipole_field(np.pi / 2)
        assert eq[2] == 0.0 and abs(pole[0]) < 1e-20
        assert abs(np.linalg.norm(pole) / np.linalg.norm(eq) - 2.0) < 1e-12
        # magnitudes in a plausible terrestrial range (20–70 uT)
        assert 2e-5 < np.linalg.norm(eq) < 7e-5


class TestDisturbance:
    def test_gates(self):
        f, m = make_truth(0.5, 0.1, -0.2)
        r, p = roll_pitch_from_accel(f)
        ok = disturbance.is_field_trustworthy(
            m, r, p, ref_intensity=50.0, ref_inclination=np.deg2rad(60),
            tol_intensity=2.0, tol_inclination=np.deg2rad(3),
        )
        assert bool(ok)
        bad = m + np.array([30.0, 0, 0])
        ok2 = disturbance.is_field_trustworthy(
            bad, r, p, ref_intensity=50.0, ref_inclination=np.deg2rad(60),
            tol_intensity=2.0, tol_inclination=np.deg2rad(3),
        )
        assert not bool(ok2)

    def test_monitor_rejects_jump(self):
        mon = disturbance.HeadingMonitor(psi0=1.0, gate=np.deg2rad(10))
        psi = mon.update(psi_mag=1.02, yaw_rate=0.0, dt=0.01)
        assert abs(psi - 1.02) < TOL_NUM and not mon.last_rejected
        psi = mon.update(psi_mag=2.5, yaw_rate=0.0, dt=0.01)  # 85 deg jump
        assert mon.last_rejected and abs(psi - 1.02) < TOL_NUM

    def test_monitor_gyro_fallback(self):
        mon = disturbance.HeadingMonitor(psi0=0.0)
        for _ in range(100):
            mon.update(psi_mag=None, yaw_rate=0.1, dt=0.01)
        assert abs(mon.psi - 0.1) < TOL_NUM
