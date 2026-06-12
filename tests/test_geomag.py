"""WMM2025 synthesis against the official NOAA/NCEI test values."""

import numpy as np
import pytest

from qnav.geomag import wmm_elements, wmm_field

# Official WMM2025 test-value rows (subset):
# (year, height_km, lat_deg, lon_deg, X_nT, Y_nT, Z_nT)
OFFICIAL = [
    (2025.0, 0.0, 80.0, 0.0, 6521.6, 145.9, 54791.5),
    (2025.0, 0.0, 0.0, 120.0, 39677.8, -109.6, -10580.2),
    (2025.0, 0.0, -80.0, 240.0, 6117.5, 15751.9, -52022.5),
    (2025.0, 100.0, 80.0, 0.0, 6216.0, 92.4, 52598.8),
    (2025.0, 100.0, 0.0, 120.0, 37688.6, -96.2, -10152.1),
    (2025.0, 100.0, -80.0, 240.0, 5907.6, 14780.3, -49540.7),
    (2027.5, 0.0, 80.0, 0.0, 6500.8, 294.5, 54869.4),
    (2027.5, 0.0, 0.0, 120.0, 39701.6, -167.4, -10381.8),
    (2027.5, 0.0, -80.0, 240.0, 6200.7, 15730.3, -51783.7),
    (2027.5, 100.0, 80.0, 0.0, 6196.7, 233.8, 52670.5),
    (2027.5, 100.0, 0.0, 120.0, 37711.5, -148.7, -9969.8),
    (2027.5, 100.0, -80.0, 240.0, 5984.0, 14760.1, -49317.7),
]


class TestWmm:
    @pytest.mark.parametrize("row", OFFICIAL)
    def test_official_values(self, row):
        year, h_km, lat, lon, x, y, z = row
        est = wmm_field(np.deg2rad(lat), np.deg2rad(lon), h_km * 1000.0, year)
        assert np.allclose(est.ned * 1e9, [x, y, z], atol=0.1)

    def test_derived_elements_consistent(self):
        est = wmm_field(np.deg2rad(48.0), np.deg2rad(11.0), 500.0, 2026.0)
        assert est.total == pytest.approx(np.linalg.norm(est.ned))
        assert est.horizontal == pytest.approx(np.hypot(est.ned[0], est.ned[1]))
        assert est.declination == pytest.approx(np.arctan2(est.ned[1], est.ned[0]))

    def test_elements_feed_heading_stack(self):
        from qnav.heading.magnetic_model import field_from_elements
        d, i, f = wmm_elements(np.deg2rad(48.0), np.deg2rad(11.0), 500.0, 2026.0)
        m = field_from_elements(d, i, f)
        est = wmm_field(np.deg2rad(48.0), np.deg2rad(11.0), 500.0, 2026.0)
        assert np.allclose(m, est.ned, atol=1e-12)

    def test_validity_window(self):
        with pytest.raises(ValueError):
            wmm_field(0.0, 0.0, 0.0, 2031.0)
        with pytest.raises(ValueError):
            wmm_field(0.0, 0.0, 0.0, 2024.9)


class TestEarthCurvature:
    def test_radii_at_45deg(self):
        from qnav.frames.earth import gaussian_radius, meridian_radius, transverse_radius
        lat = np.deg2rad(45.0)
        assert meridian_radius(lat) == pytest.approx(6367381.8, abs=1.0)
        assert transverse_radius(lat) == pytest.approx(6388838.3, abs=1.0)
        g = gaussian_radius(lat)
        assert meridian_radius(lat) < g < transverse_radius(lat)

    def test_transport_rate_signs(self):
        # northward flight rotates the local level frame about -y (pitch-down
        # of the frame); eastward flight about +x and -z (northern hemisphere)
        from qnav.frames.earth import transport_rate_ned
        lat = np.deg2rad(45.0)
        w_n = transport_rate_ned(lat, np.array([100.0, 0.0, 0.0]))
        assert w_n[0] == 0 and w_n[1] < 0 and w_n[2] == 0
        w_e = transport_rate_ned(lat, np.array([0.0, 100.0, 0.0]))
        assert w_e[0] > 0 and w_e[1] == 0 and w_e[2] < 0

    def test_bar_itzhack_extraction(self, rng):
        from qnav.attitude import dcm, quaternion as quat
        q = quat.random((), rng)
        R = dcm.from_quaternion(q)
        # exact input: agrees with Shepperd
        assert quat.angular_distance(dcm.to_quaternion_robust(R), q) < 1e-12
        # noisy input: still returns a sensible nearby attitude
        Rn = R + 1e-3 * rng.standard_normal((3, 3))
        qn = dcm.to_quaternion_robust(Rn)
        assert quat.angular_distance(qn, q) < 5e-3
