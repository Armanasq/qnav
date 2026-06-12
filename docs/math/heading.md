# Heading and Tilt

Heading estimation from low-cost MARG sensors (Magnetic, Angular Rate, Gravity) is a three-step process: estimate tilt from the accelerometer, de-rotate the magnetometer by the tilt, then compute the magnetic heading and correct for declination.

Each step has a specific failure mode: free-fall for tilt, magnetic disturbance for heading, large accelerations for both. qnav handles each explicitly.

## Tilt from accelerometer

When the vehicle is stationary (or nearly so), the accelerometer measures **specific force**:

\[
\mathbf{f}^b = -\mathbf{R}_{bn}\, \mathbf{g}^n + \delta_a
\]

In NED where \(\mathbf{g}^n = [0, 0, g]^\top\), a static sensor reading \([f_x, f_y, f_z]\) gives:

\[
\phi = \text{atan2}(-f_y, -f_z), \quad \theta = \text{atan2}(f_x, \sqrt{f_y^2 + f_z^2})
\]

(The sign of \(-f_y, -f_z\) rather than \(f_y, f_z\) comes from the specific-force convention: the accelerometer reads \(-g\) up when at rest in NED, where down is positive z.)

**Near free-fall** (\(\|\mathbf{f}^b\| < \text{threshold}\)): both atan2 arguments go to zero and the result is numerically undefined. qnav raises `DegenerateGeometryWarning` and returns the last valid estimate. Free-fall detection is a gate, not an approximation.

```python
from qnav.heading.tilt_compensation import roll_pitch_from_accel

roll, pitch = roll_pitch_from_accel(f_body, frame="NED")
```

## Tilt DCM and de-rotation

The tilt DCM is the rotation that takes the horizontal plane to the body frame:

\[
\mathbf{C}_{\text{tilt}} = R_y(\theta) R_x(\phi)
\]

De-rotating the magnetometer reading removes the tilt effect:

\[
\mathbf{m}^h = \mathbf{C}_{\text{tilt}}^\top \mathbf{m}^b
\]

The de-rotated reading \(\mathbf{m}^h = [m_x^h, m_y^h, m_z^h]\) has its x-y plane aligned with the horizontal (Earth surface), making heading extraction valid.

## Tilt-compensated heading

The magnetic heading (measured from magnetic north, clockwise):

\[
\psi_{\text{mag}} = \text{atan2}(-m_y^h, m_x^h)
\]

The explicit tilt-compensation formula (without intermediate DCM computation):

\[
m_x^h = m_x \cos\theta + m_y \sin\phi\sin\theta + m_z \cos\phi\sin\theta
\]

\[
m_y^h = m_y \cos\phi - m_z \sin\phi
\]

## Magnetic declination

Magnetic north differs from geographic north by the **declination angle** \(D\) (positive east):

\[
\psi_{\text{true}} = \psi_{\text{mag}} + D
\]

Declination varies by location (±30° in extreme cases) and changes slowly over time (~0.1°/year). For navigation, the local declination must be applied. qnav provides `apply_declination` and `remove_declination` to convert between magnetic and true headings.

For precise work, qnav ships a full **World Magnetic Model** implementation (`qnav.geomag`): degree-12 spherical-harmonic synthesis with the WMM2025 coefficients, geodetic→geocentric conversion on the WGS-84 ellipsoid, Schmidt semi-normalized Legendre recursion, and linear secular variation. It is validated against the official NOAA/NCEI test values to < 0.1 nT per component:

```python
from qnav.geomag import wmm_field, wmm_elements
import numpy as np

est = wmm_field(np.deg2rad(48.1), np.deg2rad(11.6), h=520.0, year=2026.4)
est.ned           # field vector [X, Y, Z] in NED, Tesla
est.declination   # D [rad], positive east — feed apply_declination
est.inclination   # I [rad], positive down — feed the dip disturbance gate
est.ned_sv        # secular variation, T/year

# Or directly in the form the heading stack consumes:
d, i, f = wmm_elements(np.deg2rad(48.1), np.deg2rad(11.6), 520.0, 2026.4)
```

For quick simulation work without geographic fidelity, `dipole_field` provides a simple dipole model:

```python
from qnav.heading.magnetic_model import field_from_elements, dipole_field

# Field from IGRF-style elements: declination, inclination, magnitude
M_NED = field_from_elements(
    declination=np.deg2rad(-3.0),      # 3° west
    inclination=np.deg2rad(67.0),      # 67° dip (UK typical)
    magnitude=50e-6,                   # ~50 µT
)
```

## Heading conventions: magnetic heading vs yaw

These are different things and must not be interchanged:

| | Heading | Yaw (ZYX Euler) |
|---|---|---|
| Reference | North (geographic or magnetic) | Body x-axis projected on horizontal |
| Direction | Clockwise positive | Counter-clockwise positive (right-hand rule) |
| Range | \([0°, 360°)\) | \((-180°, 180°]\) |
| Definition | Azimuth to the vehicle's forward direction | Third ZYX Euler angle |

The relationship (NED, FRD body, north = 0°):

\[
\text{heading} = \pi/2 - \psi_{\text{ZYX}} \pmod{2\pi}
\]

or equivalently \(\psi_{\text{ZYX}} = \pi/2 - \text{heading}\). qnav provides `yaw_to_heading` and `heading_to_yaw` explicitly.

## Disturbance detection

Magnetometers are sensitive to ferromagnetic objects, electrical cables, and motors. A reading corrupted by disturbance will give a wrong heading — potentially worse than no heading update at all.

qnav implements two complementary checks:

**Magnitude gate**: the field magnitude should match the expected local magnitude within tolerance:

\[
|\ \|\mathbf{m}^b\| - \|\mathbf{m}^{\text{nav}}\|\ | < \alpha_{\text{mag}} \|\mathbf{m}^{\text{nav}}\|
\]

**Dip gate**: the dip angle (vertical field fraction) should match the expected value:

\[
\left|\arcsin\!\left(\frac{m_z^b}{\|\mathbf{m}^b\|}\right) - I\right| < \alpha_{\text{dip}}
\]

A reading that passes both gates is trustworthy for a heading update. A reading that fails either should be excluded — do not fuse it, do not set it to zero, simply skip the magnetometer update for this timestep.

```python
from qnav.heading.disturbance import is_field_trustworthy

if is_field_trustworthy(m_body, m_nav_reference, mag_tol=0.1, dip_tol=0.15):
    f.update_magnetometer(M_NAV, m_body, sigma=0.02)
# else: skip the update, let the gyro propagate
```

The `HeadingMonitor` class accumulates statistics on disturbance frequency:

```python
from qnav.heading.disturbance import HeadingMonitor
monitor = HeadingMonitor(m_nav=M_NAV)
for m_body in mag_stream:
    monitor.update(m_body)
print(f"disturbance fraction: {monitor.disturbance_fraction:.1%}")
```

## Heading uncertainty

The heading variance from tilt-compensated compass:

\[
\sigma_\psi^2 \approx \frac{\sigma_m^2}{\|\mathbf{m}^h\|^2 \cos^2 I}
\]

where \(I\) is the magnetic dip angle. Near the magnetic poles (\(I \to 90°\)), the horizontal field component \(\|\mathbf{m}^h\| \cos I \to 0\) and compass heading becomes useless — this is not a calibration problem but a fundamental observability limitation.

*Source: attitudesurvey.tex §7 (tilt compensation); Kok/Hol/Schön §3.6 (magnetic field model, heading). See [`formula_catalog.md`](formula_catalog.md) §7.*
