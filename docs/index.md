# qnav

**Convention-safe attitude, heading, frame-transform, and navigation math.**

qnav is a precision navigation math library for robotics, aerospace, and sensor-fusion applications. Its central design decision is that convention ambiguity — the silent bug that has plagued navigation software for decades — is treated as a correctness property, not a style preference.

---

## The convention problem

Attitude math has no universally agreed standard. Hamilton vs JPL quaternions differ by conjugation. Active vs passive DCMs transpose each other. ZYX and XYZ Euler angles produce different rotations from the same numbers. In production code, these silent mismatches produce errors that look like sensor noise, miscalibration, or model mismatch — they are extremely difficult to isolate.

qnav eliminates this class of bug through architecture:

- Every rotation is labeled `q_AB`: source frame B, target frame A.
- Frame transforms carry their source/target as data and are checked at composition.
- JPL quaternions, scalar-last layout, and extrinsic Euler conventions are available through explicitly-named functions only — you cannot accidentally use them.
- Failure modes are typed (`FrameMismatchError`, `GimbalLockWarning`) and deterministic.

---

## Structure

<div class="grid cards" markdown>

-   **Attitude math**

    SO(3) as a Lie group: quaternion algebra, DCM, rotation vector, MRP, Rodrigues parameters. All representations, all conversions, all Jacobians. Kinematics and integrators with documented order-of-accuracy.

    [:octicons-arrow-right-24: Quaternions](math/quaternions.md) · [:octicons-arrow-right-24: SO(3)](math/so3.md)

-   **Coordinate frames**

    Typed transforms with runtime frame-consistency checks. WGS-84 geodesy (Somigliana gravity, Bowring ECEF conversion). NED/ENU/ECEF/ECI plus FRD/FLU, ROS REP-103, aerospace stability and wind frames. Frame graph with BFS path finding.

    [:octicons-arrow-right-24: Frames](math/frames.md)

-   **Attitude determination**

    Wahba problem (1965) formulation, TRIAD, Davenport q-method, QUEST, SVD, OLEQ, FLAE (quartic characteristic polynomial), plus closed-form accelerometer/magnetometer solvers — SAAM, FAMC, FQA — that need no eigendecomposition and no a-priori dip angle.

    [:octicons-arrow-right-24: Determination](math/attitude_determination.md)

-   **Filtering**

    Ten estimators spanning the complexity spectrum: complementary, Mahony, Madgwick, AQUA (decoupled tilt/yaw corrections), Fourati (LM observer), ROLEQ, Fast KF, quaternion EKF, the error-state Kalman filter (NEES-verified covariance), and an unscented filter on the SO(3) tangent for large-uncertainty regimes.

    [:octicons-arrow-right-24: Filters](math/filtering.md)

-   **Sensors and calibration**

    IMU noise models (noise density, bias random walk, Gauss-Markov), Allan variance, gyro bias from static intervals, magnetometer ellipsoid calibration (hard/soft-iron), accelerometer 6-position calibration, frame alignment from vector pairs.

    [:octicons-arrow-right-24: Sensors](api/sensors.md)

-   **Heading and tilt**

    Roll/pitch from accelerometer, tilt-compensated compass, disturbance detection (magnitude + dip gates), declination correction, and a full World Magnetic Model (WMM2025) — spherical-harmonic synthesis validated against the official test values to < 0.1 nT.

    [:octicons-arrow-right-24: Heading](math/heading.md)

</div>

---

## Quick start

```bash
pip install qnav
```

```python
import numpy as np
from qnav.attitude import quaternion as quat
from qnav.filters import Eskf

# A 6-state ESKF: attitude + gyro bias, error covariance over the right/local tangent
f = Eskf(gyro_noise_density=0.005, gyro_bias_walk=1e-5, nav_frame="NED")

for gyro_meas, accel_meas, mag_meas in sensor_stream:
    f.predict(gyro_meas, dt=0.01)
    f.update_gravity(accel_meas, sigma=0.02)       # low-dynamics assumption
    f.update_magnetometer(m_nav, mag_meas, sigma=0.02)

# attitude_std is sqrt of the diagonal of the 3×3 attitude block of P
print(f"attitude uncertainty (1σ): {np.rad2deg(f.attitude_std)} deg")
```

---

## Design philosophy

**Everything explicit.** There are no global configuration flags. No `set_convention("NED")`. The convention is part of the function call or the object constructor.

**Fail loudly.** Composing transforms with mismatched frames raises `FrameMismatchError`. Approaching gimbal lock issues `GimbalLockWarning`. Zero-norm measurements raise `ValueError`. Silent degradation is treated as a defect.

**Mathematically traceable.** Every non-trivial formula cites a source from the [`formula catalog`](math/formula_catalog.md). Discrepancies between sources are documented and resolved explicitly. The code is the reference.

**Vectorized.** All functions accept arbitrary leading batch dimensions. A quaternion has shape `(..., 4)`, a DCM `(..., 3, 3)`, a vector `(..., 3)`. No scalar-path special cases.

**NumPy only.** The core library has one runtime dependency. SciPy is not required — the NEES χ² bounds use the Wilson-Hilferty normal approximation instead of `scipy.stats.chi2`.
