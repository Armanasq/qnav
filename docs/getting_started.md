# Getting Started

## Installation

```bash
pip install qnav                     # NumPy only
pip install "qnav[test]"             # pytest + hypothesis (property tests)
pip install "qnav[docs]"             # MkDocs Material (build these docs)
pip install "qnav[dev]"              # everything + ruff + mypy
```

Development install from source:

```bash
git clone https://github.com/armanasq/qnav
cd qnav
pip install -e ".[dev]"
pytest -q                            # 248 tests
```

---

## The naming convention

qnav labels every quaternion and DCM with its frame pair: `q_AB` maps coordinates from frame B to frame A:

\[
\mathbf{v}_A = \mathbf{R}(q_{AB})\, \mathbf{v}_B
\]

This is the **passive coordinate transformation** convention. The same matrix, acting on vectors within a single frame, actively rotates them by the rotation that carries A's axes onto B's axes.

Composition chains adjacent indices:

\[
q_{AC} = q_{AB} \otimes q_{BC}
\]

The inverse (conjugate) swaps source and target:

\[
q_{BA} = q_{AB}^* 
\]

These rules are the only two things you need to remember. The code enforces them.

---

## Attitude representations

### Quaternion `[w, x, y, z]`

```python
from qnav.attitude import quaternion as quat
import numpy as np

# Identity (no rotation)
q = quat.identity()                        # [1, 0, 0, 0]

# Rotation from rotation vector (axis × angle)
phi = np.array([0.0, 0.0, np.pi / 2])     # 90° about z
q = quat.exp(phi)                          # [cos(π/4), 0, 0, sin(π/4)]

# Rotate a vector: v_A = R(q_AB) v_B
v_B = np.array([1.0, 0.0, 0.0])
v_A = quat.rotate_vector(q, v_B)          # → [0, 1, 0]

# The inverse direction: v_B = R(q_AB)ᵀ v_A
v_B_back = quat.rotate_frame(q, v_A)      # → [1, 0, 0]

# Compose: q_WL = q_WB ⊗ q_BL
q_WB = quat.exp(np.array([0.1, 0.0, 0.0]))
q_BL = quat.exp(np.array([0.0, 0.2, 0.0]))
q_WL = quat.mul(q_WB, q_BL)

# Angular distance (geodesic, sign-invariant)
d = quat.angular_distance(q_WB, q_WL)     # radians

# Interpolation
q_half = quat.power(quat.relative(q_WB, q_WL), 0.5)
```

### DCM (Direction Cosine Matrix)

```python
from qnav.attitude import dcm as dcm_mod, quaternion as quat

R = dcm_mod.from_quaternion(q)     # shape (3, 3), R ∈ SO(3)
q_back = dcm_mod.to_quaternion(R)  # Shepperd's method, 4-branch stable

# Elementary rotations
Rx = dcm_mod.rot_x(0.1)
Ry = dcm_mod.rot_y(0.2)
Rz = dcm_mod.rot_z(0.3)

# Nearest rotation in Frobenius norm (never applied silently)
from qnav.attitude.so3 import project
R_clean = project(R_noisy)
```

### Euler angles

```python
from qnav.attitude import euler
import numpy as np

# Default: intrinsic ZYX (yaw ψ, pitch θ, roll φ) — aerospace standard
angles = np.array([np.deg2rad(30), np.deg2rad(10), np.deg2rad(5)])  # ψ, θ, φ
R = euler.to_dcm(angles, seq="ZYX")
q = euler.to_quaternion(angles, seq="ZYX")

# Recovery: near gimbal lock issues GimbalLockWarning and sets third angle = 0
import warnings
with warnings.catch_warnings(record=True) as w:
    recovered = euler.from_quaternion(q_near_gimbal, seq="ZYX")

# All 18 sequences (12 Tait-Bryan + 6 proper), intrinsic (uppercase) or extrinsic (lowercase)
R_xyx = euler.to_dcm(angles, seq="XYX")    # proper Euler
R_xyz = euler.to_dcm(angles, seq="xyz")    # extrinsic = intrinsic ZYX reversed
```

### SO(3) Lie group

```python
from qnav.attitude import so3

# hat: ℝ³ → so(3)  |  vee: so(3) → ℝ³
K = so3.hat(np.array([1.0, 0.0, 0.0]))   # 3×3 skew
v = so3.vee(K)                            # back to [1, 0, 0]

# Exponential map (Rodrigues rotation formula)
phi = np.array([0.1, -0.2, 0.3])
R = so3.exp(phi)                          # exact, series below 1e-4 rad

# Logarithm (3-branch: small-angle, generic, near-π)
phi_back = so3.log(R)

# Right Jacobian: Exp(θ+δ) ≈ Exp(θ) · Exp(Jr(θ) δ)
Jr = so3.right_jacobian(phi)
Jl = so3.left_jacobian(phi)               # = Jr(-phi) = Jr(phi)ᵀ

# Manifold operations
R2 = so3.boxplus(R, np.array([0.01, 0.0, 0.0]))  # R ⊞ δ = R·Exp(δ)
delta = so3.boxminus(R2, R)                        # Log(Rᵀ R2)
```

---

## Coordinate frames

### Using typed transforms

```python
from qnav.frames import Frame, FrameTransform
from qnav.attitude import quaternion as quat
import numpy as np

BODY  = Frame("BODY")
IMU   = Frame("IMU")

# A static sensor-to-body transform: 5° misalignment about z
q_body_imu = quat.exp(np.array([0.0, 0.0, np.deg2rad(5)]))
t_body_imu = np.array([0.1, 0.0, -0.05])  # lever arm [m]

T = FrameTransform(q=q_body_imu, t=t_body_imu, from_frame=IMU, to_frame=BODY)

# Transform a velocity measured in IMU frame
v_imu = np.array([1.0, 0.0, 0.0])
v_body = T.apply_vector(v_imu)   # raises FrameMismatchError if frame is wrong

# Inverse
T_inv = T.inverse()              # from_frame=BODY, to_frame=IMU
```

### Earth frames and geodesy

```python
from qnav.frames.earth import (
    geodetic_to_ecef, ecef_to_geodetic,
    dcm_ecef_to_ned, normal_gravity, earth_rate_ned,
)
import numpy as np

lat = np.deg2rad(51.5)       # London
lon = np.deg2rad(-0.1)
h   = 11.0                   # altitude [m]

r_ecef = geodetic_to_ecef(lat, lon, h)
R_ned_ecef = dcm_ecef_to_ned(lat, lon)  # rows are NED unit vectors in ECEF
g = normal_gravity(lat, h)              # Somigliana + free-air correction [m/s²]
omega_ie_ned = earth_rate_ned(lat)      # Earth rate in NED [rad/s]

# Round-trip
lat2, lon2, h2 = ecef_to_geodetic(r_ecef)  # Bowring iteration, sub-mm accuracy
```

---

## Sensor models and calibration

### Gyro bias from static data

```python
from qnav.calibration.gyro_bias import detect_static_intervals, estimate_bias

# gyro: (N, 3), accel: (N, 3), dt: sample period
static_mask = detect_static_intervals(gyro, accel, dt=0.01)
bias, sigma = estimate_bias(gyro, static_mask)
print(f"bias: {np.rad2deg(bias * 3600)} deg/hr ± {np.rad2deg(sigma * 3600)} deg/hr")
```

Static detection uses three gates: per-axis gyro std < threshold, accelerometer-magnitude std < threshold, and mean ‖ω‖ < magnitude threshold. The magnitude gate is what makes the method robust to constant-rate spins (which pass the variance gate but are not static).

### Magnetometer ellipsoid calibration

```python
from qnav.calibration.mag_ellipsoid import fit_ellipsoid, MagCalibration

# Collect raw_data while rotating the device through all orientations
# Shape: (N, 3), no particular order required
ellipsoid = fit_ellipsoid(raw_data)    # raises CalibrationError if SPD check fails
cal = MagCalibration(ellipsoid)

# Apply to new readings
m_corrected = cal.correct(m_raw)
```

The fit uses SVD on the quadric form `‖A m + b‖² = 1`, validates positive-definiteness of A (an improper ellipsoid fit indicates bad data), and returns the Cholesky correction map that centers and spheres the data.

### Allan variance

```python
from qnav.sensors.allan import overlapping_allan_variance, identify_noise

taus, adev = overlapping_allan_variance(gyro_data[:, 0], dt=1/200)
params = identify_noise(taus, adev)
print(f"ARW: {params['noise_density']:.4f} rad/s/√Hz")
print(f"bias instability: {params['bias_instability']:.6f} rad/s")
```

---

## Running an ESKF

The ESKF is a 6-state filter over the error state `[δθ, δb] ∈ ℝ⁶`. The nominal state tracks the quaternion and gyro bias separately; the error covariance P is 6×6 over the right/local tangent perturbation `q_true = q̂ ⊗ Exp(δθ)`.

```python
from qnav.filters import Eskf
from qnav.heading.magnetic_model import field_from_elements
import numpy as np

# Build the reference field: dip 60°, declination 0°, unit magnitude
M_NAV = field_from_elements(declination=0.0, inclination=np.deg2rad(60.0), magnitude=1.0)

f = Eskf(
    gyro_noise_density=0.005,   # rad/s/√Hz
    gyro_bias_walk=1e-5,        # rad/s²/√Hz
    nav_frame="NED",
)

for k in range(n_steps):
    f.predict(gyro[k], dt)
    if k % 5 == 0:                   # fuse aiding sensors at 20 Hz
        f.update_gravity(accel[k], sigma=0.02)
        f.update_magnetometer(M_NAV, mag[k], sigma=0.02)

# Extract results
q_nav_body = f.q
bias_estimate = f.bias               # [rad/s], body frame
attitude_1sigma = f.attitude_std     # [rad], per-axis

# Covariance diagnostics
from qnav.metrics import nees_bounds
lo, hi = nees_bounds(dim=3, n_samples=100)
print(f"NEES 95% bounds: [{lo:.2f}, {hi:.2f}]")
```

**Measurement updates** call `update_direction(v_nav, v_body, sigma)` for any unit-direction observation. `update_gravity` and `update_magnetometer` are thin wrappers that normalize their input and call `update_direction`.

**Disturbance gating** is not automatic — gate magnetometer updates upstream using `qnav.heading.disturbance.is_field_trustworthy` before calling the update. This is intentional: the filter doesn't silently ignore measurements.

---

## Testing and validation

```bash
pytest                             # full suite (~45 s)
pytest tests/test_so3.py -v        # detailed SO(3) output
pytest -k "convergence"            # filter convergence tests only
pytest --tb=short -q               # compact failure output
```

The `qnav.validation` module exposes the mathematical invariants and reference cases used internally:

```python
from qnav.validation.invariants import check_all
from qnav.attitude import quaternion as quat

q = quat.random((), rng=np.random.default_rng(0))
results = check_all(q)     # returns dict of {invariant_name: pass/fail}
```
