# Coordinate Frames

Every navigation computation involves at least two coordinate frames. Mistakes in frame bookkeeping — applying a NED rotation to ENU data, composing transforms in the wrong order, forgetting a lever-arm offset — are endemic in navigation code and notoriously hard to debug.

qnav addresses this with **typed frame transforms**: objects that carry their source and target frame as data, check consistency at composition, and make implicit frame assumptions into explicit runtime errors.

## Frame definitions

| Frame | Origin | x-axis | y-axis | z-axis | Pairs with |
|---|---|---|---|---|---|
| **ECI** | Earth center | Vernal equinox | Completes RH | North pole | — |
| **ECEF** | Earth center | 0°N, 0°E | 90°E equator | North pole | — |
| **NED** | Local surface | North | East | Down | FRD |
| **ENU** | Local surface | East | North | Up | FLU |
| **FRD** | Vehicle CG | Forward | Right | Down | NED |
| **FLU** | Vehicle CG | Forward | Left | Up | ENU |

All frames are right-handed. Left-handed frames are not representable — constructing one raises `ConventionError`.

**NED vs ENU**: both are first-class in qnav. There is no default. Every function that depends on the navigation frame takes an explicit `frame` argument. The conversion between them is an involutory permutation:

\[
\mathbf{R}_{\text{ENU}\leftarrow\text{NED}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & -1 \end{bmatrix}
\]

This matrix is its own inverse: \(\mathbf{R}_{\text{NED}\leftarrow\text{ENU}} = \mathbf{R}_{\text{ENU}\leftarrow\text{NED}}\).

## FrameTransform

A `FrameTransform` encodes the rigid-body transform between two named frames:

\[
\mathbf{p}_A = \mathbf{R}_{AB}\, \mathbf{p}_B + \mathbf{t}_{AB}
\]

where \(\mathbf{t}_{AB}\) is the origin of frame B expressed in frame A (the lever arm).

```python
from qnav.frames import Frame, FrameTransform

T = FrameTransform(q=q_A_from_B, t=t_A_from_B, from_frame=B, to_frame=A)

# Vector (no translation) vs point (with translation)
v_A = T.apply_vector(v_B)
p_A = T.apply_point(p_B)

# Covariance propagation (rotation only, linearized)
cov_A = T.apply_covariance(cov_B)

# Inverse: swaps from/to frames
T_B_from_A = T.inverse()

# Composition: raises FrameMismatchError if the chain doesn't close
T_A_from_C = T.compose(T_B_from_C)   # T_B_from_C.to_frame must equal T.from_frame
```

**Inverse lever-arm**: when inverting a transform, the translation must be rotated and negated:

\[
\mathbf{t}_{BA} = -\mathbf{R}_{BA}\, \mathbf{t}_{AB}
\]

The Jacobian of the inverse (useful for covariance propagation) is:

\[
\frac{\partial(\mathbf{p}_B)}{\partial(\mathbf{p}_A)} = \begin{bmatrix} -\mathbf{R}_{AB} & \mathbf{0} \\ [\mathbf{t}_{BA}]_\times & -\mathbf{R}_{BA} \end{bmatrix}
\]

## FrameGraph

A `FrameGraph` stores a collection of `FrameTransform` objects and finds transforms between frames using BFS. It raises on ambiguous paths (two routes that produce different transforms indicate inconsistent calibration data):

```python
from qnav.frames.graph import FrameGraph

g = FrameGraph()
g.add(T_body_imu)
g.add(T_body_camera)

# Find T_camera_imu automatically via T_body_camera⁻¹ ∘ T_body_imu
T_camera_imu = g.find(from_frame=IMU, to_frame=CAMERA)
```

## WGS-84 geodesy

qnav uses the WGS-84 reference ellipsoid throughout:

| Parameter | Value |
|---|---|
| Semi-major axis \(a\) | 6 378 137.0 m |
| Flattening \(f\) | 1 / 298.257 223 563 |
| Gravitational constant \(GM\) | 3.986 004 418 × 10¹⁴ m³/s² |
| Earth rotation rate \(\Omega\) | 7.292 115 × 10⁻⁵ rad/s |

### Geodetic ↔ ECEF

Geodetic to ECEF is closed-form:

\[
N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}
\]

\[
\begin{bmatrix} x \\ y \\ z \end{bmatrix}
= \begin{bmatrix} (N+h)\cos\phi\cos\lambda \\ (N+h)\cos\phi\sin\lambda \\ (N(1-e^2)+h)\sin\phi \end{bmatrix}
\]

ECEF to geodetic uses Bowring's iterative method (fixed-point iteration on the latitude equation). It converges to sub-millimeter accuracy in ≤ 5 iterations for all terrestrial points.

### NED/ENU from ECEF

The local-level frame is defined by the geodetic latitude and longitude:

\[
\mathbf{R}_{\text{NED}\leftarrow\text{ECEF}}(\phi, \lambda) = \begin{bmatrix}
-\sin\phi\cos\lambda & -\sin\phi\sin\lambda & \cos\phi \\
-\sin\lambda & \cos\lambda & 0 \\
-\cos\phi\cos\lambda & -\cos\phi\sin\lambda & -\sin\phi
\end{bmatrix}
\]

The rows are the unit vectors pointing North, East, and Down, expressed in ECEF coordinates.

### Normal gravity (Somigliana)

\[
\gamma(\phi) = \gamma_e \frac{1 + k\sin^2\phi}{\sqrt{1 - e^2\sin^2\phi}}
\]

with \(\gamma_e = 9.780\,325\,336\) m/s² (equatorial), \(k = 0.001\,931\,852\,65\). The free-air height correction:

\[
\gamma(h) = \gamma(\phi)\left(1 - \frac{2h}{a}\bigl(1 + f + m - 2f\sin^2\phi\bigr) + \frac{3h^2}{a^2}\right)
\]

where \(m = \Omega^2 a^2 b / (GM)\). This gives the standard atmosphere gravity to parts-per-million accuracy for altitudes up to ~20 km.

## Earth rotation rate

The Earth's rotation vector expressed in NED:

\[
\boldsymbol{\omega}_{ie}^n = \Omega \begin{bmatrix} \cos\phi \\ 0 \\ -\sin\phi \end{bmatrix}
\]

At mid-latitudes, \(\|\boldsymbol{\omega}_{ie}\| \approx 7.3 \times 10^{-5}\) rad/s \(\approx 15^\circ/\text{hr}\). For MEMS gyroscopes with noise density > 0.01 deg/s/√Hz, this is below the noise floor and can be ignored. For tactical-grade and higher, it must be included in the process model.

## Vehicle and aerospace frames

qnav provides standard transforms for aerospace and robotics conventions:

```python
from qnav.frames.conventions import DCM_FLU_FRD, ned_to_enu_attitude
from qnav.frames.aerospace import dcm_body_to_stability, dcm_body_to_wind
from qnav.frames.robotics import standard_graph, quaternion_to_ros

# FRD → FLU (NED → ENU body pair)
R_flu_frd = DCM_FLU_FRD   # diag(1, -1, -1)

# Stability frame: aligned with velocity, z-down (no sideslip, small AoA)
alpha = np.deg2rad(5.0)   # angle of attack
R_stab = dcm_body_to_stability(alpha)

# Wind frame: aligned with velocity direction
alpha, beta = np.deg2rad(5.0), np.deg2rad(2.0)
R_wind = dcm_body_to_wind(alpha, beta)

# ROS REP-103: FLU body in ENU world
graph = standard_graph()   # complete transform graph for ROS navigation stack

# Convert qnav quaternion to ROS geometry_msgs/Quaternion (scalar-last)
q_ros = quaternion_to_ros(q_nav)
```

*Source: NIMA TR8350.2 (WGS-84); Kok/Hol/Schön §2 (navigation frames); standard aerospace/ROS conventions.*
