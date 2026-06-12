# Euler Angles

Euler angles are the most human-readable attitude representation and the most error-prone. Their problems — gimbal lock, convention proliferation, and discontinuities — are all manageable as long as the convention is explicit and the singular points are handled deterministically.

qnav supports all 18 valid Euler sequences (12 Tait-Bryan + 6 proper Euler), with explicit convention strings and documented gimbal-lock behavior.

## Sequence grammar

Every Euler API requires a `seq` string:

- **Uppercase** (`"ZYX"`, `"XYZ"`, `"ZXZ"`, …): **intrinsic** rotations about the *moving* frame's axes, applied left-to-right.
- **Lowercase** (`"zyx"`, `"xyz"`, `"zxz"`, …): **extrinsic** rotations about the *fixed* frame's axes.

The intrinsic–extrinsic duality: intrinsic `"ABC"` with angles \((\alpha, \beta, \gamma)\) equals extrinsic `"cba"` with angles \((\gamma, \beta, \alpha)\). The test suite enforces this identity for all 18 sequences.

Adjacent letters must differ. `"XXY"` is not a valid sequence — the middle rotation would be about the same axis as the first, collapsing two degrees of freedom.

## Default: intrinsic ZYX (aerospace yaw-pitch-roll)

The default `"ZYX"` sequence gives:

\[
\mathbf{R}_{AB} = R_z(\psi) \cdot R_y(\theta) \cdot R_x(\phi)
\]

Reading right-to-left: first roll \(\phi\) about body x, then pitch \(\theta\) about the new y, then yaw \(\psi\) about the new z. The result is the body-to-nav DCM.

Expanded:

\[
\mathbf{R}_{nb} = \begin{bmatrix}
c\theta c\psi & s\phi s\theta c\psi - c\phi s\psi & c\phi s\theta c\psi + s\phi s\psi \\
c\theta s\psi & s\phi s\theta s\psi + c\phi c\psi & c\phi s\theta s\psi - s\phi c\psi \\
-s\theta & s\phi c\theta & c\phi c\theta
\end{bmatrix}
\]

Recovery from DCM (no gimbal lock):

\[
\phi = \text{atan2}(R_{32}, R_{33}), \quad
\theta = \text{atan2}(-R_{31}, \sqrt{R_{32}^2 + R_{33}^2}), \quad
\psi = \text{atan2}(R_{21}, R_{11})
\]

## Tait-Bryan vs proper Euler

**Tait-Bryan sequences** (also called Cardan angles) use three distinct axes, e.g. `"ZYX"`, `"XYZ"`, `"ZXY"`. There are 12 of them. They are the practical choice for most engineering applications: roll/pitch/yaw in aerospace and robotics, pan/tilt/roll in camera systems.

**Proper Euler sequences** repeat the first axis, e.g. `"ZXZ"`, `"ZYZ"`, `"XYX"`. There are 6 of them. They are used in classical mechanics (precession/nutation/spin for axisymmetric bodies) and in some crystallography conventions.

The distinction matters for gimbal-lock identification: Tait-Bryan sequences lock at the **middle angle** = ±π/2, while proper sequences lock at the middle angle = 0 or π.

## Gimbal lock

Gimbal lock is not a numerical bug — it is a topological fact. The three-angle parametrization of SO(3) must have singular points (Euler's theorem on \(S^3\) coverings). What you can control is what happens at those points.

qnav's contract: when the middle angle is within `gimbal_tol` of the singular value:

1. Issue a `GimbalLockWarning`.
2. Set the **third angle to zero** (assign the lost degree of freedom entirely to the first angle).

The geometric construction: with \(c = 0\), the matrix \(\mathbf{R}_A(a) \mathbf{R}_B(b)\) must equal the measured \(\mathbf{R}\). The first-angle matrix is:

\[
\mathbf{R}_A(a) = \mathbf{R} \cdot \mathbf{R}_B(b)^\top
\]

Reading the angle from this matrix (using the principal axes of A) gives a value that is correct regardless of the Euler sequence. This generic construction replaces the sequence-specific formulas that were incorrect for 14 of the 18 sequences in earlier implementations.

## The intrinsic–extrinsic duality in code

```python
from qnav.attitude import euler
import numpy as np

angles = np.array([0.3, 0.1, -0.2])

# Intrinsic ZYX and extrinsic xyz applied with reversed angles give the same DCM
R_intrinsic = euler.to_dcm(angles, seq="ZYX")
R_extrinsic = euler.to_dcm(angles[::-1], seq="xyz")
assert np.allclose(R_intrinsic, R_extrinsic)

# Recovery is also consistent
angles2 = euler.from_dcm(R_intrinsic, seq="ZYX")
angles3 = euler.from_dcm(R_extrinsic, seq="xyz")
# angles2 == angles, angles3 == angles[::-1]
```

## When not to use Euler angles

Euler angles should not be:

1. **Stored as state in a filter.** The Euler-rate kinematic equation \(\dot{\boldsymbol{\eta}} = \mathbf{W}(\boldsymbol{\eta})\boldsymbol{\omega}\) has a singular Jacobian at gimbal lock. The ESKF uses the rotation-vector error \(\delta\boldsymbol{\theta}\) for exactly this reason.

2. **Used for interpolation.** Linear interpolation of Euler angles does not produce the geodesic path on SO(3) — it produces an arbitrary curved path whose total rotation depends on the sequence choice.

3. **Used for attitude averaging.** The mean of Euler angles is not the mean rotation.

Use quaternions (via `quat.mean`, `quat.power`, SLERP) for all these operations. Convert to Euler angles only at display boundaries.

*Source: Hashim, Paper_Hashim_SUBMIT.tex §4; Kok/Hol/Schön §3.4; attitudesurvey.tex §Euler. See [`formula_catalog.md`](formula_catalog.md) §2.6.*
