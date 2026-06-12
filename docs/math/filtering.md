# Attitude Filtering

An attitude filter combines gyroscope integration (high bandwidth, unbounded drift) with aiding sensor observations (bounded error, low bandwidth). qnav provides five filters representing different accuracy/complexity/uncertainty-quantification tradeoffs.

## The complementary structure

All attitude filters exploit the complementary noise characteristics of gyroscopes and aiding sensors:

- **Gyroscope**: white noise with bounded power spectral density, plus a slowly-varying bias. Short-term accurate; long-term drifts.
- **Accelerometer/magnetometer**: white measurement noise plus vibration sensitivity. Long-term accurate for tilt and heading in benign dynamics; corrupted by acceleration maneuvers.

The complementary filter directly implements this in the frequency domain: integrate the gyroscope (high-pass) and use the aiding sensors to correct the low-frequency drift (low-pass). The ESKF implements the same idea with a principled noise model and a covariance that tracks estimation uncertainty.

---

## ComplementaryFilter

The simplest form. Exponential predict + SLERP blend:

```
predict: q ← q ⊗ Exp(ω dt)
update:  v_body_pred = rotate_frame(q, v_nav)
         δq = axis-angle from (v_body_pred × v_body_meas)
         q ← slerp(q, q ⊗ Exp(gain · δq), 1.0)
```

The gain \(\alpha \in (0, 1)\) is the only tuning parameter — it sets the crossover frequency between gyro integration and aiding correction. There is no bias estimation and no uncertainty quantification. Use it for prototyping or when only heading is needed.

---

## MahonyFilter

PI feedback complementary filter. The cross-product error drives a PI controller on the gyro input:

\[
\mathbf{e}^b = \sum_i k_i\, \hat{\mathbf{v}}_i^b \times (R(\mathbf{q})^\top \mathbf{v}_i^{\text{nav}})
\]

\[
\hat{\boldsymbol{\omega}} = \boldsymbol{\omega}_m + k_P\, \mathbf{e}^b + k_I \int \mathbf{e}^b \, dt
\]

\[
\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \hat{\boldsymbol{\omega}}
\]

The integral term \(\int \mathbf{e}^b \, dt\) accumulates bias in the body frame; it behaves as a gyro bias estimate, though it is not a statistical estimate and does not have an associated covariance.

Tuning: \(k_P\) (proportional gain) sets convergence speed; \(k_I\) (integral gain) sets how aggressively the bias integrator corrects for steady-state error. Typical values: \(k_P = 1.0\), \(k_I = 0.1{-}0.5\). Higher \(k_P\) speeds convergence but amplifies aiding-sensor noise.

```python
from qnav.filters import MahonyFilter
import numpy as np

f = MahonyFilter(kp=1.0, ki=0.3)
v_nav = np.stack([up_ned, m_nav])      # shape (2, 3)

for gyro, accel, mag in stream:
    v_body = np.stack([accel, mag])
    f.step(gyro, dt, v_nav=v_nav, v_body=v_body)

print(f.q)      # current attitude
print(f.bias)   # integral bias estimate
```

---

## MadgwickStyleFilter

Gradient descent on the orientation objective. The objective function is the alignment error between predicted and measured reference vectors:

\[
\mathbf{F}(\mathbf{q}) = \frac{1}{2} \sum_i \|\mathbf{R}(\mathbf{q})^\top \mathbf{v}_i^{\text{nav}} - \mathbf{v}_i^{\text{body}}\|^2
\]

The gradient with respect to the quaternion:

\[
\nabla_{\mathbf{q}} F = \sum_i \frac{\partial (\mathbf{q}^* \otimes [0, \mathbf{v}_i^{\text{nav}}] \otimes \mathbf{q})}{\partial \mathbf{q}} (\mathbf{R}(\mathbf{q})^\top \mathbf{v}_i^{\text{nav}} - \mathbf{v}_i^{\text{body}})
\]

The update:

\[
\mathbf{q} \leftarrow \text{normalize}\!\left(\mathbf{q} + \Delta t \left(\frac{1}{2}\mathbf{q} \otimes [0, \boldsymbol{\omega}] - \beta \frac{\nabla_{\mathbf{q}} F}{\|\nabla_{\mathbf{q}} F\|}\right)\right)
\]

The single gain \(\beta\) [rad/s] sets the balance between gyroscope trust and gradient correction. It has units of angular rate because it directly multiplies the normalized gradient direction in the quaternion's tangent space.

The gradient Jacobian in qnav uses the **sandwich product** form of the rotation action (rather than the Rodrigues form), which is necessary for the finite-difference test to pass for off-unit perturbations.

---

## QuaternionEkf (total-state EKF)

A standard EKF with the quaternion as part of the state vector. The predict step integrates kinematics; update steps use direction observations. The covariance is 4×4 (over quaternion components) but the filter projects updates onto the tangent space to maintain unit-norm.

This filter has no bias state. Attitude error will grow without bound in a gyro-only scenario (no aiding), and will plateau in the presence of aiding — but the plateau is determined by the aiding noise, not a bias estimate. Use it when you need uncertainty quantification without the complexity of an error-state formulation.

---

## Eskf (Error-State Kalman Filter)

The production filter. State: nominal attitude \(\mathbf{q}\) (unit quaternion) + gyro bias \(\mathbf{b}\) [rad/s]. Error state: \(\delta\mathbf{x} = [\delta\boldsymbol{\theta}, \delta\mathbf{b}]^\top \in \mathbb{R}^6\) with covariance \(\mathbf{P} \in \mathbb{R}^{6\times6}\).

### Error definition

The **right/local** error convention:

\[
\mathbf{q}_{\text{true}} = \hat{\mathbf{q}} \otimes \text{Exp}(\delta\boldsymbol{\theta})
\]

\(\delta\boldsymbol{\theta}\) lives in the body frame at the current estimate \(\hat{\mathbf{q}}\). This choice is natural for gyroscope measurements (which are body-frame rates) and produces the simplest measurement Jacobians.

The alternative **left/global** error \(\mathbf{q}_{\text{true}} = \text{Exp}(\delta\boldsymbol{\theta}) \otimes \hat{\mathbf{q}}\) puts \(\delta\boldsymbol{\theta}\) in the world frame — it simplifies some navigation-frame expressions at the cost of a more complex gyro process model.

### Prediction

Given gyro measurement \(\boldsymbol{\omega}_m\), gyro bias estimate \(\hat{\mathbf{b}}\), and step \(\Delta t\):

**Nominal state**:

\[
\hat{\mathbf{q}} \leftarrow \hat{\mathbf{q}} \otimes \text{Exp}((\boldsymbol{\omega}_m - \hat{\mathbf{b}})\Delta t)
\]

**Error state transition** (discrete, Euler approximation):

\[
\mathbf{F} = \begin{bmatrix}
\text{Exp}(\hat{\boldsymbol{\omega}} \Delta t)^\top & -\mathbf{J}_r(\hat{\boldsymbol{\omega}} \Delta t) \cdot \Delta t \\
\mathbf{0} & \mathbf{I}
\end{bmatrix}
\]

where \(\hat{\boldsymbol{\omega}} = \boldsymbol{\omega}_m - \hat{\mathbf{b}}\).

**Discrete noise covariance**:

\[
\mathbf{Q}_d = \text{diag}(\sigma_g^2 \Delta t \cdot \mathbf{I}_3,\; \sigma_{bw}^2 \Delta t \cdot \mathbf{I}_3)
\]

**Covariance propagation**:

\[
\mathbf{P} \leftarrow \mathbf{F} \mathbf{P} \mathbf{F}^\top + \mathbf{Q}_d
\]

The \(\mathbf{J}_r\) term in \(\mathbf{F}\) is the key difference from a naive Euler approximation: it accounts for the non-commutativity of the rotation manifold in the cross-term between attitude and bias errors.

### Measurement update

For a unit-direction observation in the navigation frame \(\mathbf{v}^{\text{nav}}\) and body-frame measurement \(\mathbf{v}^{\text{body}}\):

**Predicted body measurement**: \(\hat{\mathbf{v}}_b = \mathbf{R}(\hat{\mathbf{q}})^\top \mathbf{v}^{\text{nav}}\)

**Observation Jacobian**:

\[
\mathbf{H} = \begin{bmatrix} [\hat{\mathbf{v}}_b]_\times & \mathbf{0}_{3\times3} \end{bmatrix} \in \mathbb{R}^{3 \times 6}
\]

The skew-symmetric form \([\hat{\mathbf{v}}_b]_\times\) arises from the first-order linearization of the rotation action around the error state \(\delta\boldsymbol{\theta}\).

**Joseph-form update** (numerically stable):

\[
\mathbf{K} = \mathbf{P}\mathbf{H}^\top(\mathbf{H}\mathbf{P}\mathbf{H}^\top + \mathbf{R})^{-1}
\]

\[
\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}(\mathbf{I} - \mathbf{K}\mathbf{H})^\top + \mathbf{K}\mathbf{R}\mathbf{K}^\top
\]

The Joseph form maintains positive-definiteness of \(\mathbf{P}\) even with numerical errors, at twice the computational cost of the simple form \((\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}\).

### Injection and reset

After computing the error-state correction \(\delta\hat{\mathbf{x}} = [\delta\hat{\boldsymbol{\theta}}, \delta\hat{\mathbf{b}}]\):

\[
\hat{\mathbf{q}} \leftarrow \hat{\mathbf{q}} \otimes \text{Exp}(\delta\hat{\boldsymbol{\theta}}), \quad \hat{\mathbf{b}} \leftarrow \hat{\mathbf{b}} + \delta\hat{\mathbf{b}}
\]

Then reset the error-state covariance (since the error state is now zeroed):

\[
\mathbf{P} \leftarrow \mathbf{G}\mathbf{P}\mathbf{G}^\top, \quad \mathbf{G} = \begin{bmatrix} \mathbf{I} - \frac{1}{2}[\delta\hat{\boldsymbol{\theta}}]_\times & \mathbf{0} \\ \mathbf{0} & \mathbf{I} \end{bmatrix}
\]

The reset Jacobian \(\mathbf{G}\) accounts for the nonlinearity of the injection. For small \(\delta\hat{\boldsymbol{\theta}}\), \(\mathbf{G} \approx \mathbf{I}\) is an acceptable approximation.

### Covariance validity: NEES test

The Normalized Estimation Error Squared (NEES) tests whether the filter's covariance is consistent with the actual errors:

\[
\epsilon_k = \delta\boldsymbol{\theta}_k^\top \mathbf{P}_{3\times3,k}^{-1} \delta\boldsymbol{\theta}_k
\]

For a correct filter, \(\epsilon_k\) follows a \(\chi^2(3)\) distribution. Averaged over \(N\) samples, \(\bar{\epsilon}\) should fall in the confidence interval \([r_1, r_2]\) given by `nees_bounds(dim=3, n_samples=N)`.

If \(\bar{\epsilon} > r_2\): the filter is **inconsistent** (underestimates uncertainty). If \(\bar{\epsilon} < r_1\): the filter is **overconfident** (overestimates uncertainty). The test suite verifies the ESKF is consistent on the canonical MARG dataset.

### Noise parameters from datasheets

IMU datasheets give noise specifications in several forms:

| Datasheet spec | qnav parameter | Conversion |
|---|---|---|
| ARW [°/hr/√Hz] or [°/√hr] | `gyro_noise_density` [rad/s/√Hz] | multiply by π/(180 × 60) |
| Bias instability [°/hr] | `gyro_bias_walk` (approx.) | multiply by π/(180 × 3600 × √τ_corr) |
| VRW [m/s/hr/√Hz] | accel noise density | multiply by 1/3600 |

The Allan variance tools in `qnav.sensors.allan` identify these parameters directly from recorded gyro data.

---

## Filter comparison

| | Complementary | Mahony | Madgwick | QuaternionEkf | ESKF |
|---|:---:|:---:|:---:|:---:|:---:|
| Bias estimation | — | Integral | — | — | ✓ statistical |
| Uncertainty (P) | — | — | — | ✓ total-state | ✓ error-state |
| NEES-consistent P | — | — | — | — | ✓ |
| Real-time @ 1 kHz | ✓ | ✓ | ✓ | ✓ | ✓ |
| Parameters | 1 (gain) | 2 (kp, ki) | 1 (β) | 2 | 2 (σg, σbw) |

The ESKF parameter count is deceptively low: the noise densities \(\sigma_g\) and \(\sigma_{bw}\) are physical parameters that can be read from the IMU datasheet or estimated from Allan variance analysis. They are not tuning knobs — they have correct values.

*Source: Solà, ErrorState.tex (ESKF); attitudesurvey.tex (Mahony, Madgwick); Kok/Hol/Schön ch.4 (complementary, EKF smoothing).*
