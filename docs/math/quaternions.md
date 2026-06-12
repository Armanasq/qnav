# Quaternion Algebra

Quaternions are the primary rotation representation in qnav. Every other representation — DCM, rotation vector, Euler angles — is a derived form that routes through quaternions for conversions.

## The Hamilton convention

qnav uses **Hamilton quaternions** exclusively. The defining relation is \(ij = k\), which makes the quaternion product right-handed: two successive rotations compose in the same order as their physical application.

The alternative — the JPL convention where \(ji = k\) — is used by NASA's SPICE library and some aerospace codebases. It represents the same physical rotation but with the conjugate quaternion. qnav provides `from_jpl` and `to_jpl` as explicit bridge functions. There is no global setting.

## Layout: scalar-first `[w, x, y, z]`

\[
\mathbf{q} = \begin{bmatrix} w \\ x \\ y \\ z \end{bmatrix}
= \begin{bmatrix} q_w \\ \mathbf{q}_v \end{bmatrix}, \quad
w = \cos\tfrac{\theta}{2}, \quad \mathbf{q}_v = \mathbf{u}\sin\tfrac{\theta}{2}
\]

where \(\theta\) is the rotation angle and \(\mathbf{u}\) is the unit rotation axis. Memory layout `[w, x, y, z]` means the scalar is index 0. SciPy and ROS use the opposite layout `[x, y, z, w]` — convert with `from_scalar_last` / `to_scalar_last`.

## Product (Hamilton product)

\[
\mathbf{p} \otimes \mathbf{q}
= \begin{bmatrix}
p_w q_w - \mathbf{p}_v^\top \mathbf{q}_v \\
p_w \mathbf{q}_v + q_w \mathbf{p}_v + \mathbf{p}_v \times \mathbf{q}_v
\end{bmatrix}
\]

The product is associative and non-commutative. Applied to a composition `q_AC = q_AB ⊗ q_BC`, the right factor `q_BC` is applied first (B→C), then `q_AB` (A→B). This mirrors matrix multiplication and matches the left-to-right reading of the composition chain.

Matrix forms useful for linear algebra:

\[
\mathbf{q}_1 \otimes \mathbf{q}_2 = [\mathbf{q}_1]_L \, \mathbf{q}_2 = [\mathbf{q}_2]_R \, \mathbf{q}_1
\]

\[
[\mathbf{q}]_L = q_w \mathbf{I}_4 + \begin{bmatrix} 0 & -\mathbf{q}_v^\top \\ \mathbf{q}_v & [\mathbf{q}_v]_\times \end{bmatrix}, \quad
[\mathbf{q}]_R = q_w \mathbf{I}_4 + \begin{bmatrix} 0 & -\mathbf{q}_v^\top \\ \mathbf{q}_v & -[\mathbf{q}_v]_\times \end{bmatrix}
\]

## Rotation semantics: passive coordinate transformation

A unit quaternion `q_AB` maps the coordinates of a fixed vector from frame B to frame A:

\[
\mathbf{v}_A = \mathbf{R}(q_{AB})\, \mathbf{v}_B
\]

This is the **passive** or **alias** interpretation. The physical vector is unchanged; only its coordinate expression changes. The same matrix, acting within a single fixed frame, **actively** rotates vectors by the rotation that carries A's axes onto B's axes.

In code:

```python
v_A = quat.rotate_vector(q_AB, v_B)   # v_A = R(q_AB) v_B
v_B = quat.rotate_frame(q_AB, v_A)    # v_B = R(q_AB)ᵀ v_A = R(q_BA) v_A
```

## Rotation matrix from quaternion

\[
\mathbf{R}(\mathbf{q}) = (q_w^2 - \mathbf{q}_v^\top \mathbf{q}_v)\,\mathbf{I}_3
+ 2\,\mathbf{q}_v \mathbf{q}_v^\top + 2\,q_w [\mathbf{q}_v]_\times
\]

Expanded:

\[
\mathbf{R} = \begin{bmatrix}
1{-}2(y^2{+}z^2) & 2(xy{-}wz) & 2(xz{+}wy) \\
2(xy{+}wz) & 1{-}2(x^2{+}z^2) & 2(yz{-}wx) \\
2(xz{-}wy) & 2(yz{+}wx) & 1{-}2(x^2{+}y^2)
\end{bmatrix}
\]

This form (using the homogeneous relation \(w^2 + x^2 + y^2 + z^2 = 1\)) avoids any divisions. All five primary references in the qnav corpus agree on this exact matrix under the Hamilton, body-to-world reading.

## Exponential and logarithm

The exponential map takes a rotation vector \(\boldsymbol{\phi} \in \mathbb{R}^3\) to a unit quaternion:

\[
\text{Exp}(\boldsymbol{\phi}) = \begin{bmatrix} \cos(\theta/2) \\ \mathbf{u}\sin(\theta/2) \end{bmatrix}, \quad \theta = \|\boldsymbol{\phi}\|, \quad \mathbf{u} = \boldsymbol{\phi}/\theta
\]

At \(\theta \to 0\), the ratio \(\sin(\theta/2)/\theta\) is replaced by its Taylor series \(\tfrac{1}{2} - \theta^2/48 + \cdots\) (threshold: \(10^{-8}\) rad). This eliminates the 0/0 form without introducing a branch discontinuity.

The logarithm is the inverse, returning the rotation vector:

\[
\text{Log}(\mathbf{q}) = 2\,\text{atan2}(\|\mathbf{q}_v\|, q_w)\, \frac{\mathbf{q}_v}{\|\mathbf{q}_v\|}
\]

Using `atan2` rather than `arccos` is deliberate: `atan2` is numerically stable for all angles including \(\theta \to 0\) and \(\theta \to \pi\), while `arccos(q_w)` loses precision near both endpoints.

```python
# Round-trip: Log(Exp(phi)) = phi for phi in (-pi, pi)
phi = np.array([0.3, -0.1, 0.7])
assert np.allclose(quat.log(quat.exp(phi)), phi)
```

## The double cover

The same physical rotation is encoded by both \(\mathbf{q}\) and \(-\mathbf{q}\):

\[
\mathbf{R}(-\mathbf{q}) = \mathbf{R}(\mathbf{q})
\]

This is not a bug or a normalization choice — it is a fundamental property of the 2:1 homomorphism \(\text{SU}(2) \to \text{SO}(3)\). Every distance metric, loss function, and interpolation in qnav must be sign-invariant. The canonical form (flip so \(w \geq 0\)) is available as `quat.canonical(q)` and is **opt-in**, never applied automatically.

The geodesic distance is:

\[
d(\mathbf{q}_1, \mathbf{q}_2) = 2\arccos(|\mathbf{q}_1^\top \mathbf{q}_2|)
\]

The absolute value is the sign-invariant correction. Without it, \(d(\mathbf{q}, -\mathbf{q}) = \pi\) instead of 0, and you would track a discontinuity as a large rotation.

## SLERP

Spherical linear interpolation traces the geodesic on SO(3):

\[
\text{slerp}(\mathbf{q}_0, \mathbf{q}_1, t) = \mathbf{q}_0 \otimes (\mathbf{q}_0^* \otimes \mathbf{q}_1)^t
\]

Before computing, if \(\mathbf{q}_0^\top \mathbf{q}_1 < 0\), negate \(\mathbf{q}_1\) to take the shorter arc (double-cover fix). When \(\mathbf{q}_0 \approx \mathbf{q}_1\), SLERP degenerates — qnav falls back to NLERP (normalized linear interpolation) below a threshold, which is accurate to first order.

## Weighted mean

The Markley et al. chordal mean minimizes the weighted sum of squared geodesic distances. It solves the eigenproblem:

\[
\bar{\mathbf{q}} = \arg\max_{\|\mathbf{q}\|=1}\, \mathbf{q}^\top M \mathbf{q}, \quad M = \sum_i w_i \mathbf{q}_i \mathbf{q}_i^\top
\]

The dominant eigenvector of the \(4 \times 4\) symmetric matrix \(M\) is the weighted mean. This is equivalent to finding the rotation closest to all inputs in the Frobenius sense, and it is the statistically optimal mean when rotations are drawn from a concentrated Fisher-Bingham distribution.

```python
qs = quat.random((20,), rng=np.random.default_rng(0))
weights = np.exp(-np.arange(20) * 0.1)   # exponential decay
q_mean = quat.mean(qs, weights)
```

## Bridges to other conventions

| Function | Direction | Note |
|---|---|---|
| `from_scalar_last(q)` | `[x,y,z,w]` → `[w,x,y,z]` | SciPy, ROS |
| `to_scalar_last(q)` | `[w,x,y,z]` → `[x,y,z,w]` | |
| `from_jpl(q)` | JPL scalar-last → Hamilton | conjugate + reorder |
| `to_jpl(q)` | Hamilton → JPL scalar-last | |

The JPL conversion is a conjugate followed by a scalar-last reorder. Both conventions represent the same physical rotation; they differ in the sign convention of the vector part. The explicit naming makes the conversion visible in code review.

## Numerical contract

- All functions accept `(..., 4)` arrays and broadcast over leading dimensions.
- Input is cast to `float64`; inputs are never mutated.
- `normalize()` raises `ValueError` on near-zero norm (threshold \(10^{-12}\)).
- `normalize(q, warn_tol=tol)` issues `NormalizationWarning` when \(|\|\mathbf{q}\| - 1| > \text{tol}\).
- The functions `mul`, `conjugate`, `rotate_vector`, `exp`, `log` do not normalize their inputs — this is caller responsibility, documented per function.

*Source: Solà, Quaternion.tex (primary); cross-checked against Hashim OVERVIEW, Kok/Hol/Schön, Parwana/Kothari. Inter-source discrepancies documented in [`formula_catalog.md`](formula_catalog.md).*
