# Attitude Determination

Attitude determination recovers a rotation from vector observations: pairs of unit vectors measured in two frames. The fundamental formulation is Wahba's problem (1965).

## Wahba's problem

Given \(N\) unit vector pairs \((\mathbf{v}_i^{\text{ref}}, \mathbf{v}_i^{\text{body}})\) with weights \(w_i > 0\), find the rotation \(\mathbf{R} \in \text{SO}(3)\) minimizing:

\[
L(\mathbf{R}) = \frac{1}{2} \sum_{i=1}^N w_i \|\mathbf{v}_i^{\text{ref}} - \mathbf{R}\, \mathbf{v}_i^{\text{body}}\|^2
\]

The convention used throughout qnav: body vectors map to reference vectors via \(\mathbf{v}^{\text{ref}} \approx \mathbf{R}_{AB}\, \mathbf{v}^{\text{body}}\), and all solvers return `q_AB` (the quaternion representing this rotation).

## The attitude profile matrix

The sufficient statistic for all algorithms is the \(3 \times 3\) matrix:

\[
\mathbf{B} = \sum_{i=1}^N w_i \, \mathbf{v}_i^{\text{ref}} \, (\mathbf{v}_i^{\text{body}})^\top
\]

The Wahba loss in terms of \(\mathbf{B}\) is:

\[
L(\mathbf{R}) = \lambda_{\text{total}} - \text{tr}(\mathbf{R}^\top \mathbf{B})
\]

where \(\lambda_{\text{total}} = \sum w_i\) (normalizing weights to sum to 1 sets \(\lambda_{\text{total}} = 1\)). Minimizing \(L\) is equivalent to maximizing \(\text{tr}(\mathbf{R}^\top \mathbf{B})\).

## Observability

A single vector observation determines 2 degrees of freedom (tilt with respect to that vector); rotation about the measurement axis is unobservable. Two non-collinear vectors fully determine attitude. The function `check_observability` tests for this: it warns with `DegenerateGeometryWarning` and returns `False` when all directions are within 1° of collinear.

## TRIAD

TRIAD constructs an orthonormal frame from two vectors without any optimization:

\[
\mathbf{t}_1 = \mathbf{v}_1^{\text{ref}}, \quad \mathbf{t}_2 = \frac{\mathbf{v}_1^{\text{ref}} \times \mathbf{v}_2^{\text{ref}}}{\|\cdot\|}, \quad \mathbf{t}_3 = \mathbf{t}_1 \times \mathbf{t}_2
\]

Repeat for body vectors, then \(\mathbf{R} = [\mathbf{t}_1^{\text{ref}}, \mathbf{t}_2^{\text{ref}}, \mathbf{t}_3^{\text{ref}}] [\mathbf{t}_1^{\text{body}}, \mathbf{t}_2^{\text{body}}, \mathbf{t}_3^{\text{body}}]^\top\).

TRIAD is exact for two observations and O(1) — it does not iterate. The drawback: it trusts the **first vector completely** and uses the second only to fix the remaining degree of freedom. If the first vector (typically gravity) has noise, that noise directly corrupts the result. For equally-weighted observations, QUEST or SVD are preferable.

```python
from qnav.determination.triad import triad
q = triad(v_ref, v_body)   # v_ref, v_body: (2, 3) arrays
```

## Davenport q-method

Reformulating Wahba's problem in quaternion form: the optimal quaternion is the eigenvector with the largest eigenvalue of the \(4 \times 4\) **Davenport K-matrix**:

\[
\mathbf{K} = \begin{bmatrix}
\mathbf{B} + \mathbf{B}^\top - \text{tr}(\mathbf{B})\mathbf{I} & \mathbf{z} \\
\mathbf{z}^\top & \text{tr}(\mathbf{B})
\end{bmatrix}
\]

where \(\mathbf{z} = \text{vee}(\mathbf{B}^\top - \mathbf{B}) = [B_{32}-B_{23},\; B_{13}-B_{31},\; B_{21}-B_{12}]^\top\).

The sign convention for \(\mathbf{z}\) is critical: the correct form uses \(\mathbf{B}^\top - \mathbf{B}\) (not \(\mathbf{B} - \mathbf{B}^\top\)). The sign error produces a quaternion that differs by a rotation about the wrong axis — it manifests as a systematic bias of order \(2\|\mathbf{z}\|\) in the estimated attitude, which can exceed several degrees for typical sensor noise.

```python
from qnav.determination.davenport import davenport
q = davenport(v_ref, v_body, weights)
```

## QUEST

QUEST (Quaternion Estimator) avoids the full eigendecomposition of the K-matrix by iteratively solving for the largest eigenvalue \(\lambda_{\max}\) using Newton's method on the characteristic polynomial.

Starting from \(\lambda_0 = \lambda_{\text{total}} = 1\) (the upper bound), one Newton step typically converges to machine precision. The quaternion is then recovered without an eigendecomposition:

\[
\mathbf{q}_{\max} = \left[(\lambda_{\max}\mathbf{I} + \mathbf{B} + \mathbf{B}^\top - \text{tr}(\mathbf{B})\mathbf{I})^{-1}\mathbf{z}, \; 1\right]^\top \quad (\text{unnormalized})
\]

QUEST degenerates when \(\lambda_{\max} \approx \pi\) (near a 180° rotation), where the 3×3 matrix becomes near-singular. qnav detects this and falls back to the full Davenport eigendecomposition:

```python
from qnav.determination.quest import solve
q = solve(v_ref, v_body, weights)    # Newton + Davenport fallback
```

## SVD method

The SVD directly maximizes \(\text{tr}(\mathbf{R}^\top \mathbf{B})\) via the polar decomposition:

\[
\mathbf{B} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top \implies \mathbf{R} = \mathbf{U}\, \text{diag}(1, 1, \det(\mathbf{U}\mathbf{V}^\top))\, \mathbf{V}^\top
\]

The determinant correction prevents a reflection when \(\mathbf{B}\) has a negative determinant (which occurs when the data is noisy enough to flip the handedness of the problem). Without it, you get the correct orientation up to a reflection — a difficult bug.

```python
from qnav.determination.svd import svd_attitude
q = svd_attitude(v_ref, v_body, weights)
```

SVD is numerically the most robust method for degenerate configurations (near-collinear vectors, near-zero weights), at the cost of being slower than QUEST for the common 2-vector case.

## OLEQ

OLEQ (Optimal Linear Attitude Estimator from Quaternions) accumulates observations through the left and right quaternion product matrices:

\[
\mathbf{M} = \sum_{i=1}^N w_i \left([\mathbf{v}_i^{\text{ref}}]_L^\top [\mathbf{v}_i^{\text{body}}]_R + [\mathbf{v}_i^{\text{body}}]_R^\top [\mathbf{v}_i^{\text{ref}}]_L\right)
\]

(where \([\mathbf{v}]_L, [\mathbf{v}]_R \in \mathbb{R}^{4\times4}\) are the left/right quaternion multiplication matrices of the pure quaternion \([0, \mathbf{v}]\)). The optimal quaternion is the dominant eigenvector of \(\mathbf{M}\).

OLEQ is useful for overdetermined systems with many observations of the same type (e.g. star trackers with many star pairs) and is easily extended to include prior information.

## Algorithm selection

| Algorithm | Vectors | Speed | Robustness | Notes |
|---|---|---|---|---|
| TRIAD | Exactly 2 | Fastest | Trusts first vector | No weight support |
| Davenport | ≥ 2 | Medium | Full eigendecomp | Reference implementation |
| QUEST | ≥ 2 | Fast | Falls back near π | Best for real-time 2-vector |
| SVD | ≥ 2 | Medium | Best for degenerate | Handles reflections |
| OLEQ | ≥ 2 | Medium | Accumulative | Good for many observations |

For the standard AHRS use case (gravity + magnetic field), QUEST is the recommended choice. For initialization or batch processing, SVD is safer.

*Source: attitudesurvey.tex §Wahba, §QUEST, §Davenport; Kok/Hol/Schön §3.6 (Davenport K reformulation); Markley & Crassidis ch.5–6.*
