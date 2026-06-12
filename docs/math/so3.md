# SO(3) Lie Group

The group \(\text{SO}(3) = \{ \mathbf{R} \in \mathbb{R}^{3 \times 3} \mid \mathbf{R}^\top \mathbf{R} = \mathbf{I},\ \det \mathbf{R} = +1 \}\) is the set of 3D rotation matrices. Its Lie algebra \(\mathfrak{so}(3)\) is the set of \(3 \times 3\) skew-symmetric matrices. The exponential map connects them.

Using SO(3) directly — rather than parametrizing through Euler angles or quaternion components — keeps covariance computation, Jacobians, and filter updates consistent with the underlying manifold geometry.

## Hat and vee

The **hat** operator maps \(\mathbb{R}^3 \to \mathfrak{so}(3)\):

\[
[\boldsymbol{\omega}]_\times = \hat{\boldsymbol{\omega}} = \begin{bmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{bmatrix}
\]

with the property \([\boldsymbol{\omega}]_\times \mathbf{v} = \boldsymbol{\omega} \times \mathbf{v}\).

The **vee** operator is its inverse. Key identities:

\[
[\mathbf{a}]_\times^\top = -[\mathbf{a}]_\times, \quad [\mathbf{u}]_\times^2 = \mathbf{u}\mathbf{u}^\top - \mathbf{I}, \quad [\mathbf{u}]_\times^3 = -[\mathbf{u}]_\times
\]

\[
[\mathbf{R}\mathbf{v}]_\times = \mathbf{R}[\mathbf{v}]_\times \mathbf{R}^\top
\]

The last identity is the equivariance of the cross product under rotation, and it appears repeatedly in Jacobian derivations.

## Exponential map (Rodrigues rotation formula)

\[
\text{Exp}(\boldsymbol{\phi}) = e^{[\boldsymbol{\phi}]_\times}
= \mathbf{I} + A[\boldsymbol{\phi}]_\times + B[\boldsymbol{\phi}]_\times^2
\]

\[
A = \frac{\sin\theta}{\theta}, \quad B = \frac{1 - \cos\theta}{\theta^2}, \quad \theta = \|\boldsymbol{\phi}\|
\]

Below threshold \(10^{-4}\) rad, series expansions replace the trigonometric ratios:

\[
A \approx 1 - \frac{\theta^2}{6}, \quad B \approx \frac{1}{2} - \frac{\theta^2}{24}
\]

The threshold \(10^{-4}\) is chosen so that the series and the closed form agree to machine precision at the boundary. This is tighter than the default in many implementations (which use \(10^{-3}\) or ignore the issue), and it matters for covariance propagation where small-angle Jacobians appear frequently.

## Logarithm map

The inverse takes a rotation matrix to its rotation vector. Three branches handle the numerical cases:

**Small angle** (\(\theta < 10^{-4}\)):

\[
\text{Log}(\mathbf{R}) \approx \frac{1}{2}\left(1 + \frac{\theta^2}{6}\right) \text{vee}(\mathbf{R} - \mathbf{R}^\top)
\]

**Generic** (\(10^{-4} \leq \theta \leq \pi - 10^{-6}\)):

\[
\text{Log}(\mathbf{R}) = \frac{\theta}{2\sin\theta} \text{vee}(\mathbf{R} - \mathbf{R}^\top), \quad \theta = \arccos\!\left(\frac{\text{tr}\,\mathbf{R} - 1}{2}\right)
\]

**Near \(\pi\)** (\(\pi - \theta < 10^{-6}\)):

The generic formula fails because \(\sin\theta \to 0\) and \(\mathbf{R} - \mathbf{R}^\top\) loses all angular information. Near \(\theta = \pi\), the rotation matrix satisfies \(\mathbf{R} \approx 2\mathbf{u}\mathbf{u}^\top - \mathbf{I}\), so:

\[
\mathbf{u}\mathbf{u}^\top = \frac{\mathbf{R} + \mathbf{I}}{2}
\]

The axis \(\mathbf{u}\) is the column with the largest diagonal entry of \(\frac{1}{2}(\mathbf{R} + \mathbf{I})\), normalized. The \(\pm\mathbf{u}\) ambiguity is resolved by choosing the sign that makes \(\mathbf{u}^\top \text{vee}(\mathbf{R} - \mathbf{R}^\top) > 0\).

## Right and left Jacobians

These are the key tools for first-order linearization on SO(3). They appear in ESKF process models, covariance propagation, and Jacobian-based optimization.

**Right Jacobian** \(\mathbf{J}_r\) satisfies:

\[
\text{Exp}(\boldsymbol{\theta} + \delta\boldsymbol{\theta}) \approx \text{Exp}(\boldsymbol{\theta}) \cdot \text{Exp}(\mathbf{J}_r(\boldsymbol{\theta})\, \delta\boldsymbol{\theta})
\]

\[
\mathbf{J}_r(\boldsymbol{\theta}) = \mathbf{I} - B[\boldsymbol{\theta}]_\times + C[\boldsymbol{\theta}]_\times^2
\]

\[
B = \frac{1 - \cos\theta}{\theta^2}, \quad C = \frac{\theta - \sin\theta}{\theta^3}
\]

**Left Jacobian**: \(\mathbf{J}_l(\boldsymbol{\theta}) = \mathbf{J}_r(-\boldsymbol{\theta}) = \mathbf{J}_r(\boldsymbol{\theta})^\top\)

**Inverses**:

\[
\mathbf{J}_r^{-1}(\boldsymbol{\theta}) = \mathbf{I} + \frac{1}{2}[\boldsymbol{\theta}]_\times + D[\boldsymbol{\theta}]_\times^2
\]

\[
D = \frac{1}{\theta^2}\left(1 - \frac{\theta\cos(\theta/2)}{2\sin(\theta/2)}\right) \to \frac{1}{12} + \frac{\theta^2}{720} + \cdots \text{ (small } \theta)
\]

The small-angle limit \(\mathbf{J}_r \approx \mathbf{I} - \frac{1}{2}[\boldsymbol{\theta}]_\times\) is what most filter implementations use; the full closed form is needed when \(\theta\) is not small (e.g., initialization from a large attitude error).

## boxplus and boxminus

**Right retraction** (local perturbation):

\[
\mathbf{R} \oplus \delta = \mathbf{R} \cdot \text{Exp}(\delta)
\]

**Local difference**:

\[
\mathbf{R}_1 \ominus \mathbf{R}_2 = \text{Log}(\mathbf{R}_2^\top \mathbf{R}_1)
\]

so that \(\mathbf{R}_2 \oplus (\mathbf{R}_1 \ominus \mathbf{R}_2) = \mathbf{R}_1\).

These are **right/local** operators: the perturbation \(\delta\) lives in the frame of \(\mathbf{R}\). The ESKF uses the same convention for its error state \(\delta\boldsymbol{\theta}\).

The left variants (perturbation in the world frame) are:

\[
\mathbf{R} \oplus_L \delta = \text{Exp}(\delta) \cdot \mathbf{R}, \quad
\mathbf{R}_1 \ominus_L \mathbf{R}_2 = \text{Log}(\mathbf{R}_1 \mathbf{R}_2^\top)
\]

These are separate functions in qnav — there is no ambiguity about which one you're calling.

## Orthogonal Procrustes projection

Given a near-rotation matrix \(\mathbf{M}\) (e.g., after accumulating floating-point drift), the nearest rotation in Frobenius norm is:

\[
\mathbf{R} = \mathbf{U} \begin{bmatrix} 1 & & \\ & 1 & \\ & & \det(\mathbf{U}\mathbf{V}^\top) \end{bmatrix} \mathbf{V}^\top
\]

from the SVD \(\mathbf{M} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top\). The determinant correction handles the case where the nearest orthogonal matrix has \(\det = -1\) (a reflection), which SVD alone does not prevent.

This is **never applied automatically** by any other qnav function. Silently correcting a drift-corrupted matrix hides the fact that drift has accumulated to the point where the matrix has left SO(3). Use it when you explicitly want to repair a matrix; otherwise investigate the source of drift.

## Geodesic distance

\[
d(\mathbf{R}_1, \mathbf{R}_2) = \|\text{Log}(\mathbf{R}_1^\top \mathbf{R}_2)\|
\]

This is the arc length of the geodesic on SO(3) connecting the two rotations, in \([0, \pi]\). It is bi-invariant: \(d(\mathbf{Q}\mathbf{R}_1, \mathbf{Q}\mathbf{R}_2) = d(\mathbf{R}_1, \mathbf{R}_2)\) for any \(\mathbf{Q} \in \text{SO}(3)\).

*Source: Solà, Quaternion.tex §SO(3); Hashim, Paper_Hashim_SUBMIT.tex. See [`formula_catalog.md`](formula_catalog.md) §3.*
