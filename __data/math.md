# Mathematical Foundations

---

## 1. Quaternion Representation

### 1.1 Convention

Throughout this work, orientations are represented as unit quaternions in scalar-first order:

$$\mathbf{q} = \bigl[w,\; x,\; y,\; z\bigr]^\top \in S^3, \qquad \|\mathbf{q}\| = 1$$

A quaternion $\mathbf{q}$ encodes the rotation from the sensor body frame to the world frame (ENU, $+z$ up), meaning a vector $\mathbf{v}_b$ in the body frame is mapped to the world frame as $\mathbf{v}_w = \mathbf{R}(\mathbf{q})\,\mathbf{v}_b$.

### 1.2 Hamilton Product

The composition of two rotations $\mathbf{q}_1$ and $\mathbf{q}_2$ is the Hamilton product $\mathbf{q}_1 \otimes \mathbf{q}_2$:

$$\mathbf{q}_1 \otimes \mathbf{q}_2 = \begin{bmatrix}
w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2 \\
w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2 \\
w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2 \\
w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2
\end{bmatrix}$$

The product is associative but not commutative. For two successive rotations applied right-to-left, $\mathbf{q}_1 \otimes \mathbf{q}_2$ applies $\mathbf{q}_2$ first, then $\mathbf{q}_1$.

### 1.3 Conjugate and Inverse

For a unit quaternion, the conjugate equals the inverse:

$$\mathbf{q}^{-1} = \mathbf{q}^* = \bigl[w,\; -x,\; -y,\; -z\bigr]^\top$$

This corresponds to the inverse rotation: $\mathbf{R}(\mathbf{q}^{-1}) = \mathbf{R}(\mathbf{q})^\top$.

### 1.4 Rotation Matrix

The rotation matrix corresponding to $\mathbf{q} = [w, x, y, z]^\top$ is:

$$\mathbf{R}(\mathbf{q}) = \begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - xw) \\
2(xz - wy) & 2(yz + xw) & 1 - 2(x^2 + y^2)
\end{bmatrix}$$

### 1.5 Exponential Map

The exponential map converts a rotation vector $\boldsymbol{\omega} \in \mathbb{R}^3$ (axis-angle with magnitude in radians) to a unit quaternion:

$$\text{Exp}(\boldsymbol{\omega}) = \begin{bmatrix}
\cos\!\left(\|\boldsymbol{\omega}\| / 2\right) \\[4pt]
\boldsymbol{\omega}\; \dfrac{\sin\!\left(\|\boldsymbol{\omega}\| / 2\right)}{\|\boldsymbol{\omega}\|}
\end{bmatrix}$$

For small $\|\boldsymbol{\omega}\| < 10^{-8}$, the sinc ratio $\sin(\theta/2)/\theta$ is replaced by its first-order Taylor expansion $\tfrac{1}{2}$ to avoid numerical division by near-zero.

This map is used to propagate orientation using gyroscope measurements:

$$\mathbf{q}_{t+1} = \mathbf{q}_t \otimes \text{Exp}(\boldsymbol{\omega}_t \Delta t_t)$$

### 1.6 Double Cover and Sign Symmetry

The map $\mathbf{q} \mapsto \mathbf{R}(\mathbf{q})$ is two-to-one: both $\mathbf{q}$ and $-\mathbf{q}$ encode the same physical rotation. Any distance or loss function operating on quaternions must therefore be sign-invariant. In particular, the distance between two orientations must satisfy $d(\mathbf{q}_1, \mathbf{q}_2) = d(\mathbf{q}_1, -\mathbf{q}_2)$.

---

## 2. Yaw-Invariant Tilt Metric

### 2.1 Motivation

In ENU coordinates with $+z$ up, the yaw rotation (azimuth heading) is a rotation around the world vertical axis $\hat{\mathbf{z}} = [0, 0, 1]^\top$. The quaternion encoding a pure yaw by angle $\psi$ is:

$$\mathbf{q}_\text{yaw}(\psi) = \bigl[\cos(\psi/2),\; 0,\; 0,\; \sin(\psi/2)\bigr]^\top$$

For this quaternion, $w^2 + z^2 = \cos^2(\psi/2) + \sin^2(\psi/2) = 1$. This observation — that the $w$ and $z$ components of a pure yaw quaternion have unit combined magnitude — motivates the RIANN tilt metric.

### 2.2 The RIANN Tilt Metric

Given a predicted quaternion $\hat{\mathbf{q}}$ and a ground-truth quaternion $\mathbf{q}_\text{gt}$, the error quaternion is:

$$\mathbf{q}_\text{err} = \hat{\mathbf{q}} \otimes \mathbf{q}_\text{gt}^{-1} = \bigl[w_\text{err},\; x_\text{err},\; y_\text{err},\; z_\text{err}\bigr]^\top$$

The tilt angle $e_\alpha$ is defined as:

$$e_\alpha = 2\arccos\!\left(\min\!\left(1,\; \sqrt{w_\text{err}^2 + z_\text{err}^2}\right)\right)$$

**Yaw invariance.** If the prediction error is a pure yaw rotation, then $\mathbf{q}_\text{err} = \mathbf{q}_\text{yaw}(\psi)$ and $w_\text{err}^2 + z_\text{err}^2 = 1$, giving $e_\alpha = 0$. Any heading offset between the prediction and the ground truth is invisible to this metric.

**Tilt sensitivity.** If the error is a pure roll or pitch, $z_\text{err} = 0$ and $w_\text{err}^2 + x_\text{err}^2 + y_\text{err}^2 = 1$. For a pure pitch by angle $\theta$ (rotation around world $y$-axis), $\mathbf{q}_\text{err} = [\cos(\theta/2), 0, \sin(\theta/2), 0]^\top$, giving $\sqrt{w_\text{err}^2 + z_\text{err}^2} = \cos(\theta/2)$ and $e_\alpha = \theta$. The metric recovers the exact tilt angle.

**Training loss.** The supervised training loss is the mean squared tilt error over a sequence of $N$ timesteps:

$$\mathcal{L}_\text{tilt} = \frac{1}{N} \sum_{t=1}^N e_{\alpha,t}^2 \quad \text{(radians}^2\text{)}$$

---

## 3. Transformer Attention

### 3.1 Scaled Dot-Product Attention

The fundamental attention operation maps a query matrix $\mathbf{Q} \in \mathbb{R}^{L \times d_k}$, key matrix $\mathbf{K} \in \mathbb{R}^{S \times d_k}$, and value matrix $\mathbf{V} \in \mathbb{R}^{S \times d_v}$ to an output $\mathbf{O} \in \mathbb{R}^{L \times d_v}$:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{A}\right) \mathbf{V}$$

where $\mathbf{A} \in \{0, -\infty\}^{L \times S}$ is an additive mask. Positions with $A_{ij} = -\infty$ receive zero attention weight after the softmax, effectively blocking attention to those positions.

Multi-head attention applies $H$ independent attention heads in parallel, projects the results, and combines:

$$\text{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\, \mathbf{W}_O$$

$$\text{head}_h = \text{Attention}(\mathbf{Q} \mathbf{W}^h_Q,\; \mathbf{K} \mathbf{W}^h_K,\; \mathbf{V} \mathbf{W}^h_V)$$

### 3.2 Causal Mask

For a sequence being processed causally (position $t$ may not attend to positions $t' > t$), the standard upper-triangular causal mask is:

$$A_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$$

This forces the attention weight matrix to be lower-triangular after the softmax.

### 3.3 Transformer-XL Mask

When processing a current chunk of length $L_c$ with XL memory of $M$ tokens, the key-value source has length $M + L_c$. The combined mask $\mathbf{A} \in \{0, -\infty\}^{L_c \times (M + L_c)}$ is:

$$A_{i,j} = \begin{cases} 0 & j < M \text{\quad (memory: full attention)} \\ 0 & j \geq M \text{ and } (j - M) \leq i \text{\quad (chunk: causal)} \\ -\infty & \text{otherwise} \end{cases}$$

The memory tokens (first $M$ positions in the key-value dimension) are fully attended from all query positions, since they all lie in the past. Within the current chunk, causality is maintained.

---

## 4. Truncated Backpropagation Through Time

### 4.1 The Gradient Flow Problem

Consider a sequence of length $T$ decomposed into $K = \lceil T / L_c \rceil$ chunks of length $L_c$. The model's prediction at step $t$ depends on all preceding inputs through the computation graph, and the loss at step $t$ produces gradients that propagate backward through this graph.

In naive full backpropagation through time (full BPTT), the computation graph for all $T$ steps is retained in memory, and gradients propagate from step $T$ back to step $0$. The memory cost is $O(T \cdot d)$ for the activations, which is prohibitive for $T = 44{,}000$ (440-second BROAD sequences at 100 Hz).

### 4.2 TBPTT with Stateful Memory

TBPTT processes each chunk independently, backpropagating only through the chunk's $L_c$ steps and discarding the computation graph thereafter. The cross-chunk information flow is maintained through the Transformer-XL memory $\mathbf{M}^{(\ell)}_k$, which is passed between chunks but detached from the gradient graph:

$$\nabla_{\mathbf{M}^{(\ell)}_k} \mathcal{L}_k = \mathbf{0} \quad \text{(memory detached)}$$

This means the gradient of the loss on chunk $k$ with respect to the model parameters reflects only the direct computation within chunk $k$ — the model learns to minimize loss in the current chunk, conditional on the detached memory context from previous chunks.

The memory-efficient attention implementation is critical here: the attention over $(M + L_c)$ key-value tokens with $L_c$ queries does not materialize the full $(L_c \times (M + L_c))$ attention weight matrix, and the detached memory tensors do not require gradient tracking, keeping the memory footprint proportional to $L_c$ rather than to the accumulated context length.

### 4.3 Training-Inference Correspondence

At inference time, the XL memory is accumulated across all chunks without detachment limits, providing the full context. At training time, the gradient is truncated at chunk boundaries. This creates a deliberate discrepancy: the model's forward pass is identical in both cases (same memory accumulation), but the backward pass during training provides gradient information only within $L_c$ steps.

The practical consequence is that the model learns to use all available context at inference time, but its parameter updates during training are driven by local $L_c$-step prediction accuracy. For well-designed models with sufficient context (sufficient $M_\text{max}$), the local learning signal is adequate to produce globally accurate attitude estimates.

---

## 5. Gravity Alignment Constraint

On quasi-static frames where $|\|\mathbf{a}_t\| - g| < \tau$ for threshold $\tau$, the accelerometer measures the body-frame gravity vector. The predicted gravity direction in the body frame, derived from the model's quaternion output, is:

$$\hat{\mathbf{g}}_t = \mathbf{R}(\hat{\mathbf{q}}_t)^\top \hat{\mathbf{z}}, \qquad \hat{\mathbf{z}} = [0, 0, 1]^\top$$

The cosine loss between the predicted and measured gravity directions is:

$$\ell_\text{grav}(\hat{\mathbf{q}}_t, \mathbf{a}_t) = 1 - \hat{\mathbf{a}}_t \cdot \hat{\mathbf{g}}_t = 1 - \frac{\mathbf{a}_t}{\|\mathbf{a}_t\|} \cdot \mathbf{R}(\hat{\mathbf{q}}_t)^\top \hat{\mathbf{z}}$$

This formulation is yaw-invariant: $\mathbf{R}(\hat{\mathbf{q}}_t)^\top \hat{\mathbf{z}}$ is the gravity direction, which depends only on tilt (roll and pitch), not heading. The minimum of $\ell_\text{grav}$ is achieved when the predicted gravity direction aligns with the measured accelerometer reading.

The aggregate loss over quasi-static frames is:

$$\mathcal{L}_\text{grav} = \frac{1}{|\mathcal{S}|} \sum_{t \in \mathcal{S}} \ell_\text{grav}(\hat{\mathbf{q}}_t, \mathbf{a}_t), \qquad \mathcal{S} = \bigl\{t : \bigl|\|\mathbf{a}_t\| - g\bigr| < 0.2\;\text{m s}^{-2}\bigr\}$$

The arccos form of this distance has a gradient singularity when $\hat{\mathbf{a}}_t \cdot \hat{\mathbf{g}}_t \to 1$ (perfect alignment), which would damage optimization near convergence. The cosine form $1 - \cos\theta$ has bounded gradient everywhere and approaches $\theta^2/2$ for small $\theta$, providing well-conditioned optimization in both the large-error and near-convergence regimes.

---

## 6. Virtual Rotation Augmentation

### 6.1 Derivation

Consider an IMU mounted on a rigid body at a fixed but unknown orientation $\mathbf{q}_\text{rand}$ relative to the body's principal frame. The sensor frame and body frame are related by $\mathbf{R}_\text{sensor→body} = \mathbf{R}(\mathbf{q}_\text{rand})$.

Under this mounting, the IMU in the sensor frame measures:

$$\mathbf{a}_\text{sensor} = \mathbf{R}(\mathbf{q}_\text{rand})\, \mathbf{a}_\text{body}, \qquad \boldsymbol{\omega}_\text{sensor} = \mathbf{R}(\mathbf{q}_\text{rand})\, \boldsymbol{\omega}_\text{body}$$

The true orientation of the body frame relative to the world is $\mathbf{q}_\text{body}$. The orientation of the sensor frame relative to the world is:

$$\mathbf{q}_\text{sensor} = \mathbf{q}_\text{body} \otimes \mathbf{q}_\text{rand}^{-1}$$

because the rotation from the sensor frame to the world frame is: first rotate sensor to body ($\mathbf{q}_\text{rand}^{-1}$), then rotate body to world ($\mathbf{q}_\text{body}$).

### 6.2 Augmentation Transform

Given original measurements $(\mathbf{a}_t, \boldsymbol{\omega}_t, \mathbf{q}_{\text{gt},t})$ from a dataset recorded in a specific sensor orientation, the virtual rotation augmentation generates a new valid training example by sampling $\mathbf{q}_\text{rand}$ uniformly from $S^3$ and applying:

$$\mathbf{a}'_t = \mathbf{R}(\mathbf{q}_\text{rand})\, \mathbf{a}_t, \qquad \boldsymbol{\omega}'_t = \mathbf{R}(\mathbf{q}_\text{rand})\, \boldsymbol{\omega}_t, \qquad \mathbf{q}'_{\text{gt},t} = \mathbf{q}_{\text{gt},t} \otimes \mathbf{q}_\text{rand}^{-1}$$

The transformation is applied once per training window (same $\mathbf{q}_\text{rand}$ for all timesteps in the window), simulating the effect of a fixed but randomly chosen mounting orientation.

### 6.3 Equivariance Property

The augmentation preserves the correctness of the estimation: if $\hat{\mathbf{q}}_t$ was the correct output for the original example, then $\hat{\mathbf{q}}_t \otimes \mathbf{q}_\text{rand}^{-1}$ is the correct output for the augmented example. The tilt loss $e_\alpha$ satisfies:

$$e_\alpha\!\left(\hat{\mathbf{q}} \otimes \mathbf{q}_\text{rand}^{-1},\; \mathbf{q}_\text{gt} \otimes \mathbf{q}_\text{rand}^{-1}\right) = e_\alpha(\hat{\mathbf{q}},\; \mathbf{q}_\text{gt})$$

because the $\mathbf{q}_\text{rand}^{-1}$ terms cancel in the error quaternion $\hat{\mathbf{q}} \otimes \mathbf{q}_\text{gt}^{-1}$. The augmentation therefore does not change the loss value for a correct estimator — it only changes the input signal, forcing the estimator to learn features that are invariant to the specific sensor mounting orientation.
