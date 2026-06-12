# qnav Formula Catalog

Extracted formula catalog with full source traceability. Every formula cites its source file
(and the LaTeX equation label / section / equation number where available). Where sources
disagree, both forms are shown and the discrepancy is noted.

**Source abbreviations**

| Abbrev. | Source |
|---|---|
| [MATH] | `__data/math.md` — qnav in-house convention baseline |
| [SOLA] | Solà, *Quaternion kinematics for the error-state Kalman filter* — `__data/Quaternion kinematics for the error-state Kalman filter/*.tex` |
| [HASH] | Hashim, *SO(3), Euler Angles, Angle-axis, Rodriguez Vector and Unit-Quaternion* — `__data/Special Orthogonal Group SO(3), .../Paper_Hashim_SUBMIT.tex` |
| [SURV] | Al-Jlailaty & Mansour, *Efficient Attitude Estimators: A Tutorial and Survey* — `__data/Efficient Attitude Estimators A Tutorial and Survey/attitudesurvey.tex` |
| [KOK]  | Kok, Hol & Schön, *Using Inertial Sensors for Position and Orientation Estimation* — `__data/Using Inertial Sensors for Position and Orientation Estimation/*.tex` |
| [PARW] | Parwana & Kothari, *Quaternions and Attitude Representation* — `__data/Quaternions and Attitude Representation/Quaternion_archive.tex` |
| [SATD] | `__data/Satdyn_mb_2010f.pdf` — satellite dynamics notes |

Unless stated otherwise, all formulas below are written in the **qnav convention**: Hamilton,
scalar-first $\mathbf{q}=[w,x,y,z]^\top$, passive body→world transform
$\mathbf{v}_w = \mathbf{R}(\mathbf{q})\,\mathbf{v}_b$, right-handed, radians.

---

## 1. Quaternion algebra

### 1.1 Definition and algebra basis

$$q = q_w + q_x i + q_y j + q_z k \in \mathbb{H},\qquad
i^2=j^2=k^2=ijk=-1,\quad ij=-ji=k,\ jk=-kj=i,\ ki=-ik=j$$

[SOLA Quaternion.tex `equ:ijkQuat`, `equ:quatAlgebra`]. The choice $ij=k$ (vs JPL's $ji=k$)
makes the quaternion **right-handed** (Hamilton). [SOLA §"Quaternion conventions", `equ:quatAlgDef`]

Vector form (scalar-first):

$$\mathbf{q} = \begin{bmatrix} q_w \\ \mathbf{q}_v \end{bmatrix}
= \begin{bmatrix} q_w \\ q_x \\ q_y \\ q_z \end{bmatrix}$$

[SOLA Quaternion.tex §1.1.1; MATH §1.1; KOK §3.4 ($q=(q_0,q_v)$); HASH `eq:OVERVIEW_PER_Q_1`]

### 1.2 Product (Hamilton product)

$$\mathbf{p}\otimes\mathbf{q} = \begin{bmatrix}
p_wq_w - p_{x}q_{x} - p_{y}q_{y} - p_{z}q_{z} \\
p_wq_{x} + p_{x}q_w + p_{y}q_{z} - p_{z}q_{y} \\
p_wq_{y} - p_{x}q_{z} + p_{y}q_w + p_{z}q_{x} \\
p_wq_{z} + p_{x}q_{y} - p_{y}q_{x} + p_{z}q_w
\end{bmatrix}
= \begin{bmatrix}
p_wq_w - \mathbf{p}_v^\top\mathbf{q}_v \\
p_w\mathbf{q}_v + q_w\mathbf{p}_v + \mathbf{p}_v\times\mathbf{q}_v
\end{bmatrix}$$

[SOLA `equ:quatProd`, `equ:quatProdVec`; identical in MATH §1.2, KOK `eq:models-quatMult` (⊙),
HASH `eq:OVERVIEW_Q_Mul` (⊙), PARW (∘)]. Associative, distributive, **not commutative**;
commutator $\mathbf{p}\otimes\mathbf{q}-\mathbf{q}\otimes\mathbf{p} = 2\,\mathbf{p}_v\times\mathbf{q}_v$
[SOLA `equ:quatCommutator`].

Matrix forms ($\mathbf{q}_1\otimes\mathbf{q}_2 = [\mathbf{q}_1]_L\,\mathbf{q}_2 = [\mathbf{q}_2]_R\,\mathbf{q}_1$,
[SOLA `equ:quatMatProd`–`equ:quatMatrix`; KOK `eq:models-leftRightQuatMult`]):

$$[\mathbf{q}]_L = q_w\mathbf{I}_4 + \begin{bmatrix} 0 & -\mathbf{q}_v^\top \\ \mathbf{q}_v & [\mathbf{q}_v]_\times \end{bmatrix},
\qquad
[\mathbf{q}]_R = q_w\mathbf{I}_4 + \begin{bmatrix} 0 & -\mathbf{q}_v^\top \\ \mathbf{q}_v & -[\mathbf{q}_v]_\times \end{bmatrix},
\qquad [\mathbf{p}]_R[\mathbf{q}]_L = [\mathbf{q}]_L[\mathbf{p}]_R$$

### 1.3 Identity, conjugate, norm, inverse

$$\mathbf{q}_1 = [1,\mathbf{0}]^\top;\qquad
\mathbf{q}^* = \begin{bmatrix} q_w \\ -\mathbf{q}_v \end{bmatrix};\qquad
(\mathbf{p}\otimes\mathbf{q})^* = \mathbf{q}^*\otimes\mathbf{p}^*$$

$$\|\mathbf{q}\| = \sqrt{\mathbf{q}\otimes\mathbf{q}^*} = \sqrt{q_w^2+q_x^2+q_y^2+q_z^2};\qquad
\|\mathbf{p}\otimes\mathbf{q}\| = \|\mathbf{p}\|\,\|\mathbf{q}\|$$

$$\mathbf{q}^{-1} = \mathbf{q}^*/\|\mathbf{q}\|^2
\;\xrightarrow{\ \|\mathbf{q}\|=1\ }\; \mathbf{q}^{-1}=\mathbf{q}^*$$

[SOLA `equ:q_norm` and surrounding; MATH §1.3; HASH `eq:OVERVIEW_PER_Q_2`]

### 1.4 Exponential of a quaternion

Pure quaternion $\mathbf{v}=\mathbf{u}\theta$, $\|\mathbf{u}\|=1$ (extended Euler formula):

$$e^{\mathbf{v}} = e^{\mathbf{u}\theta} = \cos\theta + \mathbf{u}\sin\theta
= \begin{bmatrix} \cos\theta \\ \mathbf{u}\sin\theta \end{bmatrix},\qquad
e^{-\mathbf{v}} = (e^{\mathbf{v}})^*$$

[SOLA `equ:EulerFormulaQuat`]. Small-angle Taylor safeguard:
$e^{\mathbf{v}} \approx \big[\,1-\theta^2/2,\ \ \mathbf{v}(1-\theta^2/6)\,\big]^\top \to [1,\mathbf{v}]^\top$
[SOLA Quaternion.tex, after `equ:EulerFormulaQuat`; MATH §1.5 uses threshold $\|\boldsymbol\omega\|<10^{-8}$
with sinc → 1/2 replacement].

General quaternion: $e^{\mathbf{q}} = e^{q_w}\big[\cos\|\mathbf{q}_v\|,\ \tfrac{\mathbf{q}_v}{\|\mathbf{q}_v\|}\sin\|\mathbf{q}_v\|\big]^\top$
[SOLA `equ:expGeneralQuat`].

### 1.5 Logarithm

Unit quaternion: $\log\mathbf{q} = \mathbf{u}\theta$ with

$$\theta = \arctan(\|\mathbf{q}_v\|,\,q_w),\qquad \mathbf{u} = \mathbf{q}_v/\|\mathbf{q}_v\|$$

(4-quadrant `atan2`). Small-angle series:
$\log(\mathbf{q}) \approx \tfrac{\mathbf{q}_v}{q_w}\big(1 - \tfrac{\|\mathbf{q}_v\|^2}{3q_w^2}\big) \approx \mathbf{q}_v$
[SOLA `equ:qlog` and §"Logarithm of unit quaternions"]. General quaternion:
$\log\mathbf{q} = [\log\|\mathbf{q}\|,\ \mathbf{u}\theta]^\top$.

> Note: [KOK `eq:models-logq-map`] uses $\rho=\arccos(q_0)\,q_v/\|q_v\|_2$ — equivalent on the
> $q_0\ge 0$ hemisphere but the `atan2` form [SOLA] is numerically preferred (qnav adopts atan2).

### 1.6 Power and SLERP

$$\mathbf{q}^t = \exp(t\log\mathbf{q}) = \begin{bmatrix}\cos t\theta \\ \mathbf{u}\sin t\theta\end{bmatrix}
\quad (\|\mathbf{q}\|=1)$$

[SOLA `equ:qa`]. SLERP (three equivalent methods, [SOLA §"SLERP" / `sec:slerp`]):

$$\mathbf{q}(t) = \mathbf{q}_0\otimes(\mathbf{q}_0^*\otimes\mathbf{q}_1)^t
= \mathbf{q}_0\cos(t\Delta\theta) + \mathbf{q}_\perp\sin(t\Delta\theta)
= \mathbf{q}_0\frac{\sin((1-t)\Delta\theta)}{\sin\Delta\theta}+\mathbf{q}_1\frac{\sin(t\Delta\theta)}{\sin\Delta\theta}$$

with $\Delta\theta=\arccos(\mathbf{q}_0^\top\mathbf{q}_1)$,
$\mathbf{q}_\perp = \dfrac{\mathbf{q}_1-(\mathbf{q}_0^\top\mathbf{q}_1)\mathbf{q}_0}{\|\mathbf{q}_1-(\mathbf{q}_0^\top\mathbf{q}_1)\mathbf{q}_0\|}$.
**Shortest-path fix:** if $\mathbf{q}_0^\top\mathbf{q}_1<0$, negate $\mathbf{q}_1$ (double cover)
[SOLA `fig:slerp_fix` discussion]. Matrix analogue $\mathbf{R}(t)=\mathbf{R}_0(\mathbf{R}_0^\top\mathbf{R}_1)^t$ [SOLA `equ:Rslerp`].

### 1.7 Double cover

$\mathbf{R}\{-\mathbf{q}\}=\mathbf{R}\{\mathbf{q}\}$ [SOLA `equ:rotneg`; MATH §1.6; HASH §7
"Problem of unit-quaternion"]. The 4-D angle to identity is half the 3-D rotation angle,
$\theta_{4D}=\phi/2$ [SOLA `sec:double_cover`]. **All quaternion distance/loss functions must be
sign-invariant** [MATH §1.6].

---

## 2. Representation conversions

### 2.1 Rotation vector / axis-angle → quaternion

$$\mathbf{q}\{\boldsymbol\phi\} = \mathrm{Exp}(\phi\mathbf{u}) = e^{\phi\mathbf{u}/2}
= \begin{bmatrix} \cos(\phi/2) \\ \mathbf{u}\sin(\phi/2) \end{bmatrix},
\qquad \mathrm{Exp}(\boldsymbol\phi)\triangleq\exp(\boldsymbol\phi/2)$$

[SOLA `equ:vectoquat`; MATH §1.5; HASH `eq:OVERVIEW_Q_ang_2_Q`; SURV `eq_quaternion`].

> **Sign discrepancy:** [KOK `eq:models-axisAngle-quat`] writes
> $q^{uv}(n,\alpha)=[\cos\frac{\alpha}{2},\,-n\sin\frac{\alpha}{2}]^\top$ with a **minus** sign —
> a consequence of Kok et al. defining the rotation angle for the frame (passive, opposite
> sense). qnav adopts the [SOLA]/[MATH] positive-sign form.

### 2.2 Quaternion → axis-angle (Log)

$$\phi = 2\,\mathrm{atan2}(\|\mathbf{q}_v\|,\,q_w),\qquad \mathbf{u}=\mathbf{q}_v/\|\mathbf{q}_v\|,
\qquad \mathrm{Log}(\mathbf{q}) \triangleq 2\log(\mathbf{q}) = \phi\,\mathbf{u}$$

Small-angle: $\mathrm{Log}(\mathbf{q}) \approx 2\frac{\mathbf{q}_v}{q_w}\big(1-\frac{\|\mathbf{q}_v\|^2}{3q_w^2}\big)$
[SOLA `equ:log_q_small`]. HASH variant: $\alpha=2\cos^{-1}(q_0)$, $u = q/\sin(\alpha/2)$
[HASH `eq:OVERVIEW_Q_Q_2_ang1/2`] — valid but `atan2` form is preferred near $\phi\approx 0,\ 2\pi$.

### 2.3 Quaternion → DCM

$$\mathbf{R}\{\mathbf{q}\} = (q_w^2-\mathbf{q}_v^\top\mathbf{q}_v)\,\mathbf{I}_3 + 2\,\mathbf{q}_v\mathbf{q}_v^\top + 2\,q_w[\mathbf{q}_v]_\times$$

$$\mathbf{R}\{\mathbf{q}\} = \begin{bmatrix}
q_w^2{+}q_x^2{-}q_y^2{-}q_z^2 & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\
2(q_xq_y + q_wq_z) & q_w^2{-}q_x^2{+}q_y^2{-}q_z^2 & 2(q_yq_z - q_wq_x) \\
2(q_xq_z - q_wq_y) & 2(q_yq_z + q_wq_x) & q_w^2{-}q_x^2{-}q_y^2{+}q_z^2
\end{bmatrix}$$

[SOLA Quaternion.tex "quaternion to rotation matrix" boxed eq.; identical (unit-norm form
$1-2(y^2+z^2)$ etc.) in MATH §1.4, HASH `eq:OVERVIEW_Q_R`/`eq:OVERVIEW_Q_R1`, KOK `eq:app-defRq`,
SURV DCM-from-quaternion algorithm $c_{11}\dots c_{33}$, PARW]. **All sources agree** on this
matrix under the body→world / Hamilton reading. Homomorphism properties:
$\mathbf{R}\{\mathbf{q}^*\}=\mathbf{R}\{\mathbf{q}\}^\top$,
$\mathbf{R}\{\mathbf{q}_1\otimes\mathbf{q}_2\}=\mathbf{R}\{\mathbf{q}_1\}\mathbf{R}\{\mathbf{q}_2\}$,
$\mathbf{R}\{\mathbf{q}^t\}=\mathbf{R}\{\mathbf{q}\}^t$
[SOLA `equ:rotident`–`equ:rotprod`].

### 2.4 DCM → quaternion (Shepperd-style robust extraction)

Primary branch (valid when $1+\mathrm{Tr}\,R > 0$ comfortably):

$$q_w = \tfrac12\sqrt{1+R_{11}+R_{22}+R_{33}},\qquad
\mathbf{q}_v = \frac{1}{4q_w}\begin{bmatrix} R_{32}-R_{23} \\ R_{13}-R_{31} \\ R_{21}-R_{12} \end{bmatrix}$$

[HASH `eq:OVERVIEW_Q_comp_1`; KOK appendix]. Robust variants pivoting on the largest of
$\{q_w,q_x,q_y,q_z\}$ (largest diagonal element of $R$), e.g.
$q_x = \pm\tfrac12\sqrt{1+R_{11}-R_{22}-R_{33}}$, etc.
[HASH `eq:OVERVIEW_Q_comp_2/3/4`]. qnav `attitude/conversions` must implement the 4-branch
pivoted version to avoid division by near-zero.

### 2.5 Rotation vector → DCM (Rodrigues rotation formula) and back

$$\mathbf{R} = \mathrm{Exp}(\boldsymbol\phi) = e^{[\boldsymbol\phi]_\times}
= \mathbf{I} + \sin\phi\,[\mathbf{u}]_\times + (1-\cos\phi)\,[\mathbf{u}]_\times^2
= \mathbf{I}\cos\phi + [\mathbf{u}]_\times\sin\phi + \mathbf{u}\mathbf{u}^\top(1-\cos\phi)$$

[SOLA `equ:rodrigues`, `equ:vectomat`; HASH `eq:SO3PPF_PER_AX_ANG`; KOK `eq:models-axisAngle-rotMatrix`
— note KOK writes $\mathcal{I}_3 - \sin\alpha[n\times]+(1-\cos\alpha)[n\times]^2$, the **minus sign**
again reflecting their opposite angle-sign convention; qnav uses the $+\sin$ form].

Inverse (Log map):

$$\phi = \arccos\!\Big(\frac{\mathrm{Tr}(\mathbf{R})-1}{2}\Big),\qquad
\mathbf{u} = \frac{(\mathbf{R}-\mathbf{R}^\top)^\vee}{2\sin\phi}$$

[SOLA "logarithmic maps" of $SO(3)$; HASH `eq:OVERVIEW_att_ang_alpha`, `eq:OVERVIEW_att_ang_u`].
**Singularities:** undefined axis at $\phi=0$ (use series) and extraction failure at
$\phi=\pi$ ($\sin\phi=0$; recover axis from the symmetric part $\mathbf{u}\mathbf{u}^\top = (\mathbf{R}+\mathbf{I})/2$)
[HASH §5 "Problems of angle-axis parameterization": fails at $\alpha=k\pi$].

### 2.6 Euler angles (intrinsic Z-Y'-X'' yaw–pitch–roll, aerospace)

Elementary (frame-transform / passive) matrices [SURV eq. block around `eq_CDCM`; HASH `eq:OVERVIEW_EUL_R`]:

$$C_x(\phi)=\begin{bmatrix}1&0&0\\0&c\phi&s\phi\\0&-s\phi&c\phi\end{bmatrix},\quad
C_y(\theta)=\begin{bmatrix}c\theta&0&-s\theta\\0&1&0\\s\theta&0&c\theta\end{bmatrix},\quad
C_z(\psi)=\begin{bmatrix}c\psi&s\psi&0\\-s\psi&c\psi&0\\0&0&1\end{bmatrix}$$

Body→nav DCM (qnav default; equals $\mathbf{R}_{nb}$):

$$\mathbf{R}_{nb} = R_z(\psi)\,R_y(\theta)\,R_x(\phi) = \begin{bmatrix}
c\theta c\psi & s\phi s\theta c\psi - c\phi s\psi & c\phi s\theta c\psi + s\phi s\psi\\
c\theta s\psi & s\phi s\theta s\psi + c\phi c\psi & c\phi s\theta s\psi - s\phi c\psi\\
-s\theta & s\phi c\theta & c\phi c\theta \end{bmatrix}$$

[HASH `eq:OVERVIEW_EUL_R` (body→inertial, $R_{z}R_{y}R_{x}$ with active elementary rotations);
SURV `eq_CDCM` writes the transpose, $C_n^b = C_z(\psi)C_y(\theta)C_x(\phi)$ with passive
elementary matrices — nav→body. KOK `eq:models-rotMatrix` likewise composes as
$R^{uv}=R(e_1,\phi)R(e_2,\theta)R(e_3,\psi)$ giving the nav→body matrix. **Same Z-Y-X sequence
in all sources; only the direction (body→nav vs nav→body) differs — transpose relation.**]

DCM → Euler (from $\mathbf{R}_{nb}$ above):

$$\phi = \mathrm{atan2}(R_{32}, R_{33}),\qquad
\theta = -\arcsin(R_{31}) = \mathrm{atan2}\!\big({-R_{31}},\ \sqrt{R_{32}^2+R_{33}^2}\big),\qquad
\psi = \mathrm{atan2}(R_{21}, R_{11})$$

[HASH `eq:SO3_EUL`; SURV roll/pitch/heading algorithm (expressed on $C_b^n$ elements
$c_{32},c_{33},c_{31},c_{21},c_{11}$); KOK gives the equivalent on $R^{bn}$:
$\psi=\tan^{-1}(R_{12}/R_{11})$, $\theta=-\sin^{-1}(R_{13})$, $\phi=\tan^{-1}(R_{23}/R_{33})$ —
same formulas after the transpose].

**Gimbal lock:** at $\theta=\pm\pi/2$, $\phi$ and $\psi$ are not separately observable; the
Euler-rate matrix is singular [HASH §4 "Euler parameterization problem"; KOK §3.4].

Euler → quaternion: compose elementary quaternions
$\mathbf{q} = \mathbf{q}_z(\psi)\otimes\mathbf{q}_y(\theta)\otimes\mathbf{q}_x(\phi)$ with
$\mathbf{q}_z(\psi)=[\cos\frac\psi2,0,0,\sin\frac\psi2]^\top$ etc. (qnav default Z-Y'-X'').
[PARW derives the analogous product for the 1-2-3 sequence — different sequence, see conflict
table in `docs/source_index.md`.]

Quaternion → Euler (Z-Y'-X'', from §2.3 matrix elements):

$$\phi=\mathrm{atan2}\big(2(q_yq_z+q_wq_x),\ q_w^2-q_x^2-q_y^2+q_z^2\big),\quad
\theta=\arcsin\big(2(q_wq_y-q_xq_z)\big),\quad
\psi=\mathrm{atan2}\big(2(q_xq_y+q_wq_z),\ q_w^2+q_x^2-q_y^2-q_z^2\big)$$

(derived by substituting §2.3 into the DCM→Euler atan2 formulas; clamp the $\arcsin$ argument
to $[-1,1]$.)

### 2.7 Rodrigues (Gibbs) vector

Definition and angle-axis relations [HASH `eq:OVERVIEW_SO3_lem7_1/2/3`]:

$$\boldsymbol\rho = \tan\!\big(\tfrac{\alpha}{2}\big)\,\mathbf{u},\qquad
\alpha = 2\arctan\|\boldsymbol\rho\|,\qquad \mathbf{u}=\cot\!\big(\tfrac\alpha2\big)\boldsymbol\rho$$

Rodrigues → DCM (and Cayley form) [HASH `eq:OVERVIEW_PER_ROD`, `eq:OVERVIEW_ROD_SO3_R`]:

$$\mathbf{R}_\rho = \frac{1}{1+\|\boldsymbol\rho\|^2}\Big((1-\|\boldsymbol\rho\|^2)\mathbf{I}_3
+ 2\boldsymbol\rho\boldsymbol\rho^\top + 2[\boldsymbol\rho]_\times\Big)
= (\mathbf{I}_3+[\boldsymbol\rho]_\times)(\mathbf{I}_3-[\boldsymbol\rho]_\times)^{-1}$$

DCM → Rodrigues [HASH `eq:OVERVIEW_ROD`, `eq:OVERVIEW_SO3_ROD`]:

$$\boldsymbol\rho = \mathrm{vex}\big((\mathbf{R}+\mathbf{I}_3)^{-1}(\mathbf{R}-\mathbf{I}_3)\big)
= \frac{1}{1+\mathrm{Tr}(\mathbf{R})}\begin{bmatrix} R_{32}-R_{23}\\ R_{13}-R_{31}\\ R_{21}-R_{12}\end{bmatrix}$$

Quaternion ↔ Rodrigues [HASH Lemma 4, `eq:OVERVIEW_SO3_lem4_1/2/3`]:

$$q_w = \pm\frac{1}{\sqrt{1+\|\boldsymbol\rho\|^2}},\qquad
\mathbf{q}_v = \pm\frac{\boldsymbol\rho}{\sqrt{1+\|\boldsymbol\rho\|^2}},\qquad
\boldsymbol\rho = \frac{\mathbf{q}_v}{q_w}$$

Rodrigues kinematics [HASH `eq:OVERVIEW_ROD_dot`]:
$\dot{\boldsymbol\rho} = \tfrac12\big(\mathbf{I}_3+[\boldsymbol\rho]_\times+\boldsymbol\rho\boldsymbol\rho^\top\big)\boldsymbol\Omega$.

**Singularity:** $\boldsymbol\rho\to\infty$ as $\alpha\to\pi$; the three 180° principal rotations
are unrepresentable [HASH §6 "Problems of Rodriguez vector parameterization"].

### 2.8 Modified Rodrigues Parameters (MRP)

$$\boldsymbol\sigma = \tan\!\big(\tfrac{\alpha}{4}\big)\,\mathbf{u} = \frac{\mathbf{q}_v}{1+q_w},
\qquad \text{shadow set } \boldsymbol\sigma' = -\frac{\boldsymbol\sigma}{\|\boldsymbol\sigma\|^2}$$

[HASH mentions the MRP/shadow-set transformation only in passing; no full MRP kinematics in the
corpus — qnav `attitude/mrp` will need standard literature (Markley & Crassidis) as a
supplementary source. Singularity at $\alpha=2\pi$; switch to the shadow set at $\|\boldsymbol\sigma\|>1$.]

---

## 3. SO(3): hat/vee, exp/log, Jacobians

### 3.1 Group definitions

$$SO(3)=\{\mathbf{R}\in\mathbb{R}^{3\times3}\mid \mathbf{R}^\top\mathbf{R}=\mathbf{R}\mathbf{R}^\top=\mathbf{I}_3,\ \det\mathbf{R}=+1\},\qquad
\mathfrak{so}(3)=\{\mathcal{A}\mid \mathcal{A}^\top=-\mathcal{A}\}$$

[SOLA `equ:Rorthogonal`, `equ:unitDet`; HASH `eq:OVERVIEW_PER_SO3_1/2/3`]

### 3.2 Hat / vee

$$[\mathbf{a}]_\times \equiv \mathbf{a}^\wedge = \begin{bmatrix} 0 & -a_z & a_y \\ a_z & 0 & -a_x \\ -a_y & a_x & 0 \end{bmatrix},
\qquad ([\mathbf{a}]_\times)^\vee = \mathrm{vex}([\mathbf{a}]_\times) = \mathbf{a},
\qquad [\mathbf{a}]_\times\mathbf{b} = \mathbf{a}\times\mathbf{b}$$

[SOLA `equ:skew`; HASH `eq:OVERVIEW_PER_SO3_4/5`; KOK `eq:models-crossproductMatrix`].
Identities: $[\mathbf{a}]_\times^\top=-[\mathbf{a}]_\times$,
$[\mathbf{u}]_\times^2=\mathbf{u}\mathbf{u}^\top-\mathbf{I}$, $[\mathbf{u}]_\times^3=-[\mathbf{u}]_\times$
[SOLA `equ:prop1/2`], $[R\mathbf{v}]_\times = R[\mathbf{v}]_\times R^\top$ [HASH math identities],
anti-symmetric projection $\mathcal{P}_a(\mathbf{B})=\tfrac12(\mathbf{B}-\mathbf{B}^\top)$
[HASH `eq:OVERVIEW_PER_SO3_VEX_7`].

### 3.3 Exp / Log

$$\exp:\ \mathfrak{so}(3)\to SO(3);\qquad
\mathrm{Exp}:\ \mathbb{R}^3\to SO(3),\ \ \mathrm{Exp}(\boldsymbol\phi)=e^{[\boldsymbol\phi]_\times}
\ \text{(Rodrigues, §2.5)};$$
$$\log(\mathbf{R})=[\mathbf{u}\phi]_\times,\qquad \mathrm{Log}(\mathbf{R})=(\log\mathbf{R})^\vee = \phi\,\mathbf{u}$$

[SOLA Quaternion.tex exponential/logarithmic map sections]. For the quaternion the capitalized
maps absorb the half-angle: $\mathrm{Exp}(\boldsymbol\phi)=\exp(\boldsymbol\phi/2)$,
$\mathrm{Log}(\mathbf{q})=2\log(\mathbf{q})$ [SOLA].

### 3.4 Plus/minus operators (manifold calculus)

$$\mathbf{q}_S = \mathbf{q}_R\oplus\boldsymbol\theta = \mathbf{q}_R\otimes\mathrm{Exp}(\boldsymbol\theta),
\qquad
\boldsymbol\theta = \mathbf{q}_S\ominus\mathbf{q}_R = \mathrm{Log}(\mathbf{q}_R^*\otimes\mathbf{q}_S)$$

(and identically with $\mathbf{R}$: $\mathbf{R}\oplus\boldsymbol\theta = \mathbf{R}\,\mathrm{Exp}(\boldsymbol\theta)$,
$\mathbf{R}_S\ominus\mathbf{R}_R=\mathrm{Log}(\mathbf{R}_R^\top\mathbf{R}_S)$)
[SOLA "additive and subtractive operators in SO(3)"]. These are **right (local) operators**,
matching Hamilton local perturbations.

### 3.5 Right Jacobian of SO(3) and inverse

$$\mathbf{J}_r(\boldsymbol\theta) = \mathbf{I} - \frac{1-\cos\|\boldsymbol\theta\|}{\|\boldsymbol\theta\|^2}[\boldsymbol\theta]_\times
+ \frac{\|\boldsymbol\theta\|-\sin\|\boldsymbol\theta\|}{\|\boldsymbol\theta\|^3}[\boldsymbol\theta]_\times^2$$

$$\mathbf{J}_r^{-1}(\boldsymbol\theta) = \mathbf{I} + \frac12[\boldsymbol\theta]_\times
+ \left(\frac{1}{\|\boldsymbol\theta\|^2} - \frac{1+\cos\|\boldsymbol\theta\|}{2\|\boldsymbol\theta\|\sin\|\boldsymbol\theta\|}\right)[\boldsymbol\theta]_\times^2$$

[SOLA "Right Jacobian of SO(3)", citing Chirikjian p.40]. Small angle:
$\mathbf{J}_r \approx \mathbf{I} - \tfrac12[\boldsymbol\theta]_\times$. Left Jacobian:
$\mathbf{J}_l(\boldsymbol\theta)=\mathbf{J}_r(-\boldsymbol\theta)=\mathbf{J}_r(\boldsymbol\theta)^\top$
(standard identity; not stated in corpus explicitly).

First-order BCH-type properties [SOLA `equ:Jr1` ff.]:

$$\mathrm{Exp}(\boldsymbol\theta+\delta\boldsymbol\theta) \approx \mathrm{Exp}(\boldsymbol\theta)\,\mathrm{Exp}(\mathbf{J}_r(\boldsymbol\theta)\delta\boldsymbol\theta),\qquad
\mathrm{Log}\big(\mathrm{Exp}(\boldsymbol\theta)\mathrm{Exp}(\delta\boldsymbol\theta)\big) \approx \boldsymbol\theta + \mathbf{J}_r^{-1}(\boldsymbol\theta)\,\delta\boldsymbol\theta$$

### 3.6 Useful rotation-action Jacobians

$$\frac{\partial(\mathbf{R}\mathbf{a})}{\partial\mathbf{a}} = \mathbf{R};\qquad
\frac{\partial(\mathbf{q}\otimes\mathbf{a}\otimes\mathbf{q}^*)}{\partial\delta\boldsymbol\theta}
= -\mathbf{R}\{\boldsymbol\theta\}[\mathbf{a}]_\times\mathbf{J}_r(\boldsymbol\theta)$$

[SOLA `equ:drot_da`]. Jacobian w.r.t. the (raw, 4-D) quaternion $\mathbf{q}=[w,\mathbf{v}]$:

$$\frac{\partial(\mathbf{q}\otimes\mathbf{a}\otimes\mathbf{q}^*)}{\partial\mathbf{q}}
= 2\big[\,w\mathbf{a}+\mathbf{v}\times\mathbf{a}\ \ \big|\ \ \mathbf{v}^\top\mathbf{a}\,\mathbf{I}_3 + \mathbf{v}\mathbf{a}^\top - \mathbf{a}\mathbf{v}^\top - w[\mathbf{a}]_\times\,\big]\in\mathbb{R}^{3\times4}$$

[SOLA `equ:drot_dq`]. Composition Jacobians (functions $SO(3)\times SO(3)\to SO(3)$, right
perturbations): $\partial(\mathcal{Q}\circ\mathcal{R})/\partial\mathcal{Q} = \mathbf{R}_\phi^\top$,
$\partial(\mathcal{Q}\circ\mathcal{R})/\partial\mathcal{R} = \mathbf{I}$ [SOLA "Jacobians of the
rotation composition"].

### 3.7 Perturbations: local vs global

Local (right): $\tilde{\mathbf{q}}=\mathbf{q}\otimes\mathrm{Exp}(\Delta\boldsymbol\phi_L)$,
$\Delta\boldsymbol\phi_L=\mathrm{Log}(\mathbf{q}^*\otimes\tilde{\mathbf{q}})$.
Global (left): $\tilde{\mathbf{q}}=\mathrm{Exp}(\Delta\boldsymbol\phi_G)\otimes\mathbf{q}$,
$\Delta\boldsymbol\phi_G=\mathrm{Log}(\tilde{\mathbf{q}}\otimes\mathbf{q}^*)$.
First-order: $\Delta\mathbf{q}\approx[1,\ \tfrac12\Delta\boldsymbol\phi]^\top$,
$\Delta\mathbf{R}\approx\mathbf{I}+[\Delta\boldsymbol\phi]_\times$. Conversion:
$\Delta\boldsymbol\phi_G = \mathbf{R}\,\Delta\boldsymbol\phi_L$.
[SOLA "Perturbations, uncertainties, noise" + "Global-to-local relations".]

### 3.8 SO(3) error metric (Hashim's normalized Euclidean distance)

$$\|\mathbf{R}\|_I \triangleq \tfrac14\mathrm{Tr}(\mathbf{I}_3-\mathbf{R}) \in [0,1]
= \sin^2(\alpha/2) = \frac{\|\boldsymbol\rho\|^2}{1+\|\boldsymbol\rho\|^2} = 1-q_w^2$$

$$\mathrm{vex}(\mathcal{P}_a(\mathbf{R})) = 2\cos(\tfrac\alpha2)\sin(\tfrac\alpha2)\,\mathbf{u}
= \frac{2\boldsymbol\rho}{1+\|\boldsymbol\rho\|^2} = 2q_wq_v,\qquad
\|\mathrm{vex}(\mathcal{P}_a(\mathbf{R}))\|^2 = 4(1-\|\mathbf{R}\|_I)\|\mathbf{R}\|_I$$

[HASH `eq:OVERVIEW_SO3_Ecul_Dist`, Lemmas 1, 2, 5, 6.]

---

## 4. Kinematics and integration

### 4.1 Continuous-time kinematics

Local (body) rates $\boldsymbol\omega_L$ (gyro-measured), with $\boldsymbol\Omega(\boldsymbol\omega)\triangleq[\boldsymbol\omega]_R$:

$$\dot{\mathbf{q}} = \tfrac12\,\mathbf{q}\otimes\boldsymbol\omega_L = \tfrac12\boldsymbol\Omega(\boldsymbol\omega_L)\,\mathbf{q},
\qquad
\boldsymbol\Omega(\boldsymbol\omega) = \begin{bmatrix} 0 & -\boldsymbol\omega^\top \\ \boldsymbol\omega & -[\boldsymbol\omega]_\times \end{bmatrix},
\qquad \dot{\mathbf{R}} = \mathbf{R}\,[\boldsymbol\omega_L]_\times$$

[SOLA `equ:qdotLocal`, `equ:Omega`, `equ:Rdot`; HASH `eq:OVERVIEW_Q_dot` ($\dot Q=\frac12\Gamma(\Omega)Q$,
same $\Gamma$), `eq:OVERVIEW_SO3_dyn`; KOK `eq:models-contTimeOri`; SURV "Poisson equation"
$\dot Q=\frac12 Q[\mathbf{w}\times]$; PARW $\dot q=\frac12 q\circ w'$ (body rate) $=\frac12 w\circ q$
(world rate)].

Global (world) rates: $\dot{\mathbf{q}}=\tfrac12\boldsymbol\omega_G\otimes\mathbf{q}$,
$\dot{\mathbf{R}}=[\boldsymbol\omega_G]_\times\mathbf{R}$, with
$\boldsymbol\omega_G = \mathbf{R}\boldsymbol\omega_L$ [SOLA `equ:qdotGlobal`].
Rate recovery: $\boldsymbol\omega_L = 2\mathbf{q}^*\otimes\dot{\mathbf{q}}$,
$[\boldsymbol\omega_L]_\times=\mathbf{R}^\top\dot{\mathbf{R}}$ [SOLA].

Euler-rate (Z-Y'-X'') matrix and gimbal-lock singularity at $\theta=\pm\pi/2$:

$$\begin{bmatrix}\dot\phi\\\dot\theta\\\dot\psi\end{bmatrix}
= \begin{bmatrix}
1 & \sin\phi\tan\theta & \cos\phi\tan\theta\\
0 & \cos\phi & -\sin\phi\\
0 & \sin\phi\sec\theta & \cos\phi\sec\theta
\end{bmatrix}\begin{bmatrix}p\\q\\r\end{bmatrix}$$

[HASH `eq:OVERVIEW_EUL_J`; PARW "321 Euler angle kinematic equation" — identical.]

### 4.2 Zeroth-order quaternion integration (exact for constant ω)

$$\mathbf{q}_{n+1} = \mathbf{q}_n\otimes\mathbf{q}\{\boldsymbol\omega\Delta t\},\qquad
\mathbf{q}\{\boldsymbol\omega\Delta t\}=\begin{bmatrix}
\cos(\|\boldsymbol\omega\|\Delta t/2)\\
\frac{\boldsymbol\omega}{\|\boldsymbol\omega\|}\sin(\|\boldsymbol\omega\|\Delta t/2)\end{bmatrix}$$

- Forward: $\boldsymbol\omega=\boldsymbol\omega_n$ [SOLA `equ:intZeroth`]
- Backward: $\boldsymbol\omega=\boldsymbol\omega_{n+1}$ (typical for real-time processing) [SOLA]
- Midward: $\boldsymbol\omega=\bar{\boldsymbol\omega}=\tfrac12(\boldsymbol\omega_n+\boldsymbol\omega_{n+1})$ [SOLA `equ:intFirstC`, `equ:wbar`]

[Same scheme: MATH §1.5 ($\mathbf{q}_{t+1}=\mathbf{q}_t\otimes\mathrm{Exp}(\boldsymbol\omega_t\Delta t)$);
KOK `eq:models-dynModelOri` ($q_{t+1}=q_t\odot\exp_q(\tfrac{T}{2}\omega_t)$);
HASH `eq:OVERVIEW_SO3_dyn_discrete` for DCM: $R[k{+}1]=R[k]\exp([\Omega[k]]_\times\Delta t)$;
SURV `eq_recur`/`eq_delta_phi` with Taylor-series sin/cos (`eq_delta_ser`).]

### 4.3 First-order integrator (varying rotation axis)

$$\mathbf{q}_{n+1} \approx \mathbf{q}_n\otimes\left(\mathbf{q}\{\bar{\boldsymbol\omega}\Delta t\}
+ \frac{\Delta t^2}{24}\begin{bmatrix}0\\ \boldsymbol\omega_n\times\boldsymbol\omega_{n+1}\end{bmatrix}\right)$$

[SOLA `equ:intFirstB`; equivalent to Trawny's JPL result but Hamilton form, `equ:intFirstA`].
The correction vanishes when the rotation axis is fixed
($\mathbf{q}_{n+1}=\mathbf{q}_n\otimes\mathbf{q}\{\mathbf{u}\,\Delta\theta_n\}$,
$\Delta\theta_n=\int\omega\,dt$ — exact) [SOLA `equ:first_order_constant_axis`].
**Renormalize** after first-order integration (the sum breaks unit norm) [SOLA closing note].

Related strapdown coning correction [SURV `eq_phimequation`, digital algorithm]:
$\boldsymbol\phi_m=\boldsymbol\alpha_m+\boldsymbol\beta_m$, with
$\boldsymbol\alpha_m=\int\boldsymbol\omega\,d\tau$ and the coning increment
$\Delta\boldsymbol\beta_l = \tfrac12(\boldsymbol\alpha_{l-1}+\tfrac16\Delta\boldsymbol\alpha_{l-1})\times\Delta\boldsymbol\alpha_l$;
sculling counterpart for velocity increments
$\delta\mathbf{v}_{Scul,l} = \tfrac12[(\boldsymbol\alpha_{l-1}+\tfrac16\Delta\boldsymbol\alpha_{l-1})\times\Delta\mathbf{v}_l + (\mathbf{v}_{l-1}+\tfrac16\Delta\mathbf{v}_{l-1})\times\Delta\boldsymbol\alpha_l]$
[SURV `eq_jk`]. Rotation-vector kinematics (Bortz):
$\dot{\boldsymbol\phi}=\boldsymbol\omega+\tfrac12\boldsymbol\phi\times\boldsymbol\omega
+\tfrac{1}{\phi^2}\big(1-\tfrac{\phi\sin\phi}{2(1-\cos\phi)}\big)\boldsymbol\phi\times(\boldsymbol\phi\times\boldsymbol\omega)$
[SURV "rotation vector kinematics"].

### 4.4 Runge-Kutta integration

Euler: $\mathbf{x}_{n+1}=\mathbf{x}_n+\Delta t\,f(t_n,\mathbf{x}_n)$;
midpoint: $\mathbf{x}_{n+1}=\mathbf{x}_n+\Delta t\,f(t_n+\tfrac{\Delta t}2,\ \mathbf{x}_n+\tfrac{\Delta t}2 f(t_n,\mathbf{x}_n))$;
RK4:

$$\mathbf{x}_{n+1} = \mathbf{x}_n + \frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4)$$

with the standard stages; general RK $\mathbf{x}_{n+1}=\mathbf{x}_n+\Delta t\sum_i b_ik_i$
[SOLA RungeKutta.tex `sec:NumInt`]. When applied to $\dot{\mathbf{q}}=\frac12\mathbf{q}\otimes\boldsymbol\omega$,
renormalize the quaternion after the step.

### 4.5 Transition matrix computation

Closed-form: $\Phi = e^{\mathbf{A}\Delta t} = \sum_k \frac{1}{k!}\mathbf{A}^k\Delta t^k$
[SOLA ClosedForms.tex `equ:TaylorExp`]. For pure angular-error dynamics
$\dot{\delta\boldsymbol\theta}=-[\boldsymbol\omega]_\times\delta\boldsymbol\theta$:

$$\Phi = e^{-[\boldsymbol\omega]_\times\Delta t} = \mathbf{R}\{\boldsymbol\omega\Delta t\}^\top$$

[SOLA ClosedForms.tex `sec:ClosedFormAngle`, boxed]. Truncations: system-wise
$\Phi\approx\mathbf{I}+\Delta t\,\mathbf{A}$ (finite differences) or
$\mathbf{I}+\mathbf{A}\Delta t+\tfrac12\mathbf{A}^2\Delta t^2$; block-wise truncation keeps
$\Phi_{\theta\theta}=\mathbf{R}\{\boldsymbol\omega\Delta t\}^\top$ exact and truncates
off-diagonal blocks at first significant term ($\Sigma_0=\mathbf{R}\{\boldsymbol\omega\Delta t\}^\top$,
$\Sigma_{n>0}\approx\frac{1}{n!}\mathbf{I}\Delta t^n$)
[SOLA ApproximateTruncated.tex `sec:BlockWiseTruncation`]. Time-varying $\mathbf{A}(t)$:
RK4 on $\dot\Phi = \mathbf{A}(t)\Phi$, $\Phi(t_n|t_n)=\mathbf{I}$, giving
$\Phi = \mathbf{I} + \Delta t\,\mathbf{K}$, $\mathbf{K}=\tfrac16(\mathbf{K}_1+2\mathbf{K}_2+2\mathbf{K}_3+\mathbf{K}_4)$
[SOLA TransitionRK.tex `sec:TranMatRK`].

---

## 5. Error-state Kalman filter (ESKF)

All from [SOLA ErrorState.tex / Noise.tex / ESKF_global.tex] unless noted. State decomposition:
true = nominal ⊕ error, $\mathbf{x}_t = \mathbf{x}\oplus\delta\mathbf{x}$ [SOLA `tab:errorstatevar`].

### 5.1 State, inputs, error state

$$\mathbf{x} = \begin{bmatrix}\mathbf{p}\\\mathbf{v}\\\mathbf{q}\\\mathbf{a}_b\\\boldsymbol\omega_b\\\mathbf{g}\end{bmatrix},\qquad
\delta\mathbf{x} = \begin{bmatrix}\delta\mathbf{p}\\\delta\mathbf{v}\\\delta\boldsymbol\theta\\\delta\mathbf{a}_b\\\delta\boldsymbol\omega_b\\\delta\mathbf{g}\end{bmatrix}\in\mathbb{R}^{18},\qquad
\mathbf{u}_m=\begin{bmatrix}\mathbf{a}_m\\\boldsymbol\omega_m\end{bmatrix}$$

Local (right) error: $\mathbf{q}_t=\mathbf{q}\otimes\delta\mathbf{q}$,
$\delta\mathbf{q}=e^{\delta\boldsymbol\theta/2}\approx[1,\tfrac12\delta\boldsymbol\theta]^\top$;
$\mathbf{R}_t=\mathbf{R}\,\delta\mathbf{R}$, $\delta\mathbf{R}\approx\mathbf{I}+[\delta\boldsymbol\theta]_\times$.

### 5.2 True / nominal / error kinematics (continuous time)

True [SOLA `equ:pos`–`equ:grav`]:

$$\dot{\mathbf{p}}_t=\mathbf{v}_t,\quad
\dot{\mathbf{v}}_t=\mathbf{R}_t(\mathbf{a}_m-\mathbf{a}_{bt}-\mathbf{a}_n)+\mathbf{g}_t,\quad
\dot{\mathbf{q}}_t=\tfrac12\mathbf{q}_t\otimes(\boldsymbol\omega_m-\boldsymbol\omega_{bt}-\boldsymbol\omega_n),\quad
\dot{\mathbf{a}}_{bt}=\mathbf{a}_w,\ \dot{\boldsymbol\omega}_{bt}=\boldsymbol\omega_w,\ \dot{\mathbf{g}}_t=0$$

Nominal: same without noises/biases-walk. Error (local $\delta\boldsymbol\theta$)
[SOLA `equ:efull`]:

$$\dot{\delta\mathbf{p}} = \delta\mathbf{v}$$
$$\dot{\delta\mathbf{v}} = -\mathbf{R}[\mathbf{a}_m-\mathbf{a}_b]_\times\delta\boldsymbol\theta - \mathbf{R}\delta\mathbf{a}_b + \delta\mathbf{g} - \mathbf{R}\mathbf{a}_n$$
$$\dot{\delta\boldsymbol\theta} = -[\boldsymbol\omega_m-\boldsymbol\omega_b]_\times\delta\boldsymbol\theta - \delta\boldsymbol\omega_b - \boldsymbol\omega_n$$
$$\dot{\delta\mathbf{a}}_b = \mathbf{a}_w,\qquad \dot{\delta\boldsymbol\omega}_b = \boldsymbol\omega_w,\qquad \dot{\delta\mathbf{g}}=0$$

(For isotropic accel noise, $\mathbf{R}\mathbf{a}_n$ may be replaced by $\mathbf{a}_n$.)

### 5.3 Discrete nominal propagation

$$\mathbf{p}\gets\mathbf{p}+\mathbf{v}\Delta t+\tfrac12(\mathbf{R}(\mathbf{a}_m-\mathbf{a}_b)+\mathbf{g})\Delta t^2,\quad
\mathbf{v}\gets\mathbf{v}+(\mathbf{R}(\mathbf{a}_m-\mathbf{a}_b)+\mathbf{g})\Delta t,\quad
\mathbf{q}\gets\mathbf{q}\otimes\mathbf{q}\{(\boldsymbol\omega_m-\boldsymbol\omega_b)\Delta t\}$$

[SOLA ErrorState.tex "nominal state kinematics" (discrete)].

### 5.4 Discrete error-state propagation, $F_x$, $F_i$, $Q_i$

$$\hat{\delta\mathbf{x}}\gets\mathbf{F_x}\hat{\delta\mathbf{x}}\ (=0\text{, skip in code}),\qquad
\mathbf{P}\gets\mathbf{F_x}\mathbf{P}\mathbf{F_x}^\top + \mathbf{F_i}\mathbf{Q_i}\mathbf{F_i}^\top$$

[SOLA `equ:errorMeanPred`, `equ:errorCovPred`]. With local error, Euler form
[SOLA `equ:Fx_local_euler`]:

$$\mathbf{F_x}=\begin{bmatrix}
\mathbf{I}&\mathbf{I}\Delta t&0&0&0&0\\
0&\mathbf{I}&-\mathbf{R}[\mathbf{a}_m{-}\mathbf{a}_b]_\times\Delta t&-\mathbf{R}\Delta t&0&\mathbf{I}\Delta t\\
0&0&\mathbf{R}^\top\{(\boldsymbol\omega_m{-}\boldsymbol\omega_b)\Delta t\}&0&-\mathbf{I}\Delta t&0\\
0&0&0&\mathbf{I}&0&0\\
0&0&0&0&\mathbf{I}&0\\
0&0&0&0&0&\mathbf{I}\end{bmatrix}$$

$$\mathbf{F_i}=\begin{bmatrix}0&0&0&0\\\mathbf{I}&0&0&0\\0&\mathbf{I}&0&0\\0&0&\mathbf{I}&0\\0&0&0&\mathbf{I}\\0&0&0&0\end{bmatrix},\qquad
\mathbf{Q_i}=\mathrm{diag}\big(\sigma_{\tilde a}^2\Delta t^2\mathbf{I},\ \sigma_{\tilde\omega}^2\Delta t^2\mathbf{I},\ \sigma_{a_w}^2\Delta t\,\mathbf{I},\ \sigma_{\omega_w}^2\Delta t\,\mathbf{I}\big)$$

Noise-impulse covariances: $\mathbf{V_i}=\sigma_{\tilde a_n}^2\Delta t^2\mathbf{I}$ [m²/s²],
$\Theta_\mathbf{i}=\sigma_{\tilde\omega_n}^2\Delta t^2\mathbf{I}$ [rad²],
$\mathbf{A_i}=\sigma_{a_w}^2\Delta t\,\mathbf{I}$ [m²/s⁴],
$\boldsymbol\Omega_\mathbf{i}=\sigma_{\omega_w}^2\Delta t\,\mathbf{I}$ [rad²/s²]
[SOLA ErrorState.tex + Noise.tex `sec:pertImpulses`].

General continuous→discrete noise rule [SOLA Noise.tex `equ:NoisePertCovUpdate`]:

$$\mathbf{P}_{n+1} = e^{\mathbf{A}\Delta t}\mathbf{P}_n e^{\mathbf{A}^\top\Delta t}
+ \Delta t^2\,\mathbf{B}\mathbf{U}^c\mathbf{B}^\top + \Delta t\,\mathbf{C}\mathbf{W}^c\mathbf{C}^\top$$

— sampled measurement (control) noise integrates **quadratically** ($\Delta t^2$), continuous
random-walk perturbations integrate **linearly** ($\Delta t$) [SOLA Noise.tex `tab:IntEffects`].

### 5.5 Measurement update (correction)

$$\mathbf{K}=\mathbf{P}\mathbf{H}^\top(\mathbf{H}\mathbf{P}\mathbf{H}^\top+\mathbf{V})^{-1},\qquad
\hat{\delta\mathbf{x}}\gets\mathbf{K}(\mathbf{y}-h(\hat{\mathbf{x}}_t)),\qquad
\mathbf{P}\gets(\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}$$

(Joseph form $(\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}(\mathbf{I}-\mathbf{K}\mathbf{H})^\top+\mathbf{K}\mathbf{V}\mathbf{K}^\top$
recommended for numerical stability) [SOLA ErrorState.tex §"Observation of the error state"].
Jacobian chain rule: $\mathbf{H}=\mathbf{H_x}\,\mathbf{X}_{\delta\mathbf{x}}$, where the only
non-identity block is

$$\mathbf{Q}_{\delta\boldsymbol\theta} = \frac{\partial(\mathbf{q}\otimes\delta\mathbf{q})}{\partial\delta\boldsymbol\theta}
= [\mathbf{q}]_L\,\frac12\begin{bmatrix}0&0&0\\1&0&0\\0&1&0\\0&0&1\end{bmatrix}
= \frac12\begin{bmatrix}-q_x&-q_y&-q_z\\ q_w&-q_z&q_y\\ q_z&q_w&-q_x\\ -q_y&q_x&q_w\end{bmatrix}$$

[SOLA ErrorState.tex "Jacobian computation for the filter correction"].

### 5.6 Injection and reset

Injection: $\mathbf{p}\gets\mathbf{p}+\hat{\delta\mathbf{p}}$, …,
$\mathbf{q}\gets\mathbf{q}\otimes\mathbf{q}\{\hat{\delta\boldsymbol\theta}\}$
[SOLA `equ:errorInjection`, `equ:quatErrorInjection`]. Reset:
$\hat{\delta\mathbf{x}}\gets 0$, $\mathbf{P}\gets\mathbf{G}\mathbf{P}\mathbf{G}^\top$, with

$$\mathbf{G} = \mathrm{blkdiag}\Big(\mathbf{I}_6,\ \ \mathbf{I}-\big[\tfrac12\hat{\delta\boldsymbol\theta}\big]_\times,\ \ \mathbf{I}_9\Big)$$

(often approximated $\mathbf{G}=\mathbf{I}$) [SOLA ErrorState.tex §"ESKF reset"].

### 5.7 Global-error ESKF (left error) — differences

With $\mathbf{q}_t = \delta\mathbf{q}\otimes\mathbf{q}$ [SOLA ESKF_global.tex `equ:efullglobal`,
`tab:local_to_global`]:

| item | local error | global error |
|---|---|---|
| $\dot{\delta\boldsymbol\theta}$ | $-[\boldsymbol\omega_m{-}\boldsymbol\omega_b]_\times\delta\boldsymbol\theta - \delta\boldsymbol\omega_b - \boldsymbol\omega_n$ | $-\mathbf{R}\delta\boldsymbol\omega_b - \mathbf{R}\boldsymbol\omega_n$ |
| $\partial\delta\mathbf{v}^+/\partial\delta\boldsymbol\theta$ | $-\mathbf{R}[\mathbf{a}_m{-}\mathbf{a}_b]_\times\Delta t$ | $-[\mathbf{R}(\mathbf{a}_m{-}\mathbf{a}_b)]_\times\Delta t$ |
| $\partial\delta\boldsymbol\theta^+/\partial\delta\boldsymbol\theta$ | $\mathbf{R}^\top\{(\boldsymbol\omega_m{-}\boldsymbol\omega_b)\Delta t\}$ | $\mathbf{I}$ |
| $\partial\delta\boldsymbol\theta^+/\partial\delta\boldsymbol\omega_b$ | $-\mathbf{I}\Delta t$ | $-\mathbf{R}\Delta t$ |
| injection | $\mathbf{q}\gets\mathbf{q}\otimes\mathbf{q}\{\hat{\delta\boldsymbol\theta}\}$ | $\mathbf{q}\gets\mathbf{q}\{\hat{\delta\boldsymbol\theta}\}\otimes\mathbf{q}$ |
| reset Jacobian | $\mathbf{I}-[\tfrac12\hat{\delta\boldsymbol\theta}]_\times$ | $\mathbf{I}+[\tfrac12\hat{\delta\boldsymbol\theta}]_\times$ |
| $\mathbf{Q}_{\delta\boldsymbol\theta}$ | $[\mathbf{q}]_L\tfrac12\,[\mathbf{e}]$ | $[\mathbf{q}]_R\tfrac12\,[\mathbf{e}]$ |

### 5.8 Kok-Hol-Schön EKF / smoothing variants

Quaternion-EKF and orientation-deviation ("multiplicative"/linearization-point) EKF:
$q^{nb}_t = \exp_q(\tfrac{\rho^n_t}{2})\odot\tilde q^{nb}_t$ — **global/left deviation**, unlike
SOLA's default local/right error [KOK `eq:models-oriDev`]. MAP smoothing:
$\min\sum\|e_{\omega,t}\|^2_{\Sigma_\omega^{-1}}+\sum(\|e_{a,t}\|^2_{\Sigma_a^{-1}}+\|e_{m,t}\|^2_{\Sigma_m^{-1}})$
[KOK `eq:oriEst-oriSmoothing`]; standard EKF time/measurement updates [KOK ch.4].
Discrete-time error-state KF in NED with tilt errors $\psi^n$ also in [SURV `eq_statevector`,
`eq_disc2`, `eq_delta_anmoddisc`] (13-state: 3 gyro bias, 1 accel bias, 3 tilt, 3 vel, 3 pos).

---

## 6. Vector-observation attitude determination

The corpus covers these only lightly; equations from [KOK §3.6] plus standard formulations
implied by it. (TRIAD/QUEST/Davenport derivations should additionally be traced to Markley &
Crassidis / Shuster when implementing `determination/`.)

### 6.1 Wahba problem

$$\min_{\mathbf{R}\in SO(3)}\ \tfrac12\sum_i w_i\,\|\mathbf{v}_i^n - \mathbf{R}\,\mathbf{v}_i^b\|^2$$

Quaternion form used by [KOK `eq:models-oriAccMag`] with the two vector pairs (gravity, magnetic
field):

$$\hat q^{nb} = \arg\min_{\|q\|=1}\ \|\bar{\hat g}^n - q\odot\bar{\hat g}^b\odot q^c\|_2^2
+ \|\bar{\hat m}^n - q\odot\bar{\hat m}^b\odot q^c\|_2^2$$

### 6.2 Davenport q-method (eigenvalue form)

$$\hat q = \arg\max_{\|q\|=1} q^\top \mathbf{A}\, q \quad\Rightarrow\quad
\mathbf{A}\hat q = \lambda_{\max}\hat q$$

[KOK `eq:models-oriAccMagReform` — $\mathbf{A}$ built from quaternion left/right product
matrices of the paired vectors; this is exactly the Davenport K-matrix construction].
QUEST = characteristic-polynomial solution of the same eigenproblem; SVD method = direct
solution $\mathbf{R}=\mathbf{U}\,\mathrm{diag}(1,1,\det(\mathbf{U}\mathbf{V}^\top))\,\mathbf{V}^\top$
of Wahba via $\mathbf{B}=\sum w_i \mathbf{v}_i^n (\mathbf{v}_i^b)^\top = \mathbf{U}\mathbf{S}\mathbf{V}^\top$
(standard results; not derived in the corpus).

### 6.3 TRIAD-style orthogonalized pair (initialization)

$$\hat g^n = [0,0,1]^\top,\quad \hat g^b = \frac{y_{a,1}}{\|y_{a,1}\|},\qquad
\hat m^n = [1,0,0]^\top,\quad \hat m^b = \hat g^b\times\Big(\frac{y_{m,1}}{\|y_{m,1}\|}\times\hat g^b\Big)$$

[KOK `eq:models-oriAccMagVectors`] — the magnetometer reading is projected orthogonal to
gravity before solving, i.e. the heading reference is decoupled from tilt. (Note KOK's $\hat g^n
= +e_3$ with accelerometer measuring $-R^{bn}g^n$; sign bookkeeping depends on the gravity-sign
convention, see §7.)

---

## 7. Tilt and heading from accelerometer/magnetometer

### 7.1 Accelerometer measurement of gravity (quasi-static)

$$\mathbf{y}_a = -\mathbf{R}^{bn}\mathbf{g}^n + \delta_a + e_a
\qquad\text{(zero-acceleration model)}$$

[KOK `eq:models-accMeasModelZeroAcc`]. In ENU with $\mathbf{g}^n=[0,0,-g]^\top$ a static sensor
reads $\mathbf{y}_a \approx +g\,\mathbf{R}^{bn}\hat{\mathbf{z}}$, i.e. the body-frame "up"
direction; equivalently the predicted body gravity direction is
$\hat{\mathbf{g}}_b = \mathbf{R}(\mathbf{q})^\top\hat{\mathbf{z}}$ [MATH §5].

### 7.2 Roll/pitch from accelerometer (NED, Z-Y'-X'')

With $f^b=[f_x,f_y,f_z]^\top$ the static specific force in body axes (NED, $g$ down):

$$\phi = \mathrm{atan2}(f_y,\ f_z)\ \big(\text{tilt-error form: } \mathrm{atan2}(-f_y,-f_z)\ \text{dep. sign conv.}\big),
\qquad
\theta = \mathrm{atan2}\big({-f_x},\ \sqrt{f_y^2+f_z^2}\,\big)$$

[SURV: roll/pitch extracted via DCM elements $\phi=\mathrm{atan2}(c_{32},c_{33})$,
$\theta=\mathrm{atan2}(-c_{31},\sqrt{1-c_{31}^2})$ with $c_{3j}$ reconstructed from the
normalized accelerometer reading; KOK §3.6 equivalently through the gravity direction. Sign of
$f$ vs $g$ depends on whether the accelerometer output is modeled as specific force
($-R^{bn}g$) — qnav `sensors/` must fix the sign at the model level.]

### 7.3 Tilt-compensated magnetic heading

De-rotate the body magnetometer reading by roll and pitch, then:

$$\begin{aligned}
m_x' &= m_x\cos\theta + m_y\sin\phi\sin\theta + m_z\cos\phi\sin\theta\\
m_y' &= m_y\cos\phi - m_z\sin\phi\\
\psi_{mag} &= \mathrm{atan2}(-m_y',\ m_x')\quad(\text{NED, heading from magnetic north})
\end{aligned}$$

True heading: $\psi = \psi_{mag} + D$ with $D$ the local magnetic declination. Earth-field
model with dip (inclination) angle $\delta$:
$m^n = [\cos\delta,\ 0,\ \sin\delta]^\top$ (unit norm, NED-style, north/vertical components
only) and $y_m = R^{bn}m^n + e_m$ [KOK `eq:models-localMagField`, `eq:models-magMeasModel`].
[SURV treats heading via the DCM elements $\psi=\mathrm{atan2}(c_{21},c_{11})$; the explicit
tilt-compensation identity above follows from §2.6 and is the standard implementation form for
qnav `heading/`.]

### 7.4 Yaw-invariant tilt metric (qnav metric baseline)

$$\mathbf{q}_{err} = \hat{\mathbf{q}}\otimes\mathbf{q}_{gt}^{-1},\qquad
e_\alpha = 2\arccos\Big(\min\big(1,\ \sqrt{w_{err}^2+z_{err}^2}\big)\Big)$$

Pure-yaw errors give $e_\alpha=0$; pure pitch by $\theta$ gives $e_\alpha=\theta$ exactly
[MATH §2 "RIANN tilt metric"]. **Note:** with world-z the yaw axis this metric assumes ENU
(z-up); in NED the yaw-invariant combination is the same ($w^2+z^2$) since yaw is still about
world z. Gravity-alignment cosine loss (bounded gradient, preferred over arccos near
convergence): $\ell_{grav} = 1-\hat{\mathbf{a}}\cdot\mathbf{R}(\hat{\mathbf{q}})^\top\hat{\mathbf{z}}$
[MATH §5].

---

## 8. Complementary / Mahony / Madgwick filters

### 8.1 Complementary filter

Frequency-domain split: low-pass the accelerometer/magnetometer attitude, high-pass the
gyro-integrated attitude, combined with complementary gains summing to 1
[KOK ch.4 "complementary filter"; SURV "Other Attitude Filters" intro]. Classical scalar form
per tilt axis: $\hat\theta = \alpha(\hat\theta + \omega\Delta t) + (1-\alpha)\,\theta_{acc}$.

### 8.2 Mahony filter (explicit complementary filter, PI feedback)

Error from vector cross products (body frame) [SURV `eq_hjl` and around]:

$$\mathbf{e}^b = \sum_i k_i\, \hat{\mathbf{v}}_i^b \times (\hat{C}^b_{ned}\,\mathbf{v}_i^{ref}),\qquad
\text{e.g. } \mathbf{e}_g^b = \frac{\mathbf{g}^b_{ref}}{\|\mathbf{g}^b_{ref}\|}\times\big(\hat C^b_{ned}[0,0,1]^\top\big),\quad
\mathbf{e}_\psi^b \text{ analogous from heading reference}$$

PI correction of the gyro before integration:

$$\boldsymbol\omega_{corr} = \boldsymbol\omega_m + k_P\,\mathbf{e}^b + k_I\!\int\mathbf{e}^b\,dt,
\qquad \dot{\hat{\mathbf{q}}} = \tfrac12\,\hat{\mathbf{q}}\otimes\boldsymbol\omega_{corr}$$

with the integral term acting as the gyro-bias estimate
[SURV §"Mahony Filter (feedback controller, PI compensator)"]. Mahony's gravity reference also
uses the velocity-aided form $\mathbf{g}^b_{ref}=\boldsymbol\omega^b_{gyro}\times\mathbf{V}^b_{ref}-\mathbf{f}^b_{accel}$
[SURV `eq_spec_force`].

### 8.3 Madgwick filter (gradient descent)

Objective (align rotated reference $d$ with measurement $s$) [SURV `eq_madg3/4`]:

$$\min_{\hat{\mathbf{q}}}\ \mathbf{f}(\hat{\mathbf{q}},\mathbf{d},\mathbf{s})
= \hat{\mathbf{q}}^*\otimes\mathbf{d}\otimes\hat{\mathbf{q}} - \mathbf{s}$$

Gradient step and fusion:

$$\nabla f = J^\top(\hat{\mathbf{q}})\,\mathbf{f},\qquad
\dot{\hat{\mathbf{q}}} = \tfrac12\hat{\mathbf{q}}\otimes\boldsymbol\omega_m
\;-\;\beta\,\frac{\nabla f}{\|\nabla f\|},\qquad
\hat{\mathbf{q}}_t = \hat{\mathbf{q}}_{t-1} + \dot{\hat{\mathbf{q}}}_t\,\Delta t,\ \ \text{normalize}$$

[SURV `eq_madg`, `eq_madg2`; gain $\beta$ trades gyro trust vs accel/mag trust.]

---

## 9. Sensor error models

### 9.1 Gyroscope

$$\mathbf{y}_\omega = \boldsymbol\omega^b_{nb} + \delta^b_\omega + \mathbf{e}^b_\omega,
\qquad \mathbf{e}_\omega\sim\mathcal{N}(0,\Sigma_\omega),\qquad
\delta_{\omega,t+1} = \delta_{\omega,t} + e_{\delta_\omega,t}\ \text{(random walk)}$$

[KOK `eq:models-gyrMeasModel`; SOLA `equ:gyroModel`
$\boldsymbol\omega_m=\boldsymbol\omega_t+\boldsymbol\omega_{bt}+\boldsymbol\omega_n$,
$\dot{\boldsymbol\omega}_{bt}=\boldsymbol\omega_w$]. Full model includes the earth rate:
$\omega^b_{ib} = R^{bn}(\omega^n_{ie}+\omega^n_{en}) + \omega^b_{nb}$,
$\|\omega_{ie}\| \approx 7.29\times10^{-5}$ rad/s — negligible for MEMS, not for high-end IMUs
[KOK ch.2; SOLA ErrorState.tex footnote with the same caution, $\omega_E = 15^\circ/h$].

### 9.2 Accelerometer

$$\mathbf{y}_a = \mathbf{R}^{bn}(\mathbf{a}^n_{nn}-\mathbf{g}^n) + \delta_a^b + \mathbf{e}_a^b,
\qquad
a^n_{ii} = a^n_{nn} + 2\,\omega^n_{ie}\times v^n + \omega^n_{ie}\times\omega^n_{ie}\times p^n$$

[KOK `eq:models-accMeasModel`, `eq:sensors-aii-ann` — Coriolis (~1e-4 m/s² for human motion)
and centrifugal (≤0.034 m/s², absorbed into local gravity) terms; SOLA equivalent
$\mathbf{a}_m=\mathbf{R}_t^\top(\mathbf{a}_t-\mathbf{g}_t)+\mathbf{a}_{bt}+\mathbf{a}_n$].

### 9.3 Magnetometer

$$\mathbf{y}_m = \mathbf{R}^{bn}\,\mathbf{m}^n + \mathbf{e}_m,\qquad
\mathbf{m}^n = [\cos\delta,\ 0,\ \sin\delta]^\top$$

[KOK `eq:models-magMeasModel`, `eq:models-localMagField`]. Hard/soft-iron calibration
(ellipsoid fit) discussed in [KOK ch.5].

### 9.4 Allan variance

$$\sigma_A(T_c)=\sqrt{\tfrac12\big\langle(\bar y_{k+1}-\bar y_k)^2\big\rangle}$$

white-noise region: slope −1/2 in log-log ($\sigma_A = \sigma/\sqrt{n}$); bias-instability
flattening at longer cluster times [KOK ch.2]. Continuous→discrete noise mapping for filters:
white measurement noise → $\sigma^2\Delta t^2$ impulses, random-walk drivers → $\sigma^2\Delta t$
impulses [SOLA Noise.tex, §5.4 above]; datasheet units
$\sigma_{\tilde a}\,[m/s^2]$, $\sigma_{\tilde\omega}\,[rad/s]$,
$\sigma_{a_w}\,[m/s^2\sqrt{s}]$, $\sigma_{\omega_w}\,[rad/s\sqrt{s}]$ [SOLA Noise.tex].

### 9.5 Mounting / virtual-rotation equivariance (data augmentation)

$$\mathbf{a}' = \mathbf{R}(\mathbf{q}_{rand})\mathbf{a},\quad
\boldsymbol\omega' = \mathbf{R}(\mathbf{q}_{rand})\boldsymbol\omega,\quad
\mathbf{q}_{gt}' = \mathbf{q}_{gt}\otimes\mathbf{q}_{rand}^{-1}$$

with sensor-to-world quaternion $\mathbf{q}_{sensor}=\mathbf{q}_{body}\otimes\mathbf{q}_{rand}^{-1}$
[MATH §6]. Useful for `simulation/` and for validating frame bookkeeping.

---

## 10. Rigid-body rotational dynamics (Euler's equations)

Body-frame angular momentum $\mathbf{h}=\mathbf{I}\boldsymbol\omega$; Euler's rotational
equations of motion:

$$\mathbf{I}\,\dot{\boldsymbol\omega} + \boldsymbol\omega\times(\mathbf{I}\boldsymbol\omega) = \boldsymbol\tau$$

principal-axis component form:

$$I_1\dot\omega_1 = (I_2-I_3)\,\omega_2\omega_3 + \tau_1,\qquad
I_2\dot\omega_2 = (I_3-I_1)\,\omega_3\omega_1 + \tau_2,\qquad
I_3\dot\omega_3 = (I_1-I_2)\,\omega_1\omega_2 + \tau_3$$

[SATD — satellite dynamics notes (attitude dynamics chapter); the kinematic side
$\dot{\mathbf{R}}=\mathbf{R}[\boldsymbol\omega]_\times$, $\dot{\mathbf{q}}=\tfrac12\mathbf{q}\otimes\boldsymbol\omega$
matches §4.1 of this catalog]. Torque-free special cases (axisymmetric body precession) and
gravity-gradient torques are covered in [SATD] and `__data/Lesson2.pdf` (spacecraft attitude
lecture); use them for `simulation/` truth models.

---

## 11. Attitude error metrics

| metric | definition | range | source |
|---|---|---|---|
| Geodesic angle | $\theta_{err} = \|\mathrm{Log}(\mathbf{q}_1^*\otimes\mathbf{q}_2)\| = 2\arccos\lvert\mathbf{q}_1^\top\mathbf{q}_2\rvert$ | $[0,\pi]$ | [SOLA $\ominus$ operator]; abs() enforces sign invariance [MATH §1.6] |
| DCM geodesic | $\theta_{err}=\arccos\big(\tfrac{\mathrm{Tr}(\mathbf{R}_1^\top\mathbf{R}_2)-1}{2}\big)$ | $[0,\pi]$ | [SOLA Log map; HASH `eq:OVERVIEW_att_ang_alpha`] |
| Normalized Euclidean | $\|\tilde{\mathbf{R}}\|_I=\tfrac14\mathrm{Tr}(\mathbf{I}-\tilde{\mathbf{R}})=\sin^2(\theta_{err}/2)=1-q_{w,err}^2$ | $[0,1]$ | [HASH `eq:OVERVIEW_SO3_Ecul_Dist`, Lemmas 2,6] |
| Quaternion inner-product distance | $d=1-\lvert\mathbf{q}_1^\top\mathbf{q}_2\rvert$ | $[0,1]$ | sign-invariant per [MATH §1.6] |
| Yaw-invariant tilt error | $e_\alpha = 2\arccos(\min(1,\sqrt{w_{err}^2+z_{err}^2}))$ | $[0,\pi]$ | [MATH §2] |
| Gravity cosine loss | $1-\hat{\mathbf a}\cdot\mathbf{R}(\hat{\mathbf q})^\top\hat{\mathbf z}\approx\theta^2/2$ | $[0,2]$ | [MATH §5] |

All quaternion metrics must satisfy $d(\mathbf{q}_1,\mathbf{q}_2)=d(\mathbf{q}_1,-\mathbf{q}_2)$
(double cover) [MATH §1.6].

---

## Known inter-source discrepancies (summary)

1. **Axis-angle quaternion sign** — KOK uses $q=[\cos\frac\alpha2,\,-n\sin\frac\alpha2]$ and
   Rodrigues with $-\sin\alpha[n\times]$; SOLA/MATH/HASH/SURV/PARW use $+$. Same physics,
   opposite angle-sign bookkeeping. **qnav: positive sign (SOLA form).**
2. **Error-state side** — SOLA default: local/right error $\mathbf{q}_t=\mathbf{q}\otimes\delta\mathbf{q}$;
   KOK orientation deviation and SOLA ESKF_global: global/left $\,q=\exp(\rho/2)\odot\tilde q$.
   Both fully documented in §5.7; **qnav `filters/eskf` exposes both, default local.**
3. **Quaternion log formula** — KOK `arccos` form vs SOLA `atan2` form. **qnav: atan2.**
4. **Euler-angle direction** — HASH composes body→inertial $R_zR_yR_x$; SURV/KOK write the
   nav→body (transposed) matrix with the same Z-Y-X sequence. PARW additionally analyzes the
   1-2-3 (XYZ) sequence. **qnav: intrinsic Z-Y'-X'' producing $\mathbf{R}_{nb}$ (body→nav).**
5. **Gravity sign / frame** — MATH baseline is ENU ($+z$ up, $\hat g^n=+\hat z$ as "up");
   SURV is NED (z down); KOK uses a local n-frame with either choice noted. **qnav: NED and ENU
   both first-class; gravity vector sign is owned by `frames/`, never hard-coded in filters.**
6. **Rotation-rate frame** — gyro rates are body/local in all sources; SOLA additionally gives
   the global-rate forms ($\dot q=\frac12\omega_G\otimes q$). PARW writes both and is the only
   source using the world-rate form as primary in places — take care when porting.
