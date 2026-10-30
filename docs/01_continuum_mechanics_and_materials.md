# 01. Continuum Mechanics & Constitutive Material Models

This document covers the theoretical continuum mechanics formulations, kinematics of deformation, stress representations, constitutive material models, and their underlying **physical meanings** implemented in `pSierra`.

---

## 1. Kinematics of Deformation

### 1.1 Small-Strain Kinematics (Infinitesimal Elasticity)
For small displacements $\boldsymbol{u}(\boldsymbol{x})$ where $\|\nabla \boldsymbol{u}\| \ll 1$, the linearized infinitesimal strain tensor $\boldsymbol{\varepsilon}$ is the symmetric gradient of displacement:
$$\boldsymbol{\varepsilon} = \frac{1}{2} \left( \nabla \boldsymbol{u} + (\nabla \boldsymbol{u})^T \right), \quad \varepsilon_{ij} = \frac{1}{2} \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)$$

In 3D solid mechanics, symmetry yields 6 independent strain components. In `pSierra`, we employ engineering Voigt notation:
$$\boldsymbol{\varepsilon}_{voigt} = \begin{bmatrix} \varepsilon_{xx} & \varepsilon_{yy} & \varepsilon_{zz} & \gamma_{xy} & \gamma_{yz} & \gamma_{zx} \end{bmatrix}^T$$
where engineering shear strains are defined as $\gamma_{ij} = 2 \varepsilon_{ij} = \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}$.

#### Physical Meaning & Physical Limitations of Small-Strain Theory
- **Physical Meaning**: The diagonal terms ($\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{zz}$) measure the fractional change in length (relative stretch or elongation) of material fibers aligned with the coordinate axes: $\varepsilon_{xx} = \frac{\Delta L_x}{L_x}$. The off-diagonal terms ($\gamma_{xy}, \gamma_{yz}, \gamma_{zx}$) measure the change in the angle between originally orthogonal material fibers (angular distortion in radians).
- **Why Small-Strain Fails Under Large Rotations**: Small-strain theory assumes displacements and rotations are infinitesimal. Consider a rigid body that undergoes a pure $90^\circ$ rigid rotation with zero physical deformation. The displacement gradient contains $\frac{\partial u_x}{\partial x} = -1$, which falsely computes an artificial normal strain $\varepsilon_{xx} = -1$ (i.e. $100\%$ compression). Under finite rotations, linear small-strain formulations produce non-physical "phantom" strains, causing elements to artificially expand or blow up in volume.

---

### 1.2 Finite-Strain Kinematics (Large Deformations)
Let $\boldsymbol{X} \in \Omega_0$ denote the material coordinates in the undeformed reference configuration, and let $\boldsymbol{x} \in \Omega_t$ denote the spatial coordinates in the deformed current configuration:
$$\boldsymbol{x} = \boldsymbol{\chi}(\boldsymbol{X}, t) = \boldsymbol{X} + \boldsymbol{u}(\boldsymbol{X}, t)$$

The **Deformation Gradient Tensor** $\boldsymbol{F}$ maps infinitesimal reference line elements $d\boldsymbol{X}$ to deformed line elements $d\boldsymbol{x}$:
$$\boldsymbol{F} = \frac{\partial \boldsymbol{x}}{\partial \boldsymbol{X}} = \boldsymbol{I} + \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}} = \boldsymbol{I} + \nabla_{\boldsymbol{X}} \boldsymbol{u}$$

The Jacobian determinant $J = \det(\boldsymbol{F}) > 0$ represents the local volume ratio:
$$J = \frac{d\Omega_t}{d\Omega_0} = \det(\boldsymbol{F})$$

The **Right Cauchy-Green Deformation Tensor** $\boldsymbol{C}$ and **Green-Lagrange Strain Tensor** $\boldsymbol{E}$ operate in the reference configuration:
$$\boldsymbol{C} = \boldsymbol{F}^T \boldsymbol{F}$$
$$\boldsymbol{E} = \frac{1}{2} (\boldsymbol{C} - \boldsymbol{I}) = \frac{1}{2} \left( \boldsymbol{F}^T \boldsymbol{F} - \boldsymbol{I} \right)$$
In components, $E_{IJ} = \frac{1}{2} \left( \frac{\partial u_I}{\partial X_J} + \frac{\partial u_J}{\partial X_I} + \sum_K \frac{\partial u_K}{\partial X_I} \frac{\partial u_K}{\partial X_J} \right)$, containing both linear and quadratic displacement gradient terms.

#### Physical Meaning & Geometric Intuition of Finite-Strain Tensors
- **Deformation Gradient $\boldsymbol{F}$**: Describes how an infinitesimal sphere in the reference configuration deforms into an ellipsoid in the spatial configuration (the *Cauchy deformation ellipsoid*). Through polar decomposition $\boldsymbol{F} = \boldsymbol{R}\boldsymbol{U}$, $\boldsymbol{F}$ cleanly separates pure material stretch $\boldsymbol{U}$ from rigid-body rotation $\boldsymbol{R}$.
- **Volume Ratio $J = \det(\boldsymbol{F})$**:
  - $J = 1$: Isochoric (volume-preserving) deformation, typical of incompressible rubber, water, and soft biological tissue.
  - $J > 1$: Local volumetric expansion (dilation).
  - $0 < J < 1$: Local volumetric compaction.
  - $J \le 0$: Non-physical interpenetration or element inversion (the material has folded inside out, causing infinite mathematical failure).
- **Right Cauchy-Green Tensor $\boldsymbol{C}$ and Green-Lagrange Strain $\boldsymbol{E}$**:
  Under a pure rigid-body rotation $\boldsymbol{F} = \boldsymbol{R}$ (where $\boldsymbol{R}^T \boldsymbol{R} = \boldsymbol{I}$):
  $$\boldsymbol{C} = \boldsymbol{R}^T \boldsymbol{R} = \boldsymbol{I} \implies \boldsymbol{E} = \frac{1}{2}(\boldsymbol{I} - \boldsymbol{I}) = \boldsymbol{0}$$
  $\boldsymbol{E}$ is **strictly rotation-invariant**: it isolates genuine physical stretching of material fibers and remains identically zero during arbitrary finite rigid-body motions.

---

## 2. Stress Tensors in Continuum Mechanics

`pSierra` works across both spatial and material configurations using three fundamental stress measures:

```mermaid
flowchart LR
    S["Second Piola-Kirchhoff S<br/>(Reference Domain &Omega;0)"]
    P["First Piola-Kirchhoff P<br/>(Two-point tensor)"]
    sigma["Cauchy Stress &sigma;<br/>(Current Domain &Omega;t)"]

    S -- "Push-forward by F" --> P
    P -- "Normalized by 1/J" --> sigma
    sigma -- "Pull-back: J F^-1 &sigma; F^-T" --> S
```

1. **Cauchy Stress Tensor ($\boldsymbol{\sigma}$)**: True physical force per unit deformed area in the current spatial configuration $\Omega_t$.
2. **First Piola-Kirchhoff Stress Tensor ($\boldsymbol{P}$)**: Current force per unit undeformed reference area:
   $$\boldsymbol{P} = J \boldsymbol{\sigma} \boldsymbol{F}^{-T}$$
3. **Second Piola-Kirchhoff Stress Tensor ($\boldsymbol{S}$)**: Completely pulled-back symmetric reference stress tensor:
   $$\boldsymbol{S} = \boldsymbol{F}^{-1} \boldsymbol{P} = J \boldsymbol{F}^{-1} \boldsymbol{\sigma} \boldsymbol{F}^{-T}$$

The Second Piola-Kirchhoff stress $\boldsymbol{S}$ is energetically conjugate to the Green-Lagrange strain $\boldsymbol{E}$:
$$\delta W_{int} = \int_{\Omega_0} \boldsymbol{S} : \delta \boldsymbol{E} \, d\Omega_0$$

#### Physical Meaning & Experimental Interpretation of Stress Tensors
- **Cauchy Stress $\boldsymbol{\sigma}$ ("True Stress")**:
  - **Physical Meaning**: This is the real, physically measurable force per unit instantaneous deformed area. If you place a pressure sensor or miniature load cell inside a deforming tissue, it directly measures the Cauchy traction vector $\boldsymbol{t} = \boldsymbol{\sigma} \boldsymbol{n}$.
  - **Symmetry**: Symmetric ($\boldsymbol{\sigma} = \boldsymbol{\sigma}^T$) by conservation of angular momentum (Boltzmann's axiom) in the absence of body couples.
- **First Piola-Kirchhoff Stress $\boldsymbol{P}$ ("Nominal / Engineering Stress")**:
  - **Physical Meaning**: This is what tensile testing machines (e.g. Instron) record during experimental testing: the total current pull force $F$ divided by the initial, undeformed cross-sectional area $A_0$.
  - **Asymmetry**: Generally non-symmetric because the force vector lives in the current deformed configuration, whereas the surface area normal $\boldsymbol{N}$ lives in the undeformed reference configuration.
- **Second Piola-Kirchhoff Stress $\boldsymbol{S}$ ("Reference Mathematical Stress")**:
  - **Physical Meaning**: $\boldsymbol{S}$ is completely pulled back to the reference frame. Because both force and area vectors are mapped back to the undeformed geometry, $\boldsymbol{S}$ cannot be directly measured by physical sensors.
  - **Why We Need It**: It is symmetric ($\boldsymbol{S} = \boldsymbol{S}^T$) and energetically conjugate to Green-Lagrange strain $\boldsymbol{E}$. This conjugacy makes $\boldsymbol{S}$ mathematically essential for formulating the Total Lagrangian variational weak form, enabling us to integrate equations of motion over the fixed, time-invariant reference mesh $\Omega_0$.

---

## 3. Constitutive Material Models

### 3.1 Linear Isotropic Elasticity (`Hookean`)
For small-strain elastic problems, the stress-strain relation is governed by generalized Hooke's law:
$$\boldsymbol{\sigma} = \lambda \operatorname{tr}(\boldsymbol{\varepsilon}) \boldsymbol{I} + 2 \mu \boldsymbol{\varepsilon}$$
where $\lambda$ and $\mu$ are the Lamé parameters, related to Young's modulus $E$ and Poisson's ratio $\nu$ by:
$$\lambda = \frac{E \nu}{(1 + \nu)(1 - 2\nu)}, \quad \mu = \frac{E}{2(1 + \nu)}$$

In 6x6 Voigt matrix notation $\boldsymbol{\sigma}_{voigt} = \boldsymbol{C} \boldsymbol{\varepsilon}_{voigt}$:
$$\begin{bmatrix}
\sigma_{xx} \\ \sigma_{yy} \\ \sigma_{zz} \\ \sigma_{xy} \\ \sigma_{yz} \\ \sigma_{zx}
\end{bmatrix} =
\begin{bmatrix}
\lambda + 2\mu & \lambda & \lambda & 0 & 0 & 0 \\
\lambda & \lambda + 2\mu & \lambda & 0 & 0 & 0 \\
\lambda & \lambda & \lambda + 2\mu & 0 & 0 & 0 \\
0 & 0 & 0 & \mu & 0 & 0 \\
0 & 0 & 0 & 0 & \mu & 0 \\
0 & 0 & 0 & 0 & 0 & \mu
\end{bmatrix}
\begin{bmatrix}
\varepsilon_{xx} \\ \varepsilon_{yy} \\ \varepsilon_{zz} \\ \gamma_{xy} \\ \gamma_{yz} \\ \gamma_{zx}
\end{bmatrix}$$

In Total Lagrangian non-linear formulations, this model corresponds to the **Saint Venant-Kirchhoff** material:
$$\boldsymbol{S} = \lambda \operatorname{tr}(\boldsymbol{E}) \boldsymbol{I} + 2 \mu \boldsymbol{E}$$

#### Physical Meaning of Elastic Moduli & Saint Venant-Kirchhoff Limitations
- **Young's Modulus $E$**: Tensile stiffness—the slope of the uniaxial stress-strain curve.
- **Poisson's Ratio $\nu$**: Cross-sectional coupling—the ratio of lateral contraction to longitudinal elongation under tension. For compressible metals $\nu \approx 0.3$; for incompressible rubber and biological tissue, $\nu \to 0.5$.
- **Shear Modulus $\mu$**: Resistance to angular distortion and shearing without changing volume.
- **Bulk Modulus $K = \lambda + \frac{2}{3}\mu$**: Hydrostatic stiffness—the resistance to volumetric compression under uniform pressure ($\Delta P = -K \frac{\Delta V}{V_0}$).
- **Physical Caveat of Saint Venant-Kirchhoff**: Under moderate deformations and rotations, it correctly models large rigid rotations. However, under large compressive strains ($>30\%$), the strain energy does not blow up to infinity as $V \to 0$. In fact, beyond a critical compressive strain, it predicts decreasing or negative resistance, causing catastrophic non-physical element inversion. This motivates hyperelasticity for soft tissues.

---

### 3.2 Compressible Neo-Hookean Hyperelasticity (`NeoHookean`)
Hyperelastic materials are characterized by a Helmholtz strain energy density function $\Psi(\boldsymbol{C})$ per unit reference volume. To prevent element collapse and model large soft-tissue deformations, `pSierra` implements a decoupled compressible Neo-Hookean model:
$$\Psi(\boldsymbol{C}) = \Psi_{iso}(\bar{\boldsymbol{C}}) + \Psi_{vol}(J)$$

1. **Kinematic Decomposition**:
   $$\bar{\boldsymbol{F}} = J^{-1/3} \boldsymbol{F}, \quad \bar{\boldsymbol{C}} = \bar{\boldsymbol{F}}^T \bar{\boldsymbol{F}} = J^{-2/3} \boldsymbol{C}, \quad \bar{I}_1 = \operatorname{tr}(\bar{\boldsymbol{C}}) = J^{-2/3} I_1$$
   where $I_1 = \operatorname{tr}(\boldsymbol{C}) = C_{11} + C_{22} + C_{33}$.

2. **Strain Energy Function**:
   $$\Psi = \frac{1}{2} \mu \left( J^{-2/3} I_1 - 3 \right) + \frac{1}{2} K (\ln J)^2$$
   where $\mu$ is the shear modulus and $K = \lambda + \frac{2}{3}\mu$ is the bulk modulus.

3. **Second Piola-Kirchhoff Stress Tensor**:
   $$\boldsymbol{S} = 2 \frac{\partial \Psi}{\partial \boldsymbol{C}} = \boldsymbol{S}_{iso} + \boldsymbol{S}_{vol}$$
   $$\boldsymbol{S}_{iso} = \mu J^{-2/3} \left( \boldsymbol{I} - \frac{1}{3} I_1 \boldsymbol{C}^{-1} \right)$$
   $$\boldsymbol{S}_{vol} = K \ln(J) \boldsymbol{C}^{-1}$$

4. **Fourth-Order Tangent Elasticity Tensor ($\mathbb{C}$)**:
   $$\mathbb{C} = 4 \frac{\partial^2 \Psi}{\partial \boldsymbol{C} \partial \boldsymbol{C}} = 2 \frac{\partial \boldsymbol{S}}{\partial \boldsymbol{C}} = \mathbb{C}_{iso} + \mathbb{C}_{vol}$$
   $$\mathbb{C}_{vol} = K \left( \boldsymbol{C}^{-1} \otimes \boldsymbol{C}^{-1} - 2 \ln(J) \mathbb{I}_{\boldsymbol{C}^{-1}} \right)$$
   $$\mathbb{C}_{iso} = -\frac{2}{3} \mu J^{-2/3} \left( \boldsymbol{C}^{-1} \otimes \boldsymbol{I} + \boldsymbol{I} \otimes \boldsymbol{C}^{-1} \right) + \frac{2}{9} \mu J^{-2/3} I_1 \left( \boldsymbol{C}^{-1} \otimes \boldsymbol{C}^{-1} \right) + \frac{2}{3} \mu J^{-2/3} I_1 \mathbb{I}_{\boldsymbol{C}^{-1}}$$
   where $(\mathbb{I}_{\boldsymbol{C}^{-1}})_{ijkl} = \frac{1}{2} (C_{ik}^{-1} C_{jl}^{-1} + C_{il}^{-1} C_{jk}^{-1})$.

#### Physical Meaning of the Neo-Hookean Model
- **Isochoric-Volumetric Split**: Biological soft tissues (muscles, liver, brain, blood vessels) consist mostly of water contained in a fibrous extracellular matrix. When deformed, the shape distorts readily (low shear modulus $\mu$), but the volume is virtually incompressible (high bulk modulus $K$). Splitting $\Psi$ into $\Psi_{iso}$ and $\Psi_{vol}$ mirrors this physical duality.
- **Microscopic Physics of $\Psi_{iso}$**: Rooted in statistical thermodynamics of polymer networks (Gaussian chain statistics). Stretching the material straightens out randomly coiled collagen/elastin polymer chains, reducing their entropy. $\Psi_{iso}$ represents the entropic restoring force resisting shape distortion.
- **The Logarithmic Inversion Barrier $\frac{1}{2} K (\ln J)^2$**:
  - As volume expands ($J \to \infty$), $\ln(J) \to +\infty$, resisting void growth.
  - As volume collapses ($J \to 0$), $\ln(J) \to -\infty$, creating an **infinite repulsive hydrostatic pressure**:
    $$\boldsymbol{\sigma}_{vol} = \frac{1}{J} \boldsymbol{F} \boldsymbol{S}_{vol} \boldsymbol{F}^T = \frac{K \ln(J)}{J} \boldsymbol{I} \to -\infty \cdot \boldsymbol{I}$$
    This infinite barrier physically prevents mesh elements from turning inside out during heavy surgical manipulation, tool indentation, or large dynamic impacts.

---

## 4. Computational & Algorithmic Notes

1. **Einsum Tensor Contractions**:
   In [`neohookean.py`](../src/psierra/material/neohookean.py), the 4th-order tensor operations are computed using NumPy's optimized `einsum`:
   ```python
   C_inv_outer = np.einsum("ij,kl->ijkl", C_inv, C_inv)
   I_Cinv = 0.5 * (
       np.einsum("ik,jl->ijkl", C_inv, C_inv)
       + np.einsum("il,jk->ijkl", C_inv, C_inv)
   )
   ```
2. **Voigt Matrix Transformation**:
   The full symmetric 4th-order tensor $C_{ijkl}$ is projected into the 6x6 Voigt tangent matrix using the mapping defined in [`VOIGT_MAP_3D`](../src/psierra/utility/tensor.py):
   $$\{0: (0,0), 1: (1,1), 2: (2,2), 3: (0,1), 4: (1,2), 5: (2,0)\}$$
   $$C_{voigt}[a, b] = \mathbb{C}[i, j, k, l]$$

---

## 5. Code Implementation Map

| Mathematical Quantity | Formula | Source Location | Class & Method |
|---|---|---|---|
| Elastic Parameters | $E, \nu, \lambda, \mu, K$ | [`parameters.py`](../src/psierra/material/parameters.py) | `IsotropicElasticParameters.set_from_youngs_poisson()` |
| Linear Elasticity Matrix | $\boldsymbol{C} \in \mathbb{R}^{6 \times 6}$ | [`hookean.py`](../src/psierra/material/hookean.py) | `Hookean.compute_constitutive_matrix()` |
| Deformation Gradient | $\boldsymbol{F} = \boldsymbol{x}^T \boldsymbol{H}$ | [`tet4.py`](../src/psierra/element/tet4.py) | `Tet4.compute_nonlinear_force_and_tangent_stiffness()` |
| 2nd PK Stress $\boldsymbol{S}$ | $\boldsymbol{S} = \boldsymbol{S}_{iso} + \boldsymbol{S}_{vol}$ | [`neohookean.py`](../src/psierra/material/neohookean.py) | `NeoHookean.compute_stress_and_tangent()` |
| 4th-Order Tangent $\mathbb{C}$ | $\mathbb{C} = 2 \frac{\partial \boldsymbol{S}}{\partial \boldsymbol{C}}$ | [`neohookean.py`](../src/psierra/material/neohookean.py) | `NeoHookean.compute_stress_and_tangent()` |
| Voigt Stress Mapping | $\boldsymbol{S} \to \boldsymbol{S}_{voigt}$ | [`tensor.py`](../src/psierra/utility/tensor.py) | `tensor_to_voigt_stress()` |

---

## 6. References & Further Reading

- **J. Bonet & R.D. Wood**, *Nonlinear Continuum Mechanics for Finite Element Analysis*, Chapter 5: Hyperelastic Materials.
- **G.A. Holzapfel**, *Nonlinear Solid Mechanics: A Continuum Approach for Engineering*, Chapter 6: Hyperelasticity.
- **T. Belytschko et al.**, *Nonlinear Finite Elements for Continua and Structures*, Chapter 5: Constitutive Models.
