# 04. Structural Continua: Thin & Moderately Thick Shell Elements (`Shell3`, `Shell6`)

This document details the mathematical theory, kinematics, plate bending models, and numerical stabilization for structural shell elements (`Shell3`, `Shell6`) in `pSierra`.

---

## 1. Biomechanical Motivation: Why Shell Elements?

In anatomical and biomechanical simulation, organs frequently comprise soft, volumetrically bulky parenchymal tissue encased by thin, stiff fibrous membranes:
- **Brain & Spine**: Soft brain parenchyma ($E \approx 1 - 50 \text{ kPa}$) enclosed within the tough, fibrous **dura mater** ($E \approx 10 - 50 \text{ MPa}$, thickness $t \approx 0.5 - 1.0 \text{ mm}$) and bounded by the rigid cranial skull.
- **Cardiovascular**: Aortic root and heart chambers lined with pericardial membranes.
- **Musculoskeletal**: Bone periosteum and fascial muscle sheaths.

### 1.1 The Failure of 3D Solid Continua for Thin Membranes
Attempting to discretize a $0.5\text{ mm}$ membrane using 3D solid tetrahedral elements (`Tet4` or `Tet10`) introduces fatal computational defects:
1. **Geometric Aspect Ratio Distortion**: If the lateral span is $100\text{ mm}$, generating well-shaped tetrahedra requires millions of sub-millimeter elements across the entire organ, causing DOF explosions.
2. **Numerical Ill-Conditioning**: Solid elements with high aspect ratios ($L / t \gg 20$) produce catastrophic condition numbers ($\kappa(\boldsymbol{K}) > 10^{14}$), destroying iterative solver convergence and degrading direct matrix factorization accuracy.
3. **Artificial Transverse Stiffness**: Solid continua exhibit artificial Poisson locking across the ultra-thin thickness under bending loads.

### 1.2 The Shell Solution: Dimensional Reduction ($3D \to 2D$)
Structural shell elements eliminate thickness discretization by analytically integrating stress distributions through the wall thickness $t$. This collapses the geometric representation to a 2D mid-surface mesh while preserving full 3D membrane tension, in-plane shear, transverse bending, twisting moments, and rotary inertia.

---

## 2. Shell Kinematics: 6 Degrees of Freedom per Node

Each shell node $a$ possesses **6 generalized degrees of freedom**:
$$\boldsymbol{d}_a = \begin{bmatrix} u_a & v_a & w_a & \theta_{x,a} & \theta_{y,a} & \theta_{z,a} \end{bmatrix}^T$$
- $\boldsymbol{u}_a = [u_a, v_a, w_a]^T$: Translational displacement vector of the mid-surface in global Cartesian coordinates.
- $\boldsymbol{\theta}_a = [\theta_{x,a}, \theta_{y,a}, \theta_{z,a}]^T$: Rotational vector (infinitesimal rotations of the cross-sectional director vector).

```
         ^ z_local (n)
         |
         |      o Node 2
         |     / \
         |    /   \
         |   /     \
         o--/-------\--> y_local (t2)
        Node 0       Node 1
       /
      v x_local (t1)
```

### 2.1 Local Element Surface Coordinate Triad
For each triangular shell element, a local orthonormal coordinate system $(\boldsymbol{t}_1, \boldsymbol{t}_2, \boldsymbol{n})$ is constructed from the reference nodal coordinates $\boldsymbol{X}_0, \boldsymbol{X}_1, \boldsymbol{X}_2$:
1. **Primary Tangent $\boldsymbol{t}_1$** (aligned with Edge $0 \to 1$):
   $$\boldsymbol{v}_1 = \boldsymbol{X}_1 - \boldsymbol{X}_0, \quad \boldsymbol{t}_1 = \frac{\boldsymbol{v}_1}{\|\boldsymbol{v}_1\|}$$
2. **Normal Vector $\boldsymbol{n}$** (outward surface normal):
   $$\boldsymbol{v}_2 = \boldsymbol{X}_2 - \boldsymbol{X}_0, \quad \boldsymbol{n} = \frac{\boldsymbol{t}_1 \times \boldsymbol{v}_2}{\|\boldsymbol{t}_1 \times \boldsymbol{v}_2\|}$$
3. **Secondary Tangent $\boldsymbol{t}_2$** (in-plane cross-tangent):
   $$\boldsymbol{t}_2 = \boldsymbol{n} \times \boldsymbol{t}_1$$

### 2.2 Coordinate Transformation Matrix $\boldsymbol{\Lambda}$ and $\boldsymbol{T}$
The $3 \times 3$ direction cosine transformation matrix $\boldsymbol{\Lambda}$ is:
$$\boldsymbol{\Lambda} = \begin{bmatrix} \boldsymbol{t}_1^T \\ \boldsymbol{t}_2^T \\ \boldsymbol{n}^T \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$
For each node, the transformation between global DOFs $\boldsymbol{d}_a$ and local DOFs $\boldsymbol{d}_{a, loc}$ is governed by:
$$\boldsymbol{d}_{a, loc} = \begin{bmatrix} \boldsymbol{\Lambda} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{\Lambda} \end{bmatrix} \boldsymbol{d}_a$$
The global element transformation matrix $\boldsymbol{T}$ is assembled as a block-diagonal matrix of $\boldsymbol{\Lambda}$ blocks:
$$\boldsymbol{T} = \operatorname{diag}\left( \begin{bmatrix}\boldsymbol{\Lambda} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{\Lambda}\end{bmatrix}, \dots \right) \in \mathbb{R}^{n_{dof} \times n_{dof}}$$
Any local element matrix $\boldsymbol{k}_{loc}$ transforms to the global assembly system via:
$$\boldsymbol{k}_{glob} = \boldsymbol{T}^T \boldsymbol{k}_{loc} \boldsymbol{T}$$

---

## 3. Membrane Formulations (In-Plane Action)

In-plane stretching and shearing are governed by plane-stress membrane theory.

### 3.1 Constant Strain Triangle (CST) for `Shell3`
For the 3-node triangular element:
$$\boldsymbol{\varepsilon}_m = \begin{bmatrix} \frac{\partial u}{\partial x} & \frac{\partial v}{\partial y} & \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \end{bmatrix}^T = \boldsymbol{B}_m \boldsymbol{u}_m$$
The membrane strain-displacement matrix $\boldsymbol{B}_m \in \mathbb{R}^{3 \times 6}$ is spatially constant:
$$\boldsymbol{B}_m = \frac{1}{2 A} \begin{bmatrix}
y_{12} & 0 & y_{20} & 0 & y_{01} & 0 \\
0 & x_{21} & 0 & x_{02} & 0 & x_{10} \\
x_{21} & y_{12} & x_{02} & y_{20} & x_{10} & y_{01}
\end{bmatrix}$$
where $x_{ij} = x_i - x_j$, $y_{ij} = y_i - y_j$, and $A$ is the triangle area. The membrane stiffness matrix is:
$$\boldsymbol{k}_m = A t \boldsymbol{B}_m^T \boldsymbol{C}_{plane} \boldsymbol{B}_m \in \mathbb{R}^{6 \times 6}$$
where $\boldsymbol{C}_{plane} = \frac{E}{1 - \nu^2} \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1-\nu}{2} \end{bmatrix}$ is the isotropic plane stress elasticity tensor.

### 3.2 Linear Strain Triangle (LST) for `Shell6`
For the 6-node quadratic element, quadratic serendipity shape functions interpolate in-plane displacements:
$$\boldsymbol{u}_m(x, y) = \sum_{a=0}^5 N_a(x, y) \begin{bmatrix} u_a \\ v_a \end{bmatrix}$$
The resulting membrane strain $\boldsymbol{\varepsilon}_m(x, y)$ varies linearly across the element, integrated using 3-point Dunavant numerical quadrature over the master triangle:
$$\boldsymbol{k}_m = t \sum_{q=1}^3 w_q A \, \boldsymbol{B}_m(\boldsymbol{\xi}_q)^T \boldsymbol{C}_{plane} \boldsymbol{B}_m(\boldsymbol{\xi}_q) \in \mathbb{R}^{12 \times 12}$$
This prevents in-plane shear locking and models high in-plane strain gradients around curved boundaries.

---

## 4. Plate Bending Formulations (Out-of-Plane Action)

### 4.1 Kirchhoff-Love vs Reissner-Mindlin Theories

```
Kirchhoff-Love (Thin):               Reissner-Mindlin (Thick/Moderate):
   Normals remain normal:                Normals rotate independently:
        \                                     \
         \                                     \   gamma_s (Shear Strain)
          \                                     \ /
       ====*====                             ====*====
       theta = w'                            gamma = w' - theta != 0
```

- **Kirchhoff-Love Theory**: Assumes transverse shear strains are identically zero ($\gamma_{xz} = \gamma_{yz} = 0$). Cross-sectional normals remain straight and perpendicular to the deformed mid-surface. Valid when $t / L < 0.05$.
- **Reissner-Mindlin Theory**: Permits cross-sectional normals to rotate independently of the mid-surface slope, allowing transverse shear deformation ($\gamma = \nabla w - \boldsymbol{\theta}$). Valid for both thin and moderately thick plates ($t / L \le 0.20$).

### 4.2 Discrete Kirchhoff Triangle (`DKT` in `Shell3`)
Standard $C^1$-continuous thin plate bending requires second derivatives ($\nabla^2 w$), which cannot be satisfied with simple 3-node triangles without non-conforming displacement jumps.

The **Discrete Kirchhoff Triangle (DKT)** formulation (Batoz, Bathe, & Ho, 1980) resolves this by:
1. Interpolating section rotations $(\beta_x, \beta_y)$ quadratically over the 6 mid-side and corner points.
2. Imposing the Kirchhoff thin-plate hypothesis **discretely** at the mid-points $k \in \{4, 5, 6\}$ of each of the 3 triangle edges:
   $$\gamma_s^{(k)} = \frac{\partial w}{\partial s} - \beta_s = 0$$
3. Expressing the bending slopes strictly in terms of the 9 nodal bending DOFs:
   $$\boldsymbol{w}_b = \begin{bmatrix} w_0 & \theta_{x,0} & \theta_{y,0} & w_1 & \theta_{x,1} & \theta_{y,1} & w_2 & \theta_{x,2} & \theta_{y,2} \end{bmatrix}^T$$
4. Constructing the bending strain-displacement curvature matrix $\boldsymbol{B}_b(\boldsymbol{\xi}) \in \mathbb{R}^{3 \times 9}$:
   $$\boldsymbol{\kappa} = \begin{bmatrix} \frac{\partial \beta_x}{\partial x} \\ \frac{\partial \beta_y}{\partial y} \\ \frac{\partial \beta_x}{\partial y} + \frac{\partial \beta_y}{\partial x} \end{bmatrix} = \boldsymbol{B}_b(\boldsymbol{\xi}) \boldsymbol{w}_b$$
5. Evaluating the bending stiffness matrix using 3-point Dunavant numerical quadrature:
   $$\boldsymbol{k}_b = \sum_{q=1}^3 w_q A \, \boldsymbol{B}_b(\boldsymbol{\xi}_q)^T \boldsymbol{D}_b \boldsymbol{B}_b(\boldsymbol{\xi}_q) \in \mathbb{R}^{9 \times 9}$$
   where $\boldsymbol{D}_b = \frac{E t^3}{12(1-\nu^2)} \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1-\nu}{2} \end{bmatrix}$ is the flexural bending rigidity tensor.

The DKT element is widely regarded as one of the most accurate, convergence-robust 3-node thin shell bending elements in computational mechanics, entirely free of shear locking.

### 4.3 Reissner-Mindlin Plate Bending in `Shell6`
For the 6-node quadratic element, both transverse deflection $w$ and section rotations $(\theta_x, \theta_y)$ are independently interpolated using quadratic serendipity shape functions:
- Bending curvature stiffness:
  $$\boldsymbol{k}_b = \int_A \boldsymbol{B}_b^T \boldsymbol{D}_b \boldsymbol{B}_b \, dA \in \mathbb{R}^{18 \times 18}$$
- Transverse shear stiffness:
  $$\boldsymbol{k}_s = \kappa G t \int_A \boldsymbol{B}_s^T \boldsymbol{B}_s \, dA \in \mathbb{R}^{18 \times 18}$$
  where $\kappa = 5/6$ is the Reissner shear correction factor and $\boldsymbol{B}_s$ relates shear strains $\boldsymbol{\gamma} = \begin{bmatrix} \frac{\partial w}{\partial x} - \theta_x \\ \frac{\partial w}{\partial y} - \theta_y \end{bmatrix}$ to nodal displacements and rotations.

---

## 5. The Drilling Degree of Freedom ($\theta_z$) & Stabilization

### 5.1 The Drilling Singularity
In a flat planar element lying in the local $x$-$y$ plane, in-plane membrane forces generate resistance to $u$ and $v$, while out-of-plane bending resists $w, \theta_x, \theta_y$. However, rotation about the out-of-plane surface normal $\theta_z$ (the "drilling" degree of freedom) has **zero physical stiffness** ($\boldsymbol{k}_{zz} = 0$).

When assembling planar shells into 3D curved surfaces, transforming local unconstrained $\theta_z$ DOFs to global Cartesian coordinates produces zero diagonal pivots in the global stiffness matrix $\boldsymbol{K}$, rendering the structural system singular and non-invertible.

### 5.2 Hughes-Brezzi Drilling Stabilization
To eliminate this singularity without introducing unphysical stiffness, `pSierra` incorporates Hughes-Brezzi artificial drilling stabilization. A small, fictitious rotational stiffness is added to the drilling rotation DOF at each node:
$$k_{drill} = \alpha_{drill} \cdot G \cdot t \cdot A$$
where $\alpha_{drill} = 1.0 \times 10^{-3}$ is a dimensionless stabilization parameter, $G$ is the shear modulus, $t$ is shell thickness, and $A$ is the element area.

This fictitious stiffness:
1. Prevents zero pivots in linear solvers during global factorization.
2. Invariantly leaves rigid body modes unconstrained ($k_{drill} \to 0$ under rigid rotation).
3. Produces negligible strain energy compared to true physical membrane and bending modes.

---

## 6. Consistent Mass Matrix Integration

The shell consistent mass matrix balances translational momentum and rotary inertia across the thickness:
$$\boldsymbol{m}_{e} = \int_A \boldsymbol{N}^T \begin{bmatrix} \rho t \boldsymbol{I}_3 & \boldsymbol{0} \\ \boldsymbol{0} & \frac{\rho t^3}{12} \boldsymbol{I}_3 \end{bmatrix} \boldsymbol{N} \, dA$$
- **Translational inertia**: Proportional to areal mass density $\rho t$.
- **Rotary inertia**: Proportional to thickness moment of inertia $I_\rho = \frac{\rho t^3}{12}$, resisting rapid angular acceleration of shell cross-sections.

---

## 7. Code Implementation Map

| Component | Source File | Method / Class |
| :--- | :--- | :--- |
| **3-Node Flat Shell (`Shell3`)** | [`psierra/element/shell3.py`](../src/psierra/element/shell3.py) | `Shell3.compute_linear_stiffness_matrix()`, `Shell3.compute_mass_matrix()` |
| **6-Node Quadratic Shell (`Shell6`)** | [`psierra/element/shell6.py`](../src/psierra/element/shell6.py) | `Shell6.compute_linear_stiffness_matrix()`, `Shell6.compute_mass_matrix()` |
| **Simplex Triangle Quadrature** | [`psierra/element/quadrature.py`](../src/psierra/element/quadrature.py) | `TriangleQuadrature` (1-pt centroid, 3-pt Dunavant, 4-pt Hammer) |
| **Shell Domain Representation** | [`psierra/domain/shell.py`](../src/psierra/domain/shell.py) | `ShellDomain.assemble_stiffness_matrix()`, `ShellDomain.to_pyvista()` |

---

## 8. References & Further Reading

- **J.L. Batoz, K.J. Bathe, & L.W. Ho**, *A Study of Three-Node Triangular Plate Bending Elements*, International Journal for Numerical Methods in Engineering, 15:1771-1812, 1980.
- **T.J.R. Hughes & F. Brezzi**, *On Drilling Degrees of Freedom*, Computer Methods in Applied Mechanics and Engineering, 72:105-121, 1989.
- **D.A. Dunavant**, *High Degree Efficient Symmetrical Gaussian Quadrature Rules for the Triangle*, International Journal for Numerical Methods in Engineering, 21:1129-1148, 1985.
- **O.C. Zienkiewicz & R.L. Taylor**, *The Finite Element Method for Solid and Structural Mechanics*, Chapter 4 (Plates and Shells), 7th Edition, 2013.

---

### Next Step in the Pedagogical Path
Now that all element formulations are established—from linear solids (`Tet4`) and quadratic solids (`Tet10`) to structural shells (`Shell3`, `Shell6`)—how do we assemble them into unified global algebraic systems and couple solids and shells together at physical interfaces?
- Proceed to **[05. System Assembly, Boundary Conditions & Monolithic Domain Coupling](05_system_assembly_boundaries_and_domain_coupling.md)**.

