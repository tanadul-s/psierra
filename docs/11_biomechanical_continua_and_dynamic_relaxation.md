# Chapter 11: Biomechanical Continua, Volumetric Locking Alleviation & Dynamic Relaxation

This chapter presents the advanced continuum formulations and numerical strategies in `pSierra` designed specifically for biological soft tissues, nearly-incompressible hydrogels, and complex surgical digital twins.

---

## 1. Volumetric Locking in Linear Tetrahedra (`Tet4`)

### 1.1 The Physical Problem: Incompressibility in Biological Tissues
Soft biological parenchymal tissues (e.g., human brain gray and white matter, liver, kidney) are composed predominantly of interstitial fluid and bound water (>75–80% by weight). Under physiological quasi-static deformation rates, these tissues exhibit nearly-incompressible mechanical behavior characterized by a Poisson's ratio approaching the theoretical limit:
$$\nu \to 0.5 \quad (\nu \approx 0.495 - 0.4999)$$

In terms of the Lamé parameters and bulk modulus:
$$K = \frac{E}{3(1 - 2\nu)}, \quad \mu = \frac{E}{2(1 + \nu)}$$
As $\nu \to 0.5$, the bulk modulus $K \to \infty$, while the shear modulus $\mu$ remains finite. For $\nu = 0.499$, the incompressibility ratio reaches:
$$\frac{K}{\mu} \approx \frac{2(1 + 0.499)}{3(1 - 2 \times 0.499)} = \frac{2.998}{0.006} \approx 500$$

### 1.2 Parasitic Stiffening (Volumetric Locking)
Standard 4-node linear tetrahedral elements (`Tet4Linear` and `Tet4TotalLagrangian`) exhibit constant strain throughout each element ($C^0$ displacement with linear shape functions $N_i$). For a single tetrahedron:
$$\operatorname{div} \boldsymbol{u} = \operatorname{tr}(\boldsymbol{\varepsilon}) = \text{const}$$
Enforcing the incompressibility constraint $\operatorname{div} \boldsymbol{u} = 0$ imposes one independent constraint **per element**.

In a 3D unstructured tetrahedral mesh with $N_{\text{node}}$ nodes and $N_{\text{elem}}$ elements:
$$N_{\text{elem}} \approx 5 \text{ to } 6 \times N_{\text{node}}$$
Whereas the total available displacement degrees of freedom are $3 N_{\text{node}}$. Since $N_{\text{elem}} > 3 N_{\text{node}}$, the number of volumetric constraints strictly exceeds the total kinematic degrees of freedom. This causes **volumetric locking**:
- The numerical displacement field locks completely ($u \to 0$).
- The global tangent stiffness matrix $\boldsymbol{K}_T$ becomes severely ill-conditioned (condition number $\kappa(\boldsymbol{K}_T) \sim O(10^{10})$).
- Newton-Raphson solvers fail to achieve equilibrium because of spurious volumetric force spikes.

---

## 2. Mathematical Formulation of `Tet4Fbar`

`Tet4Fbar` eliminates volumetric locking by decoupling the volumetric and isochoric (deviatoric) kinematics using the Flory multiplicative split, evaluated with **nodal-averaged volume dilatation**.

### 2.1 Flory Multiplicative Split
Let $\boldsymbol{F}_e = \boldsymbol{I} + \nabla_{\boldsymbol{X}} \boldsymbol{u}_e$ be the standard deformation gradient of element $e$, with Jacobian determinant:
$$J_e = \det \boldsymbol{F}_e$$

The modified deformation gradient $\bar{\boldsymbol{F}}_e$ replaces the local volume ratio $J_e$ with an externally smoothed volume ratio $\bar{J}_e$:
$$\bar{\boldsymbol{F}}_e = \left(\frac{\bar{J}_e}{J_e}\right)^{1/3} \boldsymbol{F}_e$$
Taking the determinant verifies exact volumetric smoothing:
$$\det \bar{\boldsymbol{F}}_e = \left[\left(\frac{\bar{J}_e}{J_e}\right)^{1/3}\right]^3 \det \boldsymbol{F}_e = \frac{\bar{J}_e}{J_e} J_e = \bar{J}_e$$

### 2.2 Nodal Patch Volume Averaging
The smoothed Jacobian $\bar{J}_e$ is constructed from nodal volume patches to ensure non-local, unconstrained volumetric degrees of freedom:

1. **Reference Nodal Volumes**:
   For each node $a \in \{1, \dots, N_{\text{node}}\}$, the reference volume of the surrounding patch is:
   $$V_{0, a} = \sum_{e \in \mathcal{E}_a} \frac{1}{4} V_e$$
   where $\mathcal{E}_a$ denotes the set of elements sharing node $a$, and $V_e$ is the reference volume of element $e$.

2. **Current Deformed Nodal Volumes**:
   At the current displacement state $\boldsymbol{u}$:
   $$v_a = \sum_{e \in \mathcal{E}_a} \frac{1}{4} V_e J_e(\boldsymbol{u})$$

3. **Nodal Volume Ratio**:
   $$J_a = \frac{v_a}{V_{0, a}}$$

4. **Element Smoothed Jacobian**:
   For element $e$ connecting vertices $(a_0, a_1, a_2, a_3)$:
   $$\bar{J}_e = \frac{1}{4} \sum_{i=0}^3 J_{a_i}$$

Because the volumetric constraint is now averaged at the nodes, the maximum number of volumetric constraints is reduced from $N_{\text{elem}}$ down to $N_{\text{node}} \ll 3 N_{\text{node}}$, completely removing the over-constraint and unlocking the mesh.

---

## 3. Total Lagrangian Kinematics and Stress Tensors

### 3.1 Constitutive Stress and Tangent
The hyperelastic constitutive law is evaluated at the smoothed state $\bar{\boldsymbol{F}}_e$:
$$(\bar{\boldsymbol{C}}_e, \bar{\boldsymbol{S}}_e) = \text{material.compute\_stress\_and\_tangent}(\bar{\boldsymbol{F}}_e)$$
where $\bar{\boldsymbol{S}}_e$ is the Second Piola-Kirchhoff (2nd PK) stress and $\bar{\boldsymbol{C}}_e$ is the 4th-order material elasticity tensor in Voigt representation.

### 3.2 Modified Strain-Displacement Matrix $\bar{\boldsymbol{B}}$
The standard Total Lagrangian strain-displacement matrix is:
$$\boldsymbol{B}_e(\boldsymbol{u}) = \boldsymbol{B}_{L, e} + \boldsymbol{B}_{NL, e}(\boldsymbol{u})$$
The volumetric strain operator row is:
$$\boldsymbol{b}_{\text{vol}, e} = \frac{1}{3} (\boldsymbol{B}_{0, e} + \boldsymbol{B}_{1, e} + \boldsymbol{B}_{2, e})$$
The deviatoric strain operator is:
$$\boldsymbol{B}_{\text{dev}, e} = \boldsymbol{B}_e - \boldsymbol{m}_{\text{voigt}} \otimes \boldsymbol{b}_{\text{vol}, e}$$
where $\boldsymbol{m}_{\text{voigt}} = [1, 1, 1, 0, 0, 0]^T$.

The consistent modified B-matrix $\bar{\boldsymbol{B}}_e$ is:
$$\bar{\boldsymbol{B}}_e = \left(\frac{\bar{J}_e}{J_e}\right)^{2/3} \boldsymbol{B}_{\text{dev}, e} + \boldsymbol{m}_{\text{voigt}} \otimes \boldsymbol{b}_{\text{vol}, e}$$

### 3.3 Internal Force Vector
The element internal force vector is:
$$\boldsymbol{f}_{\text{int}, e} = \left(\bar{\boldsymbol{B}}_e^T \bar{\boldsymbol{S}}_{\text{voigt}, e}\right) V_e$$
Global internal forces $\boldsymbol{f}_{\text{int}}$ are assembled via standard DOF indexing:
$$\boldsymbol{f}_{\text{int}} = \sum_{e=1}^{N_{\text{elem}}} \boldsymbol{A}_e^T \boldsymbol{f}_{\text{int}, e}$$

---

## 4. Consistent Global Tangent Stiffness Matrix

Achieving quadratic Newton-Raphson convergence requires the exact directional derivative:
$$\boldsymbol{K}_T = \frac{\partial \boldsymbol{f}_{\text{int}}}{\partial \boldsymbol{u}} = \boldsymbol{K}_{\text{dev}} + \boldsymbol{K}_{\text{vol}}$$

### 4.1 Deviatoric Tangent Stiffness
The deviatoric constitutive tensor is:
$$\boldsymbol{C}_{\text{dev}, e} = \bar{\boldsymbol{C}}_e - K_{\text{bulk}} (\boldsymbol{m}_{\text{voigt}} \otimes \boldsymbol{m}_{\text{voigt}})$$

For each element, the deviatoric material and geometric stiffness matrices are:
$$\boldsymbol{k}_{\text{mat}, e}^{\text{dev}} = \left(\bar{\boldsymbol{B}}_{\text{dev}, e}^T \boldsymbol{C}_{\text{dev}, e} \bar{\boldsymbol{B}}_{\text{dev}, e}\right) V_e$$
$$\boldsymbol{k}_{\text{geo}, e} = \left(\boldsymbol{G}_e^T (\boldsymbol{I}_3 \otimes \bar{\boldsymbol{S}}_e) \boldsymbol{G}_e\right) V_e$$
$$\boldsymbol{K}_{\text{dev}} = \sum_{e=1}^{N_{\text{elem}}} \boldsymbol{A}_e^T \left(\boldsymbol{k}_{\text{mat}, e}^{\text{dev}} + \boldsymbol{k}_{\text{geo}, e}\right) \boldsymbol{A}_e$$

### 4.2 Nodal Volumetric Tangent Coupling
Because $\bar{J}_e$ couples all neighboring elements through node $a$, differentiating the volumetric dilatation yields a nodal dyadic product:
$$\boldsymbol{K}_{\text{vol}} = \sum_{a=1}^{N_{\text{node}}} V_{0, a} K_{\text{bulk}} \boldsymbol{g}_a^T \boldsymbol{g}_a$$
where $\boldsymbol{g}_a$ is the global gradient of nodal volume ratio:
$$\boldsymbol{g}_a = \frac{1}{V_{0, a}} \sum_{e \in \mathcal{E}_a} \frac{1}{4} V_e J_e (\boldsymbol{B}_{0, e} + \boldsymbol{B}_{1, e} + \boldsymbol{B}_{2, e})$$

This closed-form global decomposition avoids expensive numerical differentiation, satisfies symmetry, and guarantees quadratic Newton convergence in 3 to 5 iterations.

---

## 5. Verification & Showcase Benchmark

The implementation is verified in `examples/11_advanced_continua/1101_tet4fbar_incompressibility.py` on an unstructured tetrahedral mesh with $\nu = 0.499$ ($K / \mu \approx 500$) subjected to 1.0G gravitational self-weight:

| Formulation | Poisson's Ratio $\nu$ | $K / \mu$ | 1G Gravity Sag ($|Z|$) | Locking Alleviation |
| :--- | :---: | :---: | :---: | :---: |
| **Standard Tet4** (`Tet4TotalLagrangian`) | 0.499 | 500x | $2.3951 \times 10^{-4}$ mm | Baseline (Locked) |
| **Tet4Fbar** (`Tet4Fbar`) | 0.499 | 500x | $6.7153 \times 10^{-4}$ mm | **$2.80\times$ more compliant** |

---

## 6. API Reference & Usage Guide

```python
from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4Fbar
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import GravityLoad
from psierra.analysis import NonlinearStaticAnalysis

# 1. Nearly-incompressible soft tissue parameters
params = IsotropicElasticParameters()
params.set_from_youngs_poisson(
    youngs_modulus=3000.0,   # 3.0 kPa
    poissons_ratio=0.499,    # nu -> 0.5
    mass_density=1.04e-9,    # Ton/mm^3 (1040 kg/m^3)
)
material = NeoHookean(params)

# 2. Instantiate Tet4Fbar element and SolidDomain
element = Tet4Fbar(material)
domain = SolidDomain(mesh, material, element)

# 3. 1G Gravitational body load
gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
M = domain.assemble_mass_matrix()
f_ext = gravity.compute_force_vector(M)

# 4. Solve with standard analysis pipelines
analysis = NonlinearStaticAnalysis(domain)
result = analysis.solve(dirichlet=dir_mgr, f_ext=f_ext, n_steps=5)

# 5. Export state for 3D visualization in ParaView
domain.save_state("results/1101_tet4fbar/result_tet4_fbar.vtu")
```
