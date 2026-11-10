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

---

## 7. Dynamic Relaxation Explicit Static Equilibrium Solver (`DynamicRelaxationAnalysis`)

### 7.1 Motivation: Robustness Beyond Newton-Raphson
Standard static equilibrium solvers rely on the incremental Newton-Raphson procedure:
$$\boldsymbol{K}_T(\boldsymbol{u}_k) \Delta \boldsymbol{u} = \boldsymbol{R}(\boldsymbol{u}_k) = \boldsymbol{F}_{\text{ext}} - \boldsymbol{F}_{\text{int}}(\boldsymbol{u}_k)$$
While Newton-Raphson achieves quadratic asymptotic convergence near equilibrium, it is vulnerable to:
1. **Tangent Singularities ($\det \boldsymbol{K}_T \le 0$)**: Local buckling, shear band localization, and tissue collapse make $\boldsymbol{K}_T$ singular or indefinite, causing sparse factorizations ($LU$ / Cholesky) to diverge.
2. **Contact Chatter & Non-Smoothness**: Abrupt switching between contact active/inactive states creates non-differentiable residual jumps across element facets.
3. **Memory Footprint**: Forming and factorizing $\boldsymbol{K}_T$ for fine meshes requires $O(N^{1.5} - N^2)$ storage and large fill-in memory.

### 7.2 The Fictitious Dynamic System
Dynamic Relaxation (DR) resolves static equilibrium without forming or factorizing $\boldsymbol{K}_T$. It embeds the static balance $\boldsymbol{F}_{\text{int}}(\boldsymbol{u}) - \boldsymbol{F}_{\text{ext}} = \mathbf{0}$ into an explicit, highly damped pseudo-dynamic system:
$$\boldsymbol{M}_{\text{fict}} \ddot{\boldsymbol{u}} + \boldsymbol{C}_{\text{fict}} \dot{\boldsymbol{u}} + \boldsymbol{F}_{\text{int}}(\boldsymbol{u}) = \boldsymbol{F}_{\text{ext}}$$
As kinetic energy is dissipated, $\dot{\boldsymbol{u}} \to \mathbf{0}$ and $\ddot{\boldsymbol{u}} \to \mathbf{0}$, leaving the exact static equilibrium:
$$\boldsymbol{F}_{\text{int}}(\boldsymbol{u}) = \boldsymbol{F}_{\text{ext}}$$

### 7.3 Gerschgorin-Scaled Diagonal Mass Matrix
To ensure numerical stability of the central-difference integrator without matrix inversion, a fictitious diagonal mass $\boldsymbol{M}_{\text{fict}}$ is constructed using Gerschgorin's circle theorem:
$$M_{ii} \ge \frac{\Delta t^2}{4} \left( \sum_{j} |K_{ij}| \right) \cdot \gamma_{\text{safety}}$$
where $\gamma_{\text{safety}} \ge 1.1$. This guarantees that the Courant-Friedrichs-Lewy (CFL) stability criterion:
$$\omega_{\max} \Delta t \le 2$$
is automatically satisfied for every degree of freedom, regardless of mesh size or material stiffness.

### 7.4 Barnes/Underwood Kinetic Energy Peak Damping
Rather than attempting to guess an optimal viscous damping coefficient $c$, kinetic damping operates parameter-free:
1. Integrate forward in pseudo-time with zero artificial viscous damping ($\boldsymbol{C} = \mathbf{0}$).
2. Track the total kinetic energy:
   $$E_k^{n+1/2} = \frac{1}{2} (\boldsymbol{v}^{n+1/2})^T \boldsymbol{M}_{\text{fict}} \boldsymbol{v}^{n+1/2}$$
3. When kinetic energy peaks ($E_k^{n+1/2} < E_k^{n-1/2}$), the coordinates are passing through the static potential energy minimum.
4. Underwood interpolation projects coordinates directly to the peak:
   $$\boldsymbol{u} \leftarrow \boldsymbol{u} - 1.5 \Delta t \, \boldsymbol{v}^{n-1/2} + 0.5 \Delta t^2 \boldsymbol{a}^{n-1}$$
5. The velocity is reset to zero ($\boldsymbol{v} \leftarrow \mathbf{0}$), completely purging the kinetic energy and restarting from rest.

### 7.5 Usage Example

```python
from psierra.analysis import DynamicRelaxationAnalysis

# Works identically on both linear and nonlinear domains
solver = DynamicRelaxationAnalysis(
    domain=domain,
    damping_type="kinetic",  # Parameter-free Barnes/Underwood kinetic damping
    safety_factor=1.1,
    dt=1.0,
)

result = solver.solve(
    dirichlet=dirichlet_manager,
    f_ext=f_ext,
    tol=1e-5,
    max_steps=2000,
    save_directory="results/dynamic_relaxation",
    format="xdmf",
)

print(f"Converged: {result.converged} in {result.steps_completed} steps")
print(f"Max Displacement: {result.displacement.max():.4f} mm")
```

---

## 8. Hydrostatic Pressure & Buoyancy Loads (`HydrostaticPressureLoad`, `BuoyancyLoad`)

### 8.1 Physical Motivation & Medical Relevance
In soft-tissue biomechanics, human organs are immersed in physiological fluids that provide critical Archimedean neutral buoyant support:
- The human brain ($\rho_{\text{brain}} \approx 1040\text{ kg/m}^3$) is suspended in Cerebrospinal Fluid (CSF, $\rho_{\text{CSF}} \approx 1000\text{ kg/m}^3$).
- Intact CSF buoyant lift counteracts **$96.2\%$** of the brain's downward gravitational weight, reducing the net effective mechanical load from $\sim 14.5\text{ N}$ to a gentle $\sim 0.56\text{ N}$, preventing brainstem herniation.
- During neurosurgical craniotomy (Chen et al. 2017, 2019, 2022), CSF drainage lowers the fluid level ($h_{\text{fluid}}$), stripping away buoyant relief and causing gravitational parenchymal sag and subcortical tumor migration up to $10 - 15\text{ mm}$.

### 8.2 Surface Facet Hydrostatic Pressure Formulation (`HydrostaticPressureLoad`)
The hydrostatic fluid pressure at spatial coordinate $\boldsymbol{x}$ is:
$$p(\boldsymbol{x}) = \rho_{\text{fluid}} \|\boldsymbol{g}\| \max\left(0,\, h_{\text{fluid}} - \boldsymbol{x} \cdot \hat{\boldsymbol{u}}_{\text{up}}\right)$$
where $\hat{\boldsymbol{u}}_{\text{up}} = -\boldsymbol{g} / \|\boldsymbol{g}\|$ is the upward unit vector opposite to gravity, and $h_{\text{fluid}}$ is the elevation of the fluid free surface.

The inward normal surface traction on exterior boundary facets is:
$$\boldsymbol{t}(\boldsymbol{x}) = - p(\boldsymbol{x}) \boldsymbol{n}(\boldsymbol{x})$$

For each triangular boundary facet with area $A_k$ and outward unit normal $\boldsymbol{n}_k$, the exact 2nd-order surface quadrature yields the nodal force on facet vertex $i \in \{1, 2, 3\}$:
$$\boldsymbol{F}_i^{\text{facet}} = -\boldsymbol{n}_k A_k \left( \frac{p_i + p_1 + p_2 + p_3}{12} \right)$$

By Gauss's divergence theorem, the net vertical resultant over a closed submerged surface identically recovers Archimedes' buoyant force to machine precision ($< 10^{-12}$ relative error):
$$\oint_{\partial \Omega} -p \boldsymbol{n} \, dA = \rho_{\text{fluid}} V_{\text{sub}} (-\boldsymbol{g})$$
while simultaneously capturing the lateral hydrostatic compression (squeezing) of the continuum.

### 8.3 Volumetric Buoyancy Body Force (`BuoyancyLoad`)
As an upward volumetric body force opposing gravity:
$$\boldsymbol{F}_{\text{buoyancy}} = -\rho_{\text{fluid}} \int_{\Omega_{\text{sub}}} \boldsymbol{g} \, dV$$
For fully submerged continua, evaluated via the assembled mass matrix:
$$\boldsymbol{F}_{\text{buoyancy}} = \frac{\rho_{\text{fluid}}}{\rho_{\text{solid}}} \boldsymbol{M} (-\boldsymbol{g})$$
or via tetrahedral element volume summation for partially submerged bodies.

### 8.4 Usage in Forward Analyses

```python
from psierra.boundary import GravityLoad, HydrostaticPressureLoad, BuoyancyLoad

# 1. Downward gravity body force
gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
f_gravity = gravity.compute_force_vector(M)

# 2. Upward buoyancy load
buoyancy = BuoyancyLoad(
    fluid_density=1000e-12,  # Water / CSF: 1000 kg/m^3 in tonne/mm^3
    acceleration=(0.0, 0.0, -9810.0),
    fluid_level=80.0,
)
f_buoyancy = buoyancy.compute_force_vector(domain)

# 3. Net submerged force vector
f_net = f_gravity + f_buoyancy
```

### 8.5 Verification Examples
- **[`1106_hydrostatic_buoyancy_beam.py`](../examples/11_advanced_continua/1106_hydrostatic_buoyancy_beam.py)**: Cantilever beam under gravity and water buoyancy across all 4 analyses (Linear Static, Linear Dynamic, TL Static, TL Dynamic), demonstrating a 50% buoyant sag reduction.
- **[`1107_hydrostatic_buoyancy_brain.py`](../examples/11_advanced_continua/1107_hydrostatic_buoyancy_brain.py)**: Patient-specific MRI brain shift under gravity and CSF buoyancy across all 4 analyses, demonstrating $96.2\%$ buoyant gravitational relief and subcortical tumor drift telemetry.

---

## 9. Neurosurgical Burr Hole Brain Shift: Cortical Bulging vs. Gravitational Sagging

### 9.1 Clinical Biomechanics
During cranial burr hole surgery (e.g. chronic subdural hematoma drainage, stereotactic biopsy, ventriculostomy, or deep brain stimulation lead placement), two opposing deformation phenomena occur depending on intracranial pressure and patient posture:

1. **Cortical Bulging (Herniation / Prolapse)**:
   - *Etiology*: Intracranial hypertension, cerebral edema, hypercapnia, or subcortical tumor mass effect.
   - *Mechanics*: The rigid skull and tough dura mater normally prevent outward expansion. Drilling a $12 - 16\text{ mm}$ burr hole releases the displacement constraint ($u_n = 0$) over that circular patch. Internal turgor and ventricular pressure push the parenchyma outward, causing tissue to mushroom through the aperture into the skull defect.
   - *Clinical Risk*: Cortical strangulation, vascular compromise, and parenchymal infarction against the sharp bony edge of the burr hole.

2. **Cortical Sagging (Cerebral Collapse / Pneumocephalus)**:
   - *Etiology*: Patient in supine / semi-sitting posture with a superior nondependent burr hole (e.g., Kocher's point).
   - *Mechanics*: Dural incision allows CSF to drain out rapidly while air enters the subdural space. Draining CSF destroys Archimedean buoyancy ($96.2\%$ gravitational relief lost). Under the unbuoyed downward pull of gravity ($14.5\text{ N}$), the cerebral hemisphere sags down into the occipital skull base, dropping $5 - 12\text{ mm}$ away from the burr hole opening.
   - *Clinical Risk*: Stereotactic targeting error (2–4 mm shift in deep brain nuclei), cortical bridging vein tension, and acute subdural hemorrhage.

### 9.2 Implementation & Verification (`1108_burr_hole_brain_shift.py`)
Simulation 1108 provides an end-to-end clinical digital twin:
- **Kocher's Point Burr Hole**: Right frontal convexity ($X=26.5, Y=40.0, Z=43.0\text{ mm}$, radius $R=8.0\text{ mm}$, diameter $16.0\text{ mm}$).
- **Exported 3D Aperture Geometry**:
  - `burr_hole_bone_collar.vtp`: 3D annular cranial bone table collar showing the exact skull perforation thickness ($5.0\text{ mm}$).
  - `burr_hole_perimeter_ring.vtp`: Circular contour wireframe marking the aperture rim.
  - `cranial_meninges.vtp`: Dural surface mesh with boolean scalar field `burr_hole_aperture`.
- **Solver Strategies**:
  - **Linear Static Analysis**: Fast, direct sparse solve ($\sim 5\text{ s}$ runtime).
  - **Dynamic Relaxation Analysis**: Matrix-free explicit kinetic relaxation ($< 0.5\%$ error relative to direct static).
- **Results Summary**:
  - Bulging Case: $+4.422\text{ mm}$ outward protrusion mushrooming through the burr hole ring.
  - Sagging Case: $-4.589\text{ mm}$ inward collapse opening a pneumocephalus air gap under the burr hole.



