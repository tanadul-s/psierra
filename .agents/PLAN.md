# Project pSierra Development Plan & Roadmap

This document outlines the strategic milestones, component delivery roadmap, and active task tracking for Project `pSierra`.

---

## Strategic Roadmap Overview

```
[ Phase 1: Linear FEM Core ]
            │
            ▼
[ Phase 2: Non-Linear Continua & Hyperelasticity ]
            │
            ▼
[ Phase 3: Contact Mechanics & Interactive Collision ]
            │
            ▼
[ Phase 4: Model Order Reduction (MOR) & Hyper-Reduction ]
            │
            ▼
[ Phase 5: Inverse FEM (iFEM) & Parameter Identification ]
            │
            ▼
[ Phase 6: Biomechanical Specialization & Multiphysics ]
```

---

## Phase 1: Linear FEM Foundation & Infrastructure (Completed)

- [x] Repository setup with Python 3.12, `uv`, and `pyproject.toml`.
- [x] Basic mesh abstractions (`psierra/mesh/surface.py`, `psierra/mesh/volumetric.py`).
- [x] Linear tetrahedral element formulation (`psierra/element/tet4.py`).
- [x] Isotropic linear elastic constitutive law (`psierra/material/hookean.py`).
- [x] Sparse matrix assembly infrastructure (`psierra/utility/assembly.py`).
- [x] Linear static solver with Dirichlet and load boundaries (`psierra/solver/linear_static.py`).
- [x] Linear dynamic solver with Rayleigh damping (`psierra/solver/linear_dynamic.py`, `psierra/damping/rayleigh.py`).
- [x] End-to-end runnable demonstration scripts in `examples/` authored by `developer` with PyVista 3D visualization.
- [x] **Tiered Automated Test Suite in `tests/`**:
  - [x] `tests/test_material.py`: Parameter conversions, Hookean/Neo-Hookean stress & tangent consistency via finite difference ($< 10^{-4}$).
  - [x] `tests/test_element_tet4.py`: Volume calculation, shape function partition of unity, B-matrix, patch test ($< 10^{-14}$), rigid body translation & rotation invariance ($< 10^{-10}$).
  - [x] `tests/test_mesh.py`: VolumetricMesh & SurfaceMesh loading, boundary extraction, volume/area, coordinate transforms.
  - [x] `tests/test_boundary.py`: Dirichlet constraint indexing, DOF reduction, lifecycle (`add`/`remove`/`clear`), nodal force & gravity load assembly.
  - [x] `tests/test_solvers_linear.py`: DirectSolver, FactorizedSolver, and ConjugateGradientSolver solving $A x = b$.
  - [x] `tests/test_analysis_linear.py`: Cantilever beam deflection benchmark against analytical solution ($< 5\%$), equilibrium balance ($< 10^{-8}$), Rayleigh dynamic decay.
  - [x] `tests/test_analysis_nonlinear.py`: Nonlinear static Newton-Raphson convergence, dynamic integration with Newmark-$\beta$.
- [x] Educational LaTeX theory documents authored by `teacher` (see [`docs/`](../docs/)).

---

## Phase 2: Non-Linear Mechanics & Large Deformations

- [x] **Quadratic Tetrahedral Element (`Tet10`) (Immediate Phase 2 Target)**:
  - Higher-order 10-node isoparametric tetrahedron with quadratic shape functions (`psierra/element/tet10.py`).
  - Alleviates artificial shear and volumetric locking in complex anatomical meshes (brain, tumor, liver).
  - Shape function gradient evaluations, numerical quadrature (4-point / 5-point Gauss rules in `psierra/element/quadrature.py`), and B-matrix.
  - Exact analytical consistent mass matrix integration.
  - Total Lagrangian formulation with consistent tangent stiffness matrix ($k_{tan}$ matches finite difference to $< 10^{-9}$).
  - Full suite of 13 unit tests in `tests/test_element_tet10.py` and cantilever shear locking alleviation benchmark ($1.87\times$ flexibility over Tet4) in `tests/test_analysis_linear.py`.
- [x] **Triangular Structural Shell Elements (`Shell3` / `Shell6`)**:
  - 3-node linear (`Shell3`) and 6-node quadratic (`Shell6`) triangular shell elements matching surface boundary faces of solid tetrahedral meshes (`Tet4`, `Tet10`).
  - `Shell6` provides 100% conforming quadratic interface with `Tet10` solid volume without geometric gap or faceting error.
  - Formulations:
    - Discrete Kirchhoff Triangle (`DKT`): thin membranes (dura mater, falx cerebri, tentorium, arterial walls) implemented in `Shell3` with CST membrane and Hughes-Brezzi drilling stabilization.
    - Reissner-Mindlin with Hughes-Brezzi drilling penalty: moderately thick structures (cranial skull/calvarium, cartilage) implemented in `Shell6` with LST quadratic membrane.
  - 6 DOFs per node ($u_x, u_y, u_z, \theta_x, \theta_y, \theta_z$) with drilling rotation stabilization for seamless 3D node-sharing coupling with solid continua (`Tet4`, `Tet10`).
  - Full suite of 14 tests in `tests/test_element_shell3.py` and `tests/test_element_shell6.py`:
    - Exact stiffness matrix symmetry $K = K^T$.
    - Exact 6 zero rigid-body modes ($< 10^{-8}$) and spectral gap ($> 10^8$) in arbitrary 3D space.
    - Pure translation and infinitesimal rotation invariance ($u^T K u \approx 0$).
    - $C^0$ interface conformity test between `Tet10` solid face and `Shell6` to machine precision ($10^{-14}$).
    - Analytical Euler-Bernoulli cantilever plate bending benchmarks: `Shell3` captures $>93\%$ flexibility on 2 elements; `Shell6` demonstrates monotonic convergence ($75.5\% \to 94.2\% \to 99.0\%$).
    - Mass conservation and positive definiteness of consistent mass matrices.
- [ ] Kinematic formulations:
  - Total Lagrangian (TL) with Green-Lagrange strain $E$ and 2nd Piola-Kirchhoff stress $S$.
  - Updated Lagrangian (UL) with Almansi strain and Cauchy stress.
- [ ] Hyperelastic constitutive models:
  - Compressible and nearly incompressible Neo-Hookean.
  - Mooney-Rivlin, Ogden, and Yeoh strain energy functions.
- [ ] Non-linear solver algorithms:
  - Newton-Raphson iteration with consistent tangent stiffness matrix.
  - Arc-length / Riks methods for post-buckling and softening paths.
  - Implicit non-linear dynamics with Newmark-$\beta$ and Generalized-$\alpha$ schemes.

---

## Phase 3: Contact Mechanics & Interactive Collision (Before MOR)

- [ ] **Surface Extraction & Proximity Detection**:
  - Boundary surface facet extraction from 3D tetrahedral/hexahedral meshes.
  - Fast collision detection with Bounding Volume Hierarchy (BVH) / Axis-Aligned Bounding Boxes (AABB).
  - Ray-tracing approach (Möller-Trumbore ray-triangle intersection) for high-speed signed distance, penetration depth ($g_n < 0$), and gap determination along surface normals.
  - Point-to-surface and surface-to-surface proximity / penetration distance calculations.
- [ ] **Contact Formulations**:
  - Penalty method for normal contact constraints.
  - Augmented Lagrangian multiplier method for exact non-penetration.
  - Tangential friction models (Coulomb friction with regularized slip/stick transitions).
- [ ] **Surgical Interaction & Boundary Contact**:
  - Rigid and compliant tool-tissue interaction (virtual palpation, indenter contact, surgical clamp grasping).
  - Consistent tangent matrices for contact contributions in Newton-Raphson iterations.

---

## Phase 4: Model Order Reduction (MOR) & Hyper-Reduction

*Objective*: Accelerate high-dimensional non-linear and contact-aware FEM to real-time execution (>30–1000 Hz) for medical digital twins and surgical navigation.

- [ ] **Offline Snapshot Management & Basis Construction**:
  - Full-Order Model (FOM) forward trajectory simulations with contact and deformation.
  - Proper Orthogonal Decomposition (POD / Snapshot PCA) with SVD.
  - Truncation based on cumulative energy spectrum $\sum \sigma_i / \sum \sigma_{all} \ge 0.999$.
  - Orthonormal reduced basis verification ($V^T V = I$).
- [ ] **Linear Reduced Order Modeling (Linear ROM)**:
  - Galerkin projection of linear stiffness, mass, and damping matrices:
    $$K_r = V^T K V, \quad M_r = V^T M V, \quad C_r = V^T C V$$
  - Component mode synthesis (Craig-Bampton, Guyan reduction).
  - Real-time online dynamic integration.
- [ ] **Nonlinear Hyper-Reduction**:
  - Discrete Empirical Interpolation Method (DEIM / Q-DEIM) for point-wise nonlinear internal force evaluation without full-mesh quadrature.
  - Energy-Conserving Sampling and Weighting (ECSW) for stable, energy-preserving element-level hyper-reduction in nonlinear solid dynamics.
- [ ] **Online Execution & Interactive Reconstruction**:
  - Ultra-fast online solver in low-dimensional generalized coordinates: $u \approx V u_r$.
  - Real-time surface displacement projection for interactive visual rendering and haptic feedback loops.

---

## Phase 5: Inverse FEM (iFEM) & Parameter Identification

*Objective*: Infer full-field deformations, unmeasured loads, patient-specific material properties, and stress-free configurations from sparse physical measurements.

- [ ] **Inverse Shape Sensing & Displacement Reconstruction (iFEM)**:
  - Reconstruct full 3D displacement and strain fields from sparse surface discrete sensors (e.g., optical markers, stereovision tracking, ultrasound probe points, fiber Bragg grating strain sensors).
  - Weighted least-squares variational functional matching experimental surface observations with FEM displacement fields without needing prior knowledge of external loading.
- [ ] **Inverse Material Parameter Identification**:
  - Optimization and adjoint-based inverse problem solvers to calibrate patient-specific constitutive parameters (e.g., Young's modulus, hyperelastic parameters $C_{10}, D_1$) from measured force-displacement curves (e.g., indentation or aspiration tests).
- [ ] **Stress-Free Reference Configuration Estimation**:
  - Inverse elastostatic solver to determine the in vivo zero-pressure / undeformed reference geometry from clinical scans of pre-stressed tissues (e.g., pressurized aorta, gravitationally deformed brain).

---

## Phase 6: Biomechanical Specialization & Multiphysics

- [ ] Time-dependent and porous tissue constitutive formulations:
  - Linear and non-linear viscoelasticity (Prony series, quasi-linear viscoelasticity - QLV).
  - Poroelasticity (Biot formulation for fluid-saturated porous soft tissues).
  - Hybrid hyper-visco-poroelastic models for brain and liver tissues.
- [ ] Anatomical boundary conditions:
  - Pulsatile hemodynamic pressure loading for aortic walls.
  - Skull boundary constraints and cerebrospinal fluid (CSF) interactions.
- [ ] Multiphysics coupling:
  - Vibro-acoustics simulation.
  - Fluid-Structure Interaction (FSI) coupling interfaces.

---

## Extension & Secondary Roadmap: Structured Hexahedral Mechanics (Hex8, Hex20)

*Objective*: Advanced structured hexahedral discretizations for prismatic and mapped CAD domains where structured block meshing is applicable.

- [ ] Hexahedral volume meshing infrastructure (structured sweeping, transfinite interpolation, polycube mapping).
- [ ] Linear 8-node brick element (`Hex8`) with B-bar / Selective Reduced Integration (SRI) to prevent volumetric locking.
- [ ] Quadratic 20-node serendipity brick element (`Hex20`) with 27-point / 14-point Gauss quadrature.
- [ ] Quadrilateral shell element (`Shell4` / MITC4) for structured quadrilateral surface patches.

