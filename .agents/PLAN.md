# Project pSierra Development Plan & Roadmap

This document outlines the strategic milestones, component delivery roadmap, and active task tracking for Project `pSierra`.

---

## Strategic Vision: Fast-Track to Inverse FEM (iFEM)

The primary goal of Project `pSierra` is to **close the complete work loop from forward simulation to Inverse FEM (iFEM) as quickly as possible**:
$$\text{Forward Continuum Mechanics} \longrightarrow \text{Contact Mechanics} \longrightarrow \text{Model Order Reduction (MOR)} \longrightarrow \text{Inverse FEM (iFEM)}$$

### Guiding Execution Strategy

* **Lean, High-Value Core Implementations**: For Phases 2–5, implement the robust, battle-tested numerical foundations that provide immediate engineering value and close the pipeline loop.
* **Defer Complex Research Overhead**: Defer complex, high-friction formulations (such as augmented Lagrangian mortar contact, nonlinear ECSW hyper-reduction, and multi-phase poroelasticity) to the **Post-Loop Enhancement Roadmap (Phase 6+)** to prevent stalling momentum on the highway to iFEM.

```
[ Phase 1: Linear FEM Core ] (Completed)
            │
            ▼
[ Phase 2: Non-Linear Continua & Hyperelasticity ] (In Finalization)
            │
            ▼
[ Phase 3: Contact Mechanics (Penalty & Augmented Lagrangian) ]
            │
            ▼
[ Phase 4: Model Order Reduction (POD-Galerkin, DEIM & ECSW Hyper-Reduction) ]
            │
            ▼
[ Phase 5: Inverse FEM (iFEM) & Parameter Identification ] ★ Primary End Goal ★
            │
            ▼
[ Phase 6+: Post-Loop Enhancements & Biomechanical Multiphysics ]
```

---

## Phase 1: Linear FEM Foundation & Infrastructure (Completed)

* [x] Repository setup with Python 3.12, `uv`, and `pyproject.toml`.
* [x] Basic mesh abstractions (`psierra/mesh/surface.py`, `psierra/mesh/volumetric.py`).
* [x] Linear tetrahedral element formulation (`psierra/element/tet4.py`).
* [x] Isotropic linear elastic constitutive law (`psierra/material/hookean.py`).
* [x] Sparse matrix assembly infrastructure (`psierra/utility/assembly.py`).
* [x] Linear static solver with Dirichlet and load boundaries (`psierra/solver/linear_static.py`).
* [x] Linear dynamic solver with Rayleigh damping (`psierra/solver/linear_dynamic.py`, `psierra/damping/rayleigh.py`).
* [x] End-to-end runnable demonstration scripts in `examples/` authored by `developer` with PyVista 3D visualization.
* [x] **Tiered Automated Test Suite in `tests/`**:
  * [x] `tests/test_material.py`: Parameter conversions, Hookean/Neo-Hookean stress & tangent consistency via finite difference ($< 10^{-4}$).
  * [x] `tests/test_element_tet4.py`: Volume calculation, shape function partition of unity, B-matrix, patch test ($< 10^{-14}$), rigid body translation & rotation invariance ($< 10^{-10}$).
  * [x] `tests/test_mesh.py`: VolumetricMesh & SurfaceMesh loading, boundary extraction, volume/area, coordinate transforms.
  * [x] `tests/test_boundary.py`: Dirichlet constraint indexing, DOF reduction, lifecycle (`add`/`remove`/`clear`), nodal force & gravity load assembly.
  * [x] `tests/test_solvers_linear.py`: DirectSolver, FactorizedSolver, and ConjugateGradientSolver solving $A x = b$.
  * [x] `tests/test_analysis_linear.py`: Cantilever beam deflection benchmark against analytical solution ($< 5\%$), equilibrium balance ($< 10^{-8}$), Rayleigh dynamic decay.
  * [x] `tests/test_analysis_nonlinear.py`: Nonlinear static Newton-Raphson convergence, dynamic integration with Newmark-$\beta$.
* [x] Educational LaTeX theory documents authored by `teacher` (see [`docs/`](../docs/)).

---

## Phase 2: Non-Linear Mechanics & Large Deformations

* [x] **Quadratic Tetrahedral Element (`Tet10`) (Immediate Phase 2 Target)**:
  * Higher-order 10-node isoparametric tetrahedron with quadratic shape functions (`psierra/element/tet10.py`).
  * Alleviates artificial shear and volumetric locking in complex anatomical meshes (brain, tumor, liver).
  * Shape function gradient evaluations, numerical quadrature (4-point / 5-point Gauss rules in `psierra/element/quadrature.py`), and B-matrix.
  * Exact analytical consistent mass matrix integration.
  * Total Lagrangian formulation with consistent tangent stiffness matrix ($k_{tan}$ matches finite difference to $< 10^{-9}$).
  * Full suite of 13 unit tests in `tests/test_element_tet10.py` and cantilever shear locking alleviation benchmark ($1.87\times$ flexibility over Tet4) in `tests/test_analysis_linear.py`.
* [x] **Triangular Structural Shell Elements (`Shell3` / `Shell6`)**:
  * 3-node linear (`Shell3`) and 6-node quadratic (`Shell6`) triangular shell elements matching surface boundary faces of solid tetrahedral meshes (`Tet4`, `Tet10`).
  * `Shell6` provides 100% conforming quadratic interface with `Tet10` solid volume without geometric gap or faceting error.
  * Formulations:
    * Discrete Kirchhoff Triangle (`DKT`): thin membranes (dura mater, falx cerebri, tentorium, arterial walls) implemented in `Shell3` with CST membrane and Hughes-Brezzi drilling stabilization.
    * Reissner-Mindlin with Hughes-Brezzi drilling penalty: moderately thick structures (cranial skull/calvarium, cartilage) implemented in `Shell6` with LST quadratic membrane.
  * 6 DOFs per node ($u_x, u_y, u_z, \theta_x, \theta_y, \theta_z$) with drilling rotation stabilization for seamless 3D node-sharing coupling with solid continua (`Tet4`, `Tet10`).
  * Full suite of 14 tests in `tests/test_element_shell3.py` and `tests/test_element_shell6.py`:
    * Exact stiffness matrix symmetry $K = K^T$.
    * Exact 6 zero rigid-body modes ($< 10^{-8}$) and spectral gap ($> 10^8$) in arbitrary 3D space.
    * Pure translation and infinitesimal rotation invariance ($u^T K u \approx 0$).
    * $C^0$ interface conformity test between `Tet10` solid face and `Shell6` to machine precision ($10^{-14}$).
    * Analytical Euler-Bernoulli cantilever plate bending benchmarks: `Shell3` captures $>93\%$ flexibility on 2 elements; `Shell6` demonstrates monotonic convergence ($75.5\% \to 94.2\% \to 99.0\%$).
    * Mass conservation and positive definiteness of consistent mass matrices.
* [x] **Modular Lego-Brick Element Architecture**:
  * Class naming convention: `[Element][Formulation]` (e.g., `Tet4Linear`, `Tet4Corotational`, `Tet4TotalLagrangian`, `Tet4UpdatedLagrangian`).
  * Base parent classes: `Tet4Base`, `Tet10Base`, `Shell3Base`, `Shell6Base`.
  * Zero aliases in `src/psierra/element/__init__.py`.
* [x] **Kinematic Formulations**:
  * [x] **Linear Small-Strain**: Infinitesimal strain $\boldsymbol{\varepsilon} = \operatorname{sym}(\nabla \boldsymbol{u})$ (`Tet4Linear`, `Tet10Linear`, `Shell3Linear`, `Shell6Linear`).
  * [x] **Corotational (CR)**: Polar decomposition $F = R U$ filtering rigid rotations for large displacement/rotation small strain problems (`Tet4Corotational`, `Shell3Corotational`).
  * [x] **Total Lagrangian (TL)**: Reference configuration $X$ kinematics with Green-Lagrange strain $E$ and 2nd Piola-Kirchhoff stress $S$ (`Tet4TotalLagrangian`, `Tet10TotalLagrangian`).
  * [x] **Updated Lagrangian (UL)**: Spatial current configuration $x$ kinematics with spatial gradients $h = H F^{-1}$, Cauchy stress $\sigma = \frac{1}{J} F S F^T$, spatial tangent moduli $c^{\text{spatial}}$, and spatial geometric stiffness $k_{\text{geo}}$ (`Tet4UpdatedLagrangian`, `Tet10UpdatedLagrangian`).
  * [x] Machine-precision equivalence between TL and UL verified to $< 10^{-13}$ mm.
* [x] **Hyperelastic Constitutive Models**:
  * [x] Compressible Neo-Hookean hyperelasticity with exact analytical tangent and 2nd PK stress (`NeoHookean`).
  * [x] **Yeoh Model (`Yeoh`)** (Soft Tissue Strain-Stiffening):
    * 3-parameter reduced polynomial formulation: $\Psi = \sum_{i=1}^3 C_{i0} (\bar{I}_1 - 3)^i + \frac{1}{2} K (\ln J)^2$.
    * Captures physiological exponential strain-stiffening of soft biological tissues (liver, brain, kidney) at large strains.
    * Pure closed-form invariant tensor derivatives: directly leverages existing `NeoHookean` $\partial \bar{I}_1 / \partial \boldsymbol{C}$ and $\partial^2 \bar{I}_1 / \partial \boldsymbol{C} \partial \boldsymbol{C}$ kernels.
  * [x] **Mooney-Rivlin Model (`MooneyRivlin`)**:
    * 2-parameter formulation: $\Psi = C_{10} (\bar{I}_1 - 3) + C_{01} (\bar{I}_2 - 3) + \frac{1}{2} K (\ln J)^2$.
    * Analytical derivatives for second invariant $\bar{I}_2$: $\partial I_2 / \partial \boldsymbol{C} = I_1 \boldsymbol{I} - \boldsymbol{C}$.
    * Standard benchmark for synthetic elastomers, silicones, and rubber phantoms in surgical training.
* [ ] **Non-linear Solver Algorithms & Analysis**:
  * [x] Incremental Newton-Raphson iteration with backtracking line search (`NonlinearStaticAnalysis`).
  * [x] Implicit non-linear dynamics with Newmark-$\beta$ integration and Rayleigh damping (`NonlinearDynamicAnalysis`).
  * [x] **Dynamic Steady-State Equilibrium Convergence**:
    * Automatic early stopping when static out-of-balance residual $\|\boldsymbol{f}_{\text{ext}} - \boldsymbol{f}_{\text{int}}\| / \|\boldsymbol{f}_{\text{ext}}\| \le \text{steady\_state\_tol}$ holds for `consecutive_hold_steps`.
    * Integrated across `LinearDynamicAnalysis` and `NonlinearDynamicAnalysis`.
    * Guaranteed export of final snapshot upon reaching steady state.
  * [x] **Unified Snapshot Stride Management**:
    * Snapshot saving period (`save_interval`) and PVD streaming across all static and dynamic solvers (`LinearStaticAnalysis`, `NonlinearStaticAnalysis`, `LinearDynamicAnalysis`, `NonlinearDynamicAnalysis`).
  * [x] **Generalized-$\alpha$ Dynamic Scheme (`LinearDynamicAnalysis` & `NonlinearDynamicAnalysis`)**:
    * Controllable numerical high-frequency damping parameterized by single physical knob: high-frequency spectral radius $\rho_\infty \in [0, 1]$.
    * Deprecated and removed legacy $\beta, \gamma$ parameters in favor of clean single-parameter control.
    * Preserves unconditional stability and strict 2nd-order accuracy $\mathcal{O}(\Delta t^2)$ while filtering out spurious mesh discretization chatter during fast dynamics and contact impact.
    * Full mapping: $\rho_\infty = 1.0$ (Undamped Newmark), $\rho_\infty = 0.8$ (Default Generalized-$\alpha$), $\rho_\infty = 0.5$ (HHT-$\alpha$), $\rho_\infty = 0.0$ (Asymptotic Annihilation).
    * Verified via `examples/05_simulation/sim16_dynamic_time_integration_comparison.py`.

---

## Milestone 2.5: Interactive / Online Dynamic Simulation Framework (Lean Prototype)

*Objective*: Establish a lean, real-time dynamic simulation pipeline and axial torsion / dynamic demonstration runner before introducing contact mechanics.

* [x] **Online Dynamic Simulation Runner (`InteractiveDynamicSimulation`)**:
  * Real-time dynamic loop operating with fixed time step $\Delta t$ decoupled from rendering frame rate.
  * High-performance sparse factorization reuse: pre-factorized linear dynamic step or cached Cholesky/LU factorizations for low computational overhead ($>60$ FPS, $>10,000$ Hz physics rate).
  * Real-time execution monitor tracking wall-clock simulation speed ratio (Sim Time / Real Time).
* [x] **Interactive Dynamic Load & Boundary Perturbations**:
  * Dynamic external load generators (harmonic excitation, moving point forces).
  * Interactive user disturbance interface (keyboard impulse triggers, mouse-pick surface node forces).
* [x] **Live 3D Visualization Pipeline**:
  * Real-time PyVista integration with non-blocking render loop.
  * In-place point coordinate mutation (`polydata.points = current_nodes`) to bypass expensive VTK object recreation.
* [x] **Deliverable Demonstrations & Capability Proofs**:
  * [x] `examples/05_simulation/sim17_axial_torsion_hyperelastic_beam.py`: Severe axial twisting benchmark of a 3D continuum cantilever beam (up to $90^\circ - 180^\circ$ rotation) to prove finite-strain kinematics, cross-sectional warping, Poynting axial shortening/lengthening, and hyperelastic stability under large torsional shear.
  * [x] `examples/05_simulation/sim18_tet10_nonlinear_hyperelastic_beam.py`: Finite-strain hyperelastic beam bending benchmark proving quadratic Tet10 shear-locking alleviation ($1.51\times$ compliance recovery) and TL/UL machine precision equivalence ($< 10^{-13}$ mm).
  * [x] `examples/06_interactive/sim01_online_dynamic_cantilever.py`: Unified real-time interactive cantilever beam supporting linear, corotational, Total Lagrangian, and Updated Lagrangian kinematics with Tet4 and Tet10 elements under user-controlled dynamic disturbances (impulse, harmonic).
  * [x] `examples/06_interactive/sim02_online_dynamic_axial_torsion.py`: Real-time interactive dynamic simulation of 3D continuum axial torsion under user-controlled dynamic twisting moments ($M_z$), torsional wave propagation, and elastic spring-back.

---

## Phase 3: Multi-Body Systems & Advanced Contact Mechanics

*Objective*: Establish an Abaqus-like **Assembly Layer (`MultiBodySystem`)** and a progressive contact mechanics framework capable of managing multiple independent and coupled meshes interacting in 3D space. The primary clinical/biomechanical vision is the **interaction between brain parenchyma, tumor lesion, skull boundary, and surgical instruments** (e.g. brain resting in skull cavity, pressed by an indenter tool).

### Architecture Principles & Clarifications:
1. **MultiBody Assembly Layer**: `MultiBodySystem` acts as the top-level assembly container (analogous to `*ASSEMBLY` / `*INSTANCE` in Abaqus), combining multiple domains (single or coupled) with global DOF slicing, block-diagonal mass/stiffness assembly, and state synchronization. Solvers operate on `MultiBodySystem` transparently via the standard `Domain` interface.
2. **Mesh-to-Mesh $\neq$ Deformable-to-Deformable**: *Mesh-to-mesh* refers to geometry representation—importing realistic 3D surface meshes (e.g., VTK) instead of idealized analytical primitives. Both Deformable-to-Rigid (e.g., beam impacting imported `floor1.vtk` or brain resting in `skull.vtk`) and Deformable-to-Deformable (two flexible bodies) can be mesh-to-mesh.
3. **Primary Mesh Format is VTK**: The primary 3D mesh input format across Project Sierra is **VTK** (`.vtk`, `.vtu`, `.vtp`), read cleanly via PyVista.
4. **Simple & Robust CLI**: All simulation and interactive scripts use minimal, standard `argparse` (or simple `click`) with `--help`.
5. **Unified Visualization Standard**: All domain implementations store displacements strictly as a 3D vector array in `point_data["Displacement"]` and set the active scalar to `"Displacement"`.

---

### Phase 3 Loop 0: Infrastructure & Preparatory Cleanups (Completed)
* [x] **Unified Point Data Standard**:
  * Standardized `point_data["Displacement"]` strictly as a 3D vector array in `solid.py`, `shell.py`, `coupled.py`, and `interactive_dynamic.py`.
  * Set default active scalar to `"Displacement"`. Removed redundant scalar decompositions (`_X`, `_Y`, `_Z`, `_Mag`).
* [x] **VTK Surface Mesh Reader**:
  * Implemented PyVista VTK-first reader in `psierra/mesh/surface.py` with clean PolyData extraction (preventing cell array dimension mismatches).
* [x] **MultiBody Assembly Container Blueprint (`psierra/multibody/system.py`)**:
  * Created `MultiBodySystem` container inheriting from `Domain`.
  * Implemented global DOF slicing, block-diagonal sparse mass/stiffness assembly, internal force concatenation, and multi-body VTU export with `Body ID`.
* [x] **MultiBody Unit Test Suite**:
  * Created `tests/test_multibody.py` with 6 unit tests verifying DOF mapping, state propagation, block matrix assembly, static solve, and VTK export (all passing).

---

### Phase 3 Loop 1: Minimal Working Foundation (Deformable-to-Rigid, BVH, Penalty Contact) (Completed)
*Objective*: Build a simple, complete, end-to-end working loop: MultiBody container $\to$ collision detection $\to$ deformable-to-rigid contact $\to$ penalty contact formulation $\to$ static and dynamic benchmarks.

* [x] **Collision Detection Engine (`psierra/collision/`)**:
  * `shapes.py`: Analytical collision shapes (`CollisionShape`, `RigidPlane`, `RigidSphere`, `RigidCylinder`) with signed gap and outward normal calculation.
  * `raytrace.py`: Intel Embree BVH accelerated ray-mesh obstacle (`RayMeshObstacle`) and PyVista cell locator for imported 3D VTK meshes (`floor1.vtk`).
  * `detector.py`: Proximity and closest-point projection (`CollisionDetector`) computing signed gaps $g_N$ and surface normals $\boldsymbol{n}$.
* [x] **Contact Formulation (`psierra/boundary/contact.py`)**:
  * `RigidContact`:
    * Normal penalty force: $\boldsymbol{f}_c = -k_p g_N \boldsymbol{n}$ (active when $g_N < 0$).
    * Contact velocity damping: $\boldsymbol{f}_d = -c_p (\dot{\boldsymbol{u}} \cdot \boldsymbol{n}) \boldsymbol{n}$ to eliminate contact chatter.
    * Symmetric tangent stiffness contribution: $\boldsymbol{K}_c = k_p (\boldsymbol{n} \otimes \boldsymbol{n})$.
    * Contact potential energy tracking: $\Pi_c = \frac{1}{2} k_p \sum g_N^2$.
* [x] **Solver Contact Integration**:
  * Integrate `contact` into `NonlinearStaticAnalysis` (residual $\boldsymbol{R} = \boldsymbol{F}_{\text{int}} - \boldsymbol{F}_{\text{ext}} - \boldsymbol{F}_c$, tangent stiffness $\boldsymbol{K} + \boldsymbol{K}_c$).
  * Integrate `contact` into `NonlinearDynamicAnalysis` (generalized-$\alpha$ predictor-corrector with contact force and stiffness).
* [x] **Loop 1 Unit Tests**:
  * `tests/test_collision_and_contact.py`: Collision primitives, VTK mesh ray-tracing, detector gaps, penalty force linearization, and tangent stiffness symmetry (all 9 tests passing).
* [x] **Loop 1 Deliverable Demonstrations**:
  * `examples/05_simulation/sim19_contact_static_rigid_floor.py`: Static cantilever beam bending and resting on analytical rigid floor under gravity.
  * `examples/05_simulation/sim20_contact_dynamic_bouncing_beam.py`: Dynamic free-fall, floor impact, penalty energy storage, and elastic rebound.
  * `examples/05_simulation/sim21_contact_rigid_mesh_floor.py`: Deformable beam impacting imported 3D VTK floor mesh (`psierra/data/floor1/mesh/floor1.vtk`).

---

### Phase 3 Loop 2: Advanced Multi-Body Contact (Deformable-to-Deformable & Augmented Lagrangian) (Completed)
*Objective*: Advance contact mechanics to two-way flexible interactions and exact non-penetration.

* [x] **Deformable-to-Deformable (Mesh-to-Mesh) Contact**:
  * Contact between two or more independent flexible bodies with two-way coupling (`DeformableContact`).
  * Fast dynamic master surface tracking via `vtkStaticCellLocator` ($0.136\text{ ms}$ rebuild per step).
  * Exact action-reaction interface equilibrium: $\boldsymbol{f}_s + \sum_{i=1}^3 \boldsymbol{f}_{m_i} = \mathbf{0}$.
  * Off-diagonal cross-body stiffness coupling blocks and symmetric 12x12 rank-1 contact element $\boldsymbol{K}_c^{(e)} = k_p^* (\boldsymbol{w} \otimes \boldsymbol{w}) \otimes (\boldsymbol{n} \otimes \boldsymbol{n})$:
    $$\boldsymbol{K}_{\text{coupled}} = \begin{bmatrix} \boldsymbol{K}_1 + \boldsymbol{K}_{c,11} & \boldsymbol{K}_{c,12} \\ \boldsymbol{K}_{c,21} & \boldsymbol{K}_2 + \boldsymbol{K}_{c,22} \end{bmatrix}, \quad \boldsymbol{K}_{c,12} = \boldsymbol{K}_{c,21}^T$$
* [x] **Augmented Lagrangian Multiplier Formulation**:
  * Normal contact force: $f_N = \max(0, \lambda_N - k_p g_N)$.
  * Multiplier update at step convergence: $\lambda_N^{(k+1)} = \max(0, \lambda_N^{(k)} - k_p g_N^{(k+1)})$.
  * Decouples penetration error from penalty stiffness, achieving exact non-penetration ($g_N \to 0$) without matrix ill-conditioning.
  * Multiplier updates hooked into `NonlinearStaticAnalysis` and `NonlinearDynamicAnalysis` for both `RigidContact` and `DeformableContact`.
* [x] **Regularized Tangential Friction**:
  * Regularized Coulomb stick-slip friction: $\boldsymbol{f}_t = -\mu |f_N| \tanh\left(\frac{\|\boldsymbol{v}_t\|}{v_0}\right) \boldsymbol{t}$.
* [x] **Unit & Integration Test Suite (`tests/test_deformable_contact.py`)**:
  * Barycentric interpolation on triangular facets.
  * Exact Newtonian action-reaction momentum conservation ($\sum \boldsymbol{f}_c = \mathbf{0}$).
  * Coupled tangent stiffness symmetry ($\boldsymbol{K}_c = \boldsymbol{K}_c^T$) and non-zero cross coupling ($\boldsymbol{K}_{AB} \neq \mathbf{0}$).
  * Positive semi-definiteness ($\boldsymbol{v}^T \boldsymbol{K}_c \boldsymbol{v} \ge 0$).
  * Automatic `MultiBodySystem.add_contact_pair()` detection and routing.
  * Augmented Lagrangian multiplier update convergence.
  * Two-body static compression equilibrium solve.
* [x] **Loop 2 Deliverable Demonstration**:
  * `examples/05_simulation/sim22_contact_mesh_to_mesh.py`: Simultaneous dynamic deformation and rebound between two independent flexible meshes (Upper Beam and Lower Compliant Floor) under Generalized-$\alpha$ dynamic time integration. All snapshots saved with unified `factor=0.0` for ParaView "Warp By Vector".

---

### Phase 3 Loop 3: Flagship Biomechanical Multi-Mesh Contact & Self-Contact
*Objective*: High-fidelity anatomical simulation of multi-body and self-contacting biological and geometric domains.

* [x] **Deformable Self-Contact (`SelfContact` in `psierra/boundary/contact.py`)**:
  * Autonomous contact between boundary facets of the same physical domain (essential for sulcal folding, ear compression, tissue invaginations, and geometric self-collision).
  * **Topological Star BFS Graph Expansion**: Multi-ring breadth-first search graph expansion excluding 1-ring through 5-ring topological neighbors, completely eliminating spurious false detections across thin walls/membranes.
  * **C0-Continuous Vertex Normal Interpolation**: Area-weighted vertex normal interpolation on deformed surface triangulation, preventing discontinuous force jumps and lateral chatter across facet boundaries.
  * **Exact Voronoi Triangle Projection**: Ericson's closed-form point-to-triangle projection calculating exact closest surface points and normalized barycentric coordinates.
  * **Strict Action-Reaction Momentum Conservation**: Equal and opposite action on slave node and barycentrically distributed reaction on master facet vertices ($\boldsymbol{f}_s + \sum_{i=1}^3 \boldsymbol{f}_{m_i} = \mathbf{0}$).
  * **Symmetric Positive Semi-Definite Tangent Stiffness**: Rank-1 block updates $\boldsymbol{K}_c = k_p^* (\boldsymbol{w} \otimes \boldsymbol{w}) \otimes (\boldsymbol{n} \otimes \boldsymbol{n})$, strictly symmetric ($\boldsymbol{K}_c = \boldsymbol{K}_c^T$) and translation-invariant.
  * **Uzawa Augmented Lagrangian & Penalty Schemes**: Supports both adaptive multiplier accumulation and virtual elastic penalty springs.
  * **MultiBodySystem Integration**: `MultiBodySystem.add_self_contact(body, ...)` and automatic routing in `add_contact_pair(b, b, ...)` when slave and master bodies coincide.
  * **Automated Unit & Physics Test Suite (`tests/test_self_contact.py`)**: 7 tests validating topological exclusion, exact Voronoi projections, linear momentum conservation ($< 10^{-14}$ relative error), stiffness symmetry ($< 10^{-14}$), Uzawa multiplier updates, and multi-body assembly routing (100% passing).
* [x] **Stanford Bunny Self-Contact Demonstration (`examples/05_simulation/sim23_contact_self_bunny.py`)**:
  * Volumetric tetrahedralization of user-provided `data/bunny1/scan/bunny1.stl` to high-quality watertight mesh `data/bunny1/mesh/bunny1.vtk` (1,419 nodes, 3,852 elements).
  * Canonical workflow blueprint: flat base pedestal clamped on the XY plane ($Z = 0.0$), pure self-weight gravity body load (`GravityLoad`) causing physical downward sagging of the torso, belly, and head, with the slender ears drooping and flopping backwards/downwards ($-54.2\text{ mm}$ mean, $-90.2\text{ mm}$ peak) onto the body surface without any artificial push/squeeze forces or prescribed displacements.
  * Verified robust 5/5 load-step convergence in 21.2s under pure gravity sag and self-contact (Node 766 and Node 828 contacting the body at $Z \approx 28\text{--}33\text{ mm}$, interface penetration restricted to $0.048\text{ mm}$ via Uzawa augmented Lagrangian multipliers).
  * ParaView PVD collection export with strict `factor=0.0` rule and attached 3D `"Displacement"` vector array (`active_scalars = "Displacement"`).
* [ ] **Biomechanical Brain-Skull Multi-Mesh Demonstration**:
  * `examples/05_simulation/sim24_contact_brain_skull_tool.py`:
    * Brain parenchyma (hyperelastic solid domain) resting inside imported rigid skull boundary (`skull.vtk`).
    * Rigid surgical probe / indenter pressing into brain tissue with contact reaction force feedback.
    * Multi-mesh interactive visualization in PyVista and time-series PVD collection for ParaView.

---

## Phase 4: Model Order Reduction (MOR) & Hyper-Reduction

*Objective*: Accelerate high-dimensional linear and nonlinear FEM to real-time execution (< 1 ms per evaluation), eliminating the $\mathcal{O}(N)$ full-mesh assembly bottleneck via DEIM and ECSW hyper-reduction.

* [ ] **Offline Snapshot Matrix Management & SVD Basis Generation**:
  * Collect displacement snapshots matrix $\boldsymbol{S} = [\boldsymbol{u}_1, \boldsymbol{u}_2, \dots, \boldsymbol{u}_m] \in \mathbb{R}^{n \times m}$ from forward dynamic / static simulations.
  * Thin Singular Value Decomposition (SVD): $\boldsymbol{S} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^T$.
  * Truncation based on cumulative energy spectrum ($\ge 0.999$).
  * Extract orthonormal reduced basis matrix $\boldsymbol{\Phi} = \boldsymbol{U}[:, :r] \in \mathbb{R}^{n \times r}$ ($r \approx 5\text{--}20$ modes).
* [ ] **Linear & Corotational POD-Galerkin ROM**:
  * Exact Galerkin projection of structural matrices onto low-dimensional subspace ($r \ll n$):
    $$\boldsymbol{K}_r = \boldsymbol{\Phi}^T \boldsymbol{K} \boldsymbol{\Phi}, \quad \boldsymbol{M}_r = \boldsymbol{\Phi}^T \boldsymbol{M} \boldsymbol{\Phi}, \quad \boldsymbol{f}_r = \boldsymbol{\Phi}^T \boldsymbol{f}$$
  * Fast online dynamic solver: solves the dense $r \times r$ linear system in $< 0.5$ ms.
* [ ] **Nonlinear Hyper-Reduction (DEIM & ECSW)**:
  * **Discrete Empirical Interpolation Method (DEIM / Q-DEIM)**:
    * Collect nonlinear internal force snapshots: $\boldsymbol{S}_f = [\boldsymbol{f}_{\text{int}}(\boldsymbol{u}_1), \dots, \boldsymbol{f}_{\text{int}}(\boldsymbol{u}_m)]$.
    * Compute force basis $\boldsymbol{\Psi}_f \in \mathbb{R}^{n \times m}$ via SVD.
    * Greedy selection of $m$ "magic" interpolation DOFs $\boldsymbol{P} \in \mathbb{R}^{n \times m}$.
    * Reduced nonlinear force evaluation:
      $$\boldsymbol{f}_r \approx \boldsymbol{\Phi}^T \boldsymbol{\Psi}_f (\boldsymbol{P}^T \boldsymbol{\Psi}_f)^{-1} \boldsymbol{f}_{\text{int}}(\boldsymbol{u})[\boldsymbol{P}]$$
      evaluating nonlinear internal forces only at the $m$ sample indices rather than all $n$ DOFs.
  * **Energy-Conserving Sampling and Weighting (ECSW)**:
    * Formulates a Non-Negative Least Squares (NNLS) optimization problem to select a sparse sub-mesh of elements $\tilde{\mathcal{E}} \subset \mathcal{E}$ ($|\tilde{\mathcal{E}}| \ll |\mathcal{E}|$) with strictly positive weights $w_e > 0$.
    * Reduced virtual work integration:
      $$\boldsymbol{f}_r \approx \sum_{e \in \tilde{\mathcal{E}}} w_e \boldsymbol{\Phi}_e^T \boldsymbol{f}_{\text{int}, e}(\boldsymbol{\Phi}_e \boldsymbol{u}_r)$$
    * Strictly preserves symmetry and positive-definiteness of the reduced tangent stiffness $\boldsymbol{K}_r$, guaranteeing numerical stability in nonlinear dynamic simulations.
* [ ] **Real-Time Surface Reconstruction**:
  * Pre-extracted surface sub-basis $\boldsymbol{\Phi}_{\text{surf}} = \boldsymbol{\Phi}[\text{surface\_dofs}, :]$ for instantaneous 3D surface rendering:
    $$\boldsymbol{u}_{\text{surf}} = \boldsymbol{\Phi}_{\text{surf}} \boldsymbol{u}_r$$
* [ ] **Deliverable Demonstrations & Tests**:
  * `tests/test_mor_pod_galerkin.py`: Basis orthonormality ($\boldsymbol{\Phi}^T \boldsymbol{\Phi} = \boldsymbol{I}_r$), ROM vs. FOM accuracy verification ($< 2\%$ relative error).
  * `tests/test_mor_hyperreduction.py`: DEIM and ECSW hyper-reduction speedup and energy conservation verification.
  * `examples/08_mor/sim17_realtime_reduced_dynamic_beam.py`: Real-time online dynamic simulation running at $> 500$ Hz.

---

## Phase 5: Inverse FEM (iFEM) & Parameter Identification (★ Primary End Goal ★)

*Objective*: Infer full-field internal deformations, patient-specific material properties, and stress-free configurations from sparse physical measurements (closing the forward-to-inverse loop).

* [ ] **5.1: Discrete Sensor Observation Operator ($\boldsymbol{H}$)**:
  * Linear observation matrix mapping sparse discrete sensor measurement points (e.g., 5–20 surface markers, optical tracking, stereo camera, or ultrasound probe points) to FE degrees of freedom:
    $$\hat{\boldsymbol{u}}_k = \boldsymbol{H}_k \boldsymbol{u}$$
  * Spatial point-in-element barycentric locator for arbitrary sensor positions located on the organ surface.
* [ ] **5.2: Variational Shape Sensing iFEM (Full-Field 3D Reconstruction)**:
  * Reconstruct full 3D internal displacement and strain fields throughout the entire volume from sparse surface sensor measurements without knowing external boundary loads:
    $$\mathcal{J}(\boldsymbol{u}) = \sum_{k=1}^{N_{\text{sensor}}} w_k \|\boldsymbol{H}_k \boldsymbol{u} - \hat{\boldsymbol{u}}_k\|^2 + \alpha \cdot \boldsymbol{u}^T \boldsymbol{K} \boldsymbol{u}$$
  * Stationarity $\partial \mathcal{J} / \partial \boldsymbol{u} = \boldsymbol{0}$ yields a symmetric positive-definite linear system:
    $$(\boldsymbol{H}^T \boldsymbol{W} \boldsymbol{H} + \alpha \boldsymbol{K}) \boldsymbol{u} = \boldsymbol{H}^T \boldsymbol{W} \hat{\boldsymbol{u}}$$
  * Solvable in milliseconds via sparse Cholesky factorization, or in microseconds via POD reduced basis $\boldsymbol{\Phi}$.
* [ ] **5.3: Inverse Material Parameter Identification (Constitutive Calibration)**:
  * Optimization loop (`scipy.optimize.minimize` / L-BFGS-B / Gauss-Newton) to calibrate unknown material parameters (Young's modulus $E$, hyperelastic parameter $C_{10}$) by minimizing discrepancy between simulated indenter response and measured force-displacement curves ($F$ vs. $\delta$):
    $$\min_{\boldsymbol{\theta}} \sum_{j} \left( F_{\text{FEM}}(\delta_j; \boldsymbol{\theta}) - F_{\text{exp}}(\delta_j) \right)^2$$
  * Accelerated by forward ROM from Phase 4 for rapid convergence in seconds.
* [ ] **5.4: Stress-Free Reference Configuration Solver (Zero-Pressure Configuration)**:
  * Fixed-point inverse elastostatic solver to determine the undeformed reference geometry $\boldsymbol{X}_0$ from clinical scans of deformed / gravitationally sagged organs (e.g., intraoperative brain shift, pressurized aorta):
    $$\boldsymbol{X}^{k+1} = \boldsymbol{X}_{\text{scan}} - \boldsymbol{u}(\boldsymbol{X}^k)$$
* [ ] **Deliverable Demonstrations & Tests**:
  * `tests/test_ifem_shape_sensing.py`: Reconstructing full 3D beam deformation from 6 surface sensor points ($< 5\%$ error).
  * `tests/test_ifem_parameter_identification.py`: Calibrating Young's modulus from simulated indentation test.
  * `examples/09_inverse/sim18_sparse_sensor_shape_sensing.py`: Full 3D organ deformation reconstruction from surface optical markers.
  * `examples/09_inverse/sim19_tissue_indentation_parameter_identification.py`: Automated hyperelastic parameter fitting from indenter test.

---

## Phase 6+: Post-Loop Enhancements & Advanced Biomechanics (Post-iFEM Roadmap)

*Objective*: Advanced research enhancements, non-smooth contact theories, and multiphysics to be added after the end-to-end iFEM loop is fully working and verified.

* [ ] **Advanced Non-Smooth Contact Mechanics & $C^0$ Facet Chattering Mitigation**:
  * **The 3-Pillar Contact Regularization Architecture** (Resolving high-energy dynamic impact breakdowns, facet chattering, and line-search stalls in deformable-to-deformable mesh contact):
    * *Literature References*: P. Wriggers (2006) *Computational Contact Mechanics* (Ch. 6, 10, 11); T. A. Laursen (2002) *Computational Contact and Impact Mechanics* (Ch. 4, 6); FEBio Theory Manual (Sec. 7.1.2, 7.1.3, 8.1).
    * **Pillar 1: Temporal Regularization (Adaptive Time Substepping & Automatic Step Cutback)**: `AdaptiveTimeStepController` in `NonlinearDynamicAnalysis`. Automatically halving $\Delta t \leftarrow \gamma_{\text{cut}} \Delta t$ upon line search stall, restoring state $(u_n, v_n, a_n, \lambda_n)$, retrying with bounded displacement increments across element boundaries, and ramping back up upon convergence.
    * **Pillar 2: Spatial Regularization ($C^1$-Continuous Surface Normal Smoothing)**: Upgrading `DeformableContact` to interpolate area/angle-weighted vertex pseudonormals $\boldsymbol{n}(\boldsymbol{\xi}) = \frac{\sum \xi_i \bar{\boldsymbol{n}}_i}{\|\sum \xi_i \bar{\boldsymbol{n}}_i\|}$, eliminating the directional jump $\Delta \boldsymbol{n} = \boldsymbol{n}_B - \boldsymbol{n}_A$ as nodes cross element edges and restoring quadratic Newton descent.
    * **Pillar 3: Algorithmic Regularization (Contact-Aware Line Search & Dual Rebound Relaxation)**: Distinguishing constitutive inversion ($\det J \le 0$) from non-smooth facet edge crossing; relaxing retained dual multipliers ($\lambda \leftarrow 0$) when local rebound opens a positive gap ($g > 0$) to eliminate artificial holding forces across facet boundaries.
  * Segment-to-segment mortar contact for non-conforming mesh interfaces.
  * Exact Coulomb friction with non-smooth stick-slip active set strategy.
* [ ] **Advanced Instability & Equilibrium Path Solvers**:
  * **Arc-Length / Riks Method (`ArcLengthAnalysis`)**: Tracing post-buckling, snap-through, and snap-back equilibrium paths where $\det(\boldsymbol{K}_T) \le 0$ via augmented constraint equations.
  * **Dynamic Relaxation Equilibrium Solver (`DynamicRelaxationAnalysis`)**: Barnes / Underwood kinetic damping algorithm ($\boldsymbol{v} \leftarrow \mathbf{0}$ on kinetic energy peaks) with Gerschgorin diagonal mass scaling for highly unstable softening paths.
* [ ] **Biomechanical Specialization & Multiphysics**:
  * **Viscoelastic Constitutive Formulations**: Generalized Maxwell / Prony series relaxation spectrum ($G(t) = G_\infty + \sum_m G_m e^{-t/\tau_m}$) via internal element history variables.
  * **Advanced Hyperelasticity & Anisotropy**: Holzapfel-Gasser-Ogden (`HGO`) anisotropic fiber reinforcement; Ogden spectral principal stretch model with coalescing eigenvalue perturbation handling.
  * **Incompressibility & Volumetric Locking**: `Tet4Fbar` / B-bar (Mean Dilatation) and hybrid mixed $u-p$ formulations for nearly incompressible soft tissue ($\nu \to 0.5$).
  * **Poroelasticity (Biot Consolidation)**: Coupled $u-p$ mixed formulation with saddle-point solvers and Taylor-Hood `Tet10`-`Tet4` elements satisfying the LBB / inf-sup condition.
  * **Coupled Hybrid Formulations**: Unified hyper-visco-poroelastic constitutive laws for perfused parenchymal organs.
* [ ] **Structured Hexahedral Discretizations**:
  * Hexahedral volume meshing infrastructure (structured sweeping, polycube mapping).
  * Linear 8-node brick element (`Hex8`) with B-bar / Selective Reduced Integration (SRI).
  * Quadratic 20-node serendipity brick element (`Hex20`).
  * Quadrilateral shell element (`Shell4` / MITC4).
