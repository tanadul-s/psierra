# Project pSierra Development Plan & Roadmap

This document outlines the strategic milestones, component delivery roadmap, and active task tracking for Project `pSierra`.

---

## Strategic Vision: Fast-Track to Inverse FEM (iFEM)

The primary goal of Project `pSierra` is to **close the complete work loop from forward simulation to Inverse FEM (iFEM) as quickly as possible**:
$$\text{Forward Continuum Mechanics} \longrightarrow \text{Contact Mechanics} \longrightarrow \text{Model Order Reduction (MOR)} \longrightarrow \text{Inverse FEM (iFEM)}$$

### Guiding Execution Strategy:
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
- [x] **Modular Lego-Brick Element Architecture**:
  - Class naming convention: `[Element][Formulation]` (e.g., `Tet4Linear`, `Tet4Corotational`, `Tet4TotalLagrangian`, `Tet4UpdatedLagrangian`).
  - Base parent classes: `Tet4Base`, `Tet10Base`, `Shell3Base`, `Shell6Base`.
  - Zero aliases in `src/psierra/element/__init__.py`.
- [x] **Kinematic Formulations**:
  - [x] **Linear Small-Strain**: Infinitesimal strain $\boldsymbol{\varepsilon} = \operatorname{sym}(\nabla \boldsymbol{u})$ (`Tet4Linear`, `Tet10Linear`, `Shell3Linear`, `Shell6Linear`).
  - [x] **Corotational (CR)**: Polar decomposition $F = R U$ filtering rigid rotations for large displacement/rotation small strain problems (`Tet4Corotational`, `Shell3Corotational`).
  - [x] **Total Lagrangian (TL)**: Reference configuration $X$ kinematics with Green-Lagrange strain $E$ and 2nd Piola-Kirchhoff stress $S$ (`Tet4TotalLagrangian`, `Tet10TotalLagrangian`).
  - [x] **Updated Lagrangian (UL)**: Spatial current configuration $x$ kinematics with spatial gradients $h = H F^{-1}$, Cauchy stress $\sigma = \frac{1}{J} F S F^T$, spatial tangent moduli $c^{\text{spatial}}$, and spatial geometric stiffness $k_{\text{geo}}$ (`Tet4UpdatedLagrangian`, `Tet10UpdatedLagrangian`).
  - [x] Machine-precision equivalence between TL and UL verified to $< 10^{-13}$ mm.
- [x] **Hyperelastic Constitutive Models**:
  - [x] Compressible Neo-Hookean hyperelasticity with exact analytical tangent and 2nd PK stress (`NeoHookean`).
  - [x] **Yeoh Model (`Yeoh`)** (Soft Tissue Strain-Stiffening):
    - 3-parameter reduced polynomial formulation: $\Psi = \sum_{i=1}^3 C_{i0} (\bar{I}_1 - 3)^i + \frac{1}{2} K (\ln J)^2$.
    - Captures physiological exponential strain-stiffening of soft biological tissues (liver, brain, kidney) at large strains.
    - Pure closed-form invariant tensor derivatives: directly leverages existing `NeoHookean` $\partial \bar{I}_1 / \partial \boldsymbol{C}$ and $\partial^2 \bar{I}_1 / \partial \boldsymbol{C} \partial \boldsymbol{C}$ kernels.
  - [x] **Mooney-Rivlin Model (`MooneyRivlin`)**:
    - 2-parameter formulation: $\Psi = C_{10} (\bar{I}_1 - 3) + C_{01} (\bar{I}_2 - 3) + \frac{1}{2} K (\ln J)^2$.
    - Analytical derivatives for second invariant $\bar{I}_2$: $\partial I_2 / \partial \boldsymbol{C} = I_1 \boldsymbol{I} - \boldsymbol{C}$.
    - Standard benchmark for synthetic elastomers, silicones, and rubber phantoms in surgical training.
- [ ] **Non-linear Solver Algorithms & Analysis**:
  - [x] Incremental Newton-Raphson iteration with backtracking line search (`NonlinearStaticAnalysis`).
  - [x] Implicit non-linear dynamics with Newmark-$\beta$ integration and Rayleigh damping (`NonlinearDynamicAnalysis`).
  - [x] **Dynamic Steady-State Equilibrium Convergence**:
    - Automatic early stopping when static out-of-balance residual $\|\boldsymbol{f}_{\text{ext}} - \boldsymbol{f}_{\text{int}}\| / \|\boldsymbol{f}_{\text{ext}}\| \le \text{steady\_state\_tol}$ holds for `consecutive_hold_steps`.
    - Integrated across `LinearDynamicAnalysis` and `NonlinearDynamicAnalysis`.
    - Guaranteed export of final snapshot upon reaching steady state.
  - [x] **Unified Snapshot Stride Management**:
    - Snapshot saving period (`save_interval`) and PVD streaming across all static and dynamic solvers (`LinearStaticAnalysis`, `NonlinearStaticAnalysis`, `LinearDynamicAnalysis`, `NonlinearDynamicAnalysis`).
  - [x] **Generalized-$\alpha$ Dynamic Scheme (`LinearDynamicAnalysis` & `NonlinearDynamicAnalysis`)**:
    - Controllable numerical high-frequency damping parameterized by single physical knob: high-frequency spectral radius $\rho_\infty \in [0, 1]$.
    - Deprecated and removed legacy $\beta, \gamma$ parameters in favor of clean single-parameter control.
    - Preserves unconditional stability and strict 2nd-order accuracy $\mathcal{O}(\Delta t^2)$ while filtering out spurious mesh discretization chatter during fast dynamics and contact impact.
    - Full mapping: $\rho_\infty = 1.0$ (Undamped Newmark), $\rho_\infty = 0.8$ (Default Generalized-$\alpha$), $\rho_\infty = 0.5$ (HHT-$\alpha$), $\rho_\infty = 0.0$ (Asymptotic Annihilation).
    - Verified via `examples/05_simulation/sim16_dynamic_time_integration_comparison.py`.

---

## Milestone 2.5: Interactive / Online Dynamic Simulation Framework (Lean Prototype)

*Objective*: Establish a lean, real-time dynamic simulation pipeline and axial torsion / dynamic demonstration runner before introducing contact mechanics.

- [x] **Online Dynamic Simulation Runner (`InteractiveDynamicSimulation`)**:
  - Real-time dynamic loop operating with fixed time step $\Delta t$ decoupled from rendering frame rate.
  - High-performance sparse factorization reuse: pre-factorized linear dynamic step or cached Cholesky/LU factorizations for low computational overhead ($>60$ FPS, $>10,000$ Hz physics rate).
  - Real-time execution monitor tracking wall-clock simulation speed ratio (Sim Time / Real Time).
- [x] **Interactive Dynamic Load & Boundary Perturbations**:
  - Dynamic external load generators (harmonic excitation, moving point forces).
  - Interactive user disturbance interface (keyboard impulse triggers, mouse-pick surface node forces).
- [x] **Live 3D Visualization Pipeline**:
  - Real-time PyVista integration with non-blocking render loop.
  - In-place point coordinate mutation (`polydata.points = current_nodes`) to bypass expensive VTK object recreation.
- [x] **Deliverable Demonstrations & Capability Proofs**:
  - [x] `examples/05_simulation/sim17_axial_torsion_hyperelastic_beam.py`: Severe axial twisting benchmark of a 3D continuum cantilever beam (up to $90^\circ - 180^\circ$ rotation) to prove finite-strain kinematics, cross-sectional warping, Poynting axial shortening/lengthening, and hyperelastic stability under large torsional shear.
  - [x] `examples/05_simulation/sim18_tet10_nonlinear_hyperelastic_beam.py`: Finite-strain hyperelastic beam bending benchmark proving quadratic Tet10 shear-locking alleviation ($1.51\times$ compliance recovery) and TL/UL machine precision equivalence ($< 10^{-13}$ mm).
  - [x] `examples/06_interactive/sim01_online_dynamic_cantilever.py`: Real-time interactive cantilever beam under user-controlled dynamic disturbance ($12,056 \text{ Hz}$, $24.1\times$ real-time).
  - [x] `examples/06_interactive/sim02_online_dynamic_tet10_nonlinear.py`: Real-time interactive dynamic simulation with 10-node quadratic tetrahedra (Tet10) and nonlinear hyperelasticity (Modified Newton Generalized-$\alpha$).

---

## Phase 3: Contact Mechanics & Interactive Collision

*Objective*: Provide robust tool-tissue interaction, surface indentation, and collision constraints necessary for forward testing and inverse parameter estimation, supporting both Penalty and Augmented Lagrangian formulations.

- [ ] **Dual-Tier Proximity & Collision Detection**:
  - **Broad-Phase Acceleration**: Fast bounding volume queries via `scipy.spatial.cKDTree` / AABB bounding hierarchy to prune non-colliding pairs ($< 1$ ms).
  - **Narrow-Phase Closest-Point Projection (Primary Deformable Surface Contact)**:
    - Exact orthogonal projection of slave nodes onto candidate master triangles (face interior, edge, or vertex Voronoi regions).
    - Evaluates signed normal penetration gap: $g_n = (\boldsymbol{x}_{\text{slave}} - \boldsymbol{x}_{\text{target}}) \cdot \boldsymbol{n} < 0$.
    - Prevents false-misses inherent in pure normal ray casting on curved or folding tissue boundaries.
  - **Ray-Traced Collision Detection (Möller-Trumbore Algorithm)**:
    - High-speed ray-triangle intersection dedicated to straight rigid surgical tool shafts, needles, and virtual sensor lines-of-sight.
- [ ] **Contact Formulations**:
  - **Penalty Method**:
    - Normal contact force vector: $\boldsymbol{f}_c = k_p |g_n| \boldsymbol{n}$.
    - Simple and robust for initial contact setup and dynamic stabilization.
  - **Augmented Lagrangian Multiplier Method**:
    - Normal contact force: $f_n = \max(0, \lambda_n + k_p g_n)$.
    - Multiplier update at step convergence: $\lambda_n^{(k+1)} = \max(0, \lambda_n^{(k)} + k_p g_n^{(k+1)})$.
    - Decouples penetration error from penalty stiffness $k_p$, achieving high non-penetration precision without ill-conditioning the tangent stiffness matrix.
  - **Regularized Tangential Friction**:
    - Viscous regularized friction vector: $\boldsymbol{f}_t = -\mu |f_n| \tanh\left(\frac{\|\boldsymbol{v}_t\|}{v_0}\right) \boldsymbol{t}$.
- [ ] **Rigid Indenters & Surgical Tool Contact**:
  - Analytical rigid obstacles: spherical indenter, cylindrical tool, and flat punch pressing onto deformable organ meshes.
  - Streams force-depth curves ($F$ vs. $\delta$) directly into Phase 5 Inverse Material Calibration.
- [ ] **Deliverable Demonstrations & Tests**:
  - `tests/test_contact_penalty.py`: Non-penetration verification and energy consistency for Penalty method.
  - `tests/test_contact_augmented_lagrangian.py`: Multiplier convergence and penetration error decay ($|g_n| \to 0$).
  - `examples/07_contact/sim16_spherical_indentation_soft_tissue.py`: Spherical indenter pressing into a hyperelastic soft organ block.

---

## Phase 4: Model Order Reduction (MOR) & Hyper-Reduction

*Objective*: Accelerate high-dimensional linear and nonlinear FEM to real-time execution (< 1 ms per evaluation), eliminating the $\mathcal{O}(N)$ full-mesh assembly bottleneck via DEIM and ECSW hyper-reduction.

- [ ] **Offline Snapshot Matrix Management & SVD Basis Generation**:
  - Collect displacement snapshots matrix $\boldsymbol{S} = [\boldsymbol{u}_1, \boldsymbol{u}_2, \dots, \boldsymbol{u}_m] \in \mathbb{R}^{n \times m}$ from forward dynamic / static simulations.
  - Thin Singular Value Decomposition (SVD): $\boldsymbol{S} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^T$.
  - Truncation based on cumulative energy spectrum ($\ge 0.999$).
  - Extract orthonormal reduced basis matrix $\boldsymbol{\Phi} = \boldsymbol{U}[:, :r] \in \mathbb{R}^{n \times r}$ ($r \approx 5\text{--}20$ modes).
- [ ] **Linear & Corotational POD-Galerkin ROM**:
  - Exact Galerkin projection of structural matrices onto low-dimensional subspace ($r \ll n$):
    $$\boldsymbol{K}_r = \boldsymbol{\Phi}^T \boldsymbol{K} \boldsymbol{\Phi}, \quad \boldsymbol{M}_r = \boldsymbol{\Phi}^T \boldsymbol{M} \boldsymbol{\Phi}, \quad \boldsymbol{f}_r = \boldsymbol{\Phi}^T \boldsymbol{f}$$
  - Fast online dynamic solver: solves the dense $r \times r$ linear system in $< 0.5$ ms.
- [ ] **Nonlinear Hyper-Reduction (DEIM & ECSW)**:
  - **Discrete Empirical Interpolation Method (DEIM / Q-DEIM)**:
    - Collect nonlinear internal force snapshots: $\boldsymbol{S}_f = [\boldsymbol{f}_{\text{int}}(\boldsymbol{u}_1), \dots, \boldsymbol{f}_{\text{int}}(\boldsymbol{u}_m)]$.
    - Compute force basis $\boldsymbol{\Psi}_f \in \mathbb{R}^{n \times m}$ via SVD.
    - Greedy selection of $m$ "magic" interpolation DOFs $\boldsymbol{P} \in \mathbb{R}^{n \times m}$.
    - Reduced nonlinear force evaluation:
      $$\boldsymbol{f}_r \approx \boldsymbol{\Phi}^T \boldsymbol{\Psi}_f (\boldsymbol{P}^T \boldsymbol{\Psi}_f)^{-1} \boldsymbol{f}_{\text{int}}(\boldsymbol{u})[\boldsymbol{P}]$$
      evaluating nonlinear internal forces only at the $m$ sample indices rather than all $n$ DOFs.
  - **Energy-Conserving Sampling and Weighting (ECSW)**:
    - Formulates a Non-Negative Least Squares (NNLS) optimization problem to select a sparse sub-mesh of elements $\tilde{\mathcal{E}} \subset \mathcal{E}$ ($|\tilde{\mathcal{E}}| \ll |\mathcal{E}|$) with strictly positive weights $w_e > 0$.
    - Reduced virtual work integration:
      $$\boldsymbol{f}_r \approx \sum_{e \in \tilde{\mathcal{E}}} w_e \boldsymbol{\Phi}_e^T \boldsymbol{f}_{\text{int}, e}(\boldsymbol{\Phi}_e \boldsymbol{u}_r)$$
    - Strictly preserves symmetry and positive-definiteness of the reduced tangent stiffness $\boldsymbol{K}_r$, guaranteeing numerical stability in nonlinear dynamic simulations.
- [ ] **Real-Time Surface Reconstruction**:
  - Pre-extracted surface sub-basis $\boldsymbol{\Phi}_{\text{surf}} = \boldsymbol{\Phi}[\text{surface\_dofs}, :]$ for instantaneous 3D surface rendering:
    $$\boldsymbol{u}_{\text{surf}} = \boldsymbol{\Phi}_{\text{surf}} \boldsymbol{u}_r$$
- [ ] **Deliverable Demonstrations & Tests**:
  - `tests/test_mor_pod_galerkin.py`: Basis orthonormality ($\boldsymbol{\Phi}^T \boldsymbol{\Phi} = \boldsymbol{I}_r$), ROM vs. FOM accuracy verification ($< 2\%$ relative error).
  - `tests/test_mor_hyperreduction.py`: DEIM and ECSW hyper-reduction speedup and energy conservation verification.
  - `examples/08_mor/sim17_realtime_reduced_dynamic_beam.py`: Real-time online dynamic simulation running at $> 500$ Hz.

---

## Phase 5: Inverse FEM (iFEM) & Parameter Identification (★ Primary End Goal ★)

*Objective*: Infer full-field internal deformations, patient-specific material properties, and stress-free configurations from sparse physical measurements (closing the forward-to-inverse loop).

- [ ] **5.1: Discrete Sensor Observation Operator ($\boldsymbol{H}$)**:
  - Linear observation matrix mapping sparse discrete sensor measurement points (e.g., 5–20 surface markers, optical tracking, stereo camera, or ultrasound probe points) to FE degrees of freedom:
    $$\hat{\boldsymbol{u}}_k = \boldsymbol{H}_k \boldsymbol{u}$$
  - Spatial point-in-element barycentric locator for arbitrary sensor positions located on the organ surface.
- [ ] **5.2: Variational Shape Sensing iFEM (Full-Field 3D Reconstruction)**:
  - Reconstruct full 3D internal displacement and strain fields throughout the entire volume from sparse surface sensor measurements without knowing external boundary loads:
    $$\mathcal{J}(\boldsymbol{u}) = \sum_{k=1}^{N_{\text{sensor}}} w_k \|\boldsymbol{H}_k \boldsymbol{u} - \hat{\boldsymbol{u}}_k\|^2 + \alpha \cdot \boldsymbol{u}^T \boldsymbol{K} \boldsymbol{u}$$
  - Stationarity $\partial \mathcal{J} / \partial \boldsymbol{u} = \boldsymbol{0}$ yields a symmetric positive-definite linear system:
    $$(\boldsymbol{H}^T \boldsymbol{W} \boldsymbol{H} + \alpha \boldsymbol{K}) \boldsymbol{u} = \boldsymbol{H}^T \boldsymbol{W} \hat{\boldsymbol{u}}$$
  - Solvable in milliseconds via sparse Cholesky factorization, or in microseconds via POD reduced basis $\boldsymbol{\Phi}$.
- [ ] **5.3: Inverse Material Parameter Identification (Constitutive Calibration)**:
  - Optimization loop (`scipy.optimize.minimize` / L-BFGS-B / Gauss-Newton) to calibrate unknown material parameters (Young's modulus $E$, hyperelastic parameter $C_{10}$) by minimizing discrepancy between simulated indenter response and measured force-displacement curves ($F$ vs. $\delta$):
    $$\min_{\boldsymbol{\theta}} \sum_{j} \left( F_{\text{FEM}}(\delta_j; \boldsymbol{\theta}) - F_{\text{exp}}(\delta_j) \right)^2$$
  - Accelerated by forward ROM from Phase 4 for rapid convergence in seconds.
- [ ] **5.4: Stress-Free Reference Configuration Solver (Zero-Pressure Configuration)**:
  - Fixed-point inverse elastostatic solver to determine the undeformed reference geometry $\boldsymbol{X}_0$ from clinical scans of deformed / gravitationally sagged organs (e.g., intraoperative brain shift, pressurized aorta):
    $$\boldsymbol{X}^{k+1} = \boldsymbol{X}_{\text{scan}} - \boldsymbol{u}(\boldsymbol{X}^k)$$
- [ ] **Deliverable Demonstrations & Tests**:
  - `tests/test_ifem_shape_sensing.py`: Reconstructing full 3D beam deformation from 6 surface sensor points ($< 5\%$ error).
  - `tests/test_ifem_parameter_identification.py`: Calibrating Young's modulus from simulated indentation test.
  - `examples/09_inverse/sim18_sparse_sensor_shape_sensing.py`: Full 3D organ deformation reconstruction from surface optical markers.
  - `examples/09_inverse/sim19_tissue_indentation_parameter_identification.py`: Automated hyperelastic parameter fitting from indenter test.

---

## Phase 6+: Post-Loop Enhancements & Advanced Biomechanics (Post-iFEM Roadmap)

*Objective*: Advanced research enhancements, non-smooth contact theories, and multiphysics to be added after the end-to-end iFEM loop is fully working and verified.

- [ ] **Advanced Non-Smooth Contact Mechanics**:
  - Segment-to-segment mortar contact for non-conforming mesh interfaces.
  - Exact Coulomb friction with non-smooth stick-slip active set strategy.
- [ ] **Advanced Instability & Equilibrium Path Solvers**:
  - **Arc-Length / Riks Method (`ArcLengthAnalysis`)**: Tracing post-buckling, snap-through, and snap-back equilibrium paths where $\det(\boldsymbol{K}_T) \le 0$ via augmented constraint equations.
  - **Dynamic Relaxation Equilibrium Solver (`DynamicRelaxationAnalysis`)**: Barnes / Underwood kinetic damping algorithm ($\boldsymbol{v} \leftarrow \mathbf{0}$ on kinetic energy peaks) with Gerschgorin diagonal mass scaling for highly unstable softening paths.
- [ ] **Biomechanical Specialization & Multiphysics**:
  - **Viscoelastic Constitutive Formulations**: Generalized Maxwell / Prony series relaxation spectrum ($G(t) = G_\infty + \sum_m G_m e^{-t/\tau_m}$) via internal element history variables.
  - **Advanced Hyperelasticity & Anisotropy**: Holzapfel-Gasser-Ogden (`HGO`) anisotropic fiber reinforcement; Ogden spectral principal stretch model with coalescing eigenvalue perturbation handling.
  - **Incompressibility & Volumetric Locking**: `Tet4Fbar` / B-bar (Mean Dilatation) and hybrid mixed $u-p$ formulations for nearly incompressible soft tissue ($\nu \to 0.5$).
  - **Poroelasticity (Biot Consolidation)**: Coupled $u-p$ mixed formulation with saddle-point solvers and Taylor-Hood `Tet10`-`Tet4` elements satisfying the LBB / inf-sup condition.
  - **Coupled Hybrid Formulations**: Unified hyper-visco-poroelastic constitutive laws for perfused parenchymal organs.
- [ ] **Structured Hexahedral Discretizations**:
  - Hexahedral volume meshing infrastructure (structured sweeping, polycube mapping).
  - Linear 8-node brick element (`Hex8`) with B-bar / Selective Reduced Integration (SRI).
  - Quadratic 20-node serendipity brick element (`Hex20`).
  - Quadrilateral shell element (`Shell4` / MITC4).

