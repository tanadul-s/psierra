# Project pSierra Development Plan & Roadmap (Loop 2 Archive: Phases 1–4)

> [!IMPORTANT]
> **Loop 2 (Phases 1 through 4) is finalized and archived.**
> This document preserves the historical engineering milestones, mathematical formulations, and delivery logs of Loop 2 (Biomechanical Specialization & Digital Twins).
>
> **Transition to Loop 3:**
> - Remaining uncompleted tasks and the dedicated focus on **Robust Contact Mechanics Stabilization & Mathematical Proof** are actively tracked in:
> 📂 **[`PLAN.md`](PLAN.md)**
> - For Loop 1 historical records, see:
> 📂 **[`PLAN_ARCHIVE_LOOP1.md`](PLAN_ARCHIVE_LOOP1.md)**

---

## 1. Strategic Vision: Closed Loop to Post-Loop Enhancements

Project `pSierra` has closed the primary continuum-to-inverse loop:
$$\text{Forward Continuum Mechanics} \longrightarrow \text{Contact Mechanics} \longrightarrow \text{Model Order Reduction (MOR)} \longrightarrow \text{Inverse FEM (iFEM)}$$

```
========================= LOOP 1: PRIMARY WORK LOOP (COMPLETED) =========================
[ Phase 1: Linear FEM Core ]
  - Tet4 linear continua, Hookean elasticity, sparse assembly, SuperLU, Rayleigh damping.
       │
       ▼
[ Phase 2: Non-Linear Continua & Hyperelasticity ]
  - Tet10, Shell3 (DKT), Shell6, CR, TL, UL, Neo-Hookean, Yeoh, Mooney-Rivlin, Generalized-α.
       │
       ▼
[ Phase 3: Contact Mechanics (Penalty & Augmented Lagrangian) ]
  - Rigid floor, Deformable-to-deformable mesh contact, Stanford Bunny self-contact.
       │
       ▼
[ Phase 4: Model Order Reduction (POD-Galerkin ROM & ECSW Hyper-Reduction) ]
  - Real-time dynamics (>20,000 Hz), surface sub-basis, sparse RID cubature (>900 Hz).
       │
       ▼
[ Phase 5: Inverse FEM (iFEM) & Parameter Identification ]
  - Variational shape sensing, closed-form LLS & TRF identification, Sellier Aitken inversion.
==========================================================================================
       │
       ▼
==================== LOOP 2: BIOMECHANICAL SPECIALIZATION & DIGITAL TWINS ===============
[ Phase 1: Biomechanical Continua, HDF5 Infrastructure & Dynamic Relaxation ]
  - Tet4Fbar (locking-free), Viscoelasticity (Prony series), HDF5 container & resume,
    Dynamic Relaxation solver (kinetic damping), Hydrostatic CSF load (Chen et al.).
       │
       ▼
[ Phase 2: Advanced Contact Regularization & Non-Smooth Mechanics ]
  - 3-Pillar Regularization (Adaptive cutback, C¹ vertex pseudonormals, contact-aware line search),
    Segment-to-segment Mortar contact, Exact Coulomb stick-slip friction.
       │
       ▼
[ Phase 3: Advanced Non-Linear Hyper-Reduction (DEIM & Contact-Aware ECSW) ]
  - Force snapshot streaming in HDF5, DEIM / Q-DEIM with history variables,
    Contact-Aware ECSW (Method 2: joint bulk + contact virtual work), low-latency CPU ROM.
       │
       ▼
[ Phase 4: Clinical Digital Twins & Real-Time Intraoperative Navigation ]
  - Quasi-static surgical indentation with viscoelastic relaxation,
    Online hyper-reduced parameter identification, Multi-modal sensor fusion (iUS + markers),
    Anisotropic Hyperelasticity (Holzapfel-Gasser-Ogden - HGO model for membranes).
==========================================================================================
       │ (Future Exploration)
       ▼
==================== LOOP 3: COUPLED MULTIPHYSICS & HIGH-PERFORMANCE EXTENSIONS ========
  - Taylor-Hood mixed u-p elements (Tet10-Tet4) for coupled Biot Poroelasticity (hydrocephalus).
  - GPU (CuPy) acceleration hooks for massive full-order snapshot generation & offline assembly.
  - Structured Hexahedral volume meshing & elements (Hex8 with B-bar, Hex20, Shell4 / MITC4).
==========================================================================================
```

---

## 2. Loop 1 Executive Summary (Phases 1–5, Completed)

All 5 core phases of Loop 1 have been implemented, tested, and documented with 100% test pass rate across 505 automated tests:

| Phase | Core Modules & Classes | Key Capabilities Delivered | Verification Tests | Documentation |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Linear FEM** | `psierra.element.tet4_linear`<br/>`psierra.material.hookean`<br/>`psierra.analysis.linear_static` | Simplex kinematics, sparse COO/CSC assembly, static condensation, Rayleigh dynamics. | 8 test modules<br/>(45+ tests) | [`docs/01`](../docs/01_continuum_mechanics_and_materials.md)<br/>[`docs/02`](../docs/02_solid_continua_linear_tet4.md) |
| **Phase 2: Non-Linear Continua** | `psierra.element.tet10`<br/>`psierra.element.shell3`<br/>`psierra.element.shell6`<br/>`psierra.material.neohookean` | Quadratic serendipity basis, DKT/Reissner shells, CR/TL/UL kinematics, Yeoh soft tissue stiffening. | 14 test modules<br/>(110+ tests) | [`docs/03`](../docs/03_solid_continua_quadratic_tet10_and_quadrature.md)<br/>[`docs/04`](../docs/04_structural_shells_thin_and_thick.md)<br/>[`docs/06`](../docs/06_solvers_and_analysis_pipelines.md) |
| **Phase 3: Contact Mechanics** | `psierra.collision.shapes`<br/>`psierra.boundary.contact`<br/>`psierra.multibody.system` | Embree ray-mesh BVH, action-reaction deformable contact, Uzawa Augmented Lagrangian, self-contact. | 5 test modules<br/>(85+ tests) | [`docs/05`](../docs/05_system_assembly_boundaries_and_domain_coupling.md)<br/>[`docs/07`](../docs/07_simulation_tutorials_and_verification.md) |
| **Phase 4: MOR & ECSW** | `psierra.reduction`<br/>`psierra.hyperreduction` | POD-Galerkin ROM (>20,000 Hz), surface sub-basis, ECSW cubature RID sub-mesh, NNLS optimizer. | 4 test modules<br/>(40+ tests) | [`docs/08`](../docs/08_model_order_reduction_pod_galerkin.md)<br/>[`docs/09`](../docs/09_energy_conserving_hyper_reduction_ecsw.md) |
| **Phase 5: Inverse FEM** | `psierra.inverse.observation`<br/>`psierra.inverse.shape_sensing`<br/>`psierra.inverse.parameter_identification`<br/>`psierra.inverse.stress_free` | Variational shape sensing (>50,000 Hz), closed-form LLS & TRF identification, Sellier Aitken inversion. | 5 test modules<br/>(74 tests) | [`docs/10`](../docs/10_inverse_fem_and_parameter_identification.md) |

> 📖 *For complete historical checklists, see [`.agents/PLAN_ARCHIVE_LOOP1.md`](PLAN_ARCHIVE_LOOP1.md).*

---

## 3. Loop 2 Active Roadmap: Phase 1 to Phase 4

This roadmap is laser-focused on biological soft-tissue mechanics, robust contact, and real-time clinical digital twins.

### Phase 1: Biomechanical Continua, HDF5 Infrastructure & Dynamic Relaxation
*Objective*: Establish locking-free soft-tissue kinematics, time-dependent viscoelastic relaxation, chunked single-file HDF5 storage with simulation resume capability, and matrix-free static equilibrium solving.
* [x] **Volumetric Locking Alleviation (`Tet4Fbar` / B-bar Mean Dilatation)**:
  * Implement `Tet4Fbar` with Flory volumetric/isochoric split: $\bar{\boldsymbol{F}} = (\bar{J}/J)^{1/3} \boldsymbol{F}$.
  * Nodal-averaged volumetric dilatation eliminating locking in nearly incompressible soft tissue ($\nu \to 0.5$).
  * Formulated within the Total Lagrangian framework; consistent material and geometric tangent stiffness matrices $\boldsymbol{K}_T$.
* [x] **Viscoelastic Constitutive Laws (`psierra.material.viscoelastic`)**:
  * Generalized Maxwell / Prony series relaxation spectrum: $G(t) = G_\infty + \sum_{m=1}^{N_p} G_m e^{-t / \tau_m}$, $\eta(\Delta t) = g_\infty + \sum g_m \beta_m(\Delta t)$.
  * Exact recursive exponential time integration (Simo 1987, Holzapfel 2000) with history tracking ($\boldsymbol{Q}_{m, n+1}$, $\boldsymbol{S}^0_n$).
  * Algorithmic tangent scaling: $\mathbb{C}_{n+1} = \eta(\Delta t) \mathbb{C}^0_{n+1}$.
  * Full integration with domain lifecycle (`SolidDomain.set_time_step`, `commit_state`, `revert_state`) and element classes (`Tet4Base`, `Tet4TotalLagrangian`, `Tet4Fbar`).
  * Unit tested across 13 comprehensive test scenarios (100% pass) and validated on dynamic beam and patient-specific neurosurgical brain shift benchmarks.
* [x] **Unified HDF5 Simulation Storage & Checkpointing (`h5py`, `SimulationWriter`)**:
  * Unified single class `SimulationWriter` toggling between modern `format="xdmf"` (clean 2-file binary storage with single-mesh topology and standard XDMF 3.0 XML for ParaView) and legacy `format="pvd"` (multi-file VTU snapshots).
  * Binary simulation state checkpointing: `SimulationCheckpoint(step, time, dt, u, v, a, history_variables, metadata)`.
  * Exact simulation resume verified with $0.00\times 10^0$ error against continuous baselines.
  * Successfully demonstrated across all 4 continuum simulation classes (Linear Static, Linear Dynamic, Nonlinear Static, Nonlinear Dynamic) in `1104_unified_beam_storage_and_resume.py`.
* [x] **Dynamic Relaxation Equilibrium Solver (`DynamicRelaxationAnalysis`)**:
  * Matrix-free explicit static solver using fictitious Gerschgorin-scaled diagonal mass $\boldsymbol{M}_{\text{fict}}$.
  * Barnes/Underwood kinetic damping ($\boldsymbol{v} \leftarrow \mathbf{0}$ when $dE_k/dt < 0$) and viscous mass-proportional damping.
  * Provides unconditional stability against tangent stiffness singularities ($\det(\boldsymbol{K}_T) \le 0$) during severe non-smooth contact and tissue collapse.
  * Unified single-class architecture seamlessly handling both linear elastic and nonlinear hyperelastic continua without matrix factorizations.
* [x] **CSF Hydrostatic Pressure & Buoyancy Load (`HydrostaticPressureLoad`, `BuoyancyLoad`)**:
  * *References*: Chen et al. (2017, 2019, 2022) IEEE SII & TMRB.
  * Spatially-varying boundary traction on submerged surface facets: $\boldsymbol{F}_{\text{hydro}} = -\rho_{\text{CSF}} g \max(0, h_{\text{fluid}} - z) A_k \boldsymbol{n}_k$.
  * Exact 2nd-order surface quadrature $\boldsymbol{F}_i^{\text{facet}} = -\boldsymbol{n}_k A_k (p_i + \sum p_j)/12$ identically recovering Archimedes' buoyant force to $< 10^{-12}$ relative error.
  * Volumetric body force counterpart `BuoyancyLoad` for full/partial submergence.
  * Configurable patient head orientation ($\hat{\boldsymbol{u}}_{\text{up}} = -\boldsymbol{g}/\|\boldsymbol{g}\|$) and dynamic CSF drainage fluid level $h_{\text{fluid}}$.
  * Models Archimedean neutral buoyancy (96.2% gravitational relief on brain parenchyma) and craniotomy CSF leak sag progression.
  * Complete 4-analysis suites implemented and verified across both cantilever beam (`1106_hydrostatic_buoyancy_beam.py`) and patient MRI brain anatomy (`1107_hydrostatic_buoyancy_brain.py`).
* [x] **Neurosurgical Burr Hole Brain Shift (`1108_burr_hole_brain_shift.py`)**:
  * Implemented cortical bulging (ICP / subcortical tumor edema herniation) vs. gravitational sagging (cerebral collapse / subdural sink from CSF loss) through a surgical burr hole (Kocher's Point, $R = 8.0\text{ mm}$, $\varnothing = 16\text{ mm}$).
  * Rigid calvarium & intact dural boundary envelope with circular aperture boundary release.
  * Solved with both Linear Static Analysis and Matrix-Free Dynamic Relaxation with kinetic damping ($< 0.5\%$ agreement).
  * Automatically exports 3D cranial bone table collar (`burr_hole_bone_collar.vtp`), circular perimeter ring (`burr_hole_perimeter_ring.vtp`), and cortical meninges (`cranial_meninges.vtp`) for direct ParaView spatial overlay.

---

### Phase 2: Advanced Contact Regularization & Non-Smooth Mechanics
*Objective*: Resolve high-energy dynamic impact breakdowns, facet edge chattering, and line-search stalls on locking-free continua.
* *References*: P. Wriggers (2006) *Computational Contact Mechanics*; T.A. Laursen (2002) *Computational Contact and Impact Mechanics*; FEBio Theory Manual.
* [x] **Pillar 1: Temporal Regularization (Adaptive Time Substepping & Automatic Cutback)**:
  * Implemented `AdaptiveTimeStepController` in `NonlinearDynamicAnalysis` ([`psierra/analysis/adaptive_time_step.py`](../src/psierra/analysis/adaptive_time_step.py)).
  * Automatic step cutback: $\Delta t \leftarrow \max(\Delta t_{\min}, \gamma_{\text{cut}} \Delta t)$ upon line search stall, Newton divergence, or contact non-convergence, restoring state $(\boldsymbol{u}_n, \boldsymbol{v}_n, \boldsymbol{a}_n, \boldsymbol{\lambda}_n)$ and internal viscoelastic history ($\boldsymbol{Q}_{m, n}$).
  * Dynamic ramp-up upon consecutive quadratic Newton-Raphson convergences ($\Delta t \leftarrow \min(\Delta t_{\max}, \gamma_{\text{grow}} \Delta t)$).
  * 10 automated verification tests in [`tests/test_analysis_adaptive_time_step.py`](../tests/test_analysis_adaptive_time_step.py) validating cutback recovery, dynamic growth, viscoelastic state reversion, contact multiplier restoration, and graceful exhaustion.
* [x] **Pillar 2: Spatial Regularization ($C^1$-Continuous Surface Normal Smoothing)**:
  * Upgraded `DeformableContact`, `SelfContact`, and `TriangleContactGeometry` with area/angle-weighted vertex pseudonormal interpolation:
    $$\boldsymbol{n}(\boldsymbol{\xi}) = \frac{\sum \xi_i \bar{\boldsymbol{n}}_i}{\|\sum \xi_i \bar{\boldsymbol{n}}_i\|}$$
  * Completely eliminates directional jumps $\Delta \boldsymbol{n} = \boldsymbol{n}_B - \boldsymbol{n}_A$ as nodes cross facet edges, preserving $C^1$ continuity and restoring smooth Newton-Raphson descent.
  * Preserves exact action-reaction balance ($\sum \boldsymbol{f}_c = \boldsymbol{0}$ to $< 10^{-12} \text{ N}$), exact tangent stiffness symmetry ($\boldsymbol{K}_c = \boldsymbol{K}_c^T$ to $< 10^{-12}$), and exact rigid translation invariance ($\boldsymbol{K}_c \boldsymbol{u}_{\text{trans}} \approx \boldsymbol{0}$).
  * 11 comprehensive automated tests in [`tests/test_deformable_contact_smoothing.py`](../tests/test_deformable_contact_smoothing.py) verifying jump elimination, action-reaction, tangent symmetry, translation invariance, weight choices (`angle`, `area`), and sliding continuity across facet edges.
  * 100% backward compatibility preserved via `smoothed_normals: bool = False` (all 124 existing contact tests passing untouched).
* [x] **Pillar 3: Algorithmic Regularization (Contact-Aware Line Search & Dual Relaxation)**:
  * Implemented [`ContactAwareLineSearch`](../src/psierra/analysis/line_search.py) differentiating physical continuum element inversion ($\det \boldsymbol{F} \le 0$, negative Jacobian, range limit violations) from non-smooth contact active-set switching and facet-edge crossings.
  * Inversion guard strictly rejects trials and mandates step backtracking ($\alpha \leftarrow 0.5 \alpha$), while non-smooth contact active-set transitions allow bounded full Newton step ($\alpha = 1.0$, $\|\boldsymbol{R}\| < 1.5 \|\boldsymbol{R}_0\|$) and safe contact fallback ($\|\boldsymbol{R}_{\text{best}}\| < 5.0 \|\boldsymbol{R}_0\|$).
  * Implemented Dual Rebound Relaxation in `ContactMultiplierState` and `ContactAugmentation` (`relax_rebound(gap_tol=0.0)`), dynamically zeroing multipliers ($\lambda \leftarrow 0$) when local rebound opens a positive gap ($g > g_{\text{tol}}$), eliminating artificial tensile "sticky contact" adhesion ($p = 0.0\text{ N}$).
  * Seamlessly integrated into `NonlinearDynamicAnalysis` and `NonlinearStaticAnalysis` with full backward compatibility (`rebound_relaxation=False` by default).
  * 13 automated tests in [`tests/test_analysis_line_search_and_relaxation.py`](../tests/test_analysis_line_search_and_relaxation.py) validating error classification, Armijo descent, inversion handling, fallback, rebound zeroing, and dynamic/static solver integration.
* [x] **Segment-to-Segment Mortar Contact (`MortarContact`, `MortarInterfaceIntegrator`)**:
  * Implemented Segment-to-Segment Mortar Contact for non-conforming, mismatched 3D mesh interfaces with exact action-reaction linear momentum balance ($\sum \boldsymbol{f}_s + \sum \boldsymbol{f}_m = \boldsymbol{0}$ to $< 10^{-12}\text{ N}$).
  * Wohlmuth biorthogonal dual shape functions $\Phi_i(\boldsymbol{\xi}) = 4\xi_i - 1$ satisfying $\int_\tau \Phi_i N_j d\Gamma = \frac{A_\tau}{3} \delta_{ij}$, yielding a strictly diagonal slave mortar mass matrix $\boldsymbol{D}$ and decoupled contact multipliers.
  * Sutherland-Hodgman 2D convex polygon clipping on local slave facet tangent planes, polygon triangulation, and symmetric 3-point Dunavant numerical quadrature over clipped sub-triangles.
  * Conservative tangent stiffness symmetry ($\boldsymbol{K}_c = \boldsymbol{K}_c^T$) with rigid body translation invariance ($\boldsymbol{K}_c \boldsymbol{u}_{\text{trans}} = \boldsymbol{0}$) and full two-way off-diagonal coupling blocks ($\boldsymbol{K}_{ss}, \boldsymbol{K}_{sm}, \boldsymbol{K}_{ms}, \boldsymbol{K}_{mm}$).
  * Contact patch test exact verification: uniform normal pressure transmission ($p_0 = \text{const}$) across arbitrarily non-conforming meshes without stress concentrations.
  * Dual rebound relaxation (`relax_rebound`) dynamically releasing separated interface nodes and eliminating tensile contact stickiness.
  * Full integration in `MultiBodySystem` (`add_mortar_contact()` and `add_contact_pair(..., method='mortar')`), compatible with penalty and Augmented Lagrangian formulations.
  * 9 automated mathematical verification tests in [`tests/test_mortar_contact.py`](../tests/test_mortar_contact.py) covering biorthogonality, clipping, partition of unity, momentum conservation, tangent symmetry, translation invariance, patch test, registration, and end-to-end static/dynamic finite deformation solves.
  * Delivered new executable simulations:
    * `examples/08_contact/0806_mortar_dynamic_rigid_floor.py` (deformable beam impact on rigid floor with mortar contact).
    * `examples/08_contact/0807_mortar_dynamic_deformable_floor.py` (two-way deformable beam impact on flexible floor with mortar contact).
* [x] **Exact Coulomb Stick-Slip Friction**:
  * 3D active-set Coulomb friction return mapping based on elastoplastic analogy with yield condition $\Phi^{\text{trial}} = \|\boldsymbol{f}_T^{\text{trial}}\| - \mu p_N \le 0$.
  * Elastic stick regime ($\Phi \le 0$) with zero relative slip rate ($\Delta \boldsymbol{g}_T^{\text{slip}} = \mathbf{0}$) and dissipative slip regime ($\Phi > 0$) with collinear sliding friction $\boldsymbol{f}_T = \mu p_N \boldsymbol{s}$.
  * Consistent algorithmic elastoplastic tangent operator $\boldsymbol{K}_{TT} = \frac{\mu p_N k_T}{\|\boldsymbol{f}_T^{\text{trial}}\|} (\boldsymbol{P}_T - \boldsymbol{s} \otimes \boldsymbol{s})$ exhibiting zero stiffness along the sliding direction ($\boldsymbol{K}_{TT} \boldsymbol{s} = \mathbf{0}$) and unsymmetric cross-coupling $\boldsymbol{K}_{TN} = \mu k_N (\boldsymbol{s} \otimes \boldsymbol{n})$.
  * Universal integration across all contact boundaries (`RigidContact`, `DeformableContact`, `SelfContact`, `MortarContact`) and `MultiBodySystem`.
  * Strict action-reaction momentum conservation: $\sum \boldsymbol{f}_{T, s} + \sum \boldsymbol{f}_{T, m} = \mathbf{0}$ to machine precision ($< 10^{-11}\text{ N}$) on both 4-node node-to-segment stencils and dual segment-to-segment mortar interfaces.
  * Transactional history anchor tracking (`commit_step()`) capturing hysteretic direction reversals and preserving quadratic Newton-Raphson convergence during equilibrium solves.
  * Augmented Lagrangian Uzawa multiplier projection updates: $\boldsymbol{\lambda}_T \leftarrow \operatorname{proj}_{\mu p_N}(\boldsymbol{f}_T)$.
  * 13 automated mathematical verification tests in [`tests/test_friction_coulomb.py`](../tests/test_friction_coulomb.py) covering return mapping, consistency, plastic flow condition, finite-difference tangent verification, hysteresis reversal loops, momentum balance, AL solves, dynamic frictional dissipation, and multi-body contact.
  * Executable simulation example: `examples/08_contact/0808_frictional_stick_slip.py` with unified XDMF and PVD export.

---

### Phase 3: Advanced Non-Linear Hyper-Reduction (DEIM & Contact-Aware ECSW)
*Objective*: Evaluate non-linear internal forces on low-dimensional sample points, extend hyper-reduction to deformable contact, and preserve sub-millisecond CPU ROM latency.
* [x] **Discrete Empirical Interpolation Method (DEIM / Q-DEIM)**:
  * Out-of-core force snapshot matrix assembly streamed directly into HDF5 datasets: $\boldsymbol{S}_f = [\boldsymbol{f}_{\text{int}}(\boldsymbol{u}_1), \dots, \boldsymbol{f}_{\text{int}}(\boldsymbol{u}_m)]$.
  * SVD basis extraction for force space: $\boldsymbol{\Psi}_f \in \mathbb{R}^{n \times m_f}$.
  * Greedy DEIM (Chaturantabut & Sorensen 2010) and Q-DEIM (Drmac & Gugercin 2016 QR with column pivoting) row selection determining interpolation indices $\boldsymbol{P} \in \mathbb{R}^{n \times m_f}$.
  * Pre-projected online DEIM operator: $\mathbb{E}_{\text{DEIM}} = \boldsymbol{\Phi}_f^T \boldsymbol{\Psi}_f (\boldsymbol{P}^T \boldsymbol{\Psi}_f)^{-1} \in \mathbb{R}^{r \times m_f}$.
  * Sparse element sub-mesh cache: evaluates internal forces and tangent stiffness ONLY on elements touching sampled interpolation DOFs $\boldsymbol{P}$ with vectorized scatter.
  * Support for internal history variables (rate-dependent viscoelasticity delegation) and optional tangent stiffness symmetrization.
  * Delivered core modules: [`greedy_deim`](../src/psierra/hyperreduction/deim.py), [`q_deim`](../src/psierra/hyperreduction/deim.py), [`DEIMForceBasis`](../src/psierra/hyperreduction/deim.py), [`ForceSnapshotManager`](../src/psierra/hyperreduction/deim.py), [`HyperReducedDEIMModel`](../src/psierra/hyperreduction/deim.py), and unified model-agnostic solvers [`HyperReducedDynamicSolver`](../src/psierra/hyperreduction/solver.py), [`HyperReducedStaticSolver`](../src/psierra/hyperreduction/solver.py).
  * 11 automated verification tests in [`tests/test_hyperreduction_deim.py`](../tests/test_hyperreduction_deim.py) validating greedy/Q-DEIM index selection, non-singularity, condition number, HDF5 streaming, model force fidelity, static equilibrium, and real-time Generalized-alpha dynamic solves (100% pass rate).
  * Delivered new executable simulation: [`examples/13_hyperreduction/1301_deim_hyperelastic_force.py`](../examples/13_hyperreduction/1301_deim_hyperelastic_force.py) comparing FOM, ECSW, and DEIM dynamic beam deformation under gravity load.
* [x] **Contact-Aware ECSW & Enriched POD Basis (Method 2 Hyper-Reduced Contact)**:
  * Joint cubature training in ECSW incorporating virtual work from contact boundary interfaces:
    $$\boldsymbol{A}_{\text{joint}} = \begin{bmatrix} \boldsymbol{A}_{\text{vol}} \\ \alpha_c \boldsymbol{A}_{\text{contact}} \end{bmatrix}, \quad \boldsymbol{c}_{c, i}^{(k)} = \boldsymbol{\Phi}_{s, i}^T \boldsymbol{f}_{c, i}^{(k)}$$
  * Automatic clustering of reduced integration domain (RID) elements around high-gradient contact zones (`ContactAwareECSWOptimizer` with scoring boost $\beta_c \ge 1.0$).
  * Localized modal basis enrichment: Krylov compliance modes ($K_0^{-1} f_c$) and dual-mode basis $\boldsymbol{\Phi} = [\boldsymbol{\Phi}_{\text{bulk}}, \boldsymbol{\Phi}_{\text{contact}}]$ with exact block-orthonormality ($\|\boldsymbol{\Phi}^T \boldsymbol{\Phi} - \boldsymbol{I}_r\|_F < 10^{-14}$).
  * Delivered core modules: [`EnrichedPODBasis`](../src/psierra/reduction/basis.py), [`ContactAwareReducedIntegrationDomain`](../src/psierra/hyperreduction/contact_ecsw.py), [`ContactAwareECSWTrainingMatrix`](../src/psierra/hyperreduction/contact_ecsw.py), [`ContactAwareECSWOptimizer`](../src/psierra/hyperreduction/contact_ecsw.py), [`ContactAwareHyperReducedModel`](../src/psierra/hyperreduction/contact_ecsw.py).
  * 7 automated mathematical verification tests in [`tests/test_hyperreduction_contact_aware_ecsw.py`](../tests/test_hyperreduction_contact_aware_ecsw.py) validating orthonormality, exact contact virtual work conservation, RID contact clustering, tangent stiffness symmetry, and real-time dynamic solves.
  * Delivered new executable simulation: [`examples/13_hyperreduction/1302_contact_aware_ecsw.py`](../examples/13_hyperreduction/1302_contact_aware_ecsw.py).
* [ ] **Pure CPU Low-Latency Execution**:
  * Retain CPU-optimized dense Cholesky/BLAS for online ROM execution to avoid PCIe bus transfer latency ($< 15\ \mu\text{s}$ response, $> 1,000\text{ Hz}$).

---

### Phase 4: Clinical Digital Twins, Advanced Biomechanical Constitutive Models & Grand Showcase
*Objective*: Advance closed-loop intraoperative clinical navigation, implement advanced hyperelastic soft-tissue constitutive models (Ogden and Holzapfel-Gasser-Ogden - HGO) for brain, skin, vascular (aorta), and dural membranes, and deliver a comprehensive gallery of executable examples showcasing all capabilities of the library.
* [x] **Quasi-Static Analysis & Hyper-Reduced Solver with Viscoelastic Relaxation**:
  * Implemented [`QuasiStaticAnalysis`](../src/psierra/analysis/quasi_static.py) and [`QuasiStaticResult`](../src/psierra/analysis/quasi_static.py) for physical time marching $t_{n+1} = t_n + \Delta t$ without inertia, capturing rate-dependent viscoelastic stress relaxation, creep, and dissipation.
  * Transactional history management (`set_time_step`, `commit_state`, `revert_state`), automatic adaptive time substepping cutback upon Newton divergence, and full Augmented Lagrangian contact integration.
  * Implemented [`HyperReducedQuasiStaticSolver`](../src/psierra/hyperreduction/solver.py) solving nonlinear rate-dependent equilibrium on POD-ECSW reduced coordinates $\boldsymbol{u}_r \in \mathbb{R}^r$ in microseconds ($> 600\text{ Hz}$, $1,600\times$ faster than FOM).
  * Upgraded [`HyperReducedECSWModel`](../src/psierra/hyperreduction/model.py) with element-indexed constitutive evaluation (`elem_index=entry.elem_idx`) and domain history forwarding (`set_time_step`, `commit_state`, `revert_state`, `reset_state`).
  * 6 automated tests in [`tests/test_analysis_quasi_static.py`](../tests/test_analysis_quasi_static.py) and 3 automated tests in [`tests/test_hyperreduced_quasi_static.py`](../tests/test_hyperreduced_quasi_static.py) (100% pass rate).
  * Delivered comprehensive benchmark demonstration: [`examples/11_advanced_continua/1109_quasi_static_gravity_beam.py`](../examples/11_advanced_continua/1109_quasi_static_gravity_beam.py) directly comparing Static, Dynamic Relaxation, Full-Order Quasi-Static, and Hyper-Reduced Quasi-Static beam deformation under gravity.
* [x] **Generalized Ogden Hyperelasticity (`psierra.material.ogden`)**:
  * Implemented for arbitrary order $N \in [1, 3]$ formulated on principal stretches $\lambda_a$ ($a=1,2,3$):
    $$W_{\text{Ogden}} = \sum_{p=1}^N \frac{\mu_p}{\alpha_p} \left( \bar{\lambda}_1^{\alpha_p} + \bar{\lambda}_2^{\alpha_p} + \bar{\lambda}_3^{\alpha_p} - 3 \right) + W_{\text{vol}}(J)$$
  * Dual volumetric penalty support: logarithmic $W_{\text{vol}} = \frac{K}{2}(\ln J)^2$ (preventing inversion) and quadratic polynomial $W_{\text{vol}} = \frac{K}{2}(J - 1)^2$.
  * Exact spectral decomposition of right Cauchy-Green tensor $\boldsymbol{C} = \sum_{a=1}^3 \lambda_a^2 \boldsymbol{N}_a \otimes \boldsymbol{N}_a$ with closed-form perturbation handling for coalescent/equal eigenvalues ($\lim_{\Lambda_b \to \Lambda_a} \frac{s_a - s_b}{\Lambda_a - \Lambda_b} = \frac{1}{2}(D_{aa} - D_{ab})$).
  * Consistent 2nd Piola-Kirchhoff stress $\boldsymbol{S}$ and 4th-order spatial/material tangent elasticity tensor $\mathbb{C}$ verified against numerical finite-difference to $< 4 \times 10^{-7}$ relative error.
  * Universal continuum modularity verified across `Tet4TotalLagrangian`, `Tet4UpdatedLagrangian`, `Tet4Fbar` (locking-alleviated), `Tet10TotalLagrangian` (quadratic serendipity), `Tet4Corotational`, and rate-dependent `Viscoelastic` wrapper.
  * 30 automated mathematical verification tests in [`tests/test_material_ogden.py`](../tests/test_material_ogden.py) (100% pass rate).
  * Delivered new executable simulation: [`examples/11_advanced_continua/1110_ogden_dynamic_beam.py`](../examples/11_advanced_continua/1110_ogden_dynamic_beam.py) benchmarking dynamic gravitational beam deflection across Ogden Orders 1, 2, and 3 with XDMF and PVD export.
* [x] **Anisotropic Hyperelasticity (Holzapfel-Gasser-Ogden - HGO Model, `psierra.material.hgo`)**:
  * Structural anisotropic continuum mechanics with coupled isotropic ground matrix and anisotropic collagen fiber bundles:
    $$W_{\text{HGO}} = \frac{c}{2}(\bar{I}_1 - 3) + \sum_{k=1}^M \frac{k_1}{2k_2} \left[ \exp\left( k_2 E_k^2 \right) - 1 \right] + W_{\text{vol}}(J)$$
    where $E_k = \kappa \bar{I}_1 + (1 - 3\kappa) \bar{I}_{4k} - 1$.
  * Invariant formulation ($\bar{I}_1 = J^{-2/3}\operatorname{tr}\boldsymbol{C}$, $\bar{I}_{4k} = J^{-2/3} \boldsymbol{a}_{0k} \cdot \boldsymbol{C} \boldsymbol{a}_{0k} = \bar{\lambda}_{\text{fiber}, k}^2$) eliminating costly eigenvalue decompositions and accelerating runtime solves.
  * Structural fiber dispersion $\kappa \in [0, 1/3]$ capturing fiber waviness and 3D orientation distribution ($\kappa = 0$: ideal parallel alignment; $\kappa = 1/3$: isotropic random network).
  * Physiologically accurate tension-only fiber activation ($E_k > 0$): Collagen fibers buckle with zero load-bearing under compressive strain ($E_k \le 0$) and exponentially stiffen under tensile elongation.
  * Parameter management via [`HGOParameters`](../src/psierra/material/parameters.py) with validation, arbitrary fiber families, and pre-calibrated physiological presets:
    * Human aorta layers: `media` (2 symmetric helical families at $\pm 29^\circ$, $\kappa = 0.226$), `adventitia` ($\pm 62^\circ$, $\kappa = 0.060$), `intima` ($\pm 18^\circ$, $\kappa = 0.147$).
    * Dural membrane: `falx_cerebri` ($\pm 35^\circ$ sagittal reinforcement).
  * Dual volumetric penalty formulations: logarithmic $W_{\text{vol}} = \frac{K}{2}(\ln J)^2$ and quadratic $W_{\text{vol}} = \frac{K}{2}(J - 1)^2$.
  * Consistent 2nd Piola-Kirchhoff stress $\boldsymbol{S}$ and symmetric 4th-order tangent elasticity matrix $\mathbb{C}$ verified against numerical finite-difference to $< 2.7 \times 10^{-7}$ relative error.
  * Universal continuum modularity verified across `Tet4TotalLagrangian`, `Tet4UpdatedLagrangian`, `Tet4Fbar`, `Tet10TotalLagrangian`, and rate-dependent `Viscoelastic` wrapper.
  * 19 automated mathematical verification tests in [`tests/test_material_hgo.py`](../tests/test_material_hgo.py) (100% pass rate).
  * Delivered new executable simulation: [`examples/11_advanced_continua/1111_hgo_anisotropic_bending.py`](../examples/11_advanced_continua/1111_hgo_anisotropic_bending.py) benchmarking anisotropic gravity cantilever bending across 4 configurations (Isotropic Elastin baseline, Longitudinal Fibers, Transverse Fibers, and Symmetric Helical Fibers).
* [x] **Online Hyper-Reduced Inverse Parameter Identification**:
  * Real-time in-vivo tissue stiffness calibration during active robotic tool contact.
  * Delivered [`examples/10_inverse/1008_hyperreduced_brain_parameter_identification.py`](../examples/10_inverse/1008_hyperreduced_brain_parameter_identification.py):
    * Coupled ECSW empirical cubature (12 / 90,586 elements, 7549x reduction) with Trust-Region Reflective (TRF) nonlinear least-squares optimizer.
    * Real-time in-vivo calibration of brain parenchyma Young's modulus $E$ (and joint $E, \nu$) from multi-stage robotic indentation in < 150 ms with < 0.001% error.
    * Interactive 3D PyVista GUI with live cortical indentation, recalibration hotkey `[Space]`, stiffness perturbation `[P]`, and HUD telemetry.
  * Added automated test in [`tests/test_ifem_parameter_identification.py`](../tests/test_ifem_parameter_identification.py) (`test_hyperreduced_parameter_identification`).
* [x] **Multi-Modal Intraoperative Sensor Fusion**:
  * Fusing sparse cortical surface optical markers with intraoperative ultrasound (iUS) and stereovision point clouds for enhanced deep-target tracking accuracy.
  * Delivered [`examples/10_inverse/1009_multimodal_sensor_fusion.py`](../examples/10_inverse/1009_multimodal_sensor_fusion.py):
    * Anatomically accurate burr hole trephination matching simulation `1108` ($R = 10.0\text{ mm}$, $\varnothing = 20.0\text{ mm}$, normal $\boldsymbol{n} = [-0.069, 0.654, 0.754]$) with 3D annular bone table collar and aperture perimeter ring.
    * Physically accurate brain shift under 30° lateral head posture with 85% CSF buoyancy relief (15% net gravity sag), keeping brain anatomically contained inside cranial vault ($\sim 2.18\text{ mm}$ cortical shift, $\sim 2.21\text{ mm}$ deep tumor drift).
    * Dual heterogeneous point clouds referenced to burr hole: 28 optical surface markers on exposed cortex ($\sigma = 0.05\text{ mm}$, $W = 1.0$) and 24 deep ultrasound (iUS) boundary points on lesion margin ($\sigma = 0.25\text{ mm}$, $W = 0.04$, depth $22.2\text{ mm}$).
    * Tri-modal comparative benchmark: Surface Optical Only ($0.208\text{ mm}$ error) vs. Deep iUS Only ($0.091\text{ mm}$) vs. Fused Multi-Modal ($0.085\text{ mm}$ FOM, $0.024\text{ mm}$ ROM, $59.4\%$ accuracy gain over surface alone).
    * Real-time ROM subspace inversion running in $24.7\ \mu\text{s}$ ($40{,}537\text{ Hz}$).
    * Interactive 3D PyVista GUI rendering skull vault, annular bone table collar, green optical surface point cloud, magenta ultrasound target cloud, deformed brain, and reconstructed lesion wireframe.
  * Added automated verification in [`tests/test_ifem_shape_sensing.py`](../tests/test_ifem_shape_sensing.py) (`test_multimodal_sensor_fusion_tracking`).
* [x] **Interactive Hyper-Reduced Brain Shift Inverse Identification**:
  * Real-time inverse tracking and dynamic shape sensing of surgical brain shift and subcortical tumor drift.
  * Delivered [`examples/10_inverse/1010_interactive_hyperreduced_brain_shift_sensing.py`](../examples/10_inverse/1010_interactive_hyperreduced_brain_shift_sensing.py):
    * Translates floating multi-modal sensor point clouds (cortical optical surface + deep ultrasound iUS) referenced to the anatomical burr hole aperture frame ($\hat{\boldsymbol{n}}, \boldsymbol{t}_1, \boldsymbol{t}_2$).
    * Interactive PyVista 3D GUI featuring 4 real-time slider widgets (Normal shift $\delta_n \in [-5, 5]\text{ mm}$ for sagging vs. bulging, Tangential shifts $\delta_{t1}, \delta_{t2} \in [-4, 4]\text{ mm}$, and Deep target coupling factor $\gamma$).
    * Keyboard hotkeys for surgical navigation: `[W]`/`[S]` (normal), `[A]`/`[D]` & `[Up]`/`[Down]` (tangential), `[M]` (cycle Fused/Surface/Deep modality), `[1]`/`[2]`/`[3]` (clinical presets: Sagging $-3.2\text{ mm}$, Bulging $+2.8\text{ mm}$, Compound Shear), `[R]` (reset), `[Q]` (exit).
    * Solves generalized modal coordinates in $< 70\ \mu\text{s}$ ($> 14{,}000\text{ Hz}$), instantly updating the 3D deformed brain mesh, green optical markers, magenta deep ultrasound feature points, and reconstructed subcortical lesion position at 60 FPS.
    * Automated headless multi-step verification sequence (`--headless`) exporting `interactive_fused_brain_shift.vtu`.
  * Added automated unit test in [`tests/test_ifem_shape_sensing.py`](../tests/test_ifem_shape_sensing.py) (`test_interactive_burr_hole_point_cloud_inversion`).
* [ ] **Loop 2 Grand Showcase Gallery (All-Capabilities Demonstration)**:
  * Deliver standalone executable, PyVista-visualized examples covering every simulation mode in `pSierra`:
    1. [x] `examples/11_advanced_continua/1101_tet4fbar_incompressibility.py`: 15-step NonlinearStaticAnalysis (Neo-Hookean, $\nu = 0.499, K/\mu = 500$): Tet4TotalLagrangian locking ($0.043\ \text{mm}$) vs. Tet4Fbar compliance ($0.127\ \text{mm}$, $2.99\times$ ratio). Exports `result_tet4_tl.vtu`, `result_tet4_fbar.vtu`, and `.pvd` animations.
    2. `examples/11_advanced_continua/1102_viscoelastic_relaxation.py`: Probe indentation and 30s stress relaxation holding.
    3. `examples/11_advanced_continua/1103_dynamic_relaxation_solver.py`: Matrix-free static equilibrium via kinetic damping.
    4. `examples/11_advanced_continua/1104_csf_hydrostatic_drainage.py`: Chen et al. CSF drainage and buoyancy loss during craniotomy.
    5. `examples/11_advanced_continua/1105_hdf5_checkpoint_resume.py`: Out-of-core snapshot streaming, stopping, and seamless resume.
    6. `examples/12_advanced_contact/1201_3pillar_regularized_impact.py`: High-energy dynamic impact with adaptive cutback & $C^1$ pseudonormals.
    7. `examples/12_advanced_contact/1202_mortar_contact_friction.py`: Segment-to-segment mortar contact with Coulomb stick-slip.
    8. [x] `examples/13_hyperreduction/1301_deim_hyperelastic_force.py`: Three-way dynamic beam deformation under gravity load benchmark comparing Full-Order Model (FOM Generalized-alpha), ECSW empirical cubature, and DEIM interpolation with ParaView XDMF/H5 and PVD export.
    9. [x] `examples/13_hyperreduction/1302_contact_aware_ecsw.py`: Deformable contact hyper-reduced with joint bulk + contact cubature.
    10. [x] `examples/13_hyperreduction/1303_hyperreduced_brain_shift_shell_contact.py`: Real-time ECSW hyper-reduced neurosurgical brain shift with coupled solid-shell meninges (Tet4 + Shell3) and dual multi-obstacle contact (concave inner skull vault + surgical indenter probe).
    11. [x] `examples/13_hyperreduction/1304_hyperreduced_brain_shift_dynamic_relaxation.py`: Hyper-reduced explicit Dynamic Relaxation (DR) static equilibrium solver with Barnes/Underwood kinetic energy peak damping, coupled solid-shell meninges, and dual skull + probe contact.
    12. [x] `examples/11_advanced_continua/1109_quasi_static_gravity_beam.py`: Four-way comparative benchmark: Nonlinear Static vs. Dynamic Relaxation vs. Full-Order Quasi-Static vs. Hyper-Reduced Quasi-Static under gravity.
    13. `examples/14_clinical_digital_twins/1401_neurosurgical_digital_twin_complete.py`: **The Grand Flagship Simulation** combining patient-specific brain with tumor, Tet4Fbar + viscoelasticity, falx cerebri membrane, CSF drainage, real-time ROM shape sensing, and HDF5 storage.

---

## 4. Architectural Rules, Invariants & Development Policies

Every agent and developer working on `pSierra` must uphold the following engineering standards:

1. **Strict Type Annotations**:
   - All classes, methods, and functions must carry complete type annotations (`from __future__ import annotations`).
   - Every test method must end with explicit return annotation `-> None`.
2. **Regression Verification Policy**:
   - Run unit and integration tests with `.venv/bin/pytest tests/` before concluding any turn.
   - Maintain 100% pass rate across the full suite (no regressions allowed).
3. **Git Commit Policy**:
   - **NEVER** run `git commit` automatically. Always leave changes unstaged for the user to review and commit.
4. **Clickable File Links**:
   - Always link source files and docs using GitHub-style clickable markdown links (`file:///...` or relative markdown links).
5. **Unified Visualization Standard**:
   - Displacements must be stored strictly as a 3D vector array in `point_data["Displacement"]` with `active_scalars = "Displacement"`.
   - Never decompose into separate scalar arrays (`_X`, `_Y`, `_Z`).
6. **Lean Plan Maintenance**:
   - Keep this `PLAN.md` file focused on active and upcoming tasks. When new phases are completed, summarize them and keep historical logs in the archive.
7. **Body Force & Benchmark Loading Policy**:
   - For volumetric locking alleviation, near-incompressibility ($\nu \to 0.5$), and soft-tissue continuum demonstrations, prefer distributed gravitational body force (`GravityLoad`) over concentrated point/tip loads whenever gravity load better showcases volumetric compliance.
   - Distributed body force engages the volumetric constraint $\operatorname{div} \boldsymbol{u} \approx 0$ across the entire domain without introducing artificial stress concentrations or numerical singularities associated with point loads.
