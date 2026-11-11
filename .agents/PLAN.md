# Project pSierra Development Plan & Roadmap (Loop 2: Phases 1–4)

This document outlines the active engineering milestones, mathematical formulations, and architectural roadmap for Project `pSierra`.

> [!NOTE]
> **Loop 1 (Phases 1 through 5) is fully completed and verified.**
> For the comprehensive historical checklists, mathematical derivations, and delivery logs of Loop 1, see the archive:
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
* [ ] **Discrete Empirical Interpolation Method (DEIM / Q-DEIM)**:
  * Out-of-core force snapshot matrix assembly streamed directly into HDF5 datasets: $\boldsymbol{S}_f = [\boldsymbol{f}_{\text{int}}(\boldsymbol{u}_1), \dots, \boldsymbol{f}_{\text{int}}(\boldsymbol{u}_m)]$.
  * SVD basis extraction for force space: $\boldsymbol{\Psi}_f \in \mathbb{R}^{n \times m_f}$.
  * Greedy DEIM / Q-DEIM row selection determining interpolation indices $\boldsymbol{P} \in \mathbb{R}^{n \times m_f}$.
  * Support for internal history variables (viscoelasticity).
  * Online reduced nonlinear force approximation: $\boldsymbol{f}_r \approx \boldsymbol{\Phi}^T \boldsymbol{\Psi}_f (\boldsymbol{P}^T \boldsymbol{\Psi}_f)^{-1} \boldsymbol{f}_{\text{int}}(\boldsymbol{u})[\boldsymbol{P}]$.
* [ ] **Contact-Aware ECSW & Enriched POD Basis (Method 2 Hyper-Reduced Contact)**:
  * Joint cubature training in ECSW incorporating virtual work from contact boundary interfaces:
    $$\boldsymbol{A}_{\text{joint}} = \begin{bmatrix} \boldsymbol{A}_{\text{vol}} \\ \alpha_c \boldsymbol{A}_{\text{contact}} \end{bmatrix}, \quad \boldsymbol{c}_{c, i}^{(k)} = \boldsymbol{\Phi}_{s, i}^T \boldsymbol{f}_{c, i}^{(k)}$$
  * Automatic clustering of reduced integration domain (RID) elements around high-gradient contact zones.
  * Localized modal basis enrichment: Krylov contact vectors and dual-mode basis $\boldsymbol{\Phi} = [\boldsymbol{\Phi}_{\text{bulk}}, \boldsymbol{\Phi}_{\text{contact}}]$.
* [ ] **Pure CPU Low-Latency Execution**:
  * Retain CPU-optimized dense Cholesky/BLAS for online ROM execution to avoid PCIe bus transfer latency ($< 15\ \mu\text{s}$ response, $> 1,000\text{ Hz}$).

---

### Phase 4: Clinical Digital Twins, Real-Time Navigation & Grand Showcase
*Objective*: Advance closed-loop intraoperative clinical navigation, and deliver a comprehensive gallery of executable examples showcasing all capabilities of the library.
* [ ] **Quasi-Static Surgical Indentation with Viscoelastic Relaxation**:
  * Simulation pipeline for slow surgical probe indentation and holding with stress relaxation.
* [ ] **Online Hyper-Reduced Inverse Parameter Identification**:
  * Real-time in-vivo tissue stiffness calibration during active robotic tool contact.
* [ ] **Multi-Modal Intraoperative Sensor Fusion**:
  * Fusing sparse cortical surface optical markers with intraoperative ultrasound (iUS) and stereovision point clouds for enhanced deep-target tracking accuracy.
* [ ] **Anisotropic Hyperelasticity (Holzapfel-Gasser-Ogden - HGO Model)**:
  * Fiber-reinforced biological membranes (dura mater, falx cerebri, tentorium).
* [ ] **Loop 2 Grand Showcase Gallery (All-Capabilities Demonstration)**:
  * Deliver standalone executable, PyVista-visualized examples covering every simulation mode in `pSierra`:
    1. [x] `examples/11_advanced_continua/1101_tet4fbar_incompressibility.py`: 15-step NonlinearStaticAnalysis (Neo-Hookean, $\nu = 0.499, K/\mu = 500$): Tet4TotalLagrangian locking ($0.043\ \text{mm}$) vs. Tet4Fbar compliance ($0.127\ \text{mm}$, $2.99\times$ ratio). Exports `result_tet4_tl.vtu`, `result_tet4_fbar.vtu`, and `.pvd` animations.
    2. `examples/11_advanced_continua/1102_viscoelastic_relaxation.py`: Probe indentation and 30s stress relaxation holding.
    3. `examples/11_advanced_continua/1103_dynamic_relaxation_solver.py`: Matrix-free static equilibrium via kinetic damping.
    4. `examples/11_advanced_continua/1104_csf_hydrostatic_drainage.py`: Chen et al. CSF drainage and buoyancy loss during craniotomy.
    5. `examples/11_advanced_continua/1105_hdf5_checkpoint_resume.py`: Out-of-core snapshot streaming, stopping, and seamless resume.
    6. `examples/12_advanced_contact/1201_3pillar_regularized_impact.py`: High-energy dynamic impact with adaptive cutback & $C^1$ pseudonormals.
    7. `examples/12_advanced_contact/1202_mortar_contact_friction.py`: Segment-to-segment mortar contact with Coulomb stick-slip.
    8. `examples/13_hyperreduction/1301_deim_hyperelastic_force.py`: Discrete Empirical Interpolation (DEIM) vs. ECSW.
    9. `examples/13_hyperreduction/1302_contact_aware_ecsw.py`: Deformable contact hyper-reduced with joint bulk + contact cubature.
    10. `examples/14_clinical_digital_twins/1401_neurosurgical_digital_twin_complete.py`: **The Grand Flagship Simulation** combining patient-specific brain with tumor, Tet4Fbar + viscoelasticity, falx cerebri membrane, CSF drainage, real-time ROM shape sensing, and HDF5 storage.

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
