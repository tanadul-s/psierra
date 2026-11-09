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
* [ ] **Volumetric Locking Alleviation (`Tet4Fbar` / B-bar Mean Dilatation)**:
  * Implement `Tet4Fbar` with Flory volumetric/isochoric split: $\bar{\boldsymbol{F}} = (\bar{J}/J)^{1/3} \boldsymbol{F}$.
  * Nodal-averaged volumetric dilatation eliminating locking in nearly incompressible soft tissue ($\nu \to 0.5$).
  * Formulated within the Total Lagrangian framework; consistent material and geometric tangent stiffness matrices $\boldsymbol{K}_T$.
* [ ] **Viscoelastic Constitutive Laws**:
  * Generalized Maxwell / Prony series relaxation spectrum: $G(t) = G_\infty + \sum_{m=1}^{N_p} G_m e^{-t / \tau_m}$.
  * Internal strain history variable tracking per quadrature point for stress relaxation and creep.
* [ ] **Unified HDF5 Simulation Storage & Checkpointing (`h5py`)**:
  * Single-file hierarchical storage container (`.h5`): mesh, constitutive parameters, resizable snapshot datasets, and reduced bases.
  * Simulation state checkpointing: `SimulationCheckpoint(t_n, u_n, v_n, a_n, lambda_n, dt_n)`.
  * Seamless resume capability: `analysis.resume(checkpoint, additional_steps=100)` appending directly to HDF5 datasets and PVD XML manifests without memory overhead.
* [ ] **Dynamic Relaxation Equilibrium Solver (`DynamicRelaxationAnalysis`)**:
  * Matrix-free explicit static solver using fictitious Gerschgorin-scaled diagonal mass $\boldsymbol{M}_{\text{fict}}$.
  * Barnes/Underwood kinetic damping ($\boldsymbol{v} \leftarrow \mathbf{0}$ when $dE_k/dt < 0$).
  * Provides unconditional stability against tangent stiffness singularities ($\det(\boldsymbol{K}_T) \le 0$) during severe non-smooth contact and tissue collapse.
* [ ] **CSF Hydrostatic Pressure & Buoyancy Load (`HydrostaticPressureLoad`)**:
  * *References*: Chen et al. (2017, 2019, 2022) IEEE SII & TMRB.
  * Spatially-varying boundary traction on submerged surface facets: $\boldsymbol{F}_{\text{hydro}} = -\rho_{\text{CSF}} g \max(0, h_{\text{fluid}} - z) A_k \boldsymbol{n}_k$.
  * Configurable patient head orientation ($\hat{\boldsymbol{u}}_{\text{up}} = -\boldsymbol{g}/\|\boldsymbol{g}\|$) and dynamic CSF drainage fluid level $h_{\text{fluid}}$.
  * Models Archimedean neutral buoyancy and gravity sag progression as CSF leaks during craniotomy.

---

### Phase 2: Advanced Contact Regularization & Non-Smooth Mechanics
*Objective*: Resolve high-energy dynamic impact breakdowns, facet edge chattering, and line-search stalls on locking-free continua.
* *References*: P. Wriggers (2006) *Computational Contact Mechanics*; T.A. Laursen (2002) *Computational Contact and Impact Mechanics*; FEBio Theory Manual.
* [ ] **Pillar 1: Temporal Regularization (Adaptive Time Substepping & Automatic Cutback)**:
  * Implement `AdaptiveTimeStepController` in `NonlinearDynamicAnalysis`.
  * Automatic step cutback: $\Delta t \leftarrow \gamma_{\text{cut}} \Delta t$ upon line search stall, restoring state $(\boldsymbol{u}_n, \boldsymbol{v}_n, \boldsymbol{a}_n, \boldsymbol{\lambda}_n)$.
  * Dynamic ramp-up upon consecutive quadratic Newton-Raphson convergences.
* [ ] **Pillar 2: Spatial Regularization ($C^1$-Continuous Surface Normal Smoothing)**:
  * Upgrade `DeformableContact` with area/angle-weighted vertex pseudonormal interpolation:
    $$\boldsymbol{n}(\boldsymbol{\xi}) = \frac{\sum \xi_i \bar{\boldsymbol{n}}_i}{\|\sum \xi_i \bar{\boldsymbol{n}}_i\|}$$
  * Eliminates directional jumps $\Delta \boldsymbol{n} = \boldsymbol{n}_B - \boldsymbol{n}_A$ as nodes cross facet edges, restoring quadratic Newton descent.
* [ ] **Pillar 3: Algorithmic Regularization (Contact-Aware Line Search & Dual Relaxation)**:
  * Differentiate constitutive element inversion ($\det \boldsymbol{J} \le 0$) from non-smooth facet edge crossing during line search.
  * Dual rebound relaxation: zeroing retained multipliers ($\lambda \leftarrow 0$) when local rebound opens a positive gap ($g > 0$).
* [ ] **Segment-to-Segment Mortar Contact**:
  * Mortar contact formulation for non-conforming mesh interfaces.
* [ ] **Exact Coulomb Stick-Slip Friction**:
  * Active-set Coulomb friction strategy with stick/slip transition boundaries.

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
    1. `examples/11_advanced_continua/1101_tet4fbar_incompressibility.py`: Tet4 locking vs. Tet4Fbar compliance ($\nu = 0.499$).
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
