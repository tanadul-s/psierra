# Project pSierra Development Plan & Roadmap (Loop 2: Phases 1–5)

This document outlines the strategic milestones, active task tracking, and architectural roadmap for Project `pSierra`.

> [!NOTE]
> **Loop 1 (Phases 1 through 5) is fully completed and verified.**
> For the comprehensive historical checklists, mathematical derivations, and delivery logs of Loop 1, see the archive:
> 📂 **[`PLAN_ARCHIVE_LOOP1.md`](PLAN_ARCHIVE_LOOP1.md)**

---

## 1. Strategic Vision: Closed Loop to Post-Loop Enhancements

Project `pSierra` has successfully closed the primary continuum-to-inverse loop:
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
==================== LOOP 2: ADVANCED MECHANICS & EXTENSIONS (ACTIVE ROADMAP) ============
[ Phase 1: Contact Regularization & Non-Smooth Mechanics (3-Pillars, Mortar, Friction) ]
       │
       ▼
[ Phase 2: Advanced Hyper-Reduction (DEIM, Q-DEIM & Contact-Aware ECSW) ]
       │
       ▼
[ Phase 3: Biomechanical Specialization & Multiphysics Continua ]
       │
       ▼
[ Phase 4: Advanced Instability Solvers & Structured Hexahedral Discretizations ]
       │
       ▼
[ Phase 5: High-Capacity Streaming Backends & Clinical Digital Twins ]
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

## 3. Loop 2 Active Roadmap: Phase 1 to Phase 5

This section tracks the active forward-looking engineering milestones extending `pSierra` through Loop 2.

### Phase 1: Advanced Contact Mechanics & The 3-Pillar Regularization Architecture
*Objective*: Resolve high-energy dynamic impact breakdowns, facet edge chattering, and line-search stalls in complex deformable-to-deformable mesh contact.
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

### Phase 2: Advanced Non-Linear Hyper-Reduction (DEIM, Q-DEIM & Contact-Aware ECSW)
*Objective*: Evaluate non-linear internal forces and non-polynomial constitutive laws on low-dimensional sample points, and extend hyper-reduction to deformable contact.
* [ ] **Discrete Empirical Interpolation Method (DEIM / Q-DEIM)**:
  * Force snapshot matrix assembly: $\boldsymbol{S}_f = [\boldsymbol{f}_{\text{int}}(\boldsymbol{u}_1), \dots, \boldsymbol{f}_{\text{int}}(\boldsymbol{u}_m)] \in \mathbb{R}^{n \times m}$.
  * SVD basis extraction for force space: $\boldsymbol{\Psi}_f \in \mathbb{R}^{n \times m_f}$.
  * Greedy DEIM / Q-DEIM row selection determining interpolation indices $\boldsymbol{P} \in \mathbb{R}^{n \times m_f}$.
  * Online reduced nonlinear force approximation:
    $$\boldsymbol{f}_r \approx \boldsymbol{\Phi}^T \boldsymbol{\Psi}_f (\boldsymbol{P}^T \boldsymbol{\Psi}_f)^{-1} \boldsymbol{f}_{\text{int}}(\boldsymbol{u})[\boldsymbol{P}]$$
  * Comparison benchmark between ECSW and DEIM on hyperelastic cantilever beam.
* [ ] **Contact-Aware ECSW & Enriched POD Basis (Method 2 Hyper-Reduced Contact)**:
  * Joint cubature training in ECSW incorporating virtual work from contact boundary interfaces:
    $$\boldsymbol{A}_{\text{joint}} = \begin{bmatrix} \boldsymbol{A}_{\text{vol}} \\ \alpha_c \boldsymbol{A}_{\text{contact}} \end{bmatrix}, \quad \boldsymbol{c}_{c, i}^{(k)} = \boldsymbol{\Phi}_{s, i}^T \boldsymbol{f}_{c, i}^{(k)}$$
  * Automatic clustering of reduced integration domain (RID) elements around high-gradient contact zones.
  * Localized modal basis enrichment: Krylov contact vectors and dual-mode basis $\boldsymbol{\Phi} = [\boldsymbol{\Phi}_{\text{bulk}}, \boldsymbol{\Phi}_{\text{contact}}]$.
  * Deformable-to-deformable hyper-reduced contact with dual RIDs and BVH tracking.

---

### Phase 3: Biomechanical Specialization & Multiphysics Continua
*Objective*: Model physiological time-dependent, anisotropic, nearly-incompressible, and fluid-coupled tissue phenomena.
* [ ] **Viscoelastic Constitutive Laws**:
  * Generalized Maxwell / Prony series relaxation spectrum:
    $$G(t) = G_\infty + \sum_{m=1}^{N_p} G_m e^{-t / \tau_m}$$
  * Internal history variable tracking per quadrature point.
* [ ] **Nearly-Incompressible Formulations & Volumetric Locking Alleviation**:
  * `Tet4Fbar` / B-bar (Mean Dilatation) formulation.
  * Three-field mixed $u-p-\theta$ formulation for soft tissues with $\nu \to 0.5$.
* [ ] **Anisotropic Hyperelasticity**:
  * Holzapfel-Gasser-Ogden (`HGO`) model for fiber-reinforced biological membranes (arteries, dura mater, falx cerebri).
* [ ] **Poroelasticity (Biot Consolidation)**:
  * Coupled fluid-solid mixed formulation for parenchymal fluid flow (hydrocephalus, edema).
  * Taylor-Hood `Tet10`-`Tet4` mixed elements satisfying the LBB / inf-sup condition.
* [ ] **Unified Multi-Physics Law**:
  * Coupled hyper-visco-poroelastic constitutive laws for perfused parenchymal organs.

---

### Phase 4: Advanced Instability Solvers & Structured Hexahedral Discretizations
*Objective*: Solve non-monotonic post-buckling load-displacement paths and support high-accuracy structured hexahedral elements.
* [ ] **Arc-Length / Riks Method (`ArcLengthAnalysis`)**:
  * Crisfield cylindrical arc-length constraint for post-buckling, snap-through, and snap-back equilibrium paths where $\det(\boldsymbol{K}_T) \le 0$.
* [ ] **Dynamic Relaxation Equilibrium Solver (`DynamicRelaxationAnalysis`)**:
  * Barnes/Underwood kinetic damping algorithm ($\boldsymbol{v} \leftarrow \mathbf{0}$ at kinetic energy peaks) with Gerschgorin diagonal mass scaling for highly unstable softening paths.
* [ ] **Structured Hexahedral Volume Discretizations**:
  * Hexahedral volume meshing infrastructure (structured sweeping, polycube mapping).
  * Linear 8-node brick element (`Hex8`) with B-bar / Selective Reduced Integration (SRI).
  * Quadratic 20-node serendipity brick element (`Hex20`).
  * Quadrilateral shell element (`Shell4` / MITC4).

---

### Phase 5: High-Capacity Streaming Backends & Clinical Digital Twins
*Objective*: Enable out-of-core streaming for massive models and advance closed-loop intraoperative clinical navigation.
* [ ] **Hierarchical & Streaming Dataset Backends (HDF5 / Zarr)**:
  * High-capacity out-of-core snapshot streaming and chunked storage for massive simulation runs ($> 10^6$ DOFs, continuous trajectory logging).
  * Parallel I/O integration with HDF5 and Zarr storage formats.
* [ ] **Online Hyper-Reduced Inverse Parameter Identification**:
  * Real-time in-vivo tissue stiffness calibration during active robotic tool contact.
* [ ] **Multi-Modal Intraoperative Sensor Fusion**:
  * Fusing sparse cortical surface optical markers with intraoperative ultrasound (iUS) and stereovision point clouds for enhanced deep-target tracking accuracy.

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
