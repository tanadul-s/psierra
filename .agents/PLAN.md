# Project pSierra Development Plan & Roadmap (Loop 3: Robust Contact Mechanics & Next-Gen Multiphysics)

This document outlines the active engineering milestones, mathematical formulations, and architectural roadmap for Project `pSierra` in **Loop 3**.

> [!NOTE]
> **Loops 1 and 2 are fully completed, verified, and archived:**
> - 📂 **[`PLAN_ARCHIVE_LOOP1.md`](PLAN_ARCHIVE_LOOP1.md)**: Core Continuum Mechanics, Nonlinear Hyperelasticity, POD-Galerkin ROM, and Inverse FEM / Shape Sensing.
> - 📂 **[`PLAN_ARCHIVE_LOOP2.md`](PLAN_ARCHIVE_LOOP2.md)**: Biomechanical Continua (`Tet4Fbar`, Viscoelasticity), HDF5 Infrastructure (`SimulationWriter`), Dynamic Relaxation, Contact Regularization (3-Pillars, Mortar, Friction), Advanced Hyper-Reduction (DEIM, Contact-Aware ECSW), and Clinical Digital Twins (Ogden, HGO, Sensor Fusion, Interactive Inverse Burr Hole).

---

## 1. Executive Summary: Loops 1 & 2 Deliverables & Foundation

Project `pSierra` has established a comprehensive computational biomechanics and digital twin framework:

$$\text{Forward Non-Linear Continua} \longrightarrow \text{Contact Mechanics} \longrightarrow \text{Model Order Reduction (MOR)} \longrightarrow \text{Inverse FEM (iFEM)}$$

```
========================= LOOP 1: PRIMARY WORK LOOP (ARCHIVED) =========================
[ Phase 1: Linear FEM Core ]
  - Tet4 linear continua, Hookean elasticity, sparse assembly, SuperLU, Rayleigh damping.
[ Phase 2: Non-Linear Continua & Hyperelasticity ]
  - Tet10, Shell3 (DKT), Shell6, CR, TL, UL, Neo-Hookean, Yeoh, Mooney-Rivlin, Generalized-α.
[ Phase 3: Contact Mechanics (Foundations) ]
  - Rigid floor, Deformable-to-deformable mesh contact, Stanford Bunny self-contact.
[ Phase 4: Model Order Reduction (POD-Galerkin ROM & ECSW Hyper-Reduction) ]
  - Real-time dynamics (>20,000 Hz), surface sub-basis, sparse RID cubature (>900 Hz).
[ Phase 5: Inverse FEM (iFEM) & Parameter Identification ]
  - Variational shape sensing (>50,000 Hz), closed-form LLS & TRF identification, Sellier Aitken inversion.
========================================================================================
       │
       ▼
=================== LOOP 2: BIOMECHANICAL SPECIALIZATION & DIGITAL TWINS (ARCHIVED) =====
[ Phase 1: Biomechanical Continua, HDF5 Infrastructure & Dynamic Relaxation ]
  - Tet4Fbar (locking-free), Viscoelasticity (Prony series), Unified HDF5 SimulationWriter & Checkpoints,
    Dynamic Relaxation solver (kinetic damping), Hydrostatic CSF load (Chen et al.), Burr hole brain shift.
[ Phase 2: Advanced Contact Regularization & Non-Smooth Mechanics ]
  - 3-Pillar Regularization (Adaptive cutback, C¹ vertex pseudonormals, contact-aware line search),
    Segment-to-segment Mortar contact (Wohlmuth biorthogonal), Exact Coulomb stick-slip friction.
[ Phase 3: Advanced Non-Linear Hyper-Reduction (DEIM & Contact-Aware ECSW) ]
  - Out-of-core force streaming in HDF5, Greedy / Q-DEIM row selection, Contact-Aware ECSW (Method 2).
[ Phase 4: Clinical Digital Twins & Real-Time Intraoperative Navigation ]
  - Quasi-static surgical indentation with viscoelastic relaxation, Generalized Ogden hyperelasticity,
    Anisotropic Holzapfel-Gasser-Ogden (HGO) model, Online hyper-reduced parameter identification,
    Multi-modal intraoperative sensor fusion (iUS + optical), Interactive burr hole brain shift tracking.
[ Structural Harmonization ]
  - Unified 30-script canonical benchmark across 11 categories (PLAN_RESTRUCTURE.md), physical unit consistency.
========================================================================================
       │
       ▼
=================== LOOP 3: ROBUST CONTACT MECHANICS & ADVANCED MULTIPHYSICS ============
[ Phase 1: Mathematical Proof & Stabilization of Contact Mechanics ]
  - Analytical Hertzian contact validation (sphere/plane, cylinder/plane).
  - Two-body large-strain ironing benchmark.
  - Dimensionless penalty scaling law: κ_p = (k_p * h_e) / E_bulk ∈ [5, 50].
  - Stabilization of Deformable-to-Deformable, Self-Contact, and Mortar interfaces for ultra-soft continua.
[ Phase 2: Loop 2 Unfinished Enhancements & Grand Showcase ]
  - Pure CPU low-latency execution (<15 μs response, >1,000 Hz).
  - Grand Flagship Simulation: 1401_neurosurgical_digital_twin_complete.py.
[ Phase 3: Multiphysics Expansion: Coupled Biot Poroelasticity ]
  - Mixed u-p Taylor-Hood formulation (Tet10-Tet4) for brain parenchyma fluid egress and edema.
========================================================================================
```

---

## 2. Loop 3 Critical Focus: Contact Mechanics Overhaul & Mathematical Proof

### 2.1 The Contact Divergence Diagnosis & User Hypothesis

During Loop 2 testing on the reorganized benchmark suite (`examples/08_contact/`), significant numerical stability discrepancies were observed:

1. **What Works Well**:
   - `0801_beam_floor_penalty.py` in **dynamic mode with rigid floor** converges smoothly and resolves dynamic impact/rebound against the rigid floor plane.
2. **Where Severe Divergence / Failure Occurred**:
   - **Static contact regimes** (`0801` static, `0802` static): Highly sensitive to active-set changes, often stalling Newton iterations or requiring aggressive cutbacks.
   - **Deformable-to-deformable contact** (`0801` deformable floor, `0802` deformable floor): Slave nodes penetrations cause localized mesh distortions on compliant master facets.
   - **Self-contact** (`0804_bunny_self_contact.py`, `0805_cuboid_self_contact.py`): Inward folding of lobes causes adjacent facets to continually oscillate master/slave roles, leading to active-set chattering and element collapse.
   - **Mortar contact** (`0802_beam_floor_mortar.py`): Exact in the tested linear-momentum and constant-pressure patch properties, but its current frozen-geometry tangent is not the complete derivative of the recomputed finite-deformation force. It also exhibits extreme stiffness sensitivity and divergence when coupled with ultra-soft hyperelastic continua.

> [!IMPORTANT]
> **The User's Core Hypothesis**:
> *"My material properties are too extreme and too deformable."*
> In biological tissue and silicone modeling, Young's modulus is scaled to physiological values:
> $$E \approx 2{,}100\text{ to }21{,}000\text{ Pa} \quad (2.1\text{ to }21\text{ kPa}), \quad \nu \approx 0.35\text{ to }0.49$$
> With shear modulus $\mu \approx 700\text{ to }7{,}700\text{ Pa}$.
> In contrast, standard penalty values inherited from structural engineering ($k_p = 50{,}000\text{ to }500{,}000\text{ N/mm} = 5 \times 10^{10}\text{ to }5 \times 10^{11}\text{ g/s}^2$) are **$10^6$ to $10^8$ times stiffer than the underlying elements**.
> This massive stiffness mismatch acts like a rigid needle piercing soft gelatin: localized penalty forces invert surface tetrahedra ($\det \boldsymbol{F} \le 0$) before the bulk continuum can transmit the load and displace. In dynamic mode, inertia ($\frac{1}{\beta \Delta t^2} \boldsymbol{M}$) artificially stabilizes this; in static mode, the solver diverges immediately.

> [!CAUTION]
> **Mathematical Audit Conclusion (2026-11-16)**:
> The material-stiffness mismatch is a major conditioning problem, but it is not the only cause of contact failure. The planar frictionless rigid formulation and the unsmoothed node-to-triangle normal formulation are locally consistent on a fixed active feature branch. A general correctness claim is not yet justified because curved rigid tangents, smoothed-normal derivatives, positive-gap tolerance handling, frictional finite-sliding tangents, mortar geometry linearization, and several collision-sign/geometry cases remain incomplete or inconsistent. Existing unit tests prove useful local invariants; they do not replace analytical contact benchmarks or full force-energy-tangent verification.

---

### 2.2 Phase 1: Proving Contact Correctness & Stabilizing Ultra-Soft Continua

#### Implementation Strategy: Preserve the Public Shell, Replace the Mathematical Core

The contact overhaul will use a **staged internal replacement**, not a ground-up public rewrite and not indefinite patching of duplicated formulations.

- Preserve the public classes and solver contracts of `RigidContact`, `DeformableContact`, `SelfContact`, and `MortarContact`, including multiplier snapshots, rollback, state commitment, multibody offsets, and analysis integration.
- Introduce one internal geometry contract that returns candidate-aligned contact data:
  $$\mathcal{G}(\boldsymbol{q})=
    \left(g_N,\;\nabla g_N,\;\nabla^2 g_N,\;\boldsymbol{x}_{\mathrm{closest}},\;
    \boldsymbol{n},\;\text{feature ID}\right).$$
- Separate the pipeline into four explicit layers:
  $$\text{candidate search}\longrightarrow\text{contact geometry}
    \longrightarrow\text{constraint law}\longrightarrow\text{global assembly/state}.$$
- Centralize the normal constraint law and active-set semantics so force, tangent, energy, diagnostics, damping, and friction consume the same projected pressure:
  $$p_N=\max(0,\lambda_N-k_p g_N),$$
  with $\lambda_N=0$ for pure penalty contact.
- Port the already-verified unsmoothed node-to-triangle kernel first as the reference implementation. Migrate curved rigid contact, smoothed normals, friction, mortar, and self-contact incrementally behind the preserved API.
- Maintain old-versus-new comparison tests during migration. A legacy path may be removed only after its replacement passes the mathematical, conservation, solver, and benchmark gates below.
- Do not create a long-lived public `v2` contact API. A temporary internal feature flag is permitted only for verification and must be removed when migration is complete.

#### Milestone 1.0: Repair Mathematical Consistency Before Physical Calibration
*Objective*: Build the unified internal contact core, migrate each formulation safely, and remove known force-energy-tangent inconsistencies before tuning penalty parameters or interpreting benchmark convergence as physical validation.

- [x] **Stage A — Freeze Public Contracts and Establish Characterization Baselines**:
  - Document the public constructor, evaluation, multiplier, rollback, commit, and diagnostic behavior of all four contact classes.
  - Preserve the current 167-test focused contact baseline and add characterization tests for solver coupling, DOF offsets, state restoration, and accepted-step history.
  - Mark mathematically incorrect behavior as an expected legacy discrepancy rather than encoding it as the desired replacement result.

- [x] **Stage B — Introduce Canonical Internal Data Models**:
  - Add typed, candidate-aligned internal records for search results, differentiable contact geometry, projected normal-law state, friction state, and assembly maps.
  - Require every geometry provider to state its smooth-branch validity, units, orientation convention, and behavior at feature/active-set switches.
  - Make the canonical geometry record the sole source of gaps, normals, closest points, gradients, Hessians, and feature identity used downstream.

- [x] **Stage C — Centralize Constraint Laws and Assembly**:
  - Implement one normal penalty/augmented-Lagrangian law that returns pressure, potential, first derivative, activity, and multiplier update from the same inputs.
  - Implement shared one-body and two-body assembly operators so action-reaction, DOF mapping, and sparse scatter logic are not independently reimplemented by each contact class.
  - Keep collision search tolerances out of the constitutive constraint law.

- [x] **Stage D — Migrate in Increasing-Risk Order**:
  - [x] 1. Unsmoothed deformable node-to-triangle normal contact as the reference kernel (`DeformableContact`).
  - [x] 2. Planar rigid contact, followed by exact curved analytical obstacles (`RigidContact`).
  - [x] 3. Self-contact candidate filtering using the same triangle geometry and assembly core (`SelfContact`).
  - [x] 4. Energy-consistent smoothed-normal geometry (`SmoothedTriangleGeometryProvider`).
  - [x] 5. Frictionless mortar contact with complete geometry linearization (`MortarContact`).
  - [x] 6. Coulomb friction and damping as explicitly dissipative extensions.
  - Keep each migration independently releasable and revertible until its acceptance gates pass.

- [x] **Non-Adhesive Activation and Tolerance Semantics**:
  - Separate proximity/candidate detection from physical force activation.
  - For penalty contact, enforce $p_N = k_p\langle-g_N\rangle_+$ so a separated point with $0 < g_N \le \delta_{\text{search}}$ never receives an attractive normal force.
  - Make force, stiffness, potential energy, active-node reporting, and friction activation use the same projected pressure and active-set definition.
  - Add regression tests with positive gap and nonzero search tolerance for penalty and augmented-Lagrangian contact.

- [x] **Exact Curved Rigid-Obstacle Linearization**:
  - For a signed-distance obstacle, implement
    $$\boldsymbol{K}_c=-\frac{\partial\boldsymbol{f}_c}{\partial\boldsymbol{x}}
      =k_p\left(\boldsymbol{n}\otimes\boldsymbol{n}
      +g_N\nabla^2 g_N\right)$$
    on a fixed active branch, including sphere and cylinder curvature terms.
  - Verify $\boldsymbol{K}_c=-\partial\boldsymbol{f}_c/\partial\boldsymbol{u}$ by centered finite differences at multiple radii, orientations, and length scales.

- [x] **Energy-Consistent Smoothed-Normal Contact**:
  - For $g_N=\boldsymbol{r}\cdot\boldsymbol{n}(\boldsymbol{\xi})$, include both terms
    $$\frac{\partial g_N}{\partial\boldsymbol{q}}
      =\boldsymbol{W}^{T}\boldsymbol{n}
      +\boldsymbol{r}^{T}\frac{\partial\boldsymbol{n}}{\partial\boldsymbol{q}},$$
    and differentiate the vertex pseudonormals with respect to the current master geometry when they are recomputed from that geometry.
  - Require independent finite-difference agreement among potential energy, contact force, and tangent; symmetry alone is insufficient.

- [x] **Consistent Frictional Finite-Sliding Tangent**:
  - Retain the local Coulomb return mapping, but add derivatives of the tangent projector, evolving normal, closest-point/barycentric projection, normal pressure, and committed/trial history variables required by the selected algorithm.
  - Verify stick and slip branches independently, including facet crossing and curved-master sliding.
  - Check discrete dissipation: frictional work must be non-positive within numerical tolerance.

- [x] **Complete Mortar Geometry Linearization**:
  - Differentiate the active overlap map, mortar matrices $\boldsymbol{D}$ and $\boldsymbol{M}$, quadrature gaps, projected normals, and master/slave coupling used by the recomputed residual.
  - Preserve the already-tested partition of unity, patch transfer, and action-reaction balance while adding finite-difference residual-tangent verification under rotation, curvature, and non-conforming sliding.
  - Do not describe the tangent as “consistent” until this derivative test passes away from active-set topology changes.

- [x] **Collision Geometry and Sign Contract**:
  - Resolved `LEGACY-DISC-05`: `RigidCylinder(capped=False)` preserves infinite cylinder backward compatibility, while `capped=True` (and explicit subclasses `CappedCylinder` vs `InfiniteCylinder`) implements planar caps at $z = \pm h$ and corner rim projection.
  - Implemented `psierra.collision.manifold`: `ensure_consistent_surface_orientation` (Gauss divergence theorem volume check), `is_inside_closed_surface` (topological ray-winding inside-outside test via `vtkSelectEnclosedPoints`), and `compute_angle_weighted_pseudonormals` (Bærentzen & Aanæs 2005).
  - Implemented `psierra.collision.ccd`: Möller-Trumbore ray-triangle intersection, analytical ray-plane, ray-sphere, and ray-cylinder intersection, Time-of-Impact (TOI) calculator `compute_time_of_impact`, and `conservative_advancement_step` to prevent high-speed tunneling.
  - Verified across 18 adversarial tests in `tests/test_collision_geometry_and_sign.py`.

- [x] **Acceptance Gate for Mathematical Correctness**:
  - Enforced Conservative Triad Consistency: negative energy gradient matches contact force ($-\nabla_{\boldsymbol{q}} \Pi_N = \boldsymbol{f}_c$, $< 10^{-7}$ relative error) and negative force Jacobian matches tangent stiffness matrix ($-\frac{\partial \boldsymbol{f}_c}{\partial \boldsymbol{q}} = \boldsymbol{K}_c$, $< 10^{-5}$ relative error) across all smooth branches.
  - Enforced exact tangent stiffness symmetry ($\|\boldsymbol{K}_c - \boldsymbol{K}_c^T\|_{\infty} < 10^{-12}$) and discrete momentum balance: two-body and single-body linear momentum ($\sum \boldsymbol{f}_c = \mathbf{0}$) and angular momentum ($\sum \boldsymbol{x} \times \boldsymbol{f}_c = \mathbf{0}$) to $< 10^{-12}$, and rigid translation invariance ($\boldsymbol{K}_c \boldsymbol{u}_{\text{trans}} = \mathbf{0}$ to $< 10^{-8}$).
  - Enforced Scale-Aware Invariance across 8 orders of magnitude of Young's modulus (2.1 kPa brain parenchyma $\to$ 1.0 MPa silicone $\to$ 200 GPa steel).
  - Isolated non-smooth activation boundary ($g_N = 0$) and feature switching (face $\leftrightarrow$ edge $\leftrightarrow$ vertex) in continuous semismooth verification.
  - Verified across 17 formal gate tests in `tests/test_contact_acceptance_gate.py`.

- [x] **Migration and Legacy-Retirement Gate**:
  - Run old and new implementations on identical deterministic states and classify every difference:
    * `LEGACY-DISC-01` (positive gap attraction): Verified as **Fixed Defect** (force/energy strictly 0 on separated gaps).
    * `LEGACY-DISC-02` (curved obstacle curvature Hessian): Verified as **Fixed Defect** (complete linearization matches finite differences).
    * `LEGACY-DISC-03` (smoothed normal coordinate variation): Verified as **Fixed Defect** (matches finite difference gradient to $< 10^{-8}$).
    * `LEGACY-DISC-04` (mortar frozen tangent): Verified as **Fixed Defect** (complete linearization matches residual finite difference Jacobian).
    * `LEGACY-DISC-05` (cylinder height projection): Verified as **Intentional Contract Change** (`capped=False` preserves backward compatibility; `capped=True` handles caps and rim).
    * Non-regression equivalence: on flat un-smoothed penetrations, canonical and legacy match to floating-point precision ($< 10^{-14}$).
  - Required all public API, rollback, multibody, static, dynamic, and augmented-Lagrangian integration tests to pass without a downstream solver fork.
  - Formally codified legacy retirement schedule: legacy routines and toggles (`_use_legacy_kernel`, `energy_consistent_smoothing`, `complete_linearization`, `capped`) remain operational through Milestones 1.1–1.5 benchmarks (Hertz, large-strain ironing, mortar stabilization, self-contact stabilization) before physical deletion.
  - Verified across 16 formal gate tests in `tests/test_contact_migration_and_retirement_gate.py` (total contact suite expanded to 327 passing tests).

#### Milestone 1.1: Classical 3D Hertzian Contact Benchmark (Analytical Ground Truth)
*Objective*: Disentangle contact formulation math from soft-tissue distortion by validating against exact closed-form analytical solutions on moderate-modulus materials ($E = 1.0\text{–}10.0\text{ MPa}$).
- [x] **Spherical Indentation on Rigid Plane**:
  - Implemented analytical Hertz solution in `psierra.contact.hertz.HertzSphereContact`:
    $$\text{Contact Radius: } a = \left( \frac{3 F_N R}{4 E^*} \right)^{1/3}, \quad \text{Penetration: } \delta = \frac{a^2}{R}, \quad \text{Peak Pressure: } p_0 = \frac{3 F_N}{2 \pi a^2}$$
    where $\frac{1}{E^*} = \frac{1 - \nu_1^2}{E_1} + \frac{1 - \nu_2^2}{E_2}$.
  - Delivered executable benchmark simulation: `examples/08_contact/0811_hertz_sphere_contact.py` with ParaView VTU/PVD export and Rich comparative summary table.
  - Implemented vectorized surface nodal tributary area integration (`compute_nodal_tributary_areas`) and continuous contact pressure field reconstruction (`reconstruct_nodal_pressures`).
  - Validated penetration depth $\delta$, contact area $A = \pi a^2$, peak pressure $p_0$, and radial pressure profile $p(r) = p_0 \sqrt{1 - (r/a)^2}$ against analytical formulas ($< 5\%$ error on refined mesh).
- [x] **Cylindrical Contact Benchmark**:
  - Implemented analytical cylinder contact solution in `psierra.contact.hertz.HertzCylinderContact`:
    $$b = \sqrt{\frac{4 F_N R^*}{\pi L E^*}}, \quad p_0 = \frac{2 F_N}{\pi b L} = \sqrt{\frac{F_N E^*}{\pi R^* L}}$$
  - Author and verified in `tests/test_contact_hertz_analytical.py` (21 formal tests, 100% passing). Covered analytical reciprocity, normal force integration $\int p \, dA = F_N$, modulus scaling invariance, classical subsurface stress profile ($\tau_{\max} \approx 0.31 p_0$ at $z \approx 0.48 a$), and 3D FE spherical & cylindrical indentation against rigid plane.

#### Milestone 1.2: Dimensionless Penalty Scaling Law & Automatic Stiffness Calculation
*Objective*: Eliminate arbitrary user-guessed penalty values and guarantee conditioning across 6 orders of magnitude of Young's modulus ($100\text{ MPa} \to 1\text{ kPa}$).
- [x] **Dimensionless Contact Parameter $\kappa_p$**:
  $$\kappa_p \equiv \frac{k_p \cdot h_e}{E_{\text{bulk}}} \in [5, 50]$$
  where $h_e$ is the characteristic surface element edge length and $E_{\text{bulk}}$ is the effective Young's modulus.
- [x] **Automatic Element-Adaptive Penalty Law**:
  $$k_{p, e} = \alpha_{\text{penalty}} \cdot \frac{K_{\text{bulk}} A_e}{V_e}$$
  where $K_{\text{bulk}}$ is bulk modulus, $A_e$ is facet area, $V_e$ is element volume, and $\alpha_{\text{penalty}} \approx 0.1\text{–}1.0$.
- [x] Implemented `compute_adaptive_penalty(domain, mesh)` in `psierra.contact.scaling`, exported from `psierra.boundary.contact`, `psierra.boundary`, and `psierra.contact`. Integrated `k_penalty="auto"` across `RigidContact`, `DeformableContact`, `SelfContact`, and `MortarContact`.
- [x] Parametric modulus sweep test in `tests/test_contact_modulus_scaling.py` (32 tests, 100% passing): verified that decreasing $E$ from $10\text{ MPa} \to 21\text{ kPa} \to 1\text{ kPa}$ with adaptive penalty maintains identical iteration counts (`[3, 2]`, $< 6$ Newton iterations) and zero element inversions ($\det \boldsymbol{F} > 0$). Verified absence of the 200,000x conditioning degradation seen with un-adapted rigid penalties.

#### Milestone 1.3: Two-Body Large-Deformation Ironing Benchmark
*Objective*: Prove stability of deformable-to-deformable contact and Coulomb stick-slip friction under finite sliding without facet-edge snagging.
- [x] **Benchmark Design**: A rounded deformable slider ($R = 3.5\text{ mm}$, $E = 5.0\text{ MPa}$) pressed vertically ($\delta_z = 0.15\text{ mm}$) into a deformable slab ($30 \times 8 \times 5\text{ mm}$, $E = 1.0\text{ MPa}$) and driven horizontally across multiple element widths ($d_x = 2.0\text{--}4.0\text{ mm}$).
- [x] **Two-Body Relative Tangential Kinematics**:
  Formulated reference-displacement-based relative sliding $\boldsymbol{u}_{\text{rel}} = \boldsymbol{u}_s - \boldsymbol{u}_{m}^{\text{proj}}$ and anchor tracking in `DeformableContact` and `SelfContact`, resolving the previous bug where $s_{\text{coord}} - m_{\text{proj}}$ collapsed to the normal gap vector.
- [x] **Executable Benchmark**:
  Delivered `examples/08_contact/0812_ironing_large_strain.py` with CLI flags (`--friction`, `--indentation`, `--sliding-distance`, `--smooth / --no-smooth`, `--export-pvd`, `--quiet`), Rich telemetry table, and ParaView PVD/VTU time series export.
- [x] **Verification & Acceptance Criteria**:
  * Steady-state kinetic friction: $|F_T / F_N| \approx \mu \cdot \cos \theta_{\text{contact}} \in [0.70 \mu, 1.05 \mu]$ across finite sliding.
  * Thermodynamic dissipation: Cumulative friction work $W_f = \int \boldsymbol{f}_T \cdot d\boldsymbol{u}_T \le 0$ ($D_f = -W_f \ge 0$).
  * Spatial regularization: $C^1$ smoothed pseudonormals eliminate facet-boundary force chattering and line-search stalls.
  * Continuum admissibility: Zero element inversions ($\det \boldsymbol{F} > 0.01$) across both bodies throughout finite sliding.
- [x] **Verification Test Suite**:
  Authored `tests/test_contact_ironing_benchmark.py` (5 formal tests, 100% passing). Complete regression baseline passes at 385/385 tests in 102s.
- [x] Documented in `docs/12_contact_mechanics_contracts_and_baselines.md` (Section 21).

#### Milestone 1.4: Mortar Contact Stabilization for Ultra-Soft Continua
*Objective*: Bring Segment-to-Segment Mortar Contact to robust convergence on soft hyperelastic bodies.
- [ ] **Bounded Multiplier Augmentation**:
  - Cap multiplier increment per Newton step: $|\Delta \lambda_N| \le \lambda_{\max}^{\text{step}} = \gamma_{\text{cap}} \frac{E A_e}{h_e}$.
  - Prevents wild multiplier oscillations from over-correcting soft elements in a single step.
- [ ] **Dual Contact Normal Projection**:
  - Implement area-averaged mortar projection normals to prevent non-physical facet intersections on curved interfaces.
- [ ] Verify static and dynamic mortar convergence on silicone beam and flexible floor plate.

#### Milestone 1.5: Self-Contact Stabilization (Bunny & Folding Cuboid)
*Objective*: Fix divergence in `0804_bunny_self_contact.py` and `0805_cuboid_self_contact.py`.
- [ ] **Proximity Thickness Buffer ($\delta_{\text{buffer}} = 0.15 h_e$)**:
  - Active-set detection triggered before nodes penetrate through opposing facets.
- [ ] **Consistent Normal Orientation Filtering**:
  - Reject candidate self-contact facet pairs whose undeformed reference normals point in opposite directions or share topological edges.
- [ ] Verify Stanford Bunny ear collapse and Cuboid bending self-contact without divergence.

---

### 2.3 Phase 2: Unimplemented Loop 2 Carry-Overs & Grand Showcase

- [ ] **Pure CPU Low-Latency Execution**:
  - Optimize dense Cholesky / BLAS inner loops for online ROM and hyper-reduced models.
  - Profile and ensure $< 15\ \mu\text{s}$ response time ($> 1{,}000\text{ Hz}$) on CPU for intraoperative haptic/visual streaming.
- [ ] **Grand Flagship Clinical Simulation (`1401_neurosurgical_digital_twin_complete.py`)**:
  - Patient-specific MRI brain with segmented tumor, dural falx cerebri membrane (`Shell3`), CSF drainage loss, intraoperative tool indentation, multi-modal sensor fusion (optical + iUS), and real-time inverse shape sensing.

---

### 2.4 Phase 3: Multiphysics Expansion: Poroelasticity vs. Vibroacoustics

#### Strategic Assessment & Evaluation

| Evaluation Dimension | Candidate A: Coupled Biot Poroelasticity ($\boldsymbol{u}-p$) | Candidate B: Vibroacoustics / Structural Acoustics ($P-\boldsymbol{u}$) |
| :--- | :--- | :--- |
| **Physical Formulation** | Coupled solid-fluid mixture theory (Biot 1941):<br/>$\nabla \cdot \boldsymbol{\sigma} + \rho \boldsymbol{g} = \mathbf{0}$<br/>$\frac{1}{M} \dot{p} + \alpha \nabla \cdot \dot{\boldsymbol{u}} - \nabla \cdot (\frac{k}{\mu_f} \nabla p) = q$ | Structural elastodynamics coupled to acoustic wave equation:<br/>$\nabla^2 P - \frac{1}{c^2} \ddot{P} = 0 \text{ in } \Omega_f$<br/>$\frac{\partial P}{\partial \boldsymbol{n}} = -\rho_f \ddot{u}_n \text{ on } \Gamma_{\text{int}}$ |
| **Clinical Relevance to Brain & Soft Tissue** | **CRITICAL & FUNDAMENTAL**: Brain tissue is 80% fluid. Long-term brain shift, edema, hydrocephalus, and tumor mass effect are fluid-driven de-swelling phenomena governed by Darcy pore pressure dissipation. | **SPECIALIZED**: Relevant for ultrasound transducer radiation, ARFI shear wave elastography, or cranial bone acoustics, but less central to surgical deformation and tissue manipulation. |
| **Computational Nature** | Mixed $\boldsymbol{u}-p$ parabolic-hyperbolic system. Requires Taylor-Hood (Tet10-Tet4) or MINI stabilization to satisfy the LBB inf-sup condition. Fits naturally into existing implicit Newton solvers. | Hyperbolic wave equation with high frequency demands ($f > 20\text{ kHz}$). Requires extremely fine spatial discretization to resolve acoustic wavelengths ($\lambda = c/f$), leading to very large system sizes. |
| **Synergy with Project Sierra** | **Direct continuation** of CSF hydrostatic pressure & drainage modeling (Chen et al.). Enables true physical volume loss during craniotomy instead of empirical viscoelastic approximations. | Moderate synergy; primarily serves imaging physics rather than intraoperative navigation biomechanics. |
| **Recommendation** | **PRIMARY FOCUS FOR LOOP 3 MULTIPHYSICS** | **DEFERRED TO FUTURE SPECIALIZED EXTENSIONS** |

#### Poroelasticity Implementation Roadmap (Phase 3)
- [ ] **Taylor-Hood Mixed Finite Element (`Tet10Tet4Poroelastic`)**:
  - Quadratic displacement serendipity interpolation on 10 nodes ($\boldsymbol{u}$).
  - Linear pore pressure interpolation on 4 vertex nodes ($p$).
  - Satisfies the Ladyzhenskaya-Babuška-Brezzi (LBB) stability condition for the undrained/impermeable limit ($k \to 0$).
- [ ] **Poro-Visco-Hyperelastic Constitutive Integration**:
  - Effective stress principle: $\boldsymbol{\sigma} = \boldsymbol{\sigma}_{\text{eff}}(\boldsymbol{C}) - \alpha p \boldsymbol{I}$.
  - Coupling with Neo-Hookean, Ogden, and viscoelastic solid skeletons.
- [ ] **Coupled Solver Pipeline**:
  - Monolithic two-field Newton-Raphson solver for coupled block system:
    $$\begin{bmatrix} \boldsymbol{K}_{uu} & \boldsymbol{K}_{up} \\ \boldsymbol{K}_{pu} & \boldsymbol{K}_{pp} \end{bmatrix} \begin{bmatrix} \Delta \boldsymbol{u} \\ \Delta \boldsymbol{p} \end{bmatrix} = \begin{bmatrix} -\boldsymbol{R}_u \\ -\boldsymbol{R}_p \end{bmatrix}$$
- [ ] **Neurosurgical Brain Shift De-swelling Benchmark**:
  - Coupled simulation of craniotomy CSF drainage with progressive pore pressure drop and parenchymal consolidation.

---

## 3. Engineering Invariants & Quality Standards

Every agent and contributor to Project `pSierra` must adhere to these policies:

1. **Strict Type Annotations**: Complete type hinting on all functions and methods (`from __future__ import annotations`). Tests must declare `-> None`.
2. **Strict Physical Unit System**:
   - Length: $\text{mm}$
   - Mass: $\text{g}$
   - Time: $\text{s}$
   - Force: $\text{g}\cdot\text{mm/s}^2 = 10^{-6}\text{ N} = 1\ \mu\text{N}$
   - Stress / Pressure / Modulus: $\text{Pa} = \text{g}/(\text{mm}\cdot\text{s}^2)$
   - Density: $\text{g/mm}^3$ (e.g. brain tissue: $1040 \times 10^{-6}\text{ g/mm}^3$)
   - Gravity: $9810\text{ mm/s}^2$
3. **No Automatic Git Commits**: Never run `git commit` unless explicitly instructed by the user.
4. **Clickable Links**: Reference all files with valid GitHub-style file markdown links (`file:///...` or relative links).
5. **ParaView Visualization Standard**:
   - Displacements stored as 3D vector array under `<Attribute Name="Displacement">`.
   - Provide explicit terminal instructions for applying the **Warp By Vector** filter (`Scale = 1.0`) and rescaling data ranges.
6. **Zero-Regression Verification**:
   - Every turn must conclude with automated pytest verification across modified modules.
