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
   - **Mortar contact** (`0802_beam_floor_mortar.py`): Mathematically exact in linear momentum conservation and patch tests, but exhibits extreme stiffness sensitivity and divergence when coupled with ultra-soft hyperelastic continua.

> [!IMPORTANT]
> **The User's Core Hypothesis**:
> *"My material properties are too extreme and too deformable."*
> In biological tissue and silicone modeling, Young's modulus is scaled to physiological values:
> $$E \approx 2{,}100\text{ to }21{,}000\text{ Pa} \quad (2.1\text{ to }21\text{ kPa}), \quad \nu \approx 0.35\text{ to }0.49$$
> With shear modulus $\mu \approx 700\text{ to }7{,}700\text{ Pa}$.
> In contrast, standard penalty values inherited from structural engineering ($k_p = 50{,}000\text{ to }500{,}000\text{ N/mm} = 5 \times 10^{10}\text{ to }5 \times 10^{11}\text{ g/s}^2$) are **$10^6$ to $10^8$ times stiffer than the underlying elements**.
> This massive stiffness mismatch acts like a rigid needle piercing soft gelatin: localized penalty forces invert surface tetrahedra ($\det \boldsymbol{F} \le 0$) before the bulk continuum can transmit the load and displace. In dynamic mode, inertia ($\frac{1}{\beta \Delta t^2} \boldsymbol{M}$) artificially stabilizes this; in static mode, the solver diverges immediately.

---

### 2.2 Phase 1: Proving Contact Correctness & Stabilizing Ultra-Soft Continua

#### Milestone 1.1: Classical 3D Hertzian Contact Benchmark (Analytical Ground Truth)
*Objective*: Disentangle contact formulation math from soft-tissue distortion by validating against exact closed-form analytical solutions on moderate-modulus materials ($E = 1.0\text{–}10.0\text{ MPa}$).
- [ ] **Spherical Indentation on Rigid Plane**:
  - Analytical Hertz solution for elastic sphere radius $R$, normal force $F_N$:
    $$\text{Contact Radius: } a = \left( \frac{3 F_N R}{4 E^*} \right)^{1/3}, \quad \text{Penetration: } \delta = \frac{a^2}{R}, \quad \text{Peak Pressure: } p_0 = \frac{3 F_N}{2 \pi a^2}$$
    where $\frac{1}{E^*} = \frac{1 - \nu_1^2}{E_1} + \frac{1 - \nu_2^2}{E_2}$.
  - Benchmark simulation: `examples/08_contact/0811_hertz_sphere_contact.py`.
  - Validate penetration depth $\delta$, contact area $A = \pi a^2$, and radial pressure profile $p(r) = p_0 \sqrt{1 - (r/a)^2}$ against analytical formulas ($< 3\%$ relative error).
- [ ] **Cylindrical Contact Benchmark**:
  - Two parallel cylinders / cylinder-on-flat with half-width $b = \sqrt{\frac{4 F_N R}{\pi L E^*}}$.
  - Verification test in `tests/test_contact_hertz_analytical.py`.

#### Milestone 1.2: Dimensionless Penalty Scaling Law & Automatic Stiffness Calculation
*Objective*: Eliminate arbitrary user-guessed penalty values and guarantee conditioning across 6 orders of magnitude of Young's modulus ($100\text{ MPa} \to 1\text{ kPa}$).
- [ ] **Dimensionless Contact Parameter $\kappa_p$**:
  $$\kappa_p \equiv \frac{k_p \cdot h_e}{E_{\text{bulk}}} \in [5, 50]$$
  where $h_e$ is the characteristic surface element edge length and $E_{\text{bulk}}$ is the effective Young's modulus.
- [ ] **Automatic Element-Adaptive Penalty Law**:
  $$k_{p, e} = \alpha_{\text{penalty}} \cdot \frac{K_{\text{bulk}} A_e}{V_e}$$
  where $K_{\text{bulk}}$ is bulk modulus, $A_e$ is facet area, $V_e$ is element volume, and $\alpha_{\text{penalty}} \approx 0.1\text{–}1.0$.
- [ ] Implement `compute_adaptive_penalty(domain, mesh)` in `psierra.boundary.contact`.
- [ ] Parametric modulus sweep test in `tests/test_contact_modulus_scaling.py`: verify that decreasing $E$ from $10\text{ MPa} \to 21\text{ kPa} \to 1\text{ kPa}$ with adaptive penalty maintains identical iteration counts ($< 6$ Newton iterations) and zero element inversions.

#### Milestone 1.3: Two-Body Large-Deformation Ironing Benchmark
*Objective*: Prove stability of deformable-to-deformable contact and Coulomb stick-slip friction under finite sliding without facet-edge snagging.
- [ ] **Benchmark Design**: A rounded deformable slider pressed vertically into a deformable slab and driven horizontally across multiple element widths.
- [ ] Deliver executable: `examples/08_contact/0812_ironing_large_strain.py`.
- [ ] Verify continuous horizontal contact force, energy dissipation through friction work $W_f = \int \boldsymbol{f}_T \cdot d\boldsymbol{u}_T$, and absence of edge snagging or line-search stalls.

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
