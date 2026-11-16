# OPINION.md: Collective Agent Memory

> **Purpose**: Fast memory for agents. Keep it short, plain, crisp, and high-signal. Read before coding; update after solving tricky problems.

---

## 1. Architecture & Workflow Rules (Agreed)
- **Code style**: Pure OOP for all core physics entities (`Tet4`, `MaterialBase`, `SolidDomain`, `SolverBase`). Python 3.12+. Strict types everywhere (`dataclass`, `pydantic`).
- **Utility functions**: Stateless, widely reused helper functions in `psierra/utility/` (`tensor.py`, `assembly.py`, `indexing.py`) and `psierra/boundary/selector.py` remain as functional helpers for developer ergonomics across analysis classes.
- **Math kernel exception**: Single-letter variables (`K`, `M`, `C`, `F`, `B`, `D`, `u`, `v`, `a`) are allowed inside inner numerical loops to match textbook equations.
- **4-Role Division**:
  - `manager`: Writes technical contract and API specs.
  - `developer`: Writes production code in `src/psierra/` **AND** runnable scripts in `examples/`. Does not write tests.
  - `tester`: Writes adversarial unit and benchmark tests in `tests/`.
  - `teacher`: Writes LaTeX theory docs, algorithmic acceleration notes, and textbook citations.
- **Docstring-Document 1:1 Synchronization**: Developer embeds governing equations, physical definitions, and numerical notes in code docstrings. Teacher mirrors these exact equations, symbols, and terminology in documentation, linking directly to code methods. Every part must have matched explanations between docstring and document.
- **"Lego-Brick" Composability & Example Blueprint**: Pure plug-and-play composability (`Parameters` $\to$ `Material` $\to$ `Element` $\to$ `Domain` $\to$ `Solver`). Every script in `examples/` must follow the 6-stage lifecycle modeled after `0202_gravity_beam.py`. Examples serve as direct executable blueprints where users can simply substitute their mesh or adjust parameters to run their own simulations. Coupled and multiphysics domains must strictly retain this exact simplicity.

---

## 2. Test Suite Architecture & Validation Criteria
- **Structure**: Tiered testing in `tests/`:
  1. *Fast Analytical Unit Tests*: Procedural micro-meshes (1-element patch tests, unit cubes, rigid rotations) running in $< 2$ seconds.
  2. *Integration Benchmarks*: End-to-end tests using `data/beam1/mesh/beam1.vtk` verifying static deflection and dynamic damping.
- **Tolerances**:
  - Patch tests & rigid body invariance: machine precision ($< 10^{-10}$).
  - Tangent stiffness consistency ($\frac{\partial f_{int}}{\partial u}$ vs $K_{tan}$ via finite difference): $< 10^{-5}$.
  - Analytical cantilever deflection: within discretization error ($< 5\%$).

---

## 3. FEM & Kinematics Consensus
- **3D Elements**: Linear `Tet4` first. **Immediate Phase 2 priority is `Tet10` (quadratic tetrahedral)** to eliminate artificial shear locking on complex anatomical geometries (brain, soft tissue).
- **Large strain**: Use Total Lagrangian (TL) for hyperelastic materials.
- **Hyperelasticity**: Start with Neo-Hookean, then Mooney-Rivlin, Ogden, Yeoh.
- **Symmetry**: Always exploit and preserve symmetry in $K$ and $M$ matrices.

---

## 4. Contact Mechanics (Before MOR)
- Implement contact mechanics **before** Model Order Reduction.
- Use Bounding Volume Hierarchy (BVH) for fast surface proximity searches.
- Formulations: Penalty method first for simplicity, then Augmented Lagrangian for exact non-penetration.
- Enables contact-induced stress and deformation to be captured in offline MOR snapshot matrices.

### 4.1 Deformable-to-Deformable Dynamic Contact & $C^0$ Facet Chattering (Wriggers, Laursen, FEBio Consensus — Post-iFEM Phase 6+)
- **Strategic Placement**: To preserve momentum on the primary highway to Inverse FEM (iFEM in Phase 5), advanced non-smooth contact regularization is deferred to Phase 6+.
- **The $C^0$ Facet Discontinuity Phenomenon**:
  - In node-to-surface discretization on piecewise-planar triangular meshes, outward normal vectors are discontinuous across element edges ($\Delta \boldsymbol{n} = \boldsymbol{n}_B - \boldsymbol{n}_A$).
  - When a contacting slave node slides across an edge under high dynamic pressure $p = \max(0, \lambda - k_p g)$, the contact force vector jumps instantaneously by $\Delta \boldsymbol{f}_c = p \Delta \boldsymbol{n}$.
  - Under high-speed dynamic impact (e.g. `sim22` beam impact with 410 tonnes, $p \sim 500\text{ kN}$), this jump is hundreds of kilonewtons for sub-micron movements ($0.08\ \mu\text{m}$).
  - Standard Armijo line search assumes a $C^1$ differentiable merit function; the directional force jump spikes the candidate residual above the current state, causing line search rejection, trial step shrinkage ($\alpha \to 0$), and stall (`No admissible residual-reducing trial found`).
- **The 3-Pillar Solution Architecture**:
  1. **Temporal Regularization (Adaptive Time Step Cutback)** (*FEBio Theory Manual Sec. 7.1.3 & 8.1; Wriggers 2006 Sec. 11.2*):
     - When Newton equilibrium fails line-search acceptance or exceeds max iterations, automatically halve the physical time step $\Delta t \leftarrow \Delta t / 2$, restore the last converged endpoint state $(\boldsymbol{u}_n, \boldsymbol{v}_n, \boldsymbol{a}_n, \boldsymbol{\lambda}_n)$, and retry.
     - Strictly limits the physical sliding distance and momentum shock across facet boundaries per step. Once through the impact peak, $\Delta t$ ramps back up toward $\Delta t_{\max}$.
  2. **Spatial Regularization ($C^1$ Surface Normal Smoothing)** (*Wriggers 2006 Sec. 6.4; Laursen 2002 Sec. 4.3; FEBio Theory Manual Sec. 7.1.2*):
     - Replace raw piecewise-planar face normals with area-weighted or angle-weighted vertex pseudonormals interpolated across triangles: $\boldsymbol{n}(\boldsymbol{\xi}) = \frac{\sum \xi_i \bar{\boldsymbol{n}}_i}{\|\sum \xi_i \bar{\boldsymbol{n}}_i\|}$.
     - Eliminates the directional shock $\Delta \boldsymbol{n}$ as nodes cross element edges, restoring $C^1$ continuity to the contact force field and preserving quadratic Newton-Raphson convergence.
  3. **Algorithmic Regularization (Contact-Aware Line Search & Dual Rebound Relaxation)** (*Wriggers 2006 Sec. 10.3; Laursen 2002 Sec. 6.4*):
     - Distinguish between constitutive element inversion ($\det J \le 0$) and piecewise-smooth contact active-set switching.
     - In dynamic Augmented Lagrangian contact, when a node opens a positive gap during local elastic rebound ($g > 0$), relax the retained dual multiplier ($\lambda \leftarrow 0$ or $p \leftarrow 0$) so artificial dual holding forces do not act across facet boundaries.

---

## 5. Model Order Reduction (MOR) Consensus
- **Goal**: Real-time simulation (>30–1000 Hz) for surgical digital twins (brain, aorta, liver).
- **Linear ROM**: Snapshot POD with SVD. Keep enough modes to reach 99.9% energy.
- **Hyper-reduction for non-linear FEM**:
  - Full-mesh integration ruins online speed in non-linear ROM.
  - Use **DEIM** (Discrete Empirical Interpolation Method) for fast point evaluation.
  - Use **ECSW** (Energy-Conserving Sampling and Weighting) for stable, energy-preserving element-level hyper-reduction.

---

## 6. Inverse FEM (iFEM) & Parameter Identification
- **Inverse shape sensing**: Reconstruct 3D volume deformations from sparse surface measurements (optical tracking, stereovision, ultrasound) via weighted least-squares without knowing applied forces.
- **Parameter estimation**: Adjoint-based optimization to calibrate patient-specific tissue stiffness from clinical force-indentation data.
- **Stress-free geometry**: Inverse elastostatics to estimate the zero-pressure reference geometry from clinical scans of pre-stressed organs.

---

## 7. Hardware & Performance Notes
- **Platform**: Ubuntu 24.04 LTS (x86_64), Python 3.12, managed with `uv`.
- **Sparse matrices**: Use `scipy.sparse.coo_matrix` for assembly, then convert to `csr_matrix` for solves.
- **Vectorization**: Avoid Python `for` loops over elements. Vectorize with NumPy broadcasting.
- **GPU**: CuPy (`cupy-cuda12x`) for large system solves and projection algebra.

---

## 8. Known Pitfalls & Cautions
- Inverted tetrahedral elements cause $\det(J) \le 0$; check Jacobian determinant before assembly.
- Unconstrained Dirichlet boundaries lead to singular stiffness matrices; always verify rigid body constraints.
- In nearly incompressible materials ($\nu \to 0.5$), standard displacement-based elements lock; will need mixed or selective reduced integration.
- **Canonical Voigt Ordering**: `psierra` explicitly uses `(xx, yy, zz, xy, yz, zx)` via `VOIGT_MAP_3D`. Do not confuse with Abaqus/ANSYS `(xx, yy, zz, yz, xz, xy)` ordering.
- **Floating-point Rigid Modes**: For high stiffness ($E \sim 10^{11}$ Pa), unconstrained zero eigenvalues from `eigvalsh(K)` evaluate to $\sim 10^{-5}$, which is exact IEEE 754 float64 machine epsilon ($\sim 10^{-16}$) relative to $\|K\|$. Always check relative to $\|K\|$.
- **Reaction Equilibrium**: Support reactions $R_p = K_{pf} u_f + K_{pp} u_p - F_p$ balance applied loads: $\sum R_p + \sum F_{ext} = 0$.
- **$C^0$ Contact Facet Chattering & Ridge Snapping**: In node-to-surface contact on discretized 3D meshes, crossing the edge between adjacent non-coplanar triangles causes an instantaneous jump $\Delta \boldsymbol{n} = \boldsymbol{n}_B - \boldsymbol{n}_A$ in the outward normal. Under high contact pressure $p = \max(0, \lambda - k_p g)$, this induces an unphysical force shock $\Delta \boldsymbol{f}_c = p \Delta \boldsymbol{n}$ (often hundreds of kN for sub-micron steps), causing Armijo line search rejection and Newton divergence. Must be resolved via adaptive time cutback, $C^1$ normal smoothing, or contact-aware active-set fallback.
- **Spurious Dual Forces from Frozen Uzawa Multipliers During Rebound**: In dynamic Augmented Lagrangian contact, loose AL tolerances (`al_gap_tol`, `al_tol`) permit early step acceptance with significant penetration, freezing large dual multipliers $\lambda$. When local dynamic rebound opens a positive gap ($g > 0$), $p = \max(0, \lambda - k_p g)$ maintains artificial pushing/pulling forces across facet boundaries until the outer Uzawa loop relaxes them.
- **Armijo Line Search Stalls on Non-Smooth Contact Interfaces**: Standard Armijo line search assumes a smooth $C^1$ energy landscape. Contact active-set switches create localized non-smooth ridges in the residual norm. Do not allow pure line search shrinkage to choke the time step to zero; use temporal substepping (cutback) or active-set projection fallback.

---

## 9. Loop 3 Contact Mechanics Overhaul & Milestone 1.0 Baselines (Agreed)
- **Staged Internal Replacement Strategy**: Public classes (`RigidContact`, `DeformableContact`, `SelfContact`, `MortarContact`) remain stable external interfaces. Internal mathematics are replaced in four clear layers: candidate search $\to$ contact geometry $\to$ constraint law $\to$ global assembly.
- **Stage A Characterization Baselines**:
  - Baseline preserved: all 167 existing contact tests pass without modification.
  - 22 new characterization baseline tests authored in `tests/test_contact_characterization_baseline.py` (total 189 passing contact tests).
  - Public contracts, lifecycle transitions, degree-of-freedom offsets, and solver coupling fully documented in `docs/12_contact_mechanics_contracts_and_baselines.md`.
- **Characterized Legacy Discrepancies (Do Not Bless as Ground Truth)**:
  - `LEGACY-DISC-01`: Positive-gap penalty force attraction under nonzero search tolerance ($f_N = -k_p g < 0$ when $0 < g \le \text{tolerance}$). Target in Stage C: non-adhesive law $p_N = k_p \langle -g_N \rangle_+$.
  - `LEGACY-DISC-02`: Curved obstacle tangent omission of curvature Hessian ($g_N \nabla^2 g_N$). Target in Stage D: complete analytical second derivatives.
  - `LEGACY-DISC-03`: Smoothed normals omitting $\boldsymbol{r}^T \nabla \boldsymbol{n}$ and vertex pseudonormal variation $\partial \boldsymbol{n}_v / \partial \boldsymbol{q}$. Target in Stage D: full energy-consistent variation.
  - `LEGACY-DISC-04`: Mortar tangent frozen geometry ($d\boldsymbol{D}/d\boldsymbol{u} = d\boldsymbol{M}/d\boldsymbol{u} = \mathbf{0}$). Target in Stage D: complete interface linearization.
  - `LEGACY-DISC-05`: `RigidCylinder` distance query ignoring `height` bounding caps. Target in Stage D: explicit capped or infinite collision geometry contract.
- **Stage B Canonical Internal Data Models**:
  - Introduced typed internal records in `psierra.contact.data`: `ContactCandidate`, `DifferentiableContactGeometry`, `NormalLawState`, `FrictionalConstraintState`, and `ContactAssemblyMap`.
  - Canonical `DifferentiableContactGeometry` enforces the exact tuple $\mathcal{G}(\boldsymbol{q}) = (g_N, \nabla g_N, \nabla^2 g_N, \boldsymbol{x}_{\text{closest}}, \boldsymbol{n}, \text{feature ID})$ as the sole downstream source of truth for gaps, normals, gradients, Hessians, and features.
  - Implemented analytical `ContactGeometryProvider` classes in `psierra.contact.geometry` (`PlanarGeometryProvider`, `SphereGeometryProvider`, `CylinderGeometryProvider`, `TriangleGeometryProvider`), with exact curvature and circumferential Hessians verified by centered finite differences.
  - Full suite expanded to 199 tests (167 focused baseline + 22 Stage A characterization + 10 Stage B data models), passing with 0 regressions.
- **Stage C Centralized Constraint Laws & Unified Assembly**:
  - Authored `NormalConstraintLaw` in `psierra.contact.law`: implements a single source of truth for penalty and augmented Lagrangian normal contact laws with strict non-adhesive activation ($p_N = 0$ for $g_N > 0$ when $\lambda_N = 0$), eliminating `LEGACY-DISC-01`.
  - Unified potential energy $\Pi_N = \frac{p_N^2 - \lambda_N^2}{2 k_p}$, pressure derivative $dp_N/dg_N = -k_p \cdot \mathbb{I}(\text{active})$, and exact dual multiplier update $\lambda_{N, \text{new}} = p_N = \max(0, \lambda_N - k_p g_N)$.
  - Authored `ContactAssembler` in `psierra.contact.assembly`: centralizes one-body (rigid-to-deformable), two-body (deformable-to-deformable), and self-contact assembly. Enforces action-reaction balance ($\sum \boldsymbol{f}_c = \mathbf{0}$), arbitrary multibody DOF offsets, vector scattering via `np.add.at`, and CSR sparse matrix construction via COO triplets with automatic duplicate summation.
  - Authored adversarial test suite `tests/test_contact_centralized_law_and_assembly.py` (8 tests). Full suite expanded to 207 contact tests, all passing with zero regressions.
  - Documented Section 10 in `docs/12_contact_mechanics_contracts_and_baselines.md`.
- **Stage D Migrations 1 & 2 (DeformableContact & RigidContact Staged Migration)**:
  - Migrated `DeformableContact` to consume canonical `TriangleGeometryProvider`, `NormalConstraintLaw`, and `ContactAssembler` while strictly preserving public class contracts and solver interfaces. Verified action-reaction balance, bitwise parity with legacy on penetrating stencils, and centered finite-difference consistency in `tests/test_contact_migration_deformable.py` (8 tests).
  - Migrated `RigidContact` to consume canonical providers (`PlanarGeometryProvider`, `SphereGeometryProvider`, `CylinderGeometryProvider`), `NormalConstraintLaw`, and `ContactAssembler.build_one_body_map`.
  - Resolved `LEGACY-DISC-01`: strict non-adhesive activation ($p_N = 0$ for $g_N > 0$ when $\lambda_N = 0$) prevents unphysical attractive suction under nonzero search tolerances.
  - Resolved `LEGACY-DISC-02`: exact curved obstacle linearization incorporates the complete curvature term $\boldsymbol{K}_c = (k_p + a_1 c_p)(\boldsymbol{n} \otimes \boldsymbol{n}) - p_N \nabla^2 g_N = (k_p + a_1 c_p)(\boldsymbol{n} \otimes \boldsymbol{n}) + k_p g_N \nabla^2 g_N$, verified by centered finite differences across sphere and cylinder obstacles in `tests/test_contact_migration_rigid.py` (18 tests).
  - Full suite expanded to 255 contact tests passing with 0 regressions. Documented Sections 11 & 12 in `docs/12_contact_mechanics_contracts_and_baselines.md`.
- **Stage D Migration 3 (SelfContact Staged Migration)**:
  - Migrated `SelfContact` in `src/psierra/boundary/contact.py` to consume canonical `TriangleGeometryProvider`, `NormalConstraintLaw`, and `ContactAssembler.build_self_contact_map` behind its preserved public contract.
  - Candidate selection retains topological BFS exclusion graphs (`exclusion_rings`), slave/master normal opposition filtering (`normal_threshold`), axis-aligned bounding box culling, and contact thickness distance cutoff.
  - Verified single-body linear momentum conservation ($\sum \boldsymbol{f}_c = \mathbf{0}$) and angular momentum conservation ($\sum \boldsymbol{x} \times \boldsymbol{f}_c = \mathbf{0}$) to machine precision ($< 10^{-12}$).
  - Verified exact centered finite difference tangent consistency $\boldsymbol{K}_c = -\frac{\partial \boldsymbol{f}_c}{\partial \boldsymbol{u}}$, global degree-of-freedom offsets (`dof_offset`), velocity damping, and regularized Coulomb stick-slip friction coupling.
  - Authored `tests/test_contact_migration_self.py` (11 tests). Full contact suite expanded to **266 tests, 100% passing in 100.5s**. Documented Section 13 in `docs/12_contact_mechanics_contracts_and_baselines.md`.
- **Stage D Migration 4 (Energy-Consistent Smoothed-Normal Contact)**:
  - Authored `SmoothedTriangleGeometryProvider` and `evaluate_smoothed_triangle_geometry` in `src/psierra/contact/geometry.py`.
  - Resolved `LEGACY-DISC-03`: incorporates the complete master shape and normal variation $\nabla_{\boldsymbol{q}} g_N = \boldsymbol{W}^T \boldsymbol{n} - \sum (\boldsymbol{x}_{mi} \cdot \boldsymbol{n}) \frac{\partial \xi_i}{\partial \boldsymbol{q}} + (\text{dnormal})^T \boldsymbol{r}$ and exact consistent Hessian $\nabla^2 g_N$.
  - Eliminated the ~15% directional gradient error of the legacy smoothed normal kernel, achieving $< 10^{-8}$ agreement with centered finite differences of the contact potential.
  - Preserves exact discrete linear momentum conservation ($\sum \boldsymbol{f}_c = \mathbf{0}$) to machine precision ($< 10^{-12}$), exact tangent symmetry, and rigid translation invariance ($\boldsymbol{K}_c \boldsymbol{u}_{\text{trans}} = \mathbf{0}$).
  - Supported as opt-in `energy_consistent_smoothing: bool = False` in `DeformableContact` and `SelfContact`. Authored `tests/test_contact_migration_smoothed.py` (6 tests). Documented Section 14 in `docs/12_contact_mechanics_contracts_and_baselines.md`.
- **Stage D Migration 5 (Complete Mortar Geometry Linearization)**:
  - Migrated `MortarContact` in `src/psierra/boundary/mortar.py` to consume canonical `NormalConstraintLaw`.
  - Resolved `LEGACY-DISC-04`: implemented complete interface geometry linearization via `MortarContact(complete_linearization=True)`, differentiating recomputed mortar forces $\boldsymbol{K}_c = -\frac{\partial \boldsymbol{f}_c}{\partial \boldsymbol{u}}$ w.r.t active interface DOFs, including domain clipping, dual shape matrix ($\boldsymbol{D}, \boldsymbol{M}$), and normal vector variations under finite sliding and curvature.
  - Validated against independent centered finite differences of mortar force residual ($< 10^{-4}$ error).
  - Preserves exact action-reaction linear momentum balance ($\sum \boldsymbol{f}_s + \sum \boldsymbol{f}_m = \mathbf{0}$) to $< 10^{-12}$, exact symmetry, and rigid translation invariance.
  - Authored `tests/test_contact_migration_mortar.py` (4 tests). Full contact suite expanded to **276 tests, 100% passing in 102.6s**. Documented Section 15 in `docs/12_contact_mechanics_contracts_and_baselines.md`.
- **Collision Geometry, Manifold Sign Contract & Continuous Collision Detection (CCD)**:
  - Resolved `LEGACY-DISC-05`: `RigidCylinder(capped=False)` preserves infinite cylinder radial projection for full backward compatibility, while `RigidCylinder(capped=True)` (and explicit classes `CappedCylinder` vs `InfiniteCylinder`) implements planar caps at $z = \pm h$ and circular corner rim projection.
  - Implemented `psierra.collision.manifold`:
    - `ensure_consistent_surface_orientation`: enforces consistent triangular facet winding and verifies outward orientation on closed manifolds using the Gauss divergence theorem ($\int_{\partial \Omega} \boldsymbol{x} \cdot \boldsymbol{n} \, dA = 3V > 0$).
    - `is_inside_closed_surface`: topological ray-winding inside-outside test via `vtkSelectEnclosedPoints`, immune to concave exterior pockets (e.g. L-shaped non-convex geometries).
    - `compute_angle_weighted_pseudonormals`: Bærentzen & Aanæs (2005) angle-weighted vertex and edge pseudonormals.
    - `query_closed_surface_signed_distance`: signed distance queries ($g < 0$ inside, $g \ge 0$ outside) with outward normals.
  - Implemented `psierra.collision.ccd`:
    - `ray_triangle_intersection`: Möller-Trumbore ray-triangle intersection returning exact TOI $t \in [0, 1]$ and barycentrics $(u, v)$.
    - Analytical `ray_plane_intersection`, `ray_sphere_intersection`, and `ray_cylinder_intersection`.
    - `compute_time_of_impact`: finds earliest impact $t_{\min} \in [0, 1]$ across moving slave nodes.
    - `conservative_advancement_step`: advances slave nodes safely by $\alpha = \min(1.0, \text{safety\_factor} \times t_{\min})$ to completely eliminate high-speed tunneling across thin plates, spheres, cylinders, and triangular meshes.
  - Authored `tests/test_collision_geometry_and_sign.py` (18 adversarial tests). Full suite expanded to **294 tests, 100% passing**. Documented Section 16 in `docs/12_contact_mechanics_contracts_and_baselines.md`.
- **Acceptance Gate for Mathematical Correctness (Milestone 1.0 Gate)**:
  - Validated the 4 formal mathematical gates across all four contact classes (`RigidContact`, `DeformableContact`, `SelfContact`, and `MortarContact`).
  - Gate A (Conservative Triad): negative energy gradient matches contact force ($-\nabla_{\boldsymbol{q}} \Pi_N = \boldsymbol{f}_c$, $< 10^{-7}$) and negative force Jacobian matches tangent stiffness matrix ($-\frac{\partial \boldsymbol{f}_c}{\partial \boldsymbol{q}} = \boldsymbol{K}_c$, $< 10^{-5}$) on every smooth branch; exact tangent symmetry $\boldsymbol{K}_c = \boldsymbol{K}_c^T$ to $< 10^{-12}$.
  - Gate B (Momentum Conservation & Invariants): action-reaction linear momentum ($\sum \boldsymbol{f}_c = \mathbf{0}$) and angular momentum ($\sum \boldsymbol{x} \times \boldsymbol{f}_c = \mathbf{0}$) conserved to $< 10^{-12}$; rigid translation invariance $\boldsymbol{K}_c \boldsymbol{u}_{\text{trans}} = \mathbf{0}$ to $< 10^{-8}$.
  - Gate C (Scale-Aware Invariance): relative errors in energy and tangent are scale-invariant across 8 orders of magnitude of Young's modulus (2.1 kPa $\to$ 200 GPa).
  - Gate D (Semismooth Boundary & Feature Switching): continuous force across $g_N = 0$ ($\lim_{g_N \to 0} \boldsymbol{f}_c = \mathbf{0}$) and smooth energy/force across facet feature boundaries (face $\leftrightarrow$ edge $\leftrightarrow$ vertex).
  - Authored `tests/test_contact_acceptance_gate.py` (17 tests). Full contact and collision test suite expanded to **311 tests, 100% passing**. Documented Section 17 in `docs/12_contact_mechanics_contracts_and_baselines.md`.
- **Migration and Legacy-Retirement Gate (Final Milestone 1.0 Gate)**:
  - Validated deterministic state comparison and discrepancy classification matrix:
    * `LEGACY-DISC-01`: Classified as `FIXED_DEFECT` (positive-gap attraction removed in canonical; force and energy strictly zero on opening gaps).
    * `LEGACY-DISC-02`: Classified as `FIXED_DEFECT` (obstacle curvature Hessian $g_N \nabla^2 g_N$ added in canonical; matches finite-difference Hessian).
    * `LEGACY-DISC-03`: Classified as `FIXED_DEFECT` (normal field gradient $\boldsymbol{r}^T \nabla \boldsymbol{n}$ added in canonical; matches finite-difference gradient to $< 10^{-8}$).
    * `LEGACY-DISC-04`: Classified as `FIXED_DEFECT` (frozen mortar tangent resolved in canonical; complete interface linearization matches residual finite difference Jacobian).
    * `LEGACY-DISC-05`: Classified as `INTENTIONAL_CONTRACT_CHANGE` (`capped=False` / `InfiniteCylinder` preserves backward compatibility; `capped=True` / `CappedCylinder` provides planar caps and circular rim).
    * Non-regression equivalence: on flat un-smoothed penetrations, canonical and legacy match to floating-point precision ($< 10^{-14}$).
  - Enforced public API stability, constructor typing, and default parameters across all four contact classes inheriting from `ContactMultiplierState`.
  - Verified multi-body solver coupling with `NonlinearStaticAnalysis` and `ContactAugmentation` without any downstream solver forks or private branches.
  - Verified transactional state rollback: dual multipliers and Coulomb stick-slip friction snapshots restore completely upon trial step rejection; history commits only on acceptance.
  - Formally codified legacy retirement policy: legacy pathways remain operational throughout Milestones 1.1–1.5 benchmarks before physical code deletion.
  - Authored `tests/test_contact_migration_and_retirement_gate.py` (16 tests). Full contact, collision, acceptance, and retirement test suite expanded to **327 tests, 100% passing**. Documented Section 18 in `docs/12_contact_mechanics_contracts_and_baselines.md`. Milestone 1.0 is officially 100% complete!
- **Milestone 1.1: Classical 3D Hertzian Contact Benchmark (Analytical Ground Truth)**:
  - Authored `src/psierra/contact/hertz.py` providing exact closed-form analytical solutions:
    * `HertzSphereContact`: contact radius $a = (3 F_N R^* / 4 E^*)^{1/3} = \sqrt{R^* \delta}$, penetration $\delta = a^2 / R^*$, peak pressure $p_0 = \frac{3 F_N}{2 \pi a^2}$, radial pressure profile $p(r) = p_0 \sqrt{\max(0, 1 - (r/a)^2)}$, and full subsurface stress field ($\sigma_z, \sigma_r, \tau_{\max}$).
    * `HertzCylinderContact`: contact half-width $b = \sqrt{\frac{4 F_N R^*}{\pi L E^*}}$, peak pressure $p_0 = \frac{2 F_N}{\pi b L}$, transverse pressure distribution $p(x) = p_0 \sqrt{\max(0, 1 - (x/b)^2)}$.
    * Vectorized surface nodal tributary area calculation `compute_nodal_tributary_areas` ensuring exact area conservation $\sum A_i = A_{\text{surface}}$ to $< 10^{-14}$.
    * Continuous normal contact pressure field reconstruction `reconstruct_nodal_pressures` ($p_i = f_{c, z, i} / A_i$).
  - Delivered executable simulation benchmark: `examples/08_contact/0811_hertz_sphere_contact.py`, featuring ParaView VTU/PVD export and formatted Rich comparative summary tables.
  - Authored comprehensive verification test suite: `tests/test_contact_hertz_analytical.py` (21 tests, 100% passing).
  - Validated exact mathematical reciprocity ($F_N(\delta(F_N)) = F_N$), normal force integral conservation ($\int p \, dA = F_N$ to $< 10^{-10}$), Boussinesq two-body effective modulus/curvature, classical subsurface peak shear stress location ($z \approx 0.48 a, \tau_{\max} \approx 0.31 p_0$), 3D FE spherical indentation and cylindrical compression against analytical formulas, and modulus scaling invariance across $E = 1.0\text{–}10.0\text{ MPa}$.
  - Full contact and collision test suite expanded to **348 tests, 100% passing**. Documented Section 19 in `docs/12_contact_mechanics_contracts_and_baselines.md`. Milestone 1.1 complete!
- **Milestone 1.2: Dimensionless Penalty Scaling Law & Automatic Stiffness Calculation (Agreed)**:
  - Authored `src/psierra/contact/scaling.py` providing scale-invariant penalty parameter calculation and automatic element-adaptive contact stiffness:
    * Characteristic mesh metrics extraction `compute_characteristic_mesh_metrics` ($h_e, \bar{A}_e, \bar{V}_e, A_e, V_e$).
    * Universal material stiffness extraction `extract_material_stiffness` across all continuum constitutive laws (`NeoHookean`, `Hookean`, `Yeoh`, `MooneyRivlin`, `Ogden`, `HGO`, numeric duck-types, and fallbacks).
    * Automatic element-adaptive penalty law: $k_{p, e} = \alpha_{\text{penalty}} \frac{K_{\text{bulk}} A_e}{V_e}$ (distributed) and $k_{p, \text{nodal}, e} = \alpha_{\text{penalty}} \frac{K_{\text{bulk}} A_e^2}{V_e}$ (nodal), with scalar $k_{p, \text{nodal}} = \alpha_{\text{penalty}} \frac{K_{\text{bulk}} \bar{A}_e^2}{\bar{V}_e}$.
    * Dimensionless parameter evaluation: $\kappa_p \equiv \frac{k_{p, \text{dist}} \cdot h_e}{E_{\text{bulk}}} = \frac{k_{p, \text{nodal}} \cdot h_e}{E_{\text{bulk}} \cdot A_e} \in [5, 50]$.
  - Integrated `k_penalty="auto"` across all four public contact classes (`RigidContact`, `DeformableContact`, `SelfContact`, and `MortarContact`).
  - Authored comprehensive verification test suite: `tests/test_contact_modulus_scaling.py` (32 tests, 100% passing):
    * Linear scaling verification ($k_p \propto E$, $k_p \propto K_{\text{bulk}}$) across 4 orders of magnitude.
    * Acceptance Gate: Parametric Modulus Sweep ($E = 10\text{ MPa} \to 21\text{ kPa} \to 1\text{ kPa}$) in `NonlinearStaticAnalysis` achieves identical iteration counts (`[3, 2]`, $< 6$ Newton iterations per step) and zero element inversions ($\det \boldsymbol{F} > 0$).
    * Conditioning robustness: $\|\boldsymbol{K}_c\| / \|\boldsymbol{K}_{\text{dom}}\|$ invariant across moduli, preventing the 200,000x conditioning degradation and element inversion seen with un-adapted rigid penalties ($10^5\text{ N/mm}$).
  - Full contact and collision test suite expanded to **380 tests, 100% passing in 107.55s**. Documented Section 20 in `docs/12_contact_mechanics_contracts_and_baselines.md`. Milestone 1.2 complete!
