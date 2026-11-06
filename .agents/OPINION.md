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
