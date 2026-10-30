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
