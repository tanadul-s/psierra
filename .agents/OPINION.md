# OPINION.md: Collective Agent Memory

> **Purpose**: Fast memory for agents. Keep it short, plain, crisp, and high-signal. Read before coding; update after solving tricky problems.

---

## 1. Architecture & Workflow Rules (Agreed)
- **Code style**: Pure OOP. Python 3.12+. Strict types everywhere (`dataclass`, `pydantic`).
- **No loose functions**: Keep all logic in domain classes (`Tet4`, `SolidDomain`, etc.).
- **Math kernel exception**: Single-letter variables (`K`, `M`, `C`, `F`, `B`, `D`, `u`, `v`, `a`) are allowed inside inner numerical loops to match textbook equations.
- **4-Role Division**:
  - `manager`: Writes technical contract and API specs.
  - `developer`: Writes production code in `src/psierra/` **AND** runnable scripts in `example/`. Does not write tests.
  - `tester`: Writes adversarial unit and benchmark tests in `tests/`.
  - `teacher`: Writes LaTeX theory docs, algorithmic acceleration notes, and textbook citations.

---

## 2. FEM & Kinematics Consensus
- **3D Elements**: Linear `Tet4` first, then quadratic `Tet10` and hex elements (`Hex8`, `Hex20`).
- **Large strain**: Use Total Lagrangian (TL) for hyperelastic materials.
- **Hyperelasticity**: Start with Neo-Hookean, then Mooney-Rivlin, Ogden, Yeoh.
- **Symmetry**: Always exploit and preserve symmetry in $K$ and $M$ matrices.

---

## 3. Contact Mechanics (Before MOR)
- Implement contact mechanics **before** Model Order Reduction.
- Use Bounding Volume Hierarchy (BVH) for fast surface proximity searches.
- Formulations: Penalty method first for simplicity, then Augmented Lagrangian for exact non-penetration.
- Enables contact-induced stress and deformation to be captured in offline MOR snapshot matrices.

---

## 4. Model Order Reduction (MOR) Consensus
- **Goal**: Real-time simulation (>30–1000 Hz) for surgical digital twins (brain, aorta, liver).
- **Linear ROM**: Snapshot POD with SVD. Keep enough modes to reach 99.9% energy.
- **Hyper-reduction for non-linear FEM**:
  - Full-mesh integration ruins online speed in non-linear ROM.
  - Use **DEIM** (Discrete Empirical Interpolation Method) for fast point evaluation.
  - Use **ECSW** (Energy-Conserving Sampling and Weighting) for stable, energy-preserving element-level hyper-reduction.

---

## 5. Inverse FEM (iFEM) & Parameter Identification
- **Inverse shape sensing**: Reconstruct 3D volume deformations from sparse surface measurements (optical tracking, stereovision, ultrasound) via weighted least-squares without knowing applied forces.
- **Parameter estimation**: Adjoint-based optimization to calibrate patient-specific tissue stiffness from clinical force-indentation data.
- **Stress-free geometry**: Inverse elastostatics to estimate the zero-pressure reference geometry from clinical scans of pre-stressed organs.

---

## 6. Hardware & Performance Notes
- **Platform**: Ubuntu 24.04 LTS (x86_64), Python 3.12, managed with `uv`.
- **Sparse matrices**: Use `scipy.sparse.coo_matrix` for assembly, then convert to `csr_matrix` for solves.
- **Vectorization**: Avoid Python `for` loops over elements. Vectorize with NumPy broadcasting.
- **GPU**: CuPy (`cupy-cuda12x`) for large system solves and projection algebra.

---

## 7. Known Pitfalls & Cautions
- Inverted tetrahedral elements cause $\det(J) \le 0$; check Jacobian determinant before assembly.
- Unconstrained Dirichlet boundaries lead to singular stiffness matrices; always verify rigid body constraints.
- In nearly incompressible materials ($\nu \to 0.5$), standard displacement-based elements lock; will need mixed or selective reduced integration.
