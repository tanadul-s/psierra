# Project pSierra: A Python Implementation of FEM Structural Analysis

This document serves as the permanent memory, system rules, and multi-agent coordination contract for all agents operating in this repository.

---

## 1. Project Mission & Strategic Goals

`pSierra` is a high-performance 3D Finite Element Method (FEM) simulation engine written in modern Python. The engine is designed for general structural mechanics with a primary focus on biomechanical simulation for medical digital twins and surgical navigation (e.g., soft-tissue deformation of the brain, aortic pulsatile-driven abdominal kinematics).

### Core Capabilities & Technical Scope
1. **Analysis Domains**:
   - **Structural Mechanics**: Linear and non-linear statics, implicit and explicit dynamics.
   - **Multiphysics Simulation**: Vibro-acoustics, Fluid-Structure Interaction (FSI).
2. **Kinematic Formulations**:
   - Small-strain linear elasticity.
   - Large-deformation / geometric non-linearities via **Total Lagrangian (TL)** and **Updated Lagrangian (UL)** formulations.
3. **Constitutive Material Models**:
   - **Linear Elastic**: Isotropic Hookean materials.
   - **Hyperelastic**: Neo-Hookean, Mooney-Rivlin, Ogden, Yeoh, etc.
   - **Time-Dependent & Porous Media**: Viscoelastic, Poroelastic.
   - **Coupled Hybrids**: Hyper-visco-poroelastic formulations.
4. **Finite Element Formulations**:
   - 3D Isoparametric Solid Elements: Tetrahedral (`Tet4`, `Tet10`) and Hexahedral (`Hex8`, `Hex20`), supporting both linear and quadratic interpolation.
5. **Contact Mechanics & Interactive Collision Handling (Before MOR)**:
   - Surface boundary facet extraction and Bounding Volume Hierarchy (BVH) / AABB proximity searches.
   - Penalty formulations, augmented Lagrangian multiplier methods, and regularized Coulomb friction.
   - Compliant tool-tissue contact and surgical interaction (virtual palpation, clamping, indentation).
6. **Model Order Reduction (MOR) & Hyper-Reduction (Real-Time Engine)**:
   - **Real-Time Surgical Navigation & Digital Twin Drivers**: Accelerate high-degree-of-freedom non-linear and contact FEM to interactive and haptic update rates (>30–1000 Hz) while preserving physical fidelity.
   - **Linear MOR**: Proper Orthogonal Decomposition (POD / Snapshot PCA), Modal Superposition / Truncation, Guyan Reduction, and Craig-Bampton component mode synthesis.
   - **Nonlinear MOR & Hyper-Reduction**:
     - **DEIM (Discrete Empirical Interpolation Method)** / Q-DEIM for point-wise nonlinear internal force and tangent evaluation without full-mesh quadrature.
     - **ECSW (Energy-Conserving Sampling and Weighting)** for stable, energy-preserving element-level reduced mesh sampling and weighting in nonlinear solid mechanics.
     - Nonlinear manifold and physics-informed reduced basis projection.
   - **Offline / Online Decoupling**:
     - *Offline Phase*: High-fidelity full-order model (FOM) sampling, snapshot matrix generation, singular value decomposition (SVD), reduced basis construction, and projection tensor precomputations.
     - *Online Phase*: Ultra-fast low-dimensional state integration, real-time boundary condition updates, and displacement reconstruction.
7. **Inverse FEM (iFEM) & Parameter Identification**:
   - **Inverse Shape Sensing & Deformation Reconstruction**: Reconstruct continuous 3D volumetric displacement and strain fields from sparse surface discrete observations (stereovision, optical markers, ultrasound tracking) via weighted least-squares variational formulations.
   - **Patient-Specific Parameter Identification**: Adjoint-based and optimization inverse solvers to identify constitutive tissue properties (elastic moduli, hyperelastic constants) from experimental force-displacement measurements.
   - **Stress-Free Reference Configuration Estimation**: Inverse elastostatics to determine the in vivo zero-pressure reference geometry from clinical scans of pre-stressed organs (pressurized vasculature, collapsed brain).
8. **Advanced Numerical & Computational Features**:
   - Sparse matrix storage and high-performance linear algebra (COO/CSR assembly, iterative and direct solvers, GPU acceleration via CuPy).
   - Numerical and physical damping models (e.g., Rayleigh damping, Newmark-$\beta$, Generalized-$\alpha$).

---

## 2. Universal Engineering & Code Quality Standards

All agents writing or modifying code in this repository MUST strictly adhere to the following standards:

### 2.1 Target Platform & Environment
- **Operating System**: Linux (Ubuntu 24.04 LTS x86_64).
- **Runtime**: Python 3.12+ (built-in system Python for Ubuntu 24.04 LTS).
- **Environment & Package Management**: Strictly managed using `uv` and `pyproject.toml`.
  - Add runtime dependencies to `project.dependencies` in `pyproject.toml` (e.g., `numpy`, `scipy`, `pyvista`, `cupy-cuda12x`).
  - Add testing and development tools to `[dependency-groups] dev` (e.g., `pytest`, `pytest-xdist`).
  - Always execute tasks within the managed environment using `uv run <command>`.

### 2.2 Architectural & Coding Conventions
- **Strictly Object-Oriented (OOP) Core**:
  - Organize all core physics and simulation logic into cohesive domain classes with explicit lifecycles (e.g., `Tet4`, `SolidDomain`, `NeoHookean`, `LinearStaticAnalysis`, `ProperOrthogonalDecomposition`).
  - **Utility Function Scope**: Stateless, frequently used mathematical transformations, indexing operations, COO sparsity assembly, and boundary node selectors in `psierra/utility/` and `psierra/boundary/selector.py` are explicitly permitted as pure functional helpers for ergonomic reuse across different analysis classes. All other domain and stateful logic must remain strictly class-based.
- **Pragmatic & Modern Python (3.12+)**:
  - Write clean, Pythonic, and readable code.
  - Avoid over-engineering, excessive abstraction layers, and brittle, deep inheritance hierarchies. Prefer composition over inheritance.
- **Strict Static Typing**:
  - Annotate every function parameter and return value with complete type hints.
  - Use Python standard library `dataclasses` or `pydantic` models for structured configuration, input payloads, and simulation states.
- **Mathematical Kernel Exception**:
  - Inside computational kernels, numerical integration routines, tensor/matrix assembly loops, and reduced-order projection algebra, single-letter variable names matching standard continuum mechanics notation (e.g., $K, M, C, F, B, D, u, v, a, V, \Phi, \Sigma$) are explicitly permitted and encouraged to keep mathematical expressions concise and aligned with textbook formulations. Add inline comments documenting their physical definitions.

---

## 3. Educational & Pedagogical Imperatives

`pSierra` serves a dual purpose: a production-grade simulation engine and a transparent, high-level learning platform for finite element theory, computational mechanics, and reduced-order modeling.

All agents MUST uphold these pedagogical standards:

1. **Textbook Mathematical Rigor**:
   - Root all implementations in established theoretical foundations.
   - Reference standard textbooks in docstrings and documentation:
     - **FEM & Continua**: K.J. Bathe, *Finite Element Procedures*; O.C. Zienkiewicz & R.L. Taylor, *The Finite Element Method*; T. Belytschko et al., *Nonlinear Finite Elements for Continua and Structures*; J. Bonet & R.D. Wood, *Nonlinear Continuum Mechanics for Finite Element Analysis*.
     - **Model Order Reduction & Hyper-Reduction**: P. Benner et al., *Model Order Reduction: Theory, Numerical Methods and Applications*; S. Volkwein, *Proper Orthogonal Decomposition: Theory and Reduced-Order Modelling*; S. Chaturantabut & D.C. Sorensen, *Nonlinear Model Reduction via Discrete Empirical Interpolation (DEIM)*; C. Farhat et al., *Dimensional reduction of nonlinear finite element dynamic models by an energy-conserving sampling and weighting (ECSW) technique*.
     - **Inverse FEM & Parameter Identification**: A. Tessler & H.R. Spangler, *A Variational Principle for Reconstruction of Elastic Deformations in Shear Deformable Plates and Shells* (iFEM); A. Tarantola, *Inverse Problem Theory and Methods for Model Parameter Estimation*.
2. **Computational & Numerical Acceleration Notes**:
   - Whenever translating a mathematical formulation into code, explicitly document any numerical or algorithmic techniques employed to accelerate computation (e.g., tensor vectorization, sparse matrix indexing, symmetry exploitation, quadrature optimizations, GPU dispatch, reduced basis projections).
3. **Continuous Knowledge Capture & Further Reading**:
   - Maintain clear records in docstrings, documentation, and agent logs specifying the mathematics and algorithms introduced at each development step.
   - Provide curated "Where to read further" references (chapter and section numbers) to guide in-depth learning.
4. **Bidirectional Code-Docstring & Documentation Synchronization**:
   - **Developer Contract**: Every domain class, kernel method, and public API in `src/psierra/` MUST contain structured, mathematically grounded docstrings explicitly stating:
     - The governing physical/continuum mechanics law and equation (in LaTeX or canonical notation).
     - Definitions and units of all tensors, matrices, inputs, and outputs.
     - Numerical/acceleration techniques employed (vectorization, sparsity, symmetry).
     - Cross-references to corresponding theoretical documentation and textbook chapters.
   - **Teacher Contract**: Every educational document and theory guide MUST mirror the exact mathematical terminology, variable symbols, and equations used in code docstrings, while embedding direct, clickable cross-references to the exact source files and methods (`ClassName.method_name`).
   - The user must be able to move seamlessly between code docstrings and theoretical documents with 1:1 matched explanations.

---

## 4. Symmetrical Naming Conventions & Lexicon

To maintain cognitive ergonomics and API consistency, all agents MUST enforce strict symmetrical naming conventions across classes, methods, functions, and CLI tools:

### Paired Lifecycle & Operational Antonyms
- `install` $\longleftrightarrow$ `uninstall` *(never pair asymmetric terms like `setup` with `uninstall`)*
- `start` $\longleftrightarrow$ `stop`
- `enable` $\longleftrightarrow$ `disable`
- `connect` $\longleftrightarrow$ `disconnect`
- `subscribe` $\longleftrightarrow$ `unsubscribe`
- `attach` $\longleftrightarrow$ `detach`
- `load` $\longleftrightarrow$ `unload`
- `set` $\longleftrightarrow$ `get`
- `project` $\longleftrightarrow$ `reconstruct` *(for MOR / coordinate mappings)*
- `reduce` $\longleftrightarrow$ `expand` *(for system matrices and degrees of freedom)*

### Canonical Action Verbs
- `compute`: Pure, deterministic mathematical evaluation returning values without mutating internal state (e.g., `compute_stiffness_matrix()`, `compute_strain_energy()`, `compute_reduced_basis()`).
- `solve`: Execution of numerical algorithms or iterative systems to resolve equilibrium (e.g., `solve()`).
- `update`: In-place state or kinematic mutations across iterations or time increments (e.g., `update_displacement()`, `update_geometry()`).

---

## 5. Collective Agent Memory & Retrospective Log

All agents (`manager`, `developer`, `tester`, `teacher`) MUST consult and maintain the shared project memory files located in `.agents/`:

- [`.agents/OPINION.md`](.agents/OPINION.md): Architectural consensus, critical trade-offs, numerical caveats, hardware/GPU considerations, benchmarks, and retrospective lessons learned.
- [`.agents/PLAN.md`](.agents/PLAN.md): Active development roadmap, sprint milestones, and component delivery status.

### Protocol:
- **Before Execution (Read)**: Consult both `.agents/OPINION.md` and `.agents/PLAN.md` to refine decisions against established mathematical consensus, hardware constraints, and documented pitfalls.
- **After Milestone Completion (Write)**: Update the logs with implemented architectures, trade-off decisions, discovered edge cases, convergence cautions, and open research questions.

---

## 6. Multi-Agent Development Pipeline & Team Contracts

To guarantee that code matches specifications and tests remain completely unbiased, feature development follows a strict four-role multi-agent pipeline:

```
[ manager ]  ──>  [ developer ]  ──>  [ tester ]  ──>  [ teacher ]
  Contract           Core Code          Adversarial       Educational Docs
  & Interfaces       & Examples         Test Suite        & LaTeX Math
```

Detailed role playbooks and system instructions are maintained in dedicated directories under [`.agents/agents/`](.agents/agents/):
- **Project Manager**: [`.agents/agents/manager/ROLE.md`](.agents/agents/manager/ROLE.md)
- **Software Developer**: [`.agents/agents/developer/ROLE.md`](.agents/agents/developer/ROLE.md)
- **Test Engineer**: [`.agents/agents/tester/ROLE.md`](.agents/agents/tester/ROLE.md)
- **Teacher / Educator**: [`.agents/agents/teacher/ROLE.md`](.agents/agents/teacher/ROLE.md)

### Role Summaries

| Role | Primary Mandate | Key Deliverables | Strict Constraints | Deep Handbook |
| :--- | :--- | :--- | :--- | :--- |
| **`manager`** | Translates physics & requirements into formal contracts | Class signatures, data transfer models, edge case lists, API specs | Strictly defines interfaces; does not write production implementation or tests | [`manager/ROLE.md`](.agents/agents/manager/ROLE.md) |
| **`developer`** | Implements numerical logic, domain classes, and example scripts | Clean code in `src/psierra/`, rich docstrings synchronized with docs, runnable scripts in `example/` | Pure OOP, strict typing, no loose functions, zero test bias (does not write tests in `tests/`) | [`developer/ROLE.md`](.agents/agents/developer/ROLE.md) |
| **`tester`** | Writes adversarial validation & verification suites | Independent tests in `tests/`, patch tests, benchmark verifications | Zero bias, tests independently from contract, runs `uv run pytest -n auto` | [`tester/ROLE.md`](.agents/agents/tester/ROLE.md) |
| **`teacher`** | Authors educational documentation and theoretical foundations | LaTeX theory guides, algorithmic notes, 1:1 docstring-matched code maps, citations | Plain English, retains math rigor, links equations to code lines, references developer's examples | [`teacher/ROLE.md`](.agents/agents/teacher/ROLE.md) |
