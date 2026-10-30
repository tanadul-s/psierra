# pSierra: 3D Finite Element Structural Simulation Engine

`pSierra` is a high-performance 3D Finite Element Method (FEM) simulation engine in modern Python (3.12+), designed for structural mechanics, biomechanical simulation, medical digital twins, and real-time model order reduction.

---

## Key Capabilities

- **3D Solid Elements**: 4-node constant-strain tetrahedral elements (`Tet4`) supporting both small-strain infinitesimal elasticity and large-strain Total Lagrangian kinematics.
- **Constitutive Material Models**:
  - Linear isotropic elasticity ([`Hookean`](src/psierra/material/hookean.py), Saint Venant-Kirchhoff).
  - Compressible hyperelasticity ([`NeoHookean`](src/psierra/material/neohookean.py)) with logarithmic volume penalty resisting element inversion.
- **Structural Solvers**:
  - **Linear Static**: Direct sparse factorization via SuperLU.
  - **Linear Dynamic**: Implicit Newmark-$\beta$ with Rayleigh proportional damping.
  - **Nonlinear Static**: Incremental Newton-Raphson with backtracking line search.
  - **Nonlinear Dynamic**: Finite-deformation implicit dynamic integration.
- **High-Performance Sparse Assembly**: Triplet Coordinate (COO) collection converted to Compressed Sparse Column (CSC) matrix format.
- **Boundary Conditions & Body Loads**: Multi-DOF Dirichlet managers, static condensation of non-homogeneous support movements, reaction force recovery, and consistent mass-matrix based gravity loads.
- **Data Exchange**: ParaView Data (`.pvd`) XML collections and unstructured VTK (`.vtu`) export.

---

## Quick Start

### Installation & Environment

This repository uses `uv` for Python 3.12+ environment and dependency management:

```bash
# Clone the repository
git clone https://github.com/your-org/project-sierra.git
cd project-sierra/psierra

# Run test suite
uv run pytest -n auto
```

---

## Running Simulations

The demonstration suite in [`example/05_simulation/`](example/05_simulation/) includes 7 end-to-end simulation pipelines:

```bash
# 1. Linear static beam bending under tip load
python example/05_simulation/sim01_linear_static_hookean.py

# 2. Linear static beam bending under gravity self-weight
python example/05_simulation/sim02_linear_static_hookean.py

# 3. Linear dynamic transient vibration with Rayleigh damping
python example/05_simulation/sim03_linear_dynamic_hookean.py

# 4. Nonlinear static large deflection (Saint Venant-Kirchhoff)
python example/05_simulation/sim04_nonlinear_static_hookean.py

# 5. Nonlinear dynamic transient vibration (Saint Venant-Kirchhoff)
python example/05_simulation/sim05_nonlinear_dynamic_hookean.py

# 6. Nonlinear static hyperelastic bending (Neo-Hookean)
python example/05_simulation/sim06_nonlinear_static_neohookean.py

# 7. Nonlinear dynamic hyperelastic transient (Neo-Hookean)
python example/05_simulation/sim07_nonlinear_dynamic_neohookean.py
```

---

## Theoretical Documentation

Comprehensive mathematical derivations, algorithmic notes, and 1:1 code implementation maps are available in [`docs/`](docs/):

1. [**01. Continuum Mechanics & Materials**](docs/01_continuum_mechanics_and_materials.md): Kinematics, strain measures, stress tensors, Hookean & Compressible Neo-Hookean models, and 4th-order tangents.
2. [**02. Tetrahedral Element Formulations (`Tet4`)**](docs/02_element_formulations_tet4.md): Virtual work, shape function gradients, $\boldsymbol{B}_L$ and $\boldsymbol{B}_{NL}$, material and geometric stiffness, and consistent mass.
3. [**03. System Assembly & Boundary Conditions**](docs/03_system_assembly_and_boundary_conditions.md): COO/CSC sparsity, Dirichlet condensation, support reactions, and consistent gravity loads.
4. [**04. Solvers & Analysis Pipelines**](docs/04_solvers_and_analysis_pipelines.md): Direct SuperLU, Newmark-$\beta$, Newton-Raphson with backtracking line search, and ParaView PVD exchange.
5. [**05. Simulation Tutorials & Verification**](docs/05_simulation_tutorials_and_verification.md): Analytical beam benchmarks, simulation walkthroughs, equilibrium checks, and 3D visualization.

---

## Multi-Agent Architecture

For details on the project mission, coding standards, and multi-agent roles (`manager`, `developer`, `tester`, `teacher`), see [`AGENTS.md`](AGENTS.md).
