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
git clone https://github.com/tanadul-s/project-sierra.git
cd project-sierra/psierra

# Run test suite
uv run pytest -n auto
```

---

## Running Simulations

The demonstration suite in [`examples/`](examples/) is categorized into clear engineering disciplines:

```bash
# Linear static beam bending
python examples/02_linear_static/0201_tip_load_beam.py
python examples/02_linear_static/0202_gravity_beam.py

# Linear dynamic transient vibration with Rayleigh damping
python examples/03_linear_dynamic/0301_dynamic_beam.py

# Nonlinear static large deflection (Saint Venant-Kirchhoff & Total Lagrangian Neo-Hookean)
python examples/04_nonlinear_static/0401_svk_beam.py
python examples/04_nonlinear_static/0402_tl_beam.py

# Nonlinear dynamic transient vibration
python examples/05_nonlinear_dynamic/0501_svk_beam.py
python examples/05_nonlinear_dynamic/0502_tl_beam.py

# Real-time interactive dynamics
python examples/06_interactive/0601_online_cantilever.py --headless

# Comprehensive multi-model benchmarks
python examples/07_benchmarks/0702_kinematic_formulations.py

# Contact mechanics & self-contact
python examples/08_contact/0803_dynamic_rigid_floor.py
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
