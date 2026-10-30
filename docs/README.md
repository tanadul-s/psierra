# pSierra Theoretical Foundations & Technical Documentation

Welcome to the theoretical documentation and educational guides for **`pSierra`**, a high-performance 3D Finite Element Method (FEM) simulation engine in modern Python.

This documentation bridges the mathematical foundations of continuum mechanics, numerical algorithms, and source code implementation. Every mathematical symbol, equation, and tensor operation is linked 1:1 with corresponding methods and docstrings in `psierra/src/psierra/`.

---

## Documentation Roadmap

```mermaid
flowchart TD
    D01["01. Continuum Mechanics & Material Models<br/><i>Kinematics, Hookean & Compressible Neo-Hookean</i>"]
    D02["02. Element Formulations (Tet4)<br/><i>Linear & Total Lagrangian Solid Tetrahedra</i>"]
    D03["03. System Assembly & Boundary Conditions<br/><i>COO/CSC Sparsity, Dirichlet Condensation, Gravity</i>"]
    D04["04. Solvers & Analysis Pipelines<br/><i>Direct Static, Implicit Newmark-beta Dynamics, Newton-Raphson</i>"]
    D05["05. Simulation Tutorials & Verification<br/><i>Benchmarks sim01 through sim07, ParaView Workflows</i>"]

    D01 --> D02
    D02 --> D03
    D03 --> D04
    D04 --> D05
```

---

## Detailed Guides

### 1. [Continuum Mechanics & Material Models](01_continuum_mechanics_and_materials.md)
- Kinematics of deformation: small-strain tensor $\boldsymbol{\varepsilon}$, deformation gradient $\boldsymbol{F}$, right Cauchy-Green tensor $\boldsymbol{C}$, and Green-Lagrange strain $\boldsymbol{E}$.
- Stress tensors: Cauchy stress $\boldsymbol{\sigma}$, First Piola-Kirchhoff $\boldsymbol{P}$, and Second Piola-Kirchhoff $\boldsymbol{S}$.
- Constitutive equations:
  - Linear isotropic elasticity (Hooke's law in 6x6 Voigt notation).
  - Compressible Neo-Hookean hyperelasticity with logarithmic volume penalty $K \ln(J)^2$.
  - 4th-order material tangent elasticity tensor $\mathbb{C} = 2 \frac{\partial \boldsymbol{S}}{\partial \boldsymbol{C}}$.
- Implementation map: [`Hookean`](../src/psierra/material/hookean.py) and [`NeoHookean`](../src/psierra/material/neohookean.py).

### 2. [Tetrahedral Element Formulations (`Tet4`)](02_element_formulations_tet4.md)
- Variational principle of virtual work and isoparametric mapping.
- 4-node constant-strain tetrahedron (`Tet4`): volume evaluation, linear shape functions $N_a(\boldsymbol{X})$, and gradient matrix $\boldsymbol{H}$.
- Small-strain formulation: linear strain-displacement matrix $\boldsymbol{B}_L$, element stiffness $\boldsymbol{k}_e = (\boldsymbol{B}_L^T \boldsymbol{C} \boldsymbol{B}_L) V_e$, and consistent mass matrix $\boldsymbol{m}_e$.
- Large-strain Total Lagrangian (TL) formulation:
  - Non-linear strain-displacement matrix $\boldsymbol{B}(\boldsymbol{u}) = \boldsymbol{B}_L + \boldsymbol{B}_{NL}(\boldsymbol{u})$.
  - Material tangent stiffness $\boldsymbol{k}_{mat} = (\boldsymbol{B}^T \boldsymbol{C} \boldsymbol{B}) V_e$.
  - Geometric initial-stress stiffness $\boldsymbol{k}_{geo} = (\boldsymbol{G}^T \hat{\boldsymbol{S}} \boldsymbol{G}) V_e$.
  - Internal force vector $\boldsymbol{f}_{int} = (\boldsymbol{B}^T \boldsymbol{S}_{voigt}) V_e$.
- Implementation map: [`Tet4`](../src/psierra/element/tet4.py).

### 3. [Sparse Assembly & Boundary Conditions](03_system_assembly_and_boundary_conditions.md)
- Triplet Coordinate (COO) assembly strategy for global stiffness $\boldsymbol{K}$ and mass $\boldsymbol{M}$.
- Boundary condition management:
  - Partitioning degrees of freedom into free ($\mathcal{F}$) and prescribed ($\mathcal{P}$) subsets.
  - Static condensation of non-homogeneous Dirichlet supports: $\boldsymbol{K}_{ff}\boldsymbol{u}_f = \boldsymbol{f}_f - \boldsymbol{K}_{fp}\boldsymbol{u}_p$.
  - Support reaction force recovery: $\boldsymbol{R} = \boldsymbol{K}\boldsymbol{u} - \boldsymbol{f}_{ext}$.
- Body force integration: consistent mass-matrix based gravity $\boldsymbol{f}_{grav} = \boldsymbol{M}\boldsymbol{g}$.
- Implementation map: [`SolidDomain`](../src/psierra/domain/solid.py), [`DirichletManager`](../src/psierra/boundary/dirichlet.py), [`GravityLoad`](../src/psierra/boundary/loads.py).

### 4. [Structural Analysis & Numerical Solvers](04_solvers_and_analysis_pipelines.md)
- **Linear Static Analysis**: direct sparse SuperLU factorization for $\boldsymbol{K}_{ff} \boldsymbol{u}_f = \boldsymbol{f}_{eff}$.
- **Linear Dynamic Analysis**: Rayleigh damping $\boldsymbol{C} = \alpha \boldsymbol{M} + \beta \boldsymbol{K}$, implicit Newmark-$\beta$ time integration, and pre-factorization of effective stiffness matrix $\boldsymbol{K}_{eff}$.
- **Nonlinear Static Analysis**: incremental Newton-Raphson iterations, out-of-balance residual $\boldsymbol{R} = \lambda \boldsymbol{f}_{ext} - \boldsymbol{f}_{int}(\boldsymbol{u})$, and backtracking line search guarding against element inversion ($\det(\boldsymbol{F}) \le 0$).
- **Nonlinear Dynamic Analysis**: implicit Newmark-$\beta$ integration coupled with inner Newton-Raphson equilibrium iterations.
- **ParaView Data Exchange**: `.pvd` XML collection generation linking `.vtu` snapshots.
- Implementation map: [`LinearStaticAnalysis`](../src/psierra/analysis/linear_static.py), [`LinearDynamicAnalysis`](../src/psierra/analysis/linear_dynamic.py), [`NonlinearStaticAnalysis`](../src/psierra/analysis/nonlinear_static.py), [`NonlinearDynamicAnalysis`](../src/psierra/analysis/nonlinear_dynamic.py), [`PvdCollectionWriter`](../src/psierra/exchange/pvd.py).

### 5. [Simulation Tutorials & Verification Benchmarks](05_simulation_tutorials_and_verification.md)
- Verification against Euler-Bernoulli & Timoshenko beam theory.
- Detailed walkthroughs of all 7 simulation examples (`sim01` to `sim07`).
- Verification tables: static equilibrium balance, energy conservation, dynamic oscillation, and large deformation axial shortening.
- Post-processing and ParaView visualization instructions.

---

## Theoretical Bibliography & Reference Literature

1. **K.J. Bathe**, *Finite Element Procedures*, Prentice Hall, 2nd Edition, 2014.
2. **J. Bonet & R.D. Wood**, *Nonlinear Continuum Mechanics for Finite Element Analysis*, Cambridge University Press, 2nd Edition, 2008.
3. **T. Belytschko, W.K. Liu, B. Moran, & K.I. Elkhodary**, *Nonlinear Finite Elements for Continua and Structures*, Wiley, 2nd Edition, 2014.
4. **O.C. Zienkiewicz, R.L. Taylor, & J.Z. Zhu**, *The Finite Element Method: Its Basis and Fundamentals*, Butterworth-Heinemann, 7th Edition, 2013.
5. **G.A. Holzapfel**, *Nonlinear Solid Mechanics: A Continuum Approach for Engineering*, John Wiley & Sons, 2000.
6. **N.M. Newmark**, *A Method of Computation for Structural Dynamics*, Journal of the Engineering Mechanics Division, ASCE, 85:67-94, 1959.

