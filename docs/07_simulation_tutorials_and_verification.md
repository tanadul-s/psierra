# 07. Simulation Tutorials, Analytical Verification & Benchmarks

This document provides complete tutorial walkthroughs, physical setups, theoretical benchmarks, analytical verification, and **physical insights** for all 10 demonstration simulations in `pSierra`.

---

## 1. Cantilever Beam Analytical Benchmarks

To independently verify the finite element implementations, we compare the numerical predictions against classical beam theories for a prismatic cantilever beam of length $L$, rectangular cross-section width $b$, height $h$, cross-sectional area $A = b h$, and second moment of area $I = \frac{b h^3}{12}$.

```
     |==============================================|  P (Tip Load)
     |                                              |  v
     |<- - - - - - - - - - - L - - - - - - - - - - >|
   Clamped                                         Free
   Root (X=0)                                      Tip (X=L)
```

### 1.1 Analytical Cantilever Beam Formulas
1. **Tip Point Load ($P$)**:
   - Euler-Bernoulli Theory:
     $$w_{max} = \frac{P L^3}{3 E I}$$
   - Timoshenko Theory (including shear deformation):
     $$w_{max} = \frac{P L^3}{3 E I} + \frac{P L}{\kappa G A}$$
     where $\kappa = 5/6$ for rectangular cross-sections and $G = \frac{E}{2(1+\nu)}$ is the shear modulus.
2. **Uniformly Distributed Body Weight ($q = \rho A g$)**:
   - Euler-Bernoulli Theory:
     $$w_{max} = \frac{q L^4}{8 E I}$$

#### Physical Meaning of the Timoshenko Shear Correction ($\kappa = 5/6$)
- **Why Classical Euler-Bernoulli Underpredicts Deflection**: Euler-Bernoulli beam theory assumes that plane cross-sections remain perfectly plane and normal to the deformed neutral axis ($w' = \theta$), completely neglecting internal shear distortion.
- **The Physical Reality**: In real 3D elastic bodies, transverse shear forces produce parabolic shear stress profiles across the height, warping the cross-section. The Timoshenko shear term $\frac{P L}{\kappa G A}$ accounts for this extra flexibility. Because `Tet4` solves full 3D continuum elasticity without 1D kinematic constraints, its deflection matches Timoshenko theory to within $3.5\%$, capturing both bending and shear compliance.

---

## 2. Complete Simulation Suite Walkthrough

All simulation examples are located in [`psierra/examples/05_simulation/`](../examples/05_simulation/).

### 2.1 Simulation 01: Linear Static Hookean (Distributed Tip Load)
- **Script**: [`sim01_linear_static_hookean.py`](../examples/05_simulation/sim01_linear_static_hookean.py)
- **Physics**: 3D Cantilever beam subjected to a distributed downward shear force $F_z = -150 \text{ kN}$ at the free tip.
- **Material**: Linear isotropic Hookean ($E = 21000 \text{ MPa}$, $\nu = 0.35$, $\rho = 1100 \times 10^{-6} \text{ kg/mm}^3$).
- **Mesh**: 1,457 nodes (4,371 DOFs), 6,304 linear tetrahedral elements (`Tet4`).
- **Boundary**: Clamped root at $X_{min}$ (153 nodes, 459 DOFs constrained to zero).

**Verification Results & Physical Meaning**:
- **Max Deflection**: $36.71 \text{ mm}$ (mean tip deflection $-36.08 \text{ mm}$, matching Timoshenko prediction within $3.5\%$).
- **Static Equilibrium Balance**:
  $$\sum R_z = +150000.00 \text{ N}, \quad \sum F_z = -150000.00 \text{ N}, \quad \text{Error} = 8.35 \times 10^{-8} \text{ N} \quad (\text{PASS})$$
  The support reaction identically balances the tip force, verifying Newton's third law.
- **Strain Energy**: $U = 2.7064 \times 10^6 \text{ J}$, representing the elastic energy stored in the deflected beam.

---

### 2.2 Simulation 02: Linear Static Hookean (Gravity Self-Weight)
- **Script**: [`sim02_linear_static_hookean.py`](../examples/05_simulation/sim02_linear_static_hookean.py)
- **Physics**: Static deflection under Earth's gravitational acceleration ($g_z = -9810 \text{ mm/s}^2$) formulated through the consistent mass matrix.
- **Total Integrated Mass**:
  $$M_{mesh} = \int_{\Omega} \rho \, d\Omega = 1025.20 \text{ kg} \quad (\text{Exact theoretical } \rho V = 1025.20 \text{ kg})$$
- **Total Downward Gravity Load**: $F_z = -10,057,212.00 \text{ N}$.

**Verification Results & Physical Meaning**:
- **Support Reaction**: $R_z = +10,057,212.00 \text{ N}$.
- **Equilibrium Error**: $|R_z + F_z| = 2.62 \times 10^{-6} \text{ N} \quad (\text{PASS})$.
- **Peak Resultant Deflection**: $451.82 \text{ mm}$.
- **Physical Meaning**: Demonstrates distributed continuous body loading where every differential volume of the material pulls downward with force $dF = \rho g dV$, causing the characteristic fourth-order polynomial deflection curve ($w \propto x^4$).

---

### 2.3 Simulation 03: Linear Dynamic Hookean (Implicit Newmark-$\beta$)
- **Script**: [`sim03_linear_dynamic_hookean.py`](../examples/05_simulation/sim03_linear_dynamic_hookean.py)
- **Physics**: Transient dynamic oscillation under suddenly applied gravitational acceleration with Rayleigh proportional damping ($\alpha = 2.0, \beta = 0.001$).
- **Time Stepping**: $\Delta t = 0.005 \text{ s}$, $N_{steps} = 100$ (total duration $0.50 \text{ s}$).
- **Solver Optimization**: Single pre-factorization of effective dynamic matrix $\boldsymbol{K}_{eff} = \boldsymbol{K}_{ff} + a_0 \boldsymbol{M}_{ff} + a_1 \boldsymbol{C}_{ff}$.

**Verification Results & Physical Meaning**:
- **Dynamic Amplification Factor**: Under a suddenly applied step load, the maximum transient displacement reaches $234.83 \text{ mm}$ at $t = 0.30 \text{ s}$, which is roughly **twice the steady-state deflection**. This illustrates the fundamental dynamic principle: a suddenly applied static load generates dynamic overshoot due to accumulated kinetic energy.
- **Smooth Damped Decay**: Rayleigh damping progressively dissipates this kinetic energy, causing the amplitude to decay smoothly over time.
- **Export**: 21 VTU snapshots exported to `sim03_output/output.pvd`.

---

### 2.4 Simulation 04: Nonlinear Static Hookean (Large Deformation Gravity)
- **Script**: [`sim04_nonlinear_static_hookean.py`](../examples/05_simulation/sim04_nonlinear_static_hookean.py)
- **Physics**: Finite rotation Total Lagrangian bending under gravitational body load ($0.5g = -4905 \text{ mm/s}^2$).
- **Solver**: Incremental Newton-Raphson across 10 load steps ($\lambda = 0.1 \dots 1.0$) with backtracking line search.

**Verification Results & Deep Physical Meaning**:
- **Axial Shortening (The "Pull-In" Effect)**:
  - Downward vertical deflection: $u_z = -142.74 \text{ mm}$.
  - Inward axial displacement: $u_y = -77.28 \text{ mm}$.
  - **The Physical Phenomenon**: When a beam undergoes large bending deflection, its physical material arc length is conserved:
    $$s = \int \sqrt{dx^2 + dz^2} \approx \text{constant}$$
    As the beam curves downward in $Z$, the free tip **must pull inward along $Y$**. Linear small-strain FEM predicts $u_y = 0$, which erroneously implies that bending artificially stretches the beam longer. The Total Lagrangian formulation correctly predicts this $77.28 \text{ mm}$ inward geometric shortening.
- **Convergence**: All 10 load steps converged within 4–5 Newton iterations with quadratic asymptotic convergence.

---

### 2.5 Simulation 05: Nonlinear Dynamic Hookean (Total Lagrangian Transient)
- **Script**: [`sim05_nonlinear_dynamic_hookean.py`](../examples/05_simulation/sim05_nonlinear_dynamic_hookean.py)
- **Physics**: Finite deformation dynamic oscillation under gravity ($E = 70000 \text{ MPa}$, $\Delta t = 0.005 \text{ s}$, 100 steps).

**Verification Results & Physical Meaning**:
- **Coupled Geometric-Dynamic Oscillations**: Captures large-amplitude swinging where internal stress stiffens the structure during the downswing and relaxes at the apex.
- **Convergence**: Stable convergence in 2–3 Newton iterations per time step with peak dynamic deflection of $133.11 \text{ mm}$ at $t = 0.25 \text{ s}$.

---

### 2.6 Simulation 06: Nonlinear Static Neo-Hookean (Hyperelasticity)
- **Script**: [`sim06_nonlinear_static_neohookean.py`](../examples/05_simulation/sim06_nonlinear_static_neohookean.py)
- **Physics**: Finite-strain compressible Neo-Hookean hyperelastic beam under full gravitational load ($1.0g = -9810 \text{ mm/s}^2$).
- **Material**: Compressible Neo-Hookean with shear modulus $\mu = 7777.78 \text{ MPa}$ and bulk modulus $K = 23333.33 \text{ MPa}$.

**Verification Results & Physical Meaning**:
- **Downward Deflection**: $-202.59 \text{ mm}$.
- **Axial Shortening**: $-133.72 \text{ mm}$.
- **Physical Resilience against Element Pinching**:
  Under full $1.0g$ load, Saint Venant-Kirchhoff would suffer localized compressive instabilities. In contrast, the Neo-Hookean model's logarithmic volume term $K (\ln J)^2$ produces an infinite repulsive barrier as volume decreases ($J \to 0$), enabling smooth convergence across all 10 load steps without a single element inverting.

---

### 2.7 Simulation 07: Nonlinear Dynamic Neo-Hookean (Hyperelastic Transient)
- **Script**: [`sim07_nonlinear_dynamic_neohookean.py`](../examples/05_simulation/sim07_nonlinear_dynamic_neohookean.py)
- **Physics**: Extended dynamic transient simulation ($N_{steps} = 200$, duration $1.00 \text{ s}$) of hyperelastic beam swinging under gravity with Rayleigh damping.

**Verification Results & Physical Meaning**:
- **Kinematic History**: The beam drops to a maximum dynamic swing of $243.99 \text{ mm}$ at $t = 0.40 \text{ s}$, rebounds with dissipative Rayleigh damping, and gradually stabilizes near the static equilibrium state ($\approx 205 \text{ mm}$) by $t = 1.00 \text{ s}$.
- **Physical Robustness**: 100% convergence across all 200 non-linear time steps, confirming both time integrator stability and material thermodynamic consistency.

---

### 2.8 Simulation 08: Tet10 vs Tet4 Cantilever Beam (Shear Locking Alleviation)
- **Script**: [`sim08_tet10_vs_tet4_cantilever.py`](../examples/05_simulation/sim08_tet10_vs_tet4_cantilever.py)
- **Physics**: Cantilever beam bending comparison under transverse tip shear load $F_z = -1000 \text{ N}$ to quantify artificial shear locking alleviation.
- **Formulations Compared**:
  - Base Linear Model: 245 nodes, 788 `Tet4` elements, 735 DOFs.
  - Upgraded Quadratic Model: 1,457 nodes, 788 `Tet10` elements, 4,371 DOFs (via `mesh.to_quadratic()`).
- **Material**: Hookean ($E = 21000 \text{ MPa}$, $\nu = 0.35$).

**Verification Results & Physical Meaning**:
- **Tip Deflection ($u_z$)**:
  - `Tet4`: $-0.04713 \text{ mm}$ (artificially stiff due to constant-strain shear locking).
  - `Tet10`: $-0.08834 \text{ mm}$ ($1.87\times$ more flexible, capturing true parabolic curvature).
- **Strain Energy**: $23.56 \text{ mJ}$ (Tet4) vs $44.17 \text{ mJ}$ (Tet10) ($1.87\times$ energy capture).
- **Physical Meaning**: Quadratic elements eliminate parasitic shear strain in bending, essential for capturing realistic organ compliance.

---

### 2.9 Simulation 09: Square Plate Bending Benchmark (Shell3 vs Shell6 vs Navier)
- **Script**: [`sim09_plate_bending_shell.py`](../examples/05_simulation/sim09_plate_bending_shell.py)
- **Physics**: Simply supported square thin plate ($1000 \times 1000 \times 10 \text{ mm}$) subjected to uniform transverse pressure load $q = -0.010 \text{ MPa}$ ($-10 \text{ kPa}$).
- **Material**: Isotropic Hookean ($E = 210000 \text{ MPa}$, $\nu = 0.30$, flexural rigidity $D = 6.41 \times 10^6 \text{ N}\cdot\text{mm}$).
- **Analytical Solution (Navier Double Fourier Series)**:
  $$w(x, y) = \frac{16 q}{\pi^6 D} \sum_{m=1,3,\dots}^{\infty} \sum_{n=1,3,\dots}^{\infty} \frac{\sin\left(\frac{m \pi x}{a}\right) \sin\left(\frac{n \pi y}{b}\right)}{m n \left[ \left(\frac{m}{a}\right)^2 + \left(\frac{n}{b}\right)^2 \right]^2} \implies w_{\text{max}} = -6.33672 \text{ mm}$$

**Verification Results & Physical Meaning**:
- **Shell3 (3-Node DKT)**: $w_0 = -6.29857 \text{ mm}$ (**$99.40\%$ match** to Navier analytical solution, strain energy $12,950.46 \text{ mJ}$).
- **Shell6 (6-Node Reissner-Mindlin)**: $w_0 = -5.97204 \text{ mm}$ (**$94.25\%$ match** to Navier, strain energy $12,291.29 \text{ mJ}$, accounting for transverse shear flexibility).
- **Physical Meaning**: Confirms that both Discrete Kirchhoff and Reissner-Mindlin plate bending elements converge reliably to classical elasticity theory without shear locking.

---

### 2.10 Simulation 10: Shell-Covered Soft Beam Under Self-Weight Gravity
- **Script**: [`sim10_shell_covered_beam_gravity.py`](../examples/05_simulation/sim10_shell_covered_beam_gravity.py)
- **Physics**: 3D Cantilever beam ($230 \text{ mm}$ span) undergoing gravity sag ($-9.81 \text{ m/s}^2$ downward along $Y$), demonstrating sag mitigation when a soft parenchymal tissue core is covered with a stiff outer membrane via `CoupledSolidShellDomain`.
- **Constitutive Properties**:
  - **Soft Parenchymal Core**: $E = 50.0 \text{ kPa}$, $\nu = 0.40$, $\rho = 1000 \text{ kg/m}^3$ (total weight $9.143 \text{ N}$).
  - **Dura Mater Shell Coating**: $E = 10.0 \text{ MPa}$ ($200\times$ stiffer), $\nu = 0.30$, thickness $t = 0.50 \text{ mm}$.
- **Models Compared**:
  - Model A (Regular Solid Beam): `SolidDomain` (735 DOFs).
  - Model B (Shell-Covered Beam): `CoupledSolidShellDomain` (1,281 DOFs = 735 solid $+ 546$ shell rotations).

**Verification Results & Physical Meaning**:
- **Maximum Tip Sag ($U_y$)**:
  - Regular Soft Beam: **$-143.92 \text{ mm}$** (severe sagging under self-weight).
  - Shell-Covered Beam: **$-24.25 \text{ mm}$** (restrained stability).
  - **Sag Reduction**: **$5.93\times$ reduction** ($83.1\%$ less sag).
- **Strain Energy**: $153.23 \text{ mJ}$ (regular) vs $29.50 \text{ mJ}$ (shell-covered) ($5.19\times$ stiffer structural assembly).
- **ParaView Multi-Domain Collection (`.pvd`)**: Exports `beam_shell_covered.pvd`, linking `beam_shell_covered_solid.vtu` and `beam_shell_covered_shell.vtu` for multi-field visualization.

---

## 3. Summary Comparison Table

| Simulation | Formulation | Element Type | Load Condition | Max Disp ($u_{max}$) | Convergence / Physical Observation |
|---|---|---|---|---|---|
| **`sim01`** | Linear Static | `Tet4` | Tip Shear Force ($-150 \text{ kN}$) | $36.71 \text{ mm}$ | Equilibrium error: $8.35 \times 10^{-8} \text{ N}$ (Timoshenko match) |
| **`sim02`** | Linear Static | `Tet4` | Consistent Gravity ($-1.0g$) | $451.82 \text{ mm}$ | Mass: $1025.20 \text{ kg}$, Eq error: $2.62 \times 10^{-6} \text{ N}$ |
| **`sim03`** | Linear Dynamic | `Tet4` | Dynamic Gravity (Rayleigh damped) | $234.83 \text{ mm}$ (peak) | 100/100 Steps; Dynamic overshoot $\approx 2 \times$ static |
| **`sim04`** | Non-linear Static | `Tet4` (TL) | Incremental Gravity ($-0.5g$) | $147.96 \text{ mm}$ | 10/10 Steps; Geometric pull-in: $u_y = -77.28 \text{ mm}$ |
| **`sim05`** | Non-linear Dynamic | `Tet4` (TL) | Dynamic Gravity ($-1.0g$) | $133.11 \text{ mm}$ (peak) | 100/100 Steps; 2-3 iters/step |
| **`sim06`** | Non-linear Static | `Tet4` (Neo-Hookean) | Incremental Gravity ($-1.0g$) | $219.81 \text{ mm}$ | 10/10 Steps; Inward pull-in: $u_y = -133.72 \text{ mm}$; $J > 0$ preserved |
| **`sim07`** | Non-linear Dynamic | `Tet4` (Neo-Hookean) | Dynamic Gravity ($-1.0g$, 200 steps) | $243.99 \text{ mm}$ (peak) | 200/200 Steps; Settling at static equilibrium ($205.77 \text{ mm}$) |
| **`sim08`** | Linear Static | `Tet4` vs `Tet10` | Tip Shear Load ($-1000 \text{ N}$) | $0.088 \text{ mm}$ (`Tet10`) | $1.87\times$ flexibility recovery over `Tet4`; shear locking alleviated |
| **`sim09`** | Linear Static | `Shell3` vs `Shell6` | Uniform Pressure ($-10 \text{ kPa}$) | $-6.299 \text{ mm}$ (`Shell3`) | $99.40\%$ match to Navier analytical series (Kirchhoff plate) |
| **`sim10`** | Linear Static | `CoupledSolidShellDomain` | Self-Weight Gravity ($-1.0g$) | $-24.25 \text{ mm}$ (Coupled) | $5.93\times$ sag reduction ($83.1\%$ less sag) via $0.5\text{ mm}$ stiff membrane |

---

## 4. ParaView 3D Visualization Workflow

All simulation outputs can be visualized and animated directly in [ParaView](https://www.paraview.org/):

1. **Static Solid & Shell Analyses (`sim01`, `sim02`, `sim04`, `sim06`, `sim08`, `sim09`)**:
   - File $\to$ Open $\to$ `examples/05_simulation/simXX_output/output.vtu`.
   - Click **Apply**.
   - Under **Coloring**, choose `Displacement`, `Displacement Y`, or `Displacement Magnitude`.
   - Coordinates in `.vtu` exports are automatically deformed ($x = X + u$), rendering the physical deflected state immediately.
2. **Coupled Multi-Domain Assemblies (`sim10`)**:
   - File $\to$ Open $\to$ `examples/05_simulation/sim10_output/beam_shell_covered.pvd`.
   - Click **Apply**.
   - ParaView automatically loads both the solid continuum core (`part="0"`) and outer shell skin (`part="1"`) into the 3D pipeline simultaneously.
3. **Dynamic Transient Analyses (`sim03`, `sim05`, `sim07`)**:
   - File $\to$ Open $\to$ `examples/05_simulation/simXX_output/output.pvd`.
   - Click **Apply**.
   - Click the green **Play** button in the animation toolbar to scrub through the dynamic oscillation.
