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

### 2.3 Simulation 03: Linear Dynamic Hookean (Implicit Generalized-$\alpha$)

- **Script**: [`sim03_linear_dynamic_hookean.py`](../examples/05_simulation/sim03_linear_dynamic_hookean.py)
- **Physics**: Transient dynamic oscillation under suddenly applied gravitational acceleration with Rayleigh proportional damping ($\alpha = 2.0, \beta = 0.001$).
- **Time Stepping**: $\Delta t = 0.005 \text{ s}$, $N_{steps} = 100$ (total duration $0.50 \text{ s}$).
- **Solver Optimization**: Single pre-factorization of effective dynamic matrix $\boldsymbol{K}_{eff} = \boldsymbol{K}_{ff} + a_0 \boldsymbol{M}_{ff} + a_1 \boldsymbol{C}_{ff}$.

**Verification Results & Physical Meaning**:

- **Dynamic Amplification Factor**: Under a suddenly applied step load, the maximum transient displacement reaches $234.83 \text{ mm}$ at $t = 0.30 \text{ s}$, which is roughly **twice the steady-state deflection**. This illustrates the fundamental dynamic principle: a suddenly applied static load generates dynamic overshoot due to accumulated kinetic energy.
- **Smooth Damped Decay**: Rayleigh damping progressively dissipates this kinetic energy, causing the amplitude to decay smoothly over time.
- **Export**: 21 VTU snapshots exported to `sim03_output/result.pvd`.

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

### 2.11 Simulation 11: Corotational vs Linear Self-Weight Gravity Bending

- **Script**: [`sim11_corotational_cantilever_large_rotation.py`](../examples/05_simulation/sim11_corotational_cantilever_large_rotation.py)
- **Physics**: Soft tissue cantilever beam ($230 \text{ mm}$ span) undergoing large self-weight gravity sag ($g = -9.81 \text{ m/s}^2$ downward along $-Y$), demonstrating the elimination of artificial volume expansion and the capture of curvature-induced axial shortening via `CorotationalTet4`.
- **Constitutive Properties**: Soft parenchymal organ tissue ($E = 150.0 \text{ kPa}$, $\nu = 0.35$, $\rho = 1100 \text{ kg/m}^3$, total weight $10.057 \text{ N}$).
- **Models Compared**:
  - Model A (Linear Hookean): `Tet4` with `LinearStaticAnalysis` (small-strain assumption).
  - Model B (Corotational Hookean): `CorotationalTet4` with `NonlinearStaticAnalysis` (polar decomposition rigid rotation tracking).

**Verification Results & Physical Meaning**:

- **Transverse Tip Sag ($U_y$)**: $-55.49 \text{ mm}$ (Linear) vs $-53.09 \text{ mm}$ (Corotational).
- **Axial Tip Shortening ($U_z$)**:
  - Linear Static: **$-0.028 \text{ mm}$** (no shortening; tip-to-root chord length expands artificially to $236.57 \text{ mm}$, producing $+6.57 \text{ mm}$ unphysical elongation!).
  - Corotational: **$-7.30 \text{ mm}$** (tip curls inward/backward, preserving the true inextensible $228.94 \text{ mm}$ chord length and constant volume).
- **Newton-Raphson Convergence**: 10/10 load steps converged in an average of 3 iterations per step with zero element inversion.
- **Physical Meaning**: Demonstrates that `CorotationalTet4` provides robust large-rotation non-linear capability at near-linear computational cost without the risk of numerical instability.

---

### 2.12 Simulation 12: Updated Lagrangian Static Neo-Hookean

- **Script**: [`sim12_updated_lagrangian_static_neohookean.py`](../examples/05_simulation/sim12_updated_lagrangian_static_neohookean.py)
- **Physics**: Hyperelastic cantilever beam undergoing large finite deformations under incremental gravity ($10$ steps) formulated in current spatial configuration via spatial shape function gradients ($\boldsymbol{b}_n = \nabla_{\boldsymbol{x}} N_n$) and push-forward spatial elasticity tensor $\mathbb{c}^{\tau}$.
- **Constitutive Law**: Compressible Neo-Hookean ($E = 21000 \text{ MPa}, \nu = 0.35$).
- **Verification Results**:
  - Max displacement: $219.81 \text{ mm}$, exact equilibrium match ($< 0.01\%$) with Total Lagrangian formulation (`sim06`).
  - Strict preservation of positive Jacobian determinant ($J = \det \boldsymbol{F} > 0$).
  - Full Cauchy true stress field $\boldsymbol{\sigma}$ recovery.

---

### 2.13 Simulation 13: Updated Lagrangian Dynamic Neo-Hookean (Generalized-$\alpha$)

- **Script**: [`sim13_updated_lagrangian_dynamic_neohookean.py`](../examples/05_simulation/sim13_updated_lagrangian_dynamic_neohookean.py)
- **Physics**: Transient dynamic oscillation of finite-strain hyperelastic beam formulated in Updated Lagrangian with Generalized-$\alpha$ time integration ($\rho_\infty = 0.8$) and Rayleigh damping ($\alpha = 1.0, \beta = 0.001$).
- **Verification Results**:
  - Peak dynamic deflection of $\sim 243.9 \text{ mm}$, smoothly decaying and settling at the static equilibrium position ($205.7 \text{ mm}$).
  - Full consistency between current spatial kinematics and mid-point dynamic balance.
  - ParaView `.pvd` collection with Cauchy true stress and von Mises invariants.

---

### 2.14 Simulation 14: Solid Formulations Comparison (Linear vs Corotational vs TL vs UL)

- **Script**: [`sim14_formulations_comparison_gravity.py`](../examples/05_simulation/sim14_formulations_comparison_gravity.py)
- **Physics**: 4-way benchmark on identical cantilever beam mesh under identical gravitational acceleration comparing:
  1. Small-strain Linear `Tet4`
  2. Corotational `CorotationalTet4`
  3. Total Lagrangian `Tet4TotalLagrangian` (Neo-Hookean)
  4. Updated Lagrangian `Tet4UpdatedLagrangian` (Neo-Hookean)
- **Verification Results**:
  - Eliminates artificial small-strain dilation: linear formulation produces unphysical chord elongation, whereas both Corotational and Finite-Strain formulations correctly exhibit axial shortening ($U_z < 0$).
  - Total Lagrangian and Updated Lagrangian match within $0.005\%$ in displacements and internal energy.

---

### 2.15 Simulation 15: Hyperelastic Material Models Comparison

- **Script**: [`sim15_hyperelastic_models_comparison.py`](../examples/05_simulation/sim15_hyperelastic_models_comparison.py)
- **Physics**: Side-by-side verification of non-linear constitutive formulations under tip shear loading:
  1. **Linear Hookean**: Reference constant-tangent baseline.
  2. **Neo-Hookean**: Classical 1-parameter strain-energy function for moderate polymer/rubber strains.
  3. **Mooney-Rivlin**: 2-parameter formulation capturing distinct tension vs compression stiffness asymmetry.
  4. **Yeoh**: 3-parameter polynomial formulation modeling the characteristic upturn/stiffening at large stretch ratios.
- **Verification Results**:
  - Captures high-strain stiffening in Yeoh model ($C_{30} > 0$), preventing unphysical elongation under heavy shear.
  - S-shaped inflection in Mooney-Rivlin model matches physical experimental elastomer test curves.

---

### 2.16 Simulation 16: Dynamic Time Integration Scheme Comparison

- **Script**: [`sim16_dynamic_time_integration_comparison.py`](../examples/05_simulation/sim16_dynamic_time_integration_comparison.py)
- **Physics**: Direct comparison of high-frequency spectral radius $\rho_\infty \in [0.0, 1.0]$ in Generalized-$\alpha$ dynamic analysis under sudden step load:
  1. $\rho_\infty = 1.0$ (Undamped Newmark / Midpoint Rule): Energy-preserving, zero algorithmic damping.
  2. $\rho_\infty = 0.8$ (Optimal Generalized-$\alpha$, Default): Selective high-frequency dissipation with minimal low-frequency damping.
  3. $\rho_\infty = 0.5$ (HHT-$\alpha$): Moderate algorithmic damping for stiff structural transients.
  4. $\rho_\infty = 0.0$ (Asymptotic Annihilation): Maximum dissipation, single-step cancellation of spurious high-frequency modes.
- **Verification Results**:
  - Macroscopic tip sag matches across all schemes within $0.03\%$ ($-139.01 \text{ mm}$ vs $-139.05 \text{ mm}$), confirming that low-frequency modes are unattenuated.
  - High-frequency shock waves and stress spikes are smoothly filtered: initial shock von Mises stress drops from $714.5 \text{ MPa}$ ($\rho_\infty = 1.0$) to $620.8 \text{ MPa}$ ($\rho_\infty = 0.0$), eliminating contact chatter.

---

### 2.17 Simulation 17: Severe Axial Torsion of Hyperelastic Beam (Poynting Effect & Cross-Sectional Warping)

- **Script**: [`sim17_axial_torsion_hyperelastic_beam.py`](../examples/05_simulation/sim17_axial_torsion_hyperelastic_beam.py)
- **Physics**: Severe progressive axial twisting of a 3D continuum cantilever beam up to a full half-turn ($180^\circ$ / $\pi \text{ rad}$) rotation to demonstrate finite-strain kinematics, the second-order **Poynting axial effect**, and **Saint-Venant cross-sectional warping**.
- **Kinematic Setup**:
  - Root face ($Z = -30 \text{ mm}$): Fully clamped ($u_x = u_y = u_z = 0$).
  - Tip face ($Z = 200 \text{ mm}$): In-plane rigid-body rotation prescribed ($u_x(\theta) = X(\cos\theta - 1) - Y\sin\theta, u_y(\theta) = X\sin\theta + Y(\cos\theta - 1)$), while **$u_z$ is left completely free** to enable natural axial elongation/contraction and warping.
- **Verification Results & Physical Meaning**:
  - **The Poynting Effect**: Linear elasticity predicts zero axial stress ($\sigma_{zz} = 0$) and zero axial displacement ($U_z = 0$). In finite elasticity, helical fiber elongation produces normal stresses ($\sigma_{zz} \propto r^2 \theta^2$). The mean tip axial displacement grows monotonically across all models, reaching $+5.15 \text{ mm}$ (Neo-Hookean), $+6.20 \text{ mm}$ (Mooney-Rivlin), and $+5.17 \text{ mm}$ (Yeoh) at $180^\circ$.
  - **Cross-Sectional Warping**: The non-circular rectangular cross-section exhibits out-of-plane distortion, reaching a warping amplitude $w_{\text{amp}} = \max(U_z) - \min(U_z) \approx 1.50 \text{ mm} - 1.61 \text{ mm}$ at $180^\circ$.
  - **Root Reaction Torque**: Restoring moment reaches $M_z = -214.13 \text{ kN}\cdot\text{m}$ (Neo-Hookean), $-210.03 \text{ kN}\cdot\text{m}$ (Mooney-Rivlin), and $-216.51 \text{ kN}\cdot\text{m}$ (Yeoh) at $180^\circ$.
  - **Constitutive Comparison**: All three models achieve 18/18 step convergence with zero element inversion ($\det \boldsymbol{F} > 0$).
  - **ParaView PVD Collections**: Time-series collections for side-by-side visualization exported to:
    - `examples/05_simulation/sim17_output/neohookean/result.pvd`
    - `examples/05_simulation/sim17_output/mooney_rivlin/result.pvd`
    - `examples/05_simulation/sim17_output/yeoh/result.pvd`

### 2.18 Simulation 01 (Interactive): Real-Time Online Dynamic Cantilever Simulation

- **Script**: [`sim01_online_dynamic_cantilever.py`](../examples/06_interactive/sim01_online_dynamic_cantilever.py)
- **Physics**: Real-time interactive FEA simulation of a 3D continuum cantilever beam under self-weight gravity and dynamic user perturbations (impulse forces, sinusoidal resonance driving) using Generalized-$\alpha$ integration ($\rho_\infty = 0.8$) and single-factorization SuperLU reuse.
- **Key Capabilities**:
  - **Physics / Render Clock Decoupling**: Physics advances at fixed $\Delta t = 2 \text{ ms}$ ($500 \text{ Hz}$), synchronized with wall-clock time via `advance_wall_clock()` to prevent lag spikes.
  - **Ultra-High Physics Rate**: Sparse dynamic matrix $\boldsymbol{K}_{\text{eff}}$ is factorized once at $t = 0$. Each step executes in **$82.9\,\mu\text{s}$**, achieving an effective physics rate of **$12,056 \text{ Hz}$** ($24.1\times$ real-time).
  - **In-Place VTK Mutation**: Direct point buffer update (`update_pyvista_grid()`) enables smooth $60+ \text{ FPS}$ 3D visualization without VTK memory reallocations.
  - **Interactive Disturbance Handling**: Real-time keyboard callbacks (`[I]` downward impulse, `[U]` upward impulse, `[H]` harmonic resonance excitation, `[C]` clear forces, `[R]` reset rest state).
  - **Headless & CI Execution**: Automated fallback to non-GUI demonstration sequence when running without an X11/Wayland display (`--headless`).
  - **Snapshot Streaming**: Saves time-series snapshots and collection file to `examples/06_interactive/sim01_output/sim01_online_dynamic.pvd`.

### 2.19 Simulation 18: Nonlinear Quadratic Tetrahedral (Tet10) Hyperelastic Beam Bending

- **Script**: [`sim18_tet10_nonlinear_hyperelastic_beam.py`](../examples/05_simulation/sim18_tet10_nonlinear_hyperelastic_beam.py)
- **Physics**: Finite-strain hyperelastic bending of a 3D continuum cantilever beam comparing linear Tet4, nonlinear Tet4 (Total Lagrangian), and nonlinear Tet10 (Total & Updated Lagrangian) to evaluate shear-locking alleviation and TL/UL formulation equivalence.
- **Key Findings**:
  - **Shear-Locking Elimination**: Constant-strain linear tetrahedra (`Tet4`) lock in bending, underpredicting deflection ($-18.83 \text{ mm}$). Quadratic `Tet10` elements recover true flexural curvature with **$1.51\times$ higher compliance** ($-28.43 \text{ mm}$).
  - **TL vs UL Machine Precision**: Total Lagrangian and Updated Lagrangian on quadratic `Tet10` elements yield identical displacements to $|\boldsymbol{u}_{\text{TL}} - \boldsymbol{u}_{\text{UL}}| = 4.26 \times 10^{-14} \text{ mm}$.
  - **Convergence**: Solves 5 load increments in 4-5 Newton-Raphson iterations per step with asymptotic quadratic residual reduction.
  - **ParaView Collections**: Multi-model snapshots and master PVD files exported to `examples/05_simulation/sim18_output/`.

### 2.20 Simulation 02 (Interactive): Real-Time Dynamic 3D Axial Torsion Pipeline

- **Script**: [`sim02_online_dynamic_axial_torsion.py`](../examples/06_interactive/sim02_online_dynamic_axial_torsion.py)
- **Physics**: Real-time online dynamic simulation of a 3D continuum cantilever beam subjected to pure torsional twisting moments ($M_z$), torsional wave propagation, and elastic spring-back using Generalized-$\alpha$ dynamic time integration with support for Corotational, Linear, and Hyperelastic formulations.
- **Key Capabilities**:
  - **Pure Torsional Force Couple**: Evaluates an exact tangential force couple over tip face boundary nodes with zero net linear shear force, generating pure twisting moments $M_z$.
  - **Interactive Torsion Handling**: Interactive keyboard clockwise torque (`[T]`), counter-clockwise torque (`[Y]`), harmonic torsional resonance (`[H]`), instant torque release for free spring-back vibration (`[C]`), and simulation reset (`[R]`).
  - **Diverging Colormap Visualization**: Real-time extraction of local twist rotation angles $\theta_z$ rendered in a perceptually uniform diverging `coolwarm` palette centered at $0.0^\circ$ (neutral grey), with positive clockwise twist in warm red and negative counter-clockwise twist in cool blue.
  - **Headless & Snapshot Support**: Runs automated headless verification sequence (`--headless`) and streams time-series `.vtu` and `.pvd` collections to `examples/06_interactive/sim02_output/`.

---

## 3. Summary Comparison Table

| Simulation | Formulation | Element Type | Load Condition | Max Disp ($u_{max}$) | Convergence / Physical Observation |
| --- | --- | --- | --- | --- | --- |
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
| **`sim11`** | Non-linear Static | `CorotationalTet4` | Self-Weight Gravity ($-1.0g$) | $55.49 \text{ mm}$ | True arc-length preservation; axial tip curl ($U_z = -7.30 \text{ mm}$) |
| **`sim12`** | Non-linear Static | `Tet4` (UL Neo-Hookean) | Incremental Gravity ($-1.0g$) | $219.81 \text{ mm}$ | 10/10 Steps; Cauchy stress and deformation match TL within $0.01\%$ |
| **`sim13`** | Non-linear Dynamic | `Tet4` (UL Neo-Hookean) | Dynamic Gravity (Gen-$\alpha, \rho_\infty=0.8$) | $243.9 \text{ mm}$ (peak) | Spatial mid-point equilibrium; steady-state settling |
| **`sim14`** | Comparison | Linear / CR / TL / UL | Self-Weight Gravity ($-1.0g$) | Multi-field | Elimination of artificial dilation; UL vs TL identical |
| **`sim15`** | Comparison | Hookean / NH / MR / Yeoh | Tip Shear Load ($-150 \text{ kN}$) | Multi-field | High-strain stiffening in Yeoh; Mooney-Rivlin S-curve |
| **`sim16`** | Comparison | Generalized-$\alpha$ ($\rho_\infty \in [0, 1]$) | Dynamic Gravity Transient | Multi-field | Low-frequency fidelity ($<0.03\%$ error); shock stress filtered |
| **`sim17`** | Non-linear Static | `Tet4` (TL Hyperelastic) | Severe Axial Torsion ($180^\circ$) | $84.85 \text{ mm}$ | Poynting elongation ($+5.15 \text{ mm}$), warping ($1.51 \text{ mm}$), $M_z = -214.1 \text{ kN}\cdot\text{m}$ |
| **`sim18`** | Non-linear Static | `Tet4` vs `Tet10` (TL/UL) | Transverse Shear ($-100 \text{ kN}$) | $-28.43 \text{ mm}$ (`Tet10`) | $1.51\times$ flex recovery over `Tet4`; TL vs UL match to $< 10^{-13} \text{ mm}$ |
| **`sim01 (Interactive)`** | Online Dynamic | `Tet4` / `Tet10` (Lin/CR/TL/UL) | Gravity + Impulse + Harmonic | $-7.30 \text{ mm}$ (peak) | $11,527 \text{ Hz}$ physics rate ($86.8\,\mu\text{s}/\text{step}$), $23.1\times$ real-time speedup; coolwarm diverging colormap |
| **`sim02 (Interactive)`** | Online Dynamic | `Tet4` / `Tet10` (CR/Lin/TL/UL) | Dynamic Torque $M_z$ + Harmonic | $83.61^\circ$ (peak) | Pure torsional force couple; dynamic spring-back; coolwarm twist angle map |

---

## 4. ParaView 3D Visualization Workflow

All simulation outputs can be visualized and animated directly in [ParaView](https://www.paraview.org/):

1. **Static Solid & Shell Analyses (`sim01`, `sim02`, `sim04`, `sim06`, `sim08`, `sim09`, `sim11`, `sim12`, `sim17`)**:
   - File $\to$ Open $\to$ `examples/05_simulation/simXX_output/result.vtu` (or `sim17_output/<model>/result.pvd` for animated torsion progression).
   - Click **Apply**.
   - Under **Coloring**, choose `Displacement`, `Displacement Magnitude`, `Von Mises Stress`, or `Cauchy Stress`.
   - Coordinates in `.vtu` exports are automatically deformed ($x = X + u$), rendering the physical deflected state immediately.
2. **Coupled Multi-Domain Assemblies (`sim10`)**:
   - File $\to$ Open $\to$ `examples/05_simulation/sim10_output/beam_shell_covered.pvd`.
   - Click **Apply**.
   - ParaView automatically loads both the solid continuum core (`part="0"`) and outer shell skin (`part="1"`) into the 3D pipeline simultaneously.
3. **Dynamic Transient Analyses (`sim03`, `sim05`, `sim07`, `sim13`, `sim16`)**:
   - File $\to$ Open $\to$ `examples/05_simulation/simXX_output/result.pvd` (or `sim16_output/rho_XXX/result.pvd`).
   - Click **Apply**.
   - Click the green **Play** button in the animation toolbar to scrub through the dynamic oscillation.
