# Chapter 08: Model Order Reduction (MOR) & Proper Orthogonal Decomposition (POD-Galerkin)

Model Order Reduction (MOR) establishes the computational engine necessary for real-time surgical digital twins, haptic feedback, and inverse finite element modeling (iFEM). By projecting high-dimensional continuum mechanics equations ($n \sim 10^4\text{--}10^6$ DOFs) onto an optimal low-dimensional empirical subspace ($r \sim 5\text{--}20$ modes), MOR accelerates dynamic integration to update rates exceeding $20{,}000\text{ Hz}$ ($< 0.05\text{ ms}$ per evaluation) while preserving physical fidelity ($< 2\%$ relative error).

---

## 1. Mathematical Foundations

### 1.1 Full-Order Structural Dynamic System (FOM)

The discretized equations of motion for a damped linear continuum in full-order space ($\mathbb{R}^n$) are governed by:
$$\boldsymbol{M} \ddot{\boldsymbol{u}}(t) + \boldsymbol{C} \dot{\boldsymbol{u}}(t) + \boldsymbol{K} \boldsymbol{u}(t) = \boldsymbol{F}(t)$$
where:

* $\boldsymbol{M} \in \mathbb{R}^{n \times n}$ is the symmetric positive definite (SPD) global mass matrix.
* $\boldsymbol{C} \in \mathbb{R}^{n \times n}$ is the global damping matrix (e.g. Rayleigh damping $\boldsymbol{C} = \alpha \boldsymbol{M} + \beta \boldsymbol{K}$).
* $\boldsymbol{K} \in \mathbb{R}^{n \times n}$ is the global tangent/elastic stiffness matrix.
* $\boldsymbol{u}, \dot{\boldsymbol{u}}, \ddot{\boldsymbol{u}} \in \mathbb{R}^n$ are nodal displacements, velocities, and accelerations.
* $\boldsymbol{F}(t) \in \mathbb{R}^n$ is the external body/surface load vector.

Partitioning into free degrees of freedom ($\mathcal{F}$) and prescribed Dirichlet boundary degrees of freedom ($\mathcal{P}$) with $\boldsymbol{u}_{\mathcal{P}} = \boldsymbol{u}_p$ yields the unconstrained free system:
$$\boldsymbol{M}_{ff} \ddot{\boldsymbol{u}}_f + \boldsymbol{C}_{ff} \dot{\boldsymbol{u}}_f + \boldsymbol{K}_{ff} \boldsymbol{u}_f = \boldsymbol{f}_{\text{ext}, f} - \boldsymbol{K}_{fp} \boldsymbol{u}_p$$

---

### 1.2 Offline Snapshot Matrix Harvesting & SVD Basis Generation

To construct a reduced subspace that captures the dominant deformation manifold, we record $m$ state displacement snapshot vectors $\boldsymbol{u}^{(1)}, \boldsymbol{u}^{(2)}, \dots, \boldsymbol{u}^{(m)} \in \mathbb{R}^n$ across representative load histories or dynamic time steps:
$$\boldsymbol{S} = \begin{bmatrix} \boldsymbol{u}^{(1)} & \boldsymbol{u}^{(2)} & \cdots & \boldsymbol{u}^{(m)} \end{bmatrix} \in \mathbb{R}^{n \times m} \quad (m \ll n)$$

We factorize $\boldsymbol{S}$ using thin/economic Singular Value Decomposition (SVD):
$$\boldsymbol{S} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^T$$
where:

* $\boldsymbol{U} = [\boldsymbol{\phi}_1, \boldsymbol{\phi}_2, \dots, \boldsymbol{\phi}_m] \in \mathbb{R}^{n \times m}$ contains the orthonormal left singular vectors (spatial POD modes) satisfying $\boldsymbol{U}^T \boldsymbol{U} = \boldsymbol{I}_m$.
* $\boldsymbol{\Sigma} = \operatorname{diag}(\sigma_1, \sigma_2, \dots, \sigma_m)$ with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_m \ge 0$ contains the singular values.
* $\boldsymbol{V} \in \mathbb{R}^{m \times m}$ contains the orthonormal right singular vectors (temporal coefficients).

#### Energy Spectrum Truncation Criterion

The energy (variance) of each POD mode is proportional to $\sigma_i^2$. The optimal truncation dimension $r$ is chosen according to the cumulative energy spectrum ratio:
$$E(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^m \sigma_i^2} \ge \eta$$
where $\eta \in (0, 1]$ is the user-defined energy retention threshold (typically $\eta = 0.999$, retaining $\ge 99.9\%$ of the strain energy).

The optimal orthonormal reduced basis matrix is then:
$$\boldsymbol{\Phi} = \boldsymbol{U}[:, :r] \in \mathbb{R}^{n \times r} \quad \text{with} \quad \boldsymbol{\Phi}^T \boldsymbol{\Phi} = \boldsymbol{I}_r$$

---

### 1.3 POD-Galerkin Coordinate Reduction

Let the low-dimensional generalized coordinate vector be $\boldsymbol{u}_r(t) \in \mathbb{R}^r$ with $r \ll n$ ($r \approx 5\text{--}20$). The full-field state is approximated via linear superposition:
$$\boldsymbol{u}_f(t) \approx \boldsymbol{\Phi}_f \boldsymbol{u}_r(t)$$

Substituting into the governing equations and requiring the residual to be orthogonal to the subspace spanned by $\boldsymbol{\Phi}_f$ (the classical Galerkin projection) yields:
$$\boldsymbol{\Phi}_f^T \left( \boldsymbol{M}_{ff} \boldsymbol{\Phi}_f \ddot{\boldsymbol{u}}_r + \boldsymbol{C}_{ff} \boldsymbol{\Phi}_f \dot{\boldsymbol{u}}_r + \boldsymbol{K}_{ff} \boldsymbol{\Phi}_f \boldsymbol{u}_r - \boldsymbol{f}_f \right) = \boldsymbol{0}$$

This condenses the $n$-dimensional system to a dense, coupled $r \times r$ dynamical system:
$$\boldsymbol{M}_r \ddot{\boldsymbol{u}}_r + \boldsymbol{C}_r \dot{\boldsymbol{u}}_r + \boldsymbol{K}_r \boldsymbol{u}_r = \boldsymbol{f}_r$$
where the precomputed reduced operators are:
$$\boldsymbol{M}_r = \boldsymbol{\Phi}_f^T \boldsymbol{M}_{ff} \boldsymbol{\Phi}_f \in \mathbb{R}^{r \times r}, \quad \boldsymbol{K}_r = \boldsymbol{\Phi}_f^T \boldsymbol{K}_{ff} \boldsymbol{\Phi}_f \in \mathbb{R}^{r \times r}$$
$$\boldsymbol{C}_r = \boldsymbol{\Phi}_f^T \boldsymbol{C}_{ff} \boldsymbol{\Phi}_f \in \mathbb{R}^{r \times r}, \quad \boldsymbol{f}_r = \boldsymbol{\Phi}_f^T (\boldsymbol{f}_{\text{ext}, f} - \boldsymbol{K}_{fp} \boldsymbol{u}_p) \in \mathbb{R}^r$$

Because $\boldsymbol{M}_{ff}$ and $\boldsymbol{K}_{ff}$ are symmetric positive definite and $\boldsymbol{\Phi}_f$ has full column rank $r$, $\boldsymbol{M}_r$ and $\boldsymbol{K}_r$ are strictly symmetric positive definite (SPD):
$$\lambda_{\min}(\boldsymbol{K}_r) > 0, \quad \lambda_{\min}(\boldsymbol{M}_r) > 0$$

---

### 1.4 Real-Time Generalized-$\alpha$ Dynamic Integration

To integrate the reduced system in time with unconditional stability and user-controlled high-frequency dissipation, we implement the Generalized-$\alpha$ scheme (Chung & Hulbert, 1993) parameterized by the high-frequency spectral radius $\rho_\infty \in [0.0, 1.0]$:
$$\alpha_m = \frac{2\rho_\infty - 1}{\rho_\infty + 1}, \quad \alpha_f = \frac{\rho_\infty}{\rho_\infty + 1}$$
$$\gamma = \frac{1}{2} - \alpha_m + \alpha_f, \quad \beta = \frac{1}{4}(1 - \alpha_m + \alpha_f)^2$$

The effective dynamic stiffness matrix in the reduced subspace is:
$$\boldsymbol{K}_{\text{eff}, r} = (1 - \alpha_f) \boldsymbol{K}_r + (1 - \alpha_m) a_0 \boldsymbol{M}_r + (1 - \alpha_f) a_1 \boldsymbol{C}_r \in \mathbb{R}^{r \times r}$$
where $a_0 = \frac{1}{\beta \Delta t^2}$ and $a_1 = \frac{\gamma}{\beta \Delta t}$.

Because $\boldsymbol{K}_{\text{eff}, r}$ is constant in linear dynamics and of tiny dimension $r \times r$, we pre-factorize it **once** offline using dense Cholesky decomposition:
$$\boldsymbol{L}_r \boldsymbol{L}_r^T = \boldsymbol{K}_{\text{eff}, r}$$

At each online step, the right-hand side effective load is evaluated via dense matrix-vector products:
$$\boldsymbol{m}_{\text{part}} = (1 - \alpha_m)(a_0 \boldsymbol{u}_r + a_2 \boldsymbol{v}_r + a_3 \boldsymbol{a}_r) - \alpha_m \boldsymbol{a}_r$$
$$\boldsymbol{c}_{\text{part}} = (1 - \alpha_f)(a_1 \boldsymbol{u}_r + a_4 \boldsymbol{v}_r + a_5 \boldsymbol{a}_r) - \alpha_f \boldsymbol{v}_r$$
$$\boldsymbol{f}_{\text{eff}} = \boldsymbol{f}_r + \boldsymbol{M}_r \boldsymbol{m}_{\text{part}} + \boldsymbol{C}_r \boldsymbol{c}_{\text{part}} - \alpha_f (\boldsymbol{K}_r \boldsymbol{u}_r)$$

The next reduced displacement $\boldsymbol{u}_{r, n+1}$ is computed via a single dense Cholesky back-substitution:
$$\boldsymbol{u}_{r, n+1} = \boldsymbol{L}_r^{-T} \boldsymbol{L}_r^{-1} \boldsymbol{f}_{\text{eff}}$$
followed by standard Newmark kinematic updates for acceleration and velocity. Total per-step cost is $\mathcal{O}(r^2)$, executing in $< 0.02\text{ ms}$ ($> 45{,}000\text{ Hz}$).

---

### 1.5 Instant Reduced Static Equilibrium Solving

For static deflection analysis, parameter sweeps, and haptic force feedback queries, the reduced static equilibrium equations:
$$\boldsymbol{K}_r \boldsymbol{u}_r = \boldsymbol{f}_r$$
are pre-factorized via dense Cholesky decomposition $\boldsymbol{L}_{K, r} \boldsymbol{L}_{K, r}^T = \boldsymbol{K}_r$ once. The reduced static displacement is evaluated via a single back-substitution:
$$\boldsymbol{u}_r = \boldsymbol{L}_{K, r}^{-T} \boldsymbol{L}_{K, r}^{-1} \boldsymbol{f}_r$$
executing in $< 1\ \mu\text{s}$ ($> 1{,}000{,}000\text{ Hz}$), implemented in [`ReducedLinearStaticSolver`](file:///home/tanadul/Workspace/project-sierra/psierra/src/psierra/reduction/solver.py).

---

### 1.6 Real-Time Surface Sub-Basis Reconstruction

In interactive visualization (e.g. PyVista / VTK), reconstructing all $n$ volumetric degrees of freedom every frame reintroduces an unnecessary $\mathcal{O}(N)$ computational overhead.

Because rendering requires only the outer boundary facets, we pre-slice the basis matrix along the rows corresponding to surface degrees of freedom:
$$\boldsymbol{\Phi}_{\text{surf}} = \boldsymbol{\Phi}[\text{surface\_dofs}, :] \in \mathbb{R}^{n_{\text{surf}} \times r} \quad (n_{\text{surf}} = 3 N_{\text{surf\_nodes}} \ll n)$$

Online 3D surface mesh rendering is evaluated via a single dense matrix-vector product:
$$\boldsymbol{u}_{\text{surf}} = \boldsymbol{\Phi}_{\text{surf}} \boldsymbol{u}_r \in \mathbb{R}^{n_{\text{surf}}}$$
which reshapes directly into $\mathbb{R}^{N_{\text{surf\_nodes}} \times 3}$ for in-place buffer mutation (`polydata.points = initial_points + u_surf`) in microseconds.

---

## 2. Computational & Algorithmic Notes

### 2.1 Offline-Online Decoupling

The architecture enforces strict decoupling between high-cost offline preparation and real-time online execution:

```
[ OFFLINE STAGE: Compute Once & Cache ]
  1. FOM Simulation: Harvest snapshots S = [u1, ..., um]
  2. Snapshot Caching: Auto-save/load to .npz archive
  3. Thin SVD: Compute U, Sigma and truncate to Phi (r modes)
  4. Operator Projection: Kr, Mr, Cr = Phi^T K, M, C Phi
  5. Dense Cholesky: Factorize K_eff_r = L_r L_r^T
  6. Surface Slicing: Extract Phi_surf = Phi[surface_dofs, :]
                       │
                       ▼  (Hand off to real-time loop)
[ ONLINE STAGE: Real-Time Execution (>20,000 Hz) ]
  1. Compute f_eff = fr + Mr m_part + Cr c_part - alpha_f Kr ur  [O(r^2)]
  2. Cholesky Back-Solve: ur_next = cho_solve(L_r, f_eff)        [O(r^2)]
  3. Update kinematics: vr_next, ar_next                          [O(r)]
  4. Render Surface: u_surf = Phi_surf @ ur                       [O(r * N_surf)]
```

### 2.2 Compressed `.npz` Snapshot Auto-Caching

`SnapshotManager.get_or_compute(cache_path, compute_fn)` automatically manages snapshot persistence:

* On first run: runs `compute_fn` to generate snapshots from the full-order model and compresses them to `.npz`.
* On subsequent runs: verifies array shapes, detects cache hits, and loads in $< 10\text{ ms}$, bypassing the training simulation entirely.

---

## 3. Code Implementation Map

| Mathematical Concept | Mathematical Formulation | Source File | Class / Method |
| :--- | :--- | :--- | :--- |
| **Snapshot Collection** | $\boldsymbol{S} = [\boldsymbol{u}_1, \dots, \boldsymbol{u}_m]$ | [`psierra/reduction/snapshots.py`](file:///home/tanadul/Workspace/project-sierra/psierra/src/psierra/reduction/snapshots.py) | `SnapshotManager.add_snapshot()` |
| **`.npz` Auto-Caching** | $\texttt{np.savez\_compressed} \longleftrightarrow \texttt{np.load}$ | [`psierra/reduction/snapshots.py`](file:///home/tanadul/Workspace/project-sierra/psierra/src/psierra/reduction/snapshots.py) | `SnapshotManager.get_or_compute()` |
| **Thin SVD & Truncation** | $\boldsymbol{S} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T, \quad E(r) \ge \eta$ | [`psierra/reduction/snapshots.py`](file:///home/tanadul/Workspace/project-sierra/psierra/src/psierra/reduction/snapshots.py) | `SnapshotManager.compute_basis()` |
| **POD Basis Container** | $\boldsymbol{\Phi} \in \mathbb{R}^{n \times r}, \quad \boldsymbol{\Phi}^T \boldsymbol{\Phi} = \boldsymbol{I}_r$ | [`psierra/reduction/basis.py`](file:///home/tanadul/Workspace/project-sierra/psierra/src/psierra/reduction/basis.py) | `PODBasis` |
| **Surface Sub-Basis** | $\boldsymbol{\Phi}_{\text{surf}} = \boldsymbol{\Phi}[\text{surf\_dofs}, :]$ | [`psierra/reduction/basis.py`](file:///home/tanadul/Workspace/project-sierra/psierra/src/psierra/reduction/basis.py) | `SurfacePODBasis.reconstruct_surface_displacements()` |
| **Galerkin Operator Projection** | $\boldsymbol{K}_r = \boldsymbol{\Phi}_f^T \boldsymbol{K}_{ff} \boldsymbol{\Phi}_f, \quad \boldsymbol{M}_r = \boldsymbol{\Phi}_f^T \boldsymbol{M}_{ff} \boldsymbol{\Phi}_f$ | [`psierra/reduction/model.py`](file:///home/tanadul/Workspace/project-sierra/psierra/src/psierra/reduction/model.py) | `ReducedLinearModel.from_domain()` |
| **Dense Cholesky Dynamic Step** | $\boldsymbol{u}_{r, n+1} = \boldsymbol{K}_{\text{eff}, r}^{-1} \boldsymbol{f}_{\text{eff}}$ | [`psierra/reduction/solver.py`](file:///home/tanadul/Workspace/project-sierra/psierra/src/psierra/reduction/solver.py) | `ReducedLinearDynamicSolver.step()` |
| **Dense Cholesky Static Solve** | $\boldsymbol{u}_r = \boldsymbol{K}_r^{-1} \boldsymbol{f}_r$ | [`psierra/reduction/solver.py`](file:///home/tanadul/Workspace/project-sierra/psierra/src/psierra/reduction/solver.py) | `ReducedLinearStaticSolver.solve()` |
| **Initial Dynamic Equilibrium** | $\boldsymbol{M}_r \boldsymbol{a}_{r, 0} = \boldsymbol{f}_r - \boldsymbol{K}_r \boldsymbol{u}_0 - \boldsymbol{C}_r \boldsymbol{v}_0$ | [`psierra/reduction/solver.py`](file:///home/tanadul/Workspace/project-sierra/psierra/src/psierra/reduction/solver.py) | `ReducedLinearDynamicSolver._init_equilibrium_acceleration()` |

---

## 4. Demonstration Examples

### 4.1 Cantilever Beam Dynamic Verification (`0901_reduced_dynamic_beam.py`)

* **Mesh**: 3D continuum cantilever beam (`beam1.vtk`, $771$ DOFs, $791$ elements).
* **Physics**: Gravitational dynamic deflection under Rayleigh damping ($\alpha=0.5, \beta=0.001$).
* **Results**:
  * SVD energy truncation reduces the system from $771$ DOFs down to $r = 2$ modes while capturing $> 99.99\%$ of the strain energy.
  * Relative displacement trajectory error: **$0.07\%$** ($< 2\%$ target).
  * Physics rate: **$> 45{,}000\text{ Hz}$** ($0.02\text{ ms/step}$), yielding a **$> 200\times$ speedup** over full-order sparse solves.

### 4.2 Neurosurgical Brain Shift Telemetry (`0902_reduced_brain_shift.py`)

* **Anatomy**: Parenchyma mesh with $55{,}191$ DOFs ($18{,}397$ nodes, $90{,}586$ elements) clamped at the skull base ($Y < -30\text{ mm}$), embedded with deep tumor lesion ($270$ nodes).
* **Physics**: Intraoperative gravity sag and surgical probe indentation ($0.5\text{ N}$).
* **Results**:
  * System reduced from $55{,}191$ DOFs to $r = 2$ modes ($99.97\%$ energy).
  * Online physics rate: **$> 45{,}000\text{ Hz}$**.
  * Real-time surgical telemetry tracks lesion centroid drift ($\|\Delta \boldsymbol{c}\| = 41.42\text{ mm}$) and maximum cortical surface shift ($53.42\text{ mm}$).
  * Surface sub-basis enables 3D cortical rendering in $< 0.005\text{ ms}$.

---

## 5. Further Reading & References

1. **Finite Element Dynamic Procedures**:
   * K.J. Bathe, *Finite Element Procedures*, Prentice Hall, 2nd ed., Chapter 9 (Direct Integration of Dynamic Systems).
2. **Generalized-$\alpha$ Method**:
   * J. Chung and G.M. Hulbert, "A Time Integration Algorithm for Structural Dynamics with Improved Numerical Dissipation: The Generalized-$\alpha$ Method," *Journal of Applied Mechanics*, vol. 60, no. 2, pp. 371–375, 1993.
3. **Proper Orthogonal Decomposition (POD)**:
   * S. Volkwein, *Proper Orthogonal Decomposition: Theory and Reduced-Order Modelling*, Lecture Notes, University of Konstanz, 2013.
   * P. Benner, S. Gugercin, and K. Willcox, "A Survey of Projection-Based Model Reduction Methods for Parametric Dynamical Systems," *SIAM Review*, vol. 57, no. 4, pp. 483–531, 2015.
