# Contract: Contact Mechanics v2 (clean rewrite)

Status: **DRAFT — awaiting user approval**
Branch: `contact-rewrite` (no commits unless the user asks)
Backward compatibility: **none**. Old classes, flags, tests, and docs are deleted, not aliased.

---

## 0. Goals and non-goals

**Goals**
1. One mathematically consistent contact model for deformable–deformable and self-contact, robust under large deformation of soft tissue.
2. A simple, correct obstacle contact for rigid tools and the skull.
3. One interface used by every analysis (static, quasi-static, dynamic, dynamic relaxation, reduced model).
4. Every kernel verified by finite differences, momentum balance, and an analytical benchmark before it is used by a solver.
5. No example-specific defaults. All defaults are dimensionless or scale with mesh size and stiffness.

**Non-goals (this contract)**
- Mortar contact. Removed; may be rebuilt later (Popp-style) only if a benchmark needs exact patch-test accuracy.
- Augmented Lagrangian / Uzawa. Removed; the barrier does not need it.
- Generalized-α for contact dynamics. Contact dynamics use Newmark β = 1/4, γ = 1/2 (variational form).
- GPU kernels. Code stays NumPy-vectorized and CuPy-friendly, but no CuPy path in v2.

---

## 1. Formulation

### 1.1 Notation and units
| Symbol | Meaning | Units |
|---|---|---|
| x | current nodal coordinates (contact vertices) | L |
| d | unsigned distance between two primitives | L |
| d̂ | barrier activation distance | L |
| A_k | area weight of contact pair k | L² |
| κ | barrier stiffness | F/L³ |
| E_c | contact potential | F·L |
| f_c = −∇E_c | contact force | F |
| μ | Coulomb coefficient | – |
| ε_u | friction smoothing displacement | L |

### 1.2 Primitive distances (`contact/distance.py`)
Surface = triangles; candidate primitive pairs are:
- **point–triangle (PT)**: vertex p, triangle (t0, t1, t2)
- **edge–edge (EE)**: edges (a0, a1), (b0, b1)

Each pair is classified by its closest features into one of PP, PE, PT (for point–triangle) or PP, PE, EE (for edge–edge), and the distance is evaluated on that branch. All functions work on **squared distance** s = d² (smooth where d is not) and return value, gradient (w.r.t. the 12 local coordinates), and Hessian (12×12).

Edge–edge pairs use the standard parallel-edge mollifier on ‖(a1−a0)×(b1−b0)‖² with threshold ε_× = 10⁻³ ‖a1−a0‖² ‖b1−b0‖² (reference configuration), so the potential stays C¹ when edges become parallel.

The existing face/edge/vertex kernel in `collision/triangle.py` (verified: FD error 3×10⁻¹¹ without smoothed normals) is the starting point for PT and is moved here; the smoothed-normal branch is deleted.

### 1.3 Barrier potential (`contact/barrier.py`)
On squared distance with ŝ = d̂²:

b(s) = −(s − ŝ)² ln(s / ŝ) for 0 < s < ŝ, and 0 for s ≥ ŝ.

b is C² at s = ŝ and → ∞ as s → 0, so surfaces never interpenetrate as long as every step keeps s > 0 (guaranteed by §1.5).

Contact potential: E_c(x) = κ Σ_k A_k m_k(x) b(s_k(x)) / d̂²

- m_k = edge–edge mollifier (1 for PT pairs).
- Division by d̂² makes b dimensionally L² → κ A b/d̂² has units F·L with κ in F/L³.
- **Stiffness:** κ = κ̂ · E_ref / d̂, with dimensionless κ̂ (default 1.0) and E_ref the larger Young's modulus of the two bodies. The resulting contact pressure scale is ≈ κ̂·E_ref, independent of mesh and units.
- **Area weights:** A_v = (1/3) Σ incident triangle areas for a vertex, A_e = (1/3) Σ incident triangle areas for an edge (reference configuration). PT pairs use A_v of the point, EE pairs use A_e of the first edge, each halved because every pair is counted from both sides.
- **Default d̂:** d̂ = ρ_d · h̄, with h̄ the mean reference surface edge length and ρ_d = 0.1 (dimensionless, user-settable).

Hessian: per-pair 12×12 Hessians are projected to positive semi-definite (eigenvalue clamping) before assembly, so the Newton system stays SPD when the bulk tangent is SPD.

### 1.4 Obstacle potential (`contact/obstacle.py`)
Rigid obstacles provide a signed distance φ(x) with exact ∇φ and ∇²φ:
`PlaneObstacle`, `SphereObstacle`, `CapsuleObstacle` (probe: sphere tip + shaft as one capsule), `CylinderObstacle` (infinite or capped), `MeshObstacle` (closest-point Euclidean distance on triangles with angle-weighted pseudonormal sign; requires a closed, consistently oriented surface or an explicit open-surface side).

Two potentials, chosen per obstacle contact:
- `"barrier"` (default for full-order): E = κ Σ_i A_i b(φ_i²)/d̂² for φ_i > 0; requires φ_i > 0 at the start.
- `"penalty"` (for the reduced model and for starts that are already in contact): E = ½ k_n Σ_i A_i ⟨−φ_i⟩², with k_n = κ̂ · E_ref / h̄ (F/L³). Force −∇E = k_n A_i ⟨−φ_i⟩ ∇φ_i. No positive-gap force.

Obstacles may move between steps (`set_pose`), never inside a Newton solve.

### 1.5 Continuous collision detection (`contact/ccd.py`)
`compute_max_step(x, dx) → α ∈ (0, 1]`: largest step such that no primitive pair reaches distance ≤ (1 − η)·d_current along x + t·dx, t ∈ [0, α], with η = 0.9 (keeps 10% of the current distance).

- Surface pairs: additive CCD (Li et al. 2021) for PT and EE, conservative by construction.
- Obstacles: analytic for plane/sphere/capsule/cylinder; conservative advancement using |∇φ| = 1 for `MeshObstacle`.
- Penalty obstacles do not restrict the step.

### 1.6 Friction (`contact/friction.py`)
Lagged, smoothed Coulomb friction (variational):

E_f(x) = μ Σ_k λ_k f₀(‖T_kᵀ (x − x^n)‖)

- λ_k = normal force magnitude of pair k and T_k = its tangent basis, both **frozen** (lagged) from the start of the lagging iteration.
- f₀ is the C¹ smoothed integral of the Coulomb law: f₁(y) = 2y/ε_u − y²/ε_u² for y < ε_u, else 1; f₀' = f₁.
- ε_u = ρ_f · h̄ with ρ_f = 10⁻³ (dimensionless) by default.
- Outer loop: solve with fixed λ, T; update λ, T; repeat until ‖Δλ‖/‖λ‖ < tol_f (default 10⁻³) or max 5 iterations. No stick/slip state machine, no anchors.

---

## 2. Data structures and interfaces

### 2.1 `ContactSurface` (`contact/surface.py`)
```python
@dataclass(frozen=True)
class ContactSurface:
    """Triangulated boundary of one body, mapped into global degrees of freedom."""
    body_index: int
    vertex_ids: np.ndarray      # (n_v,) node indices within the body's mesh
    faces: np.ndarray           # (n_f, 3) indices into vertex_ids, outward winding
    edges: np.ndarray           # (n_e, 2) indices into vertex_ids, unique
    vertex_dofs: np.ndarray     # (n_v, 3) global translational DOF indices
    vertex_areas: np.ndarray    # (n_v,) A_v, reference configuration [L^2]
    edge_areas: np.ndarray      # (n_e,) A_e, reference configuration [L^2]
    mean_edge_length: float     # h̄ [L]

    @classmethod
    def from_domain(cls, domain: Domain, body_index: int = 0, dof_offset: int = 0) -> ContactSurface: ...
```
- Works for `SolidDomain` (3 DOFs/node) and `ShellDomain` (6 DOFs/node; contact uses translations only).
- Rejects degenerate faces and inconsistent orientation with `ValueError`.
- `MultiBodySystem` builds one surface per body with the correct DOF offsets.

### 2.2 `ContactPotential` protocol (`contact/model.py`)
Every contact object implements exactly this; analyses use nothing else.
```python
class ContactPotential(Protocol):
    def update_candidates(self, x: np.ndarray, dx: np.ndarray | None = None) -> None:
        """Rebuild candidate pairs for state x (and the swept path x→x+dx when given). Mutates cache."""
    def compute_energy(self, x: np.ndarray) -> float: ...
    def compute_gradient(self, x: np.ndarray) -> np.ndarray:        # (n_dofs,), f_c = -gradient
        ...
    def compute_hessian(self, x: np.ndarray) -> spsp.csr_matrix:    # (n_dofs, n_dofs), PSD-projected
        ...
    def compute_max_step(self, x: np.ndarray, dx: np.ndarray) -> float: ...
    def compute_min_distance(self, x: np.ndarray) -> float: ...
    def update_lagged_state(self, x: np.ndarray) -> None:           # friction λ, T, x^n
        ...
    def compute_diagnostics(self, x: np.ndarray) -> ContactDiagnostics: ...
```
Rules:
- `compute_*` methods are pure: same x and same cached candidates → identical result, no state change.
- `x` is the global stacked nodal coordinate array `(n_nodes_total, 3)`; DOF mapping comes from `ContactSurface.vertex_dofs`.
- Candidate set is fixed between `update_candidates` calls. Analyses call `update_candidates(x, dx)` once per Newton iteration, before the line search, with the search direction dx, so the candidate set covers every trial state.

### 2.3 Concrete classes
```python
class SurfaceContact:        # deformable–deformable and self-contact, barrier potential
    def __init__(self, surfaces: Sequence[ContactSurface], *, self_contact: bool,
                 dhat_ratio: float = 0.1, stiffness_ratio: float = 1.0,
                 friction: float = 0.0, friction_smoothing_ratio: float = 1e-3,
                 youngs_modulus: float) -> None: ...

class ObstacleContact:       # deformable vs rigid obstacles
    def __init__(self, surface: ContactSurface, obstacles: Sequence[Obstacle], *,
                 potential: Literal["barrier", "penalty"] = "barrier",
                 dhat_ratio: float = 0.1, stiffness_ratio: float = 1.0,
                 friction: float = 0.0, friction_smoothing_ratio: float = 1e-3,
                 youngs_modulus: float) -> None: ...
```
- Self-contact pair exclusion is purely topological: a PT pair is excluded when the point is a vertex of the triangle, an EE pair when the edges share a vertex. No normal thresholds, exclusion rings, reference-configuration filters, or thickness buffers.
- `youngs_modulus` is explicit (E_ref). A helper may read it from a domain's material, but it is never guessed.

### 2.4 Broad phase (`contact/broadphase.py`)
```python
class BroadPhase(Protocol):
    def find_candidates(self, x: np.ndarray, dx: np.ndarray | None, inflation: float) -> CandidatePairs: ...
```
- First implementation: vectorized uniform spatial hash (NumPy), cell size = max(d̂, h̄); handles both static (inflated by d̂) and swept (x → x+dx) boxes.
- Must return a **superset** of all pairs with distance < d̂ (static) or that can come within d̂ along the sweep. Tested against brute force.
- A BVH implementation can replace it later behind the same protocol if profiling shows a need.

### 2.5 Diagnostics
```python
@dataclass(frozen=True)
class ContactDiagnostics:
    n_candidates: int
    n_active: int               # pairs with d < d̂ (barrier) or φ < 0 (penalty)
    min_distance: float         # [L]
    max_penetration: float      # [L], penalty obstacles only (barrier: always 0)
    energy: float               # [F·L]
    total_normal_force: float   # [F]
    total_friction_force: float # [F]
```

---

## 3. Solver integration

### 3.1 Prerequisite: strain energy (decision D1)
An energy line search needs Π_int(u). Currently `SolidDomain.compute_strain_energy` is linear (½uᵀKu) and hyperelastic materials/elements and shells have none.

**Proposal:** add `compute_strain_energy()` to every material (Ψ(F)), every element formulation (Σ_gp Ψ w_gp J₀), and `SolidDomain` / `ShellDomain` / `CoupledSolidShellDomain` / `MultiBodySystem`, each verified by FD (∇Π_int = f_int to < 10⁻⁶ relative).

**Fallback** for any domain without a strain energy: residual-norm backtracking, still capped by `compute_max_step`.

### 3.2 Newton step with contact (all implicit analyses)
Per iteration:
1. Assemble R = f_ext − f_int − ∇E_c(x) (plus inertia in dynamics) and K = K_int + ∇²E_c (+ M/(β Δt²)).
2. Solve K Δu = R.
3. `contact.update_candidates(x, Δx)`.
4. α₀ = min(1, `contact.compute_max_step(x, Δx)`).
5. Backtracking line search from α₀ (factor 0.5, Armijo c₁ = 10⁻⁴) on the total potential Π (or ‖R‖ fallback). **No acceptance of residual growth.** If the line search fails, the step fails (cutback in time/load, then error).
6. Convergence: ‖R_free‖ ≤ tol_r · max(‖f_ext‖, ‖f_int‖, ‖f_c‖) **and** ‖Δu‖_∞ ≤ tol_u · h̄.

Friction lagging wraps the Newton solve (§1.6).

### 3.3 Dynamics (decision D2: Newmark incremental potential)
Contact dynamics use Newmark β = 1/4, γ = 1/2 as a minimization:

Π_n(x) = 1/(2βΔt²) (x − x̃)ᵀ M (x − x̃) + Π_int(x) − f_extᵀ x + E_c(x) + E_f(x),
x̃ = x_n + Δt v_n + Δt² (½ − β) a_n.

`NonlinearDynamicAnalysis` keeps generalized-α for problems **without** contact; with contact it requires β = 1/4, γ = 1/2 and raises `ValueError` otherwise. Contact velocity damping (`c_damping`) is removed; physical damping comes from Rayleigh damping.

### 3.4 Dynamic relaxation
Uses −∇E_c as the contact force. Each pseudo-step is capped by `compute_max_step`. The fictitious mass uses a Gerschgorin bound on K_int + ∇²E_c, recomputed when the candidate set changes.

### 3.5 Reduced model (`hyperreduction/contact.py`, rebuilt)
`ReducedObstacleContact(model, obstacle_contact)` with `potential="penalty"` only:
f_r = −Φ_sᵀ ∇E, K_r = Φ_sᵀ ∇²E Φ_s on the reconstructed surface. No Uzawa, no multipliers.
Example training (Phase 5) must use physical contact snapshots (distributed probe loads or full-order contact solves at many probe positions), never single-node point forces.

---

## 4. Deletions

Deleted when their replacement lands (phase in brackets). No aliases, no compatibility shims.

| Path | Phase |
|---|---|
| `analysis/augmentation.py` | 3 |
| `analysis/line_search.py` (`ContactAwareLineSearch`) — replaced by the line search in §3.2 | 3 |
| AL/rebound/multiplier parameters in all analyses (`al_*`, `rebound_relaxation`, etc.) | 3 |
| `boundary/contact.py`, `boundary/mortar.py`, `boundary/friction.py` | 4 |
| `collision/` (whole package): `triangle.py` → moved into `contact/distance.py`; `shapes.py` → `contact/obstacle.py`; `ccd.py`, `detector.py`, `raytrace.py`, `manifold.py`, `mortar.py` deleted | 4 |
| `contact/{data,law,assembly,geometry,scaling}.py` (old internal layers) | 4 |
| `contact/hertz.py` → moved to `tests/` (or `psierra/verification/`) as analytical reference | 4 |
| `multibody/system.py`: `add_contact_pair`, `add_mortar_contact`, `add_self_contact` → replaced by `add_contact(SurfaceContact | ObstacleContact)` | 4 |
| `hyperreduction/contact.py` (old Uzawa version) | 5 |
| `examples/08_contact/0802_beam_floor_mortar.py` | 4 |
| Old contact tests (35 files, ~14.5k lines; list in Appendix A) | 4 |
| `docs/12_contact_mechanics_contracts_and_baselines.md` → rewritten | 6 |
| `.agents/OPINION.md` §4.1 and §9 → replaced by a short v2 summary | 6 |

`inverse/observation.py` switches its import to `contact/distance.py` in Phase 1.

**Between Phase 3 and Phase 4 the old contact classes no longer work with the analyses.** Examples in `08_contact`, `09_mor/0905–0906` and `11_interactive` are broken on the branch until ported.

---

## 5. Phases and exit gates

Each phase ends with a report and **stops for user review** before the next one starts.

### Phase 1 — Geometry kernels
Build: `distance.py` (PT, EE, all sub-branches, mollifier), `barrier.py`, `obstacle.py` (all obstacle types), the friction smoothing functions f₀/f₁, and the `inverse/observation.py` import.
Gate (tests written by `tester`):
- FD gradient and Hessian vs. analytical, relative error < 10⁻⁶, on randomized configurations covering every branch (PP/PE/PT/EE, parallel edges, near-zero distance).
- Translation and rotation invariance of distances (implies momentum balance of pair forces).
- Barrier: b, b′, b″ continuous at s = ŝ; b → ∞ as s → 0.
- Obstacles: φ, ∇φ, ∇²φ by FD; sign correctness inside/outside, including near mesh edges and vertices.

### Phase 2 — Surfaces, broad phase, CCD, contact models
Build: `surface.py`, `broadphase.py`, `ccd.py`, `model.py` (`SurfaceContact`, `ObstacleContact`), `ContactDiagnostics`.
Gate:
- Broad phase ⊇ brute force on randomized meshes (static and swept).
- CCD: randomized adversarial trajectories; the returned step never yields an intersection or a distance below (1 − η)·d.
- Assembled energy/gradient/Hessian vs. FD on a two-body and a self-contact patch.
- Linear and angular momentum: Σ f_c = 0 and Σ x × f_c = 0 to < 10⁻¹⁰ relative.
- PSD: assembled contact Hessian has no eigenvalue below −10⁻¹⁰·‖H‖.
- `compute_*` purity: repeated calls give bitwise-identical results.

### Phase 3 — Solver integration
Build: strain energy (D1), Newton line search §3.2, Newmark incremental potential §3.3, dynamic relaxation §3.4, friction lagging. Delete `augmentation.py`, `line_search.py`, AL parameters.
Gate:
- ∇Π_int = f_int by FD for every material/element used in examples.
- Hertz sphere-on-block (static): contact radius and force within 5% of analytical as d̂ → 0 (convergence trend shown for 3 values of ρ_d).
- Elastic block dropped on a plane (dynamic, frictionless, no damping): total energy drift < 1% through impact and rebound.
- Block sliding on an incline: sticks for tan θ < μ, slides with the correct acceleration for tan θ > μ.
- No step anywhere accepted with residual growth.

### Phase 4 — Full-order examples and cleanup
Port one at a time, each with default contact parameters: `0805` Hertz → `0801` beam–floor → `0806` stick-slip → `0807` ironing → `0803` cuboid self-contact → `0804` bunny self-contact → brain–skull (`11_interactive` full-order path). Delete everything listed for Phase 4.
Gate: each example converges with default parameters, prints min distance / force / iterations, and has no contact flags without a written reason. Full test suite passes.

### Phase 5 — Reduced model
Rebuild `hyperreduction/contact.py` (§3.5); regenerate training for `0905`, `0906`, `1102`, `1104` with physical probe snapshots; retrain ECSW.
Gate: no mode with peak/99th-percentile surface amplitude > 5 caused by a single node; probe indentation at 3 untrained locations within 10% of the full-order solution in peak displacement and force.

### Phase 6 — Documentation and memory
`teacher` rewrites `docs/12`, updates `docs/05`, `06`, `11`, `README.md`; update `.agents/OPINION.md` and `.agents/PLAN.md`.
Gate: every equation in the docs links to the method that implements it and matches its docstring.

---

## 6. Decisions needed from the user

| ID | Question | Proposal |
|---|---|---|
| D1 | Add nonlinear strain energy to all materials/elements/domains (needed for the energy line search)? | **Yes**, in Phase 3. Residual-norm fallback otherwise. |
| D2 | Contact dynamics with Newmark β = 1/4, γ = 1/2 (generalized-α kept only without contact)? | **Yes** (agreed). |
| D3 | Barrier implemented in-house (not `ipctk`)? | **Yes** (agreed). |
| D4 | Default d̂ = 0.1 · mean surface edge length? Gives a visible gap up to d̂ (≈0.36 mm on the 1102 brain mesh). | **Yes**; smaller for benchmarks needing precision. |
| D5 | Remove contact velocity damping (`c_damping`) entirely? | **Yes**; Rayleigh damping covers physical dissipation. |
| D6 | Move Hertz analytical solutions out of the library into the test/verification code? | **Yes**; they are reference data, not simulation features. |
| D7 | Brain–skull: if the brain and the decimated skull already intersect at the start, use `potential="penalty"` for the skull, or fix the mesh so the barrier can be used? | Check in Phase 2; decide then. |

---

## Appendix A — old tests to delete in Phase 4
All files in `tests/` matching any of: `test_contact_*`, `test_collision_*`, `test_mortar_*`, `test_self_contact*`, `test_deformable_contact*`, `test_friction_*`, `test_pole_pad_contact.py`, `test_analysis_augmentation.py`, `test_analysis_line_search_and_relaxation.py` (relaxation parts re-tested in Phase 3), `test_hyperreduction_contact*.py` and `test_hyperreduction_brain_shift_shell_contact.py` (Phase 5). The exact list is confirmed with the user before deletion.

## References
- Li, M., Ferguson, Z., Schneider, T., et al. (2020). Incremental Potential Contact. *ACM TOG* 39(4).
- Li, M., Kaufman, D. M., Jiang, C. (2021). Codimensional Incremental Potential Contact (additive CCD). *ACM TOG* 40(4).
- Wriggers, P. *Computational Contact Mechanics*, 2nd ed., chs. 4–6, 10.
- Kane, C., Marsden, J., Ortiz, M., West, M. (2000). Variational integrators and the Newmark algorithm. *IJNME* 49.
- Johnson, K. L. *Contact Mechanics*, ch. 4 (Hertz).
