# pSierra — Claude Code instructions

The project rules live in `AGENTS.md` (shared with other agent tools). Follow them.

@AGENTS.md

Shared memory and roadmap: `.agents/OPINION.md` (lessons and decisions) and `.agents/PLAN.md` (active roadmap). Read them when starting a milestone and update them afterwards. They are not imported here because they are long.

**Caution about `.agents/OPINION.md` §9 (contact, Milestones 1.0–1.5):** several claims there were checked in Nov 2026 and found false. Do not treat them as established:
- Mortar momentum balance "< 1e-12": the master receives 7–122% of the slave force on partial facet overlaps (`boundary/mortar.py`, slave forces use `D_active`, master forces use `D_diag`).
- Smoothed normals "fixed": `TriangleContactGeometry.evaluate` with `vertex_normals` still has ~8% gradient error against finite differences (`collision/triangle.py`).
- Friction with augmented Lagrangian: committing `λ_T = f_T` without moving the anchor doubles the stick force each augmentation (`analysis/augmentation.py` + `boundary/friction.py`).
- Thickness buffer, rebound relaxation, multiplier capping, orientation filtering, and the residual-growth line search are workarounds that mask instability, not fixes.

---

## Claude-specific notes

### Commands
- Full test suite: `uv run pytest -n auto`
- One file: `uv run pytest tests/test_<name>.py -v`
- Run an example: `uv run python examples/<group>/<script>.py` (many accept `--help`; interactive ones accept `--headless`)
- Never install with bare `pip`; add dependencies to `pyproject.toml` and use `uv`.

### Package map (actual layout under `src/psierra/`)
`analysis/` (static, dynamic, quasi-static, dynamic relaxation, line search, AL augmentation) ·
`boundary/` (Dirichlet, loads, `contact.py`, `mortar.py`, `friction.py`) ·
`collision/` (shapes, detector, triangle closest-point, ray tracing, CCD, mortar clipping) ·
`contact/` (geometry providers, normal law, assembler, penalty scaling, Hertz) ·
`domain/` · `element/` · `material/` · `mesh/` · `mesher/` · `multibody/` ·
`reduction/` (POD) · `hyperreduction/` (DEIM, ECSW, reduced contact) · `inverse/` (iFEM) ·
`solver/` · `damping/` · `exchange/` (VTK/XDMF output) · `utility/`.

AGENTS.md and the role playbooks mention `psierra/mor/` and `psierra/contact/` for everything contact-related; the real locations are the ones above.

### Reference example
The canonical 6-stage example is `examples/02_linear_static/0201_beam.py`.

### Verification rules (lessons from the contact review)
These apply to every role and override any wording that suggests otherwise:
1. **No unverified claims in docstrings.** Words like "guarantees", "exact", "conserves", "symmetric positive semi-definite" are allowed only when a test in `tests/` checks that property. Otherwise state the limitation.
2. **Finite-difference check every new kernel.** Any new gap/force/energy/tangent must have a test comparing the analytical gradient and Hessian against central differences (relative error < 1e-5 on a smooth branch).
3. **Conservation checks for contact.** Two-body and self-contact forces must pass linear- and angular-momentum balance tests, including partial-overlap and edge/vertex cases.
4. **No example-specific tuning in the library.** Library defaults must be dimensionless or scale with mesh size and material stiffness. Absolute lengths (e.g. `max(5.0, ...)` mm) and flags that exist only to make one showcase run are not allowed. If an example needs special parameters, they go in the example and the reason is written next to them.
5. **Do not hide divergence.** Do not accept residual growth in line searches, zero or clip multipliers, or loosen tolerances to make a run finish. Find and fix the cause, or report that it is unresolved.
6. **Report honestly.** When a run does not converge or a test fails, say so with the output. "Ran to completion" is not the same as "correct".

### Subagents
The four roles are available as Claude Code subagents in `.claude/agents/`: `manager`, `developer`, `tester`, `teacher`. The full four-stage pipeline is optional. For small changes, work directly. For new physics kernels or a bug fix whose correctness matters, use `tester` independently after the implementation so the tests are not written by the code's author.

### Git
Never commit unless the user explicitly asks (see AGENTS.md §2.4).
