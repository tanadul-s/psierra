"""Independent spring benchmarks for the batch solver foundation contract.

The two coordinates are a movable support and a mass connected by a spring.
Their energy is k (u_1-u_0)^2 / 2 + c (u_1-u_0)^4 / 4. This
gives exact forces/tangents without depending on the finite element kernels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

from psierra.analysis.linear_dynamic import LinearDynamicAnalysis
from psierra.analysis.nonlinear_dynamic import NonlinearDynamicAnalysis
from psierra.analysis.nonlinear_static import NonlinearStaticAnalysis
from psierra.boundary.dirichlet import DirichletManager
from psierra.domain.base import Domain
from psierra.solver.sparse_linear_solver import DirectSolver


class SpringDomain(Domain):
    """Exact two-coordinate spring; snapshot recorder implements the export API."""

    def __init__(self, k: float = 2.0, cubic: float = 0.0, mass: float = 1.0) -> None:
        self.k = k
        self.cubic = cubic
        self.mass = mass
        self.u = np.zeros(2)
        self.v = np.zeros(2)
        self.a = np.zeros(2)
        self.snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    @property
    def n_dofs(self) -> int:
        return 2

    @property
    def displacement(self) -> np.ndarray:
        return self.u.reshape(2, 1)

    @property
    def velocity(self) -> np.ndarray:
        return self.v.reshape(2, 1)

    @property
    def acceleration(self) -> np.ndarray:
        return self.a.reshape(2, 1)

    @property
    def current_nodes(self) -> np.ndarray:
        return self.displacement

    def assemble_stiffness_matrix(self) -> sp.csr_matrix:
        return sp.csr_matrix(self.k * np.array([[1.0, -1.0], [-1.0, 1.0]]))

    def assemble_tangent_stiffness_matrix(self) -> sp.csr_matrix:
        tangent = self.k + 3.0 * self.cubic * (self.u[1] - self.u[0]) ** 2
        return sp.csr_matrix(tangent * np.array([[1.0, -1.0], [-1.0, 1.0]]))

    def assemble_internal_forces(self) -> np.ndarray:
        extension = self.u[1] - self.u[0]
        force = self.k * extension + self.cubic * extension**3
        return np.array([-force, force])

    def assemble_mass_matrix(self) -> sp.csr_matrix:
        return sp.eye(2, format="csr") * self.mass

    def update_state(
        self, u: np.ndarray | None = None, v: np.ndarray | None = None,
        a: np.ndarray | None = None,
    ) -> None:
        if u is not None:
            self.u = u.copy()
        if v is not None:
            self.v = v.copy()
        if a is not None:
            self.a = a.copy()

    def save_state(self, filepath: Path, **kwargs: Any) -> None:
        self.snapshots.append((self.u.copy(), self.v.copy(), self.a.copy()))
        filepath.write_text("spring benchmark state\n")


class ReversedDirectionSolver(DirectSolver):
    """Inject an ascent direction after a chosen number of successful solves.

    This forces a normal line-search/nonconvergence exit, allowing rollback to
    be checked without singular matrices or relying on a particular iteration.
    """

    def __init__(self, successful_solves: int) -> None:
        self.successful_solves = successful_solves
        self.calls = 0

    def solve(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        self.calls += 1
        direction = super().solve(A, b)
        return direction if self.calls <= self.successful_solves else -direction


class BoundedSpringDomain(SpringDomain):
    """Mimic a constitutive admissibility boundary at displacement .75."""

    def assemble_internal_forces(self) -> np.ndarray:
        if np.max(np.abs(self.u)) > 0.75:
            raise ValueError("Constitutive state outside admissible displacement range")
        return super().assemble_internal_forces()


class NonfiniteDirectionSolver(DirectSolver):
    """Return a nonfinite Newton direction after a valid initial mass solve."""

    def __init__(self, invalid_value: float, successful_solves: int) -> None:
        self.invalid_value = invalid_value
        self.successful_solves = successful_solves
        self.calls = 0

    def solve(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.calls <= self.successful_solves:
            return super().solve(A, b)
        return np.full_like(b, self.invalid_value)


class CubicContact:
    """Conservative cubic opposing force using the existing contact API."""

    def evaluate_forces(
        self, current_nodes: np.ndarray, n_dofs: int,
        velocities: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.array([0.0, -current_nodes[1, 0]**3])

    def evaluate_stiffness(
        self, current_nodes: np.ndarray, n_dofs: int, a1: float = 0.0,
    ) -> sp.csr_matrix:
        return sp.diags([0.0, 3.0 * current_nodes[1, 0]**2], format="csr")


def constraints(support: float = 0.0) -> DirichletManager:
    manager = DirichletManager(2)
    manager.prescribe_dofs([0], support)
    return manager


DYNAMIC_ANALYSES = [LinearDynamicAnalysis, NonlinearDynamicAnalysis]


@pytest.mark.parametrize("analysis_type", DYNAMIC_ANALYSES)
def test_nonzero_support_equilibrium_and_steady_state(
    analysis_type: type[LinearDynamicAnalysis] | type[NonlinearDynamicAnalysis],
) -> None:
    """k(u_f-u_p)=F: a displaced, unloaded equilibrium must remain at rest."""
    domain = SpringDomain()
    domain.update_state(u=np.array([0.0, 0.5]))  # BC must apply before force evaluation.
    result = analysis_type(domain).solve(
        constraints(0.5), np.zeros(2), dt=0.1, n_steps=5,
        steady_state_tol=1e-10, consecutive_hold_steps=1, verbose=False,
    )
    assert result.converged
    assert result.steady_state_converged
    assert result.steps_completed == 1
    np.testing.assert_allclose(domain.u, [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(domain.v, 0.0, atol=1e-12)
    np.testing.assert_allclose(domain.a, 0.0, atol=1e-12)


@pytest.mark.parametrize("analysis_type", DYNAMIC_ANALYSES)
@pytest.mark.parametrize("n_steps", [0, 4])
def test_initial_state_export_and_output_invariance(
    analysis_type: type[LinearDynamicAnalysis] | type[NonlinearDynamicAnalysis],
    n_steps: int, tmp_path: Path,
) -> None:
    states: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for output in (False, True):
        domain = SpringDomain()
        domain.update_state(u=np.array([9.0, 0.8]), v=np.array([8.0, 0.3]), a=np.array([7.0, 6.0]))
        result = analysis_type(domain).solve(
            constraints(0.5), np.array([0.0, 0.2]), dt=0.05, n_steps=n_steps,
            save_directory=tmp_path if output else None, verbose=False,
        )
        assert result.converged
        states.append((domain.u.copy(), domain.v.copy(), domain.a.copy()))
        if output:
            initial = domain.snapshots[0]
            np.testing.assert_allclose(initial[0], [0.5, 0.8], atol=1e-12)
            np.testing.assert_allclose(initial[1], [0.0, 0.3], atol=1e-12)
            # a_f = [F-k(u_f-u_p)]/m = 0.2-2(0.8-0.5) = -0.4.
            np.testing.assert_allclose(initial[2], [0.0, -0.4], atol=1e-12)
            assert (
                (tmp_path / "result.xdmf").exists()
                or (tmp_path / "result.h5").exists()
                or (tmp_path / "result.pvd").exists()
            )
    for without_output, with_output in zip(states[0], states[1], strict=True):
        np.testing.assert_allclose(without_output, with_output, atol=1e-12)


@pytest.mark.parametrize("spectral_radius", [0.0, 0.8, 1.0])
def test_oscillator_second_order_and_cross_solver_agreement(spectral_radius: float) -> None:
    """m=1,k=4,u0=.2,v0=.3: u=.2 cos(2t)+.15 sin(2t)."""
    errors: list[float] = []
    for dt in (0.1, 0.05, 0.025):
        endpoints: list[np.ndarray] = []
        for analysis_type in DYNAMIC_ANALYSES:
            domain = SpringDomain(k=4.0)
            domain.update_state(u=np.array([0.0, 0.2]), v=np.array([0.0, 0.3]))
            analysis = analysis_type(domain, spectral_radius=spectral_radius)
            kwargs = {"nr_tol": 1e-12} if analysis_type is NonlinearDynamicAnalysis else {}
            result = analysis.solve(
                constraints(), np.zeros(2), dt=dt, n_steps=round(1.0 / dt), verbose=False, **kwargs,
            )
            assert result.converged
            endpoints.append(np.array([domain.u[1], domain.v[1]]))
        np.testing.assert_allclose(endpoints[0], endpoints[1], atol=1e-11)
        exact = np.array([0.2 * np.cos(2.0) + 0.15 * np.sin(2.0),
                          -0.4 * np.sin(2.0) + 0.3 * np.cos(2.0)])
        errors.append(float(np.linalg.norm(endpoints[0] - exact)))
    # Asymptotic ratio for a second-order method is 4; allow pre-asymptotic terms.
    assert errors[0] / errors[1] > 3.3
    assert errors[1] / errors[2] > 3.5


@pytest.mark.parametrize("analysis_type", DYNAMIC_ANALYSES)
def test_undamped_midpoint_oscillator_energy(
    analysis_type: type[LinearDynamicAnalysis] | type[NonlinearDynamicAnalysis], tmp_path: Path,
) -> None:
    domain = SpringDomain(k=4.0)
    domain.update_state(u=np.array([0.0, 0.2]), v=np.array([0.0, 0.3]))
    result = analysis_type(domain, spectral_radius=1.0).solve(
        constraints(), np.zeros(2), dt=0.1, n_steps=100, save_directory=tmp_path, verbose=False,
    )
    assert result.converged
    energy = np.array([0.5 * v[1]**2 + 2.0 * u[1]**2 for u, v, _ in domain.snapshots])
    np.testing.assert_allclose(energy, 0.125, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("dynamic", [False, True])
def test_cubic_spring_newton_overshoot_requires_residual_backtracking(dynamic: bool) -> None:
    """The exact u^3 spring tangent is tiny near u=.1; a full step overshoots."""
    domain = SpringDomain(k=0.0, cubic=1.0, mass=0.001)
    domain.update_state(u=np.array([0.0, 0.1]))
    if dynamic:
        result = NonlinearDynamicAnalysis(domain, spectral_radius=1.0).solve(
            constraints(), np.array([0.0, 1.0]), dt=1.0, n_steps=1,
            nr_iterations=10, nr_tol=1e-10, verbose=False,
        )
        # Midpoint inertia + cubic internal force must balance the applied unit force.
        midpoint_u = 0.5 * (0.1 + domain.u[1])
        midpoint_a = 0.5 * (999.0 + domain.a[1])
        assert abs(1.0 - 0.001 * midpoint_a - midpoint_u**3) < 1e-9
    else:
        result = NonlinearStaticAnalysis(domain).solve(
            constraints(), np.array([0.0, 1.0]), n_steps=1,
            nr_iterations=10, nr_tol=1e-10, verbose=False,
        )
        np.testing.assert_allclose(domain.u[1], 1.0, atol=1e-9)
    assert result.converged


@pytest.mark.parametrize("dynamic", [False, True])
def test_backtracking_uses_contact_force_in_complete_residual(dynamic: bool) -> None:
    domain = SpringDomain(k=0.01, mass=0.001)
    domain.update_state(u=np.array([0.0, 0.1]))
    if dynamic:
        result = NonlinearDynamicAnalysis(domain, spectral_radius=1.0).solve(
            constraints(), np.array([0.0, 1.0]), contact=[CubicContact()],
            dt=1.0, n_steps=1, nr_iterations=10, nr_tol=1e-10, verbose=False,
        )
        midpoint_u = 0.5 * (0.1 + domain.u[1])
        midpoint_a = 0.5 * (998.0 + domain.a[1])
        assert abs(1.0 - 0.001 * midpoint_a - 0.01 * midpoint_u - midpoint_u**3) < 1e-9
    else:
        result = NonlinearStaticAnalysis(domain).solve(
            constraints(), np.array([0.0, 1.0]), contact=[CubicContact()],
            n_steps=1, nr_iterations=10, nr_tol=1e-10, verbose=False,
        )
        assert abs(1.0 - 0.01 * domain.u[1] - domain.u[1]**3) < 1e-9
    assert result.converged


def test_dynamic_rejects_invalid_endpoint_even_with_valid_midpoint(tmp_path: Path) -> None:
    domain = BoundedSpringDomain(k=0.0)
    domain.update_state(v=np.array([0.0, 1.0]))
    result = NonlinearDynamicAnalysis(domain, spectral_radius=1.0).solve(
        constraints(), np.zeros(2), dt=1.0, n_steps=1,
        nr_iterations=8, nr_tol=1e-10, save_directory=tmp_path, verbose=False,
    )
    # The unconstrained exact endpoint is u=1, but its midpoint u=.5 is valid.
    assert not result.converged
    assert result.steps_completed == 0
    np.testing.assert_array_equal(domain.u, [0.0, 0.0])
    np.testing.assert_array_equal(domain.v, [0.0, 1.0])
    np.testing.assert_array_equal(domain.a, [0.0, 0.0])
    if (tmp_path / "result.xdmf").exists() or (tmp_path / "result.h5").exists():
        with h5py.File(tmp_path / "result.h5", "r") as h5:
            assert len([k for k in h5["time_series"].keys() if k.startswith("step_")]) == 1
    else:
        assert len(ET.parse(tmp_path / "result.pvd").findall("./Collection/DataSet")) == 1


def test_invalid_prescribed_step_restores_last_accepted_domain_state() -> None:
    domain = BoundedSpringDomain()
    domain.update_state(v=np.array([0.0, 0.3]), a=np.array([0.0, 0.4]))

    def moving_support(step: int, n_steps: int) -> DirichletManager:
        return constraints(0.5 * step)

    with pytest.raises(ValueError, match="admissible"):
        NonlinearStaticAnalysis(domain).solve(
            moving_support, np.zeros(2), n_steps=2,
            nr_iterations=4, nr_tol=1e-10, verbose=False,
        )
    np.testing.assert_allclose(domain.u, [0.5, 0.5], atol=1e-12)
    np.testing.assert_array_equal(domain.v, [0.0, 0.3])
    np.testing.assert_array_equal(domain.a, [0.0, 0.4])


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_nonfinite_newton_direction_is_rejected_without_state_contamination(
    dynamic: bool, invalid_value: float,
) -> None:
    domain = SpringDomain()
    domain.update_state(u=np.array([0.0, 0.2]), v=np.array([0.0, 0.3]), a=np.array([0.0, 0.4]))
    solver = NonfiniteDirectionSolver(invalid_value, 1 if dynamic else 0)
    kwargs: dict[str, Any] = {"n_steps": 1, "verbose": False}
    analysis: NonlinearDynamicAnalysis | NonlinearStaticAnalysis
    if dynamic:
        analysis = NonlinearDynamicAnalysis(domain, linear_solver=solver)
        kwargs["dt"] = 0.1
    else:
        analysis = NonlinearStaticAnalysis(domain, linear_solver=solver)
    result = analysis.solve(constraints(), np.array([0.0, 1.0]), **kwargs)
    assert not result.converged
    assert result.steps_completed == 0
    np.testing.assert_array_equal(domain.u, [0.0, 0.2])
    np.testing.assert_array_equal(domain.v, [0.0, 0.3])
    np.testing.assert_allclose(domain.a, [0.0, 0.6 if dynamic else 0.4], atol=1e-12)


@pytest.mark.parametrize("accepted_steps", [0, 1])
@pytest.mark.parametrize("dynamic", [False, True])
def test_failed_increment_restores_accepted_state_and_finalizes_pvd(
    accepted_steps: int, dynamic: bool, tmp_path: Path,
) -> None:
    domain = SpringDomain()
    initial = (np.array([0.0, 0.2]), np.array([0.0, 0.3]), np.array([0.0, 0.4]))
    domain.update_state(u=initial[0], v=initial[1], a=initial[2])
    load = np.array([0.0, 1.0])
    if dynamic:
        # Initialization consumes one mass solve, each linear Newton step one solve.
        solver = ReversedDirectionSolver(1 + accepted_steps)
        result = NonlinearDynamicAnalysis(domain, linear_solver=solver).solve(
            constraints(), load, dt=0.1, n_steps=2, nr_iterations=4,
            nr_tol=1e-10, save_directory=tmp_path, verbose=False,
        )
        assert result.time[-1] == pytest.approx(accepted_steps * 0.1)
        assert len(result.time) == accepted_steps + 1
    else:
        solver = ReversedDirectionSolver(accepted_steps)
        result = NonlinearStaticAnalysis(domain, linear_solver=solver).solve(
            constraints(), load, n_steps=2, nr_iterations=4,
            nr_tol=1e-10, save_directory=tmp_path, verbose=False,
        )
        np.testing.assert_allclose(result.displacement, domain.u, atol=1e-12)
        accepted_load = (accepted_steps / 2.0) * load
        np.testing.assert_allclose(result.reactions, domain.assemble_internal_forces() - accepted_load, atol=1e-12)
    assert not result.converged
    assert result.steps_completed == accepted_steps
    assert len(domain.snapshots) == accepted_steps + 1
    for actual, accepted in zip((domain.u, domain.v, domain.a), domain.snapshots[-1], strict=True):
        np.testing.assert_allclose(actual, accepted, atol=1e-12)
    if accepted_steps == 0:
        np.testing.assert_allclose(domain.u, initial[0], atol=1e-12)
        np.testing.assert_allclose(domain.v, initial[1], atol=1e-12)
        expected_a = [0.0, 0.6] if dynamic else initial[2]
        np.testing.assert_allclose(domain.a, expected_a, atol=1e-12)
    if (tmp_path / "result.xdmf").exists() or (tmp_path / "result.h5").exists():
        with h5py.File(tmp_path / "result.h5", "r") as h5:
            ts_keys = [k for k in h5["time_series"].keys() if k.startswith("step_")]
            assert len(ts_keys) == accepted_steps + 1
    else:
        datasets = ET.parse(tmp_path / "result.pvd").findall("./Collection/DataSet")
        assert len(datasets) == accepted_steps + 1
        assert [entry.attrib["file"] for entry in datasets] == [path.name for path in result.time_series_files]
        assert all((tmp_path / entry.attrib["file"]).exists() for entry in datasets)


@pytest.mark.parametrize("analysis_type", [LinearDynamicAnalysis, NonlinearDynamicAnalysis, NonlinearStaticAnalysis])
@pytest.mark.parametrize("parameter,value", [
    ("n_steps", -1), ("n_steps", 1.5), ("save_interval", 0),
    ("save_interval", 1.5),
])
def test_invalid_common_parameters_do_not_mutate_state(
    analysis_type: type, parameter: str, value: float,
) -> None:
    domain = SpringDomain()
    domain.update_state(u=np.array([0.0, 0.2]), v=np.array([0.0, 0.3]), a=np.array([0.0, 0.4]))
    initial = (domain.u.copy(), domain.v.copy(), domain.a.copy())
    kwargs: dict[str, Any] = {"n_steps": 1, parameter: value, "verbose": False}
    if analysis_type is not NonlinearStaticAnalysis:
        kwargs["dt"] = 0.1
    with pytest.raises(ValueError):
        analysis_type(domain).solve(constraints(), np.ones(2), **kwargs)
    for actual, expected in zip((domain.u, domain.v, domain.a), initial, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("analysis_type", DYNAMIC_ANALYSES)
def test_zero_optional_tolerances_accept_exact_equilibrium(analysis_type: type) -> None:
    domain = SpringDomain()
    result = analysis_type(domain).solve(
        constraints(), np.zeros(2), dt=0.1, n_steps=1,
        steady_state_tol=0.0, kinetic_energy_tol=0.0,
        consecutive_hold_steps=1, verbose=False,
    )
    assert result.converged
    assert result.steady_state_converged


@pytest.mark.parametrize("analysis_type", DYNAMIC_ANALYSES)
@pytest.mark.parametrize("parameter,value", [
    ("dt", 0.0), ("dt", -0.1), ("dt", np.nan), ("dt", np.inf),
    ("steady_state_tol", -1e-4), ("steady_state_tol", np.nan),
    ("kinetic_energy_tol", -1.0), ("kinetic_energy_tol", np.inf),
    ("consecutive_hold_steps", 0), ("consecutive_hold_steps", 1.5),
])
def test_invalid_dynamic_parameters(
    analysis_type: type, parameter: str, value: float,
) -> None:
    kwargs: dict[str, Any] = {"dt": 0.1, "n_steps": 1, parameter: value, "verbose": False}
    domain = SpringDomain()
    with pytest.raises(ValueError):
        analysis_type(domain).solve(constraints(), np.ones(2), **kwargs)
    np.testing.assert_array_equal(domain.u, 0.0)
    np.testing.assert_array_equal(domain.v, 0.0)
    np.testing.assert_array_equal(domain.a, 0.0)


@pytest.mark.parametrize("analysis_type", [NonlinearStaticAnalysis, NonlinearDynamicAnalysis])
@pytest.mark.parametrize("parameter,value", [
    ("nr_iterations", 0), ("nr_iterations", 1.5), ("nr_tol", 0.0),
    ("nr_tol", -1e-4), ("nr_tol", np.nan), ("nr_tol", np.inf),
])
def test_invalid_newton_parameters(analysis_type: type, parameter: str, value: float) -> None:
    domain = SpringDomain()
    kwargs: dict[str, Any] = {"n_steps": 1, parameter: value, "verbose": False}
    if analysis_type is NonlinearDynamicAnalysis:
        kwargs["dt"] = 0.1
    with pytest.raises(ValueError):
        analysis_type(domain).solve(constraints(), np.ones(2), **kwargs)
    np.testing.assert_array_equal(domain.u, 0.0)
