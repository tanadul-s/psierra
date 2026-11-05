"""Independent augmented-Lagrangian enforcement and state-transaction tests.

Spring/plane solutions are analytical; kp is comparable to the bulk/effective
stiffness. Dynamic checks impose endpoint contact and the stated force weighting,
without assuming impact energy conservation or a restitution law.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import scipy.sparse as sp

from psierra.analysis import NonlinearDynamicAnalysis, NonlinearStaticAnalysis
from psierra.boundary.contact import DeformableContact, RigidContact, SelfContact
from psierra.boundary.dirichlet import DirichletManager
from psierra.collision.shapes import RigidPlane
from psierra.domain.base import Domain
from psierra.solver.sparse_linear_solver import DirectSolver
from test_deformable_contact_consistency import GeometryDomain, make_contact


AL_CONTROLS = {"al_max_iterations": 50, "al_gap_tol": 1e-8, "al_tol": 1e-8}


class ContactSpringDomain(Domain):
    """Independent unit springs and unit masses on each nodal coordinate."""

    def __init__(self, n_nodes: int = 1, positive_limit: float = np.inf) -> None:
        self.reference = np.zeros((n_nodes, 3))
        self.reference[:, 0] = np.arange(n_nodes) * 2.0
        self.u = np.zeros(3 * n_nodes)
        self.v = np.zeros(3 * n_nodes)
        self.a = np.zeros(3 * n_nodes)
        self.positive_limit = positive_limit
        self.contacts: list[RigidContact] = []
        self.snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]] = []

    @property
    def n_dofs(self) -> int:
        return self.u.size

    @property
    def current_nodes(self) -> np.ndarray:
        return self.reference + self.u.reshape(-1, 3)

    @property
    def displacement(self) -> np.ndarray:
        return self.u.reshape(-1, 3)

    def assemble_stiffness_matrix(self) -> sp.csr_matrix:
        return sp.eye(self.n_dofs, format="csr")

    def assemble_tangent_stiffness_matrix(self) -> sp.csr_matrix:
        return self.assemble_stiffness_matrix()

    def assemble_mass_matrix(self) -> sp.csr_matrix:
        return sp.eye(self.n_dofs, format="csr")

    def assemble_internal_forces(self) -> np.ndarray:
        if self.u.max() > self.positive_limit:
            raise ValueError("Spring fixture constitutive admissibility limit")
        return self.u.copy()

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
        self.snapshots.append((self.u.copy(), self.v.copy(), self.a.copy(), [contact.multipliers.copy() for contact in self.contacts]))
        filepath.write_text("accepted analytical spring state\n")


class CountingDirectionSolver(DirectSolver):
    """Inject a residual-ascent Newton direction after successful reference work."""

    def __init__(self, budget: int | None = None) -> None:
        self.budget = budget
        self.calls = 0

    def solve(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        self.calls += 1
        direction = super().solve(A, b)
        return -direction if self.budget is not None and self.calls > self.budget else direction


def support(domain: ContactSpringDomain) -> DirichletManager:
    manager = DirichletManager(domain.n_dofs)
    manager.prescribe_dofs(np.array([index for index in range(domain.n_dofs) if index % 3 != 2]))
    return manager


def add_plane(
    domain: ContactSpringDomain, node: int = 0, kp: float = 10.0,
    formulation: str = "augmented_lagrangian",
) -> RigidContact:
    contact = RigidContact(
        domain, RigidPlane(normal=[0.0, 0.0, 1.0]), slave_nodes=[node],
        k_penalty=kp, formulation=formulation,
    )
    domain.contacts.append(contact)
    return contact


def assert_projected_constraint(contact: RigidContact, domain: ContactSpringDomain, node: int = 0) -> None:
    gap = domain.current_nodes[node, 2]
    multiplier = float(contact.multipliers[0])
    pressure = max(0.0, multiplier - contact.k_penalty * gap)
    assert gap >= -AL_CONTROLS["al_gap_tol"]
    scale = max(abs(pressure), abs(multiplier))
    residual = abs(pressure - multiplier) / scale if scale else 0.0
    assert residual <= AL_CONTROLS["al_tol"]


@pytest.mark.parametrize("kp", [1.0, 10.0])
def test_static_augmentation_enforces_contact_and_keeps_final_equilibrium(kp: float) -> None:
    domain = ContactSpringDomain()
    contact = add_plane(domain, kp=kp)
    load = np.array([0.0, 0.0, -1.0])
    result = NonlinearStaticAnalysis(domain).solve(
        support(domain), load, n_steps=1, nr_tol=1e-10, verbose=False, **AL_CONTROLS,
    )
    assert result.converged
    assert result.steps_completed == 1
    assert result.al_iterations_history[0] > 1
    assert_projected_constraint(contact, domain)
    force = contact.evaluate_forces(n_dofs=3)
    assert force[2] > 0.0
    np.testing.assert_allclose(load + force - domain.assemble_internal_forces(), 0.0, atol=1e-9)
    assert force[2] == pytest.approx(1.0, abs=2e-8)
    np.testing.assert_allclose(result.reactions, domain.assemble_internal_forces() - load - force, atol=1e-12)


@pytest.mark.parametrize("gap,expected_pressure", [(-0.01, 0.3), (0.01, 0.1), (0.03, 0.0)])
def test_rigid_multiplier_opening_uses_projected_pressure(gap: float, expected_pressure: float) -> None:
    domain = ContactSpringDomain()
    domain.u[2] = gap
    contact = add_plane(domain)
    contact.multipliers[:] = 0.2
    force = contact.evaluate_forces(n_dofs=3)
    np.testing.assert_allclose(force, [0.0, 0.0, expected_pressure], atol=1e-12)
    expected_energy = (expected_pressure**2 - 0.2**2) / 20.0
    assert contact.evaluate_potential_energy() == pytest.approx(expected_energy, abs=1e-12)
    tangent = contact.evaluate_stiffness(n_dofs=3).toarray()
    assert tangent[2, 2] == pytest.approx(10.0 if expected_pressure else 0.0, abs=1e-12)
    assert contact.get_contact_info()["num_active"] == (1 if expected_pressure else 0)
    contact.update_multipliers()
    np.testing.assert_allclose(contact.multipliers, expected_pressure, atol=1e-12)


def test_static_opening_releases_multiplier_and_reaches_free_equilibrium() -> None:
    domain = ContactSpringDomain()
    contact = add_plane(domain)
    contact.multipliers[:] = 1.0
    result = NonlinearStaticAnalysis(domain).solve(
        support(domain), np.array([0.0, 0.0, 0.1]), n_steps=1,
        nr_tol=1e-10, verbose=False, **AL_CONTROLS,
    )
    assert result.converged
    assert result.al_iterations_history[0] > 1
    np.testing.assert_allclose(domain.u, [0.0, 0.0, 0.1], atol=1e-10)
    np.testing.assert_array_equal(contact.multipliers, 0.0)
    np.testing.assert_array_equal(contact.evaluate_forces(n_dofs=3), 0.0)


def test_multiple_al_contacts_and_mixed_penalty_never_augment_penalty() -> None:
    domain = ContactSpringDomain(n_nodes=3)
    first = add_plane(domain, node=0, kp=10.0)
    second = add_plane(domain, node=1, kp=5.0)
    penalty = add_plane(domain, node=2, kp=10.0, formulation="penalty")
    penalty.multipliers[:] = 7.0
    load = np.zeros(9)
    load[2::3] = [-1.0, -2.0, -3.0]
    result = NonlinearStaticAnalysis(domain).solve(
        support(domain), load, n_steps=1, nr_tol=1e-10, verbose=False, **AL_CONTROLS,
    )
    assert result.converged
    assert_projected_constraint(first, domain, 0)
    assert_projected_constraint(second, domain, 1)
    np.testing.assert_allclose(domain.u[2::3], [0.0, 0.0, -3.0 / 11.0], atol=2e-8)
    np.testing.assert_array_equal(penalty.multipliers, [7.0])
    forces = sum((contact.evaluate_forces(n_dofs=9) for contact in domain.contacts), np.zeros(9))
    assert first.evaluate_forces(n_dofs=9)[2] > 0.0
    assert second.evaluate_forces(n_dofs=9)[5] > 0.0
    np.testing.assert_allclose((load + forces - domain.assemble_internal_forces())[2::3], 0.0, atol=1e-9)


@pytest.mark.parametrize("dynamic", [False, True])
def test_penalty_only_analysis_has_zero_outer_history_and_no_multiplier_changes(dynamic: bool) -> None:
    domain = ContactSpringDomain()
    contact = add_plane(domain, formulation="penalty")
    contact.multipliers[:] = 7.0
    kwargs: dict[str, Any] = {"n_steps": 2, "nr_tol": 1e-10, "verbose": False}
    analysis: NonlinearStaticAnalysis | NonlinearDynamicAnalysis
    if dynamic:
        analysis = NonlinearDynamicAnalysis(domain)
        kwargs["dt"] = 1.0
    else:
        analysis = NonlinearStaticAnalysis(domain)
    result = analysis.solve(support(domain), np.array([0.0, 0.0, -1.0]), **kwargs)
    assert result.converged
    assert result.al_iterations_history == [0, 0]
    np.testing.assert_array_equal(contact.multipliers, [7.0])
    assert contact.evaluate_forces(n_dofs=3)[2] > 0.0


@pytest.mark.parametrize("rho", [0.0, 0.8, 1.0])
def test_dynamic_augmentation_enforces_endpoint_and_weighted_force_equation(rho: float) -> None:
    domain = ContactSpringDomain()
    domain.u[2] = 0.01
    domain.v[2] = -0.1
    contact = add_plane(domain)
    load = np.array([0.0, 0.0, -1.0])
    analysis = NonlinearDynamicAnalysis(domain, spectral_radius=rho)
    result = analysis.solve(support(domain), load, dt=1.0, n_steps=1, nr_tol=1e-10, verbose=False, **AL_CONTROLS)
    assert result.converged
    assert result.al_iterations_history[0] > 1
    assert_projected_constraint(contact, domain)
    beta, gamma, am, af = analysis.beta, analysis.gamma, analysis.alpha_m, analysis.alpha_f
    initial_acceleration = -1.01
    expected_a = -0.01 / beta + 0.1 / beta - (1.0 / (2.0 * beta) - 1.0) * initial_acceleration
    expected_v = -0.1 + (1.0 - gamma) * initial_acceleration + gamma * expected_a
    midpoint_a = (1.0 - am) * expected_a + am * initial_acceleration
    expected_pressure = (1.0 + af * 0.01 + midpoint_a) / (1.0 - af)
    endpoint_force = contact.evaluate_forces(n_dofs=3)[2]
    assert endpoint_force > 0.0
    assert endpoint_force == pytest.approx(expected_pressure, rel=1e-7, abs=1e-7)
    assert domain.a[2] == pytest.approx(expected_a, abs=1e-7)
    assert domain.v[2] == pytest.approx(expected_v, abs=1e-7)
    actual_midpoint_a = (1.0 - am) * domain.a[2] + am * initial_acceleration
    actual_midpoint_u = (1.0 - af) * domain.u[2] + af * 0.01
    assert abs(-1.0 + (1.0 - af) * endpoint_force - actual_midpoint_a - actual_midpoint_u) < 1e-9


@pytest.mark.parametrize("rho", [0.0, 0.8, 1.0])
def test_dynamic_contact_rest_preserves_equilibrium(rho: float) -> None:
    domain = ContactSpringDomain()
    contact = add_plane(domain)
    contact.multipliers[:] = 1.0
    result = NonlinearDynamicAnalysis(domain, spectral_radius=rho).solve(
        support(domain), np.array([0.0, 0.0, -1.0]), dt=0.2, n_steps=3,
        nr_tol=1e-10, verbose=False, **AL_CONTROLS,
    )
    assert result.converged
    np.testing.assert_allclose(domain.u, 0.0, atol=1e-12)
    np.testing.assert_allclose(domain.v, 0.0, atol=1e-12)
    np.testing.assert_allclose(domain.a, 0.0, atol=1e-12)
    np.testing.assert_allclose(contact.multipliers, 1.0, atol=1e-12)
    assert len(result.al_iterations_history) == 3


@pytest.mark.parametrize("rho", [0.0, 0.8, 1.0])
def test_dynamic_opening_freezes_old_endpoint_force_through_augmentation(rho: float) -> None:
    domain = ContactSpringDomain()
    domain.v[2] = 0.8
    contact = add_plane(domain)
    contact.multipliers[:] = 1.0
    analysis = NonlinearDynamicAnalysis(domain, spectral_radius=rho)
    result = analysis.solve(
        support(domain), np.array([0.0, 0.0, -1.0]), dt=1.0, n_steps=1,
        nr_tol=1e-10, verbose=False, **AL_CONTROLS,
    )
    assert result.converged
    beta, am, af = analysis.beta, analysis.alpha_m, analysis.alpha_f
    expected_u = (-1.0 + af * 1.0 + (1.0 - am) * 0.8 / beta) / ((1.0 - af) + (1.0 - am) / beta)
    assert expected_u > 0.0
    assert domain.u[2] == pytest.approx(expected_u, abs=1e-9)
    np.testing.assert_array_equal(contact.multipliers, 0.0)
    np.testing.assert_array_equal(contact.evaluate_forces(n_dofs=3), 0.0)


@pytest.mark.parametrize("rho", [0.0, 0.8, 1.0])
def test_dynamic_mixed_al_endpoint_and_penalty_midpoint_laws(rho: float) -> None:
    domain = ContactSpringDomain(n_nodes=2)
    domain.u[2::3] = [0.01, -0.1]
    domain.v[2] = -0.1
    al_contact = add_plane(domain, node=0)
    penalty = add_plane(domain, node=1, formulation="penalty")
    penalty.multipliers[:] = 7.0
    load = np.zeros(6)
    load[2::3] = -1.0
    analysis = NonlinearDynamicAnalysis(domain, spectral_radius=rho)
    result = analysis.solve(support(domain), load, dt=1.0, n_steps=1, nr_tol=1e-10, verbose=False, **AL_CONTROLS)
    assert result.converged
    assert_projected_constraint(al_contact, domain)
    beta, am, af = analysis.beta, analysis.alpha_m, analysis.alpha_f
    initial_a = 0.1  # -1 - 11*(-.1)
    mass_term = (1.0 - am) * (-0.1 / beta + (1.0 / (2.0 * beta) - 1.0) * initial_a) - am * initial_a
    expected_penalty_u = (-1.0 - af * 11.0 * -0.1 + mass_term) / ((1.0 - af) * 11.0 + (1.0 - am) / beta)
    assert expected_penalty_u < 0.0
    assert domain.u[5] == pytest.approx(expected_penalty_u, abs=1e-9)
    np.testing.assert_array_equal(penalty.multipliers, [7.0])


@pytest.mark.parametrize("accepted_steps", [0, 1])
def test_static_outer_exhaustion_restores_state_multipliers_and_pvd(accepted_steps: int, tmp_path: Path) -> None:
    domain = ContactSpringDomain()
    domain.v[2], domain.a[2] = 0.3, 0.4
    contact = add_plane(domain)
    contact.multipliers[:] = 0.5 if accepted_steps else 0.3
    initial_multiplier = contact.multipliers.copy()
    result = NonlinearStaticAnalysis(domain).solve(
        support(domain), np.array([0.0, 0.0, -1.0]), n_steps=2,
        nr_tol=1e-10, al_max_iterations=1, al_gap_tol=1e-8, al_tol=1e-8,
        save_directory=tmp_path, verbose=False,
    )
    assert not result.converged
    assert result.steps_completed == accepted_steps
    assert len(result.al_iterations_history) == accepted_steps
    np.testing.assert_array_equal(domain.u, 0.0)
    np.testing.assert_array_equal(domain.v, [0.0, 0.0, 0.3])
    np.testing.assert_array_equal(domain.a, [0.0, 0.0, 0.4])
    np.testing.assert_array_equal(contact.multipliers, initial_multiplier)
    assert_accepted_output(domain, result.time_series_files, tmp_path, accepted_steps)


def assert_accepted_output(domain: ContactSpringDomain, files: list[Path], directory: Path, accepted_steps: int) -> None:
    assert len(domain.snapshots) == accepted_steps + 1
    for actual, saved in zip((domain.u, domain.v, domain.a), domain.snapshots[-1][:3], strict=True):
        np.testing.assert_allclose(actual, saved, atol=1e-12)
    for contact, saved in zip(domain.contacts, domain.snapshots[-1][3], strict=True):
        np.testing.assert_array_equal(contact.multipliers, saved)
    datasets = ET.parse(directory / "result.pvd").findall("./Collection/DataSet")
    assert [entry.attrib["file"] for entry in datasets] == [path.name for path in files]
    assert len(datasets) == accepted_steps + 1


def test_dynamic_outer_exhaustion_restores_initial_state_and_multipliers(tmp_path: Path) -> None:
    domain = ContactSpringDomain()
    domain.u[2], domain.v[2] = 0.01, -0.1
    contact = add_plane(domain)
    result = NonlinearDynamicAnalysis(domain).solve(
        support(domain), np.array([0.0, 0.0, -1.0]), dt=1.0, n_steps=1,
        nr_tol=1e-10, al_max_iterations=1, al_gap_tol=1e-8, al_tol=1e-8,
        save_directory=tmp_path, verbose=False,
    )
    assert not result.converged
    assert result.steps_completed == 0
    assert result.al_iterations_history == []
    np.testing.assert_array_equal(domain.u, [0.0, 0.0, 0.01])
    np.testing.assert_array_equal(domain.v, [0.0, 0.0, -0.1])
    np.testing.assert_allclose(domain.a, [0.0, 0.0, -1.01], atol=1e-12)
    np.testing.assert_array_equal(contact.multipliers, 0.0)
    assert_accepted_output(domain, result.time_series_files, tmp_path, 0)


@pytest.mark.parametrize("accepted_steps", [0, 1])
def test_dynamic_inner_failure_restores_last_accepted_contact_transaction(accepted_steps: int, tmp_path: Path) -> None:
    reference = ContactSpringDomain()
    reference.u[2], reference.v[2] = 0.01, -0.1
    reference_contact = add_plane(reference)
    counter = CountingDirectionSolver()
    if accepted_steps:
        result = NonlinearDynamicAnalysis(reference, linear_solver=counter).solve(
            support(reference), np.array([0.0, 0.0, -1.0]), dt=1.0, n_steps=1,
            nr_tol=1e-10, verbose=False, **AL_CONTROLS,
        )
        assert result.converged
        assert reference_contact.multipliers[0] > 0.0
    domain = ContactSpringDomain()
    domain.u[2], domain.v[2] = 0.01, -0.1
    contact = add_plane(domain)
    solver = CountingDirectionSolver(counter.calls if accepted_steps else 1)
    result = NonlinearDynamicAnalysis(domain, linear_solver=solver).solve(
        support(domain), np.array([0.0, 0.0, -1.0]), dt=1.0, n_steps=2,
        nr_tol=1e-10, save_directory=tmp_path, verbose=False, **AL_CONTROLS,
    )
    assert not result.converged
    assert result.steps_completed == accepted_steps
    assert len(result.al_iterations_history) == accepted_steps
    assert result.time[-1] == pytest.approx(float(accepted_steps))
    assert_accepted_output(domain, result.time_series_files, tmp_path, accepted_steps)
    if accepted_steps:
        for actual, expected in zip((domain.u, domain.v, domain.a), (reference.u, reference.v, reference.a), strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-10)
        np.testing.assert_allclose(contact.multipliers, reference_contact.multipliers, atol=1e-10)


def test_static_model_exception_after_accepted_step_restores_contact_transaction(tmp_path: Path) -> None:
    domain = ContactSpringDomain(positive_limit=0.15)
    contact = add_plane(domain)

    def moving_support(step: int, n_steps: int) -> DirichletManager:
        manager = DirichletManager(3)
        manager.prescribe_dofs([0, 1], np.array([0.1 * step, 0.0]))
        return manager

    with pytest.raises(ValueError, match="admissibility"):
        NonlinearStaticAnalysis(domain).solve(
            moving_support, np.array([0.0, 0.0, -1.0]), n_steps=2,
            nr_tol=1e-10, save_directory=tmp_path, verbose=False, **AL_CONTROLS,
        )
    assert len(domain.snapshots) == 2
    assert contact.multipliers[0] > 0.0
    for actual, saved in zip((domain.u, domain.v, domain.a), domain.snapshots[-1][:3], strict=True):
        np.testing.assert_allclose(actual, saved, atol=1e-12)
    np.testing.assert_array_equal(contact.multipliers, domain.snapshots[-1][3][0])
    assert len(ET.parse(tmp_path / "result.pvd").findall("./Collection/DataSet")) == 2


@pytest.mark.parametrize("analysis_type", [NonlinearStaticAnalysis, NonlinearDynamicAnalysis])
@pytest.mark.parametrize("parameter,value", [
    ("al_max_iterations", 0), ("al_max_iterations", 1.5),
    ("al_gap_tol", 0.0), ("al_gap_tol", np.nan), ("al_gap_tol", np.inf),
    ("al_tol", -1.0), ("al_tol", np.nan), ("al_tol", np.inf),
])
def test_invalid_augmentation_controls_precede_all_state_mutation(analysis_type: type, parameter: str, value: float) -> None:
    domain = ContactSpringDomain()
    domain.u[2], domain.v[2], domain.a[2] = 0.1, 0.2, 0.3
    contact = add_plane(domain)
    contact.multipliers[:] = 0.4
    initial = (domain.u.copy(), domain.v.copy(), domain.a.copy())
    kwargs: dict[str, Any] = {"n_steps": 1, "verbose": False, parameter: value}
    if analysis_type is NonlinearDynamicAnalysis:
        kwargs["dt"] = 1.0
    with pytest.raises(ValueError):
        analysis_type(domain).solve(support(domain), np.array([0.0, 0.0, -1.0]), **kwargs)
    for actual, expected in zip((domain.u, domain.v, domain.a), initial, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(contact.multipliers, [0.4])


def make_self_contact() -> tuple[SelfContact, GeometryDomain]:
    nodes = np.array([[0.25, 0.25, 0.25], [0.25, 0.5, 0.25], [0.5, 0.25, 0.25],
                      [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    domain = GeometryDomain(nodes, np.array([[0, 1, 2], [3, 4, 5]]))
    domain.mesh.nodes = domain.reference.copy()
    domain.mesh.n_nodes = len(nodes)
    contact = SelfContact(domain, contact_nodes=[0], k_penalty=10.0,
                          formulation="augmented_lagrangian", search_radius=2.0,
                          contact_thickness=0.5, exclusion_rings=0)
    domain.u.reshape(-1, 3)[:3, 2] = -0.375  # prescribed top face at z=-.125
    return contact, domain


def test_self_contact_changed_geometry_cannot_reuse_a_checksum_collision() -> None:
    contact, domain = make_self_contact()
    force = contact.evaluate_forces()
    assert force[2] > 0.0
    before = domain.current_nodes.copy()
    domain.u.reshape(-1, 3)[:3, 2] += 0.25
    domain.u.reshape(-1, 3)[1, 0] -= 0.75
    after = domain.current_nodes.copy()
    assert before.sum() == after.sum()
    assert before[0, 0] == after[0, 0]
    assert before[-1, -1] == after[-1, -1]
    np.testing.assert_array_equal(contact.evaluate_forces(), 0.0)


def test_self_contact_absent_candidate_releases_stored_multiplier() -> None:
    contact, domain = make_self_contact()
    contact.multipliers[:] = 2.0
    assert contact.evaluate_forces()[2] > 0.0
    domain.u.reshape(-1, 3)[:3, 0] += 10.0
    gaps = contact.evaluate_gaps()
    np.testing.assert_array_equal(gaps, np.inf)
    contact.update_multipliers()
    np.testing.assert_array_equal(contact.multipliers, 0.0)
    np.testing.assert_array_equal(contact.evaluate_forces(), 0.0)


def test_self_contact_stacked_body_coordinates_use_local_body_slice() -> None:
    contact, domain = make_self_contact()
    expected = contact.evaluate_forces(n_dofs=18)
    assert expected[2] > 0.0
    contact.dof_offset = 3
    stacked_nodes = np.vstack(([100.0, 100.0, 100.0], domain.current_nodes))
    actual = contact.evaluate_forces(current_nodes=stacked_nodes, n_dofs=21)
    np.testing.assert_allclose(actual, np.concatenate((np.zeros(3), expected)), atol=1e-12)
    np.testing.assert_allclose(contact.evaluate_gaps(current_nodes=stacked_nodes), [-0.125], atol=1e-12)


@pytest.mark.parametrize("family", ["rigid", "deformable", "self"])
def test_explicit_multiplier_state_copies_and_penalty_updates_are_noop(family: str) -> None:
    if family == "rigid":
        domain = ContactSpringDomain()
        domain.u[2] = -0.125
        contact: RigidContact | DeformableContact | SelfContact = add_plane(domain)
    elif family == "deformable":
        contact, _, _ = make_contact(np.array([0.25, 0.3, -0.125]), np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), "augmented_lagrangian")
    else:
        contact, _ = make_self_contact()
    before = contact.evaluate_forces().copy()
    assert np.linalg.norm(before) > 0.0
    values = np.array([2.0])
    contact.set_multipliers(values)
    values[:] = 99.0
    snapshot = contact.snapshot_multipliers()
    np.testing.assert_array_equal(snapshot, [2.0])
    snapshot[:] = 88.0
    np.testing.assert_array_equal(contact.multipliers, [2.0])
    assert np.linalg.norm(contact.evaluate_forces()) > np.linalg.norm(before)
    contact.formulation = "penalty"
    contact.update_multipliers()
    np.testing.assert_array_equal(contact.multipliers, [2.0])
