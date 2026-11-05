"""Guaranteed-active conservative self-contact and detector-limit verification.

Tiny opposing surface patches belong to one bound 3DOF-per-node body. Exact
mechanics assertions stay on smooth candidate branches; they do not establish
global self-intersection detection or an impact/restitution law.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.spatial.transform import Rotation

from psierra.analysis import NonlinearStaticAnalysis
from psierra.boundary.contact import SelfContact
from psierra.boundary.dirichlet import DirichletManager
from test_deformable_contact_consistency import GeometryDomain, reference_energy, reference_gap


def geometry_domain(nodes: np.ndarray, faces: np.ndarray) -> GeometryDomain:
    domain = GeometryDomain(nodes, faces)
    domain.mesh.nodes = nodes.copy()
    domain.mesh.n_nodes = len(nodes)
    return domain


def make_self_contact(
    point: np.ndarray | None = None, formulation: str = "penalty",
    coordinates: np.ndarray | None = None, **overrides: Any,
) -> tuple[SelfContact, GeometryDomain]:
    point = np.array([0.25, 0.3, -0.1]) if point is None else point
    if coordinates is None:
        coordinates = np.vstack((point, point + [0.0, 0.25, 0.0], point + [0.25, 0.0, 0.0],
                                 [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    domain = geometry_domain(coordinates, np.array([[0, 1, 2], [3, 4, 5]]))
    options: dict[str, Any] = {
        "contact_nodes": [0], "k_penalty": 1000.0, "formulation": formulation,
        "search_radius": 2.0, "contact_thickness": 1.0, "exclusion_rings": 0,
    }
    options.update(overrides)
    return SelfContact(domain, **options), domain


def local_energy(coordinates: np.ndarray, multiplier: float = 0.0) -> float:
    """Independent scalar energy of node0 against the triangle at nodes3,4,5."""
    return reference_energy(np.concatenate((coordinates[:3], coordinates[9:])), multiplier)


@pytest.mark.parametrize("point", [[0.25, 0.3, -0.1], [0.4, -0.2, -0.15], [-0.2, -0.3, -0.15]])
@pytest.mark.parametrize("transform", ["identity", "rotation", "deformation"])
@pytest.mark.parametrize("formulation", ["penalty", "augmented_lagrangian"])
def test_active_self_contact_matches_energy_gradient_and_complete_force_derivative(
    point: list[float], transform: str, formulation: str,
) -> None:
    _, initial = make_self_contact(np.array(point))
    nodes = initial.current_nodes
    if transform == "rotation":
        nodes = nodes @ Rotation.from_rotvec([0.4, -0.7, 0.3]).as_matrix().T + [0.7, -0.4, 0.2]
    elif transform == "deformation":
        nodes = nodes @ np.array([[1.2, 0.15, -0.1], [0.0, 0.85, 0.1], [0.2, -0.1, 1.1]]).T
    contact, domain = make_self_contact(formulation=formulation, coordinates=nodes)
    multiplier = 30.0 if formulation == "augmented_lagrangian" else 0.0
    contact.set_multipliers(np.array([multiplier]))
    q = nodes.ravel()
    gap = reference_gap(np.concatenate((q[:3], q[9:])))
    info = contact.get_contact_info()
    assert info["num_active"] == 1
    assert info["active_nodes"].tolist() == [0]
    np.testing.assert_allclose(contact.evaluate_gaps(), [gap], rtol=1e-12, atol=1e-12)
    assert contact.evaluate_potential_energy() == pytest.approx(local_energy(q, multiplier), rel=1e-12)
    force = contact.evaluate_forces(n_dofs=18)
    stiffness = contact.evaluate_stiffness(n_dofs=18).toarray()
    assert np.linalg.norm(force) > 0.0
    gradient = np.zeros(18)
    jacobian = np.zeros((18, 18))
    h = 1e-6
    for index in range(18):
        increment = np.zeros(18)
        increment[index] = h
        gradient[index] = (local_energy(q + increment, multiplier) - local_energy(q - increment, multiplier)) / (2.0 * h)
        domain.update_state(u=increment)
        positive = contact.evaluate_forces(n_dofs=18)
        domain.update_state(u=-increment)
        negative = contact.evaluate_forces(n_dofs=18)
        jacobian[:, index] = (positive - negative) / (2.0 * h)
    domain.update_state(u=np.zeros(18))
    assert np.linalg.norm(force + gradient) / np.linalg.norm(gradient) <= 1e-6
    assert np.linalg.norm(stiffness + jacobian) / np.linalg.norm(jacobian) <= 1e-6
    np.testing.assert_allclose(stiffness, stiffness.T, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(force.reshape(6, 3).sum(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.cross(nodes, force.reshape(6, 3)).sum(axis=0), 0.0, atol=1e-10)
    # Extra slave-patch vertices affect only the opposition filter on this branch.
    np.testing.assert_allclose(force[3:9], 0.0, atol=1e-12)
    for component in range(3):
        translation = np.zeros(18)
        translation[component::3] = 1.0
        np.testing.assert_allclose(stiffness @ translation, 0.0, atol=1e-10)


@pytest.mark.parametrize("point,weights", [
    ([0.25, 0.3, -0.1], [0.45, 0.25, 0.3]),
    ([0.4, -0.2, -0.15], [0.6, 0.4, 0.0]),
    ([-0.2, -0.3, -0.15], [1.0, 0.0, 0.0]),
])
def test_self_master_reactions_use_exact_closest_feature_weights(point: list[float], weights: list[float]) -> None:
    contact, _ = make_self_contact(np.array(point))
    force = contact.evaluate_forces(n_dofs=18)
    assert contact.get_contact_info()["num_active"] == 1
    assert np.linalg.norm(force[:3]) > 0.0
    np.testing.assert_allclose(force[9:].reshape(3, 3), -np.array(weights)[:, None] * force[:3], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("point", [[0.25, 0.3, -0.1], [0.4, -0.2, -0.15], [-0.2, -0.3, -0.15]])
def test_prescribed_rigid_motion_rotates_self_force_and_tangent(point: list[float]) -> None:
    contact, domain = make_self_contact(np.array(point))
    nodes = domain.current_nodes
    force = contact.evaluate_forces(n_dofs=18)
    stiffness = contact.evaluate_stiffness(n_dofs=18).toarray()
    assert contact.get_contact_info()["num_active"] == 1
    rotation = Rotation.from_rotvec([-0.5, 0.3, 0.7]).as_matrix()
    domain.update_state(u=(nodes @ rotation.T + [-2.0, 0.6, 1.5] - nodes).ravel())
    blocks = np.kron(np.eye(6), rotation)
    np.testing.assert_allclose(contact.evaluate_forces(n_dofs=18), blocks @ force, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(contact.evaluate_stiffness(n_dofs=18).toarray(), blocks @ stiffness @ blocks.T, rtol=1e-12, atol=1e-9)


def test_piecewise_planar_self_face_does_not_use_smoothed_master_normal() -> None:
    point = np.array([0.25, 0.3, 0.05])
    nodes = np.vstack((point, point + [0.25, 0.0, 0.0], point + [0.0, 0.25, 0.0],
                       [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]))
    faces = np.array([[0, 1, 2], [3, 5, 4], [3, 4, 6], [3, 6, 5], [4, 5, 6]])
    domain = geometry_domain(nodes, faces)
    contact = SelfContact(domain, contact_nodes=[0], k_penalty=1000.0,
                          search_radius=2.0, contact_thickness=1.0, exclusion_rings=0)
    force = contact.evaluate_forces(n_dofs=21)
    assert contact.get_contact_info()["num_active"] == 1
    np.testing.assert_allclose(force[:3], [0.0, 0.0, -50.0], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(np.cross(nodes, force.reshape(7, 3)).sum(axis=0), 0.0, atol=1e-10)
    assert contact.evaluate_potential_energy() == pytest.approx(1.25, rel=1e-12)


@pytest.mark.parametrize("point,multiplier", [([0.25, 0.3, 0.01], 30.0), ([0.4, -0.2, 0.15], 300.0)])
def test_self_al_opening_retains_force_until_projected_release(point: list[float], multiplier: float) -> None:
    contact, domain = make_self_contact(np.array(point), formulation="augmented_lagrangian")
    contact.set_multipliers(np.array([multiplier]))
    q = domain.current_nodes.ravel()
    gap = reference_gap(np.concatenate((q[:3], q[9:])))
    pressure = multiplier - 1000.0 * gap
    assert pressure > 0.0
    assert contact.get_contact_info()["num_active"] == 1
    assert np.linalg.norm(contact.evaluate_forces()[:3]) == pytest.approx(pressure, rel=1e-12)
    assert contact.evaluate_potential_energy() == pytest.approx(local_energy(q, multiplier), rel=1e-12)
    snapshot = contact.snapshot_multipliers()
    contact.update_multipliers()
    np.testing.assert_allclose(contact.multipliers, [pressure], rtol=1e-12)
    contact.set_multipliers(snapshot)
    np.testing.assert_allclose(contact.multipliers, [multiplier], rtol=0.0, atol=0.0)
    domain.u.reshape(6, 3)[:3, 0] += 10.0
    np.testing.assert_array_equal(contact.evaluate_gaps(), np.inf)
    np.testing.assert_array_equal(contact.evaluate_forces(), 0.0)
    assert contact.evaluate_potential_energy() == pytest.approx(-multiplier**2 / 2000.0, rel=1e-12)
    contact.update_multipliers()
    np.testing.assert_array_equal(contact.multipliers, 0.0)


def test_local_and_stacked_geometry_have_identical_offset_self_mechanics() -> None:
    contact, domain = make_self_contact(np.array([0.4, -0.2, -0.15]))
    expected_force = contact.evaluate_forces(n_dofs=18)
    expected_stiffness = contact.evaluate_stiffness(n_dofs=18).toarray()
    expected_energy = contact.evaluate_potential_energy()
    assert contact.get_contact_info()["num_active"] == 1
    contact.dof_offset = 6
    global_nodes = np.vstack(([100.0, 100.0, 100.0], [-100.0, -100.0, -100.0], domain.current_nodes))
    expected_global_force = np.concatenate((np.zeros(6), expected_force))
    expected_global_stiffness = np.zeros((24, 24))
    expected_global_stiffness[6:, 6:] = expected_stiffness
    for geometry in (domain.current_nodes, global_nodes):
        np.testing.assert_allclose(contact.evaluate_forces(current_nodes=geometry, n_dofs=24), expected_global_force, atol=1e-12)
        np.testing.assert_allclose(contact.evaluate_stiffness(current_nodes=geometry, n_dofs=24).toarray(), expected_global_stiffness, atol=1e-10)
        assert contact.evaluate_potential_energy(current_nodes=geometry) == pytest.approx(expected_energy, rel=1e-12)


@pytest.mark.parametrize("point", [[0.25, 0.3, 0.1], [0.4, -0.2, 0.15], [-0.2, -0.3, 0.15]])
def test_separated_selected_self_features_have_zero_penalty_force(point: list[float]) -> None:
    contact, _ = make_self_contact(np.array(point))
    assert contact.evaluate_gaps()[0] > 0.0
    assert contact.get_contact_info()["num_active"] == 0
    np.testing.assert_array_equal(contact.evaluate_forces(), 0.0)
    assert contact.evaluate_stiffness().nnz == 0
    assert contact.evaluate_potential_energy() == 0.0


def test_large_triangle_near_interior_remains_candidate_when_vertices_are_far() -> None:
    point = np.array([0.1, 0.2, -0.1])
    master = np.array([[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [0.0, 10.0, 0.0]])
    coordinates = np.vstack((point, point + [0.0, 0.25, 0.0], point + [0.25, 0.0, 0.0], master))
    contact, _ = make_self_contact(coordinates=coordinates, contact_thickness=0.2)
    assert np.min(np.linalg.norm(master - point, axis=1)) > 9.0
    assert contact.get_contact_info()["num_active"] == 1
    np.testing.assert_allclose(contact.evaluate_gaps(), [-0.1], atol=1e-12)
    np.testing.assert_allclose(contact.evaluate_forces()[:3], [0.0, 0.0, 100.0], atol=1e-10)


@pytest.mark.parametrize("cutoff,expected_active", [(0.2, True), (0.15, False)])
def test_diagonal_aabb_distance_preserves_true_vertex_cutoff(cutoff: float, expected_active: bool) -> None:
    contact, _ = make_self_contact(np.array([-0.1, -0.1, -0.1]), contact_thickness=cutoff)
    if expected_active:
        assert contact.get_contact_info()["num_active"] == 1
        np.testing.assert_allclose(contact.evaluate_gaps(), [-np.sqrt(0.03)], atol=1e-12)
        assert np.linalg.norm(contact.evaluate_forces()) > 0.0
    else:
        assert contact.get_contact_info()["num_active"] == 0
        np.testing.assert_array_equal(contact.evaluate_gaps(), np.inf)
        np.testing.assert_array_equal(contact.evaluate_forces(), 0.0)


def test_connected_fold_respects_configured_topological_exclusion() -> None:
    nodes = np.array([[0.25, 0.3, -0.1], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    domain = geometry_domain(nodes, np.array([[0, 2, 1], [1, 2, 3]]))
    included = SelfContact(domain, contact_nodes=[0], k_penalty=1000.0, search_radius=2.0, contact_thickness=1.0, exclusion_rings=0)
    assert included.get_contact_info()["num_active"] == 1
    assert included.evaluate_forces()[2] > 0.0
    excluded = SelfContact(domain, contact_nodes=[0], k_penalty=1000.0, search_radius=2.0, contact_thickness=1.0, exclusion_rings=1)
    np.testing.assert_array_equal(excluded.evaluate_gaps(), np.inf)
    np.testing.assert_array_equal(excluded.evaluate_forces(), 0.0)


def test_known_detector_limit_convex_boundary_node_selects_remote_opposite_wall() -> None:
    """Record a selector limitation, NOT physically correct self-contact.

    A valid undeformed tetrahedron has no self-intersection. Excluding incident
    facets leaves the remote opposite wall with a negative LOCAL signed gap.
    This counterexample forbids claiming global collision correctness here.
    """
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    domain = geometry_domain(nodes, faces)
    contact = SelfContact(domain, contact_nodes=[0], k_penalty=1000.0, search_radius=2.0, contact_thickness=2.0, exclusion_rings=0)
    np.testing.assert_allclose(contact.evaluate_gaps(), [-1.0 / np.sqrt(3.0)], rtol=1e-12)
    assert contact.get_contact_info()["num_active"] == 1


@pytest.mark.parametrize("point", [[0.25, 0.3, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
def test_zero_distance_self_features_have_finite_selected_tangent(point: list[float]) -> None:
    contact, _ = make_self_contact(np.array(point), formulation="augmented_lagrangian")
    contact.set_multipliers(np.array([30.0]))
    assert contact.get_contact_info()["num_active"] == 1
    assert np.isfinite(contact.evaluate_forces()).all()
    assert np.isfinite(contact.evaluate_stiffness().data).all()
    assert np.isfinite(contact.evaluate_potential_energy())


@pytest.mark.parametrize("parameter,value", [
    ("k_penalty", 0.0), ("k_penalty", -1.0), ("k_penalty", np.nan), ("k_penalty", np.inf),
    ("search_radius", 0.0), ("search_radius", np.inf),
    ("contact_thickness", 0.0), ("contact_thickness", np.nan),
    ("c_damping", -1.0), ("friction", np.inf), ("tolerance", -1.0),
    ("formulation", "unsupported"), ("normal_threshold", 1.1), ("normal_threshold", np.nan),
    ("exclusion_rings", -1), ("exclusion_rings", 1.5),
    ("dof_offset", -3), ("dof_offset", 1),
    ("contact_nodes", [0, 0]), ("contact_nodes", [6]), ("contact_nodes", [0.5]),
])
def test_invalid_self_contact_controls_rejected(parameter: str, value: Any) -> None:
    with pytest.raises(ValueError):
        make_self_contact(**{parameter: value})


def test_degenerate_self_master_facet_rejected_clearly() -> None:
    contact, domain = make_self_contact()
    domain.mesh.surface.polydata.points[5] = [2.0, 0.0, 0.0]
    domain.reference[5] = [2.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="[Dd]egenerate|[Aa]rea|[Cc]ollinear"):
        SelfContact(domain, contact_nodes=[0], search_radius=2.0, contact_thickness=1.0, exclusion_rings=0).evaluate_forces()


def test_empty_self_master_surface_rejected_clearly() -> None:
    domain = geometry_domain(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.empty((0, 3), dtype=int))
    with pytest.raises(ValueError):
        SelfContact(domain)


def test_undefined_self_edge_sign_pseudonormal_rejected() -> None:
    _, original = make_self_contact(np.array([0.4, -0.2, -0.15]))
    domain = geometry_domain(original.current_nodes, np.array([[0, 1, 2], [3, 4, 5], [3, 5, 4]]))
    with pytest.raises(ValueError, match="[Nn]ormal|[Ss]ign|[Oo]rient"):
        contact = SelfContact(domain, contact_nodes=[0], search_radius=2.0, contact_thickness=1.0, exclusion_rings=0)
        contact.evaluate_forces()


class SelfSpringDomain(GeometryDomain):
    """Unit spring to the reference position, with all other coordinates fixed."""

    def assemble_stiffness_matrix(self) -> sp.csr_matrix:
        return sp.eye(self.n_dofs, format="csr")

    def assemble_tangent_stiffness_matrix(self) -> sp.csr_matrix:
        return self.assemble_stiffness_matrix()

    def assemble_internal_forces(self) -> np.ndarray:
        return self.u.copy()


@pytest.mark.parametrize("formulation", ["penalty", "augmented_lagrangian"])
def test_self_contact_static_analysis_matches_spring_plane_solution(formulation: str) -> None:
    _, initial = make_self_contact(np.array([0.25, 0.3, 0.1]))
    nodes = initial.current_nodes
    # Long fixed lever arms keep the slave normal opposing even at the largest
    # free Newton trial u_z=-1. Candidate-filter switching would invalidate the
    # analytical spring/plane branch being verified here.
    nodes[1] = nodes[0] + [0.0, 10.0, 0.0]
    nodes[2] = nodes[0] + [10.0, 0.0, 0.0]
    domain = SelfSpringDomain(nodes, np.array([[0, 1, 2], [3, 4, 5]]))
    domain.mesh.nodes = domain.reference.copy()
    domain.mesh.n_nodes = 6
    contact = SelfContact(domain, contact_nodes=[0], k_penalty=10.0, formulation=formulation,
                          search_radius=2.0, contact_thickness=1.0, exclusion_rings=0)
    manager = DirichletManager(18)
    manager.prescribe_dofs([index for index in range(18) if index != 2])
    load = np.zeros(18)
    load[2] = -1.0
    result = NonlinearStaticAnalysis(domain).solve(
        manager, load, contact=contact, n_steps=1, nr_tol=1e-10,
        al_gap_tol=1e-8, al_tol=1e-8, verbose=False,
    )
    assert result.converged
    assert contact.get_contact_info()["num_active"] == 1
    force = contact.evaluate_forces(n_dofs=18)
    assert force[2] > 0.0
    assert abs(load[2] + force[2] - domain.u[2]) < 1e-9
    if formulation == "penalty":
        assert domain.current_nodes[0, 2] == pytest.approx(-0.9 / 11.0, abs=1e-10)
        assert force[2] == pytest.approx(9.0 / 11.0, abs=1e-10)
        np.testing.assert_array_equal(contact.multipliers, 0.0)
        assert result.al_iterations_history == [0]
    else:
        assert domain.current_nodes[0, 2] >= -1e-8
        assert force[2] == pytest.approx(0.9, abs=2e-8)
        pressure = max(0.0, contact.multipliers[0] - 10.0 * contact.evaluate_gaps()[0])
        assert abs(pressure - contact.multipliers[0]) / max(pressure, contact.multipliers[0]) <= 1e-8
        assert result.al_iterations_history[0] > 1
