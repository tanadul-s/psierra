"""Independent conservative node/triangle contact verification.

Finite differences are used only here as an independent reference for the
analytical production tangent. All derivative cases stay on a smooth feature
and active-set branch, with no damping or friction.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyvista as pv
import pytest
import scipy.sparse as sp
from scipy.spatial.transform import Rotation

from psierra.boundary.contact import DeformableContact
from psierra.domain.base import Domain
from psierra.mesh.surface import SurfaceMesh


class GeometryDomain(Domain):
    """Prescribed 3D coordinates isolate contact from constitutive/solver effects."""

    def __init__(self, nodes: np.ndarray, faces: np.ndarray | None = None) -> None:
        self.reference = nodes.copy()
        self.u = np.zeros(nodes.size)
        self.v = np.zeros(nodes.size)
        self.a = np.zeros(nodes.size)
        if faces is not None:
            polydata = pv.PolyData(nodes, np.column_stack((np.full(len(faces), 3), faces)).ravel())
            self.mesh = SimpleNamespace(surface=SurfaceMesh(polydata, node_indices=np.arange(len(nodes))))

    @property
    def n_dofs(self) -> int:
        return self.reference.size

    @property
    def current_nodes(self) -> np.ndarray:
        return self.reference + self.u.reshape(-1, 3)

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

    def assemble_stiffness_matrix(self) -> sp.csr_matrix:
        return sp.csr_matrix((self.n_dofs, self.n_dofs))

    def assemble_mass_matrix(self) -> sp.csr_matrix:
        return sp.eye(self.n_dofs, format="csr")

    def save_state(self, filepath: Path, **kwargs: Any) -> None:
        raise NotImplementedError("Prescribed geometry fixture has no mesh export")


def make_contact(
    point: np.ndarray, vertices: np.ndarray, formulation: str = "penalty",
) -> tuple[DeformableContact, GeometryDomain, GeometryDomain]:
    slave = GeometryDomain(point.reshape(1, 3))
    master = GeometryDomain(vertices, np.array([[0, 1, 2]]))
    contact = DeformableContact(
        slave, master, slave_nodes=[0], slave_offset=0, master_offset=3,
        k_penalty=1000.0, formulation=formulation,
    )
    return contact, slave, master


def set_coordinates(slave: GeometryDomain, master: GeometryDomain, coordinates: np.ndarray) -> None:
    slave.update_state(u=coordinates[:3] - slave.reference.ravel())
    master.update_state(u=coordinates[3:] - master.reference.ravel())


def reference_gap(coordinates: np.ndarray) -> float:
    """Independent dense projection: plane solution plus three segment candidates."""
    point, first, second, third = coordinates.reshape(4, 3)
    vertices = np.array([first, second, third])
    basis = np.column_stack((second - first, third - first))
    uv = np.linalg.lstsq(basis, point - first, rcond=None)[0]
    candidates: list[np.ndarray] = []
    if uv.min() >= 0.0 and uv.sum() <= 1.0:
        candidates.append(first + basis @ uv)
    for start, end in ((0, 1), (1, 2), (2, 0)):
        edge = vertices[end] - vertices[start]
        fraction = np.clip((point - vertices[start]) @ edge / (edge @ edge), 0.0, 1.0)
        candidates.append(vertices[start] + fraction * edge)
    closest = min(candidates, key=lambda candidate: float(np.linalg.norm(point - candidate)))
    normal = np.cross(second - first, third - first)
    sign = 1.0 if (point - closest) @ normal >= 0.0 else -1.0
    return sign * float(np.linalg.norm(point - closest))


def reference_energy(coordinates: np.ndarray, multiplier: float = 0.0) -> float:
    gap = reference_gap(coordinates)
    pressure = max(0.0, multiplier - 1000.0 * gap)
    return (pressure**2 - multiplier**2) / 2000.0


@pytest.mark.parametrize("point,expected", [
    ([0.2, 0.3, 0.4], [0.5, 0.2, 0.3]),
    ([0.4, -0.2, -0.1], [0.6, 0.4, 0.0]),
    ([1.0, 0.6, 0.2], [0.0, 0.7, 0.3]),
    ([-0.2, -0.3, -0.1], [1.0, 0.0, 0.0]),
    ([1.5, -0.2, 0.3], [0.0, 1.0, 0.0]),
])
def test_exact_closest_feature_barycentrics(point: list[float], expected: list[float]) -> None:
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    actual = DeformableContact.compute_barycentric(np.array(point), *triangle)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("feature,point", [
    ("face", [0.25, 0.3, -0.1]),
    ("edge", [0.4, -0.2, -0.15]),
    ("vertex", [-0.2, -0.3, -0.15]),
])
@pytest.mark.parametrize("transform", ["identity", "rotation", "deformation"])
@pytest.mark.parametrize("formulation", ["penalty", "augmented_lagrangian"])
def test_conservative_contact_energy_force_tangent_and_momentum(
    feature: str, point: list[float], transform: str, formulation: str,
) -> None:
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    coordinates = np.vstack((np.array(point), triangle))
    if transform == "rotation":
        matrix = Rotation.from_rotvec([0.4, -0.7, 0.3]).as_matrix()
        coordinates = coordinates @ matrix.T + [0.7, -0.4, 0.2]
    elif transform == "deformation":
        matrix = np.array([[1.2, 0.15, -0.1], [0.0, 0.85, 0.1], [0.2, -0.1, 1.1]])
        coordinates = coordinates @ matrix.T
    contact, slave, master = make_contact(coordinates[0], coordinates[1:], formulation)
    multiplier = 30.0 if formulation == "augmented_lagrangian" else 0.0
    contact.multipliers[:] = multiplier
    q = coordinates.ravel()
    force = contact.evaluate_forces(n_dofs=12)
    stiffness = contact.evaluate_stiffness(n_dofs=12).toarray()
    info = contact.get_contact_info()
    assert info["num_active"] == 1
    assert np.linalg.norm(force) > 0.0
    assert info["max_penetration"] == pytest.approx(-reference_gap(q), abs=1e-12)
    assert contact.evaluate_potential_energy() == pytest.approx(reference_energy(q, multiplier), rel=1e-12)

    h = 1e-6
    expected_force = np.zeros(12)
    force_jacobian = np.zeros((12, 12))
    for dof in range(12):
        increment = np.zeros(12)
        increment[dof] = h
        expected_force[dof] = -(reference_energy(q + increment, multiplier) - reference_energy(q - increment, multiplier)) / (2.0 * h)
        set_coordinates(slave, master, q + increment)
        positive = contact.evaluate_forces(n_dofs=12)
        set_coordinates(slave, master, q - increment)
        negative = contact.evaluate_forces(n_dofs=12)
        force_jacobian[:, dof] = (positive - negative) / (2.0 * h)
    set_coordinates(slave, master, q)
    assert np.linalg.norm(force - expected_force) / np.linalg.norm(expected_force) < 1e-5
    assert np.linalg.norm(stiffness + force_jacobian) / np.linalg.norm(force_jacobian) < 1e-5
    np.testing.assert_allclose(stiffness, stiffness.T, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(force.reshape(4, 3).sum(axis=0), 0.0, atol=1e-10)
    moment = np.cross(coordinates, force.reshape(4, 3)).sum(axis=0)
    np.testing.assert_allclose(moment, 0.0, atol=1e-10)


def test_contact_tangent_has_real_negative_geometric_curvature() -> None:
    """Stable energy finite differences disprove an unconditional PSD requirement."""
    coordinates = np.array([[0.25, 0.3, -0.1], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    contact, _, _ = make_contact(coordinates[0], coordinates[1:])
    stiffness = contact.evaluate_stiffness(n_dofs=12).toarray()
    values, vectors = np.linalg.eigh(stiffness)
    assert values[0] < -100.0
    direction = vectors[:, 0]
    q = coordinates.ravel()
    for h in (1e-3, 1e-4):
        curvature = (reference_energy(q + h * direction) - 2.0 * reference_energy(q) + reference_energy(q - h * direction)) / h**2
        assert curvature < -100.0
        assert curvature == pytest.approx(values[0], rel=1e-5, abs=1e-5)


def test_closed_tetra_face_uses_piecewise_planar_normal_and_balances_moment() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    master = GeometryDomain(vertices, faces)
    slave = GeometryDomain(np.array([[0.25, 0.3, 0.05]]))
    contact = DeformableContact(slave, master, slave_nodes=[0], master_offset=3, k_penalty=1000.0)
    force = contact.evaluate_forces(n_dofs=15)
    np.testing.assert_allclose(force[:3], [0.0, 0.0, -50.0], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(force.reshape(-1, 3).sum(axis=0), 0.0, atol=1e-10)
    positions = np.vstack((slave.current_nodes, master.current_nodes))
    np.testing.assert_allclose(np.cross(positions, force.reshape(-1, 3)).sum(axis=0), 0.0, atol=1e-10)
    assert contact.evaluate_potential_energy() == pytest.approx(1.25, rel=1e-12)


@pytest.mark.parametrize("point", [[0.25, 0.3, -0.1], [0.4, -0.2, -0.15], [-0.2, -0.3, -0.15]])
def test_contact_force_and_tangent_rotate_with_prescribed_geometry(point: list[float]) -> None:
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    coordinates = np.vstack((point, triangle))
    contact, slave, master = make_contact(coordinates[0], coordinates[1:])
    original_force = contact.evaluate_forces(n_dofs=12)
    original_stiffness = contact.evaluate_stiffness(n_dofs=12).toarray()
    assert np.linalg.norm(original_force) > 0.0
    rotation = Rotation.from_rotvec([-0.5, 0.3, 0.7]).as_matrix()
    transformed = coordinates @ rotation.T + [-2.0, 0.6, 1.5]
    set_coordinates(slave, master, transformed.ravel())
    blocks = np.kron(np.eye(4), rotation)
    np.testing.assert_allclose(contact.evaluate_forces(n_dofs=12), blocks @ original_force, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(contact.evaluate_stiffness(n_dofs=12).toarray(), blocks @ original_stiffness @ blocks.T, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("point", [[0.25, 0.3, -0.05], [0.6, 0.6, -0.1], [-0.1, -0.2, -0.15]])
def test_closed_tetra_exterior_face_edge_vertex_do_not_attract(point: list[float]) -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    master = GeometryDomain(vertices, faces)
    slave = GeometryDomain(np.array([point]))
    contact = DeformableContact(slave, master, slave_nodes=[0], master_offset=3, k_penalty=1000.0)
    np.testing.assert_array_equal(contact.evaluate_forces(n_dofs=15), 0.0)
    assert contact.get_contact_info()["num_active"] == 0
    assert contact.evaluate_potential_energy() == 0.0


@pytest.mark.parametrize("gap,pressure,energy", [(0.01, 20.0, -0.25), (0.05, 0.0, -0.45)])
def test_fixed_multiplier_potential_and_update_use_same_positive_gap(
    gap: float, pressure: float, energy: float,
) -> None:
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    contact, _, _ = make_contact(np.array([0.25, 0.3, gap]), triangle, "augmented_lagrangian")
    contact.multipliers[:] = 30.0
    np.testing.assert_allclose(contact.evaluate_forces(n_dofs=12)[:3], [0.0, 0.0, pressure], rtol=0.0, atol=1e-12)
    assert contact.evaluate_potential_energy() == pytest.approx(energy, abs=1e-12)
    assert contact.get_contact_info()["num_active"] == (1 if pressure else 0)
    info = contact.update_multipliers()
    np.testing.assert_allclose(contact.multipliers, pressure, atol=1e-12)
    assert info["max_multiplier"] == pytest.approx(pressure, abs=1e-12)


def test_active_multiplier_at_zero_face_gap_has_consistent_geometric_tangent() -> None:
    coordinates = np.array([[0.25, 0.3, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    contact, slave, master = make_contact(coordinates[0], coordinates[1:], "augmented_lagrangian")
    contact.multipliers[:] = 30.0
    q = coordinates.ravel()
    force = contact.evaluate_forces(n_dofs=12)
    np.testing.assert_allclose(force[:3], [0.0, 0.0, 30.0], atol=1e-12)
    stiffness = contact.evaluate_stiffness(n_dofs=12).toarray()
    jacobian = np.zeros((12, 12))
    h = 1e-6
    for dof in range(12):
        increment = np.zeros(12)
        increment[dof] = h
        set_coordinates(slave, master, q + increment)
        positive = contact.evaluate_forces(n_dofs=12)
        set_coordinates(slave, master, q - increment)
        negative = contact.evaluate_forces(n_dofs=12)
        jacobian[:, dof] = (positive - negative) / (2.0 * h)
    assert np.linalg.norm(stiffness + jacobian) / np.linalg.norm(jacobian) < 1e-5


@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_geometric_tolerances_preserve_scaled_triangle_contact(scale: float) -> None:
    coordinates = scale * np.array([[0.25, 0.3, -0.1], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    contact, _, _ = make_contact(coordinates[0], coordinates[1:])
    force = contact.evaluate_forces(n_dofs=12)
    np.testing.assert_allclose(force[:3], [0.0, 0.0, 100.0 * scale], rtol=1e-12, atol=1e-12 * scale)
    assert contact.get_contact_info()["max_penetration"] == pytest.approx(0.1 * scale, rel=1e-12)
    assert contact.evaluate_potential_energy() == pytest.approx(5.0 * scale**2, rel=1e-12)


@pytest.mark.parametrize("point", [
    [0.25, 0.3, 0.1], [0.4, -0.2, 0.15], [-0.2, -0.3, 0.15],
    [1.1, 0.1, 0.05],
])
def test_separated_open_triangle_features_have_no_false_contact(point: list[float]) -> None:
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    contact, _, _ = make_contact(np.array(point), triangle)
    np.testing.assert_array_equal(contact.evaluate_forces(n_dofs=12), 0.0)
    assert contact.evaluate_stiffness(n_dofs=12).nnz == 0
    assert contact.get_contact_info()["num_active"] == 0
    assert contact.evaluate_potential_energy() == 0.0


@pytest.mark.parametrize("point", [[0.2, 0.3, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
def test_zero_distance_features_return_finite_selected_tangent(point: list[float]) -> None:
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    contact, _, _ = make_contact(np.array(point), triangle)
    assert np.isfinite(contact.evaluate_forces(n_dofs=12)).all()
    assert np.isfinite(contact.evaluate_stiffness(n_dofs=12).data).all()
    assert np.isfinite(contact.evaluate_potential_energy())


def test_degenerate_master_triangle_rejected_clearly() -> None:
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="[Dd]egenerate|[Aa]rea|[Cc]ollinear"):
        contact, _, _ = make_contact(np.array([0.2, 0.1, -0.1]), triangle)
        contact.evaluate_forces(n_dofs=12)
