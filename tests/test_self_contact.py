"""
Unit & Regression Tests for Self-Contact Formulation.
====================================================
Contains mesh/API smoke checks, projection cases and MultiBodySystem routing.
Guaranteed-active mechanics verification belongs to
test_self_contact_consistency.py; bunny deformations need not activate contact.
"""

from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as spsp

from psierra.mesh.volumetric import VolumetricMesh
from psierra.domain.solid import SolidDomain
from psierra.material import IsotropicElasticParameters, Hookean, NeoHookean
from psierra.element.tet4_linear import Tet4Linear
from psierra.boundary.contact import SelfContact
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint
from psierra.multibody.system import MultiBodySystem
from psierra.analysis.nonlinear_static import NonlinearStaticAnalysis


def make_material(E: float = 1e5, nu: float = 0.3, rho: float = 1e-6) -> Hookean:
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(E, nu, rho)
    return Hookean(params)


@pytest.fixture
def beam_mesh(tmp_path: Path) -> VolumetricMesh:
    """Creates a simple volumetric beam mesh for fast testing."""
    beam_path = Path("data/beam1/mesh/beam1.vtk")
    if beam_path.exists():
        return VolumetricMesh(beam_path)
    # Fallback to bunny if beam not present
    return VolumetricMesh(Path("data/bunny1/mesh/bunny1.vtk"))


@pytest.fixture
def bunny_mesh() -> VolumetricMesh:
    """Loads Stanford Bunny volumetric tetrahedral mesh."""
    bunny_path = Path("data/bunny1/mesh/bunny1.vtk")
    if not bunny_path.exists():
        pytest.skip("bunny1.vtk not generated yet")
    return VolumetricMesh(bunny_path)


@pytest.fixture
def test_domain(beam_mesh: VolumetricMesh) -> SolidDomain:
    mat = make_material(1e5, 0.3, 1e-6)
    elem = Tet4Linear(mat)
    return SolidDomain(beam_mesh, mat, elem)


def test_self_contact_initialization(test_domain: SolidDomain) -> None:
    """Verifies SelfContact initializes properly and builds topological exclusion graph."""
    contact = SelfContact(
        domain=test_domain,
        k_penalty=5e3,
        c_damping=10.0,
        formulation="penalty",
        exclusion_rings=2,
    )

    assert contact.domain is test_domain
    assert contact.k_penalty == 5e3
    assert contact.c_damping == 10.0
    assert contact.formulation == "penalty"
    assert len(contact._excluded_faces) == contact.n_surf_nodes

    # Every surface node must exclude at least its 1-ring faces
    for surf_idx in range(contact.n_surf_nodes):
        excluded = contact._excluded_faces[surf_idx]
        assert len(excluded) > 0


def test_topological_exclusion_prevents_self_detection(test_domain: SolidDomain) -> None:
    """
    The existing beam fixture produces no active pairs with default selectors.
    This mesh-specific smoke check is not a global collision guarantee.
    """
    contact = SelfContact(domain=test_domain, k_penalty=1e4)
    active = contact._find_active_contacts()
    # In a flat beam, there are no opposing folds within search radius
    assert len(active) == 0

    f_c = contact.evaluate_forces()
    assert np.allclose(f_c, 0.0)

    K_c = contact.evaluate_stiffness()
    assert K_c.nnz == 0


def test_ericson_closest_point_triangle() -> None:
    """Tests the exact Voronoi region point-to-triangle projection."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([10.0, 0.0, 0.0])
    c = np.array([0.0, 10.0, 0.0])

    # 1. Interior point
    p_in = np.array([2.0, 2.0, 5.0])
    c_pt, (u, v, w) = SelfContact.closest_point_triangle(p_in, a, b, c)
    assert np.allclose(c_pt, [2.0, 2.0, 0.0])
    assert np.isclose(u + v + w, 1.0)
    assert u > 0 and v > 0 and w > 0

    # 2. Vertex Voronoi region (closest to B)
    p_b = np.array([15.0, -2.0, 1.0])
    c_pt, (u, v, w) = SelfContact.closest_point_triangle(p_b, a, b, c)
    assert np.allclose(c_pt, b)
    assert np.allclose([u, v, w], [0.0, 1.0, 0.0])

    # 3. Edge Voronoi region (closest to edge AB)
    p_ab = np.array([5.0, -3.0, 0.0])
    c_pt, (u, v, w) = SelfContact.closest_point_triangle(p_ab, a, b, c)
    assert np.allclose(c_pt, [5.0, 0.0, 0.0])
    assert np.isclose(u + v + w, 1.0)
    assert np.isclose(w, 0.0)


def test_bunny_self_contact_force_smoke(bunny_mesh: VolumetricMesh) -> None:
    """
    Assemble finite balanced force on a bunny perturbation, possibly inactive.
    """
    mat = make_material(5e4, 0.3, 1e-6)
    elem = Tet4Linear(mat)
    domain = SolidDomain(bunny_mesh, mat, elem)

    contact = SelfContact(
        domain=domain,
        k_penalty=1e4,
        search_radius=30.0,
        normal_threshold=-0.1,
    )

    # Artificially squeeze the bunny ears (nodes with Z > 60) toward each other to induce contact
    nodes = domain.current_nodes.copy()
    ears_mask = nodes[:, 2] > 60.0
    nodes[ears_mask, 0] *= 0.5  # Compress X span
    domain.u = (nodes - domain.mesh.nodes).ravel()

    f_c = contact.evaluate_forces()
    c_info = contact.get_contact_info()

    assert f_c.shape == (domain.n_dofs,)
    assert np.isfinite(f_c).all()
    assert 0 <= c_info["num_active"] <= len(contact.contact_nodes)
    net_force = np.sum(f_c.reshape(-1, 3), axis=0)
    assert np.allclose(net_force, 0.0, atol=1e-8)


def test_bunny_self_contact_tangent_smoke(bunny_mesh: VolumetricMesh) -> None:
    """
    Assemble a finite symmetric bunny contact tangent, possibly the zero matrix.
    """
    mat = make_material(5e4, 0.3, 1e-6)
    elem = Tet4Linear(mat)
    domain = SolidDomain(bunny_mesh, mat, elem)

    contact = SelfContact(
        domain=domain,
        k_penalty=1e4,
        search_radius=30.0,
        normal_threshold=-0.1,
    )

    # Artificially compress ears to trigger contact
    nodes = domain.current_nodes.copy()
    ears_mask = nodes[:, 2] > 60.0
    nodes[ears_mask, 0] *= 0.5
    domain.u = (nodes - domain.mesh.nodes).ravel()

    K_c = contact.evaluate_stiffness()
    c_info = contact.get_contact_info()

    assert K_c.shape == (domain.n_dofs, domain.n_dofs)
    assert np.isfinite(K_c.data).all()
    assert 0 <= c_info["num_active"] <= len(contact.contact_nodes)
    diff = K_c - K_c.T
    assert np.max(np.abs(diff.data), initial=0.0) < 1e-8


def test_bunny_self_contact_multiplier_smoke(bunny_mesh: VolumetricMesh) -> None:
    """Check finite nonnegative bunny multipliers; activation is not guaranteed."""
    mat = make_material(5e4, 0.3, 1e-6)
    elem = Tet4Linear(mat)
    domain = SolidDomain(bunny_mesh, mat, elem)

    contact = SelfContact(
        domain=domain,
        k_penalty=1e4,
        formulation="augmented_lagrangian",
        search_radius=30.0,
        normal_threshold=-0.1,
    )

    nodes = domain.current_nodes.copy()
    ears_mask = nodes[:, 2] > 60.0
    nodes[ears_mask, 0] *= 0.4
    domain.u = (nodes - domain.mesh.nodes).ravel()

    info1 = contact.update_multipliers()
    assert np.isfinite(contact.multipliers).all()
    assert np.all(contact.multipliers >= 0.0)
    if info1["num_active"] > 0:
        assert info1["max_multiplier"] > 0.0
        # Second update with same deformation increases multiplier
        info2 = contact.update_multipliers()
        assert info2["max_multiplier"] >= info1["max_multiplier"]


def test_multibody_system_add_self_contact(bunny_mesh: VolumetricMesh) -> None:
    """Verifies MultiBodySystem correctly registers and routes self-contact."""
    mat = make_material(5e4, 0.3, 1e-6)
    elem = Tet4Linear(mat)
    domain = SolidDomain(bunny_mesh, mat, elem)

    system = MultiBodySystem(bunny=domain)

    # 1. Direct call
    sc1 = system.add_self_contact("bunny", k_penalty=2e4, formulation="augmented_lagrangian")
    assert isinstance(sc1, SelfContact)
    assert sc1.k_penalty == 2e4
    assert sc1.formulation == "augmented_lagrangian"
    assert sc1.dof_offset == 0

    # 2. Routing via add_contact_pair with slave == master
    sc2 = system.add_contact_pair("bunny", "bunny", k_penalty=3e4)
    assert isinstance(sc2, SelfContact)
    assert sc2.k_penalty == 3e4
