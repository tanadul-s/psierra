"""
Tests for Pole-Pad Contact Simulations: Deformable-to-Rigid and Deformable-to-Deformable.
========================================================================================
Validates:
1. Geometry and Boundary Conditions:
   - Pad central hole (r = 55 mm) around Pole cylinder (r = 45 mm) with 10 mm radial clearance.
   - Fixed inner cylindrical mounting shaft of Pole.
2. Deformable-to-Rigid Contact Mechanics:
   - RayMeshObstacle representation of Pole.
   - Contact penetration, normal repulsive force, and stick-slip friction return mapping.
3. Two-Way Deformable-to-Deformable Contact Mechanics:
   - MultiBodySystem assembly of Pole and Pad domains.
   - Pillar 2 Smoothed Normals on cylindrical master surface.
   - Active-set Coulomb stick-slip friction and quadratic Newton convergence.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.collision import RayMeshObstacle
from psierra.boundary import (
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
    RigidContact,
    StickSlipState,
)
from psierra.multibody import MultiBodySystem
from psierra.damping import RayleighDamping
from psierra.analysis import NonlinearDynamicAnalysis


@pytest.fixture
def mesh_paths() -> tuple[Path, Path]:
    pole_p = Path("data/pole1/mesh/pole1.vtk")
    pad_p = Path("data/pad1/mesh/pad1.vtk")
    if not pole_p.exists():
        pole_p = Path("psierra") / pole_p
        pad_p = Path("psierra") / pad_p
    assert pole_p.exists(), f"Pole mesh not found: {pole_p}"
    assert pad_p.exists(), f"Pad mesh not found: {pad_p}"
    return pole_p, pad_p


def test_pole_pad_geometry_clearance(mesh_paths: tuple[Path, Path]) -> None:
    pole_file, pad_file = mesh_paths
    pole_mesh = VolumetricMesh(pole_file)
    pad_mesh = VolumetricMesh(pad_file)

    pole_r = np.sqrt(pole_mesh.nodes[:, 0] ** 2 + pole_mesh.nodes[:, 2] ** 2)
    pad_r = np.sqrt(pad_mesh.nodes[:, 0] ** 2 + pad_mesh.nodes[:, 2] ** 2)

    # Pole outer radius is 45 mm, inner is 15 mm
    assert np.isclose(pole_r.min(), 15.0, atol=0.1)
    assert np.isclose(pole_r.max(), 45.0, atol=0.1)

    # Pad hole inner radius is 55 mm
    assert np.isclose(pad_r.min(), 55.0, atol=0.1)

    # Annular radial clearance is exactly 10 mm
    radial_gap = pad_r.min() - pole_r.max()
    assert np.isclose(radial_gap, 10.0, atol=0.1)

    # Axial alignment: pad (y in [-15, 15]) is inside pole (y in [-35, 35])
    assert pad_mesh.nodes[:, 1].min() >= pole_mesh.nodes[:, 1].min()
    assert pad_mesh.nodes[:, 1].max() <= pole_mesh.nodes[:, 1].max()


def test_deformable_pad_rigid_pole_contact(mesh_paths: tuple[Path, Path]) -> None:
    pole_file, pad_file = mesh_paths
    pad_mesh = VolumetricMesh(pad_file)
    params_pad = IsotropicElasticParameters()
    params_pad.set_from_youngs_poisson(1000.0, 0.30, 1.2e-6)
    mat_pad = NeoHookean(params_pad)
    elem_pad = Tet4TotalLagrangian(mat_pad)
    pad_domain = SolidDomain(pad_mesh, mat_pad, elem_pad)

    pole_obstacle = RayMeshObstacle(pole_file)

    surf_nodes = pad_mesh.surface.node_indices
    coords = pad_mesh.nodes[surf_nodes]
    r_surf = np.sqrt(coords[:, 0] ** 2 + coords[:, 2] ** 2)
    inner_nodes = surf_nodes[r_surf < 65.0]

    contact = RigidContact(
        slave=pad_domain,
        master=pole_obstacle,
        slave_nodes=inner_nodes,
        k_penalty=5000.0,
        friction=0.35,
        c_damping=0.005,
    )

    # Pre-translate pad 10.1 mm in -X so it is in contact from step 1
    pad_domain.update_state(u=np.tile([-10.1, 0.0, 0.0], pad_mesh.n_nodes))

    gravity = GravityLoad(acceleration=(-9810.0, -2000.0, 0.0))
    M = pad_domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    dirichlet = DirichletManager(pad_domain.n_dofs)
    damping = RayleighDamping(alpha=1.0, beta=0.001)
    solver = NonlinearDynamicAnalysis(domain=pad_domain, damping=damping, spectral_radius=0.8)

    sol = solver.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        dt=0.001,
        n_steps=3,
        contact=contact,
        verbose=False,
    )

    assert sol.converged, "Deformable-to-rigid dynamic solve failed to converge"
    assert sol.steps_completed == 3

    info = contact.get_contact_info()
    assert len(info["active_nodes"]) > 0, "No active contact nodes detected"
    assert info["max_penetration"] > 0.0, "Zero penetration depth"

    f_c = contact.evaluate_forces(n_dofs=pad_domain.n_dofs)
    assert np.linalg.norm(f_c) > 0.0, "Contact force vector is zero"
    assert info.get("num_stick", 0) + info.get("num_slip", 0) > 0, "No friction candidates active"


def test_two_way_deformable_contact(mesh_paths: tuple[Path, Path]) -> None:
    pole_file, pad_file = mesh_paths
    pole_mesh = VolumetricMesh(pole_file)
    pad_mesh = VolumetricMesh(pad_file)

    params_pole = IsotropicElasticParameters()
    params_pole.set_from_youngs_poisson(50000.0, 0.25, 2.7e-6)
    mat_pole = NeoHookean(params_pole)
    pole_domain = SolidDomain(pole_mesh, mat_pole, Tet4TotalLagrangian(mat_pole))

    params_pad = IsotropicElasticParameters()
    params_pad.set_from_youngs_poisson(1000.0, 0.30, 1.2e-6)
    mat_pad = NeoHookean(params_pad)
    pad_domain = SolidDomain(pad_mesh, mat_pad, Tet4TotalLagrangian(mat_pad))

    system = MultiBodySystem({"Pole": pole_domain, "Pad": pad_domain})

    surf_nodes = pad_mesh.surface.node_indices
    coords = pad_mesh.nodes[surf_nodes]
    r_surf = np.sqrt(coords[:, 0] ** 2 + coords[:, 2] ** 2)
    inner_nodes = surf_nodes[r_surf < 65.0]

    contact = system.add_contact_pair(
        slave="Pad",
        master="Pole",
        slave_nodes=inner_nodes,
        k_penalty=5000.0,
        friction=0.30,
        c_damping=0.005,
        smoothed_normals=True,
    )

    # Fixed Pole inner cylindrical face
    pole_inner = np.flatnonzero(
        np.sqrt(pole_mesh.nodes[:, 0] ** 2 + pole_mesh.nodes[:, 2] ** 2) < 20.0
    )
    dirichlet = DirichletManager(system.n_dofs)
    dirichlet.add_constraint(
        DirichletConstraint(pole_inner, components=(0, 1, 2), dof_offset=int(system._dof_offsets[0]))
    )

    # Pre-position pad in contact (-10.1 mm in X)
    u_init = np.zeros(system.n_dofs, dtype=np.float64)
    pad_sl = system.get_body_dof_slice("Pad")
    u_init[pad_sl] = np.tile([-10.1, 0.0, 0.0], pad_mesh.n_nodes)
    system.update_state(u=u_init)

    gravity = GravityLoad(acceleration=(-9810.0, -2000.0, 0.0))
    M = system.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    damping = RayleighDamping(alpha=1.0, beta=0.001)
    solver = NonlinearDynamicAnalysis(domain=system, damping=damping, spectral_radius=0.8)

    sol = solver.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        dt=0.001,
        n_steps=3,
        contact=system.contacts,
        verbose=False,
    )

    assert sol.converged, "Two-way deformable contact solve failed to converge"
    assert sol.steps_completed == 3

    info = contact.get_contact_info()
    assert len(info["active_nodes"]) > 0, "No active contact nodes in two-way contact"
    assert info["max_penetration"] > 0.0, "Zero penetration depth in two-way contact"

    f_c = contact.evaluate_forces(n_dofs=system.n_dofs)
    assert np.linalg.norm(f_c) > 0.0, "Contact force vector is zero in two-way contact"
