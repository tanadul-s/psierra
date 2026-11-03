"""
Unit Tests for Collision Detection and Penalty Contact Mechanics.
=================================================================
Covers analytical collision primitives, Intel Embree BVH ray tracing,
mesh obstacles (floor1.vtk), penalty forces, tangent stiffness linearization,
and static/dynamic contact solvers.
"""

from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as spsp

from psierra.collision import (
    RigidPlane,
    RigidSphere,
    RigidCylinder,
    RayMeshObstacle,
    CollisionDetector,
)
from psierra.boundary.contact import RigidContact
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint
from psierra.boundary.load import GravityLoad
from psierra.material import NeoHookean
from psierra.element import Tet4TotalLagrangian, Tet4Linear
from psierra.mesh.volumetric import VolumetricMesh
from psierra.domain.solid import SolidDomain
from psierra.analysis.nonlinear_static import NonlinearStaticAnalysis
from psierra.analysis.nonlinear_dynamic import NonlinearDynamicAnalysis


# =============================================================================
# 1. Analytical Collision Primitives Tests
# =============================================================================

def test_rigid_plane_queries() -> None:
    plane = RigidPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0])

    pts = np.array([
        [10.0, 5.0, -3.0],   # above plane: gap = +5.0
        [0.0, 0.0, 0.0],     # on plane: gap = 0.0
        [-4.0, -2.5, 8.0],   # penetrating: gap = -2.5
    ])

    gaps, normals, closest = plane.query_points(pts)

    assert np.allclose(gaps, [5.0, 0.0, -2.5])
    assert np.allclose(normals, [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(closest, [[10.0, 0.0, -3.0], [0.0, 0.0, 0.0], [-4.0, 0.0, 8.0]])


def test_rigid_sphere_queries() -> None:
    sphere = RigidSphere(center=[0.0, 10.0, 0.0], radius=5.0)

    pts = np.array([
        [0.0, 20.0, 0.0],  # gap = 10 - 5 = +5.0
        [0.0, 15.0, 0.0],  # on surface: gap = 0.0
        [0.0, 12.0, 0.0],  # inside: gap = 2 - 5 = -3.0
        [0.0, 10.0, 0.0],  # at center: gap = -5.0
    ])

    gaps, normals, closest = sphere.query_points(pts)

    assert np.allclose(gaps, [5.0, 0.0, -3.0, -5.0])
    assert np.allclose(normals[0], [0.0, 1.0, 0.0])
    assert np.allclose(closest[0], [0.0, 15.0, 0.0])
    assert np.allclose(closest[2], [0.0, 15.0, 0.0])
    assert not np.any(np.isnan(normals))


def test_rigid_cylinder_queries() -> None:
    # Cylinder along Z axis through origin with radius 2.0
    cylinder = RigidCylinder(center=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], radius=2.0)

    pts = np.array([
        [5.0, 0.0, 10.0],   # r = 5, gap = 3.0
        [0.0, 2.0, -4.0],   # r = 2, gap = 0.0
        [0.0, 0.5, 3.0],    # r = 0.5, gap = -1.5
    ])

    gaps, normals, closest = cylinder.query_points(pts)

    assert np.allclose(gaps, [3.0, 0.0, -1.5])
    assert np.allclose(normals[0], [1.0, 0.0, 0.0])
    assert np.allclose(normals[1], [0.0, 1.0, 0.0])
    assert np.allclose(closest[0], [2.0, 0.0, 10.0])


# =============================================================================
# 2. Mesh Obstacle & Intel Embree Ray-Tracing Tests
# =============================================================================

def test_ray_mesh_obstacle_floor1() -> None:
    floor_vtk = Path("data/floor1/mesh/floor1.vtk")
    if not floor_vtk.exists():
        pytest.skip("floor1.vtk not found in data directory.")

    obstacle = RayMeshObstacle(floor_vtk)
    assert obstacle.polydata.n_points > 0

    # Floor top surface is at y_max
    y_top = obstacle.polydata.bounds[3]

    # Point above top surface: y_top + 5.0
    pts = np.array([[0.0, y_top + 5.0, 0.0], [0.0, y_top - 1.0, 0.0]])
    gaps, normals, closest = obstacle.query_points(pts)

    # Point at y_top + 5 is 5 units above top facet
    assert np.isclose(gaps[0], 5.0, atol=1e-2)
    assert np.allclose(normals[0], [0.0, 1.0, 0.0], atol=1e-2)
    assert np.isclose(closest[0, 1], y_top, atol=1e-2)

    # Point at y_top - 1 is inside top surface (penetrated by 1 unit)
    assert np.isclose(gaps[1], -1.0, atol=1e-2)

    # Intel Embree Ray Tracing
    origins = np.array([[0.0, y_top + 15.0, 0.0], [1000.0, 1000.0, 1000.0]])
    directions = np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]])
    ray_res = obstacle.ray_intersect(origins, directions)

    assert ray_res["hit"][0] == True
    assert ray_res["hit"][1] == False
    assert np.isclose(ray_res["distances"][0], 15.0, atol=1e-2)
    assert np.allclose(ray_res["hit_points"][0], [0.0, y_top, 0.0], atol=1e-2)


# =============================================================================
# 3. Collision Detector Tests
# =============================================================================

def test_collision_detector_multi_obstacle() -> None:
    plane = RigidPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0])
    sphere = RigidSphere(center=[0.0, 10.0, 0.0], radius=3.0)
    detector = CollisionDetector([plane, sphere])

    pts = np.array([
        [0.0, 5.0, 0.0],    # sphere gap = 5-3=2 (closest to sphere: 2 < 5)
        [0.0, -0.5, 0.0],   # plane gap = -0.5 (penetrates plane)
        [0.0, 9.0, 0.0],    # sphere gap = 1-3 = -2.0 (penetrates sphere)
    ])

    res = detector.query(pts, tolerance=0.0)

    assert res.num_active == 2
    assert set(res.active_indices) == {1, 2}
    assert np.isclose(res.max_penetration, 2.0)
    assert res.obstacle_indices[1] == 0  # hit plane
    assert res.obstacle_indices[2] == 1  # hit sphere


# =============================================================================
# 4. Penalty Contact Mechanics Tests
# =============================================================================

def test_penalty_contact_forces_and_stiffness() -> None:
    plane = RigidPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0])
    kp = 2000.0
    cp = 50.0

    nodes = np.array([
        [0.0, 1.0, 0.0],   # Node 0: above (no contact)
        [0.0, -0.2, 0.0],  # Node 1: penetrated by 0.2 mm
    ])
    vel = np.array([
        0.0, 0.0, 0.0,     # Node 0
        0.0, -10.0, 0.0,   # Node 1: moving down into floor at 10 mm/s
    ])

    contact = RigidContact(
        obstacle=plane,
        slave_nodes=[0, 1],
        k_penalty=kp,
        c_damping=cp,
        dofs_per_node=3,
    )

    # Force evaluation
    f_c = contact.evaluate_forces(current_nodes=nodes, velocities=vel, n_dofs=6)

    # Node 0 has zero force
    assert np.allclose(f_c[:3], 0.0)

    # Node 1 normal repulsive force: -kp * g * n = 2000 * 0.2 = 400 N in +Y
    # Damping force: -cp * vn * n = -50 * (-10) = 500 N in +Y
    # Total Node 1 force = 900 N
    assert np.isclose(f_c[3], 0.0)
    assert np.isclose(f_c[4], 900.0)
    assert np.isclose(f_c[5], 0.0)

    # Stiffness matrix evaluation
    K_c = contact.evaluate_stiffness(current_nodes=nodes, n_dofs=6)

    assert K_c.shape == (6, 6)
    # Symmetry check
    assert np.max(np.abs(K_c - K_c.T)) < 1e-12
    # Node 1 Y-Y diagonal entry is kp
    assert np.isclose(K_c[4, 4], kp)
    assert np.isclose(K_c[3, 3], 0.0)

    # Potential energy: 0.5 * kp * g^2 = 0.5 * 2000 * (0.2)^2 = 40.0 J (N*mm)
    energy = contact.evaluate_potential_energy(current_nodes=nodes)
    assert np.isclose(energy, 40.0)


def test_penalty_contact_linearization_consistency() -> None:
    """Verifies that K_c == -d(f_c)/du via numerical finite differences."""
    normal = np.array([1.0, 2.0, 3.0])
    normal = normal / np.linalg.norm(normal)
    plane = RigidPlane(origin=[0.0, 0.0, 0.0], normal=normal)
    kp = 5000.0

    # Penetrated node by 0.1 mm behind the plane
    nodes = np.array([[-0.1 * normal[0], -0.1 * normal[1], -0.1 * normal[2]]])
    contact = RigidContact(obstacle=plane, slave_nodes=[0], k_penalty=kp, c_damping=0.0)

    K_ana = contact.evaluate_stiffness(current_nodes=nodes, n_dofs=3).toarray()
    f0 = contact.evaluate_forces(current_nodes=nodes, n_dofs=3)

    eps = 1e-7
    K_num = np.zeros((3, 3))
    for j in range(3):
        nodes_pert = nodes.copy()
        nodes_pert[0, j] += eps
        f_pert = contact.evaluate_forces(current_nodes=nodes_pert, n_dofs=3)
        # K_c = - df_c / du
        K_num[:, j] = -(f_pert - f0) / eps

    # Compare analytical and numerical stiffness to high precision
    assert np.allclose(K_ana, K_num, rtol=1e-5, atol=1e-5)


# =============================================================================
# 5. Static Contact Simulation Test
# =============================================================================

def test_static_cantilever_resting_on_floor(
    single_element_mesh: VolumetricMesh,
    neohookean_material: NeoHookean,
) -> None:
    """Tests static beam resting on a rigid plane under downward gravity load."""
    elem = Tet4TotalLagrangian(neohookean_material)
    domain = SolidDomain(single_element_mesh, neohookean_material, elem)

    # Pin base nodes (X = 0)
    base_nodes = np.where(single_element_mesh.nodes[:, 0] < 0.1)[0]
    dirichlet = [
        DirichletConstraint(node_indices=base_nodes, components=(0, 1, 2), value=0.0)
    ]

    # Rigid plane placed at Y = -0.1 mm
    floor = RigidPlane(origin=[0.0, -0.1, 0.0], normal=[0.0, 1.0, 0.0])
    contact = RigidContact(
        obstacle=floor,
        domain=domain,
        k_penalty=1e5,
    )

    # Downward external load
    f_ext = np.zeros(domain.n_dofs)
    tip_nodes = np.where(single_element_mesh.nodes[:, 0] > 0.9)[0]
    for n in tip_nodes:
        f_ext[3 * n + 1] = -50.0  # downward load

    solver = NonlinearStaticAnalysis(domain)
    res = solver.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        contact=contact,
        n_steps=5,
        nr_tol=1e-3,
        verbose=False,
    )

    assert res.converged
    # Check that contact arrested the downward deflection near Y = -0.1
    current_y = domain.current_nodes[:, 1]
    min_y = np.min(current_y)
    # Penetration should be minimal (less than 0.01 mm)
    assert min_y >= -0.11


# =============================================================================
# 6. Dynamic Contact Simulation Test
# =============================================================================

def test_dynamic_bouncing_cube(
    five_tet_cube_mesh: VolumetricMesh,
    neohookean_material: NeoHookean,
) -> None:
    """Tests dynamic free fall and elastic rebound off a rigid ground floor."""
    mesh = five_tet_cube_mesh.copy()
    # Place cube 0.02 m above floor
    mesh.transform(translation=np.array([0.0, 0.02, 0.0]))

    elem = Tet4Linear(neohookean_material)
    domain = SolidDomain(mesh, neohookean_material, elem)

    # Rigid floor at Y = 0.0
    floor = RigidPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0])
    contact = RigidContact(
        obstacle=floor,
        domain=domain,
        k_penalty=2e6,
        c_damping=0.0,  # elastic rebound
    )

    # Gravity downwards
    grav = GravityLoad(acceleration=(0.0, -9.81, 0.0))
    M = domain.assemble_mass_matrix()
    f_grav = grav.compute_force_vector(M)

    # Unconstrained dynamic free-fall
    mgr = DirichletManager(domain.n_dofs)

    solver = NonlinearDynamicAnalysis(domain, spectral_radius=0.8)
    res = solver.solve(
        dirichlet=mgr,
        f_ext=f_grav,
        dt=0.002,
        n_steps=40,
        contact=contact,
        nr_tol=1e-3,
        verbose=False,
    )

    assert res.converged
    # The cube fell, impacted, and rebounded: min Y must stay non-penetrating
    min_y = np.min(domain.current_nodes[:, 1])
    assert min_y >= -0.01


def test_rigid_contact_augmented_lagrangian(
    five_tet_cube_mesh: VolumetricMesh,
    neohookean_material: NeoHookean,
) -> None:
    """Tests that RigidContact supports Augmented Lagrangian formulation."""
    mesh = five_tet_cube_mesh.copy()
    elem = Tet4Linear(neohookean_material)
    domain = SolidDomain(mesh, neohookean_material, elem)

    # Move node 0 to penetrate Y = 0 floor by -0.05
    u = np.zeros(domain.n_dofs)
    u[1] = -0.05
    domain.update_state(u=u)

    floor = RigidPlane(origin=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0])
    contact = RigidContact(
        obstacle=floor,
        domain=domain,
        slave_nodes=[0],
        k_penalty=1000.0,
        formulation="augmented_lagrangian",
    )

    f1 = contact.evaluate_forces()
    assert f1[1] > 0.0  # upward penalty repulsive force

    # Update multipliers
    stats = contact.update_multipliers()
    assert stats["num_active"] == 1
    assert stats["max_multiplier"] > 0.0

    # Force now incorporates multiplier
    f2 = contact.evaluate_forces()
    assert f2[1] > f1[1]
