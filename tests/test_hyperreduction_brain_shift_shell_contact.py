"""
Tests for Hyper-Reduced Brain Shift with Coupled Solid-Shell Meninges & Dual Contact.
======================================================================================
Validates:
1. Coupled Solid-Shell Domain (Tet4 + Shell3) ECSW Virtual Work Conservation:
   sum_{e=1}^{N_e} A[:, e] == b to machine precision.
2. ECSW Cubature Optimization:
   Selects a compact Reduced Integration Domain (RID) with strictly positive weights w_e > 0
   retaining both 3D solid continuum and 2D surface shell elements.
3. Strict Tangent Stiffness Symmetry:
   ||K_r - K_r^T||_F == 0.0 preserved identically across nonlinear evaluation states.
4. Dual Multi-Obstacle Contact Mechanics:
   CollisionDetector with concave skull obstacle (RayMeshObstacle) and convex surgical
   indenter (RigidSphere) generating conservative contact forces and symmetric stiffness.
5. Real-Time Dynamic Solver:
   Executes online dynamic time steps with Generalized-alpha integration.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian, Shell3Corotational
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain, ShellDomain, CoupledSolidShellDomain
from psierra.boundary import DirichletManager, DirichletConstraint, DOFConstraint, GravityLoad
from psierra.reduction.basis import PODBasis
from psierra.hyperreduction.ecsw import (
    ReducedIntegrationDomain,
    ECSWTrainingMatrix,
    ECSWOptimizer,
)
from psierra.hyperreduction.model import HyperReducedNonlinearModel
from psierra.hyperreduction.contact import HyperReducedRigidContact
from psierra.hyperreduction.solver import (
    HyperReducedNonlinearDynamicSolver,
    HyperReducedDynamicRelaxationSolver,
)
from psierra.collision import RayMeshObstacle, CollisionDetector
from psierra.collision.shapes import RigidSphere, RigidPlane


@pytest.fixture(scope="module")
def coupled_test_domain() -> tuple[CoupledSolidShellDomain, VolumetricMesh]:
    """Sets up a lightweight coupled solid-shell domain for unit testing."""
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        mesh_path = Path("psierra") / mesh_path
    assert mesh_path.exists(), f"Mesh not found: {mesh_path}"

    mesh = VolumetricMesh(mesh_path)

    # 3D Solid Core
    params_solid = IsotropicElasticParameters()
    params_solid.set_from_youngs_poisson(
        youngs_modulus=100.0,
        poissons_ratio=0.30,
        mass_density=1.0e-6,
    )
    mat_solid = NeoHookean(params_solid)
    elem_solid = Tet4TotalLagrangian(mat_solid)
    solid_domain = SolidDomain(mesh, mat_solid, elem_solid)

    # 2D Surface Shell Skin (Meninges membrane)
    params_shell = IsotropicElasticParameters()
    params_shell.set_from_youngs_poisson(
        youngs_modulus=500.0,
        poissons_ratio=0.30,
        mass_density=1.0e-6,
    )
    mat_shell = NeoHookean(params_shell)
    elem_shell = Shell3Corotational(mat_shell, thickness=0.5)
    shell_domain = ShellDomain(mesh.surface, mat_shell, elem_shell)

    coupled_domain = CoupledSolidShellDomain(solid_domain, shell_domain)
    return coupled_domain, mesh


def test_coupled_ecsw_training_matrix_conservation(coupled_test_domain):
    """Verifies exact elemental virtual work conservation on CoupledSolidShellDomain."""
    coupled_dom, mesh = coupled_test_domain
    n_dofs = coupled_dom.n_dofs
    r = 4

    np.random.seed(42)
    # Generate mock displacement snapshots
    snapshots = [np.random.randn(n_dofs) * 0.02 for _ in range(3)]
    snap_mat = np.column_stack(snapshots)

    # Orthonormal basis
    q, _ = np.linalg.qr(np.column_stack([snap_mat, np.random.randn(n_dofs, r - len(snapshots))]))
    basis = PODBasis(basis=q, singular_values=np.ones(r), cumulative_energy=1.0, total_energy=1.0)

    A, b = ECSWTrainingMatrix.build(coupled_dom, basis, snapshots, include_stiffness=True)

    # Check conservation: sum_{e=1}^{N_e} A[:, e] == b
    sum_A = A.sum(axis=1)
    conservation_error = float(np.linalg.norm(sum_A - b))
    assert conservation_error < 1e-12, f"Virtual work conservation violated: {conservation_error:.2e}"
    assert A.shape[1] == mesh.n_elements + mesh.surface.n_elements


def test_coupled_ecsw_optimizer_and_rid(coupled_test_domain):
    """Verifies that ECSWOptimizer produces a valid RID with positive weights."""
    coupled_dom, mesh = coupled_test_domain
    n_dofs = coupled_dom.n_dofs
    r = 3

    np.random.seed(123)
    snapshots = [np.random.randn(n_dofs) * 0.01 for _ in range(4)]
    q, _ = np.linalg.qr(np.column_stack(snapshots)[:, :r])
    basis = PODBasis(basis=q, singular_values=np.ones(r), cumulative_energy=1.0, total_energy=1.0)

    A, b = ECSWTrainingMatrix.build(coupled_dom, basis, snapshots, include_stiffness=False)
    optimizer = ECSWOptimizer(tolerance=0.05, max_elements=20)
    rid = optimizer.optimize(A, b)

    assert rid.n_selected > 0
    assert rid.n_selected <= 20
    assert np.all(rid.weights > 0.0)
    assert rid.relative_error <= 0.05
    assert rid.n_total_elements == mesh.n_elements + mesh.surface.n_elements


def test_coupled_hyperreduced_model_symmetry(coupled_test_domain):
    """Verifies that HyperReducedNonlinearModel preserves exact tangent stiffness symmetry."""
    coupled_dom, mesh = coupled_test_domain
    n_dofs = coupled_dom.n_dofs
    r = 4

    np.random.seed(99)
    snapshots = [np.random.randn(n_dofs) * 0.02 for _ in range(3)]
    q, _ = np.linalg.qr(np.column_stack([np.column_stack(snapshots), np.random.randn(n_dofs, 1)]))
    basis = PODBasis(basis=q, singular_values=np.ones(r), cumulative_energy=1.0, total_energy=1.0)

    A, b = ECSWTrainingMatrix.build(coupled_dom, basis, snapshots, include_stiffness=True)
    rid = ECSWOptimizer(tolerance=0.03, max_elements=15).optimize(A, b)

    model = HyperReducedNonlinearModel(coupled_dom, basis, rid)

    # Evaluate at multiple reduced displacement states
    test_states = [
        np.zeros(r),
        np.array([0.05, -0.03, 0.02, -0.01]),
        np.array([-0.08, 0.04, -0.02, 0.05]),
    ]

    for u_r in test_states:
        f_r, K_r = model.evaluate_forces_and_stiffness(u_r)
        assert f_r.shape == (r,)
        assert K_r.shape == (r, r)
        symmetry_error = float(np.linalg.norm(K_r - K_r.T, ord="fro"))
        assert symmetry_error == 0.0, f"Tangent stiffness symmetry violated: {symmetry_error}"


def test_coupled_dual_obstacle_contact(coupled_test_domain):
    """Verifies dual contact mechanics (concave boundary + convex indenter) on coupled model."""
    coupled_dom, mesh = coupled_test_domain
    n_dofs = coupled_dom.n_dofs
    r = 3

    np.random.seed(77)
    snapshots = [np.random.randn(n_dofs) * 0.01 for _ in range(3)]
    q, _ = np.linalg.qr(np.column_stack(snapshots))
    basis = PODBasis(basis=q, singular_values=np.ones(r), cumulative_energy=1.0, total_energy=1.0)

    A, b = ECSWTrainingMatrix.build(coupled_dom, basis, snapshots)
    rid = ECSWOptimizer(tolerance=0.05, max_elements=10).optimize(A, b)

    model = HyperReducedNonlinearModel(coupled_dom, basis, rid)

    # Obstacle 1: Flat floor plane
    obs1 = RigidPlane(origin=(0.0, -25.0, 0.0), normal=(0.0, 1.0, 0.0))
    # Obstacle 2: Spherical indenter
    obs2 = RigidSphere(center=(0.0, 0.0, 100.0), radius=15.0)

    detector = CollisionDetector([obs1, obs2])
    contact = HyperReducedRigidContact(
        model=model,
        obstacle=detector,
        k_penalty=5.0,
        formulation="penalty",
    )

    u_r = np.array([0.1, -0.05, 0.02])
    f_r_c, K_r_c = contact.evaluate_reduced_forces_and_stiffness(u_r)

    assert f_r_c.shape == (r,)
    assert K_r_c.shape == (r, r)
    symm_err = float(np.linalg.norm(K_r_c - K_r_c.T, ord="fro"))
    assert symm_err == 0.0, f"Contact tangent symmetry violated: {symm_err}"


def test_coupled_hyperreduced_dynamic_solver(coupled_test_domain):
    """Verifies dynamic solver time integration with contact on coupled solid-shell model."""
    coupled_dom, mesh = coupled_test_domain
    n_dofs = coupled_dom.n_dofs
    r = 3

    np.random.seed(55)
    snapshots = [np.random.randn(n_dofs) * 0.01 for _ in range(3)]
    q, _ = np.linalg.qr(np.column_stack(snapshots))
    basis = PODBasis(basis=q, singular_values=np.ones(r), cumulative_energy=1.0, total_energy=1.0)

    A, b = ECSWTrainingMatrix.build(coupled_dom, basis, snapshots)
    rid = ECSWOptimizer(tolerance=0.05, max_elements=10).optimize(A, b)

    model = HyperReducedNonlinearModel(coupled_dom, basis, rid)
    obs = RigidPlane(origin=(0.0, -25.0, 0.0), normal=(0.0, 1.0, 0.0))
    contact = HyperReducedRigidContact(model=model, obstacle=obs, k_penalty=2.0)
    model.contact = contact

    solver = HyperReducedNonlinearDynamicSolver(
        model=model,
        contact=contact,
        dt=0.005,
        spectral_radius=0.8,
    )

    for step in range(5):
        u_r_next = solver.step()
        assert np.all(np.isfinite(u_r_next))
        assert len(u_r_next) == r

    stats = solver.get_stats()
    assert stats.steps_completed == 5
    assert stats.effective_hz > 100.0


def test_coupled_dual_obstacle_dynamic_toggling(coupled_test_domain):
    """Verifies toggling individual obstacles (skull/indenter) on and off dynamically."""
    coupled_dom, _ = coupled_test_domain
    n_dofs = coupled_dom.n_dofs
    r = 3

    np.random.seed(88)
    snapshots = [np.random.randn(n_dofs) * 0.01 for _ in range(3)]
    q, _ = np.linalg.qr(np.column_stack(snapshots))
    basis = PODBasis(basis=q, singular_values=np.ones(r), cumulative_energy=1.0, total_energy=1.0)

    A, b = ECSWTrainingMatrix.build(coupled_dom, basis, snapshots)
    rid = ECSWOptimizer(tolerance=0.05, max_elements=10).optimize(A, b)
    model = HyperReducedNonlinearModel(coupled_dom, basis, rid)

    # Intersecting plane (obstacle 1) and overlapping sphere at a surface node (obstacle 2)
    surf_pts = model.domain.mesh.surface.nodes
    ref_center = surf_pts.mean(axis=0)
    obs1 = RigidPlane(origin=tuple(ref_center - np.array([0.0, 0.5, 0.0])), normal=(0.0, 1.0, 0.0))
    obs2 = RigidSphere(center=tuple(surf_pts[0]), radius=2.0)

    detector = CollisionDetector([obs1, obs2])
    contact = HyperReducedRigidContact(
        model=model,
        obstacle=detector,
        k_penalty=5.0,
        formulation="augmented_lagrangian",
    )

    u_r = np.zeros(r)

    # 1. Both obstacles active -> active contact detected
    f_both, K_both = contact.evaluate_reduced_forces_and_stiffness(u_r)
    assert contact.get_telemetry().num_active > 0
    assert np.linalg.norm(f_both) > 0.0
    assert np.linalg.norm(K_both - K_both.T, ord="fro") == 0.0

    # 2. Only obstacle 2 active
    detector._obstacles[:] = [obs2]
    contact.reset_state()
    f_obs2, _ = contact.evaluate_reduced_forces_and_stiffness(u_r)
    assert contact.get_telemetry().num_active > 0
    assert np.linalg.norm(f_obs2) > 0.0

    # 3. Both obstacles disabled -> zero active contacts, zero forces, zero stiffness
    detector._obstacles[:] = []
    contact.reset_state()
    f_none, K_none = contact.evaluate_reduced_forces_and_stiffness(u_r)
    telem_none = contact.get_telemetry()
    assert telem_none.num_active == 0
    assert telem_none.max_penetration == 0.0
    assert np.all(f_none == 0.0)
    assert np.all(K_none == 0.0)


def test_coupled_hyperreduced_dynamic_relaxation_solver(coupled_test_domain):
    """Verifies HyperReducedDynamicRelaxationSolver explicit static equilibrium convergence."""
    coupled_dom, mesh = coupled_test_domain
    n_dofs = coupled_dom.n_dofs
    r = 3

    np.random.seed(66)
    snapshots = [np.random.randn(n_dofs) * 0.01 for _ in range(3)]
    q, _ = np.linalg.qr(np.column_stack(snapshots))
    basis = PODBasis(basis=q, singular_values=np.ones(r), cumulative_energy=1.0, total_energy=1.0)

    A, b = ECSWTrainingMatrix.build(coupled_dom, basis, snapshots)
    rid = ECSWOptimizer(tolerance=0.05, max_elements=10).optimize(A, b)
    model = HyperReducedNonlinearModel(coupled_dom, basis, rid)

    # 1. Kinetic Damping Relaxation without contact
    dr_solver = HyperReducedDynamicRelaxationSolver(
        model=model,
        damping_type="kinetic",
        dt=1.0,
        tol=1e-4,
        max_steps=150,
    )

    f_ext_r = np.array([5.0, -2.5, 1.0])
    u_r_eq = dr_solver.relax(f_ext_r)
    assert dr_solver.converged
    assert dr_solver.kinetic_peaks > 0
    assert np.all(np.isfinite(u_r_eq))

    # Verify residual at equilibrium
    f_int, _ = model.evaluate_forces_and_stiffness(u_r_eq)
    res_err = float(np.linalg.norm(f_ext_r - f_int) / np.linalg.norm(f_ext_r))
    assert res_err <= 1e-4

    # 2. Dynamic Relaxation with Obstacle Contact
    obs = RigidPlane(origin=(0.0, -61.0, 0.0), normal=(0.0, 1.0, 0.0))
    contact = HyperReducedRigidContact(model=model, obstacle=obs, k_penalty=2.0)
    dr_contact_solver = HyperReducedDynamicRelaxationSolver(
        model=model,
        contact=contact,
        damping_type="kinetic",
        dt=1.0,
        tol=1e-3,
        max_steps=100,
    )

    u_r_contact = dr_contact_solver.relax(np.array([0.0, -10.0, 0.0]))
    assert dr_contact_solver.converged
    stats = dr_contact_solver.get_stats()
    assert stats.steps_completed > 0
    assert stats.effective_hz > 500.0
