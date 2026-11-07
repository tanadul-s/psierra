"""
Unit and Integration Tests for Hyper-Reduced Contact Mechanics (Method 1: Sub-Basis Projection).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from psierra.mesh.volumetric import VolumetricMesh
from psierra.material.parameters import IsotropicElasticParameters
from psierra.material.neohookean import NeoHookean
from psierra.element.tet4_total_lagrangian import Tet4TotalLagrangian
from psierra.domain.solid import SolidDomain
from psierra.boundary import DirichletManager, DirichletConstraint, GravityLoad
from psierra.boundary.selector import select_nodes_by_plane
from psierra.damping.rayleigh import RayleighDamping
from psierra.reduction.snapshots import SnapshotManager
from psierra.hyperreduction.ecsw import (
    ReducedIntegrationDomain,
    ECSWTrainingMatrix,
    ECSWOptimizer,
)
from psierra.hyperreduction.model import HyperReducedNonlinearModel
from psierra.hyperreduction.solver import (
    HyperReducedNonlinearStaticSolver,
    HyperReducedNonlinearDynamicSolver,
)
from psierra.hyperreduction.contact import HyperReducedRigidContact
from psierra.collision.shapes import RigidPlane, RigidSphere


@pytest.fixture
def beam_hyper_model() -> tuple[HyperReducedNonlinearModel, VolumetricMesh]:
    """Sets up a hyper-reduced beam model for contact tests."""
    mesh_path = Path("psierra/data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        pytest.skip("beam1.vtk not found")

    mesh = VolumetricMesh(mesh_path)
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(0.021, 0.45, 1.1e-9)
    material = NeoHookean(params)
    elem = Tet4TotalLagrangian(material)
    domain = SolidDomain(mesh=mesh, material=material, element=elem)

    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min", tolerance=1e-3)
    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0))

    # Construct small synthetic basis
    rng = np.random.default_rng(42)
    r = 3
    Phi_raw = rng.standard_normal((domain.n_dofs, r))
    Phi_raw[dirichlet.prescribed_dofs, :] = 0.0
    Phi, _ = np.linalg.qr(Phi_raw)
    mgr = SnapshotManager(n_dofs=domain.n_dofs)
    mgr.add_snapshots(Phi)
    basis = mgr.compute_basis(energy_threshold=0.999)

    snap = basis.basis_matrix @ np.array([0.2, -0.1, 0.2])
    A, b = ECSWTrainingMatrix.build(domain=domain, basis=basis, snapshots=snap, include_stiffness=True)
    rid = ECSWOptimizer(tolerance=0.05, verbose=False).optimize(A, b)

    f_grav = GravityLoad(acceleration=(0.0, -9810.0, 0.0)).compute_force_vector(
        domain.assemble_mass_matrix()
    )

    model = HyperReducedNonlinearModel(
        domain=domain,
        basis=basis,
        rid=rid,
        dirichlet=dirichlet,
        damping=RayleighDamping(0.5, 0.001),
        f_ext_base=f_grav,
        surface_nodes=mesh.surface.node_indices,
    )
    return model, mesh


class TestHyperReducedRigidContact:
    """Tests HyperReducedRigidContact formulation, projection, and solvers."""

    def test_contact_symmetry_and_positive_definiteness(self, beam_hyper_model) -> None:
        model, mesh = beam_hyper_model
        # Place obstacle so that some surface nodes are penetrating
        obstacle = RigidPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))
        contact = HyperReducedRigidContact(
            model=model,
            obstacle=obstacle,
            k_penalty=10.0,
            formulation="augmented_lagrangian",
        )

        u_r_test = np.array([0.5, -0.3, 0.4])
        f_r_c, K_r_c = contact.evaluate_reduced_forces_and_stiffness(u_r_test)

        assert f_r_c.shape == (model.r,)
        assert K_r_c.shape == (model.r, model.r)

        # STRICT SYMMETRY PRESERVATION: K_r_c == K_r_c^T to machine precision
        np.testing.assert_allclose(K_r_c, K_r_c.T, rtol=1e-14, atol=1e-14)

        # Positive semi-definiteness: eigenvalues >= 0
        eigenvalues = np.linalg.eigvalsh(K_r_c)
        assert np.all(eigenvalues >= -1e-12)

    def test_penalty_and_augmented_lagrangian_toggle(self, beam_hyper_model) -> None:
        model, mesh = beam_hyper_model
        obstacle = RigidPlane(origin=(0.0, -50.0, 0.0), normal=(0.0, 1.0, 0.0))
        contact = HyperReducedRigidContact(
            model=model,
            obstacle=obstacle,
            k_penalty=5.0,
            formulation="augmented_lagrangian",
        )

        assert contact.formulation == "augmented_lagrangian"
        new_form = contact.toggle_formulation()
        assert new_form == "penalty"
        assert contact.formulation == "penalty"

        new_form = contact.toggle_formulation()
        assert new_form == "augmented_lagrangian"
        assert contact.formulation == "augmented_lagrangian"

    def test_uzawa_multiplier_update(self, beam_hyper_model) -> None:
        model, mesh = beam_hyper_model
        obstacle = RigidPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))
        contact = HyperReducedRigidContact(
            model=model,
            obstacle=obstacle,
            k_penalty=5.0,
            formulation="augmented_lagrangian",
        )

        u_r_test = np.array([0.0, -1.0, 0.0])
        # Before update, multipliers are 0
        assert np.all(contact.multipliers == 0.0)

        stats = contact.update_multipliers(u_r_test)
        assert stats["num_active"] > 0
        assert stats["max_multiplier"] > 0.0
        assert np.any(contact.multipliers > 0.0)

    def test_static_solver_with_contact(self, beam_hyper_model) -> None:
        model, mesh = beam_hyper_model
        # Rigid plane acting as floor at Y = -70 mm
        floor = RigidPlane(origin=(0.0, -70.0, 0.0), normal=(0.0, 1.0, 0.0))
        contact = HyperReducedRigidContact(
            model=model,
            obstacle=floor,
            k_penalty=0.1,
            formulation="augmented_lagrangian",
        )

        solver = HyperReducedNonlinearStaticSolver(
            model=model, max_iter=20, tol=1e-4, contact=contact
        )
        u_r_sol = solver.solve(n_increments=2, verbose=False)
        assert u_r_sol.shape == (model.r,)

        # Telemetry should record active contact
        telem = contact.get_telemetry()
        assert isinstance(telem.max_penetration, float)

    def test_dynamic_solver_with_contact(self, beam_hyper_model) -> None:
        model, mesh = beam_hyper_model
        # Spherical indenter obstacle
        sphere = RigidSphere(center=(0.0, 0.0, 100.0), radius=50.0)
        contact = HyperReducedRigidContact(
            model=model,
            obstacle=sphere,
            k_penalty=0.5,
            formulation="augmented_lagrangian",
        )

        solver = HyperReducedNonlinearDynamicSolver(
            model=model, dt=0.005, spectral_radius=0.8, nr_max_iter=5, contact=contact
        )

        # Run 20 dynamic steps with contact
        for _ in range(20):
            solver.step()

        stats = solver.get_stats()
        assert stats.steps_completed == 20
        assert stats.effective_hz > 250.0  # Must exceed 250 Hz with contact!
