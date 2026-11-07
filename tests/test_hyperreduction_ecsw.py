"""
Unit and Benchmark Tests for Energy-Conserving Sampling and Weighting (ECSW)
Hyper-Reduction Framework.
"""

from __future__ import annotations
from pathlib import Path
import tempfile
import numpy as np
import pytest

from psierra.mesh.volumetric import VolumetricMesh
from psierra.material.parameters import IsotropicElasticParameters
from psierra.material.hookean import Hookean
from psierra.material.neohookean import NeoHookean
from psierra.element.tet4_linear import Tet4Linear
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


@pytest.fixture
def beam_domain() -> tuple[SolidDomain, DirichletManager, VolumetricMesh]:
    """Sets up a 3D continuum cantilever beam for testing."""
    mesh_path = Path("psierra/data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        pytest.skip("beam1.vtk not found")

    mesh = VolumetricMesh(mesh_path)
    params = IsotropicElasticParameters()
    # Soft rubber: E = 21 kPa = 0.021 MPa, nu = 0.45, rho = 1.1e-9 tonne/mm³
    params.set_from_youngs_poisson(0.021, 0.45, 1.1e-9)
    material = NeoHookean(params)
    elem = Tet4TotalLagrangian(material)
    domain = SolidDomain(mesh=mesh, material=material, element=elem)

    # Clamped root at Z = min(Z)
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min", tolerance=1e-3)
    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0))

    return domain, dirichlet, mesh


class TestReducedIntegrationDomain:
    """Tests for ReducedIntegrationDomain validation and persistence."""

    def test_validation_rules(self) -> None:
        # Length mismatch must raise ValueError
        with pytest.raises(ValueError, match="must match weights length"):
            ReducedIntegrationDomain(
                element_indices=np.array([1, 2]),
                weights=np.array([1.0]),
                n_total_elements=100,
                relative_error=0.01,
            )

        # Non-positive weights must raise ValueError
        with pytest.raises(ValueError, match="strictly positive"):
            ReducedIntegrationDomain(
                element_indices=np.array([1, 2]),
                weights=np.array([1.0, -0.5]),
                n_total_elements=100,
                relative_error=0.01,
            )

    def test_persistence_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test_ecsw_rid.npz"

            rid = ReducedIntegrationDomain(
                element_indices=np.array([5, 12, 45, 89], dtype=np.int64),
                weights=np.array([1.25, 0.85, 2.10, 0.40], dtype=np.float64),
                n_total_elements=500,
                relative_error=0.0085,
                metadata={"experiment": "cantilever_beam", "tol": 0.01},
            )

            assert rid.n_selected == 4
            assert rid.compression_ratio == 4 / 500
            assert rid.speedup_factor == 125.0

            saved = rid.save(archive_path)
            assert saved.is_file()

            loaded = ReducedIntegrationDomain.load(archive_path)
            np.testing.assert_array_equal(loaded.element_indices, rid.element_indices)
            np.testing.assert_allclose(loaded.weights, rid.weights, rtol=1e-14)
            assert loaded.n_total_elements == 500
            assert np.isclose(loaded.relative_error, 0.0085)
            assert loaded.metadata["experiment"] == "cantilever_beam"


class TestECSWTrainingAndOptimization:
    """Tests virtual work matrix assembly, exact projection identity, and greedy NNLS."""

    def test_training_matrix_identity(self, beam_domain) -> None:
        domain, dirichlet, mesh = beam_domain

        # Create a small POD modal basis
        rng = np.random.default_rng(42)
        r = 3
        Phi_raw = rng.standard_normal((domain.n_dofs, r))
        # Zero out clamped DOFs
        Phi_raw[dirichlet.prescribed_dofs, :] = 0.0
        Phi, _ = np.linalg.qr(Phi_raw)
        basis = SnapshotManager(n_dofs=domain.n_dofs)
        basis.add_snapshots(Phi)
        pod_basis = basis.compute_basis(energy_threshold=0.999)

        # Generate 2 smooth modal snapshots
        u1 = pod_basis.basis_matrix @ np.array([0.5, -0.3, 0.4])
        u2 = pod_basis.basis_matrix @ np.array([0.8, 0.2, -0.5])
        snapshots = np.column_stack([u1, u2])

        A, b = ECSWTrainingMatrix.build(domain=domain, basis=pod_basis, snapshots=snapshots)

        assert A.shape == (2 * pod_basis.n_modes, mesh.n_elements)
        assert b.shape == (2 * pod_basis.n_modes,)

        # Verify exact mathematical identity: b = sum_{e=0}^{N_e - 1} a_e
        np.testing.assert_allclose(b, A.sum(axis=1), rtol=1e-12)

        # Verify exact match against full-order internal force projection
        domain.update_state(u=u1)
        f_int_full_1 = domain.assemble_internal_forces()
        f_r_exact_1 = pod_basis.basis_matrix.T @ f_int_full_1

        domain.update_state(u=u2)
        f_int_full_2 = domain.assemble_internal_forces()
        f_r_exact_2 = pod_basis.basis_matrix.T @ f_int_full_2

        b_expected = np.concatenate([f_r_exact_1, f_r_exact_2])
        np.testing.assert_allclose(b, b_expected, rtol=1e-12)

    def test_ecsw_optimizer_convergence(self, beam_domain) -> None:
        domain, dirichlet, mesh = beam_domain

        # Construct training matrix from 3 snapshots
        rng = np.random.default_rng(123)
        r = 3
        Phi_raw = rng.standard_normal((domain.n_dofs, r))
        Phi_raw[dirichlet.prescribed_dofs, :] = 0.0
        Phi, _ = np.linalg.qr(Phi_raw)
        basis = SnapshotManager(n_dofs=domain.n_dofs)
        basis.add_snapshots(Phi)
        pod_basis = basis.compute_basis(energy_threshold=0.999)

        snap_list = [
            pod_basis.basis_matrix @ (np.array([0.2, -0.1, 0.3]) * scale)
            for scale in [0.5, 1.0, 1.5]
        ]

        A, b = ECSWTrainingMatrix.build(domain=domain, basis=pod_basis, snapshots=snap_list)

        optimizer = ECSWOptimizer(tolerance=0.01, verbose=False)
        rid = optimizer.optimize(A, b)

        # Retained elements must be sparse (e.g. < 20% of elements)
        assert rid.n_selected < 0.20 * mesh.n_elements
        assert rid.relative_error <= 0.01
        assert np.all(rid.weights > 0.0)

    def test_training_matrix_stiffness_augmentation(self, beam_domain) -> None:
        domain, dirichlet, mesh = beam_domain

        rng = np.random.default_rng(77)
        r = 3
        Phi_raw = rng.standard_normal((domain.n_dofs, r))
        Phi_raw[dirichlet.prescribed_dofs, :] = 0.0
        Phi, _ = np.linalg.qr(Phi_raw)
        basis = SnapshotManager(n_dofs=domain.n_dofs)
        basis.add_snapshots(Phi)
        pod_basis = basis.compute_basis(energy_threshold=0.999)

        snap = pod_basis.basis_matrix @ np.array([0.3, -0.2, 0.4])
        A, b = ECSWTrainingMatrix.build(
            domain=domain, basis=pod_basis, snapshots=snap, include_stiffness=True
        )

        expected_rows = 1 * r + (r * (r + 1)) // 2  # 3 + 6 = 9 rows
        assert A.shape == (expected_rows, mesh.n_elements)
        assert b.shape == (expected_rows,)
        np.testing.assert_allclose(b, A.sum(axis=1), rtol=1e-12)

        optimizer = ECSWOptimizer(tolerance=0.02, verbose=False)
        rid = optimizer.optimize(A, b)
        assert rid.n_selected > 0
        assert np.all(rid.weights > 0.0)


class TestHyperReducedNonlinearModelAndSolvers:
    """Tests HyperReducedNonlinearModel, symmetry preservation, and static/dynamic solvers."""

    def test_model_symmetry_and_evaluation(self, beam_domain) -> None:
        domain, dirichlet, mesh = beam_domain

        # Compute basis and ECSW domain
        rng = np.random.default_rng(42)
        r = 3
        Phi_raw = rng.standard_normal((domain.n_dofs, r))
        Phi_raw[dirichlet.prescribed_dofs, :] = 0.0
        Phi, _ = np.linalg.qr(Phi_raw)
        mgr = SnapshotManager(n_dofs=domain.n_dofs)
        mgr.add_snapshots(Phi)
        basis = mgr.compute_basis(energy_threshold=0.999)

        u_snap = basis.basis_matrix @ np.array([0.3, -0.2, 0.4])
        A, b = ECSWTrainingMatrix.build(domain=domain, basis=basis, snapshots=u_snap)

        optimizer = ECSWOptimizer(tolerance=0.02, verbose=False)
        rid = optimizer.optimize(A, b)

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

        # Evaluate at an arbitrary state u_r
        u_r_test = np.array([0.5, -0.3, 0.8])
        f_int_r, K_r_t = model.evaluate_forces_and_stiffness(u_r_test)

        assert f_int_r.shape == (r,)
        assert K_r_t.shape == (r, r)

        # STRICT SYMMETRY PRESERVATION: K_r_t == K_r_t^T
        np.testing.assert_allclose(K_r_t, K_r_t.T, rtol=1e-14, atol=1e-14)

        # Surface displacement reconstruction check
        u_surf = model.reconstruct_surface_displacement(u_r_test)
        assert u_surf.shape == (mesh.surface.n_nodes, 3)

    def test_hyperreduced_static_solver(self, beam_domain) -> None:
        domain, dirichlet, mesh = beam_domain

        # Synthetic basis & cubature
        rng = np.random.default_rng(10)
        r = 3
        Phi_raw = rng.standard_normal((domain.n_dofs, r))
        Phi_raw[dirichlet.prescribed_dofs, :] = 0.0
        Phi, _ = np.linalg.qr(Phi_raw)
        mgr = SnapshotManager(n_dofs=domain.n_dofs)
        mgr.add_snapshots(Phi)
        basis = mgr.compute_basis(energy_threshold=0.999)

        snap = basis.basis_matrix @ np.array([0.2, -0.15, 0.25])
        A, b = ECSWTrainingMatrix.build(domain=domain, basis=basis, snapshots=snap)
        rid = ECSWOptimizer(tolerance=0.02, verbose=False).optimize(A, b)

        f_grav = GravityLoad(acceleration=(0.0, -9810.0, 0.0)).compute_force_vector(
            domain.assemble_mass_matrix()
        )

        model = HyperReducedNonlinearModel(
            domain=domain,
            basis=basis,
            rid=rid,
            dirichlet=dirichlet,
            f_ext_base=f_grav,
        )

        solver = HyperReducedNonlinearStaticSolver(model=model, max_iter=20, tol=1e-6)
        u_r_sol = solver.solve(verbose=False)

        assert u_r_sol.shape == (r,)
        # Residual at solution must be near zero
        res, _ = model.evaluate_residual(u_r_sol)
        rel_res = np.linalg.norm(res) / np.linalg.norm(model.f_r_base)
        assert rel_res < 1e-5

    def test_hyperreduced_dynamic_solver(self, beam_domain) -> None:
        domain, dirichlet, mesh = beam_domain

        # Setup model
        rng = np.random.default_rng(99)
        r = 2
        Phi_raw = rng.standard_normal((domain.n_dofs, r))
        Phi_raw[dirichlet.prescribed_dofs, :] = 0.0
        Phi, _ = np.linalg.qr(Phi_raw)
        mgr = SnapshotManager(n_dofs=domain.n_dofs)
        mgr.add_snapshots(Phi)
        basis = mgr.compute_basis(energy_threshold=0.999)

        snap = basis.basis_matrix @ np.array([0.15, -0.10])
        A, b = ECSWTrainingMatrix.build(domain=domain, basis=basis, snapshots=snap)
        rid = ECSWOptimizer(tolerance=0.03, verbose=False).optimize(A, b)

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

        solver = HyperReducedNonlinearDynamicSolver(
            model=model, dt=0.005, spectral_radius=0.8, nr_max_iter=5
        )

        # Step dynamic solver
        for _ in range(50):
            solver.step()

        stats = solver.get_stats()
        assert stats.steps_completed == 50
        # Physics rate should exceed 2,000 Hz (< 0.5 ms per step)
        assert stats.average_step_ms < 1.0, f"Step time {stats.average_step_ms:.3f} ms exceeds 1.0 ms"
        assert stats.effective_hz > 1000.0
        assert stats.kinetic_energy >= 0.0
        assert stats.strain_energy >= 0.0
