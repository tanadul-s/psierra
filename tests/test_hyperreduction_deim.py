"""
Unit and Verification Tests for Discrete Empirical Interpolation Method (DEIM / Q-DEIM)
Hyper-Reduction Framework.
"""

from __future__ import annotations
from pathlib import Path
import tempfile
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
from psierra.reduction.basis import PODBasis
from psierra.hyperreduction.deim import (
    greedy_deim,
    q_deim,
    DEIMForceBasis,
    ForceSnapshotManager,
    HyperReducedDEIMModel,
)
from psierra.hyperreduction.solver import HyperReducedDynamicSolver


@pytest.fixture
def cantilever_beam() -> tuple[SolidDomain, DirichletManager, PODBasis, np.ndarray]:
    """Sets up a cantilever beam, Dirichlet conditions, and a displacement POD basis."""
    mesh_path = Path("psierra/data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        pytest.skip("beam1.vtk mesh not found")

    mesh = VolumetricMesh(mesh_path)
    params = IsotropicElasticParameters()
    # Soft rubber parameters: E = 0.05 MPa, nu = 0.45, rho = 1.1e-9 tonne/mm³
    params.set_from_youngs_poisson(0.05, 0.45, 1.1e-9)
    mat = NeoHookean(params)
    elem = Tet4TotalLagrangian(mat)
    domain = SolidDomain(mesh=mesh, material=mat, element=elem)

    # Clamped root at Z = min(Z)
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min", tolerance=1e-3)
    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0))

    # Synthetic snapshot generation for modal reduction
    rng = np.random.default_rng(42)
    n_dofs = domain.n_dofs
    free_dofs = dirichlet.free_dofs

    # Create 8 smooth displacement fields (modal-like shapes)
    u_snapshots = []
    z_coords = mesh.nodes[:, 2]
    z_max = np.max(z_coords)
    z_min = np.min(z_coords)
    z_norm = (z_coords - z_min) / max(z_max - z_min, 1e-6)

    for mode_idx in range(1, 9):
        disp_vec = np.zeros(n_dofs, dtype=np.float64)
        # Bending deflection profile: (z/L)^2
        scale = 0.5 * mode_idx
        disp_vec[0::3] = scale * (z_norm**2) * np.sin(mode_idx * np.pi * z_norm * 0.5)
        disp_vec[1::3] = 0.2 * scale * (z_norm**2) * np.cos(mode_idx * np.pi * z_norm * 0.5)
        disp_vec[2::3] = -0.1 * scale * (z_norm**2)
        disp_vec[dirichlet.prescribed_dofs] = 0.0
        u_snapshots.append(disp_vec)

    snap_manager = SnapshotManager(n_dofs=n_dofs)
    for u in u_snapshots:
        snap_manager.add_snapshot(u)

    basis = snap_manager.compute_basis(energy_threshold=0.999, max_modes=5)
    U_snapshots = np.column_stack(u_snapshots)

    return domain, dirichlet, basis, U_snapshots


class TestDEIMAlgorithms:
    """Mathematical validation tests for Greedy DEIM and Q-DEIM index selection."""

    def test_greedy_deim_mathematical_properties(self) -> None:
        """Verifies greedy index selection, uniqueness, and non-singularity."""
        rng = np.random.default_rng(123)
        n, m = 100, 6
        # Generate random orthonormal matrix
        A = rng.standard_normal((n, m))
        Q, _ = np.linalg.qr(A)

        wp = greedy_deim(Q, m_select=m)

        assert len(wp) == m
        assert len(np.unique(wp)) == m  # All indices strictly distinct
        assert np.all(wp >= 0) and np.all(wp < n)

        # First index must match argmax |Q[:, 0]|
        assert wp[0] == np.argmax(np.abs(Q[:, 0]))

        # Submatrix P^T * Q must be non-singular
        P_Q = Q[wp, :]
        assert np.linalg.matrix_rank(P_Q) == m
        cond = np.linalg.cond(P_Q)
        assert np.isfinite(cond)
        assert cond < 1e6

    def test_q_deim_mathematical_properties(self) -> None:
        """Verifies Q-DEIM QR-pivoting selection and condition number."""
        rng = np.random.default_rng(456)
        n, m = 120, 8
        A = rng.standard_normal((n, m))
        Q, _ = np.linalg.qr(A)

        wp = q_deim(Q, m_select=m)

        assert len(wp) == m
        assert len(np.unique(wp)) == m
        assert np.all(wp >= 0) and np.all(wp < n)

        P_Q = Q[wp, :]
        assert np.linalg.matrix_rank(P_Q) == m
        cond = np.linalg.cond(P_Q)
        assert np.isfinite(cond)
        assert cond < 1e6

    def test_deim_invalid_inputs(self) -> None:
        """Checks error handling on malformed dimensions."""
        with pytest.raises(ValueError, match="must be 2D"):
            greedy_deim(np.zeros(10))

        with pytest.raises(ValueError, match="Empty basis matrix"):
            greedy_deim(np.zeros((0, 0)))

        with pytest.raises(ValueError, match="m_select must be in"):
            greedy_deim(np.eye(5), m_select=10)


class TestForceSnapshotManager:
    """Tests for force snapshot collection and out-of-core HDF5 streaming."""

    def test_in_memory_collection(self) -> None:
        mgr = ForceSnapshotManager(n_dofs=30)
        assert mgr.n_snapshots == 0

        f1 = np.ones(30)
        mgr.add_snapshot(f1)
        assert mgr.n_snapshots == 1

        F = np.ones((30, 4))
        mgr.add_snapshots(F)
        assert mgr.n_snapshots == 5

        S = mgr.get_snapshot_matrix()
        assert S.shape == (30, 5)

        with pytest.raises(ValueError, match="does not match expected n_dofs"):
            mgr.add_snapshot(np.ones(20))

    def test_out_of_core_hdf5_streaming(self) -> None:
        """Verifies direct HDF5 streaming and memory-independent resizing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_file = Path(tmpdir) / "test_force_snapshots.h5"

            with ForceSnapshotManager(n_dofs=50) as mgr:
                mgr.enable_hdf5_streaming(h5_file, dataset_name="f_int", chunk_size=4)

                for i in range(12):
                    f_snap = float(i + 1) * np.ones(50)
                    mgr.add_snapshot(f_snap)

                assert mgr.n_snapshots == 12
                S_h5 = mgr.get_snapshot_matrix()
                assert S_h5.shape == (50, 12)
                assert np.allclose(S_h5[:, 0], 1.0)
                assert np.allclose(S_h5[:, 11], 12.0)

            # File should exist on disk and be readable
            assert h5_file.is_file()


class TestDEIMForceBasis:
    """Tests for DEIMForceBasis container and persistence."""

    def test_persistence_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test_deim_basis.npz"

            rng = np.random.default_rng(789)
            basis_mat = rng.standard_normal((100, 5))
            Q, _ = np.linalg.qr(basis_mat)
            wp = greedy_deim(Q, m_select=5)

            basis_obj = DEIMForceBasis(
                basis=Q,
                interpolation_indices=wp,
                singular_values=np.array([10.0, 5.0, 2.0, 1.0, 0.5]),
                cumulative_energy=0.9995,
                total_energy=130.25,
                method="greedy",
                metadata={"test": "persistence"},
            )

            assert basis_obj.n_dofs == 100
            assert basis_obj.n_modes == 5
            assert basis_obj.n_sampled == 5
            assert basis_obj.condition_number > 0.0

            saved = basis_obj.save(archive_path)
            assert saved.is_file()

            loaded = DEIMForceBasis.load(archive_path)
            assert loaded.n_dofs == 100
            assert loaded.n_modes == 5
            assert np.allclose(loaded.basis, Q)
            assert np.array_equal(loaded.interpolation_indices, wp)
            assert loaded.method == "greedy"
            assert loaded.metadata.get("test") == "persistence"

    def test_recompute_indices(self) -> None:
        rng = np.random.default_rng(101)
        Q, _ = np.linalg.qr(rng.standard_normal((80, 4)))
        basis_obj = DEIMForceBasis(
            basis=Q,
            interpolation_indices=greedy_deim(Q),
            singular_values=np.array([4.0, 3.0, 2.0, 1.0]),
            cumulative_energy=0.999,
            total_energy=30.0,
            method="greedy",
        )
        assert basis_obj.method == "greedy"
        basis_obj.recompute_indices(method="qdeim")
        assert basis_obj.method == "qdeim"
        assert len(basis_obj.interpolation_indices) == 4


class TestHyperReducedDEIMModel:
    """Integration tests for HyperReducedDEIMModel on a continuum cantilever beam."""

    def test_deim_model_construction_and_force_fidelity(
        self, cantilever_beam: tuple[SolidDomain, DirichletManager, PODBasis, np.ndarray]
    ) -> None:
        domain, dirichlet, pod_basis, u_snapshots = cantilever_beam

        # 1. Harvest force snapshots from displacement trajectory
        force_mgr = ForceSnapshotManager(n_dofs=domain.n_dofs)
        force_mgr.collect_from_displacements(domain, u_snapshots)
        assert force_mgr.n_snapshots == u_snapshots.shape[1]

        # 2. Compute DEIM basis
        deim_basis = force_mgr.compute_basis(energy_threshold=0.999, max_modes=5, method="greedy")
        assert deim_basis.n_modes >= 1
        assert deim_basis.n_sampled == deim_basis.n_modes

        # 3. Construct HyperReducedDEIMModel
        deim_model = HyperReducedDEIMModel(
            domain=domain,
            basis=pod_basis,
            deim_basis=deim_basis,
            dirichlet=dirichlet,
            symmetrize_stiffness=True,
        )

        assert deim_model.n_modes == pod_basis.n_modes
        assert deim_model.n_sampled_dofs == deim_basis.n_sampled
        assert deim_model.n_sampled_elements < domain.mesh.n_elements
        assert deim_model.compression_ratio < 1.0
        assert deim_model.speedup_factor > 1.0

        # 4. Evaluate forces and stiffness at origin
        u_r_0 = np.zeros(deim_model.n_modes)
        f_r_0, K_r_0 = deim_model.evaluate_forces_and_stiffness(u_r_0)
        assert f_r_0.shape == (deim_model.n_modes,)
        assert K_r_0.shape == (deim_model.n_modes, deim_model.n_modes)
        assert np.allclose(f_r_0, 0.0, atol=1e-10)
        # Stiffness must be strictly positive-definite and symmetric
        assert np.allclose(K_r_0, K_r_0.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(K_r_0)
        assert np.all(eigvals > 0.0)

        # 5. Evaluate forces at a perturbed state and compare with projected full-order force
        u_r_test = 0.05 * np.ones(deim_model.n_modes)
        f_r_deim, K_r_deim = deim_model.evaluate_forces_and_stiffness(u_r_test)

        # Full order projected ground truth
        u_full_test = deim_model.reconstruct_displacement(u_r_test)
        domain.update_state(u=u_full_test)
        f_full_int = domain.assemble_internal_forces()
        f_r_exact = deim_model.Phi_f.T @ f_full_int[deim_model.free_dofs]

        # DEIM force should closely approximate the exact projection
        rel_error = np.linalg.norm(f_r_deim - f_r_exact) / np.linalg.norm(f_r_exact)
        assert rel_error < 0.15, f"DEIM relative error {rel_error:.3e} exceeds expected tolerance."

    def test_deim_dynamic_solver_stepping(
        self, cantilever_beam: tuple[SolidDomain, DirichletManager, PODBasis, np.ndarray]
    ) -> None:
        """Verifies HyperReducedDynamicSolver executes real-time Generalized-alpha time steps with DEIM model."""
        domain, dirichlet, pod_basis, u_snapshots = cantilever_beam

        force_mgr = ForceSnapshotManager(n_dofs=domain.n_dofs)
        force_mgr.collect_from_displacements(domain, u_snapshots)
        deim_basis = force_mgr.compute_basis(energy_threshold=0.999, max_modes=5, method="qdeim")

        gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
        f_gravity = gravity.compute_force_vector(domain.assemble_mass_matrix())
        damping = RayleighDamping(alpha=1.0, beta=0.001)

        deim_model = HyperReducedDEIMModel(
            domain=domain,
            basis=pod_basis,
            deim_basis=deim_basis,
            dirichlet=dirichlet,
            damping=damping,
            f_ext_base=f_gravity,
            symmetrize_stiffness=True,
        )

        solver = HyperReducedDynamicSolver(
            model=deim_model,
            dt=0.002,
            spectral_radius=0.8,
            nr_max_iter=10,
            nr_tol=1e-4,
        )

        assert solver.r == deim_model.n_modes
        assert solver.steps_completed == 0

        # Step through 5 dynamic time steps
        for step in range(5):
            u_r = solver.step()
            assert u_r.shape == (solver.r,)
            assert np.all(np.isfinite(u_r))

        assert solver.steps_completed == 5
        assert solver.sim_time == pytest.approx(0.010, rel=1e-5)

        stats = solver.get_stats()
        assert stats.steps_completed == 5
        assert stats.average_step_ms >= 0.0
        assert stats.effective_hz > 0.0

        # Reconstructed displacement
        u_full = solver.reconstruct_displacement()
        assert u_full.shape == (domain.n_dofs,)
        assert np.all(np.isfinite(u_full))

    def test_deim_static_and_quasi_static_solvers(
        self, cantilever_beam: tuple[SolidDomain, DirichletManager, PODBasis, np.ndarray]
    ) -> None:
        """Verifies HyperReducedStaticSolver and HyperReducedQuasiStaticSolver with DEIM."""
        from psierra.hyperreduction.solver import (
            HyperReducedStaticSolver,
            HyperReducedQuasiStaticSolver,
        )

        domain, dirichlet, pod_basis, u_snapshots = cantilever_beam

        force_mgr = ForceSnapshotManager(n_dofs=domain.n_dofs)
        force_mgr.collect_from_displacements(domain, u_snapshots)
        deim_basis = force_mgr.compute_basis(energy_threshold=0.999, max_modes=5, method="greedy")

        gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
        f_gravity = gravity.compute_force_vector(domain.assemble_mass_matrix())

        deim_model = HyperReducedDEIMModel(
            domain=domain,
            basis=pod_basis,
            deim_basis=deim_basis,
            dirichlet=dirichlet,
            f_ext_base=f_gravity,
            symmetrize_stiffness=True,
        )

        # 1. Static solve
        static_solver = HyperReducedStaticSolver(model=deim_model, tol=1e-5)
        u_r_stat = static_solver.solve(n_increments=2)
        assert u_r_stat.shape == (deim_model.n_modes,)
        assert np.all(np.isfinite(u_r_stat))
        res_stat, _ = deim_model.evaluate_residual(u_r_stat)
        rel_res = np.linalg.norm(res_stat) / max(float(np.linalg.norm(deim_model.f_r_base)), 1.0)
        assert rel_res < 1e-3

        # 2. Quasi-static solve
        qs_solver = HyperReducedQuasiStaticSolver(model=deim_model, tol=1e-5)
        u_r_qs = qs_solver.step(dt=0.01)
        assert u_r_qs.shape == (deim_model.n_modes,)
        assert np.all(np.isfinite(u_r_qs))

    def test_deim_history_variables_delegation(
        self, cantilever_beam: tuple[SolidDomain, DirichletManager, PODBasis, np.ndarray]
    ) -> None:
        """Verifies viscoelastic history state hooks are correctly forwarded."""
        domain, dirichlet, pod_basis, u_snapshots = cantilever_beam

        force_mgr = ForceSnapshotManager(n_dofs=domain.n_dofs)
        force_mgr.collect_from_displacements(domain, u_snapshots)
        deim_basis = force_mgr.compute_basis(energy_threshold=0.999, max_modes=5, method="qdeim")

        deim_model = HyperReducedDEIMModel(
            domain=domain,
            basis=pod_basis,
            deim_basis=deim_basis,
            dirichlet=dirichlet,
        )

        # Ensure history hook methods do not raise and delegate safely
        deim_model.set_time_step(0.01)
        deim_model.commit_state()
        deim_model.revert_state()
        deim_model.reset_state()
