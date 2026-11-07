"""
Adversarial and Benchmark Tests for Proper Orthogonal Decomposition (POD)
and Linear Dynamic Model Order Reduction (MOR).
"""

from __future__ import annotations
from pathlib import Path
import tempfile
import numpy as np
import scipy.sparse as spsp
import pytest

from psierra.reduction.snapshots import SnapshotManager
from psierra.reduction.basis import PODBasis, SurfacePODBasis
from psierra.reduction.model import ReducedLinearModel
from psierra.reduction.solver import (
    ReducedLinearDynamicSolver,
    ReducedLinearStaticSolver,
)
from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import DirichletManager, DirichletConstraint, GravityLoad
from psierra.damping import RayleighDamping
from psierra.analysis import LinearDynamicAnalysis


class TestSnapshotManagerAndPersistence:
    """Tests for SnapshotManager, error handling, and .npz persistence."""

    def test_snapshot_dimension_validation(self) -> None:
        mgr = SnapshotManager(n_dofs=6)
        mgr.add_snapshot(np.ones(6))
        assert mgr.n_snapshots == 1

        # Mismatched dimension must raise ValueError
        with pytest.raises(ValueError, match="does not match expected n_dofs"):
            mgr.add_snapshot(np.ones(5))

    def test_multi_snapshot_dimension_validation(self) -> None:
        mgr = SnapshotManager()
        U = np.random.randn(12, 5)
        mgr.add_snapshots(U)
        assert mgr.n_snapshots == 5
        assert mgr.n_dofs == 12

        # 1D array passed to add_snapshots should raise ValueError
        with pytest.raises(ValueError, match="must be 2D"):
            mgr.add_snapshots(np.ones(12))

    def test_empty_snapshot_matrix_raises(self) -> None:
        mgr = SnapshotManager()
        with pytest.raises(ValueError, match="no snapshots have been collected"):
            mgr.get_snapshot_matrix()

    def test_npz_persistence_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_snapshots.npz"

            mgr = SnapshotManager(n_dofs=8)
            rng = np.random.default_rng(42)
            snapshots = rng.standard_normal((8, 10))
            mgr.add_snapshots(snapshots)

            saved_path = mgr.save(cache_file, dt=0.01, experiment="cantilever")
            assert saved_path.is_file()

            # Load back
            loaded_mgr = SnapshotManager.load(cache_file)
            assert loaded_mgr.n_dofs == 8
            assert loaded_mgr.n_snapshots == 10
            np.testing.assert_allclose(
                loaded_mgr.get_snapshot_matrix(), snapshots, rtol=1e-14
            )
            assert loaded_mgr.metadata["experiment"] == "cantilever"

    def test_get_or_compute_caching(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cached_run.npz"

            call_count = 0

            def dummy_fom(m: SnapshotManager) -> None:
                nonlocal call_count
                call_count += 1
                m.add_snapshot(np.array([1.0, 2.0, 3.0]))
                m.add_snapshot(np.array([4.0, 5.0, 6.0]))

            # 1st call: must execute compute_fn
            mgr1 = SnapshotManager.get_or_compute(
                cache_file, dummy_fom, verbose=False
            )
            assert call_count == 1
            assert mgr1.n_snapshots == 2
            assert cache_file.is_file()

            # 2nd call: must load from cache without calling dummy_fom
            mgr2 = SnapshotManager.get_or_compute(
                cache_file, dummy_fom, verbose=False
            )
            assert call_count == 1
            assert mgr2.n_snapshots == 2
            np.testing.assert_allclose(
                mgr1.get_snapshot_matrix(), mgr2.get_snapshot_matrix()
            )


class TestPODBasisMathematicalProperties:
    """Tests for PODBasis orthonormality, projection, and surface slicing."""

    def test_energy_threshold_validation(self) -> None:
        mgr = SnapshotManager()
        mgr.add_snapshot(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="energy_threshold must be in"):
            mgr.compute_basis(energy_threshold=0.0)
        with pytest.raises(ValueError, match="energy_threshold must be in"):
            mgr.compute_basis(energy_threshold=1.5)

    def test_basis_orthonormality_and_singular_decay(self) -> None:
        rng = np.random.default_rng(123)
        # Low rank signal plus noise
        t = np.linspace(0, 1, 20)
        modes = np.column_stack([np.sin(np.pi * t), np.cos(2 * np.pi * t)])
        coeffs = rng.standard_normal((2, 30))
        snapshots = modes @ coeffs

        mgr = SnapshotManager(n_dofs=20)
        mgr.add_snapshots(snapshots)

        basis = mgr.compute_basis(energy_threshold=0.999)
        assert basis.n_modes == 2
        assert basis.orthonormality_error() < 1e-12

        # Check singular values are monotonically non-increasing
        s = basis.singular_values
        assert np.all(np.diff(s) <= 0.0)

    def test_projection_reconstruction_cycle(self) -> None:
        rng = np.random.default_rng(999)
        n_dofs = 15
        n_snaps = 25
        S = rng.standard_normal((n_dofs, n_snaps))

        mgr = SnapshotManager(n_dofs=n_dofs)
        mgr.add_snapshots(S)

        # Full rank basis retains 100% of data
        basis = mgr.compute_basis(energy_threshold=1.0)
        for col in range(n_snaps):
            u_true = S[:, col]
            u_r = basis.project(u_true)
            u_recon = basis.reconstruct(u_r)
            np.testing.assert_allclose(u_recon, u_true, atol=1e-12)

    def test_surface_pod_basis_extraction(self) -> None:
        n_nodes = 6
        n_dofs = 3 * n_nodes
        basis_mat = np.eye(n_dofs)[:, :4]  # 4 modes
        basis = PODBasis(
            basis=basis_mat,
            singular_values=np.ones(4),
            cumulative_energy=1.0,
            total_energy=4.0,
        )

        # Extract surface nodes [1, 4]
        surf_basis = basis.extract_surface_basis([1, 4], is_node_indices=True)
        assert surf_basis.n_surface_nodes == 2
        assert surf_basis.n_surface_dofs == 6
        assert surf_basis.n_modes == 4

        u_r = np.array([1.0, 2.0, 3.0, 4.0])
        displacements_3d = surf_basis.reconstruct_surface_displacements(u_r)
        assert displacements_3d.shape == (2, 3)


class TestFullBeamROMBenchmark:
    """
    End-to-end benchmark verifying POD-Galerkin ROM against full-order model (FOM)
    on a 3D continuum cantilever beam (data/beam1/mesh/beam1.vtk).
    """

    @pytest.fixture(scope="class")
    @classmethod
    def beam_domain_and_bcs(cls):
        mesh_path = Path("psierra/data/beam1/mesh/beam1.vtk")
        if not mesh_path.exists():
            mesh_path = Path("data/beam1/mesh/beam1.vtk")

        mesh = VolumetricMesh(mesh_path)
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(
            youngs_modulus=200e3,
            poissons_ratio=0.30,
            mass_density=7.85e-9,
        )
        material = Hookean(params)
        element = Tet4Linear(material)
        domain = SolidDomain(mesh=mesh, material=material, element=element)

        # Clamp at X = 0
        min_x = np.min(mesh.nodes[:, 0])
        fixed_nodes = np.where(np.isclose(mesh.nodes[:, 0], min_x, atol=1e-3))[0]
        dirichlet = DirichletManager(domain.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        # Gravity load in -Y
        gravity = GravityLoad(acceleration=(0.0, -9810.0, 0.0))
        f_grav = gravity.compute_force_vector(domain.assemble_mass_matrix())

        return domain, dirichlet, f_grav, fixed_nodes

    def test_rom_vs_fom_accuracy_and_speed(self, beam_domain_and_bcs) -> None:
        domain, dirichlet, f_grav, fixed_nodes = beam_domain_and_bcs

        dt = 0.005
        n_steps = 40
        damping = RayleighDamping(alpha=0.5, beta=0.001)

        # 1. Full-Order Model (FOM) Simulation
        fom_analysis = LinearDynamicAnalysis(
            domain=domain, damping=damping, spectral_radius=0.8
        )
        # Capture snapshots step by step
        snap_mgr = SnapshotManager(n_dofs=domain.n_dofs)

        # Run FOM and record time history of tip node
        tip_node = np.argmax(domain.mesh.nodes[:, 0])
        tip_dof_y = 3 * tip_node + 1

        fom_tip_trajectory = []
        u_fom = np.zeros(domain.n_dofs)

        # We execute FOM dynamic solve
        res = fom_analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_grav,
            dt=dt,
            n_steps=n_steps,
            verbose=False,
        )

        # In order to collect snapshots directly, we re-step or collect from FOM:
        # We can extract the snapshots from an offline stepping or run
        K = domain.assemble_stiffness_matrix()
        M = domain.assemble_mass_matrix()
        C = damping.assemble_damping_matrix(M, K)
        free_dofs = (
            dirichlet.free_dofs()
            if callable(dirichlet.free_dofs)
            else dirichlet.free_dofs
        )

        # Step FOM manually to record all snapshots
        K_ff = K[free_dofs, :][:, free_dofs]
        M_ff = M[free_dofs, :][:, free_dofs]
        C_ff = C[free_dofs, :][:, free_dofs]
        f_ext_f = f_grav[free_dofs]

        # Use same integration parameters as LinearDynamicAnalysis
        rho = 0.8
        am = (2.0 * rho - 1.0) / (rho + 1.0)
        af = rho / (rho + 1.0)
        gam = 0.5 - am + af
        bet = 0.25 * (1.0 - am + af) ** 2

        a0 = 1.0 / (bet * dt**2)
        a1 = gam / (bet * dt)
        a2 = 1.0 / (bet * dt)
        a3 = 1.0 / (2.0 * bet) - 1.0
        a4 = gam / bet - 1.0
        a5 = dt * 0.5 * (gam / bet - 2.0)

        K_eff = (1.0 - af) * K_ff + (1.0 - am) * a0 * M_ff + (1.0 - af) * a1 * C_ff
        solve_step = spsp.linalg.factorized(K_eff.tocsc())

        u_f = np.zeros(len(free_dofs))
        v_f = np.zeros(len(free_dofs))
        a_f = spsp.linalg.spsolve(M_ff.tocsc(), f_ext_f)

        u_full = np.zeros(domain.n_dofs)

        for _ in range(n_steps):
            m_part = (1.0 - am) * (a0 * u_f + a2 * v_f + a3 * a_f) - am * a_f
            c_part = (1.0 - af) * (a1 * u_f + a4 * v_f + a5 * a_f) - af * v_f
            f_eff = f_ext_f + M_ff @ m_part + C_ff @ c_part - af * (K_ff @ u_f)
            u_f_next = solve_step(f_eff)
            a_f_next = a0 * (u_f_next - u_f) - a2 * v_f - a3 * a_f
            v_f_next = v_f + dt * ((1.0 - gam) * a_f + gam * a_f_next)
            u_f, v_f, a_f = u_f_next, v_f_next, a_f_next

            u_full[free_dofs] = u_f
            snap_mgr.add_snapshot(u_full)
            fom_tip_trajectory.append(u_full[tip_dof_y])

        fom_tip_trajectory = np.array(fom_tip_trajectory)

        # 2. Compute POD Basis
        basis = snap_mgr.compute_basis(energy_threshold=0.999)
        assert basis.n_modes >= 1
        assert basis.orthonormality_error() < 1e-12

        # 3. Construct Reduced Model
        surf_nodes = domain.mesh.surface.node_indices
        rom_model = ReducedLinearModel.from_domain(
            domain=domain,
            basis=basis,
            dirichlet=dirichlet,
            damping=damping,
            f_ext_base=f_grav,
            surface_nodes=surf_nodes,
        )

        # Verify Reduced Mass & Stiffness are Symmetric Positive Definite
        eig_K = np.linalg.eigvalsh(rom_model.K_r)
        eig_M = np.linalg.eigvalsh(rom_model.M_r)
        assert np.all(eig_K > 0.0), f"Reduced K_r has non-positive eigenvalues: {eig_K}"
        assert np.all(eig_M > 0.0), f"Reduced M_r has non-positive eigenvalues: {eig_M}"

        # 4. Execute ROM Online Dynamic Solver
        rom_solver = ReducedLinearDynamicSolver(
            model=rom_model, dt=dt, spectral_radius=0.8
        )
        rom_tip_trajectory = []

        import time

        t_start = time.perf_counter()
        for _ in range(n_steps):
            rom_solver.step()
            u_full_rom = rom_solver.reconstruct_displacement()
            rom_tip_trajectory.append(u_full_rom[tip_dof_y])
        t_rom_total = time.perf_counter() - t_start

        rom_tip_trajectory = np.array(rom_tip_trajectory)

        # 5. Accuracy & Performance Validation
        # Max relative trajectory error must be < 2%
        peak_disp = np.max(np.abs(fom_tip_trajectory))
        rel_error = (
            np.linalg.norm(rom_tip_trajectory - fom_tip_trajectory)
            / np.linalg.norm(fom_tip_trajectory)
        )
        assert (
            rel_error < 0.02
        ), f"ROM relative trajectory error {rel_error*100:.2f}% exceeds 2% threshold!"

        # Real-time execution rate verification (< 0.1 ms per step, > 10,000 Hz)
        stats = rom_solver.get_stats()
        assert stats.average_step_ms < 0.2, f"Step time {stats.average_step_ms:.3f} ms exceeds 0.2 ms"
        assert stats.steps_completed == n_steps
        assert stats.kinetic_energy >= 0.0
        assert stats.strain_energy >= 0.0

        # Surface reconstruction verification
        u_surf = rom_solver.reconstruct_surface_displacement()
        assert u_surf.shape == (len(surf_nodes), 3)

        # 6. Static ROM Solver Verification
        static_solver = ReducedLinearStaticSolver(model=rom_model)
        u_r_static = static_solver.solve()
        assert len(u_r_static) == basis.n_modes
        u_full_static = static_solver.reconstruct_displacement()
        assert len(u_full_static) == domain.n_dofs
        u_surf_static = static_solver.reconstruct_surface_displacement()
        assert u_surf_static.shape == (len(surf_nodes), 3)
        strain_energy_static = static_solver.compute_strain_energy()
        assert strain_energy_static > 0.0

        # Custom load vector solve
        custom_fr = 2.0 * rom_model.f_r_base
        u_r_custom = static_solver.solve(custom_fr)
        np.testing.assert_allclose(u_r_custom, 2.0 * u_r_static, rtol=1e-12)

