from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from psierra.mesh.volumetric import VolumetricMesh
from psierra.material.parameters import IsotropicElasticParameters
from psierra.material.neohookean import NeoHookean
from psierra.material.viscoelastic import Viscoelastic, PronyTerm
from psierra.element.tet4_total_lagrangian import Tet4TotalLagrangian
from psierra.domain.solid import SolidDomain
from psierra.boundary import DirichletManager, DirichletConstraint, GravityLoad
from psierra.boundary.selector import select_nodes_by_plane
from psierra.reduction.snapshots import SnapshotManager
from psierra.hyperreduction.ecsw import (
    ReducedIntegrationDomain,
    ECSWTrainingMatrix,
    ECSWOptimizer,
)
from psierra.hyperreduction.model import HyperReducedECSWModel
from psierra.hyperreduction.solver import (
    HyperReducedStaticSolver,
    HyperReducedQuasiStaticSolver,
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
    # E = 21 kPa = 0.021 MPa, nu = 0.45, rho = 1.1e-9 tonne/mm³
    params.set_from_youngs_poisson(0.021, 0.45, 1.1e-9)
    material = NeoHookean(params)
    elem = Tet4TotalLagrangian(material)
    domain = SolidDomain(mesh=mesh, material=material, element=elem)

    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min", tolerance=1e-3)
    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0))

    return domain, dirichlet, mesh


class TestHyperReducedQuasiStaticSolver:
    """Mathematical verification and integration tests for HyperReducedQuasiStaticSolver."""

    def test_hyperreduced_quasi_static_elastic_solve(self, beam_domain) -> None:
        """
        Verifies that on an elastic model, HyperReducedQuasiStaticSolver produces
        the exact same equilibrium state as HyperReducedStaticSolver.
        """
        domain, dirichlet, mesh = beam_domain

        rng = np.random.default_rng(42)
        r = 3
        Phi_raw = rng.standard_normal((domain.n_dofs, r))
        Phi_raw[dirichlet.prescribed_dofs, :] = 0.0
        Phi, _ = np.linalg.qr(Phi_raw)
        mgr = SnapshotManager(n_dofs=domain.n_dofs)
        mgr.add_snapshots(Phi)
        basis = mgr.compute_basis(energy_threshold=0.999)

        snap = basis.basis_matrix @ np.array([0.2, -0.1, 0.15])
        A, b = ECSWTrainingMatrix.build(domain=domain, basis=basis, snapshots=snap)
        rid = ECSWOptimizer(tolerance=0.03, verbose=False).optimize(A, b)

        f_grav = GravityLoad(acceleration=(0.0, -9810.0, 0.0)).compute_force_vector(
            domain.assemble_mass_matrix()
        )

        model = HyperReducedECSWModel(
            domain=domain,
            basis=basis,
            rid=rid,
            dirichlet=dirichlet,
            f_ext_base=f_grav,
        )

        # 1. Hyper-reduced static solver
        static_solver = HyperReducedStaticSolver(model=model, max_iter=25, tol=1e-6)
        u_r_static = static_solver.solve(verbose=False)

        # 2. Hyper-reduced quasi-static solver with 4 steps over 1.0s
        qs_solver = HyperReducedQuasiStaticSolver(model=model, max_iter=25, tol=1e-6)
        u_r_qs = qs_solver.solve(
            total_time=1.0,
            dt=0.25,
            load_schedule=lambda t: t / 1.0,
            verbose=False,
        )

        assert u_r_qs.shape == (r,)
        assert np.allclose(u_r_qs, u_r_static, atol=1e-5)

        # Reconstructed full-order displacement check
        u_full_qs = qs_solver.reconstruct_displacement()
        assert u_full_qs.shape == (domain.n_dofs,)
        assert np.allclose(u_full_qs[dirichlet.prescribed_dofs], 0.0)

    def test_hyperreduced_quasi_static_viscoelastic_relaxation(self, beam_domain) -> None:
        """
        Verifies that HyperReducedQuasiStaticSolver simulates rate-dependent
        viscoelastic stress relaxation and creep on the reduced subspace R^r.
        """
        domain_base, dirichlet, mesh = beam_domain

        # Construct viscoelastic domain
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(0.021, 0.45, 1.1e-9)
        base_mat = NeoHookean(params)
        prony = [PronyTerm(relative_modulus=0.5, relaxation_time=0.1)]
        visco_mat = Viscoelastic(base_mat, prony)
        elem = Tet4TotalLagrangian(visco_mat)
        domain = SolidDomain(mesh=mesh.copy(), material=visco_mat, element=elem)

        rng = np.random.default_rng(123)
        r = 3
        Phi_raw = rng.standard_normal((domain.n_dofs, r))
        Phi_raw[dirichlet.prescribed_dofs, :] = 0.0
        Phi, _ = np.linalg.qr(Phi_raw)
        mgr = SnapshotManager(n_dofs=domain.n_dofs)
        mgr.add_snapshots(Phi)
        basis = mgr.compute_basis(energy_threshold=0.999)

        snap = basis.basis_matrix @ np.array([0.15, -0.08, 0.12])
        A, b = ECSWTrainingMatrix.build(domain=domain, basis=basis, snapshots=snap)
        rid = ECSWOptimizer(tolerance=0.03, verbose=False).optimize(A, b)

        f_grav = GravityLoad(acceleration=(0.0, -9810.0, 0.0)).compute_force_vector(
            domain.assemble_mass_matrix()
        )

        model = HyperReducedECSWModel(
            domain=domain,
            basis=basis,
            rid=rid,
            dirichlet=dirichlet,
            f_ext_base=f_grav,
        )

        qs_solver = HyperReducedQuasiStaticSolver(model=model, max_iter=25, tol=1e-5)

        disp_history: list[float] = []

        def track_disp(step: int, t: float, u_r: np.ndarray) -> None:
            disp_history.append(float(np.linalg.norm(u_r)))

        # Solve 6 steps under constant held gravity load f_grav
        qs_solver.solve(
            total_time=0.3,
            dt=0.05,
            verbose=False,
            callback=track_disp,
        )

        assert len(disp_history) == 6
        # Due to viscoelastic relaxation, reduced displacement magnitude must monotonically increase
        for i in range(len(disp_history) - 1):
            assert disp_history[i + 1] >= disp_history[i] - 1e-12

        # Final displacement must exceed initial instantaneous displacement
        assert disp_history[-1] > disp_history[0] * 1.2

    def test_hyperreduced_quasi_static_telemetry_and_reset(self, beam_domain) -> None:
        """Verifies solver performance statistics and reset_state functionality."""
        domain, dirichlet, mesh = beam_domain

        rng = np.random.default_rng(77)
        r = 2
        Phi_raw = rng.standard_normal((domain.n_dofs, r))
        Phi_raw[dirichlet.prescribed_dofs, :] = 0.0
        Phi, _ = np.linalg.qr(Phi_raw)
        mgr = SnapshotManager(n_dofs=domain.n_dofs)
        mgr.add_snapshots(Phi)
        basis = mgr.compute_basis(energy_threshold=0.999)

        snap = basis.basis_matrix @ np.array([0.1, -0.05])
        A, b = ECSWTrainingMatrix.build(domain=domain, basis=basis, snapshots=snap)
        rid = ECSWOptimizer(tolerance=0.04, verbose=False).optimize(A, b)

        f_grav = GravityLoad(acceleration=(0.0, -9810.0, 0.0)).compute_force_vector(
            domain.assemble_mass_matrix()
        )

        model = HyperReducedECSWModel(
            domain=domain,
            basis=basis,
            rid=rid,
            dirichlet=dirichlet,
            f_ext_base=f_grav,
        )

        qs_solver = HyperReducedQuasiStaticSolver(model=model, max_iter=20, tol=1e-5)
        qs_solver.solve(total_time=0.1, dt=0.02, verbose=False)

        stats = qs_solver.get_stats()
        assert stats.steps_completed == 5
        assert stats.sim_time == pytest.approx(0.1, abs=1e-8)
        assert stats.effective_hz > 500.0  # Dense r x r solves in < 2 ms
        assert stats.strain_energy > 0.0

        # Reset state
        qs_solver.reset_state()
        assert np.allclose(qs_solver.u_r, 0.0)
        assert qs_solver.steps_completed == 0
        assert qs_solver.sim_time == 0.0
