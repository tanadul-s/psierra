from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from psierra.domain.solid import SolidDomain
from psierra.element import Tet4TotalLagrangian
from psierra.material import NeoHookean, MooneyRivlin, Yeoh
from psierra.mesh.volumetric import VolumetricMesh
from psierra.boundary.dirichlet import DirichletConstraint, DirichletManager
from psierra.boundary.load import NodalForce
from psierra.boundary.selector import select_nodes_by_plane
from psierra.damping.rayleigh import RayleighDamping
from psierra.analysis.nonlinear_static import (
    NonlinearStaticAnalysis,
    NonlinearStaticResult,
)
from psierra.analysis.nonlinear_dynamic import (
    NonlinearDynamicAnalysis,
    NonlinearDynamicResult,
)


class TestNonlinearAnalysis:
    """Verifies nonlinear Total Lagrangian static and dynamic procedures."""

    def test_nonlinear_static_newton_raphson_convergence(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        mesh = five_tet_cube_mesh
        element = Tet4TotalLagrangian(neohookean_material)
        domain = SolidDomain(mesh, neohookean_material, element)

        # Clamp bottom face (Z = 0)
        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        clamp = DirichletConstraint(
            node_indices=fixed_nodes, components=(0, 1, 2), value=0.0
        )
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(clamp)

        # Apply tensile force on top face (Z = 1)
        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        total_fz = 50.0  # Moderate load for 10 kPa material
        load = NodalForce(
            node_indices=top_nodes, force_vector=[0.0, 0.0, total_fz / len(top_nodes)]
        )
        f_ext = load.compute_force_vector(mesh.n_dofs)

        analysis = NonlinearStaticAnalysis(domain)
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            n_steps=4,
            nr_iterations=15,
            nr_tol=1e-4,
            verbose=True,
        )

        assert isinstance(result, NonlinearStaticResult)
        assert result.converged is True
        assert result.steps_completed == 4

        # Verify internal forces equal external forces on free DOFs
        f_int = domain.assemble_internal_forces()
        free_dofs = dirichlet.free_dofs
        residual_norm = np.linalg.norm(f_ext[free_dofs] - f_int[free_dofs])
        rel_residual = residual_norm / np.linalg.norm(f_ext[free_dofs])
        assert rel_residual < 1e-4

    def test_nonlinear_dynamic_convergence(
        self,
        five_tet_cube_mesh: VolumetricMesh,
        neohookean_material: NeoHookean,
        tmp_path: Path,
    ) -> None:
        mesh = five_tet_cube_mesh
        element = Tet4TotalLagrangian(neohookean_material)
        domain = SolidDomain(mesh, neohookean_material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        load = NodalForce(top_nodes, [0.0, 0.0, 20.0 / len(top_nodes)])
        f_ext = load.compute_force_vector(mesh.n_dofs)

        damping = RayleighDamping(alpha=1.0, beta=0.001)
        analysis = NonlinearDynamicAnalysis(domain, damping=damping)

        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.005,
            n_steps=5,
            nr_iterations=10,
            nr_tol=1e-4,
            save_directory=tmp_path / "nl_dyn_out",
            verbose=True,
        )

        assert isinstance(result, NonlinearDynamicResult)
        assert result.converged is True
        assert result.steps_completed == 5
        assert np.linalg.norm(domain.displacement) > 0.0
        assert np.linalg.norm(domain.velocity) > 0.0
        assert (tmp_path / "nl_dyn_out" / "result.pvd").exists()

    def test_nonlinear_dynamic_steady_state_early_exit(
        self,
        five_tet_cube_mesh: VolumetricMesh,
        neohookean_material: NeoHookean,
        tmp_path: Path,
    ) -> None:
        mesh = five_tet_cube_mesh
        element = Tet4TotalLagrangian(neohookean_material)
        domain = SolidDomain(mesh, neohookean_material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        load = NodalForce(top_nodes, [0.0, 0.0, 15.0 / len(top_nodes)])
        f_ext = load.compute_force_vector(mesh.n_dofs)

        # Critical damping to rapidly settle to static equilibrium (Tn ~ 2s)
        damping = RayleighDamping(alpha=4.0, beta=0.01)
        analysis = NonlinearDynamicAnalysis(domain, damping=damping)

        out_dir = tmp_path / "nl_steady_out"
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.05,
            n_steps=100,
            nr_iterations=15,
            nr_tol=1e-4,
            steady_state_tol=1e-2,
            consecutive_hold_steps=2,
            save_interval=5,
            save_directory=out_dir,
            verbose=False,
        )

        assert isinstance(result, NonlinearDynamicResult)
        assert result.converged is True
        assert result.steady_state_converged is True
        assert result.steady_state_step is not None
        assert result.steps_completed < 100
        assert result.steps_completed == result.steady_state_step

        # Verify that the final snapshot file was saved even if not a multiple of 5
        last_file = result.time_series_files[-1]
        assert last_file.exists()
        assert f"{result.steady_state_step:05d}" in last_file.name

    def test_nonlinear_static_snapshot_saving(
        self,
        five_tet_cube_mesh: VolumetricMesh,
        neohookean_material: NeoHookean,
        tmp_path: Path,
    ) -> None:
        mesh = five_tet_cube_mesh
        element = Tet4TotalLagrangian(neohookean_material)
        domain = SolidDomain(mesh, neohookean_material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(
                node_indices=fixed_nodes, components=(0, 1, 2), value=0.0
            )
        )

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        load = NodalForce(
            node_indices=top_nodes, force_vector=[0.0, 0.0, 20.0 / len(top_nodes)]
        )
        f_ext = load.compute_force_vector(mesh.n_dofs)

        out_dir = tmp_path / "nl_static_snaps"
        analysis = NonlinearStaticAnalysis(domain)
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            n_steps=6,
            nr_iterations=15,
            nr_tol=1e-4,
            save_interval=2,
            save_directory=out_dir,
            verbose=True,
        )

        assert result.converged is True
        assert (out_dir / "result.pvd").exists()
        # Step 0, step 2, step 4, step 6 = 4 snapshots
        assert len(result.time_series_files) == 4

    def test_nonlinear_static_mooney_rivlin(
        self,
        five_tet_cube_mesh: VolumetricMesh,
        mooney_rivlin_material: MooneyRivlin,
    ) -> None:
        """Verifies incremental Newton-Raphson nonlinear static solve with Mooney-Rivlin material."""
        mesh = five_tet_cube_mesh
        element = Tet4TotalLagrangian(mooney_rivlin_material)
        domain = SolidDomain(mesh, mooney_rivlin_material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(node_indices=fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        load = NodalForce(
            node_indices=top_nodes, force_vector=[0.0, 0.0, 30.0 / len(top_nodes)]
        )
        f_ext = load.compute_force_vector(mesh.n_dofs)

        analysis = NonlinearStaticAnalysis(domain)
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            n_steps=4,
            nr_iterations=15,
            nr_tol=1e-4,
            verbose=False,
        )

        assert isinstance(result, NonlinearStaticResult)
        assert result.converged is True
        assert result.steps_completed == 4

        # Verify internal force equilibrium
        f_int = domain.assemble_internal_forces()
        free_dofs = dirichlet.free_dofs
        residual_norm = np.linalg.norm(f_ext[free_dofs] - f_int[free_dofs])
        rel_residual = residual_norm / np.linalg.norm(f_ext[free_dofs])
        assert rel_residual < 1e-4

    def test_nonlinear_static_yeoh(
        self,
        five_tet_cube_mesh: VolumetricMesh,
        yeoh_material: Yeoh,
    ) -> None:
        """Verifies incremental Newton-Raphson nonlinear static solve with Yeoh strain-stiffening material."""
        mesh = five_tet_cube_mesh
        element = Tet4TotalLagrangian(yeoh_material)
        domain = SolidDomain(mesh, yeoh_material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(node_indices=fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        load = NodalForce(
            node_indices=top_nodes, force_vector=[0.0, 0.0, 30.0 / len(top_nodes)]
        )
        f_ext = load.compute_force_vector(mesh.n_dofs)

        analysis = NonlinearStaticAnalysis(domain)
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            n_steps=4,
            nr_iterations=15,
            nr_tol=1e-4,
            verbose=False,
        )

        assert isinstance(result, NonlinearStaticResult)
        assert result.converged is True
        assert result.steps_completed == 4

        # Verify internal force equilibrium
        f_int = domain.assemble_internal_forces()
        free_dofs = dirichlet.free_dofs
        residual_norm = np.linalg.norm(f_ext[free_dofs] - f_int[free_dofs])
        rel_residual = residual_norm / np.linalg.norm(f_ext[free_dofs])
        assert rel_residual < 1e-4

    def test_nonlinear_dynamic_spectral_radius_cases(
        self,
        five_tet_cube_mesh: VolumetricMesh,
        neohookean_material: NeoHookean,
    ) -> None:
        """Verifies NonlinearDynamicAnalysis Generalized-alpha parameterization."""
        mesh = five_tet_cube_mesh
        element = Tet4TotalLagrangian(neohookean_material)
        domain = SolidDomain(mesh, neohookean_material, element)

        # Invalid spectral_radius validation
        import pytest

        with pytest.raises(ValueError, match="spectral_radius must be in"):
            NonlinearDynamicAnalysis(domain, spectral_radius=-0.5)
        with pytest.raises(ValueError, match="spectral_radius must be in"):
            NonlinearDynamicAnalysis(domain, spectral_radius=2.0)

        # Check default rho_inf = 0.8
        default_solver = NonlinearDynamicAnalysis(domain)
        assert default_solver.spectral_radius == 0.8
        assert np.isclose(default_solver.alpha_m, 1.0 / 3.0)
        assert np.isclose(default_solver.alpha_f, 4.0 / 9.0)
        assert np.isclose(default_solver.gamma, 11.0 / 18.0)
        assert np.isclose(default_solver.beta, 100.0 / 324.0)

        # Verify solve works cleanly with custom rho_inf = 0.5 (HHT-alpha)
        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(node_indices=fixed_nodes, components=(0, 1, 2), value=0.0)
        )
        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        load = NodalForce(node_indices=top_nodes, force_vector=[0.0, 0.0, 10.0 / len(top_nodes)])
        f_ext = load.compute_force_vector(mesh.n_dofs)

        solver_hht = NonlinearDynamicAnalysis(domain, spectral_radius=0.5)
        assert np.isclose(solver_hht.alpha_m, 0.0)
        assert np.isclose(solver_hht.alpha_f, 1.0 / 3.0)
        assert np.isclose(solver_hht.gamma, 5.0 / 6.0)
        assert np.isclose(solver_hht.beta, 4.0 / 9.0)

        res = solver_hht.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.01,
            n_steps=5,
            nr_iterations=10,
            verbose=False,
        )
        assert res.converged is True
        assert res.steps_completed == 5

    def test_nonlinear_static_displacement_control_torsion(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        """Verifies displacement-controlled nonlinear static solve with step-dependent callable Dirichlet BCs."""
        mesh = five_tet_cube_mesh
        element = Tet4TotalLagrangian(neohookean_material)
        domain = SolidDomain(mesh, neohookean_material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        top_X = mesh.nodes[top_nodes, 0]
        top_Y = mesh.nodes[top_nodes, 1]

        clamp_base = DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)

        theta_max = np.deg2rad(30.0)

        def step_dirichlet(step: int, n_steps: int) -> DirichletManager:
            theta = (step / n_steps) * theta_max
            ux = top_X * (np.cos(theta) - 1.0) - top_Y * np.sin(theta)
            uy = top_X * np.sin(theta) + top_Y * (np.cos(theta) - 1.0)

            mgr = DirichletManager(mesh.n_dofs)
            mgr.add_constraint(clamp_base)
            mgr.add_constraint(DirichletConstraint(top_nodes, components=(0,), value=ux))
            mgr.add_constraint(DirichletConstraint(top_nodes, components=(1,), value=uy))
            # uz is left free for Poynting contraction/warping
            return mgr

        solver = NonlinearStaticAnalysis(domain)
        result = solver.solve(
            dirichlet=step_dirichlet,
            f_ext=None,  # Pure displacement control
            n_steps=3,
            nr_iterations=15,
            nr_tol=1e-4,
            verbose=False,
        )

        assert result.converged is True
        assert result.steps_completed == 3
        assert np.all(np.isfinite(result.displacement))
        assert np.max(np.abs(result.displacement)) > 0.0

    def test_nonlinear_static_tet10_newton_raphson(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        """Verifies Newton-Raphson nonlinear static convergence for Tet10 (both TL and UL)."""
        from psierra.element import Tet10TotalLagrangian, Tet10UpdatedLagrangian

        mesh10 = five_tet_cube_mesh.to_quadratic()
        fixed_nodes = select_nodes_by_plane(mesh10, plane="xy", bound="min")
        top_nodes = select_nodes_by_plane(mesh10, plane="xy", bound="max")

        clamp = DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        dm = DirichletManager(mesh10.n_dofs)
        dm.add_constraint(clamp)

        load = NodalForce(top_nodes, force_vector=[0.0, 0.0, 50.0 / len(top_nodes)])
        f_ext = load.compute_force_vector(mesh10.n_dofs)

        # 1. Total Lagrangian
        dom_tl = SolidDomain(mesh10.copy(), neohookean_material, Tet10TotalLagrangian(neohookean_material))
        res_tl = NonlinearStaticAnalysis(dom_tl).solve(
            dirichlet=dm, f_ext=f_ext, n_steps=2, nr_tol=1e-4, verbose=False
        )
        assert res_tl.converged is True

        # 2. Updated Lagrangian
        dom_ul = SolidDomain(mesh10.copy(), neohookean_material, Tet10UpdatedLagrangian(neohookean_material))
        res_ul = NonlinearStaticAnalysis(dom_ul).solve(
            dirichlet=dm, f_ext=f_ext, n_steps=2, nr_tol=1e-4, verbose=False
        )
        assert res_ul.converged is True

        # Machine-precision equivalence between TL and UL
        np.testing.assert_allclose(res_tl.displacement, res_ul.displacement, atol=1e-10)
