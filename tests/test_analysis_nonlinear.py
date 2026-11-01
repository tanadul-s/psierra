from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from psierra.domain.solid import SolidDomain
from psierra.element import Tet4TotalLagrangian
from psierra.material.neohookean import NeoHookean
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
