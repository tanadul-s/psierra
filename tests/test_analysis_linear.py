from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from psierra.domain.solid import SolidDomain
from psierra.element import Tet4Linear, Tet10Linear
from psierra.material.hookean import Hookean
from psierra.material.parameters import IsotropicElasticParameters
from psierra.mesh.volumetric import VolumetricMesh
from psierra.boundary.dirichlet import DirichletConstraint, DirichletManager
from psierra.boundary.load import NodalForce
from psierra.boundary.selector import select_nodes_by_plane
from psierra.damping.rayleigh import RayleighDamping
from psierra.analysis.linear_static import LinearStaticAnalysis, LinearStaticResult
from psierra.analysis.linear_dynamic import LinearDynamicAnalysis, LinearDynamicResult


class TestLinearAnalysis:
    """Verifies linear static and dynamic finite element analysis procedures."""

    def test_linear_static_equilibrium_and_energy(
        self, five_tet_cube_mesh: VolumetricMesh, hookean_material: Hookean
    ) -> None:
        mesh = five_tet_cube_mesh
        element = Tet4Linear(hookean_material)
        domain = SolidDomain(mesh, hookean_material, element)

        # Clamp face at Z = 0 (nodes with Z = 0)
        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        clamp = DirichletConstraint(
            node_indices=fixed_nodes, components=(0, 1, 2), value=0.0
        )
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(clamp)

        # Apply upward tensile load F_z = +1000 N on top nodes (Z = 1)
        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        f_applied = 1000.0
        load = NodalForce(
            node_indices=top_nodes, force_vector=[0.0, 0.0, f_applied / len(top_nodes)]
        )
        f_ext = load.compute_force_vector(mesh.n_dofs)

        analysis = LinearStaticAnalysis(domain)
        result = analysis.solve(dirichlet=dirichlet, f_ext=f_ext)

        assert isinstance(result, LinearStaticResult)
        assert result.displacement.shape == (mesh.n_dofs,)
        assert result.strain_energy > 0.0

        # Global Equilibrium: Sum of support reactions (R_z) + external applied forces (F_z) must equal zero
        net_equilibrium_error = np.abs(np.sum(result.reactions[2::3]) + f_applied)
        assert np.isclose(net_equilibrium_error, 0.0, atol=1e-8)

        # Top nodes must displace in positive Z direction
        top_dofs_z = top_nodes * 3 + 2
        assert np.all(result.displacement[top_dofs_z] > 0.0)

    def test_linear_static_beam_cantilever_benchmark(self, beam_vtk_path: Path) -> None:
        if not beam_vtk_path.exists():
            pytest.skip("beam1.vtk not found")

        mesh = VolumetricMesh(beam_vtk_path)
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(
            youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1.0
        )
        material = Hookean(params)
        element = Tet4Linear(material)
        domain = SolidDomain(mesh, material, element)

        root_nodes = select_nodes_by_plane(mesh, plane="xz", bound="min")
        clamp = DirichletConstraint(
            node_indices=root_nodes, components=(0, 1, 2), value=0.0
        )
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(clamp)

        tip_nodes = select_nodes_by_plane(mesh, plane="xz", bound="max")
        total_fz = -1000.0
        load = NodalForce(
            node_indices=tip_nodes, force_vector=[0.0, 0.0, total_fz / len(tip_nodes)]
        )
        f_ext = load.compute_force_vector(mesh.n_dofs)

        analysis = LinearStaticAnalysis(domain)
        result = analysis.solve(dirichlet=dirichlet, f_ext=f_ext)

        # Global equilibrium: reaction force R_z must balance total applied force F_z = -1000 N
        eq_error = np.abs(np.sum(result.reactions[2::3]) + total_fz)
        assert eq_error < 1e-6

        # Tip deflection in -Z
        tip_z = result.displacement[tip_nodes * 3 + 2]
        assert np.mean(tip_z) < 0.0

    def test_tet10_cantilever_shear_locking_alleviation(
        self, beam_vtk_path: Path
    ) -> None:
        """
        Verifies that higher-order 10-node quadratic tetrahedra (Tet10) alleviate
        artificial shear locking compared to overly stiff linear 4-node tetrahedra (Tet4)
        on the exact same underlying mesh topology.
        """
        if not beam_vtk_path.exists():
            pytest.skip("beam1.vtk not found")

        mesh_tet4 = VolumetricMesh(beam_vtk_path)
        mesh_tet10 = mesh_tet4.to_quadratic()

        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(
            youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1.0
        )
        material = Hookean(params)

        # 1. Tet4 solve
        dom4 = SolidDomain(mesh_tet4, material, Tet4Linear(material))
        root4 = select_nodes_by_plane(mesh_tet4, plane="xz", bound="min")
        dir4 = DirichletManager(mesh_tet4.n_dofs)
        dir4.add_constraint(DirichletConstraint(root4, components=(0, 1, 2), value=0.0))

        tip4 = select_nodes_by_plane(mesh_tet4, plane="xz", bound="max")
        total_fz = -1000.0
        load4 = NodalForce(tip4, force_vector=[0.0, 0.0, total_fz / len(tip4)])
        f4 = load4.compute_force_vector(mesh_tet4.n_dofs)

        res4 = LinearStaticAnalysis(dom4).solve(dirichlet=dir4, f_ext=f4, verbose=False)
        tip_def4 = np.mean(res4.displacement[tip4 * 3 + 2])

        # 2. Tet10 solve
        dom10 = SolidDomain(mesh_tet10, material, Tet10Linear(material))
        root10 = select_nodes_by_plane(mesh_tet10, plane="xz", bound="min")
        dir10 = DirichletManager(mesh_tet10.n_dofs)
        dir10.add_constraint(
            DirichletConstraint(root10, components=(0, 1, 2), value=0.0)
        )

        tip10 = select_nodes_by_plane(mesh_tet10, plane="xz", bound="max")
        load10 = NodalForce(tip10, force_vector=[0.0, 0.0, total_fz / len(tip10)])
        f10 = load10.compute_force_vector(mesh_tet10.n_dofs)

        res10 = LinearStaticAnalysis(dom10).solve(
            dirichlet=dir10, f_ext=f10, verbose=False
        )
        tip_def10 = np.mean(res10.displacement[tip10 * 3 + 2])

        # Equilibrium checks for both
        assert np.isclose(np.sum(res4.reactions[2::3]), -total_fz, rtol=1e-6)
        assert np.isclose(np.sum(res10.reactions[2::3]), -total_fz, rtol=1e-6)

        # Tet10 must be strictly more flexible than Tet4 in bending (alleviating shear locking)
        assert abs(tip_def10) > 1.5 * abs(tip_def4)

    def test_linear_dynamic_vibration_and_decay(
        self,
        five_tet_cube_mesh: VolumetricMesh,
        hookean_material: Hookean,
        tmp_path: Path,
    ) -> None:
        mesh = five_tet_cube_mesh
        element = Tet4Linear(hookean_material)
        domain = SolidDomain(mesh, hookean_material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        # Initial perturbation load
        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        load = NodalForce(top_nodes, [100.0, 0.0, 0.0])
        f_ext = load.compute_force_vector(mesh.n_dofs)

        # Strong Rayleigh damping to observe decay
        damping = RayleighDamping(alpha=10.0, beta=0.01)
        analysis = LinearDynamicAnalysis(domain, damping=damping)

        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.001,
            n_steps=20,
            save_directory=tmp_path / "dynamic_out",
            verbose=False,
        )

        assert isinstance(result, LinearDynamicResult)
        assert len(result.time) == 21
        assert (tmp_path / "dynamic_out" / "result.pvd").exists()

    def test_linear_dynamic_steady_state_early_exit(
        self,
        five_tet_cube_mesh: VolumetricMesh,
        hookean_material: Hookean,
        tmp_path: Path,
    ) -> None:
        mesh = five_tet_cube_mesh
        element = Tet4Linear(hookean_material)
        domain = SolidDomain(mesh, hookean_material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        load = NodalForce(top_nodes, [50.0, 0.0, 0.0])
        f_ext = load.compute_force_vector(mesh.n_dofs)

        # High damping to rapidly reach static state
        damping = RayleighDamping(alpha=30.0, beta=0.05)
        analysis = LinearDynamicAnalysis(domain, damping=damping)

        out_dir = tmp_path / "lin_steady_out"
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.005,
            n_steps=200,
            steady_state_tol=1e-2,
            consecutive_hold_steps=2,
            save_interval=10,
            save_directory=out_dir,
            verbose=False,
        )

        assert isinstance(result, LinearDynamicResult)
        assert result.converged is True
        assert result.steady_state_converged is True
        assert result.steady_state_step is not None
        assert result.steps_completed < 200
        assert result.steps_completed == result.steady_state_step

        last_file = result.time_series_files[-1]
        assert last_file.exists()
        assert f"{result.steady_state_step:05d}" in last_file.name

    def test_linear_static_save_file(
        self,
        five_tet_cube_mesh: VolumetricMesh,
        hookean_material: Hookean,
        tmp_path: Path,
    ) -> None:
        mesh = five_tet_cube_mesh
        element = Tet4Linear(hookean_material)
        domain = SolidDomain(mesh, hookean_material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(
                node_indices=fixed_nodes, components=(0, 1, 2), value=0.0
            )
        )

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        load = NodalForce(
            node_indices=top_nodes, force_vector=[0.0, 0.0, 100.0 / len(top_nodes)]
        )
        f_ext = load.compute_force_vector(mesh.n_dofs)

        out_dir = tmp_path / "lin_static_out"
        analysis = LinearStaticAnalysis(domain)
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            save_directory=out_dir,
            output_name="cube_linear",
            verbose=False,
        )

        assert result.saved_file is not None
        assert result.saved_file.exists()
        assert result.saved_file.name == "cube_linear.vtu"

    def test_linear_dynamic_spectral_radius_cases(
        self,
        five_tet_cube_mesh: VolumetricMesh,
        hookean_material: Hookean,
    ) -> None:
        """Verifies that spectral_radius controls numerical damping correctly."""
        mesh = five_tet_cube_mesh
        element = Tet4Linear(hookean_material)
        domain = SolidDomain(mesh, hookean_material, element)

        # Invalid bounds validation
        import pytest

        with pytest.raises(ValueError, match="spectral_radius must be in"):
            LinearDynamicAnalysis(domain, spectral_radius=-0.1)
        with pytest.raises(ValueError, match="spectral_radius must be in"):
            LinearDynamicAnalysis(domain, spectral_radius=1.5)

        # Test solver with undamped rho=1.0 and fully damped rho=0.0
        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(node_indices=fixed_nodes, components=(0, 1, 2), value=0.0)
        )
        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        load = NodalForce(node_indices=top_nodes, force_vector=[0.0, 0.0, 50.0 / len(top_nodes)])
        f_ext = load.compute_force_vector(mesh.n_dofs)

        solver_undamped = LinearDynamicAnalysis(domain, spectral_radius=1.0)
        assert solver_undamped.spectral_radius == 1.0
        assert solver_undamped.alpha_m == 0.5
        assert solver_undamped.alpha_f == 0.5
        assert solver_undamped.gamma == 0.5
        assert solver_undamped.beta == 0.25

        solver_damped = LinearDynamicAnalysis(domain, spectral_radius=0.0)
        assert solver_damped.spectral_radius == 0.0
        assert solver_damped.alpha_m == -1.0
        assert solver_damped.alpha_f == 0.0
        assert solver_damped.gamma == 1.5
        assert solver_damped.beta == 1.0

        res_undamped = solver_undamped.solve(dirichlet=dirichlet, f_ext=f_ext, dt=0.01, n_steps=10, verbose=False)
        domain.reset_state()
        res_damped = solver_damped.solve(dirichlet=dirichlet, f_ext=f_ext, dt=0.01, n_steps=10, verbose=False)

        assert res_undamped.converged is True
        assert res_damped.converged is True
