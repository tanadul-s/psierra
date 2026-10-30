from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from psierra.domain.solid import SolidDomain
from psierra.element.tet4 import Tet4
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
        element = Tet4(hookean_material)
        domain = SolidDomain(mesh, hookean_material, element)

        # Clamp face at Z = 0 (nodes with Z = 0)
        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        clamp = DirichletConstraint(node_indices=fixed_nodes, components=(0, 1, 2), value=0.0)
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(clamp)

        # Apply upward tensile load F_z = +1000 N on top nodes (Z = 1)
        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        f_applied = 1000.0
        load = NodalForce(node_indices=top_nodes, force_vector=[0.0, 0.0, f_applied / len(top_nodes)])
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

    def test_linear_static_beam_cantilever_benchmark(
        self, beam_vtk_path: Path
    ) -> None:
        if not beam_vtk_path.exists():
            pytest.skip("beam1.vtk not found")

        mesh = VolumetricMesh(beam_vtk_path)
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1.0)
        material = Hookean(params)
        element = Tet4(material)
        domain = SolidDomain(mesh, material, element)

        root_nodes = select_nodes_by_plane(mesh, plane="xz", bound="min")
        clamp = DirichletConstraint(node_indices=root_nodes, components=(0, 1, 2), value=0.0)
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(clamp)

        tip_nodes = select_nodes_by_plane(mesh, plane="xz", bound="max")
        total_fz = -1000.0
        load = NodalForce(node_indices=tip_nodes, force_vector=[0.0, 0.0, total_fz / len(tip_nodes)])
        f_ext = load.compute_force_vector(mesh.n_dofs)

        analysis = LinearStaticAnalysis(domain)
        result = analysis.solve(dirichlet=dirichlet, f_ext=f_ext)

        # Global equilibrium: reaction force R_z must balance total applied force F_z = -1000 N
        eq_error = np.abs(np.sum(result.reactions[2::3]) + total_fz)
        assert eq_error < 1e-6

        # Tip deflection in -Z
        tip_z = result.displacement[tip_nodes * 3 + 2]
        assert np.mean(tip_z) < 0.0

    def test_linear_dynamic_vibration_and_decay(
        self, five_tet_cube_mesh: VolumetricMesh, hookean_material: Hookean, tmp_path: Path
    ) -> None:
        mesh = five_tet_cube_mesh
        element = Tet4(hookean_material)
        domain = SolidDomain(mesh, hookean_material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0))

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
        assert (tmp_path / "dynamic_out" / "output.pvd").exists()
