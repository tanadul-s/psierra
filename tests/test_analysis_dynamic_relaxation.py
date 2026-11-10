from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from psierra.domain.solid import SolidDomain
from psierra.element.tet4_linear import Tet4Linear
from psierra.element.tet4_total_lagrangian import Tet4TotalLagrangian
from psierra.material.hookean import Hookean
from psierra.material.neohookean import NeoHookean
from psierra.material.parameters import IsotropicElasticParameters
from psierra.mesh.volumetric import VolumetricMesh
from psierra.boundary.dirichlet import DirichletConstraint, DirichletManager
from psierra.boundary.load import NodalForce
from psierra.boundary.selector import select_nodes_by_plane
from psierra.analysis.linear_static import LinearStaticAnalysis
from psierra.analysis.nonlinear_static import NonlinearStaticAnalysis
from psierra.analysis.dynamic_relaxation import (
    DynamicRelaxationAnalysis,
    DynamicRelaxationResult,
)


class TestDynamicRelaxationAnalysis:
    """Verifies Dynamic Relaxation static equilibrium analysis for linear and nonlinear domains."""

    def test_linear_beam_kinetic_damping_agreement_with_linear_static(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """
        Verifies that Dynamic Relaxation with kinetic damping achieves the exact static equilibrium
        of a linear elastic domain, matching LinearStaticAnalysis.
        """
        mesh = five_tet_cube_mesh
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(youngs_modulus=1000.0, poissons_ratio=0.3, mass_density=1.0)
        material = Hookean(params)
        element = Tet4Linear(material)
        domain = SolidDomain(mesh, material, element)

        # Fix bottom plane
        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        # Apply downward vertical force on top plane
        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        load = NodalForce(top_nodes, [0.0, 0.0, -10.0 / len(top_nodes)])
        f_ext = load.compute_force_vector(mesh.n_dofs)

        # 1. Direct Linear Static Solve (Analytical benchmark)
        lin_solver = LinearStaticAnalysis(domain)
        lin_res = lin_solver.solve(dirichlet=dirichlet, f_ext=f_ext, verbose=False)

        # 2. Dynamic Relaxation Solve (Kinetic Damping)
        dr_domain = SolidDomain(mesh.copy(), material, element)
        dr_solver = DynamicRelaxationAnalysis(
            dr_domain, damping_type="kinetic", safety_factor=1.1, dt=1.0
        )
        dr_res = dr_solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            tol=1e-5,
            max_steps=500,
            verbose=False,
        )

        assert dr_res.converged is True
        assert dr_res.steps_completed < 500
        assert dr_res.kinetic_peaks > 0
        assert dr_res.relative_residual <= 1e-5

        # Check displacement agreement with direct solver
        rel_diff = float(
            np.linalg.norm(dr_res.displacement - lin_res.displacement)
            / np.linalg.norm(lin_res.displacement)
        )
        assert rel_diff < 1e-3, f"DR and LinearStatic differ by {rel_diff:.2e}"

        # Check vertical reaction force equilibrium: sum(R_z) + sum(F_z) == 0
        free_dofs = dirichlet.free_dofs
        p_dofs = dirichlet.prescribed_dofs
        np.testing.assert_allclose(dr_res.reactions[free_dofs], 0.0, atol=1e-4)
        total_reaction_z = dr_res.reactions[p_dofs][2::3].sum()
        np.testing.assert_allclose(total_reaction_z, 10.0, rtol=1e-3)

    def test_linear_beam_viscous_damping_convergence(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """Verifies that Dynamic Relaxation with viscous damping converges to static equilibrium."""
        mesh = five_tet_cube_mesh
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(youngs_modulus=1000.0, poissons_ratio=0.3, mass_density=1.0)
        material = Hookean(params)
        element = Tet4Linear(material)
        domain = SolidDomain(mesh, material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        load = NodalForce(top_nodes, [0.0, 0.0, -10.0 / len(top_nodes)])
        f_ext = load.compute_force_vector(mesh.n_dofs)

        # Direct linear static benchmark
        lin_res = LinearStaticAnalysis(domain).solve(
            dirichlet=dirichlet, f_ext=f_ext, verbose=False
        )

        dr_domain = SolidDomain(mesh.copy(), material, element)
        dr_solver = DynamicRelaxationAnalysis(
            dr_domain, damping_type="viscous", viscous_damping=0.4, dt=1.0
        )
        dr_res = dr_solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            tol=1e-4,
            max_steps=500,
            verbose=False,
        )

        assert dr_res.converged is True
        rel_diff = float(
            np.linalg.norm(dr_res.displacement - lin_res.displacement)
            / np.linalg.norm(lin_res.displacement)
        )
        assert rel_diff < 1e-2

    def test_nonlinear_hyperelastic_agreement_with_newton_raphson(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        """
        Verifies that Dynamic Relaxation solves large-deformation hyperelastic static equilibrium,
        matching the result of incremental Newton-Raphson NonlinearStaticAnalysis.
        """
        mesh = five_tet_cube_mesh
        element = Tet4TotalLagrangian(neohookean_material)
        domain_nr = SolidDomain(mesh, neohookean_material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        load = NodalForce(top_nodes, [0.0, 0.0, 15.0 / len(top_nodes)])
        f_ext = load.compute_force_vector(mesh.n_dofs)

        # 1. Newton-Raphson Nonlinear Static Solve
        nr_solver = NonlinearStaticAnalysis(domain_nr)
        nr_res = nr_solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            n_steps=5,
            nr_iterations=15,
            nr_tol=1e-5,
            verbose=False,
        )
        assert nr_res.converged is True

        # 2. Dynamic Relaxation Solve on hyperelastic domain
        dr_domain = SolidDomain(mesh.copy(), neohookean_material, element)
        dr_solver = DynamicRelaxationAnalysis(
            dr_domain, damping_type="kinetic", safety_factor=1.2, dt=1.0
        )
        dr_res = dr_solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            tol=1e-4,
            max_steps=1000,
            verbose=False,
        )

        assert dr_res.converged is True
        rel_diff = float(
            np.linalg.norm(dr_res.displacement - nr_res.displacement)
            / np.linalg.norm(nr_res.displacement)
        )
        assert rel_diff < 5e-3, f"DR and NR differ by {rel_diff:.2e}"

    def test_dynamic_relaxation_time_series_export(
        self, five_tet_cube_mesh: VolumetricMesh, tmp_path: Path
    ) -> None:
        """Verifies that Dynamic Relaxation can stream relaxation steps to XDMF archive."""
        mesh = five_tet_cube_mesh
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(youngs_modulus=1000.0, poissons_ratio=0.3, mass_density=1.0)
        material = Hookean(params)
        element = Tet4Linear(material)
        domain = SolidDomain(mesh, material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        f_ext = NodalForce(top_nodes, [0.0, 0.0, -10.0 / len(top_nodes)]).compute_force_vector(mesh.n_dofs)

        out_dir = tmp_path / "dr_export"
        solver = DynamicRelaxationAnalysis(domain, damping_type="kinetic")
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            tol=1e-5,
            max_steps=100,
            save_interval=10,
            save_directory=out_dir,
            verbose=False,
        )

        assert res.converged is True
        assert (out_dir / "result.xdmf").exists()
        assert (out_dir / "result.h5").exists()
        assert len(res.time_series_files) == 2

    def test_export_default_static_equilibrium(
        self, five_tet_cube_mesh: VolumetricMesh, tmp_path: Path
    ) -> None:
        """Verifies default static solve exports converged state directly to result.xdmf (1 step)."""
        import h5py

        mesh = five_tet_cube_mesh
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(1000.0, 0.3, 1.0)
        material = Hookean(params)
        element = Tet4Linear(material)
        domain = SolidDomain(mesh, material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        f_ext = NodalForce(top_nodes, [0.0, 0.0, -10.0 / len(top_nodes)]).compute_force_vector(mesh.n_dofs)

        out_dir = tmp_path / "dr_static_export"
        solver = DynamicRelaxationAnalysis(domain, damping_type="kinetic")
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            tol=1e-5,
            max_steps=100,
            save_directory=out_dir,
            verbose=False,
        )

        assert res.converged is True
        assert (out_dir / "result.xdmf").exists()
        assert (out_dir / "result.h5").exists()
        assert res.saved_file == out_dir / "result.xdmf"

        with h5py.File(out_dir / "result.h5", "r") as f:
            steps = list(f["time_series"].keys())
            assert len(steps) == 1
            assert steps[0] == "step_000000"
            d = f["time_series/step_000000/displacement"][:]
            max_d = float(np.linalg.norm(d, axis=1).max())
            assert max_d > 0.0
            np.testing.assert_allclose(d.ravel(), res.displacement, rtol=1e-6)

    def test_invalid_parameters_raise_errors(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """Verifies parameter validation checks."""
        mesh = five_tet_cube_mesh
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(1000.0, 0.3, 1.0)
        material = Hookean(params)
        element = Tet4Linear(material)
        domain = SolidDomain(mesh, material, element)

        with pytest.raises(ValueError, match="damping_type"):
            DynamicRelaxationAnalysis(domain, damping_type="unsupported")  # type: ignore[arg-type]

        solver = DynamicRelaxationAnalysis(domain)
        all_dofs = DirichletManager(mesh.n_dofs)
        all_dofs.add_constraint(DirichletConstraint(list(range(mesh.n_nodes)), (0, 1, 2), 0.0))

        with pytest.raises(ValueError, match="constrained"):
            solver.solve(dirichlet=all_dofs)

        with pytest.raises(ValueError, match="max_steps"):
            solver.solve(dirichlet=DirichletManager(mesh.n_dofs), max_steps=0)

    def test_dynamic_relaxation_with_rigid_contact(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """Verifies Dynamic Relaxation converges to equilibrium in the presence of contact."""
        from psierra.boundary.contact import RigidContact
        from psierra.collision.shapes import RigidPlane

        mesh = five_tet_cube_mesh
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(1000.0, 0.3, 1.0)
        material = Hookean(params)
        element = Tet4Linear(material)
        domain = SolidDomain(mesh, material, element)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        )

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        f_ext = NodalForce(top_nodes, [0.0, 0.0, -50.0 / len(top_nodes)]).compute_force_vector(mesh.n_dofs)

        floor = RigidPlane(origin=np.array([0.5, 0.5, 0.95]), normal=np.array([0.0, 0.0, 1.0]))
        contact = RigidContact(obstacle=floor, slave_nodes=list(top_nodes), k_penalty=5000.0)

        solver = DynamicRelaxationAnalysis(domain, damping_type="kinetic")
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            contact=contact,
            tol=1e-4,
            max_steps=500,
            verbose=False,
        )

        assert res.converged is True
        top_z = domain.current_nodes[top_nodes, 2]
        assert np.all(top_z >= 0.93)

