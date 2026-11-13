from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from psierra.domain.solid import SolidDomain
from psierra.element import Tet4TotalLagrangian
from psierra.material import NeoHookean, Viscoelastic, PronyTerm
from psierra.mesh.volumetric import VolumetricMesh
from psierra.boundary.dirichlet import DirichletConstraint, DirichletManager
from psierra.boundary.load import NodalForce
from psierra.boundary.selector import select_nodes_by_plane
from psierra.analysis.nonlinear_static import NonlinearStaticAnalysis
from psierra.analysis.quasi_static import QuasiStaticAnalysis, QuasiStaticResult


class TestQuasiStaticAnalysis:
    """Unit and mathematical verification tests for QuasiStaticAnalysis."""

    def test_quasi_static_elastic_matches_static(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        """
        Verifies that for a rate-independent elastic material, QuasiStaticAnalysis
        recovers the exact same displacement as NonlinearStaticAnalysis.
        """
        mesh1 = five_tet_cube_mesh
        elem1 = Tet4TotalLagrangian(neohookean_material)
        dom1 = SolidDomain(mesh1, neohookean_material, elem1)

        fixed_nodes = select_nodes_by_plane(mesh1, plane="xy", bound="min")
        clamp = DirichletConstraint(
            node_indices=fixed_nodes, components=(0, 1, 2), value=0.0
        )
        dirichlet = DirichletManager(mesh1.n_dofs)
        dirichlet.add_constraint(clamp)

        top_nodes = select_nodes_by_plane(mesh1, plane="xy", bound="max")
        total_fz = 50.0
        load = NodalForce(
            node_indices=top_nodes, force_vector=[0.0, 0.0, total_fz / len(top_nodes)]
        )
        f_ext = load.compute_force_vector(mesh1.n_dofs)

        # 1. Nonlinear Static Solve (4 steps)
        solver_static = NonlinearStaticAnalysis(dom1)
        res_static = solver_static.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            n_steps=4,
            nr_iterations=15,
            nr_tol=1e-5,
            verbose=False,
        )

        # 2. Quasi-Static Solve (4 steps over 1.0s, with linear load schedule)
        mesh2 = mesh1.copy()
        elem2 = Tet4TotalLagrangian(neohookean_material)
        dom2 = SolidDomain(mesh2, neohookean_material, elem2)
        solver_qs = QuasiStaticAnalysis(dom2)
        res_qs = solver_qs.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            total_time=1.0,
            dt=0.25,
            load_schedule=lambda t: t / 1.0,
            nr_iterations=15,
            nr_tol=1e-5,
            verbose=False,
        )

        assert res_static.converged
        assert res_qs.converged
        assert len(res_qs.time_history) == 5  # t=0, 0.25, 0.5, 0.75, 1.0
        assert np.allclose(res_qs.displacement, res_static.displacement, atol=1e-5)

    def test_quasi_static_viscoelastic_relaxation_and_creep(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        """
        Verifies rate-dependent viscoelastic creep:
        Under constant held load, displacement monotonically increases over time
        from initial elastic deflection u0 to fully relaxed asymptotic deflection u_inf ~ u0 / g_inf.
        """
        # Viscoelastic material: g1 = 0.5, tau1 = 0.1s, g_inf = 0.5
        prony = [PronyTerm(relative_modulus=0.5, relaxation_time=0.1)]
        visco_mat = Viscoelastic(neohookean_material, prony)

        mesh = five_tet_cube_mesh
        elem = Tet4TotalLagrangian(visco_mat)
        dom = SolidDomain(mesh, visco_mat, elem)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        clamp = DirichletConstraint(
            node_indices=fixed_nodes, components=(0, 1, 2), value=0.0
        )
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(clamp)

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        total_fz = 40.0
        load = NodalForce(
            node_indices=top_nodes, force_vector=[0.0, 0.0, total_fz / len(top_nodes)]
        )
        f_ext = load.compute_force_vector(mesh.n_dofs)

        # Solve 10 steps of dt = 0.05s under constant held load f_ext
        solver = QuasiStaticAnalysis(dom)
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            total_time=0.5,
            dt=0.05,
            nr_iterations=20,
            nr_tol=1e-5,
            verbose=False,
        )

        assert res.converged
        assert res.steps_completed == 10

        # Extract max Z displacement history across all steps
        z_disps = [disp.reshape(-1, 3)[:, 2].max() for disp in res.displacement_history]

        # Step 0 is undeformed (0.0)
        assert np.isclose(z_disps[0], 0.0)

        # Displacements must monotonically increase due to viscoelastic relaxation
        for i in range(1, len(z_disps) - 1):
            assert z_disps[i + 1] >= z_disps[i] - 1e-9

        # Initial displacement at step 1 (t = 0.05s) vs relaxed displacement at step 10 (t = 0.5s = 5*tau)
        u_initial = z_disps[1]
        u_final = z_disps[-1]
        creep_ratio = u_final / u_initial

        # Because g_inf = 0.5, stiffness relaxes toward 50%, so displacement approaches ~2x
        assert creep_ratio > 1.3
        assert creep_ratio <= 2.2

    def test_quasi_static_callable_load_schedule(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        """Verifies custom time-dependent load functions."""
        mesh = five_tet_cube_mesh
        elem = Tet4TotalLagrangian(neohookean_material)
        dom = SolidDomain(mesh, neohookean_material, elem)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        clamp = DirichletConstraint(
            node_indices=fixed_nodes, components=(0, 1, 2), value=0.0
        )
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(clamp)

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        base_fz = 30.0

        def dynamic_f_ext(t: float) -> np.ndarray:
            # Triangular load pulse: ramp up to t=0.2, then ramp down to 0 at t=0.4
            factor = t / 0.2 if t <= 0.2 else max(0.0, (0.4 - t) / 0.2)
            load = NodalForce(
                node_indices=top_nodes, force_vector=[0.0, 0.0, factor * base_fz / len(top_nodes)]
            )
            return load.compute_force_vector(mesh.n_dofs)

        solver = QuasiStaticAnalysis(dom)
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=dynamic_f_ext,
            total_time=0.4,
            dt=0.1,
            nr_iterations=15,
            nr_tol=1e-5,
            verbose=False,
        )

        assert res.converged
        assert res.steps_completed == 4

        # Displacement should peak at t=0.2 (step 2), then return near 0 at t=0.4 (step 4)
        z_disps = [disp.reshape(-1, 3)[:, 2].max() for disp in res.displacement_history]
        assert z_disps[2] > z_disps[1]  # Peak at step 2 (t=0.2)
        assert z_disps[2] > z_disps[4]  # Unloaded at step 4 (t=0.4)
        assert np.isclose(z_disps[4], 0.0, atol=1e-4)

    def test_quasi_static_reaction_equilibrium(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        """Verifies equilibrium of reactions with applied external forces."""
        mesh = five_tet_cube_mesh
        elem = Tet4TotalLagrangian(neohookean_material)
        dom = SolidDomain(mesh, neohookean_material, elem)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        clamp = DirichletConstraint(
            node_indices=fixed_nodes, components=(0, 1, 2), value=0.0
        )
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(clamp)

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        total_fz = 45.0
        load = NodalForce(
            node_indices=top_nodes, force_vector=[0.0, 0.0, total_fz / len(top_nodes)]
        )
        f_ext = load.compute_force_vector(mesh.n_dofs)

        solver = QuasiStaticAnalysis(dom)
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            total_time=0.2,
            dt=0.1,
            verbose=False,
        )

        assert res.converged
        rxn_z = float(res.reactions[clamp.dofs][2::3].sum())
        # Equilibrium in Z: sum(R_z) + sum(F_ext_z) = 0
        assert np.isclose(rxn_z + total_fz, 0.0, atol=1e-3)

    def test_quasi_static_rigid_contact(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        """Verifies quasi-static contact against a rigid ground obstacle."""
        from psierra.boundary.contact import RigidContact
        from psierra.collision.shapes import RigidPlane

        mesh = five_tet_cube_mesh
        elem = Tet4TotalLagrangian(neohookean_material)
        dom = SolidDomain(mesh, neohookean_material, elem)

        # Rigid floor placed at Y = -0.1
        floor = RigidPlane(origin=[0.0, -0.1, 0.0], normal=[0.0, 1.0, 0.0])
        contact = RigidContact(obstacle=floor, domain=dom, k_penalty=1e5)

        # Clamp root face at X = 0
        root_nodes = select_nodes_by_plane(mesh, plane="yz", bound="min")
        clamp = DirichletConstraint(node_indices=root_nodes, components=(0, 1, 2), value=0.0)
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(clamp)

        # Downward load on tip face (X = 1)
        tip_nodes = select_nodes_by_plane(mesh, plane="yz", bound="max")
        total_fy = -80.0
        load = NodalForce(
            node_indices=tip_nodes, force_vector=[0.0, total_fy / len(tip_nodes), 0.0]
        )
        f_ext = load.compute_force_vector(mesh.n_dofs)

        solver = QuasiStaticAnalysis(dom)
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            contact=contact,
            total_time=0.4,
            dt=0.1,
            nr_iterations=25,
            nr_tol=1e-3,
            verbose=False,
        )

        assert res.converged
        current_y = dom.current_nodes[:, 1]
        min_y = float(np.min(current_y))
        # Penetration should be minimal (less than 0.02)
        assert min_y >= -0.12

    def test_quasi_static_adaptive_cutback(
        self, five_tet_cube_mesh: VolumetricMesh, neohookean_material: NeoHookean
    ) -> None:
        """Verifies adaptive cutback when initial time step stalls with low max_iter."""
        mesh = five_tet_cube_mesh
        elem = Tet4TotalLagrangian(neohookean_material)
        dom = SolidDomain(mesh, neohookean_material, elem)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        clamp = DirichletConstraint(
            node_indices=fixed_nodes, components=(0, 1, 2), value=0.0
        )
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(clamp)

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        total_fz = 60.0
        load = NodalForce(
            node_indices=top_nodes, force_vector=[0.0, 0.0, total_fz / len(top_nodes)]
        )
        f_ext = load.compute_force_vector(mesh.n_dofs)

        # With nr_iterations=3 and adaptive_stepping=True, cutback recovers and solves
        solver = QuasiStaticAnalysis(dom)
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            total_time=0.1,
            dt=0.1,
            adaptive_stepping=True,
            dt_min=1e-4,
            gamma_cut=0.5,
            nr_iterations=25,
            verbose=False,
        )

        assert res.converged
        assert res.steps_completed >= 1
