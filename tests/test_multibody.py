"""
Unit Tests for MultiBodySystem Architecture.

Verifies:
1. Multi-body aggregation of independent physical domains.
2. Global DOF offset indexing and kinematic state propagation.
3. Block-diagonal mass, stiffness, and tangent matrix assembly.
4. Internal force vector concatenation.
5. End-to-end multi-body solving via NonlinearStaticAnalysis.
6. PyVista multi-domain visualization with Body ID labeling and Displacement scalar.
"""

from __future__ import annotations
import numpy as np
import pytest

from psierra.multibody import MultiBodySystem
from psierra.domain import SolidDomain
from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear, Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.boundary import DirichletConstraint, DirichletManager, GravityLoad
from psierra.analysis import NonlinearStaticAnalysis


class TestMultiBodySystemBasics:
    """Tests basic aggregation, slicing, and state management of MultiBodySystem."""

    @pytest.fixture
    def two_cube_system(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> tuple[MultiBodySystem, SolidDomain, SolidDomain]:
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(1000.0, 0.3, 1.0)
        mat = Hookean(params)
        elem = Tet4TotalLagrangian(mat)

        body_a = SolidDomain(five_tet_cube_mesh, mat, elem)
        body_b = SolidDomain(five_tet_cube_mesh, mat, elem)

        system = MultiBodySystem([body_a, body_b], names=["CubeA", "CubeB"])
        return system, body_a, body_b

    def test_dof_indexing_and_slices(
        self, two_cube_system: tuple[MultiBodySystem, SolidDomain, SolidDomain]
    ) -> None:
        system, body_a, body_b = two_cube_system
        assert system.n_bodies == 2
        assert system.n_dofs == body_a.n_dofs + body_b.n_dofs

        # Check slices
        sl_a = system.get_body_dof_slice(0)
        sl_b = system.get_body_dof_slice(1)
        assert sl_a == slice(0, body_a.n_dofs)
        assert sl_b == slice(body_a.n_dofs, body_a.n_dofs + body_b.n_dofs)

    def test_state_propagation(
        self, two_cube_system: tuple[MultiBodySystem, SolidDomain, SolidDomain]
    ) -> None:
        system, body_a, body_b = two_cube_system
        u_rand = np.random.randn(system.n_dofs)
        v_rand = np.random.randn(system.n_dofs)
        a_rand = np.random.randn(system.n_dofs)

        system.update_state(u=u_rand, v=v_rand, a=a_rand)

        # Verify slices propagated down to constituent domains
        assert np.allclose(body_a.u, u_rand[system.get_body_dof_slice(0)])
        assert np.allclose(body_b.u, u_rand[system.get_body_dof_slice(1)])
        assert np.allclose(body_a.v, v_rand[system.get_body_dof_slice(0)])
        assert np.allclose(body_b.v, v_rand[system.get_body_dof_slice(1)])

    def test_block_matrix_assembly(
        self, two_cube_system: tuple[MultiBodySystem, SolidDomain, SolidDomain]
    ) -> None:
        system, body_a, body_b = two_cube_system

        # Mass matrix: M = diag(M_a, M_b)
        M_sys = system.assemble_mass_matrix()
        M_a = body_a.assemble_mass_matrix()
        M_b = body_b.assemble_mass_matrix()

        assert M_sys.shape == (system.n_dofs, system.n_dofs)
        sl_a = system.get_body_dof_slice(0)
        sl_b = system.get_body_dof_slice(1)

        assert np.allclose(M_sys[sl_a, sl_a].toarray(), M_a.toarray())
        assert np.allclose(M_sys[sl_b, sl_b].toarray(), M_b.toarray())
        # Off-diagonal blocks must be zero
        assert np.allclose(M_sys[sl_a, sl_b].toarray(), 0.0)
        assert np.allclose(M_sys[sl_b, sl_a].toarray(), 0.0)

        # Stiffness matrix: K = diag(K_a, K_b)
        K_sys = system.assemble_tangent_stiffness_matrix()
        K_a = body_a.assemble_tangent_stiffness_matrix()
        K_b = body_b.assemble_tangent_stiffness_matrix()
        assert np.allclose(K_sys[sl_a, sl_a].toarray(), K_a.toarray())
        assert np.allclose(K_sys[sl_b, sl_b].toarray(), K_b.toarray())

    def test_internal_forces_assembly(
        self, two_cube_system: tuple[MultiBodySystem, SolidDomain, SolidDomain]
    ) -> None:
        system, body_a, body_b = two_cube_system
        u_rand = np.random.randn(system.n_dofs) * 0.01
        system.update_state(u=u_rand)

        f_int = system.assemble_internal_forces()
        sl_a = system.get_body_dof_slice(0)
        sl_b = system.get_body_dof_slice(1)

        assert np.allclose(f_int[sl_a], body_a.assemble_internal_forces())
        assert np.allclose(f_int[sl_b], body_b.assemble_internal_forces())


class TestMultiBodySolverIntegration:
    """Verifies that standard solvers seamlessly solve a MultiBodySystem."""

    def test_multi_body_static_solve(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(1000.0, 0.3, 1.0)
        mat = Hookean(params)
        elem = Tet4Linear(mat)

        body_1 = SolidDomain(five_tet_cube_mesh, mat, elem)
        body_2 = SolidDomain(five_tet_cube_mesh, mat, elem)
        system = MultiBodySystem([body_1, body_2], names=["Body1", "Body2"])

        # Clamp base face of Body 1 and Body 2
        n_nodes_1 = body_1.n_dofs // 3  # 8 nodes
        base_nodes_1 = np.array([0, 1, 2, 3])
        base_nodes_2 = base_nodes_1 + n_nodes_1

        dirichlet = DirichletManager(system.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(
                node_indices=base_nodes_1, components=(0, 1, 2), value=0.0
            )
        )
        dirichlet.add_constraint(
            DirichletConstraint(
                node_indices=base_nodes_2,
                components=(0, 1, 2),
                value=0.0,
            )
        )

        gravity = GravityLoad(acceleration=(0.0, -9.81, 0.0))
        M = system.assemble_mass_matrix()
        f_ext = gravity.compute_force_vector(M)

        analysis = NonlinearStaticAnalysis(system)
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            n_steps=2,
            nr_iterations=10,
            nr_tol=1e-3,
            verbose=False,
        )

        assert result.converged
        # Both bodies must exhibit downward displacement
        u_body1 = system.get_body_displacement(0)
        u_body2 = system.get_body_displacement(1)
        assert u_body1[:, 1].min() < 0.0
        assert u_body2[:, 1].min() < 0.0

    def test_multi_body_pyvista_export(
        self, five_tet_cube_mesh: VolumetricMesh, tmp_path
    ) -> None:
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(1000.0, 0.3, 1.0)
        mat = Hookean(params)
        elem = Tet4Linear(mat)

        body_1 = SolidDomain(five_tet_cube_mesh, mat, elem)
        body_2 = SolidDomain(five_tet_cube_mesh, mat, elem)
        system = MultiBodySystem([body_1, body_2], names=["Left", "Right"])

        grid = system.to_pyvista()
        assert "Body ID" in grid.cell_data
        assert set(np.unique(grid.cell_data["Body ID"])) == {0, 1}
        assert "Displacement" in grid.point_data

        # Test save state
        vtu_file = tmp_path / "system.vtu"
        system.save_state(vtu_file)
        assert vtu_file.exists()

    def test_multi_body_contact_registration(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        from psierra.boundary.contact import RigidContact
        from psierra.collision import RigidPlane

        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(1000.0, 0.3, 1.0)
        mat = Hookean(params)
        elem = Tet4Linear(mat)

        body_1 = SolidDomain(five_tet_cube_mesh, mat, elem)
        body_2 = SolidDomain(five_tet_cube_mesh, mat, elem)
        system = MultiBodySystem([body_1, body_2], names=["BodyA", "BodyB"])

        floor = RigidPlane(origin=[0.0, -1.0, 0.0], normal=[0.0, 1.0, 0.0])
        contact = RigidContact(
            obstacle=floor, domain=system, body="BodyA", k_penalty=1e4
        )
        system.add_contact(contact)

        assert len(system.contacts) == 1
        assert system.contacts[0] is contact

        # Verify solve pulls contact automatically from system without passing contact to solve()
        dirichlet = DirichletManager(system.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint([0], components=(0, 1, 2), value=0.0)
        )

        solver = NonlinearStaticAnalysis(system)
        result = solver.solve(dirichlet=dirichlet, n_steps=1, verbose=False)
        assert result.converged

    def test_multi_body_dict_init_and_contact_pair(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        from psierra.collision import RigidPlane

        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(1000.0, 0.3, 1.0)
        mat = Hookean(params)
        elem = Tet4Linear(mat)

        body_1 = SolidDomain(five_tet_cube_mesh, mat, elem)
        body_2 = SolidDomain(five_tet_cube_mesh, mat, elem)

        # 1. Dict initialization
        system = MultiBodySystem({"Cube1": body_1, "Cube2": body_2})
        assert system.names == ["Cube1", "Cube2"]
        assert system["Cube1"] is body_1
        assert system["Cube2"] is body_2
        assert "Cube1" in system
        assert system.get_body_dof_slice("Cube1") == slice(0, body_1.n_dofs)
        assert system.get_body_dof_slice("Cube2") == slice(body_1.n_dofs, system.n_dofs)

        # 2. Clean assembly contact pair definition without circular references
        floor = RigidPlane(origin=[0.0, -1.0, 0.0], normal=[0.0, 1.0, 0.0])
        contact = system.add_contact_pair(
            slave="Cube1", master=floor, k_penalty=2e4, c_damping=5.0
        )
        assert len(system.contacts) == 1
        assert system.contacts[0] is contact
        assert contact.k_penalty == 2e4
        assert contact.c_damping == 5.0
        assert contact.dof_offset == 0

