from __future__ import annotations
import numpy as np
import pytest
import scipy.sparse as spsp

from psierra.boundary.dirichlet import DirichletConstraint, DirichletManager
from psierra.boundary.load import NodalForce, GravityLoad
from psierra.boundary.selector import select_nodes_by_plane, select_nodes_by_bounds
from psierra.mesh.volumetric import VolumetricMesh


class TestBoundaryModule:
    """Verifies Dirichlet constraints, Neumann forces, and node selectors."""

    def test_dirichlet_constraint_creation(self) -> None:
        nodes = np.array([2, 5], dtype=np.int64)
        c = DirichletConstraint(node_indices=nodes, components=(0, 2), value=-0.05)

        # Expected DOFs: node 2 -> 6, 8; node 5 -> 15, 17
        expected_dofs = np.array([6, 8, 15, 17])
        assert np.array_equal(c.dofs, expected_dofs)
        assert np.allclose(c.values, -0.05)

    def test_dirichlet_manager_lifecycle(self) -> None:
        total_dofs = 30
        manager = DirichletManager(total_dofs)

        assert len(manager.prescribed_dofs) == 0
        assert len(manager.free_dofs) == total_dofs

        # Add constraint 1: node 0 clamped (DOFs 0, 1, 2)
        c1 = DirichletConstraint(node_indices=[0], components=(0, 1, 2), value=0.0)
        manager.add_constraint(c1)

        assert np.array_equal(manager.prescribed_dofs, [0, 1, 2])
        assert len(manager.free_dofs) == total_dofs - 3

        # Add constraint 2: node 4 in Z (DOF 14)
        c2 = DirichletConstraint(node_indices=[4], components=(2,), value=1.5)
        manager.add_constraint(c2)

        assert np.array_equal(manager.prescribed_dofs, [0, 1, 2, 14])

        # Orthogonality & Completeness Check
        intersection = np.intersect1d(manager.prescribed_dofs, manager.free_dofs)
        union = np.union1d(manager.prescribed_dofs, manager.free_dofs)
        assert len(intersection) == 0
        assert np.array_equal(union, np.arange(total_dofs))

        # Test remove_constraint
        manager.remove_constraint(c2)
        assert np.array_equal(manager.prescribed_dofs, [0, 1, 2])

        # Test clear
        manager.clear()
        assert len(manager.prescribed_dofs) == 0
        assert len(manager.free_dofs) == total_dofs

    def test_nodal_force_vector(self) -> None:
        force = NodalForce(node_indices=[1, 3], force_vector=[0.0, 0.0, -100.0])
        total_dofs = 18

        f_vec = force.compute_force_vector(total_dofs)
        assert f_vec.shape == (total_dofs,)

        # Node 1 Z is DOF 5 (1*3 + 2); Node 3 Z is DOF 11 (3*3 + 2)
        assert f_vec[5] == -100.0
        assert f_vec[11] == -100.0
        assert np.sum(np.abs(f_vec)) == 200.0

    def test_gravity_load_vector(self) -> None:
        gravity = GravityLoad(acceleration=(0.0, 0.0, -9.81))
        # Simple diagonal mass matrix
        total_dofs = 6
        M = spsp.diags([2.0] * total_dofs).tocsr()

        f_grav = gravity.compute_force_vector(M)
        # Expected: for node 0 and node 1: F_z = 2.0 * -9.81 = -19.62
        assert np.isclose(f_grav[2], -19.62)
        assert np.isclose(f_grav[5], -19.62)
        assert f_grav[0] == 0.0 and f_grav[1] == 0.0

    def test_node_selectors(self, five_tet_cube_mesh: VolumetricMesh) -> None:
        mesh = five_tet_cube_mesh

        # In [0, 1]^3 cube, plane='xz', bound='min' corresponds to Y = 0 (nodes 0, 1, 4, 5)
        y_min_nodes = select_nodes_by_plane(mesh, plane="xz", bound="min")
        assert len(y_min_nodes) == 4
        assert np.allclose(mesh.nodes[y_min_nodes, 1], 0.0)

        # Bound box selector: select nodes with X in [0.9, 1.1]
        x_max_nodes = select_nodes_by_bounds(mesh, bounds=(0.9, 1.1, -10.0, 10.0, -10.0, 10.0))
        assert len(x_max_nodes) == 4
        assert np.allclose(mesh.nodes[x_max_nodes, 0], 1.0)
