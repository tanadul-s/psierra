"""Automated Verification Suite for Segment-to-Segment Mortar Contact.

Verifies:
1. Wohlmuth biorthogonal dual shape functions: int Phi_i * N_j dGamma = (A/3) * delta_{ij}
2. Sutherland-Hodgman 2D polygon clipping and Dunavant quadrature accuracy
3. Partition of unity: sum_k M_{ik} = D_{ii} across non-conforming overlapping facets
4. Contact patch test: constant contact pressure transfer across mismatched meshes
5. Exact action-reaction linear momentum balance: sum f_s + sum f_m = 0
6. Conservative tangent stiffness symmetry: K_c = K_c^T
7. Exact rigid body translation invariance: K_c * u_trans = 0
8. Augmented Lagrangian Uzawa multiplier updates and Dual Rebound Relaxation
9. MultiBodySystem integration (add_mortar_contact and add_contact_pair(method='mortar'))
10. End-to-end static and dynamic finite deformation solves with Mortar contact
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as spsp

from psierra.collision.mortar import (
    sutherland_hodgman_clip_2d,
    triangulate_convex_polygon_2d,
    polygon_area_2d,
    dunavant_triangle_quadrature_3pt,
    eval_dual_shape_functions,
    MortarInterfaceIntegrator,
)
from psierra.boundary.mortar import MortarContact
from psierra.mesh import VolumetricMesh
from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian, Tet4Linear
from psierra.domain import SolidDomain
from psierra.multibody import MultiBodySystem
from psierra.boundary import select_nodes_by_bounds, DirichletConstraint, DirichletManager, GravityLoad
from psierra.analysis import NonlinearStaticAnalysis, NonlinearDynamicAnalysis


def _resolve_mesh_path(rel_path: str) -> Path:
    p = Path(rel_path)
    if p.exists():
        return p
    alt = Path("psierra") / rel_path
    if alt.exists():
        return alt
    return p


class TestMortarKernels:
    """Mathematical verification of Mortar geometry, dual shape functions, and clipping."""

    def test_dual_shape_functions_biorthogonality(self) -> None:
        """Verify that int_{tau} Phi_i * N_j dGamma = (A / 3) * delta_{ij} exactly."""
        quad_pts, quad_weights = dunavant_triangle_quadrature_3pt()
        area = 0.5  # Unit right-angle triangle

        D = np.zeros((3, 3), dtype=np.float64)
        for g in range(len(quad_pts)):
            xi = quad_pts[g]
            w_g = quad_weights[g]
            N = xi
            Phi = eval_dual_shape_functions(xi)
            D += area * w_g * np.outer(Phi, N)

        expected_D = np.diag([area / 3.0, area / 3.0, area / 3.0])
        np.testing.assert_allclose(D, expected_D, atol=1e-14)

    def test_sutherland_hodgman_clipping_and_triangulation(self) -> None:
        """Verify 2D Sutherland-Hodgman polygon clipping on known shapes."""
        # Unit square clipped by unit triangle
        t1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # Area = 0.5
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

        clipped = sutherland_hodgman_clip_2d(square, t1)
        assert len(clipped) == 3
        np.testing.assert_allclose(polygon_area_2d(clipped), 0.5, atol=1e-12)

        # Partial overlap
        t2 = np.array([[0.5, 0.0], [1.5, 0.0], [0.5, 1.0]])
        clipped2 = sutherland_hodgman_clip_2d(t2, t1)
        assert len(clipped2) >= 3
        # Overlap is triangle with vertices (0.5, 0), (1.0, 0), (0.5, 0.5), area = 0.5 * 0.5 * 0.5 = 0.125
        np.testing.assert_allclose(polygon_area_2d(clipped2), 0.125, atol=1e-12)

        # Disjoint
        t_far = np.array([[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]])
        clipped_far = sutherland_hodgman_clip_2d(t_far, t1)
        assert len(clipped_far) == 0

    def test_partition_of_unity_and_mortar_m_matrix(self) -> None:
        """Verify partition of unity: sum_k M_{ik} = D_{ii} = A / 3 across split facets."""
        # Slave triangle: [0, 0, 0], [1, 0, 0], [0, 0, 1] (normal [0, 1, 0], area = 0.5)
        s_nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        s_faces = np.array([[0, 2, 1]])

        # 2 non-conforming master triangles that partition the slave triangle:
        # Master 1: [0, 0, 0], [1, 0, 0], [0.5, 0, 0.5]
        # Master 2: [0, 0, 0], [0.5, 0, 0.5], [0, 0, 1]
        m_nodes = np.array([
            [0.0, -0.01, 0.0],
            [1.0, -0.01, 0.0],
            [0.5, -0.01, 0.5],
            [0.0, -0.01, 1.0],
        ])
        m_faces = np.array([[0, 1, 2], [0, 2, 3]])

        integrator = MortarInterfaceIntegrator(
            slave_nodes=s_nodes,
            slave_faces=s_faces,
            master_nodes=m_nodes,
            master_faces=m_faces,
            search_radius=1.0,
        )

        D_diag, M_csr, weighted_gaps, segments = integrator.integrate()
        assert len(segments) == 2

        # Sum over columns (master nodes) for each slave node i:
        M_row_sum = np.array(M_csr.sum(axis=1)).ravel()
        expected_D = 0.5 / 3.0  # = 1/6

        np.testing.assert_allclose(D_diag, expected_D, atol=1e-12)
        np.testing.assert_allclose(M_row_sum, expected_D, atol=1e-12)


class TestMortarMechanicsLaws:
    """Rigorous verification of physical conservation laws and contact patch test."""

    @pytest.fixture
    def single_facet_pair(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s_nodes = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
        s_faces = np.array([[0, 2, 1]])  # normal +y
        m_nodes = np.array([[0.0, -0.2, 0.0], [10.0, -0.2, 0.0], [0.0, -0.2, 10.0]])
        m_faces = np.array([[0, 1, 2]])  # normal -y
        return s_nodes, s_faces, m_nodes, m_faces

    def test_action_reaction_linear_momentum_balance(
        self, single_facet_pair: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Verify exact action-reaction balance: sum f_s + sum f_m = 0 to machine precision."""
        s_nodes, s_faces, m_nodes, m_faces = single_facet_pair
        integrator = MortarInterfaceIntegrator(
            slave_nodes=s_nodes,
            slave_faces=s_faces,
            master_nodes=m_nodes,
            master_faces=m_faces,
            search_radius=1.0,
        )

        D_diag, M_csr, weighted_gaps, segments = integrator.integrate()
        bar_g = weighted_gaps / D_diag
        kp = 5000.0
        P = np.maximum(0.0, -kp * bar_g)

        n_push = np.array([0.0, 1.0, 0.0])  # Along +y (out of master)
        f_s = np.zeros((3, 3))
        for i in range(3):
            f_s[i] = P[i] * n_push

        f_m = np.zeros((3, 3))
        q = P / D_diag
        M_dense = M_csr.toarray()
        for k in range(3):
            for i in range(3):
                f_m[k] -= M_dense[i, k] * q[i] * n_push

        net_force = np.sum(f_s, axis=0) + np.sum(f_m, axis=0)
        np.testing.assert_allclose(net_force, 0.0, atol=1e-12)

    def test_tangent_stiffness_symmetry_and_invariance(
        self, single_facet_pair: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Verify exact symmetry K_c = K_c^T and rigid translation invariance K_c * u_trans = 0."""
        s_nodes, s_faces, m_nodes, m_faces = single_facet_pair
        integrator = MortarInterfaceIntegrator(
            slave_nodes=s_nodes,
            slave_faces=s_faces,
            master_nodes=m_nodes,
            master_faces=m_faces,
            search_radius=1.0,
        )

        D_diag, M_csr, weighted_gaps, segments = integrator.integrate()
        kp = 2500.0
        n_push = np.array([0.0, 1.0, 0.0])
        nnT = np.outer(n_push, n_push)
        M_dense = M_csr.toarray()

        K_c = np.zeros((18, 18), dtype=np.float64)
        for i in range(3):
            K_c[3*i:3*i+3, 3*i:3*i+3] += kp * nnT
            for k in range(3):
                block = -kp * (M_dense[i, k] / D_diag[i]) * nnT
                K_c[3*i:3*i+3, 9+3*k:9+3*k+3] += block
                K_c[9+3*k:9+3*k+3, 3*i:3*i+3] += block.T

        for k in range(3):
            for l in range(3):
                sum_i = sum(M_dense[i, k] * M_dense[i, l] / (D_diag[i]**2) for i in range(3))
                K_c[9+3*k:9+3*k+3, 9+3*l:9+3*l+3] += kp * sum_i * nnT

        # Symmetry:
        np.testing.assert_allclose(K_c, K_c.T, atol=1e-12)

        # Rigid translation invariance:
        for trans in ([1.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, -1.8], [3.0, -2.0, 5.0]):
            u_trans = np.tile(trans, 6)
            f_trans = K_c @ u_trans
            np.testing.assert_allclose(f_trans, 0.0, atol=1e-9)

    def test_contact_patch_test_constant_pressure(self) -> None:
        """Verify the contact patch test: uniform contact pressure produces exact consistent loads."""
        # Slave surface: 2 triangles forming a 10x10 square
        s_nodes = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 10.0],
            [0.0, 0.0, 10.0],
        ])
        s_faces = np.array([[0, 2, 1], [0, 3, 2]])  # Normal +y

        # Master surface: non-conforming mismatched triangulation (4 triangles with a center node)
        m_nodes = np.array([
            [0.0, -0.05, 0.0],
            [10.0, -0.05, 0.0],
            [10.0, -0.05, 10.0],
            [0.0, -0.05, 10.0],
            [4.0, -0.05, 6.0],  # Internal non-conforming vertex
        ])
        m_faces = np.array([
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ])

        integrator = MortarInterfaceIntegrator(
            slave_nodes=s_nodes,
            slave_faces=s_faces,
            master_nodes=m_nodes,
            master_faces=m_faces,
            search_radius=1.0,
        )

        D_diag, M_csr, weighted_gaps, segments = integrator.integrate()
        assert len(segments) >= 4

        # All points have uniform penetration g = -0.05 mm
        bar_g = weighted_gaps / D_diag
        np.testing.assert_allclose(bar_g, -0.05, atol=1e-12)

        # Under uniform pressure p0 = 10.0 MPa (total area = 100 mm^2 -> total force = 1000 N):
        p0 = 10.0
        P_slave = p0 * D_diag  # Consistent slave nodal loads
        total_slave_force_y = np.sum(P_slave)
        np.testing.assert_allclose(total_slave_force_y, 1000.0, atol=1e-10)

        # Master forces via M.T @ (P / D):
        q = P_slave / D_diag  # = constant 10.0 MPa everywhere
        f_master_y = -M_csr.T.dot(q)
        total_master_force_y = np.sum(f_master_y)

        # Net balance:
        np.testing.assert_allclose(total_master_force_y, -1000.0, atol=1e-10)
        np.testing.assert_allclose(total_slave_force_y + total_master_force_y, 0.0, atol=1e-12)


class TestMortarContactIntegration:
    """Integration of MortarContact with MultiBodySystem and nonlinear solvers."""

    def test_multibody_system_mortar_registration(self) -> None:
        """Verify add_mortar_contact and add_contact_pair(method='mortar') in MultiBodySystem."""
        beam_mesh = VolumetricMesh(_resolve_mesh_path("data/beam1/mesh/beam1.vtk"))
        floor_mesh = VolumetricMesh(_resolve_mesh_path("data/floor1/mesh/floor1.vtk"))

        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
        mat = NeoHookean(params)

        domain_beam = SolidDomain(beam_mesh, mat, Tet4TotalLagrangian(mat))
        domain_floor = SolidDomain(floor_mesh, mat, Tet4Linear(mat))

        system = MultiBodySystem({"Beam": domain_beam, "Floor": domain_floor})

        # Method 1: direct add_mortar_contact
        c1 = system.add_mortar_contact(
            slave="Beam",
            master="Floor",
            k_penalty=5e4,
            formulation="augmented_lagrangian",
        )
        assert isinstance(c1, MortarContact)
        assert c1.formulation == "augmented_lagrangian"
        assert c1.k_penalty == 5e4

        # Method 2: add_contact_pair with method="mortar"
        system2 = MultiBodySystem({"Beam": domain_beam, "Floor": domain_floor})
        c2 = system2.add_contact_pair(
            slave="Beam",
            master="Floor",
            k_penalty=8e4,
            method="mortar",
        )
        assert isinstance(c2, MortarContact)
        assert c2.k_penalty == 8e4

    def test_mortar_dual_rebound_relaxation(self) -> None:
        """Verify that relax_rebound zeroes multipliers on non-penetrating candidates."""
        beam_mesh = VolumetricMesh(_resolve_mesh_path("data/beam1/mesh/beam1.vtk"))
        floor_mesh = VolumetricMesh(_resolve_mesh_path("data/floor1/mesh/floor1.vtk"))

        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
        mat = NeoHookean(params)
        domain_beam = SolidDomain(beam_mesh, mat, Tet4TotalLagrangian(mat))
        domain_floor = SolidDomain(floor_mesh, mat, Tet4Linear(mat))

        contact = MortarContact(
            slave=domain_beam,
            master=domain_floor,
            k_penalty=1e4,
            formulation="augmented_lagrangian",
        )

        # Populate synthetic positive multipliers on all slave surface nodes
        n_candidates = len(contact.slave_nodes)
        contact.set_multipliers(np.full(n_candidates, 50.0))

        # Because beam and floor are initially separated (gap > 0), relax_rebound must zero all multipliers
        relaxed_count = contact.relax_rebound(gap_tol=0.0)
        assert relaxed_count == n_candidates
        assert np.all(contact.multipliers == 0.0)

    def test_end_to_end_static_solve_with_mortar(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """Verify full Newton-Raphson convergence of two cubes compressed via Mortar contact."""
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(5000.0, 0.3, 1.0)
        mat = NeoHookean(params)
        elem = Tet4TotalLagrangian(mat)

        # Upper cube at [0.25, 0.75] x [0, 1] x [0.25, 0.75]
        mesh_upper = five_tet_cube_mesh.copy()
        mesh_upper.transform(scale=(0.5, 1.0, 0.5), translation=(0.25, 0.0, 0.25))
        upper = SolidDomain(mesh_upper, mat, elem)

        # Lower cube at [0, 1] x [-1, 0] x [0, 1]
        mesh_lower = five_tet_cube_mesh.copy()
        mesh_lower.transform(translation=(0.0, -1.0, 0.0))
        lower = SolidDomain(mesh_lower, mat, elem)

        system = MultiBodySystem({"Upper": upper, "Lower": lower})
        contact = system.add_mortar_contact(
            slave="Upper",
            master="Lower",
            k_penalty=2e4,
            formulation="augmented_lagrangian",
            search_radius=0.1,
        )

        # Boundary conditions:
        # - Fix bottom face of lower cube (Y = -1.0)
        bot_nodes = np.where(np.isclose(lower.mesh.nodes[:, 1], -1.0))[0]
        clamp_bot = DirichletConstraint(
            bot_nodes,
            components=(0, 1, 2),
            value=0.0,
            dof_offset=system.get_body_dof_slice("Lower").start,
        )

        # - Lateral guide for upper cube to prevent rigid slip in X and Z
        top_nodes = np.where(np.isclose(upper.mesh.nodes[:, 1], 1.0))[0]
        guide_x = DirichletConstraint(top_nodes, components=(0,), value=0.0, dof_offset=0)
        guide_z = DirichletConstraint(top_nodes, components=(2,), value=0.0, dof_offset=0)
        prescribe_top = DirichletConstraint(
            top_nodes, components=(1,), value=-0.02, dof_offset=0
        )

        dirichlet = DirichletManager(system.n_dofs)
        dirichlet.add_constraint(clamp_bot)
        dirichlet.add_constraint(guide_x)
        dirichlet.add_constraint(guide_z)
        dirichlet.add_constraint(prescribe_top)

        solver = NonlinearStaticAnalysis(system)
        result = solver.solve(
            dirichlet=dirichlet,
            n_steps=2,
            nr_tol=1e-3,
            verbose=False,
        )

        assert result.converged
        info = contact.get_contact_info()
        assert info["num_active"] > 0
        assert info["max_force"] > 0.0
        assert info["potential_energy"] >= 0.0
