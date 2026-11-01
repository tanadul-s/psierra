from __future__ import annotations
import numpy as np
import pytest

from psierra.element.tet10_base import Tet10Base
from psierra.element.tet10_linear import Tet10Linear
from psierra.element.tet10_total_lagrangian import Tet10TotalLagrangian
from psierra.element.quadrature import get_tetrahedral_quadrature
from psierra.material.hookean import Hookean
from psierra.material.neohookean import NeoHookean
from psierra.mesh.volumetric import VolumetricMesh
from psierra.domain.solid import SolidDomain
from psierra.boundary.dirichlet import DirichletConstraint, DirichletManager
from psierra.boundary.load import NodalForce
from psierra.analysis.linear_static import LinearStaticAnalysis, LinearStaticResult


class TestTet10Element:
    """Rigorous physical and mathematical verification of 10-node quadratic tetrahedral elements."""

    def test_shape_functions_kronecker_delta(self) -> None:
        """Evaluates shape functions at reference nodes to verify Kronecker delta N_i(xi_j) = delta_ij."""
        ref_nodes = np.array(
            [
                [0.0, 0.0, 0.0],  # 0
                [1.0, 0.0, 0.0],  # 1
                [0.0, 1.0, 0.0],  # 2
                [0.0, 0.0, 1.0],  # 3
                [0.5, 0.0, 0.0],  # 4 (0-1)
                [0.5, 0.5, 0.0],  # 5 (1-2)
                [0.0, 0.5, 0.0],  # 6 (2-0)
                [0.0, 0.0, 0.5],  # 7 (0-3)
                [0.5, 0.0, 0.5],  # 8 (1-3)
                [0.0, 0.5, 0.5],  # 9 (2-3)
            ],
            dtype=np.float64,
        )

        for i in range(10):
            N_vals = Tet10Base.evaluate_shape_functions(*ref_nodes[i])
            expected = np.zeros(10, dtype=np.float64)
            expected[i] = 1.0
            assert np.allclose(N_vals, expected, atol=1e-15), (
                f"Kronecker delta violated at node {i}: {N_vals}"
            )

    def test_shape_functions_partition_of_unity(self) -> None:
        """Shape functions must sum to 1 everywhere in the tetrahedron."""
        np.random.seed(42)
        for _ in range(50):
            # Sample random point in master tetrahedron using Dirichlet distribution
            r = np.random.dirichlet([1.0, 1.0, 1.0, 1.0])
            xi, eta, zeta = r[1], r[2], r[3]

            N_vals = Tet10Base.evaluate_shape_functions(xi, eta, zeta)
            assert np.isclose(np.sum(N_vals), 1.0, atol=1e-14)

    def test_shape_gradients_finite_difference(self) -> None:
        """Analytical shape gradients must match finite difference approximations and sum to zero."""
        np.random.seed(123)
        eps = 1e-7

        for _ in range(25):
            r = np.random.dirichlet([1.0, 1.0, 1.0, 1.0])
            xi, eta, zeta = r[1], r[2], r[3]

            grad_ana = Tet10Base.evaluate_shape_gradients_ref(xi, eta, zeta)
            # Sum of gradients must be exactly zero (derivative of 1)
            assert np.allclose(np.sum(grad_ana, axis=0), 0.0, atol=1e-13)

            # Central finite differences in xi, eta, zeta
            for dim, delta in enumerate([[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]):
                N_plus = Tet10Base.evaluate_shape_functions(
                    xi + delta[0], eta + delta[1], zeta + delta[2]
                )
                N_minus = Tet10Base.evaluate_shape_functions(
                    xi - delta[0], eta - delta[1], zeta - delta[2]
                )
                grad_fd = (N_plus - N_minus) / (2.0 * eps)
                assert np.allclose(grad_ana[:, dim], grad_fd, atol=1e-6)

    def test_quadrature_integration_accuracy(self) -> None:
        """Tests tetrahedral quadrature rules against exact analytical polynomial integrals."""
        # Exact reference volume: 1/6
        for pts in [1, 4, 5]:
            qp, qw = get_tetrahedral_quadrature(pts)
            assert np.isclose(np.sum(qw), 1.0 / 6.0, atol=1e-15)

        # 4-point rule integrates quadratic polynomials exactly:
        # int_{Omega_ref} (xi^2) dV = 1/60
        qp4, qw4 = get_tetrahedral_quadrature(4)
        integral_xi2 = np.sum(qw4 * (qp4[:, 0] ** 2))
        assert np.isclose(integral_xi2, 1.0 / 60.0, atol=1e-14)

        # int_{Omega_ref} (xi * eta) dV = 1/120
        integral_xy = np.sum(qw4 * (qp4[:, 0] * qp4[:, 1]))
        assert np.isclose(integral_xy, 1.0 / 120.0, atol=1e-14)

        # 5-point rule integrates cubic polynomials exactly:
        # int_{Omega_ref} (xi^3) dV = 1/120
        qp5, qw5 = get_tetrahedral_quadrature(5)
        integral_xi3 = np.sum(qw5 * (qp5[:, 0] ** 3))
        assert np.isclose(integral_xi3, 1.0 / 120.0, atol=1e-14)

    def test_volume_and_gradients(
        self, hookean_material: Hookean, regular_tet10_coords: np.ndarray
    ) -> None:
        tet = Tet10Linear(hookean_material)
        V, H = tet.compute_volume_and_gradients(regular_tet10_coords)

        # Unit tetrahedron volume: 1/6
        assert np.isclose(V, 1.0 / 6.0, atol=1e-14)
        assert H.shape == (10, 3)

        # Partition of unity: sum of spatial gradients must be exactly zero
        assert np.allclose(np.sum(H, axis=0), 0.0, atol=1e-14)

        # Linear field test: f(X) = 3*X - 2*Y + 5*Z + 4
        # Grad(f) must evaluate exactly to [3, -2, 5]
        f_nodes = (
            3.0 * regular_tet10_coords[:, 0]
            - 2.0 * regular_tet10_coords[:, 1]
            + 5.0 * regular_tet10_coords[:, 2]
            + 4.0
        )
        grad_f = f_nodes @ H
        assert np.allclose(grad_f, [3.0, -2.0, 5.0], atol=1e-14)

    def test_inverted_element_raises(
        self, hookean_material: Hookean, inverted_tet10_coords: np.ndarray
    ) -> None:
        tet = Tet10Linear(hookean_material)
        with pytest.raises(ValueError, match="Jacobian determinant detected"):
            tet.compute_volume_and_gradients(inverted_tet10_coords)

    def test_mass_matrix_properties(
        self, hookean_material: Hookean, regular_tet10_coords: np.ndarray
    ) -> None:
        tet = Tet10Linear(hookean_material)
        V, _ = tet.compute_volume_and_gradients(regular_tet10_coords)
        M_e = tet.compute_mass_matrix(V)

        assert M_e.shape == (30, 30)
        # Symmetry
        assert np.allclose(M_e, M_e.T, atol=1e-15)

        # Positive-definiteness
        eigvals = np.linalg.eigvalsh(M_e)
        assert np.all(eigvals > 0.0)

        # Total mass conservation: sum of all coefficients in X-direction DOFs = total mass rho * V
        rho = hookean_material.mass_density
        total_mass_expected = rho * V
        x_dofs = np.arange(0, 30, 3)
        assert np.isclose(
            np.sum(M_e[np.ix_(x_dofs, x_dofs)]), total_mass_expected, rtol=1e-12
        )

    def test_linear_stiffness_matrix_rigid_body_modes(
        self, hookean_material: Hookean, regular_tet10_coords: np.ndarray
    ) -> None:
        tet = Tet10Linear(hookean_material)
        K_e = tet.compute_linear_stiffness_matrix(regular_tet10_coords)

        assert K_e.shape == (30, 30)
        # Symmetry
        assert np.allclose(K_e, K_e.T, atol=1e-12)

        # Exactly 6 zero eigenvalues (3 translations + 3 rotations) and 24 positive eigenvalues
        eigvals = np.linalg.eigvalsh(K_e)
        zero_eigvals = eigvals[:6]
        pos_eigvals = eigvals[6:]

        assert np.allclose(zero_eigvals / np.max(pos_eigvals), 0.0, atol=1e-12)
        assert np.all(pos_eigvals > 1e4)

    def test_pure_translation_invariance(
        self, hookean_material: Hookean, regular_tet10_coords: np.ndarray
    ) -> None:
        tet_lin = Tet10Linear(hookean_material)
        tet_tl = Tet10TotalLagrangian(hookean_material)
        K_e = tet_lin.compute_linear_stiffness_matrix(regular_tet10_coords)

        t = np.array([15.2, -42.8, 8.4], dtype=np.float64)
        u_rigid = np.tile(t, 10)

        # Linear reaction force K * u must be zero within machine precision relative to ||K||*||u||
        f_reaction = K_e @ u_rigid
        rel_error = np.linalg.norm(f_reaction) / (
            np.linalg.norm(K_e) * np.linalg.norm(u_rigid)
        )
        assert rel_error < 1e-12

        # Non-linear internal force must also be zero relative to ||K||*||u||
        u_nodes = np.tile(t, (10, 1))
        _, f_int_lin = tet_lin.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet10_coords, u_nodes
        )
        rel_error_lin = np.linalg.norm(f_int_lin) / (
            np.linalg.norm(K_e) * np.linalg.norm(u_rigid)
        )
        assert rel_error_lin < 1e-12

        _, f_int_tl = tet_tl.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet10_coords, u_nodes
        )
        rel_error_tl = np.linalg.norm(f_int_tl) / (
            np.linalg.norm(K_e) * np.linalg.norm(u_rigid)
        )
        assert rel_error_tl < 1e-12

    def test_pure_rotation_invariance_nonlinear(
        self, neohookean_material: NeoHookean, regular_tet10_coords: np.ndarray
    ) -> None:
        """Under large finite rigid body rotation, Green-Lagrange strain and f_int must be zero."""
        tet = Tet10TotalLagrangian(neohookean_material)

        # Finite rotation matrix (30 deg about axis [1, 1, 1])
        theta = np.deg2rad(30.0)
        axis = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        ux, uy, uz = axis

        K_skew = np.array(
            [[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]], dtype=np.float64
        )
        R = (
            np.identity(3, dtype=np.float64)
            + np.sin(theta) * K_skew
            + (1.0 - np.cos(theta)) * (K_skew @ K_skew)
        )

        # Deformed coords: x = R @ X
        x_nodes = (R @ regular_tet10_coords.T).T
        u_nodes = x_nodes - regular_tet10_coords

        # Total Lagrangian internal force must evaluate to zero
        _, f_int = tet.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet10_coords, u_nodes
        )
        assert np.allclose(f_int, 0.0, atol=1e-7)

    def test_patch_test_constant_strain(
        self, hookean_material: Hookean, regular_tet10_coords: np.ndarray
    ) -> None:
        """
        Standard finite element patch test:
        An imposed linear displacement field u(X) = eps_0 @ X must produce
        an exact constant strain state across all quadrature points (to < 1e-14)
        and matching linear and Total Lagrangian internal forces in the small-strain limit.
        """
        tet = Tet10TotalLagrangian(hookean_material)

        # Constant linear strain tensor
        eps_0 = np.array(
            [
                [0.002, 0.001, 0.0005],
                [0.001, -0.0015, 0.0008],
                [0.0005, 0.0008, 0.001],
            ],
            dtype=np.float64,
        )
        eps_voigt_expected = np.array(
            [
                eps_0[0, 0],
                eps_0[1, 1],
                eps_0[2, 2],
                2.0 * eps_0[0, 1],
                2.0 * eps_0[1, 2],
                2.0 * eps_0[2, 0],
            ],
            dtype=np.float64,
        )

        u_nodes = (eps_0 @ regular_tet10_coords.T).T
        u_flat = u_nodes.ravel()

        # 1. Verify exact constant strain recovery at every Gauss quadrature point
        qp, qw = get_tetrahedral_quadrature(4)
        for q in range(4):
            dxi = tet.evaluate_shape_gradients_ref(*qp[q])
            J = regular_tet10_coords.T @ dxi
            H = dxi @ np.linalg.inv(J)
            BL = tet.compute_linear_B_matrix(H)
            eps_q = BL @ u_flat
            assert np.allclose(eps_q, eps_voigt_expected, atol=1e-14)

        # 2. In small-strain limit, linear K @ u and Total Lagrangian f_int must match to < 1e-5
        u_nodes_small = 1e-3 * u_nodes
        u_flat_small = u_nodes_small.ravel()
        K_e = tet.compute_linear_stiffness_matrix(regular_tet10_coords)
        f_linear_small = K_e @ u_flat_small
        _, f_int_small = tet.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet10_coords, u_nodes_small
        )
        rel_diff = np.linalg.norm(f_linear_small - f_int_small) / np.linalg.norm(
            f_linear_small
        )
        assert rel_diff < 1e-5

    def test_nonlinear_tangent_consistency(
        self, neohookean_material: NeoHookean, regular_tet10_coords: np.ndarray
    ) -> None:
        """
        Verifies that the Total Lagrangian tangent stiffness matrix k_tan is the
        exact directional derivative of internal forces f_int via finite difference:
            d f_int / d u == k_tan
        """
        tet = Tet10TotalLagrangian(neohookean_material)

        np.random.seed(42)
        u_nodes = 0.01 * np.random.randn(10, 3)

        k_tan, _ = tet.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet10_coords, u_nodes
        )

        # Central finite difference perturbation
        h = 1e-7
        k_tan_fd = np.zeros((30, 30), dtype=np.float64)

        for dof in range(30):
            node_idx = dof // 3
            comp_idx = dof % 3

            u_plus = u_nodes.copy()
            u_minus = u_nodes.copy()
            u_plus[node_idx, comp_idx] += h
            u_minus[node_idx, comp_idx] -= h

            _, f_plus = tet.compute_nonlinear_force_and_tangent_stiffness(
                regular_tet10_coords, u_plus
            )
            _, f_minus = tet.compute_nonlinear_force_and_tangent_stiffness(
                regular_tet10_coords, u_minus
            )

            k_tan_fd[:, dof] = (f_plus - f_minus) / (2.0 * h)

        rel_error = np.linalg.norm(k_tan - k_tan_fd) / np.linalg.norm(k_tan)
        assert rel_error < 1e-5, f"Tangent stiffness error too high: {rel_error:.4e}"

    def test_end_to_end_solid_domain_solve(
        self, hookean_material: Hookean, single_element_tet10_mesh: VolumetricMesh
    ) -> None:
        """
        End-to-end integration test:
        Constructs a SolidDomain using Tet10Linear, applies Dirichlet constraints and nodal loads,
        solves for displacements, and verifies static equilibrium.
        """
        element = Tet10Linear(hookean_material)
        domain = SolidDomain(
            mesh=single_element_tet10_mesh,
            material=hookean_material,
            element=element,
        )

        assert domain.mesh.is_quadratic
        assert domain.element.nodes_per_element == 10
        assert domain.element.dofs_per_element == 30

        # Fix base face at Z = 0: nodes 0, 1, 2 and mid-edges 4, 5, 6
        base_nodes = np.array([0, 1, 2, 4, 5, 6], dtype=np.int64)
        clamp = DirichletConstraint(
            node_indices=base_nodes, components=(0, 1, 2), value=0.0
        )
        dirichlet = DirichletManager(domain.mesh.n_dofs)
        dirichlet.add_constraint(clamp)

        # Apply tensile force on apex node 3 in +Z direction
        f_applied = 1000.0
        load = NodalForce(node_indices=np.array([3]), force_vector=[0.0, 0.0, f_applied])
        f_ext = load.compute_force_vector(domain.mesh.n_dofs)

        analysis = LinearStaticAnalysis(domain)
        result = analysis.solve(dirichlet=dirichlet, f_ext=f_ext)

        assert isinstance(result, LinearStaticResult)
        assert result.displacement.shape == (domain.mesh.n_dofs,)
        assert result.strain_energy > 0.0

        # Displacements must be positive in Z for the apex
        apex_uz = domain.displacement[3, 2]
        assert apex_uz > 0.0

        # Global Equilibrium: Sum of support reactions (R_z) + external applied forces (F_z) must equal zero
        net_equilibrium_error = np.abs(np.sum(result.reactions[2::3]) + f_applied)
        assert np.isclose(net_equilibrium_error, 0.0, atol=1e-8)
