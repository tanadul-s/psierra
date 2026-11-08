"""
Unit and benchmark tests for Tet10Linear and Tet10Base quadratic tetrahedral element.

Validates:
1. Shape function Kronecker delta property N_i(xi_j) = delta_ij at all 10 nodes.
2. Shape function partition of unity sum(N_i) = 1 throughout volume.
3. Shape gradient analytical correctness matching finite differences and summing to zero.
4. Quadrature integration accuracy (1, 4, and 5-point tetrahedral rules).
5. Reference volume (V = 1/6) and exact gradient recovery for linear coordinate fields.
6. Inverted element detection (negative Jacobian determinant error).
7. Consistent mass matrix symmetry, positive-definiteness, and total mass conservation.
8. Linear stiffness matrix symmetry, rank, and 6 rigid body modes.
9. Rigid body translation invariance (zero reaction forces).
10. Constant strain patch test at all 4 quadrature points.
11. End-to-end single-element tension equilibrium.
"""

from __future__ import annotations
import numpy as np
import pytest

from psierra.element.tet10_base import Tet10Base
from psierra.element.tet10_linear import Tet10Linear
from psierra.element.quadrature import get_tetrahedral_quadrature
from psierra.material.hookean import Hookean
from psierra.mesh.volumetric import VolumetricMesh
from psierra.domain.solid import SolidDomain
from psierra.boundary.dirichlet import DirichletConstraint, DirichletManager
from psierra.boundary.load import NodalForce
from psierra.analysis.linear_static import LinearStaticAnalysis, LinearStaticResult


class TestTet10Linear:
    """Rigorous physical and mathematical verification of 10-node linear quadratic tetrahedral elements."""

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
            for dim, delta in enumerate(
                [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
            ):
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
        """Volume of unit tet10 is 1/6, gradients sum to zero, linear field is exact."""
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
        """Negative Jacobian determinant in Tet10 must raise ValueError."""
        tet = Tet10Linear(hookean_material)
        with pytest.raises(ValueError, match="Jacobian determinant detected"):
            tet.compute_volume_and_gradients(inverted_tet10_coords)

    def test_mass_matrix_properties(
        self, hookean_material: Hookean, regular_tet10_coords: np.ndarray
    ) -> None:
        """Mass matrix must be symmetric, positive-definite, and conserve total mass."""
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
        """Linear stiffness matrix must have exactly 6 zero eigenvalues and 24 positive eigenvalues."""
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
        """Under arbitrary rigid translation, reaction forces must be zero to machine precision."""
        tet_lin = Tet10Linear(hookean_material)
        K_e = tet_lin.compute_linear_stiffness_matrix(regular_tet10_coords)

        t = np.array([15.2, -42.8, 8.4], dtype=np.float64)
        u_rigid = np.tile(t, 10)

        # Linear reaction force K * u must be zero within machine precision relative to ||K||*||u||
        f_reaction = K_e @ u_rigid
        rel_error = np.linalg.norm(f_reaction) / (
            np.linalg.norm(K_e) * np.linalg.norm(u_rigid)
        )
        assert rel_error < 1e-12

    def test_patch_test_constant_strain(
        self, hookean_material: Hookean, regular_tet10_coords: np.ndarray
    ) -> None:
        """Verifies exact constant strain recovery at all 4 Gauss quadrature points."""
        tet = Tet10Linear(hookean_material)

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

        qp, qw = get_tetrahedral_quadrature(4)
        for q in range(4):
            dxi = Tet10Base.evaluate_shape_gradients_ref(*qp[q])
            J = regular_tet10_coords.T @ dxi
            H = dxi @ np.linalg.inv(J)
            BL = tet.compute_linear_B_matrix(H)
            eps_q = BL @ u_flat
            assert np.allclose(eps_q, eps_voigt_expected, atol=1e-14)

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
        result = analysis.solve(dirichlet=dirichlet, f_ext=f_ext, verbose=False)

        assert isinstance(result, LinearStaticResult)
        assert result.displacement.shape == (domain.mesh.n_dofs,)
        assert result.strain_energy > 0.0

        # Displacements must be positive in Z for the apex
        apex_uz = domain.displacement[3, 2]
        assert apex_uz > 0.0

        # Global Equilibrium: Sum of support reactions (R_z) + external applied forces (F_z) must equal zero
        net_equilibrium_error = np.abs(np.sum(result.reactions[2::3]) + f_applied)
        assert np.isclose(net_equilibrium_error, 0.0, atol=1e-8)
