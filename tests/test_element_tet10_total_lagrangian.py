"""
Unit and benchmark tests for Tet10TotalLagrangian quadratic finite element.

Validates:
1. Rigid body translation invariance (internal forces vanish to machine precision).
2. Large finite rigid body rotation invariance (Green-Lagrange strain and internal forces vanish).
3. Constant strain patch test and small-strain limit matching linear stiffness.
4. Directional derivative tangent consistency (k_tan matches d f_int / d u via finite differences).
5. Hyperelastic constitutive law compatibility (NeoHookean, Hookean).
"""

from __future__ import annotations
import numpy as np
import pytest

from psierra.element.tet10_linear import Tet10Linear
from psierra.element.tet10_total_lagrangian import Tet10TotalLagrangian
from psierra.element.quadrature import get_tetrahedral_quadrature
from psierra.material.hookean import Hookean
from psierra.material.neohookean import NeoHookean


class TestTet10TotalLagrangian:
    """Rigorous physical and mathematical verification of Total Lagrangian 10-node quadratic tetrahedral elements."""

    def test_pure_translation_invariance(
        self, hookean_material: Hookean, regular_tet10_coords: np.ndarray
    ) -> None:
        """Under arbitrary rigid translation, Total Lagrangian internal forces must vanish."""
        tet_tl = Tet10TotalLagrangian(hookean_material)

        t = np.array([15.2, -42.8, 8.4], dtype=np.float64)
        u_nodes = np.tile(t, (10, 1))

        _, f_int_tl = tet_tl.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet10_coords, u_nodes
        )
        tet_lin = Tet10Linear(hookean_material)
        K_e = tet_lin.compute_linear_stiffness_matrix(regular_tet10_coords)
        u_rigid = u_nodes.ravel()
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
        tet_lin = Tet10Linear(hookean_material)
        K_e = tet_lin.compute_linear_stiffness_matrix(regular_tet10_coords)
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

    def test_hyperelastic_material_coupling(
        self, neohookean_material: NeoHookean, regular_tet10_coords: np.ndarray
    ) -> None:
        """Verifies Total Lagrangian quadratic kinematics evaluate properly with Neo-Hookean material."""
        tet = Tet10TotalLagrangian(neohookean_material)
        u_nodes = 0.05 * np.ones((10, 3), dtype=np.float64)

        k_tan, f_int = tet.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet10_coords, u_nodes
        )
        assert k_tan.shape == (30, 30)
        assert f_int.shape == (30,)
        assert np.all(np.isfinite(k_tan))
        assert np.all(np.isfinite(f_int))
        assert np.allclose(k_tan, k_tan.T, atol=1e-8)
