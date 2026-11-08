"""
Unit and benchmark tests for Tet4TotalLagrangian finite element.

Validates:
1. Rigid body translation invariance (internal forces vanish to machine precision).
2. Large finite rigid body rotation invariance (Green-Lagrange strain and internal forces vanish).
3. Small-strain limit matching linear elasticity (f_int matches K_lin * u).
4. Directional derivative tangent consistency (k_tan matches d f_int / d u via finite differences).
5. Hyperelastic constitutive law compatibility (NeoHookean, MooneyRivlin, Yeoh).
"""

from __future__ import annotations
import numpy as np
import pytest

from psierra.element.tet4_linear import Tet4Linear
from psierra.element.tet4_total_lagrangian import Tet4TotalLagrangian
from psierra.material.hookean import Hookean
from psierra.material.neohookean import NeoHookean
from psierra.material.mooney_rivlin import MooneyRivlin
from psierra.material.yeoh import Yeoh


class TestTet4TotalLagrangian:
    """Rigorous physical and mathematical verification of Total Lagrangian 4-node tetrahedral elements."""

    def test_pure_translation_invariance(
        self, hookean_material: Hookean, regular_tet_coords: np.ndarray
    ) -> None:
        """Under arbitrary rigid translation, Total Lagrangian internal forces must vanish."""
        tet = Tet4TotalLagrangian(hookean_material)

        t = np.array([12.5, -34.2, 5.8], dtype=np.float64)
        u_nodes = np.tile(t, (4, 1))

        k_tan, f_int = tet.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet_coords, u_nodes
        )

        assert np.allclose(f_int, 0.0, atol=1e-10)

    def test_pure_rotation_invariance_nonlinear(
        self, neohookean_material: NeoHookean, regular_tet_coords: np.ndarray
    ) -> None:
        """Under large finite rigid body rotation, Green-Lagrange strain and f_int must be zero."""
        tet = Tet4TotalLagrangian(neohookean_material)

        # Finite rotation matrix (30 deg about axis [1, 1, 1])
        theta = np.deg2rad(30.0)
        axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        K_mat = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        R = np.eye(3) + np.sin(theta) * K_mat + (1.0 - np.cos(theta)) * (K_mat @ K_mat)

        # Rotated coordinates x = X @ R^T
        x_rotated = regular_tet_coords @ R.T
        u_nodes = x_rotated - regular_tet_coords

        k_tan, f_int = tet.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet_coords, u_nodes
        )

        # Internal force must vanish under pure rigid rotation
        assert np.allclose(f_int, 0.0, atol=1e-10)

    def test_small_strain_limit_matches_linear(
        self, hookean_material: Hookean, regular_tet_coords: np.ndarray
    ) -> None:
        """In the infinitesimal strain limit, Total Lagrangian internal force matches K_lin * u."""
        tet_lin = Tet4Linear(hookean_material)
        tet_tl = Tet4TotalLagrangian(hookean_material)

        K_lin = tet_lin.compute_linear_stiffness_matrix(regular_tet_coords)

        # Small strain displacement field (strain ~ 1e-5, quadratic terms ~ 1e-10)
        np.random.seed(42)
        u_nodes_small = 1e-5 * np.random.randn(4, 3)
        u_flat_small = u_nodes_small.ravel()

        f_lin_expected = K_lin @ u_flat_small
        _, f_tl = tet_tl.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet_coords, u_nodes_small
        )

        rel_diff = np.linalg.norm(f_tl - f_lin_expected) / np.linalg.norm(f_lin_expected)
        assert rel_diff < 1e-4, f"Small strain mismatch: {rel_diff:.4e}"

    def test_nonlinear_tangent_consistency(
        self, neohookean_material: NeoHookean, regular_tet_coords: np.ndarray
    ) -> None:
        """
        Verifies that tangent stiffness k_tan matches directional derivative
        of internal forces via central finite differences: d f_int / d u == k_tan.
        """
        tet = Tet4TotalLagrangian(neohookean_material)

        np.random.seed(123)
        u_nodes = 0.02 * np.random.randn(4, 3)

        k_tan, _ = tet.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet_coords, u_nodes
        )

        # Central finite difference perturbation
        h = 1e-7
        k_tan_fd = np.zeros((12, 12), dtype=np.float64)

        for dof in range(12):
            node_idx = dof // 3
            comp_idx = dof % 3

            u_plus = u_nodes.copy()
            u_minus = u_nodes.copy()
            u_plus[node_idx, comp_idx] += h
            u_minus[node_idx, comp_idx] -= h

            _, f_plus = tet.compute_nonlinear_force_and_tangent_stiffness(
                regular_tet_coords, u_plus
            )
            _, f_minus = tet.compute_nonlinear_force_and_tangent_stiffness(
                regular_tet_coords, u_minus
            )

            k_tan_fd[:, dof] = (f_plus - f_minus) / (2.0 * h)

        rel_error = np.linalg.norm(k_tan - k_tan_fd) / np.linalg.norm(k_tan)
        assert rel_error < 1e-5, f"Tangent stiffness error too high: {rel_error:.4e}"

    def test_hyperelastic_material_coupling(
        self,
        neohookean_material: NeoHookean,
        mooney_rivlin_material: MooneyRivlin,
        yeoh_material: Yeoh,
        regular_tet_coords: np.ndarray,
    ) -> None:
        """Verifies Total Lagrangian kinematics work across diverse hyperelastic materials."""
        u_nodes = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.0, 0.05],
            ],
            dtype=np.float64,
        )

        for mat in [neohookean_material, mooney_rivlin_material, yeoh_material]:
            elem = Tet4TotalLagrangian(mat)
            k_tan, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(
                regular_tet_coords, u_nodes
            )
            assert k_tan.shape == (12, 12)
            assert f_int.shape == (12,)
            assert np.all(np.isfinite(k_tan))
            assert np.all(np.isfinite(f_int))
            # Symmetry of tangent stiffness
            assert np.allclose(k_tan, k_tan.T, atol=1e-8)
