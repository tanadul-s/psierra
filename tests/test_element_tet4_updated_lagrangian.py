from __future__ import annotations
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from psierra.material import (
    IsotropicElasticParameters,
    NeoHookean,
    Hookean,
    Viscoelastic,
    PronyTerm,
)
from psierra.element import Tet4TotalLagrangian, Tet4UpdatedLagrangian


@pytest.fixture
def unit_tet_coords() -> np.ndarray:
    """Standard unit right-angled tetrahedron."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def neohookean_material() -> NeoHookean:
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=1.0e5,
        poissons_ratio=0.35,
        mass_density=1000.0,
    )
    return NeoHookean(params)


@pytest.fixture
def hookean_material() -> Hookean:
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=2.0e5,
        poissons_ratio=0.30,
        mass_density=7850.0,
    )
    return Hookean(params)


class TestTet4UpdatedLagrangian:
    """Rigorous verification of Updated Lagrangian 4-node tetrahedral element."""

    def test_pure_translation_invariance(
        self, unit_tet_coords: np.ndarray, hookean_material: Hookean
    ) -> None:
        """Verifies exact zero internal force under finite 3D rigid body translation."""
        elem = Tet4UpdatedLagrangian(hookean_material)
        t_vec = np.array([12.5, -45.0, 88.2], dtype=np.float64)
        u_nodes = np.tile(t_vec, (4, 1))

        k_tan, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(
            unit_tet_coords, u_nodes
        )

        np.testing.assert_allclose(f_int, np.zeros(12), atol=1e-10)

    def test_pure_rotation_invariance(
        self, unit_tet_coords: np.ndarray, neohookean_material: NeoHookean
    ) -> None:
        """Under finite 3D rigid body rotation, internal forces must evaluate to zero."""
        elem = Tet4UpdatedLagrangian(neohookean_material)

        rot = Rotation.from_euler("zyx", [35.0, -45.0, 60.0], degrees=True)
        R = rot.as_matrix()

        x_rot = unit_tet_coords @ R.T
        u_rot = x_rot - unit_tet_coords

        k_tan, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(
            unit_tet_coords, u_rot
        )

        np.testing.assert_allclose(f_int, np.zeros(12), atol=1e-10)

    def test_exact_equivalence_to_total_lagrangian(
        self, unit_tet_coords: np.ndarray, neohookean_material: NeoHookean
    ) -> None:
        """
        Verifies that Updated Lagrangian and Total Lagrangian produce mathematically
        identical tangent stiffness matrices and internal force vectors under large strains.
        """
        elem_tl = Tet4TotalLagrangian(neohookean_material)
        elem_ul = Tet4UpdatedLagrangian(neohookean_material)

        # Arbitrary finite deformation state: combined stretch, shear, and rotation
        u_nodes = np.array(
            [
                [0.05, -0.02, 0.01],
                [0.10, 0.04, -0.03],
                [-0.02, 0.08, 0.05],
                [0.03, -0.05, 0.12],
            ],
            dtype=np.float64,
        )

        k_tan_tl, f_int_tl = elem_tl.compute_nonlinear_force_and_tangent_stiffness(
            unit_tet_coords, u_nodes
        )
        k_tan_ul, f_int_ul = elem_ul.compute_nonlinear_force_and_tangent_stiffness(
            unit_tet_coords, u_nodes
        )

        # Internal forces must match to within machine precision
        np.testing.assert_allclose(
            f_int_ul,
            f_int_tl,
            atol=1e-10,
            err_msg="UL and TL internal force vectors do not match",
        )

        # Tangent stiffness must match to within machine precision
        np.testing.assert_allclose(
            k_tan_ul,
            k_tan_tl,
            atol=1e-9,
            err_msg="UL and TL tangent stiffness matrices do not match",
        )

    def test_tangent_consistency_finite_difference(
        self, unit_tet_coords: np.ndarray, neohookean_material: NeoHookean
    ) -> None:
        """Verifies directional derivative consistency: d(f_int)/du == k_tan."""
        elem = Tet4UpdatedLagrangian(neohookean_material)

        np.random.seed(42)
        u_nodes = 0.02 * np.random.randn(4, 3)

        k_tan, _ = elem.compute_nonlinear_force_and_tangent_stiffness(
            unit_tet_coords, u_nodes
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

            _, f_plus = elem.compute_nonlinear_force_and_tangent_stiffness(
                unit_tet_coords, u_plus
            )
            _, f_minus = elem.compute_nonlinear_force_and_tangent_stiffness(
                unit_tet_coords, u_minus
            )

            k_tan_fd[:, dof] = (f_plus - f_minus) / (2.0 * h)

        rel_error = np.linalg.norm(k_tan - k_tan_fd) / np.linalg.norm(k_tan)
        assert rel_error < 1e-5, f"Tangent stiffness error: {rel_error:.4e}"

    def test_inversion_detection(
        self, unit_tet_coords: np.ndarray, hookean_material: Hookean
    ) -> None:
        """Verifies that an inverted element raises ValueError."""
        elem = Tet4UpdatedLagrangian(hookean_material)

        u_invert = np.zeros_like(unit_tet_coords)
        u_invert[3, 2] = -2.0  # Moves Z from +1.0 to -1.0

        with pytest.raises(ValueError, match="Element inversion detected in Tet4UpdatedLagrangian"):
            elem.compute_nonlinear_force_and_tangent_stiffness(
                unit_tet_coords, u_invert
            )

    def test_viscoelastic_equivalence_to_total_lagrangian(
        self, unit_tet_coords: np.ndarray, neohookean_material: NeoHookean
    ) -> None:
        """
        Verifies that Updated Lagrangian and Total Lagrangian produce identical
        tangent stiffness matrices and internal force vectors when coupled with
        a history-dependent Viscoelastic material over multiple time steps.
        """
        prony = [
            PronyTerm(relative_modulus=0.3, relaxation_time=0.1),
            PronyTerm(relative_modulus=0.3, relaxation_time=0.5),
        ]
        mat_tl = Viscoelastic(neohookean_material, prony)
        mat_ul = Viscoelastic(neohookean_material, prony)
        mat_tl.initialize_history(1)
        mat_ul.initialize_history(1)
        dt = 0.05
        mat_tl.set_time_step(dt)
        mat_ul.set_time_step(dt)

        elem_tl = Tet4TotalLagrangian(mat_tl)
        elem_ul = Tet4UpdatedLagrangian(mat_ul)

        # Deform across 3 time steps
        for step in range(1, 4):
            factor = step * 0.05
            u_nodes = factor * np.array(
                [
                    [0.05, -0.02, 0.01],
                    [0.10, 0.04, -0.03],
                    [-0.02, 0.08, 0.05],
                    [0.03, -0.05, 0.12],
                ],
                dtype=np.float64,
            )
            k_tl, f_tl = elem_tl.compute_nonlinear_force_and_tangent_stiffness(
                unit_tet_coords, u_nodes, elem_index=0
            )
            k_ul, f_ul = elem_ul.compute_nonlinear_force_and_tangent_stiffness(
                unit_tet_coords, u_nodes, elem_index=0
            )

            np.testing.assert_allclose(f_ul, f_tl, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(k_ul, k_tl, rtol=1e-10, atol=1e-10)

            mat_tl.commit_state()
            mat_ul.commit_state()


