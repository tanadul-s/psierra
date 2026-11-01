from __future__ import annotations
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from psierra.material import IsotropicElasticParameters, NeoHookean, Hookean
from psierra.element import Tet10TotalLagrangian, Tet10UpdatedLagrangian


@pytest.fixture
def regular_tet10_coords() -> np.ndarray:
    """Standard unit right-angled 10-node quadratic tetrahedron."""
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    coords = np.zeros((10, 3), dtype=np.float64)
    coords[0:4] = v
    coords[4] = 0.5 * (v[0] + v[1])
    coords[5] = 0.5 * (v[1] + v[2])
    coords[6] = 0.5 * (v[2] + v[0])
    coords[7] = 0.5 * (v[0] + v[3])
    coords[8] = 0.5 * (v[1] + v[3])
    coords[9] = 0.5 * (v[2] + v[3])
    return coords


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


class TestTet10UpdatedLagrangian:
    """Rigorous verification of Updated Lagrangian 10-node quadratic tetrahedral element."""

    def test_pure_translation_invariance(
        self, regular_tet10_coords: np.ndarray, hookean_material: Hookean
    ) -> None:
        """Verifies exact zero internal force under finite 3D rigid body translation."""
        elem = Tet10UpdatedLagrangian(hookean_material)
        t_vec = np.array([15.2, -42.8, 8.4], dtype=np.float64)
        u_nodes = np.tile(t_vec, (10, 1))

        k_tan, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet10_coords, u_nodes
        )

        np.testing.assert_allclose(f_int, np.zeros(30), atol=1e-8)

    def test_pure_rotation_invariance(
        self, regular_tet10_coords: np.ndarray, neohookean_material: NeoHookean
    ) -> None:
        """Under finite 3D rigid body rotation, internal forces must evaluate to zero."""
        elem = Tet10UpdatedLagrangian(neohookean_material)

        rot = Rotation.from_euler("zyx", [30.0, -40.0, 50.0], degrees=True)
        R = rot.as_matrix()

        x_rot = regular_tet10_coords @ R.T
        u_rot = x_rot - regular_tet10_coords

        k_tan, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet10_coords, u_rot
        )

        np.testing.assert_allclose(f_int, np.zeros(30), atol=1e-8)

    def test_exact_equivalence_to_total_lagrangian(
        self, regular_tet10_coords: np.ndarray, neohookean_material: NeoHookean
    ) -> None:
        """
        Verifies that Updated Lagrangian and Total Lagrangian produce mathematically
        identical tangent stiffness matrices and internal force vectors under large strains.
        """
        elem_tl = Tet10TotalLagrangian(neohookean_material)
        elem_ul = Tet10UpdatedLagrangian(neohookean_material)

        np.random.seed(42)
        u_nodes = 0.03 * np.random.randn(10, 3)

        k_tan_tl, f_int_tl = elem_tl.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet10_coords, u_nodes
        )
        k_tan_ul, f_int_ul = elem_ul.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet10_coords, u_nodes
        )

        # Internal forces must match to within numerical precision
        np.testing.assert_allclose(
            f_int_ul,
            f_int_tl,
            atol=1e-10,
            err_msg="Tet10 UL and TL internal force vectors do not match",
        )

        # Tangent stiffness must match to within numerical precision
        np.testing.assert_allclose(
            k_tan_ul,
            k_tan_tl,
            atol=1e-8,
            err_msg="Tet10 UL and TL tangent stiffness matrices do not match",
        )

    def test_tangent_consistency_finite_difference(
        self, regular_tet10_coords: np.ndarray, neohookean_material: NeoHookean
    ) -> None:
        """Verifies directional derivative consistency: d(f_int)/du == k_tan."""
        elem = Tet10UpdatedLagrangian(neohookean_material)

        np.random.seed(42)
        u_nodes = 0.01 * np.random.randn(10, 3)

        k_tan, _ = elem.compute_nonlinear_force_and_tangent_stiffness(
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

            _, f_plus = elem.compute_nonlinear_force_and_tangent_stiffness(
                regular_tet10_coords, u_plus
            )
            _, f_minus = elem.compute_nonlinear_force_and_tangent_stiffness(
                regular_tet10_coords, u_minus
            )

            k_tan_fd[:, dof] = (f_plus - f_minus) / (2.0 * h)

        rel_error = np.linalg.norm(k_tan - k_tan_fd) / np.linalg.norm(k_tan)
        assert rel_error < 1e-5, f"Tangent stiffness error: {rel_error:.4e}"

    def test_inversion_detection(
        self, regular_tet10_coords: np.ndarray, hookean_material: Hookean
    ) -> None:
        """Verifies that an inverted element raises ValueError."""
        elem = Tet10UpdatedLagrangian(hookean_material)

        u_invert = np.zeros_like(regular_tet10_coords)
        u_invert[3, 2] = -2.0  # Inverts node 3 across XY plane

        with pytest.raises(ValueError, match="Element inversion detected in Tet10UpdatedLagrangian"):
            elem.compute_nonlinear_force_and_tangent_stiffness(
                regular_tet10_coords, u_invert
            )
