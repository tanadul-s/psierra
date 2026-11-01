from __future__ import annotations
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear, Tet4Corotational


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
def hookean_material() -> Hookean:
    """Standard isotropic Hookean material model."""
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=1000.0,
        poissons_ratio=0.30,
        mass_density=1.0e-6,
    )
    return Hookean(params)


class TestTet4Corotational:
    """Comprehensive test suite for Tet4Corotational finite element."""

    def test_polar_decomposition_properties(self) -> None:
        """Verifies mathematical properties of SVD polar decomposition."""
        # 1. Pure rotation
        rot = Rotation.from_euler("zyx", [35.0, 45.0, -60.0], degrees=True)
        R_true = rot.as_matrix()
        R_calc, U_calc = Tet4Corotational.compute_polar_decomposition(R_true)

        np.testing.assert_allclose(R_calc, R_true, atol=1e-12)
        np.testing.assert_allclose(U_calc, np.identity(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R_calc), 1.0, atol=1e-12)

        # 2. Combined stretch and rotation
        U_true = np.array(
            [[1.2, 0.05, 0.0], [0.05, 0.9, 0.0], [0.0, 0.0, 1.05]],
            dtype=np.float64,
        )
        F = R_true @ U_true
        R_calc, U_calc = Tet4Corotational.compute_polar_decomposition(F)

        np.testing.assert_allclose(R_calc, R_true, atol=1e-12)
        np.testing.assert_allclose(U_calc, U_true, atol=1e-12)
        assert np.isclose(np.linalg.det(R_calc), 1.0, atol=1e-12)

    def test_rigid_translation_invariance(
        self, unit_tet_coords: np.ndarray, hookean_material: Hookean
    ) -> None:
        """Verifies exact zero internal force under finite 3D rigid body translation."""
        elem = Tet4Corotational(hookean_material)
        t_vec = np.array([12.5, -45.0, 88.2], dtype=np.float64)
        u_nodes = np.tile(t_vec, (4, 1))

        k_tan, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(
            unit_tet_coords, u_nodes
        )

        # Rigid translation must produce exactly zero internal force
        np.testing.assert_allclose(f_int, np.zeros(12), atol=1e-12)

    def test_rigid_rotation_invariance(
        self, unit_tet_coords: np.ndarray, hookean_material: Hookean
    ) -> None:
        """
        Verifies that finite rigid body rotations produce zero internal force,
        eliminating the artificial volume inflation of small-strain linear elasticity.
        """
        elem_corot = Tet4Corotational(hookean_material)
        elem_linear = Tet4Linear(hookean_material)

        # Test multiple finite rotation angles around arbitrary 3D axis
        for angle_deg in [15.0, 45.0, 90.0, 135.0]:
            axis = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            axis /= np.linalg.norm(axis)
            rot = Rotation.from_rotvec(np.radians(angle_deg) * axis)
            R_target = rot.as_matrix()

            # Rotate reference coordinates
            x_rot = unit_tet_coords @ R_target.T
            u_rot = x_rot - unit_tet_coords

            # Corotational evaluation
            R_extracted = elem_corot.extract_rotation(unit_tet_coords, u_rot)
            np.testing.assert_allclose(R_extracted, R_target, atol=1e-12)

            k_tan, f_int_corot = elem_corot.compute_nonlinear_force_and_tangent_stiffness(
                unit_tet_coords, u_rot
            )

            # Corotational internal forces MUST be machine zero under pure rotation
            np.testing.assert_allclose(
                f_int_corot, np.zeros(12), atol=1e-11,
                err_msg=f"Tet4Corotational produced non-zero force at {angle_deg} deg",
            )

            # Contrast with Linear Hookean (which produces massive unphysical forces)
            K0 = elem_linear.compute_linear_stiffness_matrix(unit_tet_coords)
            f_int_linear = K0 @ u_rot.flatten()
            assert np.linalg.norm(f_int_linear) > 10.0, (
                "Linear Hookean should demonstrate large unphysical artificial force"
            )

    def test_rotated_strain_equivariance(
        self, unit_tet_coords: np.ndarray, hookean_material: Hookean
    ) -> None:
        """
        Verifies frame indifference (objectivity): rotating a strained element
        results in identically rotated internal force vectors.
        """
        elem = Tet4Corotational(hookean_material)

        # Apply a pure stretch along X
        u_stretch = np.zeros_like(unit_tet_coords)
        u_stretch[:, 0] = unit_tet_coords[:, 0] * 0.05  # 5% stretch

        _, f_ref = elem.compute_nonlinear_force_and_tangent_stiffness(
            unit_tet_coords, u_stretch
        )

        # Rotate by 60 degrees around Z axis
        rot = Rotation.from_euler("z", 60.0, degrees=True)
        R = rot.as_matrix()
        R_blk = np.kron(np.identity(4, dtype=np.float64), R)

        x_rot = (unit_tet_coords + u_stretch) @ R.T
        u_rot = x_rot - unit_tet_coords

        _, f_rot = elem.compute_nonlinear_force_and_tangent_stiffness(
            unit_tet_coords, u_rot
        )

        # Rotated reference force MUST match corotational internal force
        f_expected = R_blk @ f_ref
        np.testing.assert_allclose(f_rot, f_expected, atol=1e-12)

    def test_tangent_stiffness_symmetry_and_kernel(
        self, unit_tet_coords: np.ndarray, hookean_material: Hookean
    ) -> None:
        """Verifies that warped tangent stiffness is symmetric with 6 zero eigenvalues."""
        elem = Tet4Corotational(hookean_material)

        # Arbitrary rotation and stretch
        rot = Rotation.from_euler("xyz", [20.0, -35.0, 50.0], degrees=True)
        R = rot.as_matrix()
        u_def = (unit_tet_coords * 1.05) @ R.T - unit_tet_coords

        k_tan, _ = elem.compute_nonlinear_force_and_tangent_stiffness(
            unit_tet_coords, u_def
        )

        # 1. Exact Symmetry
        np.testing.assert_allclose(k_tan, k_tan.T, atol=1e-12)

        # 2. Kernel dimension (rigid body modes)
        eigenvalues = np.linalg.eigvalsh(k_tan)
        zero_modes = np.sum(np.abs(eigenvalues) < 1e-8)
        positive_modes = np.sum(eigenvalues >= 1e-8)

        assert zero_modes == 6, f"Expected 6 rigid body modes, found {zero_modes}"
        assert positive_modes == 6, f"Expected 6 elastic modes, found {positive_modes}"

    def test_inversion_detection(
        self, unit_tet_coords: np.ndarray, hookean_material: Hookean
    ) -> None:
        """Verifies that an inverted element (det(F) <= 0) raises ValueError."""
        elem = Tet4Corotational(hookean_material)

        # Mirror Node 3 across the XY plane (inverts tetrahedron)
        u_invert = np.zeros_like(unit_tet_coords)
        u_invert[3, 2] = -2.0  # Moves Z from +1.0 to -1.0

        with pytest.raises(ValueError, match="Element inversion detected in Tet4Corotational"):
            elem.compute_nonlinear_force_and_tangent_stiffness(
                unit_tet_coords, u_invert
            )

    def test_corotational_stress_and_strain(
        self, unit_tet_coords: np.ndarray, hookean_material: Hookean
    ) -> None:
        """Verifies Cauchy stress and engineering strain calculation."""
        elem = Tet4Corotational(hookean_material)

        # 1. Pure rigid rotation produces zero stress and strain
        rot = Rotation.from_euler("y", 45.0, degrees=True)
        R = rot.as_matrix()
        u_rot = unit_tet_coords @ R.T - unit_tet_coords

        sigma_zero, eps_zero, R_extracted = elem.compute_corotational_stress(
            unit_tet_coords, u_rot
        )
        np.testing.assert_allclose(sigma_zero, np.zeros((3, 3)), atol=1e-11)
        np.testing.assert_allclose(eps_zero, np.zeros((3, 3)), atol=1e-11)
        np.testing.assert_allclose(R_extracted, R, atol=1e-12)

        # 2. Pure uniaxial stretch along X (eps_xx = 0.01)
        u_stretch = np.zeros_like(unit_tet_coords)
        u_stretch[:, 0] = unit_tet_coords[:, 0] * 0.01

        sigma, eps, _ = elem.compute_corotational_stress(unit_tet_coords, u_stretch)
        assert np.isclose(eps[0, 0], 0.01, atol=1e-12)
        assert sigma[0, 0] > 0.0  # Tensile stress in X direction
