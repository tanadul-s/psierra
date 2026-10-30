from __future__ import annotations
import numpy as np
import pytest

from psierra.material.parameters import IsotropicElasticParameters
from psierra.material.hookean import Hookean
from psierra.material.neohookean import NeoHookean
from psierra.utility.tensor import tensor_to_voigt_stress


class TestIsotropicElasticParameters:
    """Validates elasticity parameter conversions, relationships, and error bounds."""

    def test_youngs_poisson_relationships(self) -> None:
        E = 210e9
        nu = 0.3
        rho = 7800.0

        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(youngs_modulus=E, poissons_ratio=nu, mass_density=rho)

        expected_G = E / (2.0 * (1.0 + nu))
        expected_K = E / (3.0 * (1.0 - 2.0 * nu))
        expected_lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

        assert np.isclose(params.youngs_modulus, E)
        assert np.isclose(params.poissons_ratio, nu)
        assert np.isclose(params.shear_modulus, expected_G)
        assert np.isclose(params.bulk_modulus, expected_K)
        assert np.isclose(params.lame_parameter_1, expected_lam)
        assert np.isclose(params.lame_parameter_2, expected_G)
        assert np.isclose(params.mass_density, rho)

    def test_bulk_shear_roundtrip(self) -> None:
        params1 = IsotropicElasticParameters()
        params1.set_from_youngs_poisson(100e3, 0.4, 1000.0)

        params2 = IsotropicElasticParameters()
        params2.set_from_bulk_shear(
            bulk_modulus=params1.bulk_modulus,
            shear_modulus=params1.shear_modulus,
            mass_density=1000.0,
        )

        assert np.isclose(params2.youngs_modulus, params1.youngs_modulus, rtol=1e-10)
        assert np.isclose(params2.poissons_ratio, params1.poissons_ratio, rtol=1e-10)

    def test_invalid_parameters_raise(self) -> None:
        params = IsotropicElasticParameters()
        with pytest.raises(ValueError, match="Young's modulus must be positive"):
            params.set_from_youngs_poisson(youngs_modulus=-100.0, poissons_ratio=0.3, mass_density=1000.0)

        with pytest.raises(ValueError, match="Poisson's ratio must satisfy"):
            params.set_from_youngs_poisson(youngs_modulus=1000.0, poissons_ratio=0.55, mass_density=1000.0)

        with pytest.raises(ValueError, match="Mass density must be positive"):
            params.set_from_youngs_poisson(youngs_modulus=1000.0, poissons_ratio=0.3, mass_density=-5.0)


class TestHookeanMaterial:
    """Validates linear Hookean constitutive tensor properties."""

    def test_constitutive_matrix_symmetry_and_positive_definiteness(
        self, hookean_material: Hookean
    ) -> None:
        C = hookean_material.compute_constitutive_matrix()
        assert C.shape == (6, 6)

        # Symmetry: C == C^T
        assert np.allclose(C, C.T, atol=1e-12)

        # Positive-definiteness: all eigenvalues > 0
        eigenvalues = np.linalg.eigvalsh(C)
        assert np.all(eigenvalues > 0.0)

    def test_stress_at_identity_is_zero(self, hookean_material: Hookean) -> None:
        F_identity = np.eye(3, dtype=np.float64)
        C, S = hookean_material.compute_stress_and_tangent(F_identity)

        assert C.shape == (6, 6)
        assert S.shape == (3, 3)
        assert np.allclose(S, np.zeros((3, 3)), atol=1e-14)


class TestNeoHookeanMaterial:
    """Validates non-linear hyperelastic Neo-Hookean mechanics and tangent consistency."""

    def test_zero_stress_at_reference_state(self, neohookean_material: NeoHookean) -> None:
        F_identity = np.eye(3, dtype=np.float64)
        C, S = neohookean_material.compute_stress_and_tangent(F_identity)

        assert np.allclose(S, np.zeros((3, 3)), atol=1e-14)

        # At zero deformation, tangent C must match linear Hookean C
        params = neohookean_material.parameters
        hookean = Hookean(params)
        C_hookean = hookean.compute_constitutive_matrix()
        assert np.allclose(C, C_hookean, rtol=1e-5)

    def test_tangent_consistency_finite_difference(
        self, neohookean_material: NeoHookean
    ) -> None:
        """
        Verifies that analytical tangent elasticity matrix C matches numerical
        differentiation of 2nd PK stress S with respect to Green-Lagrange strain E.
        """
        from psierra.utility.tensor import VOIGT_MAP_3D

        # Non-trivial deformation gradient F (moderate shear + stretch)
        F0 = np.array(
            [
                [1.08, 0.03, 0.01],
                [0.02, 0.96, 0.04],
                [0.01, 0.02, 1.04],
            ],
            dtype=np.float64,
        )

        C_analytical, S0 = neohookean_material.compute_stress_and_tangent(F0)
        S0_voigt = tensor_to_voigt_stress(S0)

        eps = 1e-7
        C_numerical = np.zeros((6, 6), dtype=np.float64)

        for j in range(6):
            row, col = VOIGT_MAP_3D[j]
            delta_E = np.zeros((3, 3), dtype=np.float64)
            if row == col:
                delta_E[row, col] = eps
            else:
                # Engineering shear gamma_ij = 2 * E_ij
                delta_E[row, col] = 0.5 * eps
                delta_E[col, row] = 0.5 * eps

            E0 = 0.5 * (F0.T @ F0 - np.eye(3))
            E_pert = E0 + delta_E
            C_pert = 2.0 * E_pert + np.eye(3)

            w, V = np.linalg.eigh(C_pert)
            U_pert = V @ np.diag(np.sqrt(w)) @ V.T

            C0 = F0.T @ F0
            w0, V0 = np.linalg.eigh(C0)
            U0 = V0 @ np.diag(np.sqrt(w0)) @ V0.T
            R = F0 @ np.linalg.inv(U0)
            F_pert = R @ U_pert

            _, S_pert = neohookean_material.compute_stress_and_tangent(F_pert)
            S_pert_voigt = tensor_to_voigt_stress(S_pert)

            C_numerical[:, j] = (S_pert_voigt - S0_voigt) / eps

        rel_diff = np.linalg.norm(C_analytical - C_numerical) / np.linalg.norm(C_analytical)
        assert rel_diff < 1e-4
