from __future__ import annotations
import numpy as np
import pytest

from psierra.material.parameters import (
    IsotropicElasticParameters,
    MooneyRivlinParameters,
    YeohParameters,
)
from psierra.material.hookean import Hookean
from psierra.material.neohookean import NeoHookean
from psierra.material.mooney_rivlin import MooneyRivlin
from psierra.material.yeoh import Yeoh
from psierra.utility.tensor import tensor_to_voigt_stress, VOIGT_MAP_3D


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

    def test_element_inversion_raises(self, neohookean_material: NeoHookean) -> None:
        F_inverted = np.diag([1.0, 1.0, -0.5])
        with pytest.raises(ValueError, match="Element inversion detected"):
            neohookean_material.compute_stress_and_tangent(F_inverted)


class TestMooneyRivlinParameters:
    """Validates Mooney-Rivlin material parameter container and conversions."""

    def test_constants_and_properties(self) -> None:
        params = MooneyRivlinParameters.from_constants(
            c10=3000.0,
            c01=1000.0,
            bulk_modulus=50000.0,
            mass_density=1050.0,
        )
        assert np.isclose(params.c10, 3000.0)
        assert np.isclose(params.c01, 1000.0)
        assert np.isclose(params.bulk_modulus, 50000.0)
        assert np.isclose(params.shear_modulus, 8000.0)  # 2 * (3000 + 1000)
        assert np.isclose(params.mass_density, 1050.0)
        assert params.youngs_modulus > 0.0
        assert 0.0 < params.poissons_ratio < 0.5

    def test_from_isotropic(self) -> None:
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(100e3, 0.4, 1000.0)

        mr_params = MooneyRivlinParameters.from_isotropic(iso, ratio_c10=0.7)
        assert np.isclose(mr_params.shear_modulus, iso.shear_modulus)
        assert np.isclose(mr_params.bulk_modulus, iso.bulk_modulus)
        assert np.isclose(mr_params.c10, 0.7 * 0.5 * iso.shear_modulus)
        assert np.isclose(mr_params.c01, 0.3 * 0.5 * iso.shear_modulus)

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="Sum.*must be positive"):
            MooneyRivlinParameters.from_constants(c10=-2000.0, c01=1000.0, bulk_modulus=10000.0, mass_density=1000.0)
        with pytest.raises(ValueError, match="Bulk modulus must be positive"):
            MooneyRivlinParameters.from_constants(c10=1000.0, c01=500.0, bulk_modulus=-100.0, mass_density=1000.0)
        with pytest.raises(ValueError, match="Mass density must be positive"):
            MooneyRivlinParameters.from_constants(c10=1000.0, c01=500.0, bulk_modulus=10000.0, mass_density=-10.0)


class TestYeohParameters:
    """Validates Yeoh material parameter container and conversions."""

    def test_constants_and_properties(self) -> None:
        params = YeohParameters.from_constants(
            c10=4000.0,
            c20=-300.0,
            c30=150.0,
            bulk_modulus=60000.0,
            mass_density=1020.0,
        )
        assert np.isclose(params.c10, 4000.0)
        assert np.isclose(params.c20, -300.0)
        assert np.isclose(params.c30, 150.0)
        assert np.isclose(params.bulk_modulus, 60000.0)
        assert np.isclose(params.shear_modulus, 8000.0)  # 2 * 4000
        assert np.isclose(params.mass_density, 1020.0)

    def test_from_isotropic(self) -> None:
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(100e3, 0.45, 1000.0)

        yeoh_params = YeohParameters.from_isotropic(iso, c20=-100.0, c30=50.0)
        assert np.isclose(yeoh_params.shear_modulus, iso.shear_modulus)
        assert np.isclose(yeoh_params.bulk_modulus, iso.bulk_modulus)
        assert np.isclose(yeoh_params.c10, 0.5 * iso.shear_modulus)
        assert np.isclose(yeoh_params.c20, -100.0)
        assert np.isclose(yeoh_params.c30, 50.0)

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="c10 must be positive"):
            YeohParameters.from_constants(c10=-100.0, c20=0.0, c30=0.0, bulk_modulus=10000.0, mass_density=1000.0)
        with pytest.raises(ValueError, match="Bulk modulus must be positive"):
            YeohParameters.from_constants(c10=1000.0, c20=0.0, c30=0.0, bulk_modulus=-10.0, mass_density=1000.0)


class TestMooneyRivlinMaterial:
    """Validates Mooney-Rivlin hyperelastic material physics and tangent consistency."""

    def test_zero_stress_at_reference_state(self, mooney_rivlin_material: MooneyRivlin) -> None:
        F_identity = np.eye(3, dtype=np.float64)
        C, S = mooney_rivlin_material.compute_stress_and_tangent(F_identity)

        assert np.allclose(S, np.zeros((3, 3)), atol=1e-14)

    def test_tangent_at_identity_matches_linear_matrix(
        self, mooney_rivlin_material: MooneyRivlin
    ) -> None:
        F_identity = np.eye(3, dtype=np.float64)
        C_tangent, _ = mooney_rivlin_material.compute_stress_and_tangent(F_identity)
        C_linear = mooney_rivlin_material.compute_constitutive_matrix()

        assert np.allclose(C_tangent, C_linear, rtol=1e-6)

    def test_reduces_to_neohookean_when_c01_zero(self) -> None:
        """When c01=0, Mooney-Rivlin must produce mathematically identical results to Neo-Hookean."""
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(50e3, 0.45, 1000.0)
        neo = NeoHookean(iso)

        mr_params = MooneyRivlinParameters.from_constants(
            c10=0.5 * iso.shear_modulus,
            c01=0.0,
            bulk_modulus=iso.bulk_modulus,
            mass_density=iso.mass_density,
        )
        mr = MooneyRivlin(mr_params)

        F_test = np.array(
            [
                [1.10, 0.05, 0.02],
                [0.01, 0.95, 0.03],
                [0.02, 0.01, 1.05],
            ],
            dtype=np.float64,
        )

        C_neo, S_neo = neo.compute_stress_and_tangent(F_test)
        C_mr, S_mr = mr.compute_stress_and_tangent(F_test)

        assert np.allclose(S_mr, S_neo, rtol=1e-12)
        assert np.allclose(C_mr, C_neo, rtol=1e-12)

    def test_tangent_consistency_finite_difference(
        self, mooney_rivlin_material: MooneyRivlin
    ) -> None:
        """Verifies analytical tangent C matches numerical differentiation of 2nd PK stress S."""
        F0 = np.array(
            [
                [1.08, 0.03, 0.01],
                [0.02, 0.96, 0.04],
                [0.01, 0.02, 1.04],
            ],
            dtype=np.float64,
        )

        C_analytical, S0 = mooney_rivlin_material.compute_stress_and_tangent(F0)
        S0_voigt = tensor_to_voigt_stress(S0)

        eps = 1e-7
        C_numerical = np.zeros((6, 6), dtype=np.float64)

        for j in range(6):
            row, col = VOIGT_MAP_3D[j]
            delta_E = np.zeros((3, 3), dtype=np.float64)
            if row == col:
                delta_E[row, col] = eps
            else:
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

            _, S_pert = mooney_rivlin_material.compute_stress_and_tangent(F_pert)
            S_pert_voigt = tensor_to_voigt_stress(S_pert)

            C_numerical[:, j] = (S_pert_voigt - S0_voigt) / eps

        rel_diff = np.linalg.norm(C_analytical - C_numerical) / np.linalg.norm(C_analytical)
        assert rel_diff < 1e-4

    def test_element_inversion_raises(self, mooney_rivlin_material: MooneyRivlin) -> None:
        F_inverted = np.diag([1.0, 1.0, -0.5])
        with pytest.raises(ValueError, match="Element inversion detected"):
            mooney_rivlin_material.compute_stress_and_tangent(F_inverted)


class TestYeohMaterial:
    """Validates Yeoh hyperelastic material physics, strain-stiffening, and tangent consistency."""

    def test_zero_stress_at_reference_state(self, yeoh_material: Yeoh) -> None:
        F_identity = np.eye(3, dtype=np.float64)
        C, S = yeoh_material.compute_stress_and_tangent(F_identity)

        assert np.allclose(S, np.zeros((3, 3)), atol=1e-14)

    def test_tangent_at_identity_matches_linear_matrix(
        self, yeoh_material: Yeoh
    ) -> None:
        F_identity = np.eye(3, dtype=np.float64)
        C_tangent, _ = yeoh_material.compute_stress_and_tangent(F_identity)
        C_linear = yeoh_material.compute_constitutive_matrix()

        assert np.allclose(C_tangent, C_linear, rtol=1e-6)

    def test_reduces_to_neohookean_when_c20_c30_zero(self) -> None:
        """When c20=c30=0, Yeoh must produce mathematically identical results to Neo-Hookean."""
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(50e3, 0.45, 1000.0)
        neo = NeoHookean(iso)

        yeoh_params = YeohParameters.from_constants(
            c10=0.5 * iso.shear_modulus,
            c20=0.0,
            c30=0.0,
            bulk_modulus=iso.bulk_modulus,
            mass_density=iso.mass_density,
        )
        yeoh = Yeoh(yeoh_params)

        F_test = np.array(
            [
                [1.15, 0.04, 0.01],
                [0.02, 0.92, 0.03],
                [0.01, 0.02, 1.08],
            ],
            dtype=np.float64,
        )

        C_neo, S_neo = neo.compute_stress_and_tangent(F_test)
        C_yeoh, S_yeoh = yeoh.compute_stress_and_tangent(F_test)

        assert np.allclose(S_yeoh, S_neo, rtol=1e-12)
        assert np.allclose(C_yeoh, C_neo, rtol=1e-12)

    def test_strain_stiffening_behavior(self) -> None:
        """Verifies physiological strain-stiffening: at large stretch, Yeoh stress > Neo-Hookean stress."""
        bulk = 100e3
        rho = 1000.0
        c10 = 2000.0

        # Baseline NeoHookean (equivalent to Yeoh with c20=c30=0)
        neo_params = YeohParameters.from_constants(c10=c10, c20=0.0, c30=0.0, bulk_modulus=bulk, mass_density=rho)
        mat_neo = Yeoh(neo_params)

        # Stiffening Yeoh (c30 > 0)
        stiff_params = YeohParameters.from_constants(c10=c10, c20=-100.0, c30=500.0, bulk_modulus=bulk, mass_density=rho)
        mat_stiff = Yeoh(stiff_params)

        # Large unconstrained uniaxial stretch lambda = 1.6
        lam = 1.6
        # Incompressibility stretch: lambda_trans = 1 / sqrt(lambda)
        lam_t = 1.0 / np.sqrt(lam)
        F_large = np.diag([lam, lam_t, lam_t])

        _, S_neo = mat_neo.compute_stress_and_tangent(F_large)
        _, S_stiff = mat_stiff.compute_stress_and_tangent(F_large)

        # Tensile PK2 stress along stretch direction must be strictly higher due to strain-stiffening
        assert S_stiff[0, 0] > S_neo[0, 0]

    def test_tangent_consistency_finite_difference(
        self, yeoh_material: Yeoh
    ) -> None:
        """Verifies analytical tangent C matches numerical differentiation of 2nd PK stress S."""
        F0 = np.array(
            [
                [1.08, 0.03, 0.01],
                [0.02, 0.96, 0.04],
                [0.01, 0.02, 1.04],
            ],
            dtype=np.float64,
        )

        C_analytical, S0 = yeoh_material.compute_stress_and_tangent(F0)
        S0_voigt = tensor_to_voigt_stress(S0)

        eps = 1e-7
        C_numerical = np.zeros((6, 6), dtype=np.float64)

        for j in range(6):
            row, col = VOIGT_MAP_3D[j]
            delta_E = np.zeros((3, 3), dtype=np.float64)
            if row == col:
                delta_E[row, col] = eps
            else:
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

            _, S_pert = yeoh_material.compute_stress_and_tangent(F_pert)
            S_pert_voigt = tensor_to_voigt_stress(S_pert)

            C_numerical[:, j] = (S_pert_voigt - S0_voigt) / eps

        rel_diff = np.linalg.norm(C_analytical - C_numerical) / np.linalg.norm(C_analytical)
        assert rel_diff < 1e-4

    def test_element_inversion_raises(self, yeoh_material: Yeoh) -> None:
        F_inverted = np.diag([1.0, 1.0, -0.5])
        with pytest.raises(ValueError, match="Element inversion detected"):
            yeoh_material.compute_stress_and_tangent(F_inverted)

