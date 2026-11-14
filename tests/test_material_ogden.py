from __future__ import annotations
import numpy as np
import pytest

from psierra.material.parameters import (
    IsotropicElasticParameters,
    OgdenParameters,
)
from psierra.material.hookean import Hookean
from psierra.material.ogden import Ogden
from psierra.material.viscoelastic import Viscoelastic, PronyTerm
from psierra.utility.tensor import tensor_to_voigt_stress, VOIGT_MAP_3D

from psierra.element.tet4_total_lagrangian import Tet4TotalLagrangian
from psierra.element.tet4_updated_lagrangian import Tet4UpdatedLagrangian
from psierra.element.tet4_fbar import Tet4Fbar
from psierra.element.tet10_total_lagrangian import Tet10TotalLagrangian
from psierra.element.tet4_corotational import Tet4Corotational


class TestOgdenParameters:
    """Validates Ogden parameter representations, derivations, and safety guards."""

    def test_order1_parameters_from_constants(self) -> None:
        params = OgdenParameters.from_constants(
            mu=[7777.77],
            alpha=[2.0],
            bulk_modulus=50000.0,
            mass_density=1100e-6,
        )
        assert params.order == 1
        assert np.isclose(params.mu[0], 7777.77)
        assert np.isclose(params.alpha[0], 2.0)
        assert np.isclose(params.bulk_modulus, 50000.0)
        assert np.isclose(params.shear_modulus, 7777.77)  # 0.5 * (7777.77 * 2)
        assert np.isclose(params.mass_density, 1100e-6)
        assert params.youngs_modulus > 0.0
        assert 0.0 < params.poissons_ratio < 0.5

    def test_order2_parameters_from_isotropic(self) -> None:
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1100e-6)

        ogden_params = OgdenParameters.from_isotropic(iso, order=2, alpha=[2.0, -2.0])
        assert ogden_params.order == 2
        assert np.isclose(ogden_params.shear_modulus, iso.shear_modulus)
        assert np.isclose(ogden_params.bulk_modulus, iso.bulk_modulus)
        assert np.isclose(ogden_params.youngs_modulus, iso.youngs_modulus)
        assert np.isclose(ogden_params.poissons_ratio, iso.poissons_ratio)

    def test_order3_parameters_from_isotropic(self) -> None:
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1100e-6)

        ogden_params = OgdenParameters.from_isotropic(iso, order=3)
        assert ogden_params.order == 3
        assert np.isclose(ogden_params.shear_modulus, iso.shear_modulus)
        assert np.isclose(ogden_params.bulk_modulus, iso.bulk_modulus)
        assert np.isclose(ogden_params.youngs_modulus, iso.youngs_modulus)
        assert np.isclose(ogden_params.poissons_ratio, iso.poissons_ratio)

    def test_brain_parenchyma_preset(self) -> None:
        params = OgdenParameters.from_brain_parenchyma(bulk_modulus=10000.0, mass_density=1040.0)
        assert params.order == 1
        assert params.shear_modulus > 0.0  # -842.0 * -4.7 > 0
        assert np.isclose(params.alpha[0], -4.7)

    def test_rubber_preset(self) -> None:
        params = OgdenParameters.from_rubber(bulk_modulus=1e7, mass_density=1100.0)
        assert params.order == 3
        assert params.shear_modulus > 0.0

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="Length of mu.*must match"):
            OgdenParameters.from_constants(mu=[1000.0, 500.0], alpha=[2.0], bulk_modulus=50000.0, mass_density=1000.0)

        with pytest.raises(ValueError, match="cannot be zero"):
            OgdenParameters.from_constants(mu=[1000.0], alpha=[0.0], bulk_modulus=50000.0, mass_density=1000.0)

        with pytest.raises(ValueError, match="positive initial shear stiffness"):
            OgdenParameters.from_constants(mu=[-1000.0], alpha=[2.0], bulk_modulus=50000.0, mass_density=1000.0)

        with pytest.raises(ValueError, match="Bulk modulus must be positive"):
            OgdenParameters.from_constants(mu=[1000.0], alpha=[2.0], bulk_modulus=-500.0, mass_density=1000.0)

        with pytest.raises(ValueError, match="Mass density must be positive"):
            OgdenParameters.from_constants(mu=[1000.0], alpha=[2.0], bulk_modulus=50000.0, mass_density=-10.0)


class TestOgdenMaterial:
    """Validates Ogden continuum mechanics, spectral formulation, and tangent consistency."""

    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("vol_model", ["log", "quadratic"])
    def test_zero_stress_at_reference_state(self, order: int, vol_model: str) -> None:
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1100e-6)
        mat = Ogden(iso, order=order, volumetric_model=vol_model)  # type: ignore[arg-type]

        F_identity = np.eye(3, dtype=np.float64)
        C, S = mat.compute_stress_and_tangent(F_identity)

        assert np.allclose(S, np.zeros((3, 3)), atol=1e-14)
        assert np.isclose(mat.compute_strain_energy(F_identity), 0.0, atol=1e-14)

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_tangent_at_identity_matches_linear_hookean(self, order: int) -> None:
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1100e-6)
        mat = Ogden(iso, order=order)

        F_identity = np.eye(3, dtype=np.float64)
        C_ogden, _ = mat.compute_stress_and_tangent(F_identity)

        hookean = Hookean(iso)
        C_hookean = hookean.compute_constitutive_matrix()

        assert np.allclose(C_ogden, C_hookean, rtol=1e-6)
        assert np.allclose(C_ogden, mat.compute_constitutive_matrix(), rtol=1e-6)

    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("vol_model", ["log", "quadratic"])
    def test_tangent_consistency_finite_difference(self, order: int, vol_model: str) -> None:
        """
        Verifies that analytical tangent elasticity matrix C matches numerical
        differentiation of 2nd PK stress S with respect to Green-Lagrange strain E.
        """
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1100e-6)
        mat = Ogden(iso, order=order, volumetric_model=vol_model)  # type: ignore[arg-type]

        # Non-trivial 3D deformation gradient (combined shear + stretch)
        F0 = np.array(
            [
                [1.08, 0.03, 0.01],
                [0.02, 0.96, 0.04],
                [0.01, 0.02, 1.04],
            ],
            dtype=np.float64,
        )

        C_analytical, S0 = mat.compute_stress_and_tangent(F0)
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

            _, S_pert = mat.compute_stress_and_tangent(F_pert)
            S_pert_voigt = tensor_to_voigt_stress(S_pert)

            C_numerical[:, j] = (S_pert_voigt - S0_voigt) / eps

        rel_diff = np.linalg.norm(C_analytical - C_numerical) / np.linalg.norm(C_analytical)
        assert rel_diff < 1e-4

        # Tangent symmetry check
        assert np.allclose(C_analytical, C_analytical.T, atol=1e-8)

    def test_tangent_coalescent_equibiaxial(self) -> None:
        """Verifies perturbation limit when two eigenvalues coalesce (equibiaxial stretch)."""
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1100e-6)
        mat = Ogden(iso, order=3)

        F0 = np.diag([1.2, 1.2, 0.8])
        C_analytical, S0 = mat.compute_stress_and_tangent(F0)
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

            w0, V0 = np.linalg.eigh(F0.T @ F0)
            U0 = V0 @ np.diag(np.sqrt(w0)) @ V0.T
            R = F0 @ np.linalg.inv(U0)
            F_pert = R @ U_pert

            _, S_pert = mat.compute_stress_and_tangent(F_pert)
            S_pert_voigt = tensor_to_voigt_stress(S_pert)

            C_numerical[:, j] = (S_pert_voigt - S0_voigt) / eps

        rel_diff = np.linalg.norm(C_analytical - C_numerical) / np.linalg.norm(C_analytical)
        assert rel_diff < 1e-4
        assert np.allclose(C_analytical, C_analytical.T, atol=1e-8)

    def test_tangent_coalescent_hydrostatic(self) -> None:
        """Verifies perturbation limit when all three eigenvalues coalesce (pure dilation)."""
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1100e-6)
        mat = Ogden(iso, order=3)

        F0 = 1.15 * np.eye(3)
        C_analytical, S0 = mat.compute_stress_and_tangent(F0)
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

            w0, V0 = np.linalg.eigh(F0.T @ F0)
            U0 = V0 @ np.diag(np.sqrt(w0)) @ V0.T
            R = F0 @ np.linalg.inv(U0)
            F_pert = R @ U_pert

            _, S_pert = mat.compute_stress_and_tangent(F_pert)
            S_pert_voigt = tensor_to_voigt_stress(S_pert)

            C_numerical[:, j] = (S_pert_voigt - S0_voigt) / eps

        rel_diff = np.linalg.norm(C_analytical - C_numerical) / np.linalg.norm(C_analytical)
        assert rel_diff < 1e-4
        assert np.allclose(C_analytical, C_analytical.T, atol=1e-8)

    def test_element_inversion_raises(self) -> None:
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
        mat = Ogden(iso)

        F_inverted = np.diag([1.0, 1.0, -0.5])
        with pytest.raises(ValueError, match="Element inversion detected"):
            mat.compute_stress_and_tangent(F_inverted)

        with pytest.raises(ValueError, match="Element inversion detected"):
            mat.compute_strain_energy(F_inverted)


class TestOgdenElementIntegration:
    """Verifies seamless integration of Ogden material across all continuum element families."""

    @pytest.fixture
    def tet4_nodes(self) -> np.ndarray:
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
    def tet10_nodes(self) -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ],
            dtype=np.float64,
        )

    def test_tet4_total_lagrangian_integration(self, tet4_nodes: np.ndarray) -> None:
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
        mat = Ogden(iso, order=1)
        elem = Tet4TotalLagrangian(mat)

        u_def = np.zeros((4, 3))
        u_def[1, 0] = 0.05
        u_def[2, 1] = -0.02
        u_def[3, 2] = 0.03

        K, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(tet4_nodes, u_def)

        assert K.shape == (12, 12)
        assert f_int.shape == (12,)
        assert np.allclose(K, K.T, atol=1e-8)
        assert not np.isnan(K).any()
        assert not np.isnan(f_int).any()

    def test_tet4_updated_lagrangian_integration(self, tet4_nodes: np.ndarray) -> None:
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
        mat = Ogden(iso, order=2)
        elem = Tet4UpdatedLagrangian(mat)

        u_def = np.zeros((4, 3))
        u_def[1, 0] = 0.05

        K, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(tet4_nodes, u_def)

        assert K.shape == (12, 12)
        assert f_int.shape == (12,)
        assert np.allclose(K, K.T, atol=1e-8)

    def test_tet4_fbar_locking_free_integration(self, tet4_nodes: np.ndarray) -> None:
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(21000.0, 0.499, 1100e-6)  # Nearly incompressible
        mat = Ogden(iso, order=3)
        elem = Tet4Fbar(mat)

        u_def = np.zeros((4, 3))
        u_def[1, 0] = 0.05

        K, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(tet4_nodes, u_def)

        assert K.shape == (12, 12)
        assert f_int.shape == (12,)
        assert np.allclose(K, K.T, atol=1e-8)

    def test_tet10_total_lagrangian_integration(self, tet10_nodes: np.ndarray) -> None:
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
        mat = Ogden(iso, order=1)
        elem = Tet10TotalLagrangian(mat)

        u_def = np.zeros((10, 3))
        u_def[1, 0] = 0.05

        K, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(tet10_nodes, u_def)

        assert K.shape == (30, 30)
        assert f_int.shape == (30,)
        assert np.allclose(K, K.T, atol=1e-8)

    def test_tet4_corotational_integration(self, tet4_nodes: np.ndarray) -> None:
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
        mat = Ogden(iso, order=2)
        elem = Tet4Corotational(mat)

        u_def = np.zeros((4, 3))
        u_def[1, 0] = 0.05

        K, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(tet4_nodes, u_def)

        assert K.shape == (12, 12)
        assert f_int.shape == (12,)
        assert np.allclose(K, K.T, atol=1e-8)

    def test_viscoelastic_ogden_wrapper(self) -> None:
        """Verifies Ogden hyperelastic base material wrapped by viscoelastic Prony series."""
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
        ogden = Ogden(iso, order=3)

        prony_terms = [PronyTerm(relative_modulus=0.5, relaxation_time=0.1)]
        visco = Viscoelastic(base_material=ogden, prony_series=prony_terms)
        visco.set_time_step(0.01)

        F = np.array(
            [
                [1.05, 0.02, 0.0],
                [0.0, 0.98, 0.01],
                [0.01, 0.0, 1.02],
            ],
            dtype=np.float64,
        )

        C, S = visco.compute_stress_and_tangent(F)
        assert C.shape == (6, 6)
        assert S.shape == (3, 3)
        assert np.allclose(C, C.T, atol=1e-8)

