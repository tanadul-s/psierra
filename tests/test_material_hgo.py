from __future__ import annotations
import numpy as np
import pytest

from psierra.material.parameters import (
    IsotropicElasticParameters,
    HGOParameters,
)
from psierra.material.hookean import Hookean
from psierra.material.hgo import HGO
from psierra.material.viscoelastic import Viscoelastic, PronyTerm
from psierra.utility.tensor import tensor_to_voigt_stress, VOIGT_MAP_3D

from psierra.element.tet4_total_lagrangian import Tet4TotalLagrangian
from psierra.element.tet4_updated_lagrangian import Tet4UpdatedLagrangian
from psierra.element.tet4_fbar import Tet4Fbar
from psierra.element.tet10_total_lagrangian import Tet10TotalLagrangian
from psierra.element.tet4_corotational import Tet4Corotational


class TestHGOParameters:
    """Validates HGO parameter representations, derivations, presets, and safety bounds."""

    def test_constants_initialization(self) -> None:
        params = HGOParameters.from_constants(
            c=15.0e3,
            k1=300.0e3,
            k2=25.0,
            kappa=0.15,
            fiber_directions=[[1.0, 1.0, 0.0]],  # should be auto-normalized
            bulk_modulus=100.0e3,
            mass_density=1050.0,
        )
        assert np.isclose(params.c, 15.0e3)
        assert np.isclose(params.k1, 300.0e3)
        assert np.isclose(params.k2, 25.0)
        assert np.isclose(params.kappa, 0.15)
        assert params.n_fiber_families == 1
        expected_dir = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        assert np.allclose(params.fiber_directions[0], expected_dir)
        assert np.isclose(params.shear_modulus, 15.0e3)
        assert np.isclose(params.bulk_modulus, 100.0e3)
        assert params.youngs_modulus > 0.0
        assert 0.0 < params.poissons_ratio < 0.5

    def test_from_isotropic(self) -> None:
        iso = IsotropicElasticParameters()
        iso.set_from_youngs_poisson(youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1100e-6)

        hgo_params = HGOParameters.from_isotropic(
            iso,
            k1_ratio=3.0,
            k2=15.0,
            kappa=0.1,
            fiber_directions=[[1.0, 0.0, 0.0]],
        )
        assert np.isclose(hgo_params.c, iso.shear_modulus)
        assert np.isclose(hgo_params.k1, 3.0 * iso.shear_modulus)
        assert np.isclose(hgo_params.bulk_modulus, iso.bulk_modulus)
        assert np.isclose(hgo_params.mass_density, iso.mass_density)

    def test_aorta_presets(self) -> None:
        # Test all 3 anatomical layers: media, adventitia, intima
        for layer in ["media", "adventitia", "intima"]:
            params = HGOParameters.from_aorta(
                bulk_modulus=500.0e3,
                mass_density=1060.0,
                layer=layer,
            )
            assert params.n_fiber_families == 2
            assert params.c > 0.0
            assert params.k1 > 0.0
            assert params.k2 > 0.0
            assert 0.0 <= params.kappa <= 1.0 / 3.0
            # Fibers must be symmetric about X axis
            v1, v2 = params.fiber_directions
            assert np.isclose(v1[0], v2[0])
            assert np.isclose(v1[1], -v2[1])

    def test_falx_cerebri_preset(self) -> None:
        params = HGOParameters.from_falx_cerebri(bulk_modulus=10.0e6, mass_density=1040.0)
        assert params.n_fiber_families == 1
        assert np.allclose(params.fiber_directions[0], [0.0, 1.0, 0.0])
        assert params.c > 0.0
        assert params.k1 > 0.0

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="matrix modulus c must be positive"):
            HGOParameters.from_constants(-10.0, 100.0, 10.0, 0.1, [[1, 0, 0]], 5000.0, 1000.0)

        with pytest.raises(ValueError, match="k1 must be non-negative"):
            HGOParameters.from_constants(10.0, -100.0, 10.0, 0.1, [[1, 0, 0]], 5000.0, 1000.0)

        with pytest.raises(ValueError, match="k2 must be positive"):
            HGOParameters.from_constants(10.0, 100.0, -5.0, 0.1, [[1, 0, 0]], 5000.0, 1000.0)

        with pytest.raises(ValueError, match="dispersion kappa must satisfy"):
            HGOParameters.from_constants(10.0, 100.0, 10.0, 0.45, [[1, 0, 0]], 5000.0, 1000.0)

        with pytest.raises(ValueError, match="cannot be zero vector"):
            HGOParameters.from_constants(10.0, 100.0, 10.0, 0.1, [[0, 0, 0]], 5000.0, 1000.0)


class TestHGOMaterial:
    """Validates HGO continuum mechanics, tension-only mechanics, and tangent consistency."""

    @pytest.mark.parametrize("vol_model", ["log", "quadratic"])
    def test_zero_stress_at_reference_state(self, vol_model: str) -> None:
        params = HGOParameters.from_aorta(bulk_modulus=500.0e3, mass_density=1060.0)
        mat = HGO(params, volumetric_model=vol_model)  # type: ignore[arg-type]

        F_identity = np.eye(3, dtype=np.float64)
        C, S = mat.compute_stress_and_tangent(F_identity)

        assert np.allclose(S, np.zeros((3, 3)), atol=1e-14)
        assert np.isclose(mat.compute_strain_energy(F_identity), 0.0, atol=1e-14)

    def test_tangent_at_reference_matches_ground_matrix(self) -> None:
        """At reference state (fibers unloaded E_bar=0), tangent must match isotropic matrix."""
        params = HGOParameters.from_aorta(bulk_modulus=500.0e3, mass_density=1060.0)
        mat = HGO(params)

        F_identity = np.eye(3, dtype=np.float64)
        C_hgo, _ = mat.compute_stress_and_tangent(F_identity)

        # Baseline isotropic parameters with G = c
        iso = IsotropicElasticParameters()
        iso.set_from_bulk_shear(params.bulk_modulus, params.c, params.mass_density)
        hookean = Hookean(iso)
        C_hookean = hookean.compute_constitutive_matrix()

        assert np.allclose(C_hgo, C_hookean, rtol=1e-6)
        assert np.allclose(C_hgo, mat.compute_constitutive_matrix(), rtol=1e-6)

    @pytest.mark.parametrize("vol_model", ["log", "quadratic"])
    def test_tangent_consistency_finite_difference(self, vol_model: str) -> None:
        """
        Verifies that analytical 4th-order tangent elasticity matrix C matches numerical
        differentiation of 2nd PK stress S with respect to Green-Lagrange strain E.
        """
        params = HGOParameters.from_aorta(
            bulk_modulus=500.0e3,
            mass_density=1060.0,
            pitch_angle_deg=35.0,
        )
        mat = HGO(params, volumetric_model=vol_model)  # type: ignore[arg-type]

        # Non-trivial 3D finite deformation gradient with active fiber tension
        F0 = np.array(
            [
                [1.15, 0.03, 0.01],
                [0.02, 1.08, 0.02],
                [0.01, 0.01, 0.88],
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

            C_numerical[:, j] = (S_pert_v := S_pert_voigt - S0_voigt) / eps

        rel_diff = np.linalg.norm(C_analytical - C_numerical) / np.linalg.norm(C_analytical)
        assert rel_diff < 1e-4

        # Tangent symmetry check
        assert np.allclose(C_analytical, C_analytical.T, atol=1e-8)

    def test_tension_only_activation(self) -> None:
        """
        Verifies tension-only activation: under compression along the fiber axis,
        collagen fibers buckle with zero load-bearing and stress matches pure matrix.
        """
        params = HGOParameters.from_constants(
            c=10.0e3,
            k1=500.0e3,
            k2=30.0,
            kappa=0.0,  # perfectly aligned along X
            fiber_directions=[[1.0, 0.0, 0.0]],
            bulk_modulus=100.0e3,
            mass_density=1050.0,
        )
        mat_hgo = HGO(params, tension_only=True)

        # Pure compression along X (fiber axis)
        F_comp = np.diag([0.85, 1.05, 1.05])
        _, S_hgo = mat_hgo.compute_stress_and_tangent(F_comp)

        # Baseline isotropic model with c = 10 kPa
        iso = IsotropicElasticParameters()
        iso.set_from_bulk_shear(params.bulk_modulus, params.c, params.mass_density)
        from psierra.material.neohookean import NeoHookean
        mat_neo = NeoHookean(iso)
        _, S_neo = mat_neo.compute_stress_and_tangent(F_comp)

        # S_hgo must match pure isotropic S_neo since fiber is buckled (E_bar <= 0)
        assert np.allclose(S_hgo, S_neo, rtol=1e-5)

    def test_anisotropic_tensile_stiffening(self) -> None:
        """
        Verifies strong anisotropic stiffening: stretch along fiber direction (X)
        produces much higher stress than stretch perpendicular to fiber direction (Y).
        """
        params = HGOParameters.from_constants(
            c=10.0e3,
            k1=200.0e3,
            k2=10.0,
            kappa=0.0,  # perfectly aligned along X
            fiber_directions=[[1.0, 0.0, 0.0]],
            bulk_modulus=100.0e3,
            mass_density=1050.0,
        )
        mat = HGO(params)

        # Stretch along X (fiber axis)
        F_fiber = np.diag([1.25, 1.0, 0.8])
        _, S_fiber = mat.compute_stress_and_tangent(F_fiber)

        # Stretch along Y (perpendicular to fiber axis)
        F_trans = np.diag([1.0, 1.25, 0.8])
        _, S_trans = mat.compute_stress_and_tangent(F_trans)

        # Fiber-aligned stress S_11 must vastly exceed transverse stress S_22
        assert S_fiber[0, 0] > 5.0 * S_trans[1, 1]

    def test_element_inversion_raises(self) -> None:
        params = HGOParameters.from_aorta(500.0e3, 1060.0)
        mat = HGO(params)

        F_inverted = np.diag([1.0, 1.0, -0.5])
        with pytest.raises(ValueError, match="Element inversion detected"):
            mat.compute_stress_and_tangent(F_inverted)

        with pytest.raises(ValueError, match="Element inversion detected"):
            mat.compute_strain_energy(F_inverted)


class TestHGOElementIntegration:
    """Verifies universal integration of HGO material across all continuum element families."""

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
        params = HGOParameters.from_aorta(500.0e3, 1060.0)
        mat = HGO(params)
        elem = Tet4TotalLagrangian(mat)

        u_def = np.zeros((4, 3))
        u_def[1, 0] = 0.05
        u_def[2, 1] = 0.02

        K, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(tet4_nodes, u_def)

        assert K.shape == (12, 12)
        assert f_int.shape == (12,)
        assert np.allclose(K, K.T, atol=1e-8)
        assert not np.isnan(K).any()
        assert not np.isnan(f_int).any()

    def test_tet4_updated_lagrangian_integration(self, tet4_nodes: np.ndarray) -> None:
        params = HGOParameters.from_aorta(500.0e3, 1060.0)
        mat = HGO(params)
        elem = Tet4UpdatedLagrangian(mat)

        u_def = np.zeros((4, 3))
        u_def[1, 0] = 0.05

        K, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(tet4_nodes, u_def)

        assert K.shape == (12, 12)
        assert f_int.shape == (12,)
        assert np.allclose(K, K.T, atol=1e-8)

    def test_tet4_fbar_locking_free_integration(self, tet4_nodes: np.ndarray) -> None:
        params = HGOParameters.from_aorta(500.0e3, 1060.0)
        mat = HGO(params)
        elem = Tet4Fbar(mat)

        u_def = np.zeros((4, 3))
        u_def[1, 0] = 0.05

        K, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(tet4_nodes, u_def)

        assert K.shape == (12, 12)
        assert f_int.shape == (12,)
        assert np.allclose(K, K.T, atol=1e-8)

    def test_tet10_total_lagrangian_integration(self, tet10_nodes: np.ndarray) -> None:
        params = HGOParameters.from_aorta(500.0e3, 1060.0)
        mat = HGO(params)
        elem = Tet10TotalLagrangian(mat)

        u_def = np.zeros((10, 3))
        u_def[1, 0] = 0.05

        K, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(tet10_nodes, u_def)

        assert K.shape == (30, 30)
        assert f_int.shape == (30,)
        assert np.allclose(K, K.T, atol=1e-8)

    def test_tet4_corotational_integration(self, tet4_nodes: np.ndarray) -> None:
        params = HGOParameters.from_aorta(500.0e3, 1060.0)
        mat = HGO(params)
        elem = Tet4Corotational(mat)

        u_def = np.zeros((4, 3))
        u_def[1, 0] = 0.05

        K, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(tet4_nodes, u_def)

        assert K.shape == (12, 12)
        assert f_int.shape == (12,)
        assert np.allclose(K, K.T, atol=1e-8)

    def test_viscoelastic_hgo_wrapper(self) -> None:
        """Verifies HGO anisotropic hyperelastic base material wrapped by viscoelastic Prony series."""
        params = HGOParameters.from_aorta(500.0e3, 1060.0)
        hgo = HGO(params)

        prony_terms = [PronyTerm(relative_modulus=0.4, relaxation_time=0.1)]
        visco = Viscoelastic(base_material=hgo, prony_series=prony_terms)
        visco.set_time_step(0.01)

        F = np.array(
            [
                [1.08, 0.02, 0.0],
                [0.0, 0.98, 0.01],
                [0.01, 0.0, 1.02],
            ],
            dtype=np.float64,
        )

        C, S = visco.compute_stress_and_tangent(F)
        assert C.shape == (6, 6)
        assert S.shape == (3, 3)
        assert np.allclose(C, C.T, atol=1e-8)
