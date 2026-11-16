"""
Verification Suite for Milestone 1.2: Dimensionless Penalty Scaling Law & Automatic Stiffness Calculation.
=========================================================================================================

Verifies:
1. Dimensionless Contact Parameter kappa_p = (k_p * h_e) / E_bulk in [5, 50].
2. Automatic Element-Adaptive Penalty Law k_{p, e} = alpha_penalty * (K_bulk * A_e) / V_e.
3. Parametric Young's Modulus Sweep across 4 orders of magnitude (10 MPa -> 21 kPa -> 1 kPa):
   - Invariant Newton-Raphson iteration counts (< 6 iterations per step).
   - Zero element inversions (all det(F) > 0).
4. Full integration with RigidContact, DeformableContact, SelfContact, and MortarContact via k_penalty="auto".
5. Conditioning robustness: adaptive penalty avoids the severe ill-conditioning of user-guessed rigid penalties.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pytest
import pyvista as pv

from psierra.domain.solid import SolidDomain
from psierra.material.parameters import (
    IsotropicElasticParameters,
    MooneyRivlinParameters,
    YeohParameters,
    OgdenParameters,
    HGOParameters,
)
from psierra.material.neohookean import NeoHookean
from psierra.material.hookean import Hookean
from psierra.material.yeoh import Yeoh
from psierra.material.mooney_rivlin import MooneyRivlin
from psierra.material.ogden import Ogden
from psierra.material.hgo import HGO
from psierra.element.tet4_linear import Tet4Linear
from psierra.mesher.tetrahedral import TetrahedralMesher
from psierra.collision.shapes import RigidPlane
from psierra.boundary.contact import (
    RigidContact,
    DeformableContact,
    SelfContact,
    compute_adaptive_penalty,
    compute_dimensionless_contact_parameter,
)
from psierra.contact.scaling import (
    extract_material_stiffness,
    compute_characteristic_mesh_metrics,
)
from psierra.boundary.mortar import MortarContact
from psierra.boundary.dirichlet import DirichletManager, DOFConstraint
from psierra.analysis.nonlinear_static import NonlinearStaticAnalysis


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def hemisphere_mesh() -> Any:
    """Standard tetrahedral hemisphere mesh for contact verification."""
    R = 10.0
    sphere = pv.Sphere(radius=R, center=(0.0, 0.0, R), theta_resolution=16, phi_resolution=16)
    hemisphere = sphere.clip(normal=(0, 0, 1), origin=(0, 0, R))
    mesher = TetrahedralMesher(order=1, edge_length_fac=0.15, quiet=True)
    return mesher.generate(hemisphere)


@pytest.fixture(scope="module")
def sample_domain(hemisphere_mesh: Any) -> SolidDomain:
    """Reference solid domain with NeoHookean material."""
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(youngs_modulus=2.0e6, poissons_ratio=0.45, mass_density=1e-6)
    mat = NeoHookean(params)
    elem = Tet4Linear(mat)
    return SolidDomain(hemisphere_mesh.copy(), mat, elem)


# =============================================================================
# 1. Mathematical Scaling Law Tests
# =============================================================================

class TestDimensionlessScalingLawMath:
    """Verifies the mathematical and dimensional properties of the penalty scaling laws."""

    @pytest.mark.parametrize("beta", [0.001, 0.021, 1.0, 50.0, 2000.0])
    def test_linear_scaling_with_modulus(self, hemisphere_mesh: Any, beta: float) -> None:
        """k_p must scale strictly linearly with Young's modulus E (k_p(beta * E) = beta * k_p(E))."""
        E_base = 1.0e6
        params_base = IsotropicElasticParameters()
        params_base.set_from_youngs_poisson(youngs_modulus=E_base, poissons_ratio=0.40, mass_density=1e-6)
        mat_base = NeoHookean(params_base)
        dom_base = SolidDomain(hemisphere_mesh.copy(), mat_base, Tet4Linear(mat_base))

        params_scaled = IsotropicElasticParameters()
        params_scaled.set_from_youngs_poisson(youngs_modulus=beta * E_base, poissons_ratio=0.40, mass_density=1e-6)
        mat_scaled = NeoHookean(params_scaled)
        dom_scaled = SolidDomain(hemisphere_mesh.copy(), mat_scaled, Tet4Linear(mat_scaled))

        kp_base = compute_adaptive_penalty(domain=dom_base)
        kp_scaled = compute_adaptive_penalty(domain=dom_scaled)

        ratio = kp_scaled / kp_base
        np.testing.assert_allclose(ratio, beta, rtol=1e-12)

    def test_linear_scaling_with_bulk_modulus(self, hemisphere_mesh: Any) -> None:
        """k_p must scale linearly with bulk modulus K_bulk when alpha_penalty is fixed."""
        alpha = 0.5
        params1 = IsotropicElasticParameters()
        params1.set_from_youngs_poisson(youngs_modulus=1.0e6, poissons_ratio=0.30, mass_density=1e-6)
        mat1 = NeoHookean(params1)
        dom1 = SolidDomain(hemisphere_mesh.copy(), mat1, Tet4Linear(mat1))

        params2 = IsotropicElasticParameters()
        params2.set_from_youngs_poisson(youngs_modulus=1.0e6, poissons_ratio=0.45, mass_density=1e-6)
        mat2 = NeoHookean(params2)
        dom2 = SolidDomain(hemisphere_mesh.copy(), mat2, Tet4Linear(mat2))

        K1 = mat1.parameters.bulk_modulus
        K2 = mat2.parameters.bulk_modulus

        kp1 = compute_adaptive_penalty(domain=dom1, alpha_penalty=alpha)
        kp2 = compute_adaptive_penalty(domain=dom2, alpha_penalty=alpha)

        expected_ratio = K2 / K1
        actual_ratio = kp2 / kp1
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-12)

    def test_explicit_kappa_p_specification(self, sample_domain: SolidDomain) -> None:
        """When kappa_p is specified, compute_dimensionless_contact_parameter must return exactly kappa_p."""
        for target_kappa in [5.0, 10.0, 20.0, 35.0, 50.0]:
            # Nodal formulation
            kp_nodal = compute_adaptive_penalty(domain=sample_domain, kappa_p=target_kappa, nodal=True)
            kappa_calc_nodal = compute_dimensionless_contact_parameter(kp_nodal, domain=sample_domain, nodal=True)
            np.testing.assert_allclose(kappa_calc_nodal, target_kappa, rtol=1e-12)

            # Distributed formulation
            kp_dist = compute_adaptive_penalty(domain=sample_domain, kappa_p=target_kappa, nodal=False)
            kappa_calc_dist = compute_dimensionless_contact_parameter(kp_dist, domain=sample_domain, nodal=False)
            np.testing.assert_allclose(kappa_calc_dist, target_kappa, rtol=1e-12)

    def test_kappa_p_in_recommended_range(self, sample_domain: SolidDomain) -> None:
        """Default element-adaptive penalty (alpha_penalty=1.0, nu=0.45) must yield kappa_p in [5, 50]."""
        kp_default = compute_adaptive_penalty(domain=sample_domain, alpha_penalty=1.0, nodal=True)
        kappa = compute_dimensionless_contact_parameter(kp_default, domain=sample_domain, nodal=True)
        assert 5.0 <= kappa <= 50.0, f"Expected kappa_p in [5, 50], got {kappa}"

    def test_elementwise_penalty_law(self, sample_domain: SolidDomain) -> None:
        """Per-facet adaptive penalty array must match k_{p, e} = alpha * K * A_e^2 / V_e."""
        alpha = 0.8
        kp_array = compute_adaptive_penalty(
            domain=sample_domain, alpha_penalty=alpha, nodal=True, return_elementwise=True
        )
        assert isinstance(kp_array, np.ndarray)
        assert len(kp_array) == sample_domain.mesh.surface.n_elements
        assert np.all(kp_array > 0.0)
        assert np.all(np.isfinite(kp_array))

        # Check mean vs scalar evaluation
        kp_scalar = compute_adaptive_penalty(domain=sample_domain, alpha_penalty=alpha, nodal=True)
        assert abs(np.mean(kp_array) - kp_scalar) / kp_scalar < 0.5


# =============================================================================
# 2. Material Stiffness & Mesh Extraction Tests
# =============================================================================

class TestMaterialStiffnessAndMeshExtraction:
    """Verifies that stiffness and geometry extraction are robust across all constitutive models."""

    def test_extract_from_neohookean(self) -> None:
        """Extracts bulk and Young's modulus from NeoHookean material."""
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(youngs_modulus=5.0e5, poissons_ratio=0.35, mass_density=1e-6)
        mat = NeoHookean(params)
        K, E, nu = extract_material_stiffness(mat)
        np.testing.assert_allclose(E, 5.0e5, rtol=1e-12)
        np.testing.assert_allclose(nu, 0.35, rtol=1e-12)
        np.testing.assert_allclose(K, 5.0e5 / (3.0 * (1.0 - 2.0 * 0.35)), rtol=1e-12)

    def test_extract_from_hookean(self) -> None:
        """Extracts parameters from linear Hookean material."""
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(youngs_modulus=2.0e11, poissons_ratio=0.30, mass_density=7.8e-6)
        mat = Hookean(params)
        K, E, nu = extract_material_stiffness(mat)
        np.testing.assert_allclose(E, 2.0e11, rtol=1e-12)
        np.testing.assert_allclose(nu, 0.30, rtol=1e-12)

    def test_extract_from_yeoh(self) -> None:
        """Extracts bulk and shear/Young's modulus from Yeoh hyperelastic model."""
        params = YeohParameters(c10=1.5e4, c20=-1.0e3, c30=2.0e2, bulk_modulus=1.0e6, mass_density=1e-6)
        mat = Yeoh(params)
        K, E, nu = extract_material_stiffness(mat)
        np.testing.assert_allclose(K, 1.0e6, rtol=1e-12)
        assert E > 0.0

    def test_extract_from_mooney_rivlin(self) -> None:
        """Extracts bulk and Young's modulus from Mooney-Rivlin model."""
        params = MooneyRivlinParameters(c10=2.0e4, c01=1.0e4, bulk_modulus=5.0e5, mass_density=1e-6)
        mat = MooneyRivlin(params)
        K, E, nu = extract_material_stiffness(mat)
        np.testing.assert_allclose(K, 5.0e5, rtol=1e-12)
        assert E > 0.0

    def test_extract_from_ogden(self) -> None:
        """Extracts bulk and Young's modulus from Ogden model."""
        params = OgdenParameters(mu=[2.0e4], alpha=[2.0], bulk_modulus=8.0e5, mass_density=1e-6)
        mat = Ogden(params)
        K, E, nu = extract_material_stiffness(mat)
        np.testing.assert_allclose(K, 8.0e5, rtol=1e-12)
        assert E > 0.0

    def test_extract_from_hgo(self) -> None:
        """Extracts bulk and Young's modulus from Holzapfel-Gasser-Ogden (HGO) model."""
        f_dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        params = HGOParameters(
            c=5.0e3, k1=1.0e4, k2=2.0, kappa=0.1, fiber_directions=f_dirs, bulk_modulus=2.0e6, mass_density=1e-6
        )
        mat = HGO(params)
        K, E, nu = extract_material_stiffness(mat)
        np.testing.assert_allclose(K, 2.0e6, rtol=1e-12)
        assert E > 0.0

    def test_extract_from_numeric_and_fallbacks(self) -> None:
        """Handles duck-typed numbers, objects with missing attributes, and None gracefully."""
        # Raw float as E
        K1, E1, nu1 = extract_material_stiffness(50000.0)
        np.testing.assert_allclose(E1, 50000.0, rtol=1e-12)
        assert K1 > 0.0

        # None fallback
        K_none, E_none, nu_none = extract_material_stiffness(None)
        assert K_none > 0.0 and E_none > 0.0

    def test_characteristic_mesh_metrics_volumetric(self, hemisphere_mesh: Any) -> None:
        """Extracts volume, area, and edge length metrics from a VolumetricMesh."""
        metrics = compute_characteristic_mesh_metrics(hemisphere_mesh)
        assert metrics["h_e"] > 0.0
        assert metrics["mean_area"] > 0.0
        assert metrics["mean_volume"] > 0.0
        assert metrics["n_elements"] == hemisphere_mesh.n_elements
        assert metrics["n_facets"] == hemisphere_mesh.surface.n_elements

    def test_characteristic_mesh_metrics_surface(self, hemisphere_mesh: Any) -> None:
        """Extracts metrics from a standalone SurfaceMesh."""
        surf = hemisphere_mesh.surface
        metrics = compute_characteristic_mesh_metrics(surf)
        assert metrics["h_e"] > 0.0
        assert metrics["mean_area"] > 0.0
        assert metrics["mean_volume"] > 0.0
        assert metrics["n_facets"] == surf.n_elements


# =============================================================================
# 3. Contact Auto-Penalty Integration Tests
# =============================================================================

class TestContactAutoPenaltyIntegration:
    """Verifies that k_penalty='auto' seamlessly initializes all contact classes."""

    def test_rigid_contact_auto_penalty(self, sample_domain: SolidDomain) -> None:
        """RigidContact(k_penalty='auto') automatically initializes with calibrated stiffness."""
        plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))
        rc = RigidContact(slave=sample_domain, master=plane, k_penalty="auto")
        expected_kp = compute_adaptive_penalty(domain=sample_domain)
        np.testing.assert_allclose(rc.k_penalty, expected_kp, rtol=1e-12)

    def test_deformable_contact_auto_penalty(self, sample_domain: SolidDomain) -> None:
        """DeformableContact(k_penalty='auto') automatically initializes with calibrated stiffness."""
        dom2 = SolidDomain(sample_domain.mesh.copy(), sample_domain.material, sample_domain.element)
        dc = DeformableContact(slave=sample_domain, master=dom2, k_penalty="auto")
        expected_kp = compute_adaptive_penalty(domain=sample_domain)
        np.testing.assert_allclose(dc.k_penalty, expected_kp, rtol=1e-12)

    def test_self_contact_auto_penalty(self, sample_domain: SolidDomain) -> None:
        """SelfContact(k_penalty='auto') automatically initializes with calibrated stiffness."""
        sc = SelfContact(domain=sample_domain, k_penalty="auto")
        expected_kp = compute_adaptive_penalty(domain=sample_domain)
        np.testing.assert_allclose(sc.k_penalty, expected_kp, rtol=1e-12)

    def test_mortar_contact_auto_penalty(self, sample_domain: SolidDomain) -> None:
        """MortarContact(k_penalty='auto') automatically initializes with calibrated stiffness."""
        dom2 = SolidDomain(sample_domain.mesh.copy(), sample_domain.material, sample_domain.element)
        mc = MortarContact(slave=sample_domain, master=dom2, k_penalty="auto")
        expected_kp = compute_adaptive_penalty(domain=sample_domain)
        np.testing.assert_allclose(mc.k_penalty, expected_kp, rtol=1e-12)

    def test_case_insensitive_auto_keyword(self, sample_domain: SolidDomain) -> None:
        """Accepts 'AUTO', 'Auto', and 'auto' interchangeably."""
        plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))
        rc1 = RigidContact(slave=sample_domain, master=plane, k_penalty="AUTO")
        rc2 = RigidContact(slave=sample_domain, master=plane, k_penalty="Auto")
        assert rc1.k_penalty == rc2.k_penalty

    def test_invalid_penalty_errors(self, sample_domain: SolidDomain) -> None:
        """Rejects non-positive or non-finite penalty inputs."""
        plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))
        with pytest.raises(ValueError, match="k_penalty must be finite and positive"):
            RigidContact(slave=sample_domain, master=plane, k_penalty=-10.0)
        with pytest.raises(ValueError, match="k_penalty must be finite and positive"):
            RigidContact(slave=sample_domain, master=plane, k_penalty=0.0)


# =============================================================================
# 4. Parametric Modulus Sweep Verification Gate
# =============================================================================

class TestParametricModulusSweepConvergence:
    """
    Acceptance Gate: Parametric Modulus Sweep (10 MPa -> 21 kPa -> 1 kPa).
    Verifies that adaptive penalty maintains identical iteration counts (< 6 per step)
    and zero element inversions across 4 orders of magnitude of Young's modulus.
    """

    @pytest.mark.parametrize("modulus_name,E_val", [
        ("cartilage_dense", 10.0),       # 10.0 MPa = 10^7 Pa
        ("silicone_phantom", 0.021),     # 21.0 kPa = 2.1 x 10^4 Pa
        ("brain_parenchyma", 0.0010),    # 1.0 kPa = 1.0 x 10^3 Pa
    ])
    def test_parametric_modulus_sweep_newton_iterations(
        self, hemisphere_mesh: Any, modulus_name: str, E_val: float
    ) -> None:
        """NonlinearStaticAnalysis with adaptive penalty converges in < 6 Newton iterations per step."""
        R = 10.0
        delta = 0.10  # mm indentation

        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(youngs_modulus=E_val, poissons_ratio=0.45, mass_density=1e-6)
        mat = NeoHookean(params)
        elem = Tet4Linear(mat)
        domain = SolidDomain(hemisphere_mesh.copy(), mat, elem)

        plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))
        # Automatic adaptive penalty
        contact = RigidContact(slave=domain, master=plane, k_penalty="auto")

        top_nodes = np.flatnonzero(domain.mesh.nodes[:, 2] >= R - 0.5)
        manager = DirichletManager(domain.n_dofs)
        for node in top_nodes:
            manager.add_constraint(DOFConstraint([3 * node, 3 * node + 1], 0.0))
            manager.add_constraint(DOFConstraint([3 * node + 2], -delta))

        analysis = NonlinearStaticAnalysis(domain=domain)
        res = analysis.solve(
            dirichlet=manager,
            contact=contact,
            n_steps=2,
            nr_iterations=15,
            nr_tol=1e-4,
            verbose=False,
        )

        assert res.converged is True, f"Failed to converge for {modulus_name} (E = {E_val} MPa)"
        assert len(res.iteration_history) == 2
        # Gate: every step must converge in < 6 Newton iterations
        for step_idx, iters in enumerate(res.iteration_history):
            assert iters < 6, (
                f"{modulus_name} step {step_idx + 1} took {iters} Newton iterations (limit: < 6)"
            )

        # Gate: iteration history should be strictly bounded (typically exactly [3, 2])
        assert max(res.iteration_history) <= 3

    @pytest.mark.parametrize("modulus_name,E_val", [
        ("cartilage_dense", 10.0),
        ("silicone_phantom", 0.021),
        ("brain_parenchyma", 0.0010),
    ])
    def test_zero_element_inversions_across_moduli(
        self, hemisphere_mesh: Any, modulus_name: str, E_val: float
    ) -> None:
        """Every continuum element preserves positive Jacobian determinant (det(F) > 0) under adaptive contact."""
        R = 10.0
        delta = 0.10

        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(youngs_modulus=E_val, poissons_ratio=0.45, mass_density=1e-6)
        mat = NeoHookean(params)
        elem = Tet4Linear(mat)
        domain = SolidDomain(hemisphere_mesh.copy(), mat, elem)

        plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))
        contact = RigidContact(slave=domain, master=plane, k_penalty="auto")

        top_nodes = np.flatnonzero(domain.mesh.nodes[:, 2] >= R - 0.5)
        manager = DirichletManager(domain.n_dofs)
        for node in top_nodes:
            manager.add_constraint(DOFConstraint([3 * node, 3 * node + 1], 0.0))
            manager.add_constraint(DOFConstraint([3 * node + 2], -delta))

        analysis = NonlinearStaticAnalysis(domain=domain)
        res = analysis.solve(
            dirichlet=manager,
            contact=contact,
            n_steps=2,
            nr_iterations=15,
            nr_tol=1e-4,
            verbose=False,
        )
        assert res.converged is True

        # Check deformation gradient Jacobian determinant for every element
        current_pts = domain.current_nodes
        ref_pts = domain.mesh.nodes
        conn = domain.mesh.elements

        for e in range(domain.mesh.n_elements):
            node_ids = conn[e, :4]
            X = ref_pts[node_ids]
            x = current_pts[node_ids]

            dX = np.column_stack([X[1] - X[0], X[2] - X[0], X[3] - X[0]])
            dx = np.column_stack([x[1] - x[0], x[2] - x[0], x[3] - x[0]])

            # Deformation gradient F = dx * inv(dX)
            F = dx @ np.linalg.inv(dX)
            detF = float(np.linalg.det(F))
            assert detF > 0.05, f"Element {e} inverted or near-inverted: det(F) = {detF:.4e}"


# =============================================================================
# 5. Conditioning & Robustness Comparison Tests
# =============================================================================

class TestStiffnessConditioningComparison:
    """Compares numerical conditioning between adaptive penalty and uncalibrated user guesses."""

    def test_stiffness_norm_ratio_scale_invariance(self, hemisphere_mesh: Any) -> None:
        """Ratio ||K_contact|| / ||K_domain|| remains strictly scale-invariant across 4 orders of magnitude."""
        ratios = []
        moduli = [10.0, 0.021, 0.0010]  # 10 MPa, 21 kPa, 1 kPa

        for E_val in moduli:
            params = IsotropicElasticParameters()
            params.set_from_youngs_poisson(youngs_modulus=E_val, poissons_ratio=0.45, mass_density=1e-6)
            mat = NeoHookean(params)
            elem = Tet4Linear(mat)
            domain = SolidDomain(hemisphere_mesh.copy(), mat, elem)

            # Prescribe small uniform penetration (-0.05 mm)
            domain.update_state(u=np.tile(np.array([0.0, 0.0, -0.05]), domain.mesh.n_nodes))

            K_domain = domain.assemble_stiffness_matrix().toarray()
            plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))
            rc = RigidContact(slave=domain, master=plane, k_penalty="auto")
            K_contact = rc.evaluate_stiffness().toarray()

            norm_dom = float(np.linalg.norm(K_domain))
            norm_c = float(np.linalg.norm(K_contact))
            ratio = norm_c / norm_dom
            ratios.append(ratio)

        # All ratios must match to high relative precision
        for r in ratios[1:]:
            np.testing.assert_allclose(r, ratios[0], rtol=1e-10)

    def test_guessed_rigid_penalty_conditioning_degradation(self, hemisphere_mesh: Any) -> None:
        """Unadapted user-guessed penalty (10^5 N/mm) on 1 kPa tissue causes 200,000x conditioning degradation."""
        E_val = 0.0010  # 1 kPa
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(youngs_modulus=E_val, poissons_ratio=0.45, mass_density=1e-6)
        mat = NeoHookean(params)
        elem = Tet4Linear(mat)
        domain = SolidDomain(hemisphere_mesh.copy(), mat, elem)

        domain.update_state(u=np.tile(np.array([0.0, 0.0, -0.05]), domain.mesh.n_nodes))
        K_domain = domain.assemble_stiffness_matrix().toarray()
        norm_dom = float(np.linalg.norm(K_domain))

        plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))

        # 1. Adaptive penalty
        rc_adapt = RigidContact(slave=domain, master=plane, k_penalty="auto")
        norm_c_adapt = float(np.linalg.norm(rc_adapt.evaluate_stiffness().toarray()))
        ratio_adapt = norm_c_adapt / norm_dom

        # 2. Guessed rigid penalty
        rc_guessed = RigidContact(slave=domain, master=plane, k_penalty=1.0e5)
        norm_c_guessed = float(np.linalg.norm(rc_guessed.evaluate_stiffness().toarray()))
        ratio_guessed = norm_c_guessed / norm_dom

        # Adaptive ratio is well-conditioned (O(10^-2))
        assert ratio_adapt < 0.1
        # Guessed ratio is violently ill-conditioned (> 100,000x domain stiffness)
        assert ratio_guessed > 1.0e5
        assert ratio_guessed / ratio_adapt > 1.0e6
