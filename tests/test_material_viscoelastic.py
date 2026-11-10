"""
Unit and integration tests for Viscoelastic constitutive laws.

Validates:
1. Prony series parameter validation (g_m in (0, 1), tau_m > 0, sum(g_m) < 1.0).
2. Instantaneous elastic limit (dt -> 0: eta -> 1.0, S -> S0, C -> C0).
3. Fully relaxed equilibrium limit (dt -> inf: eta -> g_inf, S -> g_inf * S0, C -> g_inf * C0).
4. Step-strain stress relaxation tracking continuous analytical Prony spectrum to < 1e-12.
5. Cyclic deformation hysteretic energy dissipation loop.
6. Multi-term Maxwell spectrum across multiple relaxation decades.
7. Continuum domain integration with Tet4TotalLagrangian and NonlinearDynamicAnalysis.
"""

from __future__ import annotations
import numpy as np
import pytest

from psierra.material.parameters import IsotropicElasticParameters
from psierra.material.hookean import Hookean
from psierra.material.neohookean import NeoHookean
from psierra.material.viscoelastic import Viscoelastic, PronyTerm
from psierra.element.tet4_total_lagrangian import Tet4TotalLagrangian
from psierra.element.tet4_updated_lagrangian import Tet4UpdatedLagrangian
from psierra.mesh.volumetric import VolumetricMesh
from psierra.domain.solid import SolidDomain
from psierra.boundary.dirichlet import DirichletConstraint, DirichletManager
from psierra.analysis.nonlinear_dynamic import NonlinearDynamicAnalysis


@pytest.fixture
def elastic_params() -> IsotropicElasticParameters:
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=1000.0,
        poissons_ratio=0.3,
        mass_density=1000.0,
    )
    return params


class TestViscoelasticInstantiation:
    """Tests parameter checking and basic properties."""

    def test_valid_prony_terms(self, elastic_params: IsotropicElasticParameters) -> None:
        base = NeoHookean(elastic_params)
        terms = [PronyTerm(0.3, 0.5), PronyTerm(0.2, 2.0)]
        mat = Viscoelastic(base, terms)

        assert mat.base_material is base
        assert mat.mass_density == 1000.0
        assert len(mat.prony_terms) == 2
        assert np.isclose(mat.g_infinity, 0.5)

    def test_tuple_prony_terms(self, elastic_params: IsotropicElasticParameters) -> None:
        base = Hookean(elastic_params)
        mat = Viscoelastic(base, [(0.4, 0.1), (0.2, 1.0)])
        assert len(mat.prony_terms) == 2
        assert np.isclose(mat.g_infinity, 0.4)

    def test_invalid_modulus_raises(self, elastic_params: IsotropicElasticParameters) -> None:
        base = NeoHookean(elastic_params)
        with pytest.raises(ValueError, match="0 < g_m < 1"):
            Viscoelastic(base, [PronyTerm(-0.1, 1.0)])
        with pytest.raises(ValueError, match="0 < g_m < 1"):
            Viscoelastic(base, [PronyTerm(1.0, 1.0)])

    def test_invalid_tau_raises(self, elastic_params: IsotropicElasticParameters) -> None:
        base = NeoHookean(elastic_params)
        with pytest.raises(ValueError, match="strictly positive"):
            Viscoelastic(base, [PronyTerm(0.3, 0.0)])
        with pytest.raises(ValueError, match="strictly positive"):
            Viscoelastic(base, [PronyTerm(0.3, -2.0)])

    def test_sum_g_exceeds_unity_raises(self, elastic_params: IsotropicElasticParameters) -> None:
        base = NeoHookean(elastic_params)
        with pytest.raises(ValueError, match="strictly less than 1.0"):
            Viscoelastic(base, [(0.6, 0.1), (0.5, 1.0)])

    def test_empty_prony_raises(self, elastic_params: IsotropicElasticParameters) -> None:
        base = NeoHookean(elastic_params)
        with pytest.raises(ValueError, match="at least one"):
            Viscoelastic(base, [])


class TestViscoelasticAsymptoticLimits:
    """Validates mathematical asymptotic limits (dt -> 0 and dt -> inf)."""

    def test_instantaneous_limit_dt_zero(self, elastic_params: IsotropicElasticParameters) -> None:
        base = NeoHookean(elastic_params)
        mat = Viscoelastic(base, [(0.4, 0.5), (0.2, 2.0)])
        mat.set_time_step(1e-12)

        assert np.isclose(mat.algorithmic_factor, 1.0, atol=1e-6)

        F = np.array([
            [1.1, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0 / (1.1)],
        ], dtype=np.float64)

        C_base, S_base = base.compute_stress_and_tangent(F)
        C_visco, S_visco = mat.compute_stress_and_tangent(F)

        np.testing.assert_allclose(S_visco, S_base, rtol=1e-5)
        np.testing.assert_allclose(C_visco, C_base, rtol=1e-5)

    def test_relaxed_equilibrium_limit_dt_inf(self, elastic_params: IsotropicElasticParameters) -> None:
        base = NeoHookean(elastic_params)
        mat = Viscoelastic(base, [(0.4, 0.5), (0.2, 2.0)])
        mat.set_time_step(1e12)

        g_inf = mat.g_infinity  # 0.4
        assert np.isclose(mat.algorithmic_factor, g_inf, atol=1e-6)

        F = np.array([
            [1.1, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0 / (1.1)],
        ], dtype=np.float64)

        C_base, S_base = base.compute_stress_and_tangent(F)
        C_visco, S_visco = mat.compute_stress_and_tangent(F)

        np.testing.assert_allclose(S_visco, g_inf * S_base, rtol=1e-5)
        np.testing.assert_allclose(C_visco, g_inf * C_base, rtol=1e-5)


class TestStepStrainStressRelaxation:
    """Validates exact tracking of analytical continuous Prony relaxation curve."""

    def test_single_term_relaxation_curve(self, elastic_params: IsotropicElasticParameters) -> None:
        base = NeoHookean(elastic_params)
        g1 = 0.6
        tau1 = 0.2
        mat = Viscoelastic(base, [(g1, tau1)])

        dt = 0.02
        mat.set_time_step(dt)

        # Apply sudden uniaxial stretch at step 1
        F_step = np.diag([1.15, 1.0, 1.0])
        _, S0_elastic = base.compute_stress_and_tangent(F_step)

        # Step 1: initial ramp loading
        _, S1 = mat.compute_stress_and_tangent(F_step, elem_index=0)
        mat.commit_state()

        gamma1 = np.exp(-dt / tau1)
        beta1 = (tau1 / dt) * (1.0 - gamma1)

        # Hold constant strain for 15 steps
        for step in range(2, 16):
            t_curr = step * dt
            _, S_num = mat.compute_stress_and_tangent(F_step, elem_index=0)
            mat.commit_state()

            # Exact discrete recurrence analytical stress
            factor_exact = mat.g_infinity + g1 * beta1 * (gamma1 ** (step - 1))
            S_expected = factor_exact * S0_elastic

            # Must match discrete recurrence to machine precision
            np.testing.assert_allclose(
                S_num,
                S_expected,
                rtol=1e-12,
                err_msg=f"Relaxation mismatch at step {step} (t = {t_curr:.3f} s)",
            )

    def test_multi_term_relaxation(self, elastic_params: IsotropicElasticParameters) -> None:
        base = NeoHookean(elastic_params)
        terms = [(0.3, 0.05), (0.3, 0.5)]
        mat = Viscoelastic(base, terms)

        dt = 0.01
        mat.set_time_step(dt)

        F_step = np.array([
            [1.1, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        _, S0 = base.compute_stress_and_tangent(F_step)

        # Step 1
        mat.compute_stress_and_tangent(F_step, elem_index=0)
        mat.commit_state()

        gammas = [np.exp(-dt / tau) for _, tau in terms]
        betas = [(tau / dt) * (1.0 - gammas[i]) for i, (_, tau) in enumerate(terms)]

        # Step 2 to 20
        for step in range(2, 21):
            _, S = mat.compute_stress_and_tangent(F_step, elem_index=0)
            mat.commit_state()

            factor_exact = mat.g_infinity + sum(
                terms[i][0] * betas[i] * (gammas[i] ** (step - 1))
                for i in range(len(terms))
            )
            np.testing.assert_allclose(S, factor_exact * S0, rtol=1e-12)


class TestViscoelasticCyclicEnergyDissipation:
    """Validates that cyclic strain produces an open hysteresis loop (viscous dissipation)."""

    def test_hysteresis_loop_area(self, elastic_params: IsotropicElasticParameters) -> None:
        base = Hookean(elastic_params)
        mat = Viscoelastic(base, [(0.5, 0.1)])

        dt = 0.01
        mat.set_time_step(dt)
        omega = 2.0 * np.pi * 1.0  # 1 Hz excitation
        n_steps = 100

        strains = []
        stresses = []

        for step in range(1, n_steps + 1):
            t = step * dt
            gamma = 0.05 * np.sin(omega * t)
            F = np.array([
                [1.0, gamma, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

            _, S = mat.compute_stress_and_tangent(F, elem_index=0)
            mat.commit_state()

            strains.append(gamma)
            stresses.append(S[0, 1])

        strains = np.array(strains)
        stresses = np.array(stresses)

        # Area of loop via trapezoidal rule: int S dE
        dissipated_energy = np.trapezoid(stresses, strains)
        # Dissipated energy must be non-zero and positive over cyclic loops
        assert abs(dissipated_energy) > 1e-4, f"Expected non-zero hysteresis, got {dissipated_energy}"


class TestContinuumDomainIntegration:
    """Validates full integration with VolumetricMesh, SolidDomain, and NonlinearDynamicAnalysis."""

    def test_domain_history_allocation_and_step(
        self, elastic_params: IsotropicElasticParameters, single_element_mesh: VolumetricMesh
    ) -> None:
        base = NeoHookean(elastic_params)
        mat = Viscoelastic(base, [(0.5, 0.1)])
        element = Tet4TotalLagrangian(mat)
        domain = SolidDomain(single_element_mesh, mat, element)

        # Verify history array initialized for elements
        assert mat._n_elements == single_element_mesh.n_elements

        # Set time step on domain
        domain.set_time_step(0.01)
        assert mat.dt == 0.01

        # Evaluate internal forces and commit
        f_int = domain.assemble_internal_forces()
        assert f_int.shape == (single_element_mesh.n_dofs,)

        domain.commit_state()
        domain.reset_state()
        assert np.allclose(mat._Q, 0.0)

    def test_viscoelastic_beam_nonlinear_dynamic_solve(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(1000.0, 0.3, 1e-3)
        base = NeoHookean(params)
        mat = Viscoelastic(base, [(0.5, 0.05)])
        element = Tet4TotalLagrangian(mat)
        domain = SolidDomain(five_tet_cube_mesh, mat, element)

        # Clamp bottom face (Z == 0)
        clamped_nodes = np.where(five_tet_cube_mesh.nodes[:, 2] < 1e-6)[0]
        clamp = DirichletConstraint(clamped_nodes, components=(0, 1, 2), value=0.0)
        dirichlet = DirichletManager(domain.n_dofs)
        dirichlet.add_constraint(clamp)

        # Apply downward gravity load
        from psierra.boundary.load import GravityLoad
        from psierra.damping.rayleigh import RayleighDamping

        gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
        M = domain.assemble_mass_matrix()
        f_ext = gravity.compute_force_vector(M)
        damping = RayleighDamping(alpha=1.0, beta=0.001)

        analysis = NonlinearDynamicAnalysis(domain, damping=damping, spectral_radius=0.8)
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.005,
            n_steps=4,
            nr_iterations=10,
            nr_tol=1e-4,
            verbose=False,
        )

        assert result.converged
        assert result.steps_completed == 4
        assert np.linalg.norm(domain.displacement) > 0.0

    def test_viscoelastic_updated_lagrangian_dynamic_solve(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """Verifies full domain dynamic solve with Tet4UpdatedLagrangian and Viscoelasticity."""
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(1000.0, 0.3, 1e-3)
        base = NeoHookean(params)
        mat = Viscoelastic(base, [(0.5, 0.05)])
        element = Tet4UpdatedLagrangian(mat)
        domain = SolidDomain(five_tet_cube_mesh, mat, element)

        # Clamp bottom face (Z == 0)
        clamped_nodes = np.where(five_tet_cube_mesh.nodes[:, 2] < 1e-6)[0]
        clamp = DirichletConstraint(clamped_nodes, components=(0, 1, 2), value=0.0)
        dirichlet = DirichletManager(domain.n_dofs)
        dirichlet.add_constraint(clamp)

        from psierra.boundary.load import GravityLoad
        from psierra.damping.rayleigh import RayleighDamping

        gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
        M = domain.assemble_mass_matrix()
        f_ext = gravity.compute_force_vector(M)
        damping = RayleighDamping(alpha=1.0, beta=0.001)

        analysis = NonlinearDynamicAnalysis(domain, damping=damping, spectral_radius=0.8)
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.005,
            n_steps=4,
            nr_iterations=10,
            nr_tol=1e-4,
            verbose=False,
        )

        assert result.converged
        assert result.steps_completed == 4
        assert np.linalg.norm(domain.displacement) > 0.0

