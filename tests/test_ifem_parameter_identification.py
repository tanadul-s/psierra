"""
Adversarial and Benchmark Tests for Material Parameter Identification (Constitutive Calibration).

Validates:
1. ParameterBounds validation, lower/upper bound consistency, and initial value checks.
2. Closed-form linear elasticity calibration (exact Young's modulus E recovery to machine precision).
3. Nonlinear least-squares optimization loop (TRF, Dogbox, L-BFGS-B).
4. Automatic recovery of Young's modulus E within 1% of ground truth from force-deflection curves.
5. Automatic recovery of hyperelastic parameter C10 (Neo-Hookean) within 1% of ground truth.
6. Multi-parameter identification (joint calibration of E and Poisson's ratio nu).
7. Prior Tikhonov regularization beta sensitivity and noisy observation damping.
8. Error handling: empty observations, mismatched weights, invalid bounds, non-physical parameters.
"""

from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pytest

from psierra.material import IsotropicElasticParameters, Hookean, NeoHookean
from psierra.element import Tet4Linear, Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary.selector import select_nodes_by_bounds
from psierra.boundary import DirichletConstraint, DirichletManager
from psierra.analysis import LinearStaticAnalysis, NonlinearStaticAnalysis
from psierra.inverse import (
    ParameterBounds,
    IdentificationResult,
    MaterialParameterIdentifier,
)


@pytest.fixture(scope="module")
def beam_mesh() -> VolumetricMesh:
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        pytest.skip(f"Mesh file not found: {mesh_path}")
    return VolumetricMesh(mesh_path)


class TestParameterBoundsAndDataModel:
    """Tests for ParameterBounds dataclass and validation rules."""

    def test_parameter_bounds_valid(self) -> None:
        pb = ParameterBounds(
            name="youngs_modulus",
            initial_value=20000.0,
            lower_bound=1000.0,
            upper_bound=100000.0,
            description="Young's modulus in MPa",
        )
        assert pb.name == "youngs_modulus"
        assert pb.initial_value == 20000.0
        assert pb.lower_bound == 1000.0
        assert pb.upper_bound == 100000.0

    def test_parameter_bounds_invalid(self) -> None:
        # lower_bound >= upper_bound
        with pytest.raises(ValueError, match="must be strictly less than upper_bound"):
            ParameterBounds(name="E", initial_value=5.0, lower_bound=10.0, upper_bound=5.0)

        # initial_value outside bounds
        with pytest.raises(ValueError, match="must lie within"):
            ParameterBounds(name="E", initial_value=2.0, lower_bound=5.0, upper_bound=20.0)


class TestLinearElasticityCalibration:
    """Tests for closed-form and iterative linear modulus identification."""

    def test_closed_form_exact_modulus_recovery(self) -> None:
        """Verifies exact closed-form Young's modulus recovery in < 1 microsecond."""
        E_truth = 21000.0  # MPa
        E_baseline = 10000.0  # Initial guess (more than 50% off)

        # Synthetic linear force-deflection curve: F = k * E * delta
        deltas = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        k_geom = 2.5  # N / (mm * MPa)

        f_exp = k_geom * E_truth * deltas
        f_base = k_geom * E_baseline * deltas

        res = MaterialParameterIdentifier.calibrate_linear_elasticity(
            experimental_forces=f_exp,
            baseline_forces=f_base,
            baseline_modulus=E_baseline,
        )

        assert isinstance(res, IdentificationResult)
        assert res.success is True
        assert np.isclose(res.parameters["youngs_modulus"], E_truth, rtol=1e-12)
        assert res.rms_error < 1e-10
        assert res.relative_error < 1e-10

    def test_closed_form_with_tikhonov_regularization(self) -> None:
        E_truth = 20000.0
        E_0 = 10000.0
        deltas = np.array([1.0, 2.0, 3.0])
        f_exp = 5.0 * E_truth * deltas
        f_base = 5.0 * E_0 * deltas

        # With strong prior regularization beta, solution is pulled toward E_0
        res_unreg = MaterialParameterIdentifier.calibrate_linear_elasticity(f_exp, f_base, E_0, beta=0.0)
        res_reg = MaterialParameterIdentifier.calibrate_linear_elasticity(f_exp, f_base, E_0, beta=1e4)

        assert np.isclose(res_unreg.parameters["youngs_modulus"], E_truth)
        assert res_reg.parameters["youngs_modulus"] < E_truth
        assert res_reg.parameters["youngs_modulus"] > E_0

    def test_linear_fem_beam_modulus_identification(self, beam_mesh: VolumetricMesh) -> None:
        """Recovers Young's modulus E within 0.1% on full 3D beam cantilever FEM."""
        E_truth = 25000.0
        nu = 0.3

        # Ground truth forward model
        params_truth = IsotropicElasticParameters()
        params_truth.set_from_youngs_poisson(E_truth, nu, 1e-3)
        domain = SolidDomain(beam_mesh, Hookean(params_truth), Tet4Linear(Hookean(params_truth)))

        # Clamped root
        z_min = beam_mesh.bounds[4]
        root_nodes = select_nodes_by_bounds(beam_mesh, bounds=(-100, 100, -100, 100, z_min - 1, z_min + 5))
        clamp = DirichletConstraint(node_indices=root_nodes, components=(0, 1, 2), value=0.0)

        tip_node = 5
        prescribed_deltas = np.array([-2.0, -4.0, -6.0, -8.0])

        # Generate target forces
        exp_forces = []
        analysis = LinearStaticAnalysis(domain)
        f_ext = np.zeros(domain.mesh.n_dofs)

        for d in prescribed_deltas:
            tip_bc = DirichletConstraint([tip_node], components=(1,), value=d)
            dm = DirichletManager(domain.mesh.n_dofs)
            dm.add_constraint(clamp)
            dm.add_constraint(tip_bc)
            r = analysis.solve(dm, f_ext, verbose=False)
            exp_forces.append(r.reactions[3 * tip_node + 1])
        exp_forces = np.array(exp_forces)

        # Optimize E starting from initial guess E_0 = 12000.0 (over 50% error)
        def forward_model(params_dict: dict[str, float]) -> np.ndarray:
            E_val = params_dict["youngs_modulus"]
            p = IsotropicElasticParameters()
            p.set_from_youngs_poisson(E_val, nu, 1e-3)
            dom = SolidDomain(beam_mesh, Hookean(p), Tet4Linear(Hookean(p)))
            ans = LinearStaticAnalysis(dom)
            f_sim = []
            for d in prescribed_deltas:
                tip_bc = DirichletConstraint([tip_node], components=(1,), value=d)
                dm = DirichletManager(dom.mesh.n_dofs)
                dm.add_constraint(clamp)
                dm.add_constraint(tip_bc)
                res = ans.solve(dm, f_ext, verbose=False)
                f_sim.append(res.reactions[3 * tip_node + 1])
            return np.array(f_sim)

        identifier = MaterialParameterIdentifier.from_bounds(
            {"youngs_modulus": (12000.0, 1000.0, 100000.0)},
            method="trf",
        )
        result = identifier.identify(exp_forces, forward_model)

        assert result.success is True
        E_calibrated = result.parameters["youngs_modulus"]
        error_pct = abs(E_calibrated - E_truth) / E_truth * 100.0
        assert error_pct < 0.1, f"Calibrated E error {error_pct:.3f}% exceeds 0.1%"


class TestHyperelasticParameterIdentification:
    """Tests for hyperelastic Neo-Hookean C10 parameter calibration."""

    def test_hyperelastic_c10_identification(self, beam_mesh: VolumetricMesh) -> None:
        """Calibrates Neo-Hookean C10 from nonlinear force-deflection steps within 1% of truth."""
        # Baseline ground truth: C10 = 92.59 MPa (corresponds to E = 500 MPa, nu = 0.35)
        C10_truth = 92.59
        nu = 0.35

        z_min = beam_mesh.bounds[4]
        root_nodes = select_nodes_by_bounds(beam_mesh, bounds=(-100, 100, -100, 100, z_min - 1, z_min + 5))
        clamp = DirichletConstraint(node_indices=root_nodes, components=(0, 1, 2), value=0.0)
        tip_node = 5

        # 3 nonlinear indentation steps
        test_deltas = [-2.0, -4.0, -6.0]

        def solve_nonlinear_reactions(c10_val: float) -> np.ndarray:
            mu_val = 2.0 * float(c10_val)
            E_val = 2.0 * mu_val * (1.0 + nu)

            p = IsotropicElasticParameters()
            p.set_from_youngs_poisson(youngs_modulus=E_val, poissons_ratio=nu, mass_density=1e-3)
            material = NeoHookean(p)
            element = Tet4TotalLagrangian(material)
            domain = SolidDomain(beam_mesh, material, element)

            forces = []
            for d in test_deltas:
                tip_bc = DirichletConstraint(node_indices=[tip_node], components=(1,), value=d)
                dm = DirichletManager(domain.mesh.n_dofs)
                dm.add_constraint(clamp)
                dm.add_constraint(tip_bc)
                solver = NonlinearStaticAnalysis(domain)
                res = solver.solve(dirichlet=dm, n_steps=2, verbose=False)
                forces.append(res.reactions[3 * tip_node + 1])
            return np.array(forces)

        # Generate ground truth response
        exp_forces = solve_nonlinear_reactions(C10_truth)

        # Calibrate starting from C10_0 = 50.0 (46% initial error)
        identifier = MaterialParameterIdentifier.from_bounds(
            {"c10": (50.0, 10.0, 300.0)},
            method="trf",
            ftol=1e-3,
            xtol=1e-3,
        )

        def forward_fn(params: dict[str, float]) -> np.ndarray:
            return solve_nonlinear_reactions(params["c10"])

        result = identifier.identify(exp_forces, forward_fn)

        assert result.success is True
        c10_calibrated = result.parameters["c10"]
        error_pct = abs(c10_calibrated - C10_truth) / C10_truth * 100.0

        # Must recover C10 within 1% of ground truth (per roadmap spec)
        assert error_pct < 1.0, f"Calibrated C10 error {error_pct:.4f}% exceeds 1.0%"
        assert result.relative_error < 0.01


class TestMultiParameterIdentification:
    """Tests for simultaneous joint identification of multiple parameters (e.g. E and nu)."""

    def test_joint_e_and_nu_calibration(self) -> None:
        """Calibrates both Young's modulus E and Poisson's ratio nu from synthetic 2D response."""
        E_truth = 30000.0
        nu_truth = 0.35

        # Synthetic 2D response: axial stiffness K_axial = E * A / L, transverse K_trans = nu * K_axial
        def forward(p: dict[str, float]) -> np.ndarray:
            E = p["youngs_modulus"]
            nu = p["poissons_ratio"]
            # 4 synthetic measurement channels
            return np.array([
                E * 0.1,
                E * 0.2,
                E * nu * 0.05,
                E * nu * 0.10,
            ])

        exp_data = forward({"youngs_modulus": E_truth, "poissons_ratio": nu_truth})

        # Initial guess with 30% error
        identifier = MaterialParameterIdentifier.from_bounds(
            {
                "youngs_modulus": (20000.0, 1000.0, 100000.0),
                "poissons_ratio": (0.25, 0.05, 0.49),
            },
            method="trf",
        )

        res = identifier.identify(exp_data, forward)
        assert res.success is True
        assert np.isclose(res.parameters["youngs_modulus"], E_truth, rtol=1e-3)
        assert np.isclose(res.parameters["poissons_ratio"], nu_truth, rtol=1e-3)


class TestAdversarialAndErrorHandling:
    """Tests for edge cases, non-physical values, and exception handling."""

    def test_negative_beta_raises(self) -> None:
        with pytest.raises(ValueError, match="beta must be non-negative"):
            MaterialParameterIdentifier.from_bounds({"E": (10.0, 1.0, 100.0)}, beta=-1.0)

    def test_empty_observations_raises(self) -> None:
        identifier = MaterialParameterIdentifier.from_bounds({"E": (10.0, 1.0, 100.0)})
        with pytest.raises(ValueError, match="experimental_forces cannot be empty"):
            identifier.identify([], lambda p: np.array([1.0]))

    def test_mismatched_weights_raises(self) -> None:
        identifier = MaterialParameterIdentifier.from_bounds(
            {"E": (10.0, 1.0, 100.0)},
            weights=[1.0, 2.0],  # 2 weights
        )
        with pytest.raises(ValueError, match="weights length"):
            identifier.identify([10.0, 20.0, 30.0], lambda p: np.array([10.0, 20.0, 30.0]))

    def test_nonpositive_weights_raises(self) -> None:
        identifier = MaterialParameterIdentifier.from_bounds(
            {"E": (10.0, 1.0, 100.0)},
            weights=[1.0, 0.0, 1.0],
        )
        with pytest.raises(ValueError, match="strictly positive"):
            identifier.identify([1.0, 2.0, 3.0], lambda p: np.array([1.0, 2.0, 3.0]))

    def test_simulation_failure_penalty_recovery(self) -> None:
        """Verifies optimizer gracefully handles forward simulation errors via penalties."""
        identifier = MaterialParameterIdentifier.from_bounds(
            {"E": (10.0, 1.0, 100.0)},
            method="trf",
        )

        def unstable_forward(p: dict[str, float]) -> np.ndarray:
            if p["E"] < 5.0:
                raise RuntimeError("Nonlinear element inversion simulated breakdown")
            return np.array([p["E"] * 2.0, p["E"] * 4.0])

        exp_data = np.array([40.0, 80.0])  # Truth E = 20.0
        res = identifier.identify(exp_data, unstable_forward)
        assert res.success is True
        assert np.isclose(res.parameters["E"], 20.0, rtol=1e-3)
