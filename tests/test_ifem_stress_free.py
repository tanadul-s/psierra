"""
Adversarial and Benchmark Tests for Stress-Free Reference Configuration Solver.

Validates:
1. StressFreeResult container attributes, properties, mesh factory methods, and summaries.
2. StressFreeConfigurationSolver initialization, argument validation, and Dirichlet managers.
3. Exact recovery of undeformed cantilever beam reference geometry from gravity-sagged scan (< 10^-3 mm error).
4. Elimination of double-gravity bias during surgical posture reorientation (prone / inverted loading).
5. Sensitivity and stability across relaxation factors (omega = 0.5, 0.7, 1.0) and Aitken dynamic relaxation.
6. Nonlinear finite-strain hyperelastic reference configuration recovery (Total Lagrangian NeoHookean).
7. Robust error handling for dimension mismatches, malformed coordinates, and invalid arguments.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from psierra.material import IsotropicElasticParameters, Hookean, NeoHookean
from psierra.element import Tet4Linear, Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary.selector import select_nodes_by_bounds
from psierra.boundary import DirichletConstraint, DirichletManager, GravityLoad
from psierra.analysis import LinearStaticAnalysis, NonlinearStaticAnalysis
from psierra.inverse import StressFreeResult, StressFreeConfigurationSolver


@pytest.fixture(scope="module")
def beam_linear_setup():
    """Sets up a cantilever beam undergoing linear elastostatic gravity deflection."""
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        pytest.skip(f"Mesh file not found: {mesh_path}")

    mesh = VolumetricMesh(mesh_path)
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000.0,
        poissons_ratio=0.35,
        mass_density=5.0e-5,
    )
    material = Hookean(params)
    element = Tet4Linear(material)
    domain = SolidDomain(mesh, material, element)

    # Clamped root at Z_min
    z_min = mesh.bounds[4]
    root_nodes = select_nodes_by_bounds(
        mesh, bounds=(-100, 100, -100, 100, z_min - 1, z_min + 5)
    )
    clamp = DirichletConstraint(
        node_indices=root_nodes, components=(0, 1, 2), value=0.0
    )
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Gravity self-weight along -Y (sagging downward)
    gravity = GravityLoad(acceleration=(0.0, -9.81e3, 0.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(mass_matrix=M)

    analysis = LinearStaticAnalysis(domain)
    res_forward = analysis.solve(dirichlet=dirichlet, f_ext=f_ext, verbose=False)

    X_true = mesh.nodes.copy()
    u_forward = res_forward.displacement
    X_scan = X_true + u_forward.reshape(-1, 3)

    return {
        "mesh": mesh,
        "params": params,
        "material": material,
        "element": element,
        "domain": domain,
        "clamp": clamp,
        "dirichlet": dirichlet,
        "gravity": gravity,
        "X_true": X_true,
        "X_scan": X_scan,
        "u_forward": u_forward,
        "root_nodes": root_nodes,
    }


class TestStressFreeResultDataModel:
    """Tests for StressFreeResult dataclass and helper methods."""

    def test_result_attributes_and_properties(self, beam_linear_setup) -> None:
        X_true = beam_linear_setup["X_true"]
        X_scan = beam_linear_setup["X_scan"]
        u = beam_linear_setup["u_forward"]
        n_nodes = len(X_true)

        res = StressFreeResult(
            reference_nodes=X_true,
            scan_nodes=X_scan,
            predicted_scan_nodes=X_scan,
            displacement=u,
            residual_error=1.5e-5,
            relative_error=2.3e-6,
            rms_error=1.1e-5,
            iterations=8,
            converged=True,
            residual_history=[1.2, 0.5, 0.1, 1.5e-5],
            wall_time_seconds=0.045,
        )

        assert res.n_nodes == n_nodes
        assert res.displacement_3d.shape == (n_nodes, 3)
        assert res.converged is True
        assert res.iterations == 8
        assert res.residual_error == 1.5e-5
        assert len(res.residual_history) == 4

        summary = res.summary()
        assert "Converged in 8 iterations" in summary
        assert "1.5000e-05 mm" in summary

    def test_create_reference_and_predicted_mesh(self, beam_linear_setup) -> None:
        mesh = beam_linear_setup["mesh"]
        X_true = beam_linear_setup["X_true"]
        X_scan = beam_linear_setup["X_scan"]
        u = beam_linear_setup["u_forward"]

        res = StressFreeResult(
            reference_nodes=X_true,
            scan_nodes=X_scan,
            predicted_scan_nodes=X_scan,
            displacement=u,
            residual_error=0.0,
            relative_error=0.0,
            rms_error=0.0,
            iterations=1,
            converged=True,
        )

        ref_mesh = res.create_reference_mesh(mesh)
        assert isinstance(ref_mesh, VolumetricMesh)
        assert np.allclose(ref_mesh.nodes, X_true)

        pred_mesh = res.create_predicted_mesh(mesh)
        assert isinstance(pred_mesh, VolumetricMesh)
        assert np.allclose(pred_mesh.nodes, X_scan)

    def test_create_mesh_dimension_mismatch_raises(self, beam_linear_setup) -> None:
        X_true = beam_linear_setup["X_true"]
        X_scan = beam_linear_setup["X_scan"]
        u = beam_linear_setup["u_forward"]

        res = StressFreeResult(
            reference_nodes=X_true[:10],
            scan_nodes=X_scan[:10],
            predicted_scan_nodes=X_scan[:10],
            displacement=u[:30],
            residual_error=0.0,
            relative_error=0.0,
            rms_error=0.0,
            iterations=1,
            converged=True,
        )

        mesh = beam_linear_setup["mesh"]
        with pytest.raises(ValueError, match="expected 10"):
            res.create_reference_mesh(mesh)


class TestStressFreeSolverInitialization:
    """Tests for StressFreeConfigurationSolver initialization and validation."""

    def test_valid_initialization(self, beam_linear_setup) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]

        solver = StressFreeConfigurationSolver(
            domain=domain,
            load=gravity,
            dirichlet=dirichlet,
            relaxation_factor=0.6,
            tolerance=1e-5,
            max_iterations=30,
            analysis_type="linear",
        )

        assert solver.relaxation_factor == 0.6
        assert solver.tolerance == 1e-5
        assert solver.max_iterations == 30
        assert solver.analysis_type == "linear"

    def test_sequence_dirichlet_initialization(self, beam_linear_setup) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        clamp = beam_linear_setup["clamp"]

        solver = StressFreeConfigurationSolver(
            domain=domain,
            load=gravity,
            dirichlet=[clamp],
            relaxation_factor=0.7,
        )
        assert isinstance(solver._dirichlet, DirichletManager)

    @pytest.mark.parametrize("invalid_omega", [0.0, -0.5, 1.5, 2.0])
    def test_invalid_relaxation_factor_raises(
        self, beam_linear_setup, invalid_omega
    ) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]

        with pytest.raises(ValueError, match="relaxation_factor must lie in"):
            StressFreeConfigurationSolver(
                domain=domain,
                load=gravity,
                dirichlet=dirichlet,
                relaxation_factor=invalid_omega,
            )

    @pytest.mark.parametrize("invalid_tol", [0.0, -1e-4])
    def test_invalid_tolerance_raises(self, beam_linear_setup, invalid_tol) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]

        with pytest.raises(ValueError, match="tolerance must be strictly positive"):
            StressFreeConfigurationSolver(
                domain=domain,
                load=gravity,
                dirichlet=dirichlet,
                tolerance=invalid_tol,
            )

    def test_invalid_max_iterations_raises(self, beam_linear_setup) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]

        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            StressFreeConfigurationSolver(
                domain=domain,
                load=gravity,
                dirichlet=dirichlet,
                max_iterations=0,
            )

    def test_invalid_analysis_type_raises(self, beam_linear_setup) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]

        with pytest.raises(ValueError, match="analysis_type must be 'linear' or 'nonlinear'"):
            StressFreeConfigurationSolver(
                domain=domain,
                load=gravity,
                dirichlet=dirichlet,
                analysis_type="modal",  # type: ignore
            )


class TestLinearStressFreeInversion:
    """Benchmark tests recovering undeformed reference configuration under linear elasticity."""

    def test_exact_recovery_of_reference_geometry(self, beam_linear_setup) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]
        X_true = beam_linear_setup["X_true"]
        X_scan = beam_linear_setup["X_scan"]

        solver = StressFreeConfigurationSolver(
            domain=domain,
            load=gravity,
            dirichlet=dirichlet,
            relaxation_factor=0.7,
            tolerance=1e-4,
            max_iterations=25,
        )

        result = solver.solve(X_scan, verbose=False)

        assert result.converged is True
        assert result.residual_error <= 1e-4

        # Recovery error: difference between recovered X_0* and true undeformed X_true
        recovery_error = np.max(np.abs(result.reference_nodes - X_true))
        assert recovery_error < 1e-3, f"Recovery error {recovery_error:.4e} mm exceeds 1e-3 mm threshold."

        # Forward prediction error: X_0* + u* matches X_scan
        pred_error = np.max(np.abs(result.predicted_scan_nodes - X_scan))
        assert pred_error <= 1e-4

    def test_monotonic_residual_reduction(self, beam_linear_setup) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]
        X_scan = beam_linear_setup["X_scan"]

        solver = StressFreeConfigurationSolver(
            domain=domain,
            load=gravity,
            dirichlet=dirichlet,
            relaxation_factor=0.7,
            tolerance=1e-4,
            max_iterations=25,
        )

        result = solver.solve(X_scan, verbose=False)
        history = result.residual_history
        assert len(history) >= 3

        # Check general downward trend
        assert history[-1] < history[0] * 1e-4
        for k in range(len(history) - 1):
            assert history[k + 1] <= history[k] + 1e-10

    def test_solve_with_mesh_object_and_initial_guess(self, beam_linear_setup) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]
        mesh = beam_linear_setup["mesh"]
        X_true = beam_linear_setup["X_true"]
        X_scan = beam_linear_setup["X_scan"]

        scan_mesh = mesh.copy()
        scan_mesh.nodes = X_scan.copy()

        solver = StressFreeConfigurationSolver(
            domain=domain,
            load=gravity,
            dirichlet=dirichlet,
            relaxation_factor=0.8,
            tolerance=1e-4,
        )

        # Solve passing VolumetricMesh as scan input and explicit initial guess
        init_guess = X_scan.copy()
        result = solver.solve(scan_mesh, initial_guess=init_guess, verbose=False)

        assert result.converged is True
        recovery_error = np.max(np.abs(result.reference_nodes - X_true))
        assert recovery_error < 1e-3


class TestDoubleGravityBiasElimination:
    """Verifies that posture reorientation without stress-free inversion suffers double-gravity bias."""

    def test_double_gravity_bias_elimination(self, beam_linear_setup) -> None:
        domain = beam_linear_setup["domain"]
        material = beam_linear_setup["material"]
        element = beam_linear_setup["element"]
        gravity_scan = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]
        X_true = beam_linear_setup["X_true"]
        X_scan = beam_linear_setup["X_scan"]
        mesh = beam_linear_setup["mesh"]

        # 1. Recover stress-free reference geometry X_0* from scan
        solver = StressFreeConfigurationSolver(
            domain=domain,
            load=gravity_scan,
            dirichlet=dirichlet,
            relaxation_factor=0.8,
            tolerance=1e-4,
        )
        res_sf = solver.solve(X_scan, verbose=False)
        assert res_sf.converged is True
        X_0_star = res_sf.reference_nodes

        # 2. Simulate posture change: reorient posture so gravity is inverted (0, +9.81e3, 0)
        # e.g., patient moving from supine (sagging down) to prone (sagging up)
        gravity_surgery = GravityLoad(acceleration=(0.0, 9.81e3, 0.0))

        # True Physical State (forward solve from true reference X_true under inverted gravity):
        true_mesh = mesh.copy()
        true_mesh.nodes = X_true.copy()
        dom_true = SolidDomain(true_mesh, material, element)
        M_true = dom_true.assemble_mass_matrix()
        f_surg_true = gravity_surgery.compute_force_vector(mass_matrix=M_true)
        res_true = LinearStaticAnalysis(dom_true).solve(dirichlet=dirichlet, f_ext=f_surg_true, verbose=False)
        x_true_operative = X_true + res_true.displacement.reshape(-1, 3)

        # A. Unbiased State (forward solve from recovered X_0* under inverted gravity):
        unbiased_mesh = mesh.copy()
        unbiased_mesh.nodes = X_0_star.copy()
        dom_unbiased = SolidDomain(unbiased_mesh, material, element)
        M_unbiased = dom_unbiased.assemble_mass_matrix()
        f_surg_unbiased = gravity_surgery.compute_force_vector(mass_matrix=M_unbiased)
        res_unbiased = LinearStaticAnalysis(dom_unbiased).solve(dirichlet=dirichlet, f_ext=f_surg_unbiased, verbose=False)
        x_unbiased_operative = X_0_star + res_unbiased.displacement.reshape(-1, 3)

        # B. Biased State (naive simulation taking X_scan directly as reference geometry):
        biased_mesh = mesh.copy()
        biased_mesh.nodes = X_scan.copy()
        dom_biased = SolidDomain(biased_mesh, material, element)
        M_biased = dom_biased.assemble_mass_matrix()
        f_surg_biased = gravity_surgery.compute_force_vector(mass_matrix=M_biased)
        res_biased = LinearStaticAnalysis(dom_biased).solve(dirichlet=dirichlet, f_ext=f_surg_biased, verbose=False)
        x_biased_operative = X_scan + res_biased.displacement.reshape(-1, 3)

        # The biased state has massive error (~19 mm in this cantilever setup)
        biased_error = float(np.max(np.abs(x_biased_operative - x_true_operative)))
        unbiased_error = float(np.max(np.abs(x_unbiased_operative - x_true_operative)))

        assert biased_error > 10.0, f"Biased error ({biased_error:.2f} mm) should be large."
        assert unbiased_error < 1e-3, f"Unbiased error ({unbiased_error:.4e} mm) should be microscopic."


class TestRelaxationFactorsAndAdaptiveAitken:
    """Tests convergence across relaxation factors and adaptive Aitken acceleration."""

    @pytest.mark.parametrize("omega", [0.5, 0.7, 1.0])
    def test_fixed_relaxation_factors(self, beam_linear_setup, omega: float) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]
        X_true = beam_linear_setup["X_true"]
        X_scan = beam_linear_setup["X_scan"]

        solver = StressFreeConfigurationSolver(
            domain=domain,
            load=gravity,
            dirichlet=dirichlet,
            relaxation_factor=omega,
            tolerance=1e-4,
            max_iterations=30,
        )

        result = solver.solve(X_scan, verbose=False)
        assert result.converged is True
        recovery_err = np.max(np.abs(result.reference_nodes - X_true))
        assert recovery_err < 1e-3

    def test_adaptive_aitken_relaxation(self, beam_linear_setup) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]
        X_true = beam_linear_setup["X_true"]
        X_scan = beam_linear_setup["X_scan"]

        solver = StressFreeConfigurationSolver(
            domain=domain,
            load=gravity,
            dirichlet=dirichlet,
            relaxation_factor=0.7,
            adaptive_relaxation=True,
            tolerance=1e-4,
            max_iterations=25,
        )

        result = solver.solve(X_scan, verbose=False)
        assert result.converged is True
        recovery_err = np.max(np.abs(result.reference_nodes - X_true))
        assert recovery_err < 1e-3


class TestNonlinearStressFreeInversion:
    """Benchmark tests for nonlinear finite-strain hyperelastic reference configuration recovery."""

    def test_nonlinear_neohookean_inversion(self, beam_linear_setup) -> None:
        mesh = beam_linear_setup["mesh"]
        params = beam_linear_setup["params"]
        dirichlet = beam_linear_setup["dirichlet"]

        # Hyperelastic NeoHookean Total Lagrangian domain with smaller load for quick test
        material = NeoHookean(params)
        element = Tet4TotalLagrangian(material)
        domain = SolidDomain(mesh.copy(), material, element)

        # Smaller gravity for fast 1-step nonlinear benchmark
        gravity = GravityLoad(acceleration=(0.0, -2.0e3, 0.0))
        M = domain.assemble_mass_matrix()
        f_ext = gravity.compute_force_vector(mass_matrix=M)

        nl_analysis = NonlinearStaticAnalysis(domain)
        res_fwd = nl_analysis.solve(dirichlet=dirichlet, f_ext=f_ext, n_steps=1, nr_iterations=15, verbose=False)
        X_true = mesh.nodes.copy()
        X_scan = X_true + res_fwd.displacement.reshape(-1, 3)

        solver = StressFreeConfigurationSolver(
            domain=domain,
            load=gravity,
            dirichlet=dirichlet,
            relaxation_factor=0.8,
            tolerance=1e-3,
            max_iterations=10,
            analysis_type="nonlinear",
            nonlinear_kwargs={"n_steps": 1, "nr_iterations": 15, "nr_tol": 1e-4},
        )

        result = solver.solve(X_scan, verbose=False)
        assert result.converged is True
        recovery_err = np.max(np.abs(result.reference_nodes - X_true))
        assert recovery_err < 5e-3


class TestAdversarialAndErrorHandling:
    """Tests verifying input validation and informative error messages."""

    def test_scan_nodes_shape_mismatch_raises(self, beam_linear_setup) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]
        X_scan = beam_linear_setup["X_scan"]

        solver = StressFreeConfigurationSolver(
            domain=domain,
            load=gravity,
            dirichlet=dirichlet,
        )

        # Wrong node count
        with pytest.raises(ValueError, match="node count .* does not match"):
            solver.solve(X_scan[:50])

        # Wrong dimensionality (1D array)
        with pytest.raises(ValueError, match="shape .*N_nodes, 3"):
            solver.solve(X_scan.flatten())

    def test_initial_guess_shape_mismatch_raises(self, beam_linear_setup) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]
        X_scan = beam_linear_setup["X_scan"]

        solver = StressFreeConfigurationSolver(
            domain=domain,
            load=gravity,
            dirichlet=dirichlet,
        )

        with pytest.raises(ValueError, match="initial_guess shape .* does not match"):
            solver.solve(X_scan, initial_guess=X_scan[:10])

    def test_unsupported_scan_input_type_raises(self, beam_linear_setup) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]

        solver = StressFreeConfigurationSolver(
            domain=domain,
            load=gravity,
            dirichlet=dirichlet,
        )

        with pytest.raises(TypeError, match="must be VolumetricMesh or np.ndarray"):
            solver.solve("invalid_mesh_type")  # type: ignore

    def test_constant_load_vector_and_callable(self, beam_linear_setup) -> None:
        domain = beam_linear_setup["domain"]
        gravity = beam_linear_setup["gravity"]
        dirichlet = beam_linear_setup["dirichlet"]
        X_true = beam_linear_setup["X_true"]
        X_scan = beam_linear_setup["X_scan"]

        M = domain.assemble_mass_matrix()
        f_vec = gravity.compute_force_vector(mass_matrix=M)

        # Test solver with constant numpy ndarray load
        solver_vec = StressFreeConfigurationSolver(
            domain=domain,
            load=f_vec,
            dirichlet=dirichlet,
            relaxation_factor=0.8,
            tolerance=1e-4,
        )
        res_vec = solver_vec.solve(X_scan)
        assert res_vec.converged is True

        # Test solver with callable load
        solver_fn = StressFreeConfigurationSolver(
            domain=domain,
            load=lambda d: f_vec,
            dirichlet=dirichlet,
            relaxation_factor=0.8,
            tolerance=1e-4,
        )
        res_fn = solver_fn.solve(X_scan)
        assert res_fn.converged is True
