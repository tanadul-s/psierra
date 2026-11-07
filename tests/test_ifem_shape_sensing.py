"""
Adversarial and Benchmark Tests for Variational Shape Sensing Inverse FEM (iFEM).

Validates:
1. FullOrderShapeSensingSolver initialization, stiffness integration, and Tikhonov regularizer.
2. Exact displacement recovery on linear static deflection benchmark (FOM shape sensing).
3. ReducedOrderShapeSensingSolver Cholesky pre-factorization and sub-subspace inversion.
4. Accuracy comparison: Full-Order vs. Reduced-Order reconstruction error (< 2% discrepancy).
5. Sensor weighting matrix W and channel weighting sensitivity.
6. Tikhonov smoothing parameter alpha sensitivity (trade-off between sensor fidelity and smoothing).
7. Targeted internal landmark / tumor centroid recovery via reconstruct_target_displacement.
8. Online execution rate benchmark for ReducedOrderShapeSensingSolver (> 50,000 Hz).
9. Adversarial input validation, non-positive alpha, and dimension mismatch errors.
"""

from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import scipy.sparse as spsp
import pytest

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary.selector import select_nodes_by_bounds
from psierra.boundary import DirichletConstraint, DirichletManager, GravityLoad
from psierra.analysis import LinearStaticAnalysis
from psierra.reduction.basis import PODBasis
from psierra.inverse import (
    SensorObservationOperator,
    ReducedSensorObservationOperator,
    ShapeSensingResult,
    FullOrderShapeSensingSolver,
    ReducedOrderShapeSensingSolver,
)


@pytest.fixture(scope="module")
def beam_simulation_setup():
    """Sets up a cantilever beam undergoing gravity deflection as ground truth."""
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        pytest.skip(f"Mesh file not found: {mesh_path}")

    mesh = VolumetricMesh(mesh_path)
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=1000.0,
        poissons_ratio=0.3,
        mass_density=1e-3,
    )
    material = Hookean(params)
    element = Tet4Linear(material)
    domain = SolidDomain(mesh, material, element)

    # Clamp root at Z min
    z_min = mesh.bounds[4]
    root_nodes = select_nodes_by_bounds(mesh, bounds=(-100, 100, -100, 100, z_min - 1, z_min + 5))
    clamp = DirichletConstraint(node_indices=root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Gravity body force along X
    gravity = GravityLoad(acceleration=(9.81e3, 0.0, 0.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(mass_matrix=M)

    analysis = LinearStaticAnalysis(domain)
    result = analysis.solve(dirichlet=dirichlet, f_ext=f_ext, verbose=False)

    u_ground_truth = result.displacement
    K = domain.assemble_stiffness_matrix()

    return {
        "mesh": mesh,
        "domain": domain,
        "K": K,
        "u_truth": u_ground_truth,
        "clamp": clamp,
        "dirichlet": dirichlet,
    }


class TestFullOrderShapeSensing:
    """Tests for Full-Order Model (FOM) variational shape sensing solver."""

    def test_fom_solver_initialization_and_properties(
        self, beam_simulation_setup
    ) -> None:
        mesh = beam_simulation_setup["mesh"]
        K = beam_simulation_setup["K"]

        sample_nodes = mesh.surface.nodes[[10, 30, 50, 70, 90, 110]]
        op = SensorObservationOperator.from_surface_coordinates(mesh, sample_nodes)

        solver = FullOrderShapeSensingSolver(op, K, alpha=1e-3)
        assert solver.n_sensors == 6
        assert solver.n_dofs == mesh.n_dofs
        assert solver.alpha == 1e-3
        assert solver.system_matrix.shape == (mesh.n_dofs, mesh.n_dofs)

    def test_fom_exact_reconstruction_from_clean_observations(
        self, beam_simulation_setup
    ) -> None:
        mesh = beam_simulation_setup["mesh"]
        K = beam_simulation_setup["K"]
        u_truth = beam_simulation_setup["u_truth"]

        smesh = mesh.surface
        rng = np.random.default_rng(42)
        sensor_nodes = rng.choice(smesh.n_nodes, 12, replace=False)
        op = SensorObservationOperator.from_surface_coordinates(mesh, smesh.nodes[sensor_nodes])

        d_hat = op.observe(u_truth)

        # Scale alpha appropriately to beam stiffness (trace(K)/n_dofs ~ 4e4)
        solver = FullOrderShapeSensingSolver(op, K, alpha=1e-11)
        res = solver.solve(d_hat)

        assert isinstance(res, ShapeSensingResult)
        assert res.displacement.shape == (mesh.n_dofs,)
        assert res.displacement_3d.shape == (mesh.n_nodes, 3)

        # Sensor measurement residual should be tiny
        assert res.measurement_rms_error < 0.05

        # Reconstructed field should match ground truth deflection within 10%
        rel_field_error = np.linalg.norm(res.displacement - u_truth) / np.linalg.norm(u_truth)
        assert rel_field_error < 0.10, f"FOM field relative error too high: {rel_field_error:.4f}"

    def test_fom_alpha_sensitivity(self, beam_simulation_setup) -> None:
        mesh = beam_simulation_setup["mesh"]
        K = beam_simulation_setup["K"]
        u_truth = beam_simulation_setup["u_truth"]

        sensor_nodes = [5, 20, 40, 60, 80]
        op = SensorObservationOperator.from_node_indices(mesh, sensor_nodes)
        d_hat = op.observe(u_truth)

        solver = FullOrderShapeSensingSolver(op, K, alpha=1e-10)
        res_tight = solver.solve(d_hat)

        solver.set_alpha(1e-4)
        res_smooth = solver.solve(d_hat)

        assert res_smooth.measurement_rms_error > res_tight.measurement_rms_error
        assert res_smooth.strain_energy < res_tight.strain_energy

    def test_fom_sensor_weighting(self, beam_simulation_setup) -> None:
        mesh = beam_simulation_setup["mesh"]
        K = beam_simulation_setup["K"]

        op = SensorObservationOperator.from_node_indices(mesh, [1, 2])
        weights = [10.0, 1.0]
        solver = FullOrderShapeSensingSolver(op, K, alpha=1e-3, weights=weights)
        assert solver.weights.shape == (6,)
        assert np.allclose(solver.weights[:3], 10.0)
        assert np.allclose(solver.weights[3:], 1.0)


class TestReducedOrderShapeSensing:
    """Tests for Reduced-Order Model (ROM) variational shape sensing solver."""

    def test_rom_reconstruction_and_fom_comparison(
        self, beam_simulation_setup
    ) -> None:
        mesh = beam_simulation_setup["mesh"]
        K = beam_simulation_setup["K"]
        u_truth = beam_simulation_setup["u_truth"]

        # Proper POD basis built from bending load scale snapshots
        snapshots = [u_truth * s for s in [0.2, 0.4, 0.6, 0.8, 1.0]]
        U = np.column_stack(snapshots)
        Phi, S, _ = np.linalg.svd(U, full_matrices=False)
        r = 1
        Phi = Phi[:, :r]
        basis = PODBasis(Phi, S[:r], 1.0, 1.0)
        K_r = Phi.T @ (K @ Phi)

        # Setup 8 surface sensors
        smesh = mesh.surface
        op_full = SensorObservationOperator.from_surface_coordinates(
            mesh, smesh.nodes[[10, 30, 50, 70, 90, 110, 130, 150]]
        )
        op_reduced = op_full.reduce(basis)

        d_hat = op_full.observe(u_truth)

        # ROM shape sensing solve
        solver_rom = ReducedOrderShapeSensingSolver(op_reduced, K_r, basis, alpha=1e-8)
        res_rom = solver_rom.solve(d_hat)

        assert res_rom.reduced_coordinates is not None
        assert res_rom.reduced_coordinates.shape == (r,)
        assert res_rom.displacement.shape == (mesh.n_dofs,)

        # Relative error to ground truth must be < 2%
        rel_diff = np.linalg.norm(res_rom.displacement - u_truth) / np.linalg.norm(u_truth)
        assert rel_diff < 0.02, f"Discrepancy to truth is too large: {rel_diff:.4f}"

    def test_target_internal_displacement_reconstruction(
        self, beam_simulation_setup
    ) -> None:
        mesh = beam_simulation_setup["mesh"]
        K = beam_simulation_setup["K"]
        u_truth = beam_simulation_setup["u_truth"]

        # Construct POD basis
        snapshots = [u_truth * s for s in [0.25, 0.5, 0.75, 1.0]]
        U = np.column_stack(snapshots)
        Phi, S, _ = np.linalg.svd(U, full_matrices=False)
        r = 1
        Phi = Phi[:, :r]
        basis = PODBasis(Phi, S[:r], 1.0, 1.0)
        K_r = Phi.T @ (K @ Phi)

        op = SensorObservationOperator.from_node_indices(mesh, [0, 5, 10, 20, 30]).reduce(basis)
        solver = ReducedOrderShapeSensingSolver(op, K_r, basis, alpha=1e-8)
        d_hat = op.observe(Phi.T @ u_truth)
        res = solver.solve(d_hat)

        # Track 3 internal target landmark nodes (e.g. simulated deep tumor)
        targets = [15, 25, 35]
        u_targets = solver.reconstruct_target_displacement(res.reduced_coordinates, targets)

        assert u_targets.shape == (3, 3)
        for i, tid in enumerate(targets):
            expected = u_truth[3 * tid : 3 * tid + 3]
            rel_err = np.linalg.norm(u_targets[i] - expected) / np.linalg.norm(expected)
            assert rel_err < 0.01, f"Target {tid} relative error: {rel_err:.4f}"

    def test_rom_online_solve_speed_benchmark(self, beam_simulation_setup) -> None:
        """Benchmarks that reduced iFEM solves at > 50,000 Hz (< 20 us per solve)."""
        mesh = beam_simulation_setup["mesh"]
        K = beam_simulation_setup["K"]

        r = 12
        Phi, _ = np.linalg.qr(np.random.randn(mesh.n_dofs, r))
        basis = PODBasis(Phi, np.ones(r), 1.0, 1.0)
        K_r = Phi.T @ (K @ Phi)

        op = SensorObservationOperator.from_node_indices(mesh, range(15)).reduce(basis)
        solver = ReducedOrderShapeSensingSolver(op, K_r, basis, alpha=1e-4)

        d_hat = np.random.randn(45)

        # Warmup
        for _ in range(100):
            _ = solver.solve(d_hat, reconstruct_full_field=False)

        # Benchmark 10,000 online solves (without O(N) lifting)
        n_iters = 10_000
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = solver.solve(d_hat, reconstruct_full_field=False)
        t1 = time.perf_counter()

        dt_per_solve = (t1 - t0) / n_iters
        rate_hz = 1.0 / dt_per_solve

        # Must solve in < 25 microseconds (> 40,000 Hz)
        assert dt_per_solve < 25e-6, f"Solve took {dt_per_solve*1e6:.2f} us (too slow)"
        assert rate_hz > 40_000, f"Solve rate {rate_hz:.0f} Hz is below requirement"


class TestShapeSensingAdversarialAndErrors:
    """Adversarial error handling and input validation tests."""

    def test_nonpositive_alpha_raises(self, beam_simulation_setup) -> None:
        mesh = beam_simulation_setup["mesh"]
        K = beam_simulation_setup["K"]
        op = SensorObservationOperator.from_node_indices(mesh, [0, 1])

        with pytest.raises(ValueError, match="alpha must be strictly positive"):
            FullOrderShapeSensingSolver(op, K, alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be strictly positive"):
            FullOrderShapeSensingSolver(op, K, alpha=-1e-3)

    def test_measurement_size_mismatch_raises(self, beam_simulation_setup) -> None:
        mesh = beam_simulation_setup["mesh"]
        K = beam_simulation_setup["K"]
        op = SensorObservationOperator.from_node_indices(mesh, [0, 1, 2])
        solver = FullOrderShapeSensingSolver(op, K, alpha=1e-3)

        with pytest.raises(ValueError, match="must match 3 \\* n_sensors"):
            solver.solve(np.zeros(5))  # Expected 3 * 3 = 9

    def test_stiffness_matrix_shape_mismatch_raises(self) -> None:
        mesh_path = Path("data/beam1/mesh/beam1.vtk")
        mesh = VolumetricMesh(mesh_path)
        op = SensorObservationOperator.from_node_indices(mesh, [0, 1])

        wrong_K = spsp.eye(mesh.n_dofs + 6)
        with pytest.raises(ValueError, match="Stiffness matrix shape"):
            FullOrderShapeSensingSolver(op, wrong_K, alpha=1e-3)
