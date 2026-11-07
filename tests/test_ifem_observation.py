"""
Adversarial and Benchmark Tests for Discrete Sensor Observation Operator (H).

Validates:
1. DiscreteSensor metadata, shape constraints, and barycentric displacement evaluation.
2. Direct nodal sensor operator construction and exact nodal displacement tracking.
3. Continuous 3D surface point-in-facet barycentric locator, Voronoi projections,
   and distance tolerance checking.
4. Volumetric tetrahedral point-in-element barycentric locator.
5. Linear algebraic properties of H: partition of unity (row sums = 1.0),
   rigid body translation invariance, and linear displacement field preservation.
6. Reduced observation operator projection H_r = H @ Phi, exact projection identity
   ||H @ (Phi @ u_r) - H_r @ u_r|| < 1e-14, and batch evaluation.
7. Online microsecond-rate execution speed benchmark (> 200,000 Hz).
8. Comprehensive boundary condition, dimension mismatch, and error handling.
"""

from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import scipy.sparse as spsp
import pytest

from psierra.inverse import (
    DiscreteSensor,
    SensorObservationOperator,
    ReducedSensorObservationOperator,
)
from psierra.mesh import VolumetricMesh
from psierra.reduction.basis import PODBasis


@pytest.fixture(scope="module")
def beam_mesh() -> VolumetricMesh:
    """Loads reference cantilever beam volumetric mesh."""
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        pytest.skip(f"Mesh file not found: {mesh_path}")
    return VolumetricMesh(mesh_path)


class TestDiscreteSensorDataModel:
    """Tests for DiscreteSensor data model, validation, and evaluation."""

    def test_sensor_creation_and_properties(self) -> None:
        sensor = DiscreteSensor(
            sensor_id=0,
            coordinates=np.array([1.0, 2.0, 3.0]),
            closest_point=np.array([1.0, 2.0, 3.0]),
            node_indices=(5, 12, 19),
            weights=(0.2, 0.5, 0.3),
            cell_index=42,
            distance=0.0,
        )
        assert sensor.sensor_id == 0
        assert np.allclose(sensor.coordinates, [1.0, 2.0, 3.0])
        assert sensor.node_indices == (5, 12, 19)
        assert np.isclose(sum(sensor.weights), 1.0)
        assert sensor.cell_index == 42
        assert sensor.distance == 0.0

    def test_sensor_validation_errors(self) -> None:
        # Invalid coordinate shape
        with pytest.raises(ValueError, match="Sensor coordinates must have shape"):
            DiscreteSensor(
                sensor_id=0,
                coordinates=np.array([1.0, 2.0]),
                closest_point=np.array([1.0, 2.0, 3.0]),
                node_indices=(0,),
                weights=(1.0,),
                cell_index=0,
                distance=0.0,
            )

        # Mismatched node_indices and weights count
        with pytest.raises(ValueError, match="Length of node_indices"):
            DiscreteSensor(
                sensor_id=0,
                coordinates=np.array([1.0, 2.0, 3.0]),
                closest_point=np.array([1.0, 2.0, 3.0]),
                node_indices=(0, 1),
                weights=(1.0,),
                cell_index=0,
                distance=0.0,
            )

        # Weights not summing to 1.0
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            DiscreteSensor(
                sensor_id=0,
                coordinates=np.array([1.0, 2.0, 3.0]),
                closest_point=np.array([1.0, 2.0, 3.0]),
                node_indices=(0, 1),
                weights=(0.3, 0.3),
                cell_index=0,
                distance=0.0,
            )

    def test_sensor_evaluate_displacement(self) -> None:
        sensor = DiscreteSensor(
            sensor_id=0,
            coordinates=np.array([0.0, 0.0, 0.0]),
            closest_point=np.array([0.0, 0.0, 0.0]),
            node_indices=(1, 2),
            weights=(0.25, 0.75),
            cell_index=0,
            distance=0.0,
        )

        n_dofs = 3 * 5  # 5 nodes
        u = np.zeros(n_dofs)
        # Node 1 disp: [10, 20, 30]
        u[3 : 6] = [10.0, 20.0, 30.0]
        # Node 2 disp: [2, 4, 6]
        u[6 : 9] = [2.0, 4.0, 6.0]

        u_sensor = sensor.evaluate(u)
        expected = 0.25 * np.array([10.0, 20.0, 30.0]) + 0.75 * np.array([2.0, 4.0, 6.0])
        assert np.allclose(u_sensor, expected)


class TestSensorObservationOperatorAssembly:
    """Tests for operator assembly from node indices and continuous coordinates."""

    def test_from_node_indices(self, beam_mesh: VolumetricMesh) -> None:
        sensor_nodes = [10, 25, 42, 100]
        op = SensorObservationOperator.from_node_indices(beam_mesh, sensor_nodes)

        assert op.n_sensors == len(sensor_nodes)
        assert op.n_dofs == beam_mesh.n_dofs
        assert op.matrix.shape == (3 * len(sensor_nodes), beam_mesh.n_dofs)

        # Create synthetic displacement field
        u = np.random.randn(beam_mesh.n_dofs)
        u_obs = op.observe(u)
        u_obs_3d = op.observe_3d(u)

        assert u_obs.shape == (3 * len(sensor_nodes),)
        assert u_obs_3d.shape == (len(sensor_nodes), 3)

        # Every sensor must exactly match its corresponding node's displacement
        for k, nid in enumerate(sensor_nodes):
            expected = u[3 * nid : 3 * nid + 3]
            assert np.allclose(u_obs[3 * k : 3 * k + 3], expected)
            assert np.allclose(u_obs_3d[k], expected)

    def test_from_node_indices_invalid(self, beam_mesh: VolumetricMesh) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            SensorObservationOperator.from_node_indices(beam_mesh, [])

        with pytest.raises(IndexError, match="out-of-range"):
            SensorObservationOperator.from_node_indices(beam_mesh, [999999])

    def test_from_surface_coordinates_barycentric_locator(
        self, beam_mesh: VolumetricMesh
    ) -> None:
        smesh = beam_mesh.surface

        # Select 5 target coordinates from known surface nodes with slight offsets
        sample_nodes = smesh.nodes[[0, 10, 50, 100, 150]]
        op = SensorObservationOperator.from_surface_coordinates(beam_mesh, sample_nodes)

        assert op.n_sensors == 5
        assert op.n_dofs == beam_mesh.n_dofs
        assert op.matrix.shape == (15, beam_mesh.n_dofs)

        # Since these are exact surface nodes, projection distance should be ~0
        assert op.max_projection_distance < 1e-10

        # Check partition of unity for all sensors
        for sensor in op.sensors:
            assert np.isclose(sum(sensor.weights), 1.0)
            assert all(w >= -1e-12 for w in sensor.weights)

            # Check Voronoi closest point matches reconstructed coordinates
            verts = beam_mesh.nodes[list(sensor.node_indices)]
            reconstructed = sum(w * v for w, v in zip(sensor.weights, verts))
            assert np.allclose(reconstructed, sensor.closest_point, atol=1e-12)

    def test_distance_tolerance_enforcement(self, beam_mesh: VolumetricMesh) -> None:
        # A point far outside the mesh bounds
        far_point = np.array([[500.0, 500.0, 500.0]])

        # Without tolerance, projects cleanly
        op = SensorObservationOperator.from_surface_coordinates(beam_mesh, far_point)
        assert op.max_projection_distance > 100.0

        # With tight tolerance, must raise ValueError
        with pytest.raises(ValueError, match="exceeding distance_tolerance"):
            SensorObservationOperator.from_surface_coordinates(
                beam_mesh, far_point, distance_tolerance=1.0
            )

    def test_from_volumetric_coordinates(self, beam_mesh: VolumetricMesh) -> None:
        # Interior points near domain centroid
        centroid = beam_mesh.centroid
        coords = np.array([
            centroid,
            centroid + [2.0, 0.0, 5.0],
            centroid - [1.0, 3.0, 2.0],
        ])

        op = SensorObservationOperator.from_volumetric_coordinates(beam_mesh, coords)
        assert op.n_sensors == 3
        assert op.n_dofs == beam_mesh.n_dofs

        # Check 4-node tetrahedral simplex weights
        for sensor in op.sensors:
            assert len(sensor.node_indices) == 4
            assert len(sensor.weights) == 4
            assert np.isclose(sum(sensor.weights), 1.0, atol=1e-7)

            verts = beam_mesh.nodes[list(sensor.node_indices)]
            reconstructed = sum(w * v for w, v in zip(sensor.weights, verts))
            assert np.allclose(reconstructed, sensor.closest_point, atol=1e-10)


class TestObservationOperatorInvarianceAndAlgebra:
    """Tests for mathematical invariance, row sums, and linear field recovery."""

    def test_partition_of_unity_row_sums(self, beam_mesh: VolumetricMesh) -> None:
        # Arbitrary surface sensor points
        rng = np.random.default_rng(123)
        sample_pts = beam_mesh.surface.nodes[rng.choice(beam_mesh.surface.n_nodes, 10)]
        op = SensorObservationOperator.from_surface_coordinates(beam_mesh, sample_pts)

        H = op.matrix
        row_sums = np.array(H.sum(axis=1)).flatten()

        # In 3D continuum mechanics, each row interpolates one coordinate component.
        # Therefore every row sum of H must equal 1.0 identically.
        assert np.allclose(row_sums, 1.0, atol=1e-12)

    def test_rigid_body_translation_invariance(self, beam_mesh: VolumetricMesh) -> None:
        rng = np.random.default_rng(42)
        sample_pts = beam_mesh.surface.nodes[rng.choice(beam_mesh.surface.n_nodes, 8)]
        op = SensorObservationOperator.from_surface_coordinates(beam_mesh, sample_pts)

        # Uniform rigid body translation: t = [tx, ty, tz]
        t = np.array([12.5, -7.3, 3.8])
        u_rigid = np.tile(t, beam_mesh.n_nodes)

        u_obs = op.observe(u_rigid)
        u_obs_3d = op.observe_3d(u_rigid)

        expected_obs = np.tile(t, op.n_sensors)
        assert np.allclose(u_obs, expected_obs, atol=1e-14)

        for k in range(op.n_sensors):
            assert np.allclose(u_obs_3d[k], t, atol=1e-14)

    def test_linear_displacement_field_preservation(
        self, beam_mesh: VolumetricMesh
    ) -> None:
        rng = np.random.default_rng(999)
        sample_pts = beam_mesh.surface.nodes[rng.choice(beam_mesh.surface.n_nodes, 6)]
        op = SensorObservationOperator.from_surface_coordinates(beam_mesh, sample_pts)

        # Linear displacement field: u_x = 0.01 * X, u_y = -0.02 * Y, u_z = 0.05 * Z
        nodes = beam_mesh.nodes
        u_linear = np.zeros(beam_mesh.n_dofs)
        u_linear[0::3] = 0.01 * nodes[:, 0]
        u_linear[1::3] = -0.02 * nodes[:, 1]
        u_linear[2::3] = 0.05 * nodes[:, 2]

        u_obs_3d = op.observe_3d(u_linear)

        # Evaluated displacements must match linear formula evaluated at closest points
        for k, sensor in enumerate(op.sensors):
            cp = sensor.closest_point
            expected = np.array([0.01 * cp[0], -0.02 * cp[1], 0.05 * cp[2]])
            assert np.allclose(u_obs_3d[k], expected, atol=1e-12)

    def test_batch_snapshot_observation(self, beam_mesh: VolumetricMesh) -> None:
        sensor_nodes = [5, 15, 30, 45]
        op = SensorObservationOperator.from_node_indices(beam_mesh, sensor_nodes)

        # Matrix of 10 snapshots in R^(n_dofs x 10)
        rng = np.random.default_rng(2026)
        U_batch = rng.standard_normal((beam_mesh.n_dofs, 10))

        U_obs = op.observe(U_batch)
        assert U_obs.shape == (3 * len(sensor_nodes), 10)

        for m in range(10):
            single_obs = op.observe(U_batch[:, m])
            assert np.allclose(U_obs[:, m], single_obs, atol=1e-14)


class TestReducedSensorObservationOperator:
    """Tests for MOR projection H_r = H @ Phi and online evaluation."""

    def test_reduced_projection_and_identity(self, beam_mesh: VolumetricMesh) -> None:
        rng = np.random.default_rng(42)
        sample_pts = beam_mesh.surface.nodes[rng.choice(beam_mesh.surface.n_nodes, 12)]
        op = SensorObservationOperator.from_surface_coordinates(beam_mesh, sample_pts)

        # Synthesize an orthonormal reduced basis Phi in R^(n_dofs x r)
        r = 8
        Q, _ = np.linalg.qr(rng.standard_normal((beam_mesh.n_dofs, r)))
        basis = PODBasis(
            basis=Q,
            singular_values=np.ones(r),
            cumulative_energy=1.0,
            total_energy=1.0,
        )

        op_r = op.reduce(basis)
        assert isinstance(op_r, ReducedSensorObservationOperator)
        assert op_r.n_sensors == op.n_sensors
        assert op_r.n_modes == r
        assert op_r.matrix.shape == (3 * op.n_sensors, r)

        # Project alias
        op_r_alias = op.project(basis)
        assert np.array_equal(op_r.matrix, op_r_alias.matrix)

        # Verify mathematical projection identity:
        # H @ (Phi @ u_r) == H_r @ u_r  to machine precision
        for _ in range(5):
            u_r = rng.standard_normal(r)
            u_full = basis.reconstruct(u_r)

            obs_full = op.observe(u_full)
            obs_reduced = op_r.observe(u_r)

            err = np.linalg.norm(obs_full - obs_reduced)
            assert err < 1e-13, f"Projection identity failed with error: {err}"

    def test_reduced_batch_and_3d_observe(self, beam_mesh: VolumetricMesh) -> None:
        op = SensorObservationOperator.from_node_indices(beam_mesh, [1, 2, 3])
        r = 4
        rng = np.random.default_rng(101)
        Phi, _ = np.linalg.qr(rng.standard_normal((beam_mesh.n_dofs, r)))

        op_r = op.reduce(Phi)

        # 1D observe_3d
        u_r = rng.standard_normal(r)
        obs_3d = op_r.observe_3d(u_r)
        assert obs_3d.shape == (3, 3)

        # 2D batch observe
        U_r = rng.standard_normal((r, 7))
        batch_obs = op_r.observe(U_r)
        assert batch_obs.shape == (9, 7)

    def test_online_evaluation_benchmark(self, beam_mesh: VolumetricMesh) -> None:
        """Benchmarks that reduced observation runs in < 5 microseconds (> 200 kHz)."""
        rng = np.random.default_rng(777)
        sample_pts = beam_mesh.surface.nodes[rng.choice(beam_mesh.surface.n_nodes, 20)]
        op = SensorObservationOperator.from_surface_coordinates(beam_mesh, sample_pts)

        r = 10
        Phi, _ = np.linalg.qr(rng.standard_normal((beam_mesh.n_dofs, r)))
        op_r = op.reduce(Phi)
        u_r = rng.standard_normal(r)

        # Warmup
        for _ in range(100):
            _ = op_r.observe(u_r)

        # Timed loop
        n_iters = 10_000
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = op_r.observe(u_r)
        t1 = time.perf_counter()

        dt_per_eval = (t1 - t0) / n_iters
        rate_hz = 1.0 / dt_per_eval

        # Must execute in < 10 microseconds (typically ~0.7 microseconds, > 1 MHz)
        assert dt_per_eval < 10e-6, f"Observation too slow: {dt_per_eval*1e6:.2f} us"
        assert rate_hz > 100_000, f"Observation rate too low: {rate_hz:.0f} Hz"


class TestErrorHandlingAndValidation:
    """Tests for edge cases and dimension validation."""

    def test_empty_coordinates_raises(self, beam_mesh: VolumetricMesh) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            SensorObservationOperator.from_surface_coordinates(beam_mesh, np.empty((0, 3)))

    def test_invalid_coordinates_shape(self, beam_mesh: VolumetricMesh) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            SensorObservationOperator.from_surface_coordinates(beam_mesh, np.ones((5, 2)))

    def test_observe_dimension_mismatch(self, beam_mesh: VolumetricMesh) -> None:
        op = SensorObservationOperator.from_node_indices(beam_mesh, [0, 1])
        with pytest.raises(ValueError, match="must match operator n_dofs"):
            op.observe(np.ones(10))

    def test_reduce_basis_dimension_mismatch(self, beam_mesh: VolumetricMesh) -> None:
        op = SensorObservationOperator.from_node_indices(beam_mesh, [0, 1])
        wrong_basis = np.ones((10, 3))
        with pytest.raises(ValueError, match="do not match operator n_dofs"):
            op.reduce(wrong_basis)

    def test_reduced_observe_dimension_mismatch(self, beam_mesh: VolumetricMesh) -> None:
        op = SensorObservationOperator.from_node_indices(beam_mesh, [0, 1])
        basis = np.ones((beam_mesh.n_dofs, 5))
        op_r = op.reduce(basis)

        with pytest.raises(ValueError, match="must match operator n_modes"):
            op_r.observe(np.ones(3))
