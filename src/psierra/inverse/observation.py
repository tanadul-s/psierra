"""
Discrete Sensor Observation Operator (H) for Inverse Finite Element Methods (iFEM).

Provides spatial point-in-element barycentric localization and observation operators:
    u_hat = H @ u               (full-order observation in R^(3 * N_sensor))
    u_hat = H_r @ u_r           (reduced-order observation in R^(3 * N_sensor))

Mathematical Formulation:
-------------------------
For a physical domain discretized with n_dof = 3 * N_nodes degrees of freedom,
a discrete sensor k located on a boundary facet with vertices {a_1, a_2, a_3}
interpolates displacement via continuous barycentric coordinates:
    u_hat_k = H_k @ u = sum_{a=1}^3 N_a(xi_k) u_a

Full-order observation matrix H in R^(3 * N_sensor x n_dof):
    H = [ H_0^T, H_1^T, ..., H_{N-1}^T ]^T

Reduced-order observation matrix projected onto modal basis Phi in R^(n_dof x r):
    H_r = H @ Phi in R^(3 * N_sensor x r)

Online evaluation complexity:
    O(N_sensor * r) executed in < 1 microsecond.

References:
-----------
- A. Tessler & H.R. Spangler (2005). "A Variational Principle for Reconstruction
  of Elastic Deformations in Shear Deformable Plates and Shells". Computer Methods
  in Applied Mechanics and Engineering, 194(2-5), 327-347.
- C. Ericson (2004). "Real-Time Collision Detection". CRC Press (Sec. 5.1.5,
  Closest Point on Triangle).
- P. Benner et al. (2015). "Model Order Reduction: Theory, Numerical Methods
  and Applications". De Gruyter.
"""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as spsp
import vtk

from psierra.collision.triangle import TriangleContactGeometry
from psierra.mesh.surface import SurfaceMesh
from psierra.mesh.volumetric import VolumetricMesh
from psierra.reduction.basis import PODBasis


@dataclass(frozen=True)
class DiscreteSensor:
    """
    Metadata and spatial interpolation weights for a single discrete sensor.

    Attributes:
    -----------
    sensor_id : int
        Zero-based index of the sensor.
    coordinates : np.ndarray of shape (3,)
        Physical 3D target coordinates [X, Y, Z] in reference configuration.
    closest_point : np.ndarray of shape (3,)
        Exact projected point on the surface facet or volumetric element.
    node_indices : tuple[int, ...]
        Global volumetric node indices of the facet or element vertices.
    weights : tuple[float, ...]
        Partition-of-unity barycentric interpolation weights (summing to 1.0).
    cell_index : int
        Index of the surface facet (or volumetric element) containing the sensor.
    distance : float
        Euclidean projection distance between target coordinates and closest_point.
    """

    sensor_id: int
    coordinates: np.ndarray
    closest_point: np.ndarray
    node_indices: tuple[int, ...]
    weights: tuple[float, ...]
    cell_index: int
    distance: float

    def __post_init__(self) -> None:
        if self.coordinates.shape != (3,):
            raise ValueError(
                f"Sensor coordinates must have shape (3,), got {self.coordinates.shape}"
            )
        if self.closest_point.shape != (3,):
            raise ValueError(
                f"Closest point coordinates must have shape (3,), got {self.closest_point.shape}"
            )
        if len(self.node_indices) != len(self.weights):
            raise ValueError(
                f"Length of node_indices ({len(self.node_indices)}) does not match "
                f"weights ({len(self.weights)})."
            )
        if not np.isclose(sum(self.weights), 1.0, atol=1e-7):
            raise ValueError(
                f"Sensor barycentric weights must sum to 1.0, got {sum(self.weights)}"
            )

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        """
        Evaluates 3D interpolated displacement vector at this sensor location:
            u_sensor = sum_{a} w_a * u_{node_a} in R^3.

        Parameters:
        -----------
        u : np.ndarray of shape (n_dofs,)
            Full-field displacement vector.

        Returns:
        --------
        u_sensor : np.ndarray of shape (3,)
            Local 3D displacement vector [u_x, u_y, u_z].
        """
        u_sensor = np.zeros(3, dtype=np.float64)
        for node_idx, weight in zip(self.node_indices, self.weights):
            dof_start = 3 * node_idx
            u_sensor += weight * u[dof_start : dof_start + 3]
        return u_sensor


@dataclass(frozen=True)
class ReducedSensorObservationOperator:
    """
    Reduced-order discrete observation operator H_r in R^(3 * N_sensor x r).

    Formed by the modal Galerkin projection of the full-order observation matrix:
        H_r = H @ Phi in R^(3 * N_sensor x r)

    Enables microsecond-rate sensor observation evaluation during online
    digital twin tracking, Kalman filtering, and inverse shape sensing:
        u_hat = H_r @ u_r in R^(3 * N_sensor)
    """

    matrix: np.ndarray
    sensors: tuple[DiscreteSensor, ...]
    n_dofs: int
    basis: PODBasis | np.ndarray

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError(
                f"Reduced observation matrix must be 2D, got shape {self.matrix.shape}"
            )
        expected_rows = 3 * len(self.sensors)
        if self.matrix.shape[0] != expected_rows:
            raise ValueError(
                f"Reduced matrix rows ({self.matrix.shape[0]}) does not match "
                f"3 * n_sensors ({expected_rows})."
            )

    @property
    def n_sensors(self) -> int:
        """Total number of discrete sensors N_sensor."""
        return len(self.sensors)

    @property
    def n_modes(self) -> int:
        """Number of retained reduced modal coordinates r."""
        return self.matrix.shape[1]

    @property
    def observation_matrix(self) -> np.ndarray:
        """Dense observation projection matrix H_r in R^(3 * N_sensor x r)."""
        return self.matrix

    def observe(self, u_r: np.ndarray) -> np.ndarray:
        """
        Computes observed sensor displacements from reduced generalized coordinates:
            u_hat = H_r @ u_r in R^(3 * N_sensor)

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,) or (r, M)
            Generalized coordinates vector or batch trajectory matrix.

        Returns:
        --------
        u_hat : np.ndarray of shape (3 * N_sensor,) or (3 * N_sensor, M)
        """
        u_r = np.asarray(u_r, dtype=np.float64)
        if u_r.ndim == 1:
            if len(u_r) != self.n_modes:
                raise ValueError(
                    f"Reduced state vector length ({len(u_r)}) must match "
                    f"operator n_modes ({self.n_modes})."
                )
            return self.matrix @ u_r
        elif u_r.ndim == 2:
            if u_r.shape[0] != self.n_modes:
                raise ValueError(
                    f"Reduced state rows ({u_r.shape[0]}) must match "
                    f"operator n_modes ({self.n_modes})."
                )
            return self.matrix @ u_r
        else:
            raise ValueError(
                f"Reduced state must be 1D or 2D array, got ndim={u_r.ndim}."
            )

    def observe_3d(self, u_r: np.ndarray) -> np.ndarray:
        """
        Computes observed sensor displacements formatted as 3D vectors:
            U_hat in R^(N_sensor, 3) or R^(N_sensor, 3, M).

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,) or (r, M)

        Returns:
        --------
        u_hat_3d : np.ndarray of shape (N_sensor, 3) or (N_sensor, 3, M)
        """
        flat = self.observe(u_r)
        if flat.ndim == 1:
            return flat.reshape(self.n_sensors, 3)
        return flat.reshape(self.n_sensors, 3, -1)


class SensorObservationOperator:
    """
    Full-Order Discrete Sensor Observation Operator H in R^(3 * N_sensor x n_dof).

    Maps the full-field structural displacement vector u in R^n_dof to discrete
    measurements u_hat in R^(3 * N_sensor) via spatial barycentric interpolation
    over boundary surface facets or volumetric elements:
        u_hat = H @ u

    Attributes:
    -----------
    sensors : tuple[DiscreteSensor, ...]
        Immutable collection of located discrete sensors.
    n_dofs : int
        Dimension of full-order displacement vector (3 * N_nodes).
    matrix : scipy.sparse.csr_matrix
        Sparse observation operator matrix of shape (3 * N_sensor, n_dofs).
    """

    def __init__(
        self,
        mesh_or_sensors: VolumetricMesh | SurfaceMesh | Sequence[DiscreteSensor] | Any,
        coordinates_or_ndofs: np.ndarray | Sequence[Sequence[float]] | int | None = None,
        distance_tolerance: float | None = None,
    ) -> None:
        """
        Instantiates SensorObservationOperator.

        Supports two construction patterns:
        1. Direct instantiation:
           SensorObservationOperator(sensors: Sequence[DiscreteSensor], n_dofs: int)
        2. Convenience mesh-and-coordinates instantiation:
           SensorObservationOperator(mesh, coordinates, distance_tolerance=None)

        For advanced or explicit construction, prefer classmethods:
        - from_surface_coordinates(...)
        - from_node_indices(...)
        - from_volumetric_coordinates(...)
        - from_sensors(...)
        """
        if isinstance(coordinates_or_ndofs, int):
            # Direct pattern: (sensors, n_dofs)
            sensors = tuple(mesh_or_sensors)
            n_dofs = int(coordinates_or_ndofs)
            self._init_from_sensors(sensors, n_dofs)
        else:
            # Convenience pattern: (mesh, coordinates)
            if coordinates_or_ndofs is None:
                raise ValueError(
                    "SensorObservationOperator requires either (sensors, n_dofs) or (mesh, coordinates)."
                )
            op = self.from_surface_coordinates(
                mesh=mesh_or_sensors,
                coordinates=coordinates_or_ndofs,
                distance_tolerance=distance_tolerance,
            )
            self._sensors = op.sensors
            self._n_dofs = op.n_dofs
            self._matrix = op.matrix
            self._sensor_coordinates = op.sensor_coordinates
            self._closest_points = op.closest_points
            self._distances = op.projection_distances

    def _init_from_sensors(
        self, sensors: Sequence[DiscreteSensor], n_dofs: int
    ) -> None:
        if len(sensors) == 0:
            raise ValueError("Cannot construct SensorObservationOperator with zero sensors.")
        if n_dofs <= 0:
            raise ValueError(f"n_dofs must be positive, got {n_dofs}.")

        self._sensors = tuple(sensors)
        self._n_dofs = n_dofs
        self._matrix = self._assemble_sparse_matrix(self._sensors, self._n_dofs)

        coords = [s.coordinates for s in self._sensors]
        pts = [s.closest_point for s in self._sensors]
        dists = [s.distance for s in self._sensors]

        self._sensor_coordinates = np.array(coords, dtype=np.float64)
        self._closest_points = np.array(pts, dtype=np.float64)
        self._distances = np.array(dists, dtype=np.float64)

    @classmethod
    def from_sensors(
        cls, sensors: Sequence[DiscreteSensor], n_dofs: int
    ) -> SensorObservationOperator:
        """Constructs an operator from pre-computed DiscreteSensor objects."""
        return cls(sensors, n_dofs)

    @classmethod
    def from_surface_coordinates(
        cls,
        mesh: VolumetricMesh | SurfaceMesh | Any,
        coordinates: np.ndarray | Sequence[Sequence[float]],
        distance_tolerance: float | None = None,
    ) -> SensorObservationOperator:
        """
        Locates arbitrary 3D target coordinates on the boundary surface mesh and
        constructs the observation operator H.

        Parameters:
        -----------
        mesh : VolumetricMesh, SurfaceMesh, or SolidDomain
            Structural domain or mesh containing the boundary geometry.
        coordinates : np.ndarray of shape (N_sensor, 3)
            Physical 3D target coordinates in reference space.
        distance_tolerance : float, optional
            Maximum allowable distance from a sensor coordinate to the mesh surface.
            Raises ValueError if any sensor exceeds this tolerance.

        Returns:
        --------
        operator : SensorObservationOperator
        """
        # Resolve mesh and surface mesh
        vmesh, smesh, n_dofs = cls._resolve_mesh_and_surface(mesh)

        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"Sensor coordinates must have shape (N_sensor, 3), got {coords.shape}"
            )
        if len(coords) == 0:
            raise ValueError("Sensor coordinates cannot be empty.")

        # Build VTK Static Cell Locator on surface triangulation
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(smesh.polydata)
        locator.BuildLocator()

        sensors: list[DiscreteSensor] = []
        closest = np.zeros(3, dtype=np.float64)
        cell_id = vtk.mutable(0)
        sub_id = vtk.mutable(0)
        dist2 = vtk.mutable(0.0)

        for k, pt in enumerate(coords):
            locator.FindClosestPoint(pt, closest, cell_id, sub_id, dist2)
            cid = int(cell_id)
            distance = float(np.sqrt(float(dist2)))

            if distance_tolerance is not None and distance > distance_tolerance:
                raise ValueError(
                    f"Sensor {k} at {pt} is {distance:.4e} from surface, "
                    f"exceeding distance_tolerance={distance_tolerance:.4e}."
                )

            # Retrieve facet nodes
            face_local_nodes = smesh.elements[cid]  # length 3 (or 6)
            if len(face_local_nodes) > 3:
                # Use linear corner vertices for simplex barycentrics
                face_local_nodes = face_local_nodes[:3]

            vertices = smesh.nodes[face_local_nodes]  # (3, 3)

            # Compute exact non-negative Voronoi barycentric weights
            weights_arr = TriangleContactGeometry.closest_barycentric(closest, vertices)
            weights = (float(weights_arr[0]), float(weights_arr[1]), float(weights_arr[2]))

            # Map local surface node indices to global volumetric node indices
            if smesh.node_indices is not None:
                global_nodes = tuple(int(smesh.node_indices[i]) for i in face_local_nodes)
            else:
                global_nodes = tuple(int(i) for i in face_local_nodes)

            sensors.append(
                DiscreteSensor(
                    sensor_id=k,
                    coordinates=pt.copy(),
                    closest_point=closest.copy(),
                    node_indices=global_nodes,
                    weights=weights,
                    cell_index=cid,
                    distance=distance,
                )
            )

        return cls.from_sensors(sensors, n_dofs)

    @classmethod
    def from_node_indices(
        cls,
        mesh: VolumetricMesh | SurfaceMesh | Any,
        node_indices: Sequence[int] | np.ndarray,
    ) -> SensorObservationOperator:
        """
        Constructs an observation operator for sensors placed directly at known mesh nodes.

        Parameters:
        -----------
        mesh : VolumetricMesh, SurfaceMesh, or SolidDomain
        node_indices : Sequence[int] or np.ndarray of shape (N_sensor,)
            Global node indices where discrete measurements are observed.

        Returns:
        --------
        operator : SensorObservationOperator
        """
        vmesh, smesh, n_dofs = cls._resolve_mesh_and_surface(mesh)
        nodes_ref = vmesh.nodes if vmesh is not None else smesh.nodes

        indices = np.asarray(node_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(f"node_indices must be a 1D sequence, got {indices.shape}")
        if len(indices) == 0:
            raise ValueError("node_indices cannot be empty.")

        n_total_nodes = len(nodes_ref)
        if np.any(indices < 0) or np.any(indices >= n_total_nodes):
            invalid = indices[(indices < 0) | (indices >= n_total_nodes)]
            raise IndexError(
                f"node_indices contains out-of-range indices: {invalid} (valid: [0, {n_total_nodes - 1}])"
            )

        sensors: list[DiscreteSensor] = []
        for k, node_id in enumerate(indices):
            nid = int(node_id)
            coord = nodes_ref[nid].copy()
            sensors.append(
                DiscreteSensor(
                    sensor_id=k,
                    coordinates=coord.copy(),
                    closest_point=coord.copy(),
                    node_indices=(nid,),
                    weights=(1.0,),
                    cell_index=-1,
                    distance=0.0,
                )
            )

        return cls.from_sensors(sensors, n_dofs)

    @classmethod
    def from_volumetric_coordinates(
        cls,
        mesh: VolumetricMesh | Any,
        coordinates: np.ndarray | Sequence[Sequence[float]],
    ) -> SensorObservationOperator:
        """
        Locates arbitrary 3D target coordinates inside a 3D volumetric tetrahedral mesh
        and constructs the observation operator H.

        Parameters:
        -----------
        mesh : VolumetricMesh or SolidDomain
        coordinates : np.ndarray of shape (N_sensor, 3)

        Returns:
        --------
        operator : SensorObservationOperator
        """
        if hasattr(mesh, "mesh"):
            mesh = mesh.mesh
        if not isinstance(mesh, VolumetricMesh):
            raise TypeError(
                f"from_volumetric_coordinates requires a VolumetricMesh, got {type(mesh)}."
            )

        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"Sensor coordinates must have shape (N_sensor, 3), got {coords.shape}"
            )
        if len(coords) == 0:
            raise ValueError("Sensor coordinates cannot be empty.")

        n_dofs = mesh.n_dofs

        # Locator on volumetric tetrahedral grid
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(mesh.ugrid)
        locator.BuildLocator()

        sensors: list[DiscreteSensor] = []
        closest = np.zeros(3, dtype=np.float64)
        cell_id = vtk.mutable(0)
        sub_id = vtk.mutable(0)
        dist2 = vtk.mutable(0.0)

        for k, pt in enumerate(coords):
            locator.FindClosestPoint(pt, closest, cell_id, sub_id, dist2)
            cid = int(cell_id)
            distance = float(np.sqrt(float(dist2)))

            cell = mesh.ugrid.get_cell(cid)
            pt_ids = list(cell.point_ids)[:4]
            verts = mesh.nodes[pt_ids]  # (4, 3)

            # Linear simplex barycentric coordinate solve
            J = (verts[1:] - verts[0]).T  # 3x3
            w123 = np.linalg.solve(J, closest - verts[0])
            w0 = 1.0 - float(np.sum(w123))
            weights = (w0, float(w123[0]), float(w123[1]), float(w123[2]))

            sensors.append(
                DiscreteSensor(
                    sensor_id=k,
                    coordinates=pt.copy(),
                    closest_point=closest.copy(),
                    node_indices=tuple(pt_ids),
                    weights=weights,
                    cell_index=cid,
                    distance=distance,
                )
            )

        return cls.from_sensors(sensors, n_dofs)

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _resolve_mesh_and_surface(
        mesh: Any,
    ) -> tuple[VolumetricMesh | None, SurfaceMesh, int]:
        """Resolves mesh, boundary SurfaceMesh, and total model DOFs."""
        if hasattr(mesh, "mesh"):
            mesh = mesh.mesh

        if isinstance(mesh, VolumetricMesh):
            return mesh, mesh.surface, mesh.n_dofs
        elif isinstance(mesh, SurfaceMesh):
            if mesh.node_indices is not None:
                n_dofs = 3 * (int(np.max(mesh.node_indices)) + 1)
            else:
                n_dofs = mesh.n_dofs
            return None, mesh, n_dofs
        else:
            raise TypeError(
                f"Expected VolumetricMesh, SurfaceMesh, or SolidDomain, got {type(mesh)}."
            )

    @staticmethod
    def _assemble_sparse_matrix(
        sensors: Sequence[DiscreteSensor], n_dofs: int
    ) -> spsp.csr_matrix:
        """Assembles sparse observation matrix H in R^(3 * N_sensor x n_dofs)."""
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for k, sensor in enumerate(sensors):
            row_base = 3 * k
            for node_idx, weight in zip(sensor.node_indices, sensor.weights):
                col_base = 3 * node_idx
                for d in range(3):
                    rows.append(row_base + d)
                    cols.append(col_base + d)
                    vals.append(weight)

        n_rows = 3 * len(sensors)
        coo = spsp.coo_matrix(
            (vals, (rows, cols)),
            shape=(n_rows, n_dofs),
            dtype=np.float64,
        )
        return coo.tocsr()

    # =========================================================================
    # Public Properties
    # =========================================================================

    @property
    def sensors(self) -> tuple[DiscreteSensor, ...]:
        """Tuple of discrete sensors."""
        return self._sensors

    @property
    def n_sensors(self) -> int:
        """Total number of discrete sensors N_sensor."""
        return len(self._sensors)

    @property
    def n_dofs(self) -> int:
        """Dimension of full-order displacement vector n_dofs = 3 * N_nodes."""
        return self._n_dofs

    @property
    def matrix(self) -> spsp.csr_matrix:
        """Sparse observation matrix H in R^(3 * N_sensor x n_dofs)."""
        return self._matrix

    @property
    def observation_matrix(self) -> spsp.csr_matrix:
        """Alias for matrix."""
        return self._matrix

    @property
    def sensor_coordinates(self) -> np.ndarray:
        """Target sensor coordinates array of shape (N_sensor, 3)."""
        return self._sensor_coordinates

    @property
    def closest_points(self) -> np.ndarray:
        """Projected sensor surface points array of shape (N_sensor, 3)."""
        return self._closest_points

    @property
    def projection_distances(self) -> np.ndarray:
        """Distances between target coordinates and projected surface points (N_sensor,)."""
        return self._distances

    @property
    def max_projection_distance(self) -> float:
        """Maximum projection distance among all sensors."""
        return float(np.max(self._distances)) if len(self._distances) > 0 else 0.0

    # =========================================================================
    # Observation Methods
    # =========================================================================

    def observe(self, u: np.ndarray) -> np.ndarray:
        """
        Computes observed sensor displacements from full-order state vector:
            u_hat = H @ u in R^(3 * N_sensor)

        Parameters:
        -----------
        u : np.ndarray of shape (n_dofs,) or (n_dofs, M)
            Full-field displacement vector or snapshot collection.

        Returns:
        --------
        u_hat : np.ndarray of shape (3 * N_sensor,) or (3 * N_sensor, M)
        """
        u = np.asarray(u, dtype=np.float64)
        if u.ndim == 1:
            if len(u) != self.n_dofs:
                raise ValueError(
                    f"Displacement vector length ({len(u)}) must match "
                    f"operator n_dofs ({self.n_dofs})."
                )
            return self.matrix @ u
        elif u.ndim == 2:
            if u.shape[0] != self.n_dofs:
                raise ValueError(
                    f"Displacement rows ({u.shape[0]}) must match "
                    f"operator n_dofs ({self.n_dofs})."
                )
            return self.matrix @ u
        else:
            raise ValueError(
                f"Displacement must be 1D or 2D array, got ndim={u.ndim}."
            )

    def observe_3d(self, u: np.ndarray) -> np.ndarray:
        """
        Computes observed sensor displacements formatted as 3D vectors:
            U_hat in R^(N_sensor, 3) or R^(N_sensor, 3, M).

        Parameters:
        -----------
        u : np.ndarray of shape (n_dofs,) or (n_dofs, M)

        Returns:
        --------
        u_hat_3d : np.ndarray of shape (N_sensor, 3) or (N_sensor, 3, M)
        """
        flat = self.observe(u)
        if flat.ndim == 1:
            return flat.reshape(self.n_sensors, 3)
        return flat.reshape(self.n_sensors, 3, -1)

    # =========================================================================
    # Model Order Reduction Projection
    # =========================================================================

    def reduce(
        self, basis: PODBasis | np.ndarray
    ) -> ReducedSensorObservationOperator:
        """
        Projects full-order observation matrix H onto reduced basis Phi:
            H_r = H @ Phi in R^(3 * N_sensor x r)

        Parameters:
        -----------
        basis : PODBasis or np.ndarray of shape (n_dofs, r)
            Proper Orthogonal Decomposition modal basis matrix.

        Returns:
        --------
        reduced_operator : ReducedSensorObservationOperator
        """
        if isinstance(basis, PODBasis):
            Phi = basis.basis
        elif isinstance(basis, np.ndarray):
            Phi = basis
        else:
            raise TypeError(
                f"basis must be PODBasis or np.ndarray, got {type(basis)}."
            )

        if Phi.ndim != 2:
            raise ValueError(
                f"Basis matrix must be 2D of shape (n_dofs, r), got {Phi.shape}."
            )
        if Phi.shape[0] != self.n_dofs:
            raise ValueError(
                f"Basis rows ({Phi.shape[0]}) do not match operator n_dofs ({self.n_dofs})."
            )

        # Sparse-dense matrix product: (3 * N_sensor, n_dofs) @ (n_dofs, r) -> (3 * N_sensor, r)
        H_r = self.matrix @ Phi

        return ReducedSensorObservationOperator(
            matrix=np.ascontiguousarray(H_r, dtype=np.float64),
            sensors=self.sensors,
            n_dofs=self.n_dofs,
            basis=basis,
        )

    def project(
        self, basis: PODBasis | np.ndarray
    ) -> ReducedSensorObservationOperator:
        """Alias for reduce(basis)."""
        return self.reduce(basis)
