from __future__ import annotations
from pathlib import Path
from typing import Literal
from collections.abc import Sequence

import meshio
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation


class SurfaceMesh:
    """
    2D Boundary Surface Mesh (Triangular Mesh).

    Represents either the boundary skin of a VolumetricMesh or an independent
    2D surface mesh (e.g. rigid skull STL, surgical tool, floor boundary).
    """

    def __init__(
        self,
        source: Path | pv.PolyData,
        node_indices: np.ndarray | None = None,
    ) -> None:
        """
        Parameters:
        -----------
        source : Path or pv.PolyData
            File path to a surface mesh (.stl, .ply, .obj, .vtp) OR an existing PyVista PolyData.
        node_indices : np.ndarray, optional
            Global volumetric node indices corresponding to each surface node:
            vmesh.nodes[smesh.node_indices] == smesh.nodes.
        """
        if isinstance(source, Path):
            self._polydata = self._read_file(source)
        elif isinstance(source, pv.PolyData):
            self._polydata = source
        else:
            raise TypeError(
                f"SurfaceMesh expects a pathlib.Path or pv.PolyData, got {type(source)}."
            )

        # Ensure surface is fully triangulated
        if not self._polydata.is_all_triangles:
            self._polydata = self._polydata.triangulate()

        self._node_indices = (
            np.asarray(node_indices, dtype=np.int64)
            if node_indices is not None
            else None
        )

    # =========================================================================
    # Internal File Reader
    # =========================================================================

    @staticmethod
    def _read_file(path: Path) -> pv.PolyData:
        if not path.exists():
            raise FileNotFoundError(f"Surface mesh file not found: {path}")

        path_str = str(path)
        try:
            pdata = pv.read(path_str)
            if isinstance(pdata, pv.UnstructuredGrid):
                pdata = pdata.extract_surface(algorithm=None)
            return pdata
        except Exception:
            try:
                m = meshio.read(path_str)
                grid = pv.from_meshio(m)
                return (
                    grid.extract_surface(algorithm=None)
                    if isinstance(grid, pv.UnstructuredGrid)
                    else grid
                )
            except Exception as e:
                raise RuntimeError(f"Failed to read surface mesh from {path}: {e}")

    # =========================================================================
    # Data Queries
    # =========================================================================

    @property
    def polydata(self) -> pv.PolyData:
        """Access underlying PyVista PolyData object."""
        return self._polydata

    @property
    def nodes(self) -> np.ndarray:
        """Surface nodal coordinates array of shape (N_surf_nodes, 3)."""
        return np.array(self._polydata.points, dtype=np.float64)

    @nodes.setter
    def nodes(self, array: np.ndarray) -> None:
        self._polydata.points = np.asarray(array, dtype=np.float64)

    @property
    def elements(self) -> np.ndarray:
        """Triangular element connectivity array of shape (N_triangles, 3)."""
        return np.array(self._polydata.regular_faces, dtype=np.int64)

    @property
    def node_indices(self) -> np.ndarray | None:
        """
        Global node indices in parent VolumetricMesh (None if independent surface).
        vmesh.nodes[smesh.node_indices] == smesh.nodes
        """
        return self._node_indices

    @property
    def n_nodes(self) -> int:
        """Total number of surface nodes."""
        return self._polydata.n_points

    @property
    def n_elements(self) -> int:
        """Total number of surface triangular elements."""
        return self.elements.shape[0]

    @property
    def n_dofs(self) -> int:
        """Total translational degrees of freedom for the surface (3 * n_nodes)."""
        return self.n_nodes * 3

    @property
    def bounds(self) -> np.ndarray:
        """Bounding box: [xmin, xmax, ymin, ymax, zmin, zmax]."""
        return np.array(self._polydata.bounds)

    @property
    def triangle_mask(self) -> np.ndarray:
        """Boolean mask of cells in polydata that are 2D triangles."""
        return self._polydata.celltypes == pv.CellType.TRIANGLE

    # =========================================================================
    # Point Data & Cell Data Access
    # =========================================================================

    @property
    def point_data(self) -> pv.DataSetAttributes:
        """Access or assign nodal field data (dict-like)."""
        return self._polydata.point_data

    @property
    def cell_data(self) -> pv.DataSetAttributes:
        """Access or assign element field data (dict-like)."""
        return self._polydata.cell_data

    def __getitem__(self, key: str) -> np.ndarray:
        """Shortcut access to point_data or cell_data by field name: smesh['Traction']."""
        return self._polydata[key]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        """Shortcut assignment to point_data or cell_data: smesh['Contact_Pressure'] = p."""
        self._polydata[key] = value

    # =========================================================================
    # Geometric Properties & Quality
    # =========================================================================

    @property
    def areas(self) -> np.ndarray:
        """Area of each surface triangle (N_triangles,)."""
        cell_sizes = self._polydata.compute_cell_sizes()
        return np.array(cell_sizes.cell_data["Area"], dtype=np.float64)

    @property
    def area(self) -> float:
        """Total surface area."""
        return float(np.sum(self.areas))

    @property
    def centroids(self) -> np.ndarray:
        """Centroid coordinates [X, Y, Z] of each surface triangle (N_triangles, 3)."""
        return np.array(self._polydata.cell_centers().points, dtype=np.float64)

    @property
    def centroid(self) -> np.ndarray:
        """Geometric center of the entire surface mesh [X, Y, Z]."""
        return np.mean(self.nodes, axis=0)

    @property
    def normals(self) -> np.ndarray:
        """Outward unit normal vectors for each triangle face (N_triangles, 3)."""
        pdata_normals = self._polydata.compute_normals(
            cell_normals=True, point_normals=False
        )
        return np.array(pdata_normals.cell_data["Normals"], dtype=np.float64)

    @property
    def nodal_normals(self) -> np.ndarray:
        """Outward unit normal vectors at each surface node (N_surf_nodes, 3)."""
        pdata_normals = self._polydata.compute_normals(
            cell_normals=False, point_normals=True
        )
        return np.array(pdata_normals.point_data["Normals"], dtype=np.float64)

    def compute_qualities(
        self,
        metric: Literal[
            "aspect_ratio", "min_angle", "max_angle", "radius_ratio"
        ] = "aspect_ratio",
    ) -> np.ndarray:
        """
        Computes element mesh qualities for all triangles.

        Parameters:
        -----------
        metric : {'aspect_ratio', 'min_angle', 'max_angle', 'radius_ratio'}, default='aspect_ratio'
        """
        q_mesh = self._polydata.cell_quality(quality_measure=metric)
        return np.array(q_mesh.cell_data[metric], dtype=np.float64)

    # =========================================================================
    # Copy, Rigid Transformation, & Saving
    # =========================================================================

    def copy(self) -> SurfaceMesh:
        """Returns a deep copy of the SurfaceMesh."""
        return SurfaceMesh(
            self._polydata.copy(deep=True),
            node_indices=(
                self._node_indices.copy() if self._node_indices is not None else None
            ),
        )

    def transform(
        self,
        translation: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
        orientation: Rotation = Rotation.from_euler("xyz", (0, 0, 0)),
        about_point: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Translates and rotates the surface mesh in-place using a 4x4 affine transformation.
        """
        about_point = np.asarray(about_point, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64)

        tf1 = np.eye(4)
        tf1[:3, 3] = -about_point

        tf2 = np.eye(4)
        tf2[:3, :3] = orientation.as_matrix()
        tf2[:3, 3] = translation + about_point

        tf = tf2 @ tf1

        self._polydata.transform(tf, inplace=True)

    def save(self, filepath: Path) -> None:
        """Saves surface mesh to disk (.stl, .ply, .vtp), creating parent directories if needed."""
        if not isinstance(filepath, Path):
            filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._polydata.save(str(filepath))

    def __repr__(self) -> str:
        return (
            f"SurfaceMesh"
            f"\n  N Nodes         : {self.n_nodes} nodes"
            f"\n  N DoFs          : {self.n_dofs} DoFs"
            f"\n  N Elements      : {self.n_elements} elements"
            f"\n  X Bounds        : [{self.bounds[0]:.2f}, {self.bounds[1]:.2f}] unit"
            f"\n  Y Bounds        : [{self.bounds[2]:.2f}, {self.bounds[3]:.2f}] unit"
            f"\n  Z Bounds        : [{self.bounds[4]:.2f}, {self.bounds[5]:.2f}] unit"
            f"\n  Domain Centroid : ({self.centroid[0]:.2f}, {self.centroid[1]:.2f}, {self.centroid[2]:.2f}) unit"
            f"\n  Surface Area    : {self.area:.2f} unit^2"
        )