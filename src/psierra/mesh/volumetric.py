from __future__ import annotations
from pathlib import Path
from typing import Literal
from collections.abc import Sequence

import meshio
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from psierra.mesh.surface import SurfaceMesh


class VolumetricMesh:
    """
    3D FEM Volumetric Mesh (Tetrahedral Mesh).

    Represents the 3D solid continuum reference domain Omega_0.
    Directly wraps a PyVista UnstructuredGrid (`ugrid`), leveraging VTK/PyVista
    for geometric processing, spatial queries, quality metrics, and visualization.
    """

    def __init__(self, source: Path | pv.UnstructuredGrid) -> None:
        """
        Parameters:
        -----------
        source : Path or pv.UnstructuredGrid
            File path to a 3D mesh (.vtk, .vtu, .msh, etc.) OR an existing PyVista UnstructuredGrid.
        """
        if isinstance(source, Path):
            self._ugrid = self._read_file(source)
        elif isinstance(source, pv.UnstructuredGrid):
            self._ugrid = source
        else:
            raise TypeError(
                f"VolumetricMesh expects a pathlib.Path or pv.UnstructuredGrid, got {type(source)}."
            )

        has_tet4 = pv.CellType.TETRA in self._ugrid.cells_dict
        has_tet10 = pv.CellType.QUADRATIC_TETRA in self._ugrid.cells_dict

        if not (has_tet4 or has_tet10):
            raise ValueError(
                "Mesh does not contain tetrahedral cells (CellType.TETRA or CellType.QUADRATIC_TETRA)."
            )

        self._cell_type = (
            pv.CellType.QUADRATIC_TETRA if has_tet10 else pv.CellType.TETRA
        )

        # Ensure ugrid contains ONLY 3D tetrahedral solid elements.
        # Files from Gmsh/CAD often bundle 0D points, 1D lines, and 2D triangles.
        if not np.all(self._ugrid.celltypes == self._cell_type):
            self._ugrid = self._ugrid.extract_cells(
                self._ugrid.celltypes == self._cell_type
            )

        # Cached SurfaceMesh
        self._surface: SurfaceMesh | None = None

    # =========================================================================
    # Internal File Reader
    # =========================================================================

    @staticmethod
    def _read_file(path: Path) -> pv.UnstructuredGrid:
        if not path.exists():
            raise FileNotFoundError(f"Mesh file not found: {path}")

        path_str = str(path)
        try:
            grid = pv.read(path_str)
            if isinstance(grid, pv.PolyData):
                raise TypeError(
                    f"File {path} contains a 2D surface PolyData. A 3D volumetric mesh is required."
                )
            return grid
        except Exception:
            try:
                m = meshio.read(path_str)
                return pv.from_meshio(m)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read mesh from {path} with both PyVista and Meshio: {e}"
                )

    # =========================================================================
    # Volumetric Data Queries
    # =========================================================================

    @property
    def ugrid(self) -> pv.UnstructuredGrid:
        """Access underlying PyVista UnstructuredGrid."""
        return self._ugrid

    @property
    def nodes(self) -> np.ndarray:
        """Reference nodal coordinates array of shape (N_nodes, 3)."""
        return np.array(self._ugrid.points, dtype=np.float64)

    @nodes.setter
    def nodes(self, array: np.ndarray) -> None:
        self._ugrid.points = np.asarray(array, dtype=np.float64)
        self._surface = None

    @property
    def cell_type(self) -> pv.CellType:
        """Active tetrahedral cell type (CellType.TETRA or CellType.QUADRATIC_TETRA)."""
        return self._cell_type

    @property
    def order(self) -> int:
        """Element interpolation order: 1 for linear Tet4, 2 for quadratic Tet10."""
        return 2 if self._cell_type == pv.CellType.QUADRATIC_TETRA else 1

    @property
    def nodes_per_element(self) -> int:
        """Number of nodes per element (4 for Tet4, 10 for Tet10)."""
        return 10 if self._cell_type == pv.CellType.QUADRATIC_TETRA else 4

    @property
    def elements(self) -> np.ndarray:
        """Tetrahedral element connectivity array of shape (N_tetra, 4) or (N_tetra, 10)."""
        return np.array(self._ugrid.cells_dict[self._cell_type], dtype=np.int64)

    @property
    def is_linear(self) -> bool:
        """True if mesh consists of linear Tet4 elements (order 1)."""
        return self.order == 1

    @property
    def is_quadratic(self) -> bool:
        """True if mesh consists of quadratic Tet10 elements (order 2)."""
        return self.order == 2

    @property
    def corner_nodes(self) -> np.ndarray:
        """
        Nodal coordinates of the primary element corner vertices.
        For Tet4, this equals self.nodes.
        For Tet10, returns the coordinates of the 4 corner vertices per element.
        """
        if self.is_linear:
            return self.nodes
        corner_indices = np.unique(self.elements[:, :4])
        return self.nodes[corner_indices]

    @property
    def midedge_nodes(self) -> np.ndarray:
        """
        Nodal coordinates of the mid-edge nodes (Tet10 only).
        Returns empty array of shape (0, 3) for linear Tet4 meshes.
        """
        if self.is_linear:
            return np.empty((0, 3), dtype=np.float64)
        midedge_indices = np.unique(self.elements[:, 4:])
        return self.nodes[midedge_indices]

    def to_quadratic(self, reorder: bool = False) -> VolumetricMesh:
        """
        Returns a new VolumetricMesh upgraded to quadratic Tet10 elements.
        Guarantees that all corner vertices, element topology, and volume are preserved,
        adding exactly one mid-edge node per unique edge in canonical VTK ordering.

        Parameters:
        -----------
        reorder : bool, default=False
            If True, automatically applies Reverse Cuthill-McKee (RCM) renumbering
            to minimize sparse matrix bandwidth and maximize cache locality.
        """
        if self.is_quadratic:
            res = self.copy()
        else:
            from psierra.mesher.order import upgrade_tet4_to_tet10
            res = upgrade_tet4_to_tet10(self)  # type: ignore[assignment]

        return res.reorder_nodes("rcm") if reorder else res

    def to_linear(self) -> VolumetricMesh:
        """
        Returns a new VolumetricMesh downgraded to linear Tet4 elements.
        Extracts the 4 corner vertices per tetrahedron.
        """
        if self.is_linear:
            return self.copy()
        from psierra.mesher.order import downgrade_tet10_to_tet4
        return downgrade_tet10_to_tet4(self)  # type: ignore[return-value]

    @property
    def bandwidth(self) -> int:
        """
        Maximum nodal index difference interacting within any single element:
            B = max_e (max_k(e_k) - min_k(e_k))
        Directly governs sparse matrix profile, bandwidth, and cache locality in FEM.
        """
        from psierra.mesher.reorder import compute_mesh_bandwidth
        return compute_mesh_bandwidth(self.elements)

    def reorder_nodes(self, method: Literal["rcm"] = "rcm") -> VolumetricMesh:
        """
        Renumbers mesh nodes using graph ordering algorithms to minimize matrix bandwidth.

        Parameters:
        -----------
        method : {'rcm'}, default='rcm'
            Reverse Cuthill-McKee graph ordering.
        """
        from psierra.mesher.reorder import reorder_mesh_nodes
        return reorder_mesh_nodes(self, method=method)  # type: ignore[return-value]

    @property
    def n_nodes(self) -> int:
        """Total number of volumetric nodes."""
        return self._ugrid.n_points

    @property
    def n_elements(self) -> int:
        """Total number of tetrahedral elements."""
        return self._ugrid.n_cells

    @property
    def n_dofs(self) -> int:
        """Total translational degrees of freedom (3 * n_nodes)."""
        return self.n_nodes * 3

    @property
    def bounds(self) -> np.ndarray:
        """Bounding box: [xmin, xmax, ymin, ymax, zmin, zmax]."""
        return np.array(self._ugrid.bounds)

    # =========================================================================
    # Point Data & Cell Data Access
    # =========================================================================

    @property
    def point_data(self) -> pv.DataSetAttributes:
        """Access or assign nodal field data (dict-like)."""
        return self._ugrid.point_data

    @property
    def cell_data(self) -> pv.DataSetAttributes:
        """Access or assign element field data (dict-like)."""
        return self._ugrid.cell_data

    def __getitem__(self, key: str) -> np.ndarray:
        """Shortcut access to point_data or cell_data by field name: vmesh['Displacement']."""
        return self._ugrid[key]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        """Shortcut assignment to point_data or cell_data: vmesh['VonMises'] = stress."""
        self._ugrid[key] = value

    # =========================================================================
    # Geometric Properties & Quality
    # =========================================================================

    @property
    def volumes(self) -> np.ndarray:
        """Reference volume of each tetrahedral element (N_elems,)."""
        cell_sizes = self._ugrid.compute_cell_sizes()
        return np.array(cell_sizes.cell_data["Volume"], dtype=np.float64)

    @property
    def volume(self) -> float:
        """Total reference volumetric domain volume."""
        return float(np.sum(self.volumes))

    @property
    def centroids(self) -> np.ndarray:
        """Centroid coordinates [X, Y, Z] of every tetrahedron (N_elems, 3)."""
        return np.array(self._ugrid.cell_centers().points, dtype=np.float64)

    @property
    def centroid(self) -> np.ndarray:
        """Geometric center of mass / domain centroid [X, Y, Z]."""
        return np.mean(self.nodes, axis=0)

    def compute_qualities(
        self,
        metric: Literal[
            "scaled_jacobian", "aspect_ratio", "radius_ratio", "collapse_ratio"
        ] = "scaled_jacobian",
    ) -> np.ndarray:
        """
        Computes element mesh qualities for all tetrahedra in compiled C++.

        Parameters:
        -----------
        metric : {'scaled_jacobian', 'aspect_ratio', 'radius_ratio', 'collapse_ratio'}, default='scaled_jacobian'
        """
        if self._cell_type == pv.CellType.QUADRATIC_TETRA:
            # vtkMeshQuality evaluates linear simplices; compute quality on corner vertices
            base_tet4 = pv.UnstructuredGrid(
                {pv.CellType.TETRA: self.elements[:, :4]}, self.nodes
            )
            q_mesh = base_tet4.cell_quality(quality_measure=metric)
            return np.array(q_mesh.cell_data[metric], dtype=np.float64)

        q_mesh = self._ugrid.cell_quality(quality_measure=metric)
        return np.array(q_mesh.cell_data[metric], dtype=np.float64)

    # =========================================================================
    # Nested Boundary Surface Mesh & Boundary Node Mask
    # =========================================================================

    @property
    def surface(self) -> SurfaceMesh:
        """Access the boundary SurfaceMesh object."""
        if self._surface is None:
            surf_polydata = self._ugrid.extract_surface(algorithm=None)
            node_idx = np.array(self._ugrid.surface_indices(), dtype=np.int64)
            self._surface = SurfaceMesh(surf_polydata, node_idx)
        return self._surface

    @property
    def surface_node_mask(self) -> np.ndarray:
        """
        Boolean mask of shape (N_nodes,) indicating which volumetric nodes lie on the boundary.
        mask[i] is True if node i is on the outer surface.
        """
        mask = np.zeros(self.n_nodes, dtype=bool)
        mask[self.surface.node_indices] = True
        return mask

    # =========================================================================
    # Rigid Transformation & Saving
    # =========================================================================

    def copy(self) -> VolumetricMesh:
        """Returns a deep copy of the VolumetricMesh."""
        return VolumetricMesh(self._ugrid.copy(deep=True))

    def transform(
        self,
        translation: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
        orientation: Rotation = Rotation.from_euler("xyz", (0, 0, 0)),
        about_point: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Translates and rotates the mesh in-place using a 4x4 affine transformation.
        Invalidates cached surface structures.
        """
        about_point = np.asarray(about_point, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64)

        # 1. Translate to origin relative to pivot point
        tf1 = np.eye(4)
        tf1[:3, 3] = -about_point

        # 2. Apply rotation and translation to target position
        tf2 = np.eye(4)
        tf2[:3, :3] = orientation.as_matrix()
        tf2[:3, 3] = translation + about_point

        tf = tf2 @ tf1

        # Apply in-place to coordinates (fast, zero duplicate memory)
        self._ugrid.transform(tf, inplace=True)
        self._surface = None  # Invalidate cached surface

    def save(self, filepath: Path) -> None:
        """Saves mesh to disk as .vtu or .vtk (creates parent directories if needed)."""
        if not isinstance(filepath, Path):
            filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._ugrid.save(str(filepath))

    def __repr__(self) -> str:
        cell_desc = "Tet10 (Quadratic)" if self.order == 2 else "Tet4 (Linear)"
        return (
            f"VolumetricMesh [{cell_desc}]"
            f"\n  N Nodes         : {self.n_nodes} nodes"
            f"\n  N DoFs          : {self.n_dofs} DoFs"
            f"\n  N Elements      : {self.n_elements} elements"
            f"\n  X Bounds        : [{self.bounds[0]:.2f}, {self.bounds[1]:.2f}] unit"
            f"\n  Y Bounds        : [{self.bounds[2]:.2f}, {self.bounds[3]:.2f}] unit"
            f"\n  Z Bounds        : [{self.bounds[4]:.2f}, {self.bounds[5]:.2f}] unit"
            f"\n  Domain Centroid : ({self.centroid[0]:.2f}, {self.centroid[1]:.2f}, {self.centroid[2]:.2f}) unit"
            f"\n  Domain Volume   : {self.volume:.2f} unit^3"
            f"\n  Surface Area    : {self.surface.area:.2f} unit^2"
        )
