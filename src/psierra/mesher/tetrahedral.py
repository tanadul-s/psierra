from __future__ import annotations
from pathlib import Path
from typing import Literal

import numpy as np
import pytetwild
import pyvista as pv

from psierra.mesher.cad import read_step
from psierra.mesher.io import read_surface, export_mesh, import_mesh
from psierra.mesher.order import upgrade_tet4_to_tet10
from psierra.mesh.surface import SurfaceMesh
from psierra.mesh.volumetric import VolumetricMesh


class TetrahedralMesher:
    """
    Robust 3D Tetrahedral Mesher for Complex Organic and CAD Geometries.

    Leverages envelope-constrained tetrahedralization (PyTetWild) to generate watertight,
    strictly positive-Jacobian tetrahedral meshes from noisy surfaces (STL, STEP/CAD, OBJ).
    Supports generating both linear (Tet4) and quadratic (Tet10) volumetric meshes with
    seamless PyVista and Meshio roundtrip capability.
    """

    def __init__(
        self,
        order: Literal[1, 2] = 1,
        edge_length_fac: float = 0.05,
        edge_length_abs: float | None = None,
        optimize: bool = True,
        simplify: bool = True,
        epsilon: float = 0.001,
        stop_energy: float = 10.0,
        coarsen: bool = False,
        num_threads: int = 0,
        num_opt_iter: int = 80,
        quiet: bool = True,
        reorder: bool = False,
    ) -> None:
        """
        Parameters:
        -----------
        order : {1, 2}, default=1
            Interpolation order: 1 for linear Tet4 elements, 2 for quadratic Tet10 elements.
        edge_length_fac : float, default=0.05
            Target edge length as a fraction of the bounding box diagonal.
        edge_length_abs : float, optional
            Target edge length in absolute geometric coordinates (overrides edge_length_fac).
        optimize : bool, default=True
            Improve element Jacobians using AMIPS continuous energy optimization.
        simplify : bool, default=True
            Simplify input surface geometry before tetrahedralization.
        epsilon : float, default=0.001
            Envelope tolerance size relative to bounding box.
        stop_energy : float, default=10.0
            Stopping threshold for mesh optimization energy.
        coarsen : bool, default=False
            Coarsen the interior of the mesh while maintaining boundary quality.
        num_threads : int, default=0
            Number of CPU threads to use (0 = all available cores).
        num_opt_iter : int, default=80
            Maximum number of optimization iterations.
        quiet : bool, default=True
            Suppress PyTetWild console output.
        reorder : bool, default=False
            If True, applies Reverse Cuthill-McKee (RCM) node renumbering to minimize
            matrix bandwidth and profile for faster sparse solver execution.
        """
        if order not in (1, 2):
            raise ValueError(f"Order must be 1 (Tet4) or 2 (Tet10), got {order}.")

        self.order = order
        self.edge_length_fac = edge_length_fac
        self.edge_length_abs = edge_length_abs
        self.optimize = optimize
        self.simplify = simplify
        self.epsilon = epsilon
        self.stop_energy = stop_energy
        self.coarsen = coarsen
        self.num_threads = num_threads
        self.num_opt_iter = num_opt_iter
        self.quiet = quiet
        self.reorder = reorder

    def generate(
        self,
        source: (
            Path
            | str
            | pv.PolyData
            | SurfaceMesh
            | pv.UnstructuredGrid
            | VolumetricMesh
        ),
    ) -> VolumetricMesh:
        """
        Executes the tetrahedral generation pipeline on the provided geometry source.

        If source is already a 3D volumetric mesh (VolumetricMesh or pv.UnstructuredGrid),
        it directly upgrades/downgrades element order without re-tetrahedralizing from scratch.

        If source is a surface (STL, STEP, PolyData, SurfaceMesh), it runs envelope-constrained
        tetrahedralization once via PyTetWild, and then upgrades to Tet10 if order=2.

        Parameters:
        -----------
        source : Path, str, pv.PolyData, SurfaceMesh, pv.UnstructuredGrid, or VolumetricMesh
            Input geometric representation. Supports *.stl, *.step, *.stp, *.ply, *.obj,
            or existing 3D mesh objects.

        Returns:
        --------
        VolumetricMesh
            3D continuum volumetric mesh containing either Tet4 or Tet10 elements.
        """
        # 1. Handle existing 3D volumetric mesh directly (DO NOT re-tetrahedralize)
        if isinstance(source, VolumetricMesh):
            res = source.to_quadratic(reorder=self.reorder) if self.order == 2 else source.to_linear()
            return res.reorder_nodes("rcm") if (self.order == 1 and self.reorder) else res

        if isinstance(source, pv.UnstructuredGrid):
            has_tet4 = pv.CellType.TETRA in source.cells_dict
            has_tet10 = pv.CellType.QUADRATIC_TETRA in source.cells_dict
            if has_tet4 or has_tet10:
                vmesh = VolumetricMesh(source)
                res = vmesh.to_quadratic(reorder=self.reorder) if self.order == 2 else vmesh.to_linear()
                return res.reorder_nodes("rcm") if (self.order == 1 and self.reorder) else res

        # Check if file path points to an existing 3D volumetric mesh file (.vtu, .vtk)
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() in [".vtu", ".vtk"]:
                try:
                    loaded = import_mesh(path)
                    has_tet4 = pv.CellType.TETRA in loaded.cells_dict
                    has_tet10 = pv.CellType.QUADRATIC_TETRA in loaded.cells_dict
                    if has_tet4 or has_tet10:
                        vmesh = VolumetricMesh(loaded)
                        res = vmesh.to_quadratic(reorder=self.reorder) if self.order == 2 else vmesh.to_linear()
                        return res.reorder_nodes("rcm") if (self.order == 1 and self.reorder) else res
                except Exception:
                    pass

        # 2. Source is a 2D surface geometry: generate Tet4 via PyTetWild
        surface = self._extract_surface(source)

        ugrid_tet4 = pytetwild.tetrahedralize_pv(
            surface,
            edge_length_fac=self.edge_length_fac,
            edge_length_abs=self.edge_length_abs,
            optimize=self.optimize,
            simplify=self.simplify,
            epsilon=self.epsilon,
            stop_energy=self.stop_energy,
            coarsen=self.coarsen,
            num_threads=self.num_threads,
            num_opt_iter=self.num_opt_iter,
            quiet=self.quiet,
        )

        vmesh = VolumetricMesh(ugrid_tet4)

        # Clean up temporary file created by fTetWild in working directory
        stray_stl = Path("__tracked_surface.stl")
        if stray_stl.exists():
            try:
                stray_stl.unlink()
            except OSError:
                pass

        # 3. Upgrade to Tet10 if order=2 (preserves exact same corner nodes!)
        if self.order == 2:
            return vmesh.to_quadratic(reorder=self.reorder)

        return vmesh.reorder_nodes("rcm") if self.reorder else vmesh

    def upgrade(
        self,
        mesh: VolumetricMesh | pv.UnstructuredGrid,
        reorder: bool | None = None,
    ) -> VolumetricMesh:
        """
        Upgrades an existing linear tetrahedral mesh (Tet4) to quadratic (Tet10).
        Guarantees that all corner vertices, edge connections, and volume are preserved,
        adding exactly one mid-edge node per unique edge in canonical VTK ordering.

        Parameters:
        -----------
        mesh : VolumetricMesh or pv.UnstructuredGrid
            Source linear mesh.
        reorder : bool, optional
            Whether to apply Reverse Cuthill-McKee (RCM) bandwidth minimization.
            Defaults to self.reorder if not specified.
        """
        apply_reorder = self.reorder if reorder is None else reorder
        vmesh = mesh if isinstance(mesh, VolumetricMesh) else VolumetricMesh(mesh)
        return vmesh.to_quadratic(reorder=apply_reorder)

    def _extract_surface(
        self,
        source: (
            Path
            | str
            | pv.PolyData
            | SurfaceMesh
            | pv.UnstructuredGrid
            | VolumetricMesh
        ),
    ) -> pv.PolyData:
        """Normalizes any supported input geometry into a triangulated pv.PolyData surface."""
        if isinstance(source, (str, Path)):
            return read_surface(source)

        if isinstance(source, SurfaceMesh):
            return source.polydata

        if isinstance(source, pv.PolyData):
            pdata = source.copy(deep=True)
            if not pdata.is_all_triangles:
                pdata = pdata.triangulate()
            return pdata.clean()

        if isinstance(source, VolumetricMesh):
            return source.surface.polydata

        if isinstance(source, pv.UnstructuredGrid):
            return source.extract_surface(algorithm=None).triangulate().clean()

        raise TypeError(
            f"Unsupported source type for tetrahedral generation: {type(source)}."
        )


# Ergonomic alias
TetMesher = TetrahedralMesher


def generate_tetrahedra(
    source: (
        Path | str | pv.PolyData | SurfaceMesh | pv.UnstructuredGrid | VolumetricMesh
    ),
    order: Literal[1, 2] = 1,
    edge_length_fac: float = 0.05,
    edge_length_abs: float | None = None,
    optimize: bool = True,
    simplify: bool = True,
    epsilon: float = 0.001,
    coarsen: bool = False,
    quiet: bool = True,
    reorder: bool = False,
) -> VolumetricMesh:
    """
    Convenience functional interface to TetrahedralMesher.

    Parameters:
    -----------
    source : Path, str, pv.PolyData, SurfaceMesh, pv.UnstructuredGrid, or VolumetricMesh
        Geometric input (*.stl, *.step, *.stp, or mesh object).
    order : {1, 2}, default=1
        1 for linear Tet4, 2 for quadratic Tet10.
    edge_length_fac : float, default=0.05
        Relative element edge length.
    edge_length_abs : float, optional
        Absolute element edge length.
    optimize : bool, default=True
        Run AMIPS element quality optimization.
    simplify : bool, default=True
        Simplify surface prior to meshing.
    epsilon : float, default=0.001
        Envelope tolerance relative to bounding box.
    coarsen : bool, default=False
        Coarsen interior mesh.
    quiet : bool, default=True
        Suppress solver logs.
    reorder : bool, default=False
        If True, applies Reverse Cuthill-McKee (RCM) node renumbering to minimize matrix bandwidth.

    Returns:
    --------
    VolumetricMesh
        Generated 3D tetrahedral continuum mesh.
    """
    mesher = TetrahedralMesher(
        order=order,
        edge_length_fac=edge_length_fac,
        edge_length_abs=edge_length_abs,
        optimize=optimize,
        simplify=simplify,
        epsilon=epsilon,
        coarsen=coarsen,
        quiet=quiet,
        reorder=reorder,
    )
    return mesher.generate(source)
