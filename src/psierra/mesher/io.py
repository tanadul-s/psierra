from __future__ import annotations
from pathlib import Path
from typing import Literal

import meshio
import numpy as np
import pyvista as pv

from psierra.mesher.cad import read_step


def read_surface(
    filepath: Path | str,
    backend: Literal["pyvista", "meshio", "auto"] = "auto",
) -> pv.PolyData:
    """
    Reads a 2D surface mesh from disk into a PyVista PolyData representation.
    Supports STL, OBJ, PLY, VTP, and CAD STEP/STP formats.

    Parameters:
    -----------
    filepath : Path or str
        Path to surface file.
    backend : {'pyvista', 'meshio', 'auto'}, default='auto'
        Reader backend. 'auto' selects the most suitable engine based on extension.

    Returns:
    --------
    pv.PolyData
        Clean, triangulated surface mesh.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Surface file not found: {path}")

    suffix = path.suffix.lower()

    # Route CAD STEP/STP to OpenCASCADE Gmsh reader
    if suffix in [".step", ".stp"]:
        return read_step(path)

    pdata: pv.PolyData | None = None

    if backend in ["pyvista", "auto"]:
        try:
            raw = pv.read(str(path))
            if isinstance(raw, pv.UnstructuredGrid):
                pdata = raw.extract_surface(algorithm=None)
            elif isinstance(raw, pv.PolyData):
                pdata = raw
        except Exception:
            if backend == "pyvista":
                raise

    if pdata is None and backend in ["meshio", "auto"]:
        try:
            m = meshio.read(str(path))
            raw = pv.from_meshio(m)
            if isinstance(raw, pv.UnstructuredGrid):
                pdata = raw.extract_surface(algorithm=None)
            elif isinstance(raw, pv.PolyData):
                pdata = raw
        except Exception as e:
            raise RuntimeError(
                f"Failed to load surface mesh from '{path}' with PyVista and Meshio: {e}"
            )

    if pdata is None:
        raise RuntimeError(f"Could not parse surface mesh from '{path}'.")

    # Ensure triangles only and clean duplicate nodes
    if not pdata.is_all_triangles:
        pdata = pdata.triangulate()

    return pdata.clean()


def export_mesh(
    mesh: pv.UnstructuredGrid | Any,
    filepath: Path | str,
    backend: Literal["pyvista", "meshio", "auto"] = "auto",
    file_format: str | None = None,
) -> Path:
    """
    Exports a 3D volumetric mesh (Tet4 or Tet10) to disk using either PyVista or Meshio.

    Parameters:
    -----------
    mesh : pv.UnstructuredGrid or VolumetricMesh
        Mesh to export.
    filepath : Path or str
        Target file path (.vtu, .vtk, .msh, .xdmf, .med, etc.).
    backend : {'pyvista', 'meshio', 'auto'}, default='auto'
        Export backend.
    file_format : str, optional
        Specific format name for meshio (e.g. 'gmsh', 'xdmf', 'vtk').
        Defaults to 'gmsh' for .msh files to avoid ANSYS Fluent limitation with quadratic cells.

    Returns:
    --------
    Path
        Absolute path to the exported file.
    """
    from psierra.mesh.volumetric import VolumetricMesh

    ugrid = mesh.ugrid if isinstance(mesh, VolumetricMesh) else mesh
    if not isinstance(ugrid, pv.UnstructuredGrid):
        raise TypeError(f"Expected pv.UnstructuredGrid or VolumetricMesh, got {type(mesh)}.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    # Determine backend if auto
    chosen_backend = backend
    if chosen_backend == "auto":
        if suffix in [".vtu", ".vtk"]:
            chosen_backend = "pyvista"
        else:
            chosen_backend = "meshio"

    if chosen_backend == "pyvista":
        ugrid.save(str(path))
    elif chosen_backend == "meshio":
        points = np.array(ugrid.points, dtype=np.float64)
        cells = []

        # Map VTK cell types to Meshio cell types
        for cell_type, conn in ugrid.cells_dict.items():
            if cell_type == pv.CellType.TETRA:
                cells.append(("tetra", conn))
            elif cell_type == pv.CellType.QUADRATIC_TETRA:
                cells.append(("tetra10", conn))
            elif cell_type == pv.CellType.TRIANGLE:
                cells.append(("triangle", conn))
            else:
                cells.append((cell_type.name.lower(), conn))

        point_data = {k: np.array(v) for k, v in ugrid.point_data.items()}
        cell_data = {k: [np.array(v)] for k, v in ugrid.cell_data.items()}

        m = meshio.Mesh(
            points=points,
            cells=cells,
            point_data=point_data,
            cell_data=cell_data,
        )

        fmt = file_format
        if fmt is None and suffix == ".msh":
            fmt = "gmsh"

        if fmt is not None:
            meshio.write(str(path), m, file_format=fmt)
        else:
            meshio.write(str(path), m)
    else:
        raise ValueError(f"Unknown export backend '{backend}'. Choose 'pyvista', 'meshio', or 'auto'.")

    return path.resolve()


def import_mesh(
    filepath: Path | str,
    backend: Literal["pyvista", "meshio", "auto"] = "auto",
) -> pv.UnstructuredGrid:
    """
    Imports a 3D volumetric mesh from disk into a PyVista UnstructuredGrid.

    Parameters:
    -----------
    filepath : Path or str
        Path to mesh file (.vtu, .vtk, .msh, etc.).
    backend : {'pyvista', 'meshio', 'auto'}, default='auto'

    Returns:
    --------
    pv.UnstructuredGrid
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    suffix = path.suffix.lower()
    chosen_backend = backend
    if chosen_backend == "auto":
        chosen_backend = "pyvista" if suffix in [".vtu", ".vtk"] else "meshio"

    if chosen_backend == "pyvista":
        grid = pv.read(str(path))
        if isinstance(grid, pv.PolyData):
            raise TypeError(f"File '{path}' is a surface PolyData, not a 3D volumetric mesh.")
        return grid

    # Meshio backend
    m = meshio.read(str(path))
    return pv.from_meshio(m)
