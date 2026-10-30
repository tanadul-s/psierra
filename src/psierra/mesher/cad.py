from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


def is_gmsh_available() -> bool:
    """Returns True if the 'gmsh' Python package is installed and importable."""
    try:
        import gmsh  # type: ignore[import-not-found]
        return True
    except ImportError:
        return False


def read_step(
    filepath: Path | str,
    mesh_size: float | None = None,
    min_mesh_size: float | None = None,
    max_mesh_size: float | None = None,
    optimize: bool = True,
) -> pv.PolyData:
    """
    Reads an ISO 10303 CAD boundary representation file (*.step, *.stp) and
    discretizes its external boundary into a 2D triangular surface mesh (pv.PolyData).

    Parameters:
    -----------
    filepath : Path or str
        Path to the STEP/STP CAD model file.
    mesh_size : float, optional
        Target characteristic element size for the boundary discretization.
    min_mesh_size : float, optional
        Minimum characteristic element size constraint.
    max_mesh_size : float, optional
        Maximum characteristic element size constraint.
    optimize : bool, default=True
        Whether to perform 2D surface mesh optimization after generation.

    Returns:
    --------
    pv.PolyData
        Discretized triangular boundary surface mesh.

    Raises:
    -------
    FileNotFoundError
        If the given STEP file does not exist.
    ImportError
        If 'gmsh' is not installed in the Python environment.
    RuntimeError
        If CAD import or surface meshing fails.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CAD STEP file not found: {path}")

    if not is_gmsh_available():
        raise ImportError(
            f"Reading STEP/STP CAD files ({path.suffix}) requires the 'gmsh' library. "
            "Please install it in your environment (e.g., 'pip install gmsh') or "
            "export your CAD model as an STL surface mesh using your CAD software."
        )

    import gmsh  # type: ignore[import-not-found]

    was_initialized = gmsh.isInitialized()
    if not was_initialized:
        gmsh.initialize()

    model_name = f"step_import_{path.stem}"
    gmsh.model.add(model_name)

    try:
        # Import B-Rep shapes using OpenCASCADE kernel
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()

        # Configure mesh sizing options
        if mesh_size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        if min_mesh_size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMin", min_mesh_size)
        if max_mesh_size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMax", max_mesh_size)

        # Set 2D surface meshing algorithm (Frontal-Delaunay for 2D meshes)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        if optimize:
            gmsh.option.setNumber("Mesh.Optimize", 1)

        # Generate 2D surface triangulation
        gmsh.model.mesh.generate(2)

        # Extract nodal coordinates
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords, dtype=np.float64).reshape(-1, 3)

        # Map 1-based gmsh node tags to 0-based array indices
        tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}

        # Extract 2D triangular element connectivity (Gmsh element type 2 is 3-node triangle)
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)

        triangles_list: list[list[int]] = []
        for el_type, tags in zip(elem_types, elem_node_tags):
            if el_type == 2:  # 3-node triangle
                tag_arr = np.array(tags, dtype=np.int64).reshape(-1, 3)
                for tri in tag_arr:
                    triangles_list.append([tag_to_idx[tri[0]], tag_to_idx[tri[1]], tag_to_idx[tri[2]]])

        if not triangles_list:
            raise RuntimeError(
                f"No 2D triangular surface elements could be generated from CAD file: {path}"
            )

        tri_connectivity = np.array(triangles_list, dtype=np.int64)

        # Format for PyVista PolyData: [3, n0, n1, n2, 3, n0, n1, n2, ...]
        n_faces = len(tri_connectivity)
        faces = np.hstack([np.full((n_faces, 1), 3, dtype=np.int64), tri_connectivity]).ravel()

        polydata = pv.PolyData(coords, faces=faces)
        return polydata.clean().triangulate()

    finally:
        # Clean up model
        try:
            gmsh.model.remove()
        except Exception:
            pass
        if not was_initialized:
            gmsh.finalize()

