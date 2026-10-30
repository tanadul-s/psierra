from __future__ import annotations
from typing import TYPE_CHECKING, overload

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from psierra.mesh.volumetric import VolumetricMesh


def compute_tet10_connectivity_and_points(
    nodes: np.ndarray,
    elements_tet4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes quadratic 10-node tetrahedral connectivity and expanded nodal coordinates
    from linear 4-node tetrahedral mesh arrays using vectorized edge deduplication.

    Follows canonical VTK node ordering for CellType.QUADRATIC_TETRA (VTK ID 24):
      - Nodes 0-3: Corner vertices
      - Node 4: Midpoint of edge (0, 1)
      - Node 5: Midpoint of edge (1, 2)
      - Node 6: Midpoint of edge (2, 0)
      - Node 7: Midpoint of edge (0, 3)
      - Node 8: Midpoint of edge (1, 3)
      - Node 9: Midpoint of edge (2, 3)

    Parameters:
    -----------
    nodes : np.ndarray
        Original vertex coordinates of shape (N_vertices, 3).
    elements_tet4 : np.ndarray
        Linear tetrahedral connectivity array of shape (N_elements, 4).

    Returns:
    --------
    nodes_10 : np.ndarray
        Expanded coordinates including original vertices and unique edge midpoints,
        shape (N_vertices + N_unique_edges, 3).
    elements_tet10 : np.ndarray
        Quadratic tetrahedral connectivity array of shape (N_elements, 10).
    """
    pts = np.asarray(nodes, dtype=np.float64)
    c4 = np.asarray(elements_tet4, dtype=np.int64)

    if c4.ndim != 2 or c4.shape[1] != 4:
        raise ValueError(
            f"Expected elements_tet4 of shape (N, 4), got {c4.shape}."
        )

    # 6 canonical VTK edges for each tetrahedron
    e0 = c4[:, [0, 1]]
    e1 = c4[:, [1, 2]]
    e2 = c4[:, [2, 0]]
    e3 = c4[:, [0, 3]]
    e4 = c4[:, [1, 3]]
    e5 = c4[:, [2, 3]]

    # Stack to shape (N_elements, 6, 2)
    edges = np.stack([e0, e1, e2, e3, e4, e5], axis=1)

    # Undirected canonical edge representation (min_node, max_node)
    sorted_edges = np.sort(edges, axis=-1)
    flat_edges = sorted_edges.reshape(-1, 2)

    # Deduplicate unique edges across all elements
    unique_edges, inverse_mapping = np.unique(
        flat_edges, axis=0, return_inverse=True
    )

    # Compute midpoints for unique edges
    midpoint_coords = 0.5 * (pts[unique_edges[:, 0]] + pts[unique_edges[:, 1]])

    n_orig_nodes = len(pts)
    mid_node_indices = (n_orig_nodes + inverse_mapping).reshape(-1, 6)

    # Assemble 10-node element connectivity
    elements_tet10 = np.hstack([c4, mid_node_indices])
    nodes_10 = np.vstack([pts, midpoint_coords])

    return nodes_10, elements_tet10


def upgrade_tet4_to_tet10(
    source: pv.UnstructuredGrid | VolumetricMesh | tuple[np.ndarray, np.ndarray],
) -> pv.UnstructuredGrid | VolumetricMesh | tuple[np.ndarray, np.ndarray]:
    """
    Upgrades a linear tetrahedral mesh (Tet4) to a quadratic tetrahedral mesh (Tet10)
    by computing unique mid-edge nodes adhering strictly to the VTK canonical numbering.

    Preserves existing point_data (interpolating linearly to midpoints) and cell_data.

    Parameters:
    -----------
    source : pv.UnstructuredGrid, VolumetricMesh, or tuple of (nodes, elements)
        The input linear tetrahedral mesh.

    Returns:
    --------
    Same type as source:
        Quadratic tetrahedral representation with CellType.QUADRATIC_TETRA (VTK ID 24).
    """
    # Import locally to avoid circular dependency
    from psierra.mesh.volumetric import VolumetricMesh

    if isinstance(source, VolumetricMesh):
        ugrid_in = source.ugrid
        is_volumetric_mesh = True
    elif isinstance(source, pv.UnstructuredGrid):
        ugrid_in = source
        is_volumetric_mesh = False
    elif isinstance(source, tuple) and len(source) == 2:
        nodes, elements = source
        return compute_tet10_connectivity_and_points(nodes, elements)
    else:
        raise TypeError(
            f"Unsupported source type for upgrade_tet4_to_tet10: {type(source)}."
        )

    # If already quadratic, return copy
    if pv.CellType.QUADRATIC_TETRA in ugrid_in.cells_dict and pv.CellType.TETRA not in ugrid_in.cells_dict:
        return VolumetricMesh(ugrid_in.copy(deep=True)) if is_volumetric_mesh else ugrid_in.copy(deep=True)

    if pv.CellType.TETRA not in ugrid_in.cells_dict:
        raise ValueError(
            "Input mesh does not contain linear tetrahedral cells (CellType.TETRA)."
        )

    orig_nodes = np.array(ugrid_in.points, dtype=np.float64)
    orig_elements = np.array(ugrid_in.cells_dict[pv.CellType.TETRA], dtype=np.int64)

    nodes_10, elements_tet10 = compute_tet10_connectivity_and_points(
        orig_nodes, orig_elements
    )

    grid10 = pv.UnstructuredGrid(
        {pv.CellType.QUADRATIC_TETRA: elements_tet10}, nodes_10
    )

    # Interpolate point_data to mid-edge nodes
    n_orig = len(orig_nodes)
    for key, data_array in ugrid_in.point_data.items():
        arr = np.asarray(data_array)
        # Extract unique edge indices from elements_tet10
        # elements_tet10[:, 4:] contains the mid-node indices
        # We need the values for indices [n_orig, len(nodes_10) - 1]
        new_data = np.zeros((len(nodes_10), *arr.shape[1:]), dtype=arr.dtype)
        new_data[:n_orig] = arr

        # Re-derive unique edge connections to interpolate
        e0 = orig_elements[:, [0, 1]]
        e1 = orig_elements[:, [1, 2]]
        e2 = orig_elements[:, [2, 0]]
        e3 = orig_elements[:, [0, 3]]
        e4 = orig_elements[:, [1, 3]]
        e5 = orig_elements[:, [2, 3]]
        edges = np.stack([e0, e1, e2, e3, e4, e5], axis=1)
        sorted_edges = np.sort(edges, axis=-1).reshape(-1, 2)
        unique_edges = np.unique(sorted_edges, axis=0)

        mid_vals = 0.5 * (arr[unique_edges[:, 0]] + arr[unique_edges[:, 1]])
        new_data[n_orig:] = mid_vals
        grid10.point_data[key] = new_data

    # Transfer cell_data directly (1-to-1 element correspondence)
    for key, cell_array in ugrid_in.cell_data.items():
        grid10.cell_data[key] = np.array(cell_array, copy=True)

    if is_volumetric_mesh:
        return VolumetricMesh(grid10)
    return grid10


def downgrade_tet10_to_tet4(
    source: pv.UnstructuredGrid | VolumetricMesh | tuple[np.ndarray, np.ndarray],
) -> pv.UnstructuredGrid | VolumetricMesh | tuple[np.ndarray, np.ndarray]:
    """
    Downgrades a quadratic tetrahedral mesh (Tet10) to a linear tetrahedral mesh (Tet4)
    by extracting the 4 corner vertices of each element.

    Parameters:
    -----------
    source : pv.UnstructuredGrid, VolumetricMesh, or tuple of (nodes, elements)
        The input quadratic tetrahedral mesh.

    Returns:
    --------
    Same type as source:
        Linear tetrahedral representation with CellType.TETRA (VTK ID 10).
    """
    from psierra.mesh.volumetric import VolumetricMesh

    if isinstance(source, VolumetricMesh):
        ugrid_in = source.ugrid
        is_volumetric_mesh = True
    elif isinstance(source, pv.UnstructuredGrid):
        ugrid_in = source
        is_volumetric_mesh = False
    elif isinstance(source, tuple) and len(source) == 2:
        nodes, elements = source
        c10 = np.asarray(elements, dtype=np.int64)
        return np.asarray(nodes, dtype=np.float64), c10[:, :4]
    else:
        raise TypeError(
            f"Unsupported source type for downgrade_tet10_to_tet4: {type(source)}."
        )

    if pv.CellType.QUADRATIC_TETRA not in ugrid_in.cells_dict:
        # Already linear, return copy
        return VolumetricMesh(ugrid_in.copy(deep=True)) if is_volumetric_mesh else ugrid_in.copy(deep=True)

    c10 = np.array(ugrid_in.cells_dict[pv.CellType.QUADRATIC_TETRA], dtype=np.int64)
    c4 = c10[:, :4]
    points = np.array(ugrid_in.points, dtype=np.float64)

    # Clean unreferenced mid-edge nodes
    unique_corners, inverse_corners = np.unique(c4, return_inverse=True)
    clean_c4 = inverse_corners.reshape(c4.shape)
    clean_points = points[unique_corners]

    grid4 = pv.UnstructuredGrid({pv.CellType.TETRA: clean_c4}, clean_points)

    # Transfer point_data (for corner nodes) and cell_data
    for key, arr in ugrid_in.point_data.items():
        arr_np = np.asarray(arr)
        grid4.point_data[key] = arr_np[unique_corners]
    for key, arr in ugrid_in.cell_data.items():
        grid4.cell_data[key] = np.array(arr, copy=True)

    if is_volumetric_mesh:
        return VolumetricMesh(grid4)
    return grid4

