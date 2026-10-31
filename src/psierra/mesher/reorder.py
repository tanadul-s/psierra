from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix, csgraph

if TYPE_CHECKING:
    from psierra.mesh.volumetric import VolumetricMesh


def compute_mesh_bandwidth(elements: np.ndarray) -> int:
    """
    Computes the nodal bandwidth of a finite element mesh.

    In FEM, the bandwidth is the maximum difference between any two node indices
    interacting within the same element:
        B = max_e (max_k(e_k) - min_k(e_k))

    Parameters:
    -----------
    elements : np.ndarray
        Element connectivity array of shape (N_elements, nodes_per_element).

    Returns:
    --------
    int
        Maximum nodal index difference across all elements.
    """
    elem_arr = np.asarray(elements, dtype=np.int64)
    if elem_arr.ndim != 2 or elem_arr.shape[0] == 0:
        return 0

    span_per_elem = np.max(elem_arr, axis=1) - np.min(elem_arr, axis=1)
    return int(np.max(span_per_elem))


def reorder_mesh_nodes(
    source: pv.UnstructuredGrid | VolumetricMesh,
    method: Literal["rcm"] = "rcm",
) -> pv.UnstructuredGrid | VolumetricMesh:
    """
    Renubmers the nodes of a volumetric finite element mesh to minimize matrix bandwidth
    and optimize sparse solver efficiency and CPU cache locality.

    Parameters:
    -----------
    source : pv.UnstructuredGrid or VolumetricMesh
        Input mesh to be renumbered.
    method : {'rcm'}, default='rcm'
        Reordering graph algorithm:
          - 'rcm': Reverse Cuthill-McKee algorithm via breadth-first search.

    Returns:
    --------
    Same type as source:
        Renumbered mesh with identical geometric shape, volume, and element topologies,
        with nodes reordered to minimize the profile and bandwidth.
    """
    from psierra.mesh.volumetric import VolumetricMesh

    is_volumetric_mesh = isinstance(source, VolumetricMesh)
    ugrid_in = source.ugrid if is_volumetric_mesh else source

    if not isinstance(ugrid_in, pv.UnstructuredGrid):
        raise TypeError(
            f"Expected pv.UnstructuredGrid or VolumetricMesh, got {type(source)}."
        )

    # Determine active tetrahedral cell type
    cell_type: pv.CellType | None = None
    if pv.CellType.QUADRATIC_TETRA in ugrid_in.cells_dict:
        cell_type = pv.CellType.QUADRATIC_TETRA
    elif pv.CellType.TETRA in ugrid_in.cells_dict:
        cell_type = pv.CellType.TETRA
    else:
        # Fallback to the first available 3D cell type
        for ct in ugrid_in.cells_dict:
            cell_type = ct
            break

    if cell_type is None:
        return VolumetricMesh(ugrid_in.copy(deep=True)) if is_volumetric_mesh else ugrid_in.copy(deep=True)

    conn = np.array(ugrid_in.cells_dict[cell_type], dtype=np.int64)
    n_nodes = ugrid_in.n_points
    n_per_elem = conn.shape[1]

    # Construct nodal adjacency graph
    rows = np.repeat(conn, n_per_elem)
    cols = np.tile(conn, (1, n_per_elem)).ravel()
    adj = coo_matrix(
        (np.ones_like(rows, dtype=bool), (rows, cols)),
        shape=(n_nodes, n_nodes),
    ).tocsr()

    # Compute graph permutation
    if method == "rcm":
        perm = csgraph.reverse_cuthill_mckee(adj)
    else:
        raise ValueError(
            f"Unknown reordering method '{method}'. Supported methods: 'rcm'."
        )

    # Invert permutation: inv_perm[old_index] = new_index
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))

    # Permute nodal coordinates
    new_points = np.array(ugrid_in.points[perm], dtype=np.float64)

    # Re-index element connectivity
    new_conn = inv_perm[conn]

    # Assemble new PyVista UnstructuredGrid
    new_grid = pv.UnstructuredGrid({cell_type: new_conn}, new_points)

    # Permute point data fields
    for key, data_array in ugrid_in.point_data.items():
        arr = np.asarray(data_array)
        new_grid.point_data[key] = arr[perm]

    # Transfer cell data fields directly (element identities unchanged)
    for key, cell_array in ugrid_in.cell_data.items():
        new_grid.cell_data[key] = np.array(cell_array, copy=True)

    if is_volumetric_mesh:
        return VolumetricMesh(new_grid)
    return new_grid

