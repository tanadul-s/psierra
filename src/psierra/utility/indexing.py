from __future__ import annotations
import numpy as np


def node_indices_to_dof_indices(
    node_indices: np.ndarray, dofs_per_node: int = 3
) -> np.ndarray:
    """
    Maps 1D array of node indices to 1D array of global degree of freedom (DOF) indices.

    For 3D solid mechanics (dofs_per_node = 3), node i maps to [3*i, 3*i+1, 3*i+2].

    Parameters:
    -----------
    node_indices : np.ndarray of shape (N,)
        Array of node indices.
    dofs_per_node : int, default=3
        Number of DOFs per node (e.g., 3 for translation in X, Y, Z).

    Returns:
    --------
    dof_indices : np.ndarray of shape (N * dofs_per_node,)
        Flattened array of global DOF indices.
    """
    node_indices = np.asarray(node_indices, dtype=np.int64)
    if node_indices.ndim != 1:
        node_indices = node_indices.ravel()

    # Repeat node indices and add axis offsets: [0, 1, 2]
    base_dofs = np.repeat(node_indices * dofs_per_node, dofs_per_node)
    offsets = np.tile(np.arange(dofs_per_node), len(node_indices))
    return base_dofs + offsets


def compute_element_dofs(elements: np.ndarray, dofs_per_node: int = 3) -> np.ndarray:
    """
    Vectorized computation of global DOF indices for all elements in a mesh.

    Parameters:
    -----------
    elements : np.ndarray of shape (N_elems, N_nodes_per_elem)
        Element connectivity table containing node indices for each element.
    dofs_per_node : int, default=3
        Number of DOFs per node.

    Returns:
    --------
    element_dofs : np.ndarray of shape (N_elems, N_nodes_per_elem * dofs_per_node)
        Global DOF indices for each element.
    """
    elements = np.asarray(elements, dtype=np.int64)
    n_elems, n_nodes_per_elem = elements.shape
    n_elem_dofs = n_nodes_per_elem * dofs_per_node

    # elements[:, :, None] * 3 -> shape (N_elems, N_nodes, 1)
    # add offsets [0, 1, 2]    -> shape (N_elems, N_nodes, 3)
    offsets = np.arange(dofs_per_node)
    elem_dofs = elements[:, :, None] * dofs_per_node + offsets[None, None, :]
    return elem_dofs.reshape(n_elems, n_elem_dofs)
