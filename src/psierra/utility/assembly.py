from __future__ import annotations
import numpy as np
import scipy.sparse as spsp


def create_coo_sparsity_indices(
    element_dofs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precomputes the global row and column index arrays (I, J) for sparse matrix assembly.

    Parameters:
    -----------
    element_dofs : np.ndarray of shape (N_elems, n_elem_dofs)
        Global DOF indices for every element.

    Returns:
    --------
    rows : np.ndarray of shape (N_elems * n_elem_dofs * n_elem_dofs,)
        Flattened row index array for COO triplet assembly.
    cols : np.ndarray of shape (N_elems * n_elem_dofs * n_elem_dofs,)
        Flattened column index array for COO triplet assembly.
    """
    n_elems, n_elem_dofs = element_dofs.shape

    # Broadcast element DOFs to create local grid (N_elems, n_elem_dofs, n_elem_dofs)
    rows = np.broadcast_to(
        element_dofs[:, :, None], (n_elems, n_elem_dofs, n_elem_dofs)
    ).reshape(-1)
    cols = np.broadcast_to(
        element_dofs[:, None, :], (n_elems, n_elem_dofs, n_elem_dofs)
    ).reshape(-1)

    return rows, cols


def assemble_sparse_matrix(
    element_matrices: np.ndarray,
    element_dofs: np.ndarray | None = None,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
    total_dofs: int | None = None,
) -> spsp.csr_matrix:
    """
    Assembles local element matrices into a global CSR sparse matrix.

    Parameters:
    -----------
    element_matrices : np.ndarray of shape (N_elems, n_elem_dofs, n_elem_dofs)
        Local stiffness or mass matrices for all elements.
    element_dofs : np.ndarray of shape (N_elems, n_elem_dofs), optional
        Global DOF indices for every element (used if rows/cols are not precomputed).
    rows : np.ndarray, optional
        Precomputed COO row index array from `create_coo_sparsity_indices`.
    cols : np.ndarray, optional
        Precomputed COO col index array from `create_coo_sparsity_indices`.
    total_dofs : int, optional
        Total number of global DOFs (matrix dimension N x N). If None, inferred from max index + 1.

    Returns:
    --------
    global_matrix : spsp.csr_matrix of shape (total_dofs, total_dofs)
        Assembled global sparse matrix.
    """
    if rows is None or cols is None:
        if element_dofs is None:
            raise ValueError("Either (rows, cols) or element_dofs must be provided.")
        rows, cols = create_coo_sparsity_indices(element_dofs)

    data = np.asarray(element_matrices, dtype=np.float64).reshape(-1)

    if total_dofs is None:
        total_dofs = int(max(rows.max(), cols.max())) + 1

    coo = spsp.coo_matrix((data, (rows, cols)), shape=(total_dofs, total_dofs))
    return coo.tocsr()


def assemble_vector(
    element_vectors: np.ndarray,
    element_dofs: np.ndarray,
    total_dofs: int,
) -> np.ndarray:
    """
    Assembles local element vectors (e.g. internal or body force vectors) into a global vector.

    Parameters:
    -----------
    element_vectors : np.ndarray of shape (N_elems, n_elem_dofs)
        Local element vectors.
    element_dofs : np.ndarray of shape (N_elems, n_elem_dofs)
        Global DOF indices for each element.
    total_dofs : int
        Total number of global DOFs.

    Returns:
    --------
    global_vector : np.ndarray of shape (total_dofs,)
        Assembled global vector.
    """
    global_vector = np.zeros(total_dofs, dtype=np.float64)
    # np.add.at performs unbuffered in-place addition for duplicate indices
    np.add.at(
        global_vector,
        element_dofs.ravel(),
        element_vectors.ravel(),
    )
    return global_vector
