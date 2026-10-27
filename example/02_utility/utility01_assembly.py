import numpy as np

from psierra.utility.indexing import compute_element_dofs
from psierra.utility.assembly import assemble_sparse_matrix


def main():
    # Two simple tetrahedral elements:
    elements = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    elem_dofs = compute_element_dofs(elements)
    print("Element DOFs shape:", elem_dofs.shape)  # (2, 12)

    # Dummy element stiffness matrices:
    k_local = np.eye(12)
    k_all = np.stack([k_local, k_local])  # (2, 12, 12)

    # Assemble:
    K_global = assemble_sparse_matrix(
        element_matrices=k_all, element_dofs=elem_dofs, total_dofs=15
    )
    print("Global matrix shape:", K_global.shape)  # (15, 15)
    print("K:", K_global.toarray())


if __name__ == "__main__":
    main()
