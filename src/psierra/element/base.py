from __future__ import annotations
import numpy as np

from psierra.material.base import Material


class Element:
    """
    Base class for all finite elements in psierra.

    Manages the association with a constitutive MaterialModel model and establishes
    the standard interface for element kinematic computations, stiffness, mass,
    and internal force assembly.

    Parameters:
    -----------
    material : MaterialModel
        The constitutive material model associated with this element.
    """

    def __init__(self, material: Material) -> None:
        self._material = material

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def material(self) -> Material:
        """The constitutive material model for this element."""
        return self._material

    @material.setter
    def material(self, mat: Material) -> None:
        self._material = mat

    @property
    def nodes_per_element(self) -> int:
        """Number of nodes that define this element (e.g. 4 for Tet4, 8 for Hex8)."""
        raise NotImplementedError("Subclasses must implement nodes_per_element.")

    @property
    def dofs_per_node(self) -> int:
        """Number of degrees of freedom per node (3 for 3D translation: ux, uy, uz)."""
        return 3

    @property
    def dofs_per_element(self) -> int:
        """Total degrees of freedom for the element (nodes_per_element * dofs_per_node)."""
        return self.nodes_per_element * self.dofs_per_node

    # =========================================================================
    # Linear FEM Interface
    # =========================================================================

    def compute_linear_stiffness_matrix(self, node_coords: np.ndarray) -> np.ndarray:
        """
        Computes the local linear element stiffness matrix k_e.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (nodes_per_element, 3)
            Reference coordinates X of the element vertices.

        Returns:
        --------
        k_e : np.ndarray of shape (dofs_per_element, dofs_per_element)
            Local stiffness matrix.
        """
        raise NotImplementedError(
            "Subclasses must implement compute_linear_stiffness_matrix."
        )

    def compute_mass_matrix(self, volume: float) -> np.ndarray:
        """
        Computes the local consistent element mass matrix m_e.

        Parameters:
        -----------
        volume : float
            Element volume.

        Returns:
        --------
        m_e : np.ndarray of shape (dofs_per_element, dofs_per_element)
            Local consistent mass matrix.
        """
        raise NotImplementedError("Subclasses must implement compute_mass_matrix.")

    # =========================================================================
    # Nonlinear (Total Lagrangian) FEM Interface
    # =========================================================================

    def compute_nonlinear_force_and_tangent_stiffness(
        self, node_coords: np.ndarray, u_nodes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the local tangent stiffness matrix k_tan and internal force vector f_int
        for the given deformed configuration u.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (nodes_per_element, 3)
            Reference coordinates X of the element vertices.
        u_nodes : np.ndarray of shape (nodes_per_element, 3)
            Current nodal displacements u.

        Returns:
        --------
        k_tan : np.ndarray of shape (dofs_per_element, dofs_per_element)
            Tangent stiffness matrix: k_mat + k_geo.
        f_int : np.ndarray of shape (dofs_per_element,)
            Internal force vector.
        """
        raise NotImplementedError(
            "Subclasses must implement compute_nonlinear_force_and_tangent_stiffness."
        )
