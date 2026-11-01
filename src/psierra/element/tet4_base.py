from __future__ import annotations
import numpy as np

from psierra.element.base import Element
from psierra.material.base import Material


class Tet4Base(Element):
    """
    Base class for 4-Node Linear Tetrahedral Solid Elements.

    Provides core geometry, shape function gradients, linear strain-displacement
    matrix B_L, geometric matrix G, linear stiffness matrix k_e, and consistent mass matrix m_e.
    """

    def __init__(self, material: Material) -> None:
        super().__init__(material)
        self._nodes_per_element = 4
        self._dofs_per_node = 3
        self._dofs_per_element = 12

    @property
    def nodes_per_element(self) -> int:
        return self._nodes_per_element

    @property
    def dofs_per_element(self) -> int:
        return self._dofs_per_element

    # =========================================================================
    # Geometry & Shape Function Gradients
    # =========================================================================

    def compute_volume_and_gradients(
        self, node_coords: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Computes reference element volume and shape function gradient matrix H.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (4, 3)
            Reference nodal coordinates [X, Y, Z] for the 4 vertices.

        Returns:
        --------
        volume : float
            Reference volume V_e.
        H : np.ndarray of shape (4, 3)
            Shape function gradient matrix in reference configuration:
            [[dN0/dX, dN0/dY, dN0/dZ], ..., [dN3/dX, dN3/dY, dN3/dZ]]
        """
        A = np.ones((4, 4), dtype=np.float64)
        A[:, 1:] = node_coords

        detA = np.linalg.det(A)
        if detA <= 0.0:
            raise ValueError(
                f"Negative or zero element volume detected (det(A) = {detA}). "
                "Check element vertex ordering."
            )

        volume = detA / 6.0
        A_inv = np.linalg.inv(A)
        H = A_inv[1:, :].T  # Shape (4, 3)

        return volume, H

    # =========================================================================
    # Strain-Displacement (B) and Geometric (G) Matrices
    # =========================================================================

    def compute_linear_B_matrix(self, H: np.ndarray) -> np.ndarray:
        """
        Computes the linear strain-displacement matrix B_L (6 x 12).
        """
        BL = np.zeros((6, 12), dtype=np.float64)

        for i in range(4):
            b, c, d = H[i, 0], H[i, 1], H[i, 2]
            col = i * 3
            BL[0, col + 0] = b
            BL[1, col + 1] = c
            BL[2, col + 2] = d

            BL[3, col + 0] = c
            BL[3, col + 1] = b

            BL[4, col + 1] = d
            BL[4, col + 2] = c

            BL[5, col + 0] = d
            BL[5, col + 2] = b

        return BL

    def compute_geometric_matrix(self, H: np.ndarray) -> np.ndarray:
        """
        Computes the geometric gradient matrix G (9 x 12).
        Maps nodal displacements to the 9-element displacement gradient vector.
        """
        G = np.zeros((9, 12), dtype=np.float64)

        for i in range(4):
            b, c, d = H[i, 0], H[i, 1], H[i, 2]
            col = i * 3

            # du/dX, du/dY, du/dZ
            G[0, col + 0] = b
            G[1, col + 0] = c
            G[2, col + 0] = d

            # dv/dX, dv/dY, dv/dZ
            G[3, col + 1] = b
            G[4, col + 1] = c
            G[5, col + 1] = d

            # dw/dX, dw/dY, dw/dZ
            G[6, col + 2] = b
            G[7, col + 2] = c
            G[8, col + 2] = d

        return G

    # =========================================================================
    # Linear Stiffness & Mass Matrices
    # =========================================================================

    def compute_linear_stiffness_matrix(self, node_coords: np.ndarray) -> np.ndarray:
        """Computes the 12x12 linear stiffness matrix k_e = (B^T C B) * V_e."""
        V, H = self.compute_volume_and_gradients(node_coords)
        BL = self.compute_linear_B_matrix(H)
        C = self._material.compute_constitutive_matrix()
        return (BL.T @ C @ BL) * V

    def compute_mass_matrix(self, volume: float) -> np.ndarray:
        """Computes the 12x12 consistent element mass matrix."""
        rho = self._material.mass_density
        factor = (rho * volume) / 20.0

        I3 = np.identity(3, dtype=np.float64)
        m_e = np.zeros((12, 12), dtype=np.float64)

        for i in range(4):
            for j in range(4):
                weight = 2.0 if i == j else 1.0
                m_e[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = weight * factor * I3

        return m_e

