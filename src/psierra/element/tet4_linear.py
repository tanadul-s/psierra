from __future__ import annotations
import numpy as np

from psierra.element.tet4_base import Tet4Base
from psierra.material.base import Material


class Tet4Linear(Tet4Base):
    """
    Linear (Small-Strain) 4-Node Tetrahedral Solid Element.

    Assumes infinitesimal displacements and strains. Tangent stiffness is constant
    and internal forces scale linearly with nodal displacements: f_int = k_0 @ u.
    """

    def __init__(self, material: Material) -> None:
        super().__init__(material)

    def compute_nonlinear_force_and_tangent_stiffness(
        self, node_coords: np.ndarray, u_nodes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes linear tangent stiffness matrix k_e (12x12)
        and linear internal force vector f_int = k_e @ u (12,).

        Parameters:
        -----------
        node_coords : np.ndarray of shape (4, 3)
            Reference coordinates X.
        u_nodes : np.ndarray of shape (4, 3)
            Nodal displacements u.

        Returns:
        --------
        k_tan : np.ndarray of shape (12, 12)
            Linear stiffness matrix k_0.
        f_int : np.ndarray of shape (12,)
            Internal force vector k_0 @ u.
        """
        k_lin = self.compute_linear_stiffness_matrix(node_coords)
        f_int = k_lin @ u_nodes.reshape(-1)
        return k_lin, f_int

