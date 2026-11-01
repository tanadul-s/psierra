from __future__ import annotations
import numpy as np

from psierra.element.tet10_base import Tet10Base
from psierra.material.base import Material


class Tet10Linear(Tet10Base):
    """
    Linear (Small-Strain) 10-Node Quadratic Tetrahedral Solid Element.

    Assumes infinitesimal displacements and strains. Tangent stiffness is constant
    and internal forces scale linearly with nodal displacements: f_int = k_0 @ u.
    """

    def __init__(self, material: Material, default_quadrature_order: int = 4) -> None:
        super().__init__(material, default_quadrature_order=default_quadrature_order)

    def compute_nonlinear_force_and_tangent_stiffness(
        self,
        node_coords: np.ndarray,
        u_nodes: np.ndarray,
        quadrature_order: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes linear tangent stiffness matrix k_e (30x30)
        and linear internal force vector f_int = k_e @ u (30,).

        Parameters:
        -----------
        node_coords : np.ndarray of shape (10, 3)
            Reference coordinates X.
        u_nodes : np.ndarray of shape (10, 3)
            Nodal displacements u.
        quadrature_order : int, optional
            Number of Gauss integration points (default: 4).

        Returns:
        --------
        k_tan : np.ndarray of shape (30, 30)
            Linear stiffness matrix k_0.
        f_int : np.ndarray of shape (30,)
            Internal force vector k_0 @ u.
        """
        k_lin = self.compute_linear_stiffness_matrix(node_coords, quadrature_order)
        f_int = k_lin @ u_nodes.reshape(-1)
        return k_lin, f_int

