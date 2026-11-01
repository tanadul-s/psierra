from __future__ import annotations
import numpy as np

from psierra.element.shell3_base import Shell3Base
from psierra.material.base import Material


class Shell3Linear(Shell3Base):
    """
    Linear (Small-Strain) 3-Node Triangular Shell Element.

    Assumes infinitesimal displacements and rotations. For finite 3D rigid rotations,
    prefer Shell3Corotational.
    """

    def __init__(
        self,
        material: Material,
        thickness: float,
        drilling_factor: float = 1e-3,
    ) -> None:
        super().__init__(
            material=material,
            thickness=thickness,
            drilling_factor=drilling_factor,
        )

    def compute_nonlinear_force_and_tangent_stiffness(
        self,
        node_coords: np.ndarray,
        u_nodes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the small-strain linearized tangent stiffness matrix K_tan (18x18)
        and internal force vector f_int = K_0 @ u (18,) for Shell3Linear.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (3, 3)
            Reference coordinates X.
        u_nodes : np.ndarray of shape (3, 6) or (18,)
            Current nodal displacements and rotations.

        Returns:
        --------
        k_tan : np.ndarray of shape (18, 18)
            Linear stiffness matrix K_0.
        f_int : np.ndarray of shape (18,)
            Internal force vector K_0 @ u.
        """
        K = self.compute_linear_stiffness_matrix(node_coords)
        u_vec = np.asarray(u_nodes, dtype=np.float64).ravel()
        f_int = K @ u_vec
        return K, f_int

