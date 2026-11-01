from __future__ import annotations
import numpy as np

from psierra.element.shell6_base import Shell6Base
from psierra.material.base import Material


class Shell6Linear(Shell6Base):
    """
    Linear (Small-Strain) 6-Node Conforming Quadratic Triangular Shell Element.

    Assumes infinitesimal displacements and rotations. Conforms to Tet10 solid elements.
    """

    def __init__(
        self,
        material: Material,
        thickness: float,
        drilling_factor: float = 1e-3,
        shear_correction_factor: float = 5.0 / 6.0,
    ) -> None:
        super().__init__(
            material=material,
            thickness=thickness,
            drilling_factor=drilling_factor,
            shear_correction_factor=shear_correction_factor,
        )

    def compute_nonlinear_force_and_tangent_stiffness(
        self,
        node_coords: np.ndarray,
        u_nodes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the small-strain linearized tangent stiffness matrix K_tan (36x36)
        and internal force vector f_int = K_0 @ u (36,) for Shell6Linear.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (6, 3)
            Reference coordinates X.
        u_nodes : np.ndarray of shape (6, 6) or (36,)
            Current nodal displacements and rotations.

        Returns:
        --------
        k_tan : np.ndarray of shape (36, 36)
            Linear stiffness matrix K_0.
        f_int : np.ndarray of shape (36,)
            Internal force vector K_0 @ u.
        """
        K = self.compute_linear_stiffness_matrix(node_coords)
        u_vec = np.asarray(u_nodes, dtype=np.float64).ravel()
        f_int = K @ u_vec
        return K, f_int

