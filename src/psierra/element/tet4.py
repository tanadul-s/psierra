from __future__ import annotations
import numpy as np

from psierra.element.base import Element
from psierra.material.base import Material
from psierra.utility.tensor import tensor_to_voigt_stress


class Tet4(Element):
    """
    Unified 4-Node Linear & Total Lagrangian Tetrahedral Solid Element.

    Provides:
    - Shape function gradients & volume calculation
    - Linear stiffness matrix (k_e) and mass matrix (m_e)
    - Nonlinear Total Lagrangian internal force vector (f_int) and tangent stiffness (k_tan)
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

    def compute_nonlinear_B_matrix(
        self, H: np.ndarray, u_nodes: np.ndarray
    ) -> np.ndarray:
        """
        Computes the Total Lagrangian strain-displacement matrix B(u) = B_L + B_NL(u).

        Parameters:
        -----------
        H : np.ndarray of shape (4, 3)
            Shape function gradient matrix.
        u_nodes : np.ndarray of shape (4, 3)
            Current nodal displacements for the 4 vertices.

        Returns:
        --------
        B : np.ndarray of shape (6, 12)
        """
        BL = self.compute_linear_B_matrix(H)
        BNL = np.zeros((6, 12), dtype=np.float64)

        # Displacement gradient l = u^T @ H (shape: 3 x 3)
        # l[i, j] = du_i / dX_j
        l = u_nodes.T @ H
        du, dv, dw = l[0], l[1], l[2]

        for i in range(4):
            b, c, d = H[i, 0], H[i, 1], H[i, 2]
            col = i * 3

            # Normal strain derivatives: du_k/dX_i * dN_a/dX_i
            BNL[0, col : col + 3] = du[0] * b, dv[0] * b, dw[0] * b
            BNL[1, col : col + 3] = du[1] * c, dv[1] * c, dw[1] * c
            BNL[2, col : col + 3] = du[2] * d, dv[2] * d, dw[2] * d

            # Shear strain derivatives
            BNL[3, col : col + 3] = (
                du[0] * c + du[1] * b,
                dv[0] * c + dv[1] * b,
                dw[0] * c + dw[1] * b,
            )
            BNL[4, col : col + 3] = (
                du[1] * d + du[2] * c,
                dv[1] * d + dv[2] * c,
                dw[1] * d + dw[2] * c,
            )
            BNL[5, col : col + 3] = (
                du[0] * d + du[2] * b,
                dv[0] * d + dv[2] * b,
                dw[0] * d + dw[2] * b,
            )

        return BL + BNL

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

    # =========================================================================
    # Nonlinear Total Lagrangian Stiffness & Internal Force
    # =========================================================================

    def compute_nonlinear_force_and_tangent_stiffness(
        self, node_coords: np.ndarray, u_nodes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the Total Lagrangian tangent stiffness matrix k_tan (12x12)
        and internal load vector f_int (12,) for the current deformed state.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (4, 3)
            Reference coordinates X.
        u_nodes : np.ndarray of shape (4, 3)
            Nodal displacements u.

        Returns:
        --------
        k_tan : np.ndarray of shape (12, 12)
            Tangent stiffness matrix: k_mat + k_geo.
        f_int : np.ndarray of shape (12,)
            Internal force vector.
        """
        V, H = self.compute_volume_and_gradients(node_coords)

        # Current coordinates: x = X + u
        x = node_coords + u_nodes

        # Deformation gradient: F = x^T @ H (shape: 3 x 3)
        F = x.T @ H
        J = np.linalg.det(F)
        if J <= 0.0:
            raise ValueError(f"Element inversion detected: det(F) = {J:.4e} <= 0.")

        # Query material for constitutive tangent C and 2nd PK stress S
        C, S = self._material.compute_stress_and_tangent(F)

        # Matrices B(u) and G
        B = self.compute_nonlinear_B_matrix(H, u_nodes)
        G = self.compute_geometric_matrix(H)

        # 1. Material Stiffness: k_mat = B^T @ C @ B * V
        k_mat = (B.T @ C @ B) * V

        # 2. Geometric Stiffness: k_geo = G^T @ (I3 ⊗ S) @ G * V
        S_hat = np.kron(np.identity(3, dtype=np.float64), S)
        k_geo = (G.T @ S_hat @ G) * V

        k_tan = k_mat + k_geo

        # 3. Internal Force: f_int = B^T @ S_voigt * V
        S_voigt = tensor_to_voigt_stress(S)
        f_int = (B.T @ S_voigt) * V

        return k_tan, f_int
