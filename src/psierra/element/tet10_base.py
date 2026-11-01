from __future__ import annotations
import numpy as np

from psierra.element.base import Element
from psierra.element.quadrature import get_tetrahedral_quadrature
from psierra.material.base import Material


# Exact analytical dimensionless reference mass matrix for canonical 10-node tetrahedron:
# M_ref[i, j] = (1 / V_e) * \int_{Omega_e} N_i(X) * N_j(X) dV
# Values are exact integers divided by 2520.
_TET10_MASS_MATRIX_INT = np.array(
    [
        [36, 6, 6, 6, -24, -36, -24, -24, -36, -36],
        [6, 36, 6, 6, -24, -24, -36, -36, -24, -36],
        [6, 6, 36, 6, -36, -24, -24, -36, -36, -24],
        [6, 6, 6, 36, -36, -36, -36, -24, -24, -24],
        [-24, -24, -36, -36, 192, 96, 96, 96, 96, 48],
        [-36, -24, -24, -36, 96, 192, 96, 48, 96, 96],
        [-24, -36, -24, -36, 96, 96, 192, 96, 48, 96],
        [-24, -36, -36, -24, 96, 48, 96, 192, 96, 96],
        [-36, -24, -36, -24, 96, 96, 48, 96, 192, 96],
        [-36, -36, -24, -24, 48, 96, 96, 96, 96, 192],
    ],
    dtype=np.float64,
)
_TET10_REFERENCE_MASS_MATRIX = _TET10_MASS_MATRIX_INT / 2520.0


class Tet10Base(Element):
    """
    Base class for 10-Node Quadratic Isoparametric Tetrahedral Solid Elements.

    Canonical Node Ordering (VTK CellType.QUADRATIC_TETRA, VTK ID 24):
      - Nodes 0-3: Corner vertices
      - Node 4: Midpoint of edge (0, 1)
      - Node 5: Midpoint of edge (1, 2)
      - Node 6: Midpoint of edge (2, 0)
      - Node 7: Midpoint of edge (0, 3)
      - Node 8: Midpoint of edge (1, 3)
      - Node 9: Midpoint of edge (2, 3)

    Barycentric / Volume Coordinates:
      L_1 = xi, L_2 = eta, L_3 = zeta, L_0 = 1 - xi - eta - zeta

    Shape Functions:
      - Corners (i = 0, 1, 2, 3): N_i = L_i * (2 * L_i - 1)
      - Mid-edges:
          N_4 = 4 * L_0 * L_1
          N_5 = 4 * L_1 * L_2
          N_6 = 4 * L_2 * L_0
          N_7 = 4 * L_0 * L_3
          N_8 = 4 * L_1 * L_3
          N_9 = 4 * L_2 * L_3

    Parameters:
    -----------
    material : Material
        Constitutive material law governing stress-strain relationships.
    default_quadrature_order : int, default=4
        Default number of Gauss quadrature points (4 is exact for linear elasticity).
    """

    def __init__(self, material: Material, default_quadrature_order: int = 4) -> None:
        super().__init__(material)
        self._nodes_per_element = 10
        self._dofs_per_node = 3
        self._dofs_per_element = 30
        self._default_quadrature_order = default_quadrature_order

    @property
    def nodes_per_element(self) -> int:
        return self._nodes_per_element

    @property
    def dofs_per_element(self) -> int:
        return self._dofs_per_element

    @property
    def default_quadrature_order(self) -> int:
        return self._default_quadrature_order

    # =========================================================================
    # Shape Functions & Reference Gradients
    # =========================================================================

    @staticmethod
    def evaluate_shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
        """
        Evaluates the 10 quadratic shape functions at natural coordinates (xi, eta, zeta).

        Returns:
        --------
        N : np.ndarray of shape (10,)
        """
        L1 = xi
        L2 = eta
        L3 = zeta
        L0 = 1.0 - L1 - L2 - L3

        N = np.zeros(10, dtype=np.float64)
        # 4 Corner vertices
        N[0] = L0 * (2.0 * L0 - 1.0)
        N[1] = L1 * (2.0 * L1 - 1.0)
        N[2] = L2 * (2.0 * L2 - 1.0)
        N[3] = L3 * (2.0 * L3 - 1.0)

        # 6 Mid-edge nodes
        N[4] = 4.0 * L0 * L1
        N[5] = 4.0 * L1 * L2
        N[6] = 4.0 * L2 * L0
        N[7] = 4.0 * L0 * L3
        N[8] = 4.0 * L1 * L3
        N[9] = 4.0 * L2 * L3
        return N

    @staticmethod
    def evaluate_shape_gradients_ref(
        xi: float, eta: float, zeta: float
    ) -> np.ndarray:
        """
        Evaluates derivatives of the 10 shape functions with respect to natural
        reference coordinates (xi, eta, zeta).

        Returns:
        --------
        dN_dxi : np.ndarray of shape (10, 3)
            [[dN_0/dxi, dN_0/deta, dN_0/dzeta], ..., [dN_9/dxi, dN_9/deta, dN_9/dzeta]]
        """
        L1 = xi
        L2 = eta
        L3 = zeta
        L0 = 1.0 - L1 - L2 - L3

        dN = np.zeros((10, 3), dtype=np.float64)

        # Node 0 (corner): N_0 = L0 * (2 * L0 - 1)
        dN[0, :] = -(4.0 * L0 - 1.0)

        # Node 1 (corner): N_1 = L1 * (2 * L1 - 1)
        dN[1, 0] = 4.0 * L1 - 1.0

        # Node 2 (corner): N_2 = L2 * (2 * L2 - 1)
        dN[2, 1] = 4.0 * L2 - 1.0

        # Node 3 (corner): N_3 = L3 * (2 * L3 - 1)
        dN[3, 2] = 4.0 * L3 - 1.0

        # Node 4 (mid 0-1): N_4 = 4 * L0 * L1
        dN[4, 0] = 4.0 * (L0 - L1)
        dN[4, 1] = -4.0 * L1
        dN[4, 2] = -4.0 * L1

        # Node 5 (mid 1-2): N_5 = 4 * L1 * L2
        dN[5, 0] = 4.0 * L2
        dN[5, 1] = 4.0 * L1
        dN[5, 2] = 0.0

        # Node 6 (mid 2-0): N_6 = 4 * L2 * L0
        dN[6, 0] = -4.0 * L2
        dN[6, 1] = 4.0 * (L0 - L2)
        dN[6, 2] = -4.0 * L2

        # Node 7 (mid 0-3): N_7 = 4 * L0 * L3
        dN[7, 0] = -4.0 * L3
        dN[7, 1] = -4.0 * L3
        dN[7, 2] = 4.0 * (L0 - L3)

        # Node 8 (mid 1-3): N_8 = 4 * L1 * L3
        dN[8, 0] = 4.0 * L3
        dN[8, 1] = 0.0
        dN[8, 2] = 4.0 * L1

        # Node 9 (mid 2-3): N_9 = 4 * L2 * L3
        dN[9, 0] = 0.0
        dN[9, 1] = 4.0 * L3
        dN[9, 2] = 4.0 * L2

        return dN

    # =========================================================================
    # Geometry, Volume, & Spatial Gradients
    # =========================================================================

    def compute_volume_and_gradients(
        self,
        node_coords: np.ndarray,
        xi: np.ndarray | tuple[float, float, float] | None = None,
    ) -> tuple[float, np.ndarray]:
        """
        Computes the reference element volume and spatial shape function gradients H
        evaluated at natural coordinate `xi` (defaults to the element centroid (1/4, 1/4, 1/4)).

        Parameters:
        -----------
        node_coords : np.ndarray of shape (10, 3)
            Reference nodal coordinates [X, Y, Z] for the 10 nodes.
        xi : array-like of shape (3,), optional
            Natural coordinates (xi, eta, zeta). If None, defaults to centroid [0.25, 0.25, 0.25].

        Returns:
        --------
        volume : float
            Integrated reference volume V_e.
        H : np.ndarray of shape (10, 3)
            Spatial shape function gradients dN_a / dX_j evaluated at `xi`.
        """
        if node_coords.shape != (10, 3):
            raise ValueError(
                f"Expected node_coords of shape (10, 3), got {node_coords.shape}."
            )

        # Integrate volume accurately with 4-point quadrature
        qp, qw = get_tetrahedral_quadrature(4)
        volume = 0.0
        for q in range(4):
            dN_dxi_q = self.evaluate_shape_gradients_ref(*qp[q])
            J_q = node_coords.T @ dN_dxi_q
            detJ_q = np.linalg.det(J_q)
            if detJ_q <= 0.0:
                raise ValueError(
                    f"Negative or zero element Jacobian determinant detected "
                    f"(det(J) = {detJ_q:.4e}). Check element vertex ordering."
                )
            volume += qw[q] * detJ_q

        # Spatial gradients evaluated at target xi (default: centroid)
        if xi is None:
            xi_eval = (0.25, 0.25, 0.25)
        else:
            xi_eval = tuple(xi)

        dN_dxi = self.evaluate_shape_gradients_ref(*xi_eval)
        J = node_coords.T @ dN_dxi
        detJ = np.linalg.det(J)
        if detJ <= 0.0:
            raise ValueError(
                f"Negative or zero element Jacobian determinant detected at xi={xi_eval} "
                f"(det(J) = {detJ:.4e})."
            )
        H = dN_dxi @ np.linalg.inv(J)

        return float(volume), H

    # =========================================================================
    # Strain-Displacement (B) and Geometric (G) Matrices
    # =========================================================================

    @staticmethod
    def compute_linear_B_matrix(H: np.ndarray) -> np.ndarray:
        """
        Computes the linear strain-displacement matrix B_L (6 x 30) from spatial gradients H.

        Voigt Order (Standard psierra VOIGT_MAP_3D):
          [epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_yz, gamma_zx]^T
        """
        BL = np.zeros((6, 30), dtype=np.float64)

        for i in range(10):
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

    @staticmethod
    def compute_geometric_matrix(H: np.ndarray) -> np.ndarray:
        """
        Computes the geometric gradient matrix G (9 x 30).
        Maps 30 nodal displacements to 9 displacement gradients.
        """
        G = np.zeros((9, 30), dtype=np.float64)

        for i in range(10):
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

    def compute_linear_stiffness_matrix(
        self,
        node_coords: np.ndarray,
        quadrature_order: int | None = None,
    ) -> np.ndarray:
        """
        Computes the 30x30 linear element stiffness matrix k_e via numerical quadrature:
            k_e = sum_{q=1}^{N_qp} w_q * (B_L^T @ C @ B_L) * det(J_q)
        """
        order = (
            self._default_quadrature_order
            if quadrature_order is None
            else quadrature_order
        )
        qp, qw = get_tetrahedral_quadrature(order)
        C = self._material.compute_constitutive_matrix()

        k_e = np.zeros((30, 30), dtype=np.float64)

        for q in range(len(qw)):
            dN_dxi = self.evaluate_shape_gradients_ref(*qp[q])
            J = node_coords.T @ dN_dxi
            detJ = np.linalg.det(J)
            if detJ <= 0.0:
                raise ValueError(
                    f"Negative or zero Jacobian determinant: det(J) = {detJ:.4e}."
                )

            H = dN_dxi @ np.linalg.inv(J)
            BL = self.compute_linear_B_matrix(H)
            k_e += qw[q] * (BL.T @ C @ BL) * detJ

        return k_e

    def compute_mass_matrix(self, volume: float) -> np.ndarray:
        """
        Computes the 30x30 consistent element mass matrix m_e using the exact
        analytical reference mass tensor scaled by rho * volume.

        Parameters:
        -----------
        volume : float
            Element reference volume V_e.

        Returns:
        --------
        m_e : np.ndarray of shape (30, 30)
            Exact consistent element mass matrix.
        """
        rho = self._material.mass_density
        factor = rho * volume

        I3 = np.identity(3, dtype=np.float64)
        m_e = np.zeros((30, 30), dtype=np.float64)

        for i in range(10):
            for j in range(10):
                weight = _TET10_REFERENCE_MASS_MATRIX[i, j]
                m_e[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = (
                    weight * factor * I3
                )

        return m_e

