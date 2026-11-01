from __future__ import annotations
import numpy as np

from psierra.element.base import Element
from psierra.element.quadrature import get_triangle_quadrature
from psierra.material.base import Material


class Shell3(Element):
    """
    3-Node Flat Triangular Shell Element (DKT Bending + CST Membrane + Drilling).

    Couples in-plane membrane stretching, out-of-plane DKT (Discrete Kirchhoff Triangle)
    plate bending, and Hughes-Brezzi stabilized drilling rotations.

    Degrees of Freedom per Node (6 DOFs):
      [u_x, u_y, u_z, theta_x, theta_y, theta_z]
      - (u_x, u_y, u_z): 3D translational displacements
      - (theta_x, theta_y, theta_z): 3D rotational angles (right-hand rule)

    Total Element DOFs: 3 nodes * 6 DOFs = 18 DOFs.

    Parameters:
    -----------
    material : Material
        Constitutive material model providing elastic parameters (E, nu, rho).
    thickness : float
        Physical shell thickness t (e.g. 0.0005 m for 0.5 mm dura mater).
    drilling_factor : float, default=1e-3
        Non-dimensional stabilization parameter for in-plane drilling rotation (theta_z).
    """

    def __init__(
        self,
        material: Material,
        thickness: float,
        drilling_factor: float = 1e-3,
    ) -> None:
        super().__init__(material)
        if thickness <= 0.0:
            raise ValueError(f"Shell thickness must be strictly positive, got {thickness}.")
        self._thickness = float(thickness)
        self._drilling_factor = float(drilling_factor)
        self._nodes_per_element = 3
        self._dofs_per_node = 6
        self._dofs_per_element = 18

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def thickness(self) -> float:
        """Physical shell wall thickness t."""
        return self._thickness

    @thickness.setter
    def thickness(self, val: float) -> None:
        if val <= 0.0:
            raise ValueError(f"Shell thickness must be positive, got {val}.")
        self._thickness = float(val)

    @property
    def drilling_factor(self) -> float:
        """Drilling rotation stabilization parameter."""
        return self._drilling_factor

    @property
    def nodes_per_element(self) -> int:
        return self._nodes_per_element

    @property
    def dofs_per_node(self) -> int:
        return self._dofs_per_node

    @property
    def dofs_per_element(self) -> int:
        return self._dofs_per_element

    # =========================================================================
    # Local Coordinate System & Geometry
    # =========================================================================

    @staticmethod
    def compute_local_coordinate_system(
        node_coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Constructs an orthonormal local coordinate system [e1, e2, e3] and computes
        local planar coordinates for the 3 vertices.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (3, 3)
            Global Cartesian coordinates of the 3 triangle vertices.

        Returns:
        --------
        Lambda : np.ndarray of shape (3, 3)
            Transformation matrix where rows are [e1, e2, e3].
        X_loc : np.ndarray of shape (3, 2)
            In-plane local coordinates (x', y') for the 3 vertices.
        area : float
            Surface area of the triangle.
        """
        if node_coords.shape != (3, 3):
            raise ValueError(f"Expected node_coords of shape (3, 3), got {node_coords.shape}.")

        v12 = node_coords[1] - node_coords[0]
        v13 = node_coords[2] - node_coords[0]

        normal = np.cross(v12, v13)
        normal_len = np.linalg.norm(normal)
        if normal_len <= 1e-15:
            raise ValueError("Collinear or degenerate triangle vertices detected (zero surface area).")

        area = 0.5 * normal_len
        e3 = normal / normal_len
        e1 = v12 / np.linalg.norm(v12)
        e2 = np.cross(e3, e1)

        Lambda = np.array([e1, e2, e3], dtype=np.float64)

        # In-plane coordinates: X_loc[i] = (node_coords[i] - node_coords[0]) @ Lambda.T
        rel_coords = node_coords - node_coords[0]
        X_loc_3d = rel_coords @ Lambda.T
        X_loc_2d = X_loc_3d[:, :2]

        return Lambda, X_loc_2d, float(area)

    # =========================================================================
    # Linear Stiffness Assembly
    # =========================================================================

    def compute_linear_stiffness_matrix(self, node_coords: np.ndarray) -> np.ndarray:
        """
        Computes the 18x18 global linear shell stiffness matrix K_e in 3D Cartesian coordinates.

        Integrates:
          1. Constant Strain Triangle (CST) membrane stiffness (6x6).
          2. Hughes-Brezzi drilling rotation stabilization (9x9 on u, v, theta_z).
          3. Discrete Kirchhoff Triangle (DKT) plate bending stiffness (9x9 on w, theta_x, theta_y).
          4. Transformation to global 3D degrees of freedom.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (3, 3)
            Reference nodal coordinates [X, Y, Z] of the 3 vertices.

        Returns:
        --------
        K_global : np.ndarray of shape (18, 18)
            Global symmetric element stiffness matrix.
        """
        Lambda, X_loc, area = self.compute_local_coordinate_system(node_coords)
        x = X_loc[:, 0]
        y = X_loc[:, 1]

        # Geometry helpers
        x21 = x[1] - x[0]
        y21 = y[1] - y[0]
        x31 = x[2] - x[0]
        y31 = y[2] - y[0]
        x32 = x[2] - x[1]
        y32 = y[2] - y[1]

        x12 = -x21
        y12 = -y21
        x23 = -x32
        y23 = -y32
        x13 = -x31
        y13 = -y31

        t = self._thickness
        E, nu = self._material.parameters.youngs_modulus, self._material.parameters.poissons_ratio
        G = E / (2.0 * (1.0 + nu))

        # ---------------------------------------------------------------------
        # 1. CST Membrane Stiffness (6x6)
        # ---------------------------------------------------------------------
        # B_m maps [u1, v1, u2, v2, u3, v3] to in-plane strains [eps_xx, eps_yy, gamma_xy]
        Bm = (1.0 / (2.0 * area)) * np.array(
            [
                [y23, 0.0, y31, 0.0, y12, 0.0],
                [0.0, x32, 0.0, x13, 0.0, x21],
                [x32, y23, x13, y31, x21, y12],
            ],
            dtype=np.float64,
        )

        Dm = (E * t / (1.0 - nu**2)) * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ],
            dtype=np.float64,
        )
        Km = area * (Bm.T @ Dm @ Bm)

        # ---------------------------------------------------------------------
        # 2. Hughes-Brezzi Drilling Rotation Stabilization (9x9)
        # ---------------------------------------------------------------------
        # Relates drilling rotation theta_z to in-plane rotation omega_z = 0.5 * (dv/dx - du/dy)
        r = 0.5 / (2.0 * area)
        row_omega = np.array(
            [-r * y23, r * x32, 0.0, -r * y31, r * x13, 0.0, -r * y12, r * x21, 0.0],
            dtype=np.float64,
        )
        B_drill = np.zeros((3, 9), dtype=np.float64)
        for i in range(3):
            B_drill[i] = row_omega
            B_drill[i, i * 3 + 2] = -1.0

        gamma_drill = self._drilling_factor * G * t * area
        K_drill_coupled = gamma_drill * (B_drill.T @ B_drill)

        # ---------------------------------------------------------------------
        # 3. DKT Plate Bending Stiffness (9x9)
        # ---------------------------------------------------------------------
        l4_sq = x12**2 + y12**2
        l5_sq = x23**2 + y23**2
        l6_sq = x31**2 + y31**2

        a4 = -x12 / l4_sq
        b4 = 0.75 * x12 * y12 / l4_sq
        c4 = (0.25 * x12**2 - 0.5 * y12**2) / l4_sq
        d4 = -y12 / l4_sq
        e4 = (0.25 * y12**2 - 0.5 * x12**2) / l4_sq

        a5 = -x23 / l5_sq
        b5 = 0.75 * x23 * y23 / l5_sq
        c5 = (0.25 * x23**2 - 0.5 * y23**2) / l5_sq
        d5 = -y23 / l5_sq
        e5 = (0.25 * y23**2 - 0.5 * x23**2) / l5_sq

        a6 = -x31 / l6_sq
        b6 = 0.75 * x31 * y31 / l6_sq
        c6 = (0.25 * x31**2 - 0.5 * y31**2) / l6_sq
        d6 = -y31 / l6_sq
        e6 = (0.25 * y31**2 - 0.5 * x31**2) / l6_sq

        def get_dkt_Bb(xi: float, eta: float) -> np.ndarray:
            L1 = 1.0 - xi - eta
            L2 = xi
            L3 = eta

            dN_dxi = np.array(
                [-(4 * L1 - 1), 4 * L2 - 1, 0.0, 4 * (L1 - L2), 4 * L3, -4 * L3],
                dtype=np.float64,
            )
            dN_deta = np.array(
                [-(4 * L1 - 1), 0.0, 4 * L3 - 1, -4 * L2, 4 * L2, 4 * (L1 - L3)],
                dtype=np.float64,
            )

            def eval_H_from_dN(dN: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                N1, N2, N3, N4, N5, N6 = dN
                Hx = np.zeros(9, dtype=np.float64)
                Hy = np.zeros(9, dtype=np.float64)

                # Node 1
                Hx[0] = 1.5 * (a6 * N6 - a4 * N4)
                Hx[1] = b6 * N6 + b4 * N4
                Hx[2] = N1 - c6 * N6 - c4 * N4
                Hy[0] = 1.5 * (d6 * N6 - d4 * N4)
                Hy[1] = -N1 + e6 * N6 + e4 * N4
                Hy[2] = -b6 * N6 - b4 * N4

                # Node 2
                Hx[3] = 1.5 * (a4 * N4 - a5 * N5)
                Hx[4] = b4 * N4 + b5 * N5
                Hx[5] = N2 - c4 * N4 - c5 * N5
                Hy[3] = 1.5 * (d4 * N4 - d5 * N5)
                Hy[4] = -N2 + e4 * N4 + e5 * N5
                Hy[5] = -b4 * N4 - b5 * N5

                # Node 3
                Hx[6] = 1.5 * (a5 * N5 - a6 * N6)
                Hx[7] = b5 * N5 + b6 * N6
                Hx[8] = N3 - c5 * N5 - c6 * N6
                Hy[6] = 1.5 * (d5 * N5 - d6 * N6)
                Hy[7] = -N3 + e5 * N5 + e6 * N6
                Hy[8] = -b5 * N5 - b6 * N6

                return Hx, Hy

            dHx_dxi, dHy_dxi = eval_H_from_dN(dN_dxi)
            dHx_deta, dHy_deta = eval_H_from_dN(dN_deta)

            fac = 1.0 / (2.0 * area)
            dHx_dx = fac * (y31 * dHx_dxi - y21 * dHx_deta)
            dHx_dy = fac * (-x31 * dHx_dxi + x21 * dHx_deta)
            dHy_dx = fac * (y31 * dHy_dxi - y21 * dHy_deta)
            dHy_dy = fac * (-x31 * dHy_dxi + x21 * dHy_deta)

            Bb = np.zeros((3, 9), dtype=np.float64)
            Bb[0] = dHx_dx
            Bb[1] = dHy_dy
            Bb[2] = dHx_dy + dHy_dx
            return Bb

        D_factor = (E * t**3) / (12.0 * (1.0 - nu**2))
        Db = D_factor * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ],
            dtype=np.float64,
        )

        qp, qw = get_triangle_quadrature(3)
        Kb = np.zeros((9, 9), dtype=np.float64)
        for q in range(3):
            Bb = get_dkt_Bb(qp[q, 0], qp[q, 1])
            Kb += 2.0 * area * qw[q] * (Bb.T @ Db @ Bb)

        # ---------------------------------------------------------------------
        # 4. Assemble 18x18 Local Stiffness Matrix
        # ---------------------------------------------------------------------
        # Local DOF mapping for each node i: [u, v, w, th_x, th_y, th_z]
        # In-plane DOFs: u(0), v(1), th_z(5), u(6), v(7), th_z(11), u(12), v(13), th_z(17)
        K_local = np.zeros((18, 18), dtype=np.float64)

        # CST membrane in (u, v)
        m_dofs = [0, 1, 6, 7, 12, 13]
        K_local[np.ix_(m_dofs, m_dofs)] = Km

        # Hughes-Brezzi drilling in (u, v, th_z)
        ip_dofs = [0, 1, 5, 6, 7, 11, 12, 13, 17]
        K_local[np.ix_(ip_dofs, ip_dofs)] += K_drill_coupled

        # DKT bending in (w, th_x, th_y)
        b_dofs = [2, 3, 4, 8, 9, 10, 14, 15, 16]
        K_local[np.ix_(b_dofs, b_dofs)] = Kb

        # Enforce exact symmetry in float64
        K_local = 0.5 * (K_local + K_local.T)

        # ---------------------------------------------------------------------
        # 5. Global 3D Coordinate Transformation
        # ---------------------------------------------------------------------
        T_block = np.zeros((6, 6), dtype=np.float64)
        T_block[:3, :3] = Lambda
        T_block[3:, 3:] = Lambda

        T = np.zeros((18, 18), dtype=np.float64)
        T[0:6, 0:6] = T_block
        T[6:12, 6:12] = T_block
        T[12:18, 12:18] = T_block

        K_global = T.T @ K_local @ T
        return 0.5 * (K_global + K_global.T)

    # =========================================================================
    # Mass Matrix Assembly
    # =========================================================================

    def compute_mass_matrix(
        self,
        volume: float | None = None,
        area: float | None = None,
        node_coords: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Computes the 18x18 consistent shell mass matrix M_e in 3D global coordinates.

        Parameters:
        -----------
        volume : float, optional
            Element volume V = area * thickness.
        area : float, optional
            Surface area of the triangle.
        node_coords : np.ndarray of shape (3, 3), optional
            Coordinates of the 3 vertices.

        Returns:
        --------
        M_global : np.ndarray of shape (18, 18)
        """
        if area is None:
            if volume is not None:
                area = volume / self._thickness
            elif node_coords is not None:
                _, _, area = self.compute_local_coordinate_system(node_coords)
            else:
                raise ValueError("Must provide either area, volume, or node_coords to compute mass matrix.")

        rho = self._material.mass_density
        total_mass = rho * self._thickness * area

        # Consistent mass matrix for membrane/translational DOFs:
        # M_trans_block = (total_mass / 12) * [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
        I3 = np.identity(3, dtype=np.float64)
        M_local = np.zeros((18, 18), dtype=np.float64)

        # Translational DOFs (u, v, w) for nodes 1, 2, 3:
        # Node i translations at indices 6*i + [0, 1, 2]
        m_diag = (2.0 / 12.0) * total_mass
        m_off = (1.0 / 12.0) * total_mass

        for i in range(3):
            for j in range(3):
                weight = m_diag if i == j else m_off
                M_local[i * 6 : i * 6 + 3, j * 6 : j * 6 + 3] = weight * I3

        # Rotational inertia DOFs: (theta_x, theta_y, theta_z) at indices 6*i + [3, 4, 5]
        # Rotational inertia per node: (rho * t^3 * area) / 36
        rot_inertia = (rho * (self._thickness**3) * area) / 36.0
        for i in range(3):
            M_local[i * 6 + 3 : i * 6 + 6, i * 6 + 3 : i * 6 + 6] = rot_inertia * I3

        return M_local

