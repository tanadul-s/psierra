from __future__ import annotations
import numpy as np

from psierra.element.base import Element
from psierra.element.quadrature import get_triangle_quadrature
from psierra.material.base import Material


class Shell6Base(Element):
    """
    Base class for 6-Node Conforming Quadratic Triangular Shell Elements.

    Conforms directly to quadratic tetrahedral solid elements (Tet10) boundary faces,
    guaranteeing C^0 displacement continuity across the solid-shell interface without
    spurious interfacial gaps or kinematic locking.

    Couples:
      1. Linear Strain Triangle (LST) quadratic membrane stretching (12 DOFs).
      2. Reissner-Mindlin quadratic plate bending with shear deformation (18 DOFs).
      3. Hughes-Brezzi drilling rotation stabilization across all 6 nodes (6 DOFs).

    Node Topology:
      - Nodes 0, 1, 2: Corner vertices.
      - Node 3: Mid-edge between vertex 0 and vertex 1.
      - Node 4: Mid-edge between vertex 1 and vertex 2.
      - Node 5: Mid-edge between vertex 2 and vertex 0.

    Degrees of Freedom per Node (6 DOFs):
      [u_x, u_y, u_z, theta_x, theta_y, theta_z]
      - (u_x, u_y, u_z): 3D translational displacements
      - (theta_x, theta_y, theta_z): 3D right-hand-rule rotation angles

    Total Element DOFs: 6 nodes * 6 DOFs = 36 DOFs.

    Parameters:
    -----------
    material : Material
        Constitutive material model providing elastic parameters (E, nu, rho).
    thickness : float
        Physical shell thickness t (e.g. 0.0005 m for dura mater or 0.004 m for calvarium).
    drilling_factor : float, default=1e-3
        Non-dimensional stabilization parameter for in-plane drilling rotation (theta_z).
    shear_correction_factor : float, default=5.0/6.0
        Reissner-Mindlin transverse shear correction coefficient kappa_s.
    """

    def __init__(
        self,
        material: Material,
        thickness: float,
        drilling_factor: float = 1e-3,
        shear_correction_factor: float = 5.0 / 6.0,
    ) -> None:
        super().__init__(material)
        if thickness <= 0.0:
            raise ValueError(f"Shell thickness must be strictly positive, got {thickness}.")
        self._thickness = float(thickness)
        self._drilling_factor = float(drilling_factor)
        self._kappa_s = float(shear_correction_factor)
        self._nodes_per_element = 6
        self._dofs_per_node = 6
        self._dofs_per_element = 36

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
    def shear_correction_factor(self) -> float:
        """Reissner-Mindlin shear correction factor kappa_s."""
        return self._kappa_s

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
    # Shape Functions & Geometric Derivatives
    # =========================================================================

    @staticmethod
    def evaluate_shape_functions(xi: float, eta: float) -> np.ndarray:
        """
        Evaluates the 6 quadratic Lagrange shape functions on the master reference triangle:
          Omega_ref = {(xi, eta) : xi >= 0, eta >= 0, xi + eta <= 1}

        Parameters:
        -----------
        xi, eta : float
            Natural coordinates on the reference master triangle.

        Returns:
        --------
        N : np.ndarray of shape (6,)
        """
        L1 = 1.0 - xi - eta
        L2 = xi
        L3 = eta
        return np.array(
            [
                L1 * (2.0 * L1 - 1.0),  # Node 0 (corner 0: xi=0, eta=0)
                L2 * (2.0 * L2 - 1.0),  # Node 1 (corner 1: xi=1, eta=0)
                L3 * (2.0 * L3 - 1.0),  # Node 2 (corner 2: xi=0, eta=1)
                4.0 * L1 * L2,          # Node 3 (mid 0-1: xi=0.5, eta=0)
                4.0 * L2 * L3,          # Node 4 (mid 1-2: xi=0.5, eta=0.5)
                4.0 * L3 * L1,          # Node 5 (mid 2-0: xi=0, eta=0.5)
            ],
            dtype=np.float64,
        )

    @staticmethod
    def evaluate_shape_gradients_ref(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the parametric derivatives [dN/dxi, dN/deta] on the reference triangle.
        """
        L1 = 1.0 - xi - eta
        L2 = xi
        L3 = eta

        dN_dxi = np.array(
            [
                -(4.0 * L1 - 1.0),
                4.0 * L2 - 1.0,
                0.0,
                4.0 * (L1 - L2),
                4.0 * L3,
                -4.0 * L3,
            ],
            dtype=np.float64,
        )
        dN_deta = np.array(
            [
                -(4.0 * L1 - 1.0),
                0.0,
                4.0 * L3 - 1.0,
                -4.0 * L2,
                4.0 * L2,
                4.0 * (L1 - L3),
            ],
            dtype=np.float64,
        )
        return dN_dxi, dN_deta

    @classmethod
    def compute_shape_gradients_physical(
        cls,
        xi: float,
        eta: float,
        local_coords_2d: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Computes spatial derivatives [dN/dx, dN/dy] and the planar Jacobian determinant det(J).

        Parameters:
        -----------
        xi, eta : float
            Evaluation natural coordinates.
        local_coords_2d : np.ndarray of shape (6, 2)
            Local planar Cartesian coordinates of the 6 nodes.

        Returns:
        --------
        dN_dxy : np.ndarray of shape (2, 6)
            Spatial derivatives row 0 = dN/dx, row 1 = dN/dy.
        detJ : float
            Jacobian determinant of the isoparametric coordinate mapping.
        """
        dN_dxi, dN_deta = cls.evaluate_shape_gradients_ref(xi, eta)

        # 2x2 Jacobian: J = [dx/dxi, dy/dxi; dx/deta, dy/deta]
        J = np.zeros((2, 2), dtype=np.float64)
        J[0, 0] = np.dot(dN_dxi, local_coords_2d[:, 0])
        J[0, 1] = np.dot(dN_dxi, local_coords_2d[:, 1])
        J[1, 0] = np.dot(dN_deta, local_coords_2d[:, 0])
        J[1, 1] = np.dot(dN_deta, local_coords_2d[:, 1])

        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if detJ <= 1e-15:
            raise ValueError(
                f"Degenerate or inverted shell element geometry detected (detJ = {detJ:.3e} <= 0)."
            )

        invJ = (1.0 / detJ) * np.array(
            [
                [J[1, 1], -J[0, 1]],
                [-J[1, 0], J[0, 0]],
            ],
            dtype=np.float64,
        )
        dN_dxy = invJ @ np.vstack([dN_dxi, dN_deta])
        return dN_dxy, float(detJ)

    # =========================================================================
    # Local Coordinate System & Geometry
    # =========================================================================

    @staticmethod
    def compute_local_coordinate_system(
        node_coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Constructs an orthonormal local coordinate system [e1, e2, e3] and computes
        local planar coordinates for the 6 nodes.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (6, 3)
            Global Cartesian coordinates of the 6 nodes.

        Returns:
        --------
        Lambda : np.ndarray of shape (3, 3)
            Orthonormal transformation matrix where rows are [e1, e2, e3].
        X_loc : np.ndarray of shape (6, 2)
            In-plane local coordinates (x', y') for the 6 nodes.
        area : float
            Surface area computed from corner vertices.
        """
        if node_coords.shape != (6, 3):
            raise ValueError(f"Expected node_coords of shape (6, 3), got {node_coords.shape}.")

        # Reference plane defined by the 3 corner vertices
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
        Computes the 36x36 global linear shell stiffness matrix K_e in 3D Cartesian coordinates.

        Integrates:
          1. Linear Strain Triangle (LST) quadratic membrane stiffness (12 DOFs).
          2. Reissner-Mindlin plate bending and transverse shear stiffness (18 DOFs).
          3. Hughes-Brezzi drilling rotation stabilization (6 DOFs).
          4. Global 3D coordinate transformation.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (6, 3)
            Reference nodal coordinates [X, Y, Z] of the 6 nodes.

        Returns:
        --------
        K_global : np.ndarray of shape (36, 36)
            Global symmetric element stiffness matrix.
        """
        Lambda, X_loc, area = self.compute_local_coordinate_system(node_coords)
        t = self._thickness
        E = self._material.parameters.youngs_modulus
        nu = self._material.parameters.poissons_ratio
        G = E / (2.0 * (1.0 + nu))

        # Quadrature for quadratic triangle: 3-point Dunavant / Hammer rule (exact for quadratics)
        qp, qw = get_triangle_quadrature(3)

        K_local = np.zeros((36, 36), dtype=np.float64)

        # ---------------------------------------------------------------------
        # 1. Constitutive Matrices
        # ---------------------------------------------------------------------
        # Membrane plane stress (3x3)
        Dm = (E * t / (1.0 - nu**2)) * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ],
            dtype=np.float64,
        )

        # Plate bending (3x3)
        Db = ((E * t**3) / (12.0 * (1.0 - nu**2))) * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ],
            dtype=np.float64,
        )

        # Transverse shear (2x2)
        Ds = self._kappa_s * G * t * np.identity(2, dtype=np.float64)

        # ---------------------------------------------------------------------
        # 2. Quadrature Integration of Membrane, Bending, and Shear
        # ---------------------------------------------------------------------
        for q in range(3):
            xi_q, eta_q = qp[q, 0], qp[q, 1]
            dN_dxy, detJ = self.compute_shape_gradients_physical(xi_q, eta_q, X_loc)
            N = self.evaluate_shape_functions(xi_q, eta_q)

            # B_m: membrane strain-displacement matrix (3x36)
            Bm = np.zeros((3, 36), dtype=np.float64)
            # B_b: bending curvature matrix (3x36)
            Bb = np.zeros((3, 36), dtype=np.float64)
            # B_s: transverse shear matrix (2x36)
            Bs = np.zeros((2, 36), dtype=np.float64)

            for i in range(6):
                # Node i DOF indices in local system:
                # 6*i + 0: u
                # 6*i + 1: v
                # 6*i + 2: w
                # 6*i + 3: th_x
                # 6*i + 4: th_y
                # 6*i + 5: th_z

                # Membrane strains: eps_xx = du/dx, eps_yy = dv/dy, gamma_xy = du/dy + dv/dx
                Bm[0, 6 * i] = dN_dxy[0, i]
                Bm[1, 6 * i + 1] = dN_dxy[1, i]
                Bm[2, 6 * i] = dN_dxy[1, i]
                Bm[2, 6 * i + 1] = dN_dxy[0, i]

                # Bending curvatures: kappa_xx = d(th_y)/dx, kappa_yy = -d(th_x)/dy,
                #                     kappa_xy = d(th_y)/dy - d(th_x)/dx
                Bb[0, 6 * i + 4] = dN_dxy[0, i]
                Bb[1, 6 * i + 3] = -dN_dxy[1, i]
                Bb[2, 6 * i + 3] = -dN_dxy[0, i]
                Bb[2, 6 * i + 4] = dN_dxy[1, i]

                # Transverse shear: gamma_xz = dw/dx + th_y, gamma_yz = dw/dy - th_x
                Bs[0, 6 * i + 2] = dN_dxy[0, i]
                Bs[0, 6 * i + 4] = N[i]
                Bs[1, 6 * i + 2] = dN_dxy[1, i]
                Bs[1, 6 * i + 3] = -N[i]

            weight = qw[q] * detJ
            K_local += weight * (Bm.T @ Dm @ Bm + Bb.T @ Db @ Bb + Bs.T @ Ds @ Bs)

        # ---------------------------------------------------------------------
        # 3. Hughes-Brezzi Drilling Rotation Stabilization (6 nodes)
        # ---------------------------------------------------------------------
        # At each of the 6 nodes, couples the drilling rotation th_z to the in-plane
        # rotation omega_z = 0.5 * (dv/dx - du/dy)
        node_xi_eta = [
            (0.0, 0.0),      # Node 0
            (1.0, 0.0),      # Node 1
            (0.0, 1.0),      # Node 2
            (0.5, 0.0),      # Node 3
            (0.5, 0.5),      # Node 4
            (0.0, 0.5),      # Node 5
        ]
        gamma_drill = self._drilling_factor * G * t * (area / 6.0)
        for k in range(6):
            xi_k, eta_k = node_xi_eta[k]
            dN_dxy, _ = self.compute_shape_gradients_physical(xi_k, eta_k, X_loc)
            B_drill = np.zeros((1, 36), dtype=np.float64)
            for j in range(6):
                B_drill[0, 6 * j] = -0.5 * dN_dxy[1, j]      # -0.5 * du/dy
                B_drill[0, 6 * j + 1] = 0.5 * dN_dxy[0, j]   # +0.5 * dv/dx
            B_drill[0, 6 * k + 5] = -1.0                      # - th_zk
            K_local += gamma_drill * (B_drill.T @ B_drill)

        # Symmetrize local stiffness matrix
        K_local = 0.5 * (K_local + K_local.T)

        # ---------------------------------------------------------------------
        # 4. Global 3D Coordinate Transformation
        # ---------------------------------------------------------------------
        T_block = np.zeros((6, 6), dtype=np.float64)
        T_block[:3, :3] = Lambda
        T_block[3:, 3:] = Lambda

        T = np.zeros((36, 36), dtype=np.float64)
        for i in range(6):
            T[i * 6 : (i + 1) * 6, i * 6 : (i + 1) * 6] = T_block

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
        Computes the 36x36 consistent shell mass matrix M_e in 3D global coordinates.

        Parameters:
        -----------
        volume : float, optional
            Element volume V = area * thickness.
        area : float, optional
            Surface area of the triangle.
        node_coords : np.ndarray of shape (6, 3), optional
            Coordinates of the 6 nodes.

        Returns:
        --------
        M_global : np.ndarray of shape (36, 36)
        """
        if area is None:
            if volume is not None:
                area = volume / self._thickness
            elif node_coords is not None:
                _, _, area = self.compute_local_coordinate_system(node_coords)
            else:
                raise ValueError("Must provide either area, volume, or node_coords to compute mass matrix.")

        rho = self._material.mass_density
        t = self._thickness

        qp, qw = get_triangle_quadrature(3)
        M_scalar = np.zeros((6, 6), dtype=np.float64)
        for q in range(3):
            N = self.evaluate_shape_functions(qp[q, 0], qp[q, 1])
            # dA = 2 * area * qw[q]
            M_scalar += (2.0 * area * qw[q]) * np.outer(N, N)

        I3 = np.identity(3, dtype=np.float64)
        M_local = np.zeros((36, 36), dtype=np.float64)

        # Translational DOFs: rho * t * int N_i N_j dA
        for i in range(6):
            for j in range(6):
                m_trans_ij = rho * t * M_scalar[i, j]
                M_local[i * 6 : i * 6 + 3, j * 6 : j * 6 + 3] = m_trans_ij * I3

        # Rotational inertia DOFs: (rho * t^3 / 12) * int N_i N_j dA
        rot_factor = (rho * (t**3)) / 12.0
        for i in range(6):
            for j in range(6):
                m_rot_ij = rot_factor * M_scalar[i, j]
                M_local[i * 6 + 3 : i * 6 + 6, j * 6 + 3 : j * 6 + 6] = m_rot_ij * I3

        return 0.5 * (M_local + M_local.T)

