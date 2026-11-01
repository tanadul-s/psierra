from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

from psierra.element.shell3 import Shell3
from psierra.material.base import Material


class CorotationalShell3(Shell3):
    """
    Corotational 3-Node Flat Triangular Shell Element (DKT Bending + CST Membrane + Drilling).

    Decomposes element motion into finite rigid-body motion (translation + finite 3D rotation)
    and a purely deformational displacement field in the local corotating element frame.
    This formulation filters out large rigid-body rotations, eliminating geometric distortion
    and parasitic stress stiffening while retaining the linear DKT/CST formulation locally.

    Kinematics:
      - Initial reference vertices: X_1, X_2, X_3 in R^3
      - Initial reference frame   : Lambda_0 = [e1_0, e2_0, e3_0]^T in SO(3)
      - Current deformed vertices : x_i = X_i + u_i
      - Current deformed frame    : Lambda_def = [e1_def, e2_def, e3_def]^T in SO(3)
      - Element rigid rotation    : R = Lambda_def^T @ Lambda_0 in SO(3)
      - Local deformational disp  : u_l,i = Lambda_def @ (x_i - x_1) - Lambda_0 @ (X_i - X_1)
      - Local deformational rot   : theta_l,i = rotvec(Lambda_0 @ (R^T @ R_i) @ Lambda_0^T)
      - Local internal forces     : f_l = K_0,loc @ d_l
      - Global internal forces    : f_int = T_def^T @ f_l
      - Warped tangent stiffness  : K_tan = T_def^T @ K_0,loc @ T_def = R_blk @ K_0 @ R_blk^T

    Properties:
      - Exactly zero internal forces under arbitrary 3D rigid translations and rotations.
      - Exactly 6 zero eigenvalues corresponding to the 6 rigid-body modes of a 3D shell.
      - Unconditionally symmetric and positive semi-definite tangent stiffness matrix K_tan.
      - Fully compatible with NonlinearStaticAnalysis and NonlinearDynamicAnalysis.
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

    # =========================================================================
    # Corotational Frame & Kinematic State Decomposition
    # =========================================================================

    def compute_local_reference_stiffness(self, node_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the 18x18 local stiffness matrix K_0,loc and the reference transformation T_0.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (3, 3)
            Reference Cartesian coordinates of the 3 vertices.

        Returns:
        --------
        K_loc : np.ndarray of shape (18, 18)
            Reference stiffness matrix in the element local coordinate system.
        Lambda_0 : np.ndarray of shape (3, 3)
            Reference orthonormal coordinate transformation matrix.
        """
        Lambda_0, _, _ = self.compute_local_coordinate_system(node_coords)
        K_0 = self.compute_linear_stiffness_matrix(node_coords)

        T_blk0 = np.zeros((6, 6), dtype=np.float64)
        T_blk0[:3, :3] = Lambda_0
        T_blk0[3:, 3:] = Lambda_0

        T_0 = np.zeros((18, 18), dtype=np.float64)
        for i in range(3):
            T_0[6 * i : 6 * (i + 1), 6 * i : 6 * (i + 1)] = T_blk0

        K_loc = T_0 @ K_0 @ T_0.T
        return 0.5 * (K_loc + K_loc.T), Lambda_0

    def extract_element_rotation(
        self,
        node_coords: np.ndarray,
        u_nodes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts the element rigid-body rotation matrix R in SO(3), reference frame Lambda_0,
        and deformed frame Lambda_def.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (3, 3)
            Reference Cartesian coordinates X.
        u_nodes : np.ndarray of shape (3, 6) or (18,)
            Current nodal displacements and rotations [u_x, u_y, u_z, th_x, th_y, th_z].

        Returns:
        --------
        R : np.ndarray of shape (3, 3)
            Rigid-body rotation matrix in SO(3).
        Lambda_0 : np.ndarray of shape (3, 3)
            Reference transformation matrix.
        Lambda_def : np.ndarray of shape (3, 3)
            Deformed transformation matrix.
        """
        u_mat = np.asarray(u_nodes, dtype=np.float64).reshape(3, 6)
        x = node_coords + u_mat[:, :3]

        Lambda_0, _, _ = self.compute_local_coordinate_system(node_coords)
        Lambda_def, _, _ = self.compute_local_coordinate_system(x)

        # Rigid rotation R maps reference basis vectors E_0 to deformed basis vectors E_def
        # E_0 = Lambda_0.T, E_def = Lambda_def.T => R = E_def @ E_0.T = Lambda_def.T @ Lambda_0
        R = Lambda_def.T @ Lambda_0
        return R, Lambda_0, Lambda_def

    # =========================================================================
    # Nonlinear Force & Tangent Stiffness Assembly
    # =========================================================================

    def compute_nonlinear_force_and_tangent_stiffness(
        self,
        node_coords: np.ndarray,
        u_nodes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the Corotational tangent stiffness matrix K_tan (18x18)
        and internal force vector f_int (18,) for the current deformed shell state.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (3, 3)
            Reference coordinates X of the 3 element vertices.
        u_nodes : np.ndarray of shape (3, 6) or (18,)
            Current nodal displacements and rotations [u_x, u_y, u_z, th_x, th_y, th_z].

        Returns:
        --------
        K_tan : np.ndarray of shape (18, 18)
            Warped corotational tangent stiffness matrix in global coordinates.
        f_int : np.ndarray of shape (18,)
            Internal nodal force vector in global coordinates.
        """
        u_mat = np.asarray(u_nodes, dtype=np.float64).reshape(3, 6)
        u_trans = u_mat[:, :3]
        u_rot = u_mat[:, 3:]

        # 1. Reference & Deformed Local Coordinates
        K_loc, Lambda_0 = self.compute_local_reference_stiffness(node_coords)

        x = node_coords + u_trans
        Lambda_def, _, _ = self.compute_local_coordinate_system(x)

        # Rigid rotation of the element frame
        R = Lambda_def.T @ Lambda_0

        # Local positions in respective element frames
        u_l = (x @ Lambda_def.T) - (node_coords @ Lambda_0.T)

        # 3. Local Deformational Rotations
        # Extracts the relative nodal rotation with respect to the corotated frame
        th_l = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            th_vec = u_rot[i]
            norm_th = np.linalg.norm(th_vec)
            if norm_th < 1e-12:
                R_i = np.eye(3, dtype=np.float64)
            else:
                R_i = Rotation.from_rotvec(th_vec).as_matrix()

            # Relative rotation in global frame: R_rel = R.T @ R_i
            # Expressed in element local frame: R_loc = Lambda_0 @ R_rel @ Lambda_0.T
            R_rel_loc = Lambda_0 @ (R.T @ R_i) @ Lambda_0.T
            th_l[i] = Rotation.from_matrix(R_rel_loc).as_rotvec()

        # 4. Assemble 18-element Local Deformational Vector d_l
        d_l = np.zeros(18, dtype=np.float64)
        for i in range(3):
            d_l[6 * i : 6 * i + 3] = u_l[i]
            d_l[6 * i + 3 : 6 * i + 6] = th_l[i]

        # 5. Local Internal Forces
        f_l = K_loc @ d_l

        # 6. Global Transformation
        T_blk_def = np.zeros((6, 6), dtype=np.float64)
        T_blk_def[:3, :3] = Lambda_def
        T_blk_def[3:, 3:] = Lambda_def

        T_def = np.zeros((18, 18), dtype=np.float64)
        for i in range(3):
            T_def[6 * i : 6 * (i + 1), 6 * i : 6 * (i + 1)] = T_blk_def

        f_int = T_def.T @ f_l
        K_tan = T_def.T @ K_loc @ T_def
        K_tan = 0.5 * (K_tan + K_tan.T)

        return K_tan, f_int
