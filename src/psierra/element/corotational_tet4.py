from __future__ import annotations
import numpy as np

from psierra.element.tet4 import Tet4
from psierra.material.base import Material


class CorotationalTet4(Tet4):
    """
    Corotational 4-Node Linear Tetrahedral Solid Element.

    Decomposes element motion into a rigid-body motion (translation + finite rotation R)
    and a purely deformational displacement field u_l in the local corotating frame.
    This formulation eliminates artificial volume dilation (ballooning) under large rotations
    while retaining the speed, robustness, and simplicity of linear elasticity locally.

    Kinematics:
      - Deformed coordinates: x = X + u
      - Deformation gradient: F = x^T @ H  (strictly constant across element)
      - Polar decomposition : F = R @ U  (R in SO(3), U symmetric positive-definite stretch)
      - Local unrotated coordinates: p = x @ R  (p_i = R^T @ x_i)
      - Local deformational displacement: u_l = p - X
      - Local elastic forces: f_l = K_0 @ u_l
      - Global internal force: f_int = diag(R, R, R, R) @ f_l
      - Global tangent stiffness: K_tan = diag(R, R, R, R) @ K_0 @ diag(R, R, R, R)^T
    """

    def __init__(self, material: Material) -> None:
        super().__init__(material)

    # =========================================================================
    # Polar Decomposition & Rotation Extraction
    # =========================================================================

    @staticmethod
    def compute_polar_decomposition(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the polar decomposition of the 3x3 deformation gradient F = R @ U.

        Uses Singular Value Decomposition (SVD):
          F = W @ diag(s) @ V^T
          R = W @ V^T
          U = V @ diag(s) @ V^T

        Ensures R is a pure, right-handed proper rotation in SO(3) with det(R) = +1.

        Parameters:
        -----------
        F : np.ndarray of shape (3, 3)
            Deformation gradient tensor.

        Returns:
        --------
        R : np.ndarray of shape (3, 3)
            Rigid body rotation tensor in SO(3).
        U : np.ndarray of shape (3, 3)
            Right stretch tensor (symmetric positive-definite).
        """
        W, s, Vt = np.linalg.svd(F)
        R = W @ Vt

        # Safeguard against reflection/mirroring (ensure det(R) = +1)
        if np.linalg.det(R) < 0.0:
            W_mod = W.copy()
            W_mod[:, -1] *= -1.0
            R = W_mod @ Vt
            s_mod = s.copy()
            s_mod[-1] *= -1.0
            U = Vt.T @ np.diag(s_mod) @ Vt
        else:
            U = Vt.T @ np.diag(s) @ Vt

        return R, U

    def extract_rotation(
        self, node_coords: np.ndarray, u_nodes: np.ndarray
    ) -> np.ndarray:
        """
        Extracts the rigid-body rotation matrix R in SO(3) of the tetrahedron.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (4, 3)
            Reference nodal coordinates X.
        u_nodes : np.ndarray of shape (4, 3)
            Current nodal displacements u.

        Returns:
        --------
        R : np.ndarray of shape (3, 3)
            Orthonormal rotation matrix representing element orientation.
        """
        _, H = self.compute_volume_and_gradients(node_coords)
        x = node_coords + u_nodes
        F = x.T @ H
        R, _ = self.compute_polar_decomposition(F)
        return R

    # =========================================================================
    # Nonlinear Force & Tangent Stiffness Assembly
    # =========================================================================

    def compute_nonlinear_force_and_tangent_stiffness(
        self, node_coords: np.ndarray, u_nodes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the Corotational tangent stiffness matrix k_tan (12x12)
        and internal force vector f_int (12,) for the current deformed state.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (4, 3)
            Reference coordinates X of the element vertices.
        u_nodes : np.ndarray of shape (4, 3)
            Current nodal displacements u of the element vertices.

        Returns:
        --------
        k_tan : np.ndarray of shape (12, 12)
            Warped corotational tangent stiffness matrix: R_blk @ K_0 @ R_blk^T.
        f_int : np.ndarray of shape (12,)
            Internal nodal force vector in global coordinates.
        """
        V, H = self.compute_volume_and_gradients(node_coords)

        # Deformed coordinates: x = X + u
        x = node_coords + u_nodes

        # Deformation gradient: F = x^T @ H (strictly constant across Tet4)
        F = x.T @ H
        J = np.linalg.det(F)
        if J <= 0.0:
            raise ValueError(f"Element inversion detected in CorotationalTet4: det(F) = {J:.4e} <= 0.")

        # Extract pure rigid rotation R in SO(3)
        R, _ = self.compute_polar_decomposition(F)

        # Unrotated local coordinates: p_i = R^T @ x_i  =>  p = x @ R
        p = x @ R

        # Purely deformational displacement in local frame: u_l = p - X
        u_l = (p - node_coords).reshape(-1)

        # Reference linear elastic stiffness matrix: K_0 = (B^T C B) * V
        K0 = self.compute_linear_stiffness_matrix(node_coords)

        # Local internal force: f_l = K_0 @ u_l
        f_l = K0 @ u_l

        # Construct 12x12 block diagonal rotation operator R_blk = diag(R, R, R, R)
        R_blk = np.zeros((12, 12), dtype=np.float64)
        for i in range(4):
            R_blk[i * 3 : (i + 1) * 3, i * 3 : (i + 1) * 3] = R

        # Global internal force: f_int = R_blk @ f_l
        f_int = R_blk @ f_l

        # Global tangent stiffness matrix: K_tan = R_blk @ K_0 @ R_blk^T
        # This warped stiffness is unconditionally symmetric and positive semi-definite
        k_tan = R_blk @ K0 @ R_blk.T

        return k_tan, f_int

    # =========================================================================
    # Corotational Stresses
    # =========================================================================

    def compute_corotational_stress(
        self, node_coords: np.ndarray, u_nodes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates local and rotated Cauchy stress and engineering strain tensors.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (4, 3)
            Reference nodal coordinates X.
        u_nodes : np.ndarray of shape (4, 3)
            Current nodal displacements u.

        Returns:
        --------
        sigma : np.ndarray of shape (3, 3)
            Cauchy stress tensor in global coordinates: sigma = R @ sigma_l @ R^T.
        strain : np.ndarray of shape (3, 3)
            Small strain tensor in global coordinates: eps = R @ eps_l @ R^T.
        R : np.ndarray of shape (3, 3)
            Element rigid rotation matrix in SO(3).
        """
        _, H = self.compute_volume_and_gradients(node_coords)
        x = node_coords + u_nodes
        F = x.T @ H
        R, _ = self.compute_polar_decomposition(F)

        # Local coordinates and displacement
        p = x @ R
        u_l_nodes = p - node_coords

        # Local displacement gradient: grad_u_l = u_l^T @ H
        grad_u_l = u_l_nodes.T @ H

        # Local small strain tensor: eps_l = 0.5 * (grad_u_l + grad_u_l^T)
        eps_l = 0.5 * (grad_u_l + grad_u_l.T)

        # Voigt strain [eps_xx, eps_yy, eps_zz, 2*eps_xy, 2*eps_yz, 2*eps_zx]
        eps_voigt = np.array(
            [
                eps_l[0, 0],
                eps_l[1, 1],
                eps_l[2, 2],
                2.0 * eps_l[0, 1],
                2.0 * eps_l[1, 2],
                2.0 * eps_l[2, 0],
            ],
            dtype=np.float64,
        )

        C_mat = self._material.compute_constitutive_matrix()
        sigma_voigt = C_mat @ eps_voigt

        sigma_l = np.array(
            [
                [sigma_voigt[0], sigma_voigt[3], sigma_voigt[5]],
                [sigma_voigt[3], sigma_voigt[1], sigma_voigt[4]],
                [sigma_voigt[5], sigma_voigt[4], sigma_voigt[2]],
            ],
            dtype=np.float64,
        )

        # Rotate local stress and strain back to global orientation
        sigma = R @ sigma_l @ R.T
        strain = R @ eps_l @ R.T

        return sigma, strain, R

