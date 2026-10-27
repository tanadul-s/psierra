from __future__ import annotations
import numpy as np

from psierra.material.base import Material
from psierra.material.parameters import IsotropicElasticParameters
from psierra.utility.tensor import VOIGT_MAP_3D


class NeoHookean(Material):
    """
    Compressible Neo-Hookean Hyperelastic Material.

    Strain energy function:
        Psi = 0.5 * mu * (J^(-2/3) * I1 - 3) + 0.5 * K * (ln J)^2

    Provides infinite resistance against volume collapse (J -> 0),
    preventing element inversion in large deformations and soft tissues.
    """

    def __init__(self, parameters: IsotropicElasticParameters) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> IsotropicElasticParameters:
        """Access underlying elastic parameters."""
        return self._parameters

    @property
    def mass_density(self) -> float:
        """Mass density."""
        return self._parameters.mass_density

    def compute_constitutive_matrix(self) -> np.ndarray:
        """
        Computes the small-strain 6x6 isotropic elasticity matrix C.
        Identical to Hooke's law at the undeformed reference state.
        """
        lam = self._parameters.lame_parameter_1
        mu = self._parameters.shear_modulus
        c11 = lam + 2.0 * mu

        C = np.zeros((6, 6), dtype=np.float64)

        # Normal block
        C[0, 0] = C[1, 1] = C[2, 2] = c11
        C[0, 1] = C[1, 0] = lam
        C[0, 2] = C[2, 0] = lam
        C[1, 2] = C[2, 1] = lam

        # Shear block
        C[3, 3] = C[4, 4] = C[5, 5] = mu

        return C

    def compute_stress_and_tangent(
        self, deformation_gradient: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the 6x6 tangent constitutive matrix C (Voigt notation)
        and 3x3 Second Piola-Kirchhoff stress S for the current deformation gradient.

        Parameters:
        -----------
        deformation_gradient : np.ndarray of shape (3, 3)
            Deformation gradient tensor F.

        Returns:
        --------
        C_voigt : np.ndarray of shape (6, 6)
            Tangent elasticity matrix.
        S : np.ndarray of shape (3, 3)
            2nd Piola-Kirchhoff stress tensor.
        """
        F = deformation_gradient
        J = float(np.linalg.det(F))
        if J <= 0.0:
            raise ValueError(f"Element inversion detected: det(F) = {J:.4e} <= 0.")

        C = F.T @ F
        C_inv = np.linalg.inv(C)
        I1 = float(np.trace(C))

        mu = self._parameters.shear_modulus
        K_bulk = self._parameters.bulk_modulus
        I3 = np.identity(3, dtype=np.float64)

        factor = mu * (J ** (-2.0 / 3.0))

        # 1. Second Piola-Kirchhoff Stress S = S_iso + S_vol
        S_iso = factor * (I3 - (1.0 / 3.0) * I1 * C_inv)
        S_vol = K_bulk * np.log(J) * C_inv
        S = S_iso + S_vol

        # 2. 4th-Order Material Tangent Elasticity Tensor C4
        C_inv_outer = np.einsum("ij,kl->ijkl", C_inv, C_inv)
        I_Cinv = 0.5 * (
            np.einsum("ik,jl->ijkl", C_inv, C_inv)
            + np.einsum("il,jk->ijkl", C_inv, C_inv)
        )

        # Volumetric contribution
        C4_vol = K_bulk * C_inv_outer - 2.0 * K_bulk * np.log(J) * I_Cinv

        # Isochoric contribution
        C_inv_I = np.einsum("ij,kl->ijkl", C_inv, I3) + np.einsum(
            "ij,kl->ijkl", I3, C_inv
        )
        C4_iso = (
            -(2.0 / 3.0) * factor * C_inv_I
            + (2.0 / 9.0) * factor * I1 * C_inv_outer
            + (2.0 / 3.0) * factor * I1 * I_Cinv
        )

        C4 = C4_vol + C4_iso

        # 3. Convert 4th-Order Tensor to 6x6 Voigt Notation Matrix using VOIGT_MAP_3D
        C_voigt = np.zeros((6, 6), dtype=np.float64)
        for a, (i, j) in VOIGT_MAP_3D.items():
            for b, (k, l) in VOIGT_MAP_3D.items():
                C_voigt[a, b] = C4[i, j, k, l]

        return C_voigt, S
