from __future__ import annotations
import numpy as np

from psierra.material.base import Material
from psierra.material.parameters import (
    IsotropicElasticParameters,
    MooneyRivlinParameters,
)
from psierra.utility.tensor import VOIGT_MAP_3D


class MooneyRivlin(Material):
    """
    Compressible Mooney-Rivlin Hyperelastic Material.

    Strain energy function:
        Psi = c10 * (I1_bar - 3) + c01 * (I2_bar - 3) + 0.5 * K * (ln J)^2

    Provides classical two-parameter hyperelastic behavior for rubber-like materials,
    elastomers, and medical phantom polymers. In the infinitesimal limit, the initial
    shear modulus is mu = 2 * (c10 + c01).
    """

    def __init__(
        self,
        parameters: MooneyRivlinParameters | IsotropicElasticParameters,
    ) -> None:
        if isinstance(parameters, IsotropicElasticParameters):
            self._parameters = MooneyRivlinParameters.from_isotropic(parameters)
        elif isinstance(parameters, MooneyRivlinParameters):
            self._parameters = parameters
        else:
            raise TypeError(
                f"Expected MooneyRivlinParameters or IsotropicElasticParameters, got {type(parameters).__name__}"
            )

    @property
    def parameters(self) -> MooneyRivlinParameters:
        """Access underlying Mooney-Rivlin material parameters."""
        return self._parameters

    @property
    def mass_density(self) -> float:
        """Mass density (kg/m^3)."""
        return self._parameters.mass_density

    def compute_constitutive_matrix(self) -> np.ndarray:
        """
        Computes the small-strain 6x6 isotropic elasticity matrix C in Voigt notation.
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
            Tangent elasticity matrix in Voigt notation.
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
        C2 = C @ C
        I2 = 0.5 * (I1**2 - float(np.trace(C2)))

        c10 = self._parameters.c10
        c01 = self._parameters.c01
        K_bulk = self._parameters.bulk_modulus
        I3 = np.identity(3, dtype=np.float64)

        # 1. Second Piola-Kirchhoff Stress S = S_iso + S_vol
        factor1 = 2.0 * c10 * (J ** (-2.0 / 3.0))
        factor2 = 2.0 * c01 * (J ** (-4.0 / 3.0))

        S1 = factor1 * (I3 - (1.0 / 3.0) * I1 * C_inv)
        S2 = factor2 * (I1 * I3 - C - (2.0 / 3.0) * I2 * C_inv)
        S_iso = S1 + S2
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

        # Isochoric contribution from I1_bar
        C_inv_I = np.einsum("ij,kl->ijkl", C_inv, I3) + np.einsum(
            "ij,kl->ijkl", I3, C_inv
        )
        C4_iso1 = (
            -(2.0 / 3.0) * factor1 * C_inv_I
            + (2.0 / 9.0) * factor1 * I1 * C_inv_outer
            + (2.0 / 3.0) * factor1 * I1 * I_Cinv
        )

        # Isochoric contribution from I2_bar
        I_outer = np.einsum("ij,kl->ijkl", I3, I3)
        I_sym = 0.5 * (
            np.einsum("ik,jl->ijkl", I3, I3)
            + np.einsum("il,jk->ijkl", I3, I3)
        )
        B2 = I1 * I3 - C
        C_inv_B2 = np.einsum("ij,kl->ijkl", C_inv, B2) + np.einsum(
            "ij,kl->ijkl", B2, C_inv
        )
        C4_iso2 = (c01 * (J ** (-4.0 / 3.0))) * (
            4.0 * (I_outer - I_sym)
            - (8.0 / 3.0) * C_inv_B2
            + (16.0 / 9.0) * I2 * C_inv_outer
            + (8.0 / 3.0) * I2 * I_Cinv
        )

        C4 = C4_vol + C4_iso1 + C4_iso2

        # 3. Convert 4th-Order Tensor to 6x6 Voigt Notation Matrix using VOIGT_MAP_3D
        C_voigt = np.zeros((6, 6), dtype=np.float64)
        for a, (i, j) in VOIGT_MAP_3D.items():
            for b, (k, l) in VOIGT_MAP_3D.items():
                C_voigt[a, b] = C4[i, j, k, l]

        return C_voigt, S
