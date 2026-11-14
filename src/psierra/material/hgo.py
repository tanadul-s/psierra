from __future__ import annotations
from typing import Literal
import numpy as np

from psierra.material.base import Material
from psierra.material.parameters import (
    IsotropicElasticParameters,
    HGOParameters,
)
from psierra.utility.tensor import VOIGT_MAP_3D


class HGO(Material):
    """
    Holzapfel-Gasser-Ogden (HGO) Anisotropic Hyperelastic Material Model.

    Structural anisotropic continuum formulation combining an isotropic ground matrix (elastin)
    with one or more families of dispersed collagen fiber bundles:
        W = W_iso(I1_bar) + sum_{k=1}^M W_fiber,k(E_bar_k) + W_vol(J)

    where:
        W_iso = 0.5 * c * (I1_bar - 3)
        W_fiber,k = (k1 / (2 * k2)) * [exp(k2 * E_bar_k^2) - 1]     (for E_bar_k > 0)
        E_bar_k = kappa * I1_bar + (1 - 3 * kappa) * I4_bar_k - 1

    Key Features:
        1. Invariant formulation (I1 = tr C, I4k = a_0k . C a_0k = lambda_fiber,k^2) eliminating
           costly spectral eigenvalue decompositions and accelerating finite element assembly.
        2. Structural fiber dispersion kappa in [0, 1/3]:
           - kappa = 0: perfectly aligned parallel collagen fibers.
           - kappa = 1/3: 3D random isotropic distribution.
        3. Tension-only activation criterion (E_bar_k > 0): Fibers buckle with zero load-bearing
           under compressive stress and lock up under tensile loading.
        4. Consistent 2nd Piola-Kirchhoff stress S and 4th-order material tangent elasticity tensor C.
        5. Target clinical applications:
           - Aorta & arterial wall (adventitia / media with symmetric helical fibers).
           - Falx cerebri & tentorium cerebelli (internal dural suspension arches).
           - Skin dermis (Langer's tension lines).
    """

    def __init__(
        self,
        parameters: HGOParameters | IsotropicElasticParameters,
        tension_only: bool = True,
        volumetric_model: Literal["log", "quadratic"] = "log",
    ) -> None:
        if isinstance(parameters, IsotropicElasticParameters):
            self._parameters = HGOParameters.from_isotropic(parameters)
        elif isinstance(parameters, HGOParameters):
            self._parameters = parameters
        else:
            raise TypeError(
                f"Expected HGOParameters or IsotropicElasticParameters, got {type(parameters).__name__}"
            )

        if volumetric_model not in ("log", "quadratic"):
            raise ValueError(
                f"volumetric_model must be 'log' or 'quadratic', got '{volumetric_model}'"
            )
        self._volumetric_model: Literal["log", "quadratic"] = volumetric_model
        self._tension_only: bool = bool(tension_only)

        # Pre-compute structural tensors A_k = a_0k (x) a_0k
        self._A_matrices: list[np.ndarray] = [
            np.outer(np.array(a0, dtype=np.float64), np.array(a0, dtype=np.float64))
            for a0 in self._parameters.fiber_directions
        ]

    @property
    def parameters(self) -> HGOParameters:
        """Access underlying HGO material parameters."""
        return self._parameters

    @property
    def tension_only(self) -> bool:
        """Whether tension-only fiber activation is enforced."""
        return self._tension_only

    @property
    def volumetric_model(self) -> str:
        """Active volumetric strain energy model ('log' or 'quadratic')."""
        return self._volumetric_model

    @property
    def mass_density(self) -> float:
        """Mass density (kg/m^3 or tonne/mm^3)."""
        return self._parameters.mass_density

    def compute_constitutive_matrix(self) -> np.ndarray:
        """
        Computes the small-strain 6x6 isotropic elasticity matrix C in Voigt notation
        governed by the ground matrix and bulk modulus at the reference state.
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
        self,
        deformation_gradient: np.ndarray,
        elem_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the 6x6 tangent constitutive matrix C (Voigt notation)
        and 3x3 Second Piola-Kirchhoff stress S for the current deformation gradient.

        Parameters:
        -----------
        deformation_gradient : np.ndarray of shape (3, 3)
            Deformation gradient tensor F.
        elem_index : int | None, optional
            Ignored; provided for interface compatibility.

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
        J_pow23 = J ** (-2.0 / 3.0)
        J_pow43 = J ** (-4.0 / 3.0)
        I1_bar = J_pow23 * I1
        I3 = np.identity(3, dtype=np.float64)

        c = self._parameters.c
        k1 = self._parameters.k1
        k2 = self._parameters.k2
        kappa = self._parameters.kappa
        K_bulk = self._parameters.bulk_modulus

        # 1. Isotropic Elastin Matrix Contribution
        S_iso = c * J_pow23 * (I3 - (1.0 / 3.0) * I1 * C_inv)

        # 4th-order geometric tensors
        C_inv_outer = np.einsum("ij,kl->ijkl", C_inv, C_inv)
        I_Cinv = 0.5 * (
            np.einsum("ik,jl->ijkl", C_inv, C_inv)
            + np.einsum("il,jk->ijkl", C_inv, C_inv)
        )
        C_inv_I = np.einsum("ij,kl->ijkl", C_inv, I3) + np.einsum(
            "ij,kl->ijkl", I3, C_inv
        )

        C4_iso = c * J_pow23 * (
            (2.0 / 3.0) * I1 * I_Cinv
            + (2.0 / 9.0) * I1 * C_inv_outer
            - (2.0 / 3.0) * C_inv_I
        )

        # 2. Volumetric Compressibility Contribution
        if self._volumetric_model == "log":
            S_vol = K_bulk * np.log(J) * C_inv
            C4_vol = K_bulk * C_inv_outer - 2.0 * K_bulk * np.log(J) * I_Cinv
        else:  # quadratic
            S_vol = K_bulk * (J - 1.0) * J * C_inv
            C4_vol = (
                K_bulk * J * (2.0 * J - 1.0) * C_inv_outer
                - 2.0 * K_bulk * (J - 1.0) * J * I_Cinv
            )

        S = S_iso + S_vol
        C4 = C4_iso + C4_vol

        # 3. Anisotropic Collagen Fiber Family Contributions
        for A in self._A_matrices:
            I4 = float(np.sum(C * A))
            I4_bar = J_pow23 * I4

            E_bar = kappa * I1_bar + (1.0 - 3.0 * kappa) * I4_bar - 1.0

            # Tension-only activation check
            if self._tension_only and E_bar <= 0.0:
                continue

            exp_term = np.exp(k2 * (E_bar**2))
            psi_prime = k1 * E_bar * exp_term
            psi_double_prime = k1 * (1.0 + 2.0 * k2 * (E_bar**2)) * exp_term

            H = kappa * I3 + (1.0 - 3.0 * kappa) * A
            Gamma = kappa * I1 + (1.0 - 3.0 * kappa) * I4
            H_bar_star = H - (1.0 / 3.0) * Gamma * C_inv

            # Fiber 2nd Piola-Kirchhoff stress contribution
            S_fiber = 2.0 * psi_prime * J_pow23 * H_bar_star
            S += S_fiber

            # Fiber 4th-order tangent elasticity contribution
            H_star_outer = np.einsum("ij,kl->ijkl", H_bar_star, H_bar_star)
            C_inv_H = np.einsum("ij,kl->ijkl", C_inv, H) + np.einsum(
                "ij,kl->ijkl", H, C_inv
            )

            C4_fiber = (
                4.0 * psi_double_prime * J_pow43 * H_star_outer
                + 4.0 * psi_prime * J_pow23 * (
                    (1.0 / 3.0) * Gamma * I_Cinv
                    + (1.0 / 9.0) * Gamma * C_inv_outer
                    - (1.0 / 3.0) * C_inv_H
                )
            )
            C4 += C4_fiber

        # 4. Map 4th-order elasticity tensor to 6x6 Voigt notation matrix
        C_voigt = np.zeros((6, 6), dtype=np.float64)
        for k_idx, (i, j) in VOIGT_MAP_3D.items():
            for l_idx, (m, n) in VOIGT_MAP_3D.items():
                C_voigt[k_idx, l_idx] = C4[i, j, m, n]

        return C_voigt, S

    def compute_strain_energy(self, deformation_gradient: np.ndarray) -> float:
        """
        Computes the total strain energy density W = W_iso + sum W_fiber,k + W_vol.

        Parameters:
        -----------
        deformation_gradient : np.ndarray of shape (3, 3)
            Deformation gradient tensor F.

        Returns:
        --------
        W : float
            Total strain energy density per unit reference volume.
        """
        F = deformation_gradient
        J = float(np.linalg.det(F))
        if J <= 0.0:
            raise ValueError(f"Element inversion detected: det(F) = {J:.4e} <= 0.")

        C = F.T @ F
        I1 = float(np.trace(C))
        J_pow23 = J ** (-2.0 / 3.0)
        I1_bar = J_pow23 * I1

        # Isotropic ground matrix energy
        W_iso = 0.5 * self._parameters.c * (I1_bar - 3.0)

        # Anisotropic fiber energy
        W_fiber = 0.0
        k1 = self._parameters.k1
        k2 = self._parameters.k2
        kappa = self._parameters.kappa

        for A in self._A_matrices:
            I4 = float(np.sum(C * A))
            I4_bar = J_pow23 * I4
            E_bar = kappa * I1_bar + (1.0 - 3.0 * kappa) * I4_bar - 1.0

            if self._tension_only and E_bar <= 0.0:
                continue

            W_fiber += (k1 / (2.0 * k2)) * float(np.exp(k2 * (E_bar**2)) - 1.0)

        # Volumetric energy
        K_bulk = self._parameters.bulk_modulus
        if self._volumetric_model == "log":
            W_vol = 0.5 * K_bulk * (np.log(J) ** 2)
        else:
            W_vol = 0.5 * K_bulk * ((J - 1.0) ** 2)

        return float(W_iso + W_fiber + W_vol)
