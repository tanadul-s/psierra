from __future__ import annotations
from typing import Literal
import numpy as np

from psierra.material.base import Material
from psierra.material.parameters import (
    IsotropicElasticParameters,
    OgdenParameters,
)
from psierra.utility.tensor import VOIGT_MAP_3D


class Ogden(Material):
    """
    Generalized Ogden Hyperelastic Material Model.

    Formulated on principal stretches lambda_a (a=1,2,3) for arbitrary order N in [1, 3]:
        W = W_iso(lambda_bar_1, lambda_bar_2, lambda_bar_3) + W_vol(J)

    where:
        W_iso = sum_{p=1}^N (mu_p / alpha_p) * (lambda_bar_1^alpha_p + lambda_bar_2^alpha_p + lambda_bar_3^alpha_p - 3)
        lambda_bar_a = J^(-1/3) * lambda_a

    Volumetric models supported:
        - "log":       W_vol = 0.5 * K * (ln J)^2      (default: infinite resistance to volume collapse)
        - "quadratic": W_vol = 0.5 * K * (J - 1)^2     (standard polynomial penalty)

    Key Features:
        1. Exact spectral decomposition of right Cauchy-Green tensor C = sum lambda_a^2 N_a (x) N_a.
        2. Closed-form perturbation handling for coalescent/equal eigenvalues (lambda_a approx lambda_b)
           eliminating numerical singularities at reference, spherical, and equibiaxial states.
        3. Thermodynamically consistent 2nd Piola-Kirchhoff stress S and symmetric 4th-order material
           tangent elasticity tensor C preserving quadratic asymptotic convergence in Newton-Raphson solvers.
        4. Target applications:
           - Brain parenchyma (order N=1, Miller & Chinzei 2002 tension-compression asymmetry).
           - Skin / Dermis (order N=2, 3, toe-region and steep collagen stiffening).
           - Elastomers and phantom polymers (order N=3, Ogden 1972 rubber).
    """

    def __init__(
        self,
        parameters: OgdenParameters | IsotropicElasticParameters,
        order: int = 1,
        volumetric_model: Literal["log", "quadratic"] = "log",
    ) -> None:
        if isinstance(parameters, IsotropicElasticParameters):
            self._parameters = OgdenParameters.from_isotropic(parameters, order=order)
        elif isinstance(parameters, OgdenParameters):
            self._parameters = parameters
        else:
            raise TypeError(
                f"Expected OgdenParameters or IsotropicElasticParameters, got {type(parameters).__name__}"
            )

        if volumetric_model not in ("log", "quadratic"):
            raise ValueError(
                f"volumetric_model must be 'log' or 'quadratic', got '{volumetric_model}'"
            )
        self._volumetric_model: Literal["log", "quadratic"] = volumetric_model

    @property
    def parameters(self) -> OgdenParameters:
        """Access underlying Ogden material parameters."""
        return self._parameters

    @property
    def order(self) -> int:
        """Order N of the Ogden model."""
        return self._parameters.order

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
            Ignored; provided for interface compatibility with element formulations.

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

        # 1. Spectral decomposition of right Cauchy-Green tensor C = F^T F
        C = F.T @ F
        w, N = np.linalg.eigh(C)  # w: eigenvalues Lambda_a, N[:, a]: eigenvectors
        Lambda = np.maximum(w, 1e-16)
        lam = np.sqrt(Lambda)

        # Isochoric principal stretches
        J_pow = J ** (-1.0 / 3.0)
        lam_bar = J_pow * lam

        # 2. Isochoric Kirchhoff stresses and derivatives
        tau_bar = np.zeros(3, dtype=np.float64)
        h = np.zeros(3, dtype=np.float64)
        for mu_p, alpha_p in zip(self._parameters.mu, self._parameters.alpha):
            term = mu_p * (lam_bar**alpha_p)
            tau_bar += term
            h += alpha_p * term

        p_bar_iso = float(np.mean(tau_bar))
        tau_iso = tau_bar - p_bar_iso

        # 3. Volumetric Kirchhoff stress and tangent coefficient
        K_bulk = self._parameters.bulk_modulus
        if self._volumetric_model == "log":
            tau_vol = K_bulk * np.log(J)
            kappa_vol = K_bulk
        else:  # quadratic: 0.5 * K * (J - 1)^2
            tau_vol = K_bulk * (J - 1.0) * J
            kappa_vol = K_bulk * J * (2.0 * J - 1.0)

        tau = tau_iso + tau_vol
        s = tau / Lambda  # Principal 2nd Piola-Kirchhoff stresses

        # 4. 2nd Piola-Kirchhoff Stress Tensor: S = sum_a s_a (N_a (x) N_a)
        S = np.zeros((3, 3), dtype=np.float64)
        for a in range(3):
            S += s[a] * np.outer(N[:, a], N[:, a])

        # 5. Tangent Elasticity Tensor: D_ab = 2 * d(s_a) / d(Lambda_b)
        h_bar = float(np.mean(h))
        T = np.zeros((3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                delta_ab = 1.0 if a == b else 0.0
                T[a, b] = (
                    h[a] * delta_ab
                    - (1.0 / 3.0) * (h[a] + h[b])
                    + (1.0 / 3.0) * h_bar
                    + kappa_vol
                )

        D = np.zeros((3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                delta_ab = 1.0 if a == b else 0.0
                D[a, b] = (
                    T[a, b] / (Lambda[a] * Lambda[b])
                    - (2.0 * s[a] / Lambda[a]) * delta_ab
                )

        # 6. Eigenbasis 4th-Order Tangent C_hat with closed-form coalescent handling
        C_hat = np.zeros((3, 3, 3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                C_hat[a, a, b, b] = D[a, b]

        tol = 1e-6 * max(1.0, float(np.max(Lambda)))
        for a in range(3):
            for b in range(3):
                if a != b:
                    diff = Lambda[a] - Lambda[b]
                    if abs(diff) < tol:
                        # Coalescent eigenvalue limit: lim_{diff -> 0} (s_a - s_b) / diff = 0.5 * (D_aa - D_ab)
                        gamma = 0.5 * (D[a, a] - D[a, b])
                    else:
                        gamma = (s[a] - s[b]) / diff
                    C_hat[a, b, a, b] = gamma
                    C_hat[a, b, b, a] = gamma

        # 7. Push-forward from Eigenbasis to Cartesian Reference Coordinates
        C4 = np.einsum("abcd,ia,jb,kc,ld->ijkl", C_hat, N, N, N, N)

        # 8. Convert 4th-Order Tensor to 6x6 Voigt Notation Matrix
        C_voigt = np.zeros((6, 6), dtype=np.float64)
        for k, (i, j) in VOIGT_MAP_3D.items():
            for l, (m, n) in VOIGT_MAP_3D.items():
                C_voigt[k, l] = C4[i, j, m, n]

        return C_voigt, S

    def compute_strain_energy(self, deformation_gradient: np.ndarray) -> float:
        """
        Computes the total strain energy density W = W_iso + W_vol.

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
        w = np.linalg.eigvalsh(C)
        Lambda = np.maximum(w, 1e-16)
        lam = np.sqrt(Lambda)
        lam_bar = (J ** (-1.0 / 3.0)) * lam

        W_iso = 0.0
        for mu_p, alpha_p in zip(self._parameters.mu, self._parameters.alpha):
            W_iso += (mu_p / alpha_p) * float(np.sum(lam_bar**alpha_p) - 3.0)

        K_bulk = self._parameters.bulk_modulus
        if self._volumetric_model == "log":
            W_vol = 0.5 * K_bulk * (np.log(J) ** 2)
        else:
            W_vol = 0.5 * K_bulk * ((J - 1.0) ** 2)

        return float(W_iso + W_vol)
