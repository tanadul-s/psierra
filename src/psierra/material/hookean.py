from __future__ import annotations
import numpy as np

from psierra.material.base import Material
from psierra.material.parameters import IsotropicElasticParameters


class Hookean(Material):
    """
    Linear Isotropic Elastic Material (Hooke's Law).

    Constructed with an IsotropicElasticParameters instance.
    Provides the 6x6 constitutive matrix C in Voigt notation:
    [sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx]^T = C * [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx]^T
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
        Computes the 6x6 linear isotropic elasticity matrix C.

        Returns:
        --------
        C : np.ndarray of shape (6, 6)
            Symmetric constitutive matrix in Voigt notation:
            Order: (xx, yy, zz, xy, yz, zx)
        """
        lam = self._parameters.lame_parameter_1
        mu = self._parameters.lame_parameter_2  # or shear_modulus
        c11 = lam + 2.0 * mu  # Also p_wave_modulus

        C = np.zeros((6, 6), dtype=np.float64)

        # Normal stress-strain block (3x3)
        C[0, 0] = c11
        C[1, 1] = c11
        C[2, 2] = c11

        C[0, 1] = C[1, 0] = lam
        C[0, 2] = C[2, 0] = lam
        C[1, 2] = C[2, 1] = lam

        # Shear stress-strain block (3x3 diagonal)
        C[3, 3] = mu
        C[4, 4] = mu
        C[5, 5] = mu

        return C

    def compute_stress_and_tangent(
        self, deformation_gradient: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Saint Venant-Kirchhoff linear response for Hookean material: S = C : E."""
        F = deformation_gradient
        # Green-Lagrange strain tensor E = 0.5 * (F^T F - I)
        E_tensor = 0.5 * (F.T @ F - np.identity(3, dtype=np.float64))

        lam = self._parameters.lame_parameter_1
        mu = self._parameters.shear_modulus

        # S = lambda * tr(E) * I + 2 * mu * E
        trE = np.trace(E_tensor)
        S = lam * trE * np.identity(3, dtype=np.float64) + 2.0 * mu * E_tensor

        C = self.compute_constitutive_matrix()
        return C, S
