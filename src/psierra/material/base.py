from __future__ import annotations
import numpy as np


class Material:
    """
    Base class for all continuum mechanics material models.

    Defines the standard interface for material properties and constitutive laws.
    Subclasses override these methods for specific constitutive models (Hookean, NeoHookean, etc.).
    """

    @property
    def mass_density(self) -> float:
        """Mass density of the material."""
        raise NotImplementedError("Subclasses must implement mass_density.")

    def compute_constitutive_matrix(self) -> np.ndarray:
        """
        Computes the 6x6 elasticity constitutive matrix in Voigt notation.

        Returns:
        --------
        C : np.ndarray of shape (6, 6)
        """
        raise NotImplementedError(
            "Subclasses must implement compute_constitutive_matrix."
        )

    def compute_stress_and_tangent(
        self, deformation_gradient: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the 6x6 tangent constitutive matrix C and the 3x3 Second Piola-Kirchhoff stress S.

        Parameters:
        -----------
        deformation_gradient : np.ndarray of shape (3, 3)
            Deformation gradient tensor F.

        Returns:
        --------
        C : np.ndarray of shape (6, 6)
            Tangent elasticity matrix.
        S : np.ndarray of shape (3, 3)
            2nd Piola-Kirchhoff stress tensor.
        """
        raise NotImplementedError(
            "Subclasses must implement compute_stress_and_tangent."
        )
