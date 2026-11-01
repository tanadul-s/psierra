from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import scipy.sparse as spsp


class Domain(ABC):
    """
    Abstract Base Class for Finite Element Domains in psierra.

    Binds geometric mesh, constitutive material, and element kinematics,
    and manages assembly of global sparse system matrices (K, M) and state vectors.
    """

    @property
    @abstractmethod
    def n_dofs(self) -> int:
        """Total degrees of freedom in the domain."""
        ...

    @abstractmethod
    def assemble_stiffness_matrix(self) -> spsp.csr_matrix:
        """Assembles and returns the global sparse stiffness matrix K."""
        ...

    @abstractmethod
    def assemble_mass_matrix(self) -> spsp.csr_matrix:
        """Assembles and returns the global sparse consistent mass matrix M."""
        ...

    @abstractmethod
    def update_state(
        self,
        u: np.ndarray | None = None,
        v: np.ndarray | None = None,
        a: np.ndarray | None = None,
    ) -> None:
        """Updates kinematic state vectors (displacement, velocity, acceleration)."""
        ...

    @abstractmethod
    def save_state(self, filepath: Path, **kwargs) -> None:
        """Exports the current deformed state to disk as a VTK/VTU unstructured mesh."""
        ...

