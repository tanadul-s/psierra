from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class Analysis(ABC):
    """
    Abstract base class for all physical simulation analyses in psierra.

    Subclasses orchestrate equilibrium equations, load incrementation,
    time integration schemes, and boundary condition enforcement.
    """

    @abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the simulation analysis procedure."""
        raise NotImplementedError("Subclasses must implement solve.")
