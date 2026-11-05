from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Analysis(ABC):
    """
    Abstract base class for all physical simulation analyses in psierra.

    Subclasses orchestrate equilibrium equations, load incrementation,
    time integration schemes, and boundary condition enforcement.
    """

    @staticmethod
    def _validate_count(name: str, value: int, minimum: int = 1) -> None:
        """Reject noninteger counts before altering the physical state."""
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value}")

    @staticmethod
    def _validate_scalar(name: str, value: float, allow_zero: bool = False) -> None:
        """Require finite positive numerical controls (optionally including zero)."""
        if not np.isfinite(value) or (
            value < 0.0 if allow_zero else value <= 0.0
        ):
            relation = ">= 0" if allow_zero else "> 0"
            raise ValueError(f"{name} must be finite and {relation}, got {value}")

    @classmethod
    def _validate_dynamic_controls(
        cls,
        dt: float,
        n_steps: int,
        save_interval: int,
        steady_state_tol: float | None,
        kinetic_energy_tol: float | None,
        consecutive_hold_steps: int,
    ) -> None:
        """Validate time controls; zero steps retain the consistent initial state."""
        cls._validate_scalar("dt", dt)
        cls._validate_count("n_steps", n_steps, minimum=0)
        cls._validate_count("save_interval", save_interval)
        cls._validate_count("consecutive_hold_steps", consecutive_hold_steps)
        for name, value in (
            ("steady_state_tol", steady_state_tol),
            ("kinetic_energy_tol", kinetic_energy_tol),
        ):
            if value is not None:
                cls._validate_scalar(name, value, allow_zero=True)

    @abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the simulation analysis procedure."""
        raise NotImplementedError("Subclasses must implement solve.")
