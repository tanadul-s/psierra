from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from psierra.analysis.base import Analysis


@dataclass
class AdaptiveTimeStepController:
    """
    Adaptive Time Step Controller for Non-Smooth and Non-Linear Dynamics (Pillar 1).

    Manages automatic time substepping and step cutbacks upon line search stalls,
    Newton-Raphson divergence, contact chattering, or constitutive non-convergence,
    paired with dynamic time-step growth upon consecutive quadratic Newton convergences.

    Formulation & Algorithm (Pillar 1: Temporal Regularization):
    ------------------------------------------------------------
    1. Automatic Step Cutback:
       Upon solver non-convergence or line search stall at step n+1:
           dt_{new} = max(dt_min, gamma_cut * dt)
       The domain state (u_n, v_n, a_n), contact multipliers lambda_n, and
       internal viscoelastic history variables Q_{m, n} are fully rolled back.
       The time increment constants a_0 ... a_5 for Generalized-alpha are recomputed,
       and the step is re-attempted.

    2. Dynamic Step Growth:
       When consecutive steps converge rapidly (Newton iterations <= target_iterations)
       for consecutive_success_for_grow consecutive steps:
           dt_{new} = min(dt_max, gamma_grow * dt)
       restoring nominal time resolution after a non-smooth impact or contact transition.

    Parameters:
    -----------
    dt_min : float, default=1e-6
        Minimum allowable time step size in seconds. If a cutback attempts to reduce
        dt below dt_min, the cutback is rejected and simulation fails gracefully.
    dt_max : float | None, default=None
        Maximum allowable time step size in seconds. If None, initialized to initial_dt.
    gamma_cut : float, default=0.5
        Cutback multiplier applied upon step failure: dt_new = gamma_cut * dt. Must be in (0, 1).
    gamma_grow : float, default=1.25
        Growth factor applied upon consecutive smooth convergences: dt_new = gamma_grow * dt. Must be > 1.0.
    max_cutbacks : int, default=5
        Maximum consecutive cutbacks allowed at the same physical time point before aborting.
    consecutive_success_for_grow : int, default=3
        Number of consecutive successful steps required before increasing dt.
    target_iterations : int, default=4
        Nominal Newton-Raphson iteration target. Steps converging in <= target_iterations
        are counted toward consecutive smooth convergences.
    max_total_steps : int, default=10000
        Safety ceiling for total substeps taken when adaptive substepping is active.
    """

    dt_min: float = 1e-6
    dt_max: float | None = None
    gamma_cut: float = 0.5
    gamma_grow: float = 1.25
    max_cutbacks: int = 5
    consecutive_success_for_grow: int = 3
    target_iterations: int = 4
    max_total_steps: int = 10000

    # Internal runtime state
    current_dt: float = field(init=False, default=0.0)
    initial_dt: float = field(init=False, default=0.0)
    cutback_count: int = field(init=False, default=0)
    total_cutbacks: int = field(init=False, default=0)
    consecutive_successes: int = field(init=False, default=0)
    substeps_taken: int = field(init=False, default=0)
    initialized: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        Analysis._validate_scalar("dt_min", self.dt_min)
        if self.dt_max is not None:
            Analysis._validate_scalar("dt_max", self.dt_max)
            if self.dt_max < self.dt_min:
                raise ValueError(
                    f"dt_max ({self.dt_max}) must be >= dt_min ({self.dt_min})"
                )
        if not (0.0 < self.gamma_cut < 1.0):
            raise ValueError(
                f"gamma_cut must be in (0.0, 1.0), got {self.gamma_cut}"
            )
        if self.gamma_grow <= 1.0:
            raise ValueError(
                f"gamma_grow must be strictly > 1.0, got {self.gamma_grow}"
            )
        Analysis._validate_count("max_cutbacks", self.max_cutbacks, minimum=1)
        Analysis._validate_count(
            "consecutive_success_for_grow",
            self.consecutive_success_for_grow,
            minimum=1,
        )
        Analysis._validate_count("target_iterations", self.target_iterations, minimum=1)
        Analysis._validate_count("max_total_steps", self.max_total_steps, minimum=1)

    def initialize(self, initial_dt: float) -> None:
        """
        Initializes the controller for a new simulation run with baseline time step initial_dt.
        """
        Analysis._validate_scalar("initial_dt", initial_dt)
        initial_dt_val = float(initial_dt)
        if self.dt_max is None:
            self.dt_max = initial_dt_val
        elif self.dt_max < initial_dt_val:
            # Allow explicit user dt_max even if initial_dt is different, as long as valid
            pass

        if self.dt_min > initial_dt_val:
            self.dt_min = initial_dt_val * 1e-4

        self.initial_dt = initial_dt_val
        self.current_dt = initial_dt_val
        self.cutback_count = 0
        self.total_cutbacks = 0
        self.consecutive_successes = 0
        self.substeps_taken = 0
        self.initialized = True

    def can_cutback(self) -> bool:
        """
        Checks whether another time step cutback is permissible.
        """
        if not self.initialized:
            return False
        has_cutbacks_left = self.cutback_count < self.max_cutbacks
        above_min_dt = (self.current_dt * self.gamma_cut) >= (self.dt_min - 1e-15)
        return has_cutbacks_left and above_min_dt

    def cutback(self) -> float:
        """
        Executes a time step cutback: dt <- max(dt_min, gamma_cut * dt).

        Returns:
        --------
        new_dt : float
            The reduced time step size.
        """
        if not self.can_cutback():
            raise RuntimeError(
                f"Cannot execute cutback: cutbacks={self.cutback_count}/{self.max_cutbacks}, "
                f"current_dt={self.current_dt:.4e}, dt_min={self.dt_min:.4e}"
            )
        new_dt = max(self.dt_min, self.current_dt * self.gamma_cut)
        self.cutback_count += 1
        self.total_cutbacks += 1
        self.consecutive_successes = 0
        self.current_dt = new_dt
        return self.current_dt

    def record_success(self, iterations: int) -> float:
        """
        Notifies the controller that a time step successfully converged in the given iterations.
        Manages consecutive quadratic convergence counts and dynamic time step growth.

        Returns:
        --------
        next_dt : float
            The suggested time step size for the subsequent step.
        """
        self.substeps_taken += 1
        self.cutback_count = 0

        if iterations <= self.target_iterations:
            self.consecutive_successes += 1
            if self.consecutive_successes >= self.consecutive_success_for_grow:
                growth_dt = self.current_dt * self.gamma_grow
                max_allowed = self.dt_max if self.dt_max is not None else growth_dt
                self.current_dt = min(max_allowed, growth_dt)
                self.consecutive_successes = 0
        else:
            self.consecutive_successes = 0

        return self.current_dt

    def reset_step(self) -> None:
        """Resets the per-step cutback counter when transitioning to a new time point."""
        self.cutback_count = 0

    def reset(self) -> None:
        """Resets the controller back to an uninitialized state."""
        self.current_dt = 0.0
        self.initial_dt = 0.0
        self.cutback_count = 0
        self.total_cutbacks = 0
        self.consecutive_successes = 0
        self.substeps_taken = 0
        self.initialized = False
