"""Centralized Normal Constraint Law for Contact Mechanics in Project Sierra.

Milestone 1.0 Stage C of Loop 3 Contact Mechanics Overhaul.
-----------------------------------------------------------
This module implements the unified normal penalty / augmented-Lagrangian constraint
law that serves as the single mathematical kernel for all contact formulations:

    p_N = max(0, lambda_N - k_p * g_N)   with lambda_N = 0 for pure penalty.

Key Principles:
1. Non-Adhesive Physical Activation: Search proximity tolerances (delta_search)
   are strictly decoupled from force activation. For penalty contact, a separated
   point with 0 < g_N <= delta_search receives zero normal force (p_N = 0).
2. Conservative Triad Consistency:
   - Potential energy: Pi_N = (p_N^2 - lambda_N^2) / (2 * k_p)
   - Interface force: f_local = -dPi_N/dq = p_N * grad_q(g_N)
   - Tangent stiffness: K_local = -df_local/dq = k_p * (grad_q(g_N) (x) grad_q(g_N)) - p_N * Hess_q(g_N)
3. Dynamic Velocity Damping:
   Optional regularized normal damping active when penetrating velocity v_N = (v . n) < 0.
"""

from __future__ import annotations

from typing import Literal
import numpy as np

from psierra.contact.data import DifferentiableContactGeometry, NormalLawState


class NormalConstraintLaw:
    """Unified normal penalty and augmented-Lagrangian constraint law.

    Evaluates contact pressure, potential energy, force vector, and consistent
    tangent stiffness matrix from canonical DifferentiableContactGeometry.
    """

    def __init__(
        self,
        k_penalty: float = 1e4,
        c_damping: float = 0.0,
        formulation: Literal["penalty", "augmented_lagrangian"] = "penalty",
    ) -> None:
        if not np.isfinite(k_penalty) or k_penalty <= 0.0:
            raise ValueError(f"k_penalty must be finite and positive, got {k_penalty}")
        if not np.isfinite(c_damping) or c_damping < 0.0:
            raise ValueError(f"c_damping must be finite and nonnegative, got {c_damping}")
        if formulation not in ("penalty", "augmented_lagrangian"):
            raise ValueError(f"formulation must be 'penalty' or 'augmented_lagrangian', got '{formulation}'")

        self.k_penalty = float(k_penalty)
        self.c_damping = float(c_damping)
        self.formulation = formulation

    def evaluate_state(
        self,
        geometry: DifferentiableContactGeometry,
        multiplier: float = 0.0,
    ) -> NormalLawState:
        """Evaluate scalar normal law state (p_N, Pi_N, activity, derivatives)."""
        return NormalLawState.evaluate(
            geometry=geometry,
            k_penalty=self.k_penalty,
            multiplier=multiplier,
            formulation=self.formulation,
        )

    def evaluate_force(
        self,
        geometry: DifferentiableContactGeometry,
        multiplier: float = 0.0,
        velocity_stencil: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate local contact force vector f_local = p_N * grad_q(g_N) + f_damping.

        Parameters:
        -----------
        geometry : DifferentiableContactGeometry
            Canonical contact geometry at the candidate interaction.
        multiplier : float, default=0.0
            Retained dual multiplier lambda_N >= 0 [force].
        velocity_stencil : np.ndarray, optional
            Local velocity vector aligned with the stencil DOFs.

        Returns:
        --------
        f_local : np.ndarray of shape (M,)
            Local contact repulsive force vector.
        """
        state = self.evaluate_state(geometry, multiplier=multiplier)
        if not state.is_active or state.pressure <= 0.0:
            return np.zeros_like(geometry.gradient)

        # Conservative normal force: f = p_N * grad_q(g_N)
        f_local = state.pressure * geometry.gradient

        # Optional normal velocity damping
        if velocity_stencil is not None and self.c_damping > 0.0:
            vel = np.asarray(velocity_stencil, dtype=np.float64).ravel()
            if len(vel) == len(geometry.gradient):
                # Normal relative velocity: v_N = vel . grad_q(g_N)
                v_N = float(np.dot(vel, geometry.gradient))
                if v_N < 0.0:  # Moving into deeper penetration
                    f_local += -self.c_damping * v_N * geometry.gradient

        return f_local

    def evaluate_tangent(
        self,
        geometry: DifferentiableContactGeometry,
        multiplier: float = 0.0,
        a1: float = 0.0,
    ) -> np.ndarray:
        """Evaluate local tangent stiffness matrix K_local = -df_local/dq.

        Exact analytical derivative on a smooth fixed branch:
            K_local = (k_p + a1 * c_p) * (grad_q(g_N) (x) grad_q(g_N)) - p_N * Hess_q(g_N)

        Parameters:
        -----------
        geometry : DifferentiableContactGeometry
            Canonical contact geometry.
        multiplier : float, default=0.0
            Retained dual multiplier lambda_N >= 0.
        a1 : float, default=0.0
            Algorithmic dynamic scale factor gamma / (beta * dt) for damping.

        Returns:
        --------
        K_local : np.ndarray of shape (M, M)
            Local tangent stiffness matrix. Symmetric on smooth branch.
        """
        state = self.evaluate_state(geometry, multiplier=multiplier)
        M = len(geometry.gradient)
        if not state.is_active or state.pressure <= 0.0:
            return np.zeros((M, M), dtype=np.float64)

        eff_kp = self.k_penalty + a1 * self.c_damping
        grad = geometry.gradient
        hess = geometry.hessian

        # K_local = eff_kp * (grad outer grad) - p_N * Hess
        K_local = eff_kp * np.outer(grad, grad) - state.pressure * hess

        # Ensure exact analytical symmetry
        return 0.5 * (K_local + K_local.T)

    def evaluate_potential_energy(
        self,
        geometry: DifferentiableContactGeometry,
        multiplier: float = 0.0,
    ) -> float:
        """Evaluate scalar normal contact potential energy Pi_N."""
        return self.evaluate_state(geometry, multiplier=multiplier).potential_energy
