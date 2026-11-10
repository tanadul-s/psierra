"""Contact-aware line search and algorithmic regularizations for non-smooth mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


def is_element_inversion_error(exc: BaseException) -> bool:
    """Return True if exception indicates physical continuum element inversion or constitutive violation.

    Detects negative Jacobians, inverted elements, admissible range violations,
    and floating point overflows arising from unphysical element collapse.
    """
    if isinstance(exc, FloatingPointError):
        return True
    if isinstance(exc, ValueError):
        msg = str(exc).lower()
        if any(
            keyword in msg
            for keyword in (
                "element inversion",
                "negative jacobian",
                "det(f)",
                "outside admissible",
                "admissible range",
                "zero surface area",
                "degenerate",
            )
        ):
            return True
    return False


@dataclass(frozen=True)
class LineSearchResult:
    """Result of a contact-aware backtracking line search.

    Attributes
    ----------
    success : bool
        Whether an admissible step size alpha was found.
    alpha : float
        Selected step length scale in (0, 1].
    candidate_norm : float
        Residual norm ||R(u + alpha * du)|| at selected step.
    iterations : int
        Number of step contractions evaluated.
    element_inversion_encountered : bool
        Whether any trial step triggered continuum element inversion (det(F) <= 0).
    accepted_as_contact_fallback : bool
        Whether the step was accepted through contact-aware fallback when Armijo
        condition was prevented by contact active set non-smoothness.
    rejection_reason : str | None
        Diagnostic explanation if line search failed.
    """

    success: bool
    alpha: float
    candidate_norm: float
    iterations: int
    element_inversion_encountered: bool
    accepted_as_contact_fallback: bool
    rejection_reason: str | None = None


class ContactAwareLineSearch:
    """Contact-aware backtracking line search for non-linear finite element systems.

    Differentiates between:
    1. Physical continuum breakdown (constitutive element inversion det(F) <= 0),
       which strictly invalidates the trial step and mandates backtracking.
    2. Non-smooth contact mechanics (active-set switching, facet edge crossings, and
       normal penalty stiffness activation), which produce non-monotonic residual
       jumps without volume inversion.

    Parameters
    ----------
    max_iterations : int, default=16
        Maximum number of backtracking bisections.
    contraction_factor : float, default=0.5
        Step contraction ratio alpha <- alpha * contraction_factor.
    armijo_c1 : float, default=2.0e-4
        Sufficient decrease curvature parameter for Armijo condition:
        ||R(alpha)||^2 <= (1 - 2*c1*alpha) * ||R_0||^2.
    full_step_tolerance : float, default=1.5
        Allowed residual growth factor for full Newton step (alpha=1.0) when no
        element inversion occurs, facilitating contact active-set discovery.
    contact_fallback_tolerance : float, default=5.0
        Maximum allowable residual growth for accepting the best non-inverted
        candidate when Armijo descent is prevented by non-smooth contact bumps.
    """

    def __init__(
        self,
        max_iterations: int = 16,
        contraction_factor: float = 0.5,
        armijo_c1: float = 2.0e-4,
        full_step_tolerance: float = 1.5,
        contact_fallback_tolerance: float = 5.0,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if not 0.0 < contraction_factor < 1.0:
            raise ValueError("contraction_factor must be in (0, 1)")
        if not 0.0 < armijo_c1 < 0.5:
            raise ValueError("armijo_c1 must be in (0, 0.5)")
        if full_step_tolerance < 1.0:
            raise ValueError("full_step_tolerance must be at least 1.0")
        if contact_fallback_tolerance < 1.0:
            raise ValueError("contact_fallback_tolerance must be at least 1.0")

        self.max_iterations = int(max_iterations)
        self.contraction_factor = float(contraction_factor)
        self.armijo_c1 = float(armijo_c1)
        self.full_step_tolerance = float(full_step_tolerance)
        self.contact_fallback_tolerance = float(contact_fallback_tolerance)

    def search(
        self,
        eval_fn: Callable[[float], float],
        current_norm: float,
        inner_tolerance: float,
        *,
        contact_active: bool = False,
    ) -> LineSearchResult:
        """Execute contact-aware backtracking line search.

        Parameters
        ----------
        eval_fn : Callable[[float], float]
            Function taking step scale alpha in (0, 1] and returning candidate residual
            norm ||R(alpha)||. Must raise ValueError / FloatingPointError on element inversion.
        current_norm : float
            Residual norm ||R_0|| at beginning of Newton iteration (alpha=0).
        inner_tolerance : float
            Absolute residual tolerance for direct convergence acceptance.
        contact_active : bool, default=False
            Whether contact constraints are currently present or active in the model.
        """
        alpha = 1.0
        best_norm = np.inf
        best_alpha = 1.0
        inversion_encountered = False

        for iter_idx in range(1, self.max_iterations + 1):
            is_inverted = False
            try:
                cand_norm = float(eval_fn(alpha))
            except Exception as exc:
                if is_element_inversion_error(exc):
                    inversion_encountered = True
                    is_inverted = True
                    cand_norm = np.inf
                else:
                    raise

            if is_inverted or not np.isfinite(cand_norm):
                inversion_encountered = True
                alpha *= self.contraction_factor
                continue

            # Candidate is constitutively valid (no element inversion!)
            if cand_norm < best_norm:
                best_norm = cand_norm
                best_alpha = alpha

            # 1. Direct convergence
            if cand_norm <= inner_tolerance:
                return LineSearchResult(
                    success=True,
                    alpha=alpha,
                    candidate_norm=cand_norm,
                    iterations=iter_idx,
                    element_inversion_encountered=inversion_encountered,
                    accepted_as_contact_fallback=False,
                )

            # 2. Strict residual reduction
            if cand_norm <= current_norm:
                return LineSearchResult(
                    success=True,
                    alpha=alpha,
                    candidate_norm=cand_norm,
                    iterations=iter_idx,
                    element_inversion_encountered=inversion_encountered,
                    accepted_as_contact_fallback=False,
                )

            # 3. Standard Armijo condition
            if cand_norm**2 <= (1.0 - 2.0 * self.armijo_c1 * alpha) * current_norm**2:
                return LineSearchResult(
                    success=True,
                    alpha=alpha,
                    candidate_norm=cand_norm,
                    iterations=iter_idx,
                    element_inversion_encountered=inversion_encountered,
                    accepted_as_contact_fallback=False,
                )

            # 4. Full Newton step with bounded residual (contact-aware)
            if (
                contact_active
                and iter_idx == 1
                and alpha == 1.0
                and cand_norm < self.full_step_tolerance * current_norm
                and not is_inverted
            ):
                return LineSearchResult(
                    success=True,
                    alpha=alpha,
                    candidate_norm=cand_norm,
                    iterations=iter_idx,
                    element_inversion_encountered=inversion_encountered,
                    accepted_as_contact_fallback=False,
                )

            alpha *= self.contraction_factor

        # Post-loop: Check Contact-Aware Fallback
        if (
            np.isfinite(best_norm)
            and best_norm < self.contact_fallback_tolerance * current_norm
            and (contact_active or not inversion_encountered)
        ):
            return LineSearchResult(
                success=True,
                alpha=best_alpha,
                candidate_norm=best_norm,
                iterations=self.max_iterations,
                element_inversion_encountered=inversion_encountered,
                accepted_as_contact_fallback=True,
            )

        # Failure: No admissible trial found
        reason = (
            "Element inversion detected across all evaluated step lengths"
            if (not np.isfinite(best_norm) and inversion_encountered)
            else f"Residual {best_norm:.4e} exceeded fallback tolerance {self.contact_fallback_tolerance * current_norm:.4e}"
        )
        return LineSearchResult(
            success=False,
            alpha=best_alpha,
            candidate_norm=best_norm,
            iterations=self.max_iterations,
            element_inversion_encountered=inversion_encountered,
            accepted_as_contact_fallback=False,
            rejection_reason=reason,
        )
