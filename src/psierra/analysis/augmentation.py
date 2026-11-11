"""Bounded augmented-Lagrangian state and convergence for batch analyses."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from psierra.analysis.base import Analysis
from psierra.boundary.contact import ContactMultiplierState


@dataclass(frozen=True)
class AugmentationStatus:
    """Projected dual state and constraint defects at one equilibrated state."""

    proposed: list[np.ndarray]
    max_penetration: float
    relative_multiplier_residual: float
    converged: bool


class ContactAugmentation:
    """Manage fixed-multiplier equilibrium solves and transactional dual state.

    For each AL contact, p=max(0,lambda-kp*g). Acceptance requires maximum
    penetration <= gap_tol [length] and ||p-lambda||inf/max(||p||inf,||lambda||inf)
    <= tolerance [dimensionless], with zero defect when both norms vanish.
    Each contact passes independently; penalty contacts are never augmented.
    A passing state retains its equilibrated lambda, with no unchecked update.

    Inner equilibrium is tightened by dual-force and gap-force targets. This
    Uzawa iteration is bounded; convergence can be slow when bulk/effective
    dynamic stiffness is large relative to kp. See FEBio Theory, sec. 7.1;
    docs/06_solvers_and_analysis_pipelines.md.
    """

    def __init__(
        self,
        contacts: list[ContactMultiplierState],
        max_iterations: int,
        gap_tol: float,
        tolerance: float,
        rebound_relaxation: bool = False,
    ) -> None:
        Analysis._validate_count("al_max_iterations", max_iterations)
        Analysis._validate_scalar("al_gap_tol", gap_tol)
        Analysis._validate_scalar("al_tol", tolerance)
        self.contacts = [
            c for c in contacts
            if getattr(c, "formulation", None) == "augmented_lagrangian"
        ]
        self.max_iterations = max_iterations if self.contacts else 1
        self.gap_tol = gap_tol
        self.tolerance = tolerance
        self.rebound_relaxation = bool(rebound_relaxation)
        self._friction_snapshots: list[Any] = []

    def snapshot(self, current_nodes: np.ndarray | None = None) -> list[np.ndarray]:
        """Copy accepted dual state; resolve any lazy candidate dimensions first."""
        for contact in self.contacts:
            contact.evaluate_gaps(current_nodes)
        snapshots = [contact.snapshot_multipliers() for contact in self.contacts]
        for values in snapshots:
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError("Initial AL multipliers must be finite and nonnegative")
        self._friction_snapshots = [
            c.snapshot_friction_state() if hasattr(c, "snapshot_friction_state") else None
            for c in self.contacts
        ]
        return snapshots

    def restore(self, snapshot: list[np.ndarray]) -> None:
        """Restore accepted multipliers and invalidate derived query state."""
        for contact, multipliers in zip(self.contacts, snapshot, strict=True):
            contact.set_multipliers(multipliers)
        if self._friction_snapshots and len(self._friction_snapshots) == len(self.contacts):
            for contact, f_snap in zip(self.contacts, self._friction_snapshots, strict=True):
                if hasattr(contact, "restore_friction_state") and f_snap is not None:
                    contact.restore_friction_state(f_snap)

    def assess(self, current_nodes: np.ndarray | None = None) -> AugmentationStatus:
        """Evaluate projected multipliers and defects without changing state."""
        proposed: list[np.ndarray] = []
        max_penetration = 0.0
        max_relative = 0.0
        for contact in self.contacts:
            gaps = contact.evaluate_gaps(current_nodes)
            multipliers = contact.snapshot_multipliers()
            if (
                gaps.shape != multipliers.shape
                or np.any(np.isnan(gaps))
                or np.any(np.isneginf(gaps))
            ):
                raise ValueError(
                    "Contact gaps must align with multipliers and be finite or +infinity"
                )
            pressure = np.maximum(0.0, multipliers - contact.k_penalty * gaps)
            if self.rebound_relaxation:
                pressure = np.where(gaps > self.gap_tol, 0.0, pressure)
            if not np.all(np.isfinite(pressure)):
                raise ValueError("Projected AL multipliers must be finite")
            scale = max(
                float(np.max(pressure, initial=0.0)),
                float(np.max(multipliers, initial=0.0)),
            )
            difference = float(np.max(np.abs(pressure - multipliers), initial=0.0))
            relative = difference / scale if scale > 0.0 else 0.0
            max_relative = max(max_relative, relative)
            max_penetration = max(
                max_penetration,
                float(np.max(np.maximum(-gaps, 0.0), initial=0.0)),
            )
            proposed.append(pressure)
        return AugmentationStatus(
            proposed, max_penetration, max_relative,
            max_penetration <= self.gap_tol and max_relative <= self.tolerance,
        )

    def commit(self, status: AugmentationStatus) -> None:
        """Commit projected nonnegative dual values before another equilibrium solve."""
        for contact, pressure in zip(self.contacts, status.proposed, strict=True):
            contact.set_multipliers(pressure)
            if (
                hasattr(contact, "friction")
                and contact.friction > 0.0
                and hasattr(contact, "friction_state")
                and contact.friction_state is not None
            ):
                contact.friction_state.multipliers[:] = contact.friction_state.forces[:]

    def relax_rebound(self) -> int:
        """Zero retained multipliers on non-penetrating candidates across all AL contacts."""
        if not self.rebound_relaxation:
            return 0
        return sum(c.relax_rebound(self.gap_tol) for c in self.contacts)

    def inner_force_tolerance(self, requested: float, force_scale: float) -> float:
        """Bound force error by requested, dual-force, and gap-force accuracy.

        All arguments/results have force units. Target at most one tenth of
        al_tol times the current force scale or kp*al_gap_tol, so inner Newton
        error cannot mask projected-dual stabilization. Penalty-only analyses
        retain their original requested tolerance.
        """
        if not self.contacts:
            return requested
        dual_scale = max(
            force_scale,
            max(float(np.max(c.multipliers, initial=0.0)) for c in self.contacts),
            np.finfo(np.float64).tiny,
        )
        gap_force = min(c.k_penalty * self.gap_tol for c in self.contacts)
        return min(requested, 0.1 * self.tolerance * dual_scale, 0.1 * gap_force)
