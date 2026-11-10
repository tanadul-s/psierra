"""
Unit & Integration Tests for Pillar 3: Algorithmic Regularization.
===================================================================
Verifies:
1. ContactAwareLineSearch error classification (is_element_inversion_error).
2. ContactAwareLineSearch Armijo descent, direct convergence, and strict reduction.
3. Element inversion detection (det(F) <= 0) forcing backtracking away from unphysical collapse.
4. Line search failure when element inversion cannot be resolved at any step scale.
5. Full Newton step acceptance with bounded residual growth on contact active set changes.
6. Contact-aware fallback acceptance when Armijo descent is blocked by contact non-smoothness.
7. Dual rebound relaxation in ContactMultiplierState (relax_rebound).
8. Dual rebound relaxation in ContactAugmentation (rebound_relaxation).
9. Elimination of artificial sticky contact adhesion upon dynamic rebound in NonlinearDynamicAnalysis.
10. Seamless integration with NonlinearStaticAnalysis under contact loading.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import pyvista as pv
import scipy.sparse as sp

from psierra.analysis.augmentation import ContactAugmentation
from psierra.analysis.line_search import (
    ContactAwareLineSearch,
    LineSearchResult,
    is_element_inversion_error,
)
from psierra.analysis.nonlinear_dynamic import NonlinearDynamicAnalysis
from psierra.analysis.nonlinear_static import NonlinearStaticAnalysis
from psierra.boundary.contact import RigidContact
from psierra.collision.shapes import RigidPlane
from psierra.domain.base import Domain
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint


class SingleDofSpringDomain(Domain):
    """1D spring domain in 3D coordinates for testing contact and line search mechanics."""

    def __init__(self, k_spring: float = 100.0, mass: float = 1.0) -> None:
        self.k_spring = float(k_spring)
        self.mass = float(mass)
        self.reference = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        self.u = np.zeros(3, dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.a = np.zeros(3, dtype=np.float64)
        self.invert_at_z: float | None = None  # Mock element inversion trigger

    @property
    def n_dofs(self) -> int:
        return 3

    @property
    def current_nodes(self) -> np.ndarray:
        return self.reference + self.u.reshape(1, 3)

    def update_state(
        self,
        u: np.ndarray | None = None,
        v: np.ndarray | None = None,
        a: np.ndarray | None = None,
    ) -> None:
        if u is not None:
            self.u = np.asarray(u, dtype=np.float64).copy()
        if v is not None:
            self.v = np.asarray(v, dtype=np.float64).copy()
        if a is not None:
            self.a = np.asarray(a, dtype=np.float64).copy()

    def assemble_internal_forces(self) -> np.ndarray:
        if self.invert_at_z is not None and self.u[2] < self.invert_at_z:
            raise ValueError(f"Element inversion detected: det(F) = {self.u[2]:.4e} <= 0.")
        return self.k_spring * self.u

    def assemble_tangent_stiffness_matrix(self) -> sp.csr_matrix:
        return sp.diags([self.k_spring, self.k_spring, self.k_spring], format="csr")

    def assemble_stiffness_matrix(self) -> sp.csr_matrix:
        return self.assemble_tangent_stiffness_matrix()

    def assemble_mass_matrix(self) -> sp.csr_matrix:
        return sp.diags([self.mass, self.mass, self.mass], format="csr")

    def save_state(self, filepath: Path, **kwargs: Any) -> None:
        pass


class TestLineSearchErrorClassification:
    """Tests differentiation between element inversion and regular exceptions."""

    def test_element_inversion_detected(self) -> None:
        assert is_element_inversion_error(
            ValueError("Element inversion detected: det(F) = -1.234e-03 <= 0.")
        )
        assert is_element_inversion_error(
            ValueError("Element inversion detected in Tet4Corotational: det(F) = 0.0000e+00 <= 0.")
        )
        assert is_element_inversion_error(ValueError("Negative Jacobian encountered"))
        assert is_element_inversion_error(FloatingPointError("overflow in stress evaluation"))

    def test_non_inversion_errors_differentiated(self) -> None:
        assert not is_element_inversion_error(ValueError("k_penalty must be finite and positive"))
        assert not is_element_inversion_error(RuntimeError("Sparse SuperLU factorization failed"))
        assert not is_element_inversion_error(KeyError("missing key"))


class TestContactAwareLineSearchMechanics:
    """Tests line search descent conditions, element inversion backtracks, and contact fallbacks."""

    def test_direct_convergence_acceptance(self) -> None:
        ls = ContactAwareLineSearch()
        res = ls.search(lambda alpha: 1e-8, current_norm=10.0, inner_tolerance=1e-6)
        assert res.success
        assert res.alpha == 1.0
        assert res.candidate_norm == 1e-8
        assert not res.element_inversion_encountered

    def test_strict_reduction_acceptance(self) -> None:
        ls = ContactAwareLineSearch()
        res = ls.search(lambda alpha: 5.0, current_norm=10.0, inner_tolerance=1e-6)
        assert res.success
        assert res.alpha == 1.0
        assert res.candidate_norm == 5.0

    def test_armijo_backtracking_quadratic(self) -> None:
        ls = ContactAwareLineSearch(contraction_factor=0.5, armijo_c1=1e-4)
        # Function where alpha=1.0 overshoots (norm=12.0 > 10.0), but alpha=0.5 decreases (norm=4.0)
        def eval_fn(alpha: float) -> float:
            return 12.0 if alpha > 0.6 else 4.0

        res = ls.search(eval_fn, current_norm=10.0, inner_tolerance=1e-6)
        assert res.success
        assert res.alpha == 0.5
        assert res.candidate_norm == 4.0
        assert res.iterations == 2

    def test_element_inversion_forces_backtracking(self) -> None:
        ls = ContactAwareLineSearch(contraction_factor=0.5)

        # alpha=1.0 and 0.5 invert; alpha=0.25 is valid
        def eval_fn(alpha: float) -> float:
            if alpha > 0.3:
                raise ValueError("Element inversion detected: det(F) = -0.05 <= 0.")
            return 5.0

        res = ls.search(eval_fn, current_norm=10.0, inner_tolerance=1e-6)
        assert res.success
        assert res.alpha == 0.25
        assert res.element_inversion_encountered
        assert not res.accepted_as_contact_fallback

    def test_persistent_element_inversion_fails_gracefully(self) -> None:
        ls = ContactAwareLineSearch(max_iterations=5)

        def eval_fn(alpha: float) -> float:
            raise ValueError("Element inversion detected: det(F) = -0.5 <= 0.")

        res = ls.search(eval_fn, current_norm=10.0, inner_tolerance=1e-6)
        assert not res.success
        assert res.element_inversion_encountered
        assert "Element inversion detected" in (res.rejection_reason or "")

    def test_full_newton_step_bounded_residual_for_contact(self) -> None:
        ls = ContactAwareLineSearch(full_step_tolerance=1.5)
        # Contact activation causes 20% residual bump at alpha=1.0
        res = ls.search(
            lambda alpha: 1.2 * 10.0,
            current_norm=10.0,
            inner_tolerance=1e-6,
            contact_active=True,
        )
        assert res.success
        assert res.alpha == 1.0
        assert res.candidate_norm == 12.0

    def test_contact_aware_fallback_when_armijo_stalls(self) -> None:
        ls = ContactAwareLineSearch(
            max_iterations=4,
            full_step_tolerance=1.1,
            contact_fallback_tolerance=3.0,
        )
        # All alphas have residual ~2.0 * R_0 due to non-smooth contact barrier
        res = ls.search(
            lambda alpha: 2.0 * 10.0,
            current_norm=10.0,
            inner_tolerance=1e-6,
            contact_active=True,
        )
        assert res.success
        assert res.accepted_as_contact_fallback
        assert res.candidate_norm == 20.0


class TestDualReboundRelaxation:
    """Tests zeroing retained AL multipliers upon physical separation (g > 0)."""

    def test_contact_multiplier_state_relax_rebound(self) -> None:
        domain = SingleDofSpringDomain()
        plane = RigidPlane(origin=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 1.0]))
        contact = RigidContact(domain, plane, k_penalty=1000.0, formulation="augmented_lagrangian")

        # Scenario A: Penetrating (gap = -0.05 < 0)
        domain.u[2] = -0.05
        contact.multipliers[:] = 50.0
        relaxed = contact.relax_rebound(gap_tol=1e-6)
        assert relaxed == 0
        assert contact.multipliers[0] == 50.0

        # Scenario B: Separated (gap = +0.02 > 0)
        domain.u[2] = +0.02
        relaxed = contact.relax_rebound(gap_tol=1e-6)
        assert relaxed == 1
        assert contact.multipliers[0] == 0.0

    def test_augmentation_rebound_relaxation_assess(self) -> None:
        domain = SingleDofSpringDomain()
        plane = RigidPlane(origin=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 1.0]))
        contact = RigidContact(domain, plane, k_penalty=1000.0, formulation="augmented_lagrangian")
        contact.multipliers[:] = 50.0

        # Node in the air at z = +0.02 (gap = +0.02)
        domain.u[2] = 0.02

        # Standard augmentation retains sticky pressure 50 - 1000*0.02 = 30 N
        aug_standard = ContactAugmentation(
            [contact], max_iterations=5, gap_tol=1e-6, tolerance=1e-6, rebound_relaxation=False
        )
        status_std = aug_standard.assess(domain.current_nodes)
        assert status_std.proposed[0][0] == pytest.approx(30.0, abs=1e-10)

        # Relaxed augmentation immediately zeroes sticky pressure (0.0 N)
        aug_relaxed = ContactAugmentation(
            [contact], max_iterations=5, gap_tol=1e-6, tolerance=1e-6, rebound_relaxation=True
        )
        status_rel = aug_relaxed.assess(domain.current_nodes)
        assert status_rel.proposed[0][0] == 0.0

    def test_dynamic_rebound_relaxation_integration(self) -> None:
        """Verify dynamic solver eliminates sticky contact force upon physical rebound."""
        domain = SingleDofSpringDomain(k_spring=200.0, mass=1.0)
        plane = RigidPlane(origin=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 1.0]))
        contact = RigidContact(domain, plane, k_penalty=2000.0, formulation="augmented_lagrangian")

        # Initial state: node penetrating with upward rebound velocity
        domain.u[2] = -0.01
        domain.v[2] = 0.5  # high upward velocity pushing node into separation
        contact.multipliers[:] = 40.0

        dirichlet = DirichletManager(domain.n_dofs)
        dirichlet.add_constraint(DirichletConstraint([0], components=(0, 1), value=0.0))

        analysis = NonlinearDynamicAnalysis(domain)
        result = analysis.solve(
            dirichlet=dirichlet,
            f_ext=np.zeros(3),
            dt=0.1,
            n_steps=2,
            contact=contact,
            rebound_relaxation=True,
            al_max_iterations=10,
            al_gap_tol=1e-5,
            al_tol=1e-4,
            verbose=False,
        )

        assert result.converged
        # Node has separated into positive z without being pulled down by artificial adhesion
        assert domain.u[2] > 0.0
        assert contact.multipliers[0] == 0.0
        forces = contact.evaluate_forces(n_dofs=3)
        assert np.allclose(forces, 0.0, atol=1e-12)

    def test_static_contact_aware_line_search_integration(self) -> None:
        """Verify NonlinearStaticAnalysis solves equilibrium smoothly with ContactAwareLineSearch."""
        domain = SingleDofSpringDomain(k_spring=100.0)
        plane = RigidPlane(origin=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 1.0]))
        contact = RigidContact(domain, plane, k_penalty=1000.0, formulation="augmented_lagrangian")

        dirichlet = DirichletManager(domain.n_dofs)
        dirichlet.add_constraint(DirichletConstraint([0], components=(0, 1), value=0.0))

        # Push downward into floor with -50 N load
        f_ext = np.array([0.0, 0.0, -50.0])

        solver = NonlinearStaticAnalysis(domain)
        result = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            contact=contact,
            n_steps=2,
            rebound_relaxation=True,
            al_max_iterations=20,
            al_gap_tol=1e-6,
            al_tol=1e-6,
            verbose=False,
        )

        assert result.converged
        # Equilibrium: spring force k*u + contact force f_c == 50 N
        # With AL gap_tol=1e-6: u_z approx 0.0, contact multiplier approx 50 N
        assert abs(domain.u[2]) < 1e-4
        assert contact.multipliers[0] == pytest.approx(50.0, rel=1e-3)
