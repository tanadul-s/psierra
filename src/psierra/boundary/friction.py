"""Coulomb Stick-Slip Friction Mechanics for Project Sierra.
==========================================================
Implements 3D active-set Coulomb friction with return-mapping elastoplastic
analogy, consistent algorithmic tangent operators, stick-slip state tracking,
and Augmented Lagrangian tangential multiplier updates.

References:
-----------
- Wriggers, P. (2006). Computational Contact Mechanics. Springer, 2nd ed., ch. 6.
- Laursen, T. A. (2002). Computational Contact and Impact Mechanics. Springer, ch. 4-5.
- Alart, P., & Curnier, A. (1991). A mixed formulation for frictional contact problems
  prone to Newton like solution methods. CMAME, 92(3), 353-375.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import scipy.sparse as spsp


class StickSlipState(IntEnum):
    """Active-set Coulomb contact friction status."""

    INACTIVE = 0  # No contact pressure (p_N <= 0)
    STICK = 1     # Sticking: ||f_T^trial|| <= mu * p_N (zero relative slip rate)
    SLIP = 2      # Slipping: ||f_T^trial|| > mu * p_N (sliding along tangential traction)


@dataclass
class CoulombFrictionSnapshot:
    """Snapshot of candidate friction state for transactional solver updates."""

    anchors: np.ndarray
    multipliers: np.ndarray
    states: np.ndarray
    slip_distances: np.ndarray
    trial_anchors: np.ndarray
    trial_slip_increments: np.ndarray


class CoulombFrictionState:
    """Manages history variables and tangential dual multipliers for candidate contact nodes.

    Parameters:
    -----------
    n_candidates : int
        Number of candidate contact nodes/features.
    """

    def __init__(self, n_candidates: int) -> None:
        self.n_candidates = int(n_candidates)
        # Spatial reference coordinates where stick began (shape: (n_candidates, 3))
        # NaN indicates node is not currently in stick contact
        self.anchors = np.full((self.n_candidates, 3), np.nan, dtype=np.float64)
        # Trial sliding anchors during active-set return mapping within a step
        self.trial_anchors = np.full((self.n_candidates, 3), np.nan, dtype=np.float64)
        # Tangential Augmented Lagrangian multipliers [force] (shape: (n_candidates, 3))
        self.multipliers = np.zeros((self.n_candidates, 3), dtype=np.float64)
        # Current active-set status (shape: (n_candidates,), dtype int)
        self.states = np.full(self.n_candidates, StickSlipState.INACTIVE, dtype=np.int32)
        # Unit slip directions (shape: (n_candidates, 3))
        self.slip_dirs = np.zeros((self.n_candidates, 3), dtype=np.float64)
        # Tangential contact forces [force] (shape: (n_candidates, 3))
        self.forces = np.zeros((self.n_candidates, 3), dtype=np.float64)
        # Cumulative scalar slip distance [length] (shape: (n_candidates,))
        self.slip_distances = np.zeros(self.n_candidates, dtype=np.float64)
        # Incremental scalar slip distance within current step [length] (shape: (n_candidates,))
        self.trial_slip_increments = np.zeros(self.n_candidates, dtype=np.float64)

    def snapshot(self) -> CoulombFrictionSnapshot:
        """Create a deep copy snapshot of the current accepted friction state."""
        return CoulombFrictionSnapshot(
            anchors=self.anchors.copy(),
            multipliers=self.multipliers.copy(),
            states=self.states.copy(),
            slip_distances=self.slip_distances.copy(),
            trial_anchors=self.trial_anchors.copy(),
            trial_slip_increments=self.trial_slip_increments.copy(),
        )

    def restore(self, snap: CoulombFrictionSnapshot) -> None:
        """Restore friction state from a snapshot upon solver step/iteration rejection."""
        self.anchors[:] = snap.anchors
        self.multipliers[:] = snap.multipliers
        self.states[:] = snap.states
        self.slip_distances[:] = snap.slip_distances
        self.trial_anchors[:] = snap.trial_anchors
        self.trial_slip_increments[:] = snap.trial_slip_increments
        self.forces.fill(0.0)
        self.slip_dirs.fill(0.0)

    def reset(self) -> None:
        """Reset all candidate history to uncontacted state."""
        self.anchors.fill(np.nan)
        self.trial_anchors.fill(np.nan)
        self.multipliers.fill(0.0)
        self.states.fill(StickSlipState.INACTIVE)
        self.slip_dirs.fill(0.0)
        self.forces.fill(0.0)
        self.slip_distances.fill(0.0)
        self.trial_slip_increments.fill(0.0)

    def commit_step(self) -> None:
        """Commits converged trial anchors and accumulates slip increments across converged step."""
        slip_mask = (self.states == StickSlipState.SLIP) & ~np.isnan(self.trial_anchors[:, 0])
        self.anchors[slip_mask] = self.trial_anchors[slip_mask]
        self.slip_distances[slip_mask] += self.trial_slip_increments[slip_mask]
        self.trial_slip_increments.fill(0.0)

    def resize(self, n_new: int) -> None:
        """Resize storage if candidate count changes dynamically."""
        if n_new == self.n_candidates:
            return
        new_anchors = np.full((n_new, 3), np.nan, dtype=np.float64)
        new_trial_anchors = np.full((n_new, 3), np.nan, dtype=np.float64)
        new_mult = np.zeros((n_new, 3), dtype=np.float64)
        new_states = np.full(n_new, StickSlipState.INACTIVE, dtype=np.int32)
        new_dirs = np.zeros((n_new, 3), dtype=np.float64)
        new_forces = np.zeros((n_new, 3), dtype=np.float64)
        new_dist = np.zeros(n_new, dtype=np.float64)
        new_trial_inc = np.zeros(n_new, dtype=np.float64)

        n_copy = min(self.n_candidates, n_new)
        if n_copy > 0:
            new_anchors[:n_copy] = self.anchors[:n_copy]
            new_trial_anchors[:n_copy] = self.trial_anchors[:n_copy]
            new_mult[:n_copy] = self.multipliers[:n_copy]
            new_states[:n_copy] = self.states[:n_copy]
            new_dirs[:n_copy] = self.slip_dirs[:n_copy]
            new_forces[:n_copy] = self.forces[:n_copy]
            new_dist[:n_copy] = self.slip_distances[:n_copy]
            new_trial_inc[:n_copy] = self.trial_slip_increments[:n_copy]

        self.anchors = new_anchors
        self.trial_anchors = new_trial_anchors
        self.multipliers = new_mult
        self.states = new_states
        self.slip_dirs = new_dirs
        self.forces = new_forces
        self.slip_distances = new_dist
        self.trial_slip_increments = new_trial_inc
        self.n_candidates = n_new
        self.multipliers = new_mult
        self.states = new_states
        self.slip_dirs = new_dirs
        self.forces = new_forces
        self.slip_distances = new_dist
        self.n_candidates = n_new


class CoulombFrictionModel:
    """Algorithmic 3D Coulomb friction kernel with return mapping and consistent tangent.

    Formulation:
    ------------
    Given normal contact force p_N >= 0, contact normal n, and relative displacement
    delta_u_rel between slave and master contact points:
    1. Tangential projector:
           P_T = I - n (x) n
    2. Tangential relative motion increment:
           delta_g_T = P_T * delta_u_rel
    3. Trial tangential force (opposing slave motion relative to master):
           f_T_trial = lambda_T - k_T * delta_g_T
       where lambda_T is the stored tangential multiplier (lambda_T . n = 0).
    4. Coulomb yield criterion:
           Phi_trial = ||f_T_trial|| - mu * p_N
    5. Active-set evaluation:
       - If p_N <= 0 or mu <= 0:
           f_T = 0, state = INACTIVE / NONE
       - If Phi_trial <= 0:
           f_T = f_T_trial, state = STICK
           K_TT = -k_T * P_T
           K_TN = 0
       - If Phi_trial > 0:
           s = f_T_trial / ||f_T_trial||  (unit slip direction)
           f_T = mu * p_N * s, state = SLIP
           Consistent algorithmic tangent:
           K_TT = -(mu * p_N * k_T / ||f_T_trial||) * (P_T - s (x) s)
           K_TN = -mu * k_N * (s (x) n)  [cross-coupling from normal penetration]
    """

    @staticmethod
    def return_mapping(
        p_N: float,
        normal: np.ndarray,
        delta_u_rel: np.ndarray,
        mu: float,
        k_T: float,
        k_N: float,
        lambda_T: np.ndarray | None = None,
        symmetric_tangent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, StickSlipState, np.ndarray]:
        """Perform radial return mapping for a single contact point.

        Parameters:
        -----------
        p_N : float
            Current compressive normal contact force magnitude [force] (>= 0).
        normal : np.ndarray of shape (3,)
            Unit contact normal (pointing outwards into slave).
        delta_u_rel : np.ndarray of shape (3,)
            Relative displacement vector of slave relative to master stick reference.
        mu : float
            Coulomb coefficient of friction (>= 0).
        k_T : float
            Tangential penalty / stick stiffness [force/length] (> 0).
        k_N : float
            Normal penalty contact stiffness [force/length] (> 0).
        lambda_T : np.ndarray of shape (3,), optional
            Stored tangential Augmented Lagrangian multiplier [force].
        symmetric_tangent : bool, default=False
            If True, omits the unsymmetric normal-tangential cross-coupling term K_TN
            (suitable when an iterative symmetric linear solver such as CG is required).

        Returns:
        --------
        f_T : np.ndarray of shape (3,)
            Tangential friction force vector acting on slave [force].
        K_TT : np.ndarray of shape (3, 3)
            Tangential-tangential stiffness contribution: -df_T / d(delta_u_rel).
        K_TN : np.ndarray of shape (3, 3)
            Tangential-normal cross-coupling stiffness: -df_T / du_N.
        state : StickSlipState
            Active-set contact status (INACTIVE, STICK, or SLIP).
        slip_dir : np.ndarray of shape (3,)
            Unit sliding direction vector s (in tangent plane).
        """
        n = np.asarray(normal, dtype=np.float64)
        n_sq = float(np.dot(n, n))
        if n_sq > 1e-14:
            n = n / np.sqrt(n_sq)
        else:
            return (
                np.zeros(3, dtype=np.float64),
                np.zeros((3, 3), dtype=np.float64),
                np.zeros((3, 3), dtype=np.float64),
                StickSlipState.INACTIVE,
                np.zeros(3, dtype=np.float64),
            )

        # Inactive or frictionless
        if p_N <= 0.0 or mu <= 0.0:
            return (
                np.zeros(3, dtype=np.float64),
                np.zeros((3, 3), dtype=np.float64),
                np.zeros((3, 3), dtype=np.float64),
                StickSlipState.INACTIVE if p_N <= 0.0 else StickSlipState.STICK,
                np.zeros(3, dtype=np.float64),
            )

        # Tangential projector: P_T = I - n (x) n
        I3 = np.eye(3, dtype=np.float64)
        P_T = I3 - np.outer(n, n)

        # Tangential relative displacement
        delta_g_T = P_T @ np.asarray(delta_u_rel, dtype=np.float64)

        # Trial tangential force
        f_T_trial = -k_T * delta_g_T
        if lambda_T is not None:
            # Multiplier must also lie in the tangent plane
            lam_T = P_T @ np.asarray(lambda_T, dtype=np.float64)
            f_T_trial += lam_T

        f_trial_norm = float(np.linalg.norm(f_T_trial))
        coulomb_limit = mu * p_N

        # Yield condition: Phi = ||f_T_trial|| - mu * p_N
        if f_trial_norm <= coulomb_limit or f_trial_norm < 1e-14:
            # STICK STATE
            f_T = f_T_trial
            # Algorithmic tangent for stick: -df_T / d(u_rel) = k_T * P_T
            K_TT = k_T * P_T
            K_TN = np.zeros((3, 3), dtype=np.float64)
            s_dir = f_T_trial / f_trial_norm if f_trial_norm > 1e-14 else np.zeros(3, dtype=np.float64)
            return f_T, K_TT, K_TN, StickSlipState.STICK, s_dir

        # SLIP STATE
        # Unit slip direction along trial friction force
        s_dir = f_T_trial / f_trial_norm
        f_T = coulomb_limit * s_dir

        # Consistent algorithmic tangent operator:
        # K_TT = -df_T / d(u_rel) = (mu * p_N * k_T / f_trial_norm) * (P_T - s (x) s)
        ratio = (coulomb_limit * k_T) / f_trial_norm
        K_TT = ratio * (P_T - np.outer(s_dir, s_dir))

        # Cross-coupling tangent with normal penetration:
        # Since p_N = max(0, lambda_N - k_N * g_N), dp_N / du_s = -k_N * n
        # df_T / du_s = mu * (dp_N / du_s) * s = -mu * k_N * (s (x) n)
        # Therefore LHS stiffness matrix contribution is K_TN = mu * k_N * (s (x) n)
        if not symmetric_tangent:
            K_TN = (mu * k_N) * np.outer(s_dir, n)
        else:
            K_TN = np.zeros((3, 3), dtype=np.float64)

        return f_T, K_TT, K_TN, StickSlipState.SLIP, s_dir

    @staticmethod
    def project_tangential_multiplier(
        lambda_T: np.ndarray,
        delta_g_T: np.ndarray,
        p_N: float,
        mu: float,
        k_T: float,
        normal: np.ndarray,
    ) -> np.ndarray:
        """Update tangential Augmented Lagrangian multiplier via Uzawa projection.

        lambda_T^{(k+1)} = proj_{mu * p_N}( lambda_T^{(k)} - k_T * delta_g_T )
        """
        n = np.asarray(normal, dtype=np.float64)
        n_sq = float(np.dot(n, n))
        if n_sq > 1e-14:
            n = n / np.sqrt(n_sq)
        P_T = np.eye(3, dtype=np.float64) - np.outer(n, n)

        trial_lam = P_T @ (lambda_T - k_T * delta_g_T)
        trial_norm = float(np.linalg.norm(trial_lam))
        limit = mu * p_N
        if trial_norm <= limit or trial_norm < 1e-14:
            return trial_lam
        return (limit / trial_norm) * trial_lam

    @staticmethod
    def assemble_two_body_stiffness(
        K_TT: np.ndarray,
        K_TN: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Assemble the 12x12 local friction stiffness matrix for node-to-facet contact.

        Parameters:
        -----------
        K_TT : np.ndarray of shape (3, 3)
            Tangential tangent block.
        K_TN : np.ndarray of shape (3, 3)
            Normal-tangential cross-coupling block.
        weights : np.ndarray of shape (4,)
            Barycentric weights: [1.0, -xi_1, -xi_2, -xi_3].

        Returns:
        --------
        K_local : np.ndarray of shape (12, 12)
            Local stiffness coupling slave and 3 master facet nodes.
        """
        # Full 3x3 block
        K_block = K_TT + K_TN
        # Kronecker product with outer product of weights
        W = np.outer(weights, weights)  # (4, 4)
        return np.kron(W, K_block)  # (12, 12)
