"""Canonical Internal Data Models for Contact Mechanics in Project Sierra.

Milestone 1.0 Stage B of Loop 3 Contact Mechanics Overhaul.
-----------------------------------------------------------
This module defines the typed, candidate-aligned internal records for:
1. ContactCandidate: Search and proximity detection candidate results.
2. DifferentiableContactGeometry: The canonical differentiable geometry record
   G(q) = (g_N, grad g_N, Hess g_N, x_closest, n, feature_id) that serves as
   the SOLE downstream source of geometric quantities.
3. NormalLawState: Projected normal penalty and augmented Lagrangian state.
4. FrictionalConstraintState: Regularized Coulomb stick-slip friction state.
5. ContactAssemblyMap: Global degree of freedom indices and scatter stencils.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np

from psierra.boundary.friction import StickSlipState


@dataclass(frozen=True)
class ContactCandidate:
    """Candidate contact pair detected during spatial proximity search.

    Attributes:
    -----------
    candidate_index : int
        Candidate identifier in candidate-aligned arrays (0 <= index < n_candidates).
    slave_node : int
        Index of the slave node undergoing potential contact.
    master_feature_id : str
        Identifier for the interacting master obstacle or facet (e.g. 'plane', 'tri_12').
    master_nodes : tuple[int, ...]
        Tuple of global master node indices defining the interacting facet/feature.
    search_distance : float
        Proximity search distance / bounding box cutoff [length].
    is_admissible : bool
        Whether this candidate satisfies topological exclusions and normal orientation filters.
    metadata : dict[str, Any]
        Diagnostic search attributes (e.g. bounding box overlap, topological hop count).
    """

    candidate_index: int
    slave_node: int
    master_feature_id: str
    master_nodes: tuple[int, ...] = ()
    search_distance: float = 0.0
    is_admissible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.candidate_index < 0:
            raise ValueError(f"candidate_index must be nonnegative, got {self.candidate_index}")
        if self.slave_node < 0:
            raise ValueError(f"slave_node must be nonnegative, got {self.slave_node}")
        if self.search_distance < 0.0 or not np.isfinite(self.search_distance):
            raise ValueError(f"search_distance must be finite and nonnegative, got {self.search_distance}")


@dataclass(frozen=True)
class DifferentiableContactGeometry:
    """Canonical differentiable contact geometry G(q) at a candidate interaction.

    This record is the SOLE downstream source of geometric quantities for
    normal/tangential constraint laws, energy potentials, and stiffness assembly.

    Attributes:
    -----------
    gap : float
        Signed normal distance g_N [length].
        g_N < 0: penetration.
        g_N > 0: separation.
        +inf: inactive / no admissible feature.
    normal : np.ndarray
        Outward unit normal vector n in R^3 pointing out of master into slave.
        ||n|| = 1.0 (or zero vector if gap is +inf).
    closest_point : np.ndarray
        Orthogonal spatial projection x_closest in R^3 [length] on master surface.
    gradient : np.ndarray
        Local gap gradient grad_q(g_N) in R^M (dimensionless).
        M = 3 for rigid obstacle, M = 12 for node-to-triangle stencil.
    hessian : np.ndarray
        Local gap Hessian Hess_q(g_N) in R^{M x M} [1/length].
        Symmetric: Hess_q(g_N) = Hess_q(g_N)^T.
    feature_id : str
        Canonical feature identifier ('plane', 'sphere', 'cylinder', 'face', 'edge', 'vertex', 'mortar').
    feature_index : int
        Topological index of the active feature (e.g. face ID, edge ID, vertex ID).
    barycentric : np.ndarray | None
        Dimensionless closest-point projection weights xi on master facet (sum(xi) = 1).
    is_smooth_branch : bool
        True if q lies strictly in the interior of a smooth differentiable branch.
        False at feature ridges, vertices, or active-set boundaries.
    units : str
        Dimensional units of length, default 'mm'.
    orientation_convention : str
        Directional convention: 'master_outward' (normal points from master to slave).
    """

    gap: float
    normal: np.ndarray
    closest_point: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray
    feature_id: str
    feature_index: int = 0
    barycentric: np.ndarray | None = None
    is_smooth_branch: bool = True
    units: str = "mm"
    orientation_convention: str = "master_outward"

    def __post_init__(self) -> None:
        if np.isnan(self.gap):
            raise ValueError("Contact gap cannot be NaN")
        norm = np.asarray(self.normal, dtype=np.float64)
        if norm.shape != (3,):
            raise ValueError(f"Normal must be a 3D vector, got shape {norm.shape}")
        if np.isfinite(self.gap):
            n_len = np.linalg.norm(norm)
            if not (0.999 <= n_len <= 1.001 or n_len == 0.0):
                raise ValueError(f"Normal vector must have unit length, got norm={n_len}")
        grad = np.asarray(self.gradient, dtype=np.float64)
        hess = np.asarray(self.hessian, dtype=np.float64)
        if grad.ndim != 1:
            raise ValueError(f"Gradient must be 1D, got ndim={grad.ndim}")
        if hess.ndim != 2 or hess.shape[0] != hess.shape[1]:
            raise ValueError(f"Hessian must be a square matrix, got shape {hess.shape}")
        if hess.shape[0] != grad.shape[0]:
            raise ValueError(
                f"Hessian dimensions {hess.shape} must match gradient length {grad.shape[0]}"
            )
        # Check Hessian symmetry
        if not np.allclose(hess, hess.T, atol=1e-10, rtol=1e-8):
            raise ValueError("Hessian matrix must be symmetric")

    @property
    def feature(self) -> str:
        """Legacy alias for feature_id."""
        return self.feature_id


@dataclass(frozen=True)
class NormalLawState:
    """Projected normal constraint state evaluated from canonical geometry and multipliers.

    Governing law:
        p_N = max(0, lambda_N - k_p * g_N)
    with lambda_N = 0 for pure penalty contact.

    Attributes:
    -----------
    pressure : float
        Projected contact normal force / pressure p_N >= 0 [force].
    multiplier : float
        Retained dual multiplier lambda_N >= 0 [force] (0.0 for pure penalty).
    gap : float
        Signed gap g_N [length] from canonical geometry.
    is_active : bool
        Activity flag: True if p_N > 0.0 (or in penalty: g_N < 0.0).
    potential_energy : float
        Scalar normal potential Pi_N = (p_N^2 - lambda_N^2) / (2 * k_p) [energy].
    first_derivative : float
        Derivative dp_N / dg_N = -k_p on active branch, 0.0 on inactive branch.
    multiplier_update : float
        Projected updated multiplier for Uzawa step: lambda_N^{new} = max(0, lambda_N - k_p * g_N).
    k_penalty : float
        Penalty stiffness parameter k_p [force/length].
    """

    pressure: float
    multiplier: float
    gap: float
    is_active: bool
    potential_energy: float
    first_derivative: float
    multiplier_update: float
    k_penalty: float

    def __post_init__(self) -> None:
        if self.k_penalty <= 0.0 or not np.isfinite(self.k_penalty):
            raise ValueError(f"k_penalty must be finite and positive, got {self.k_penalty}")
        if self.multiplier < 0.0 or not np.isfinite(self.multiplier):
            raise ValueError(f"multiplier must be finite and nonnegative, got {self.multiplier}")
        if self.pressure < -1e-12 or not np.isfinite(self.pressure):
            raise ValueError(f"pressure must be finite and nonnegative, got {self.pressure}")

    @classmethod
    def evaluate(
        cls,
        geometry: DifferentiableContactGeometry,
        k_penalty: float,
        multiplier: float = 0.0,
        formulation: Literal["penalty", "augmented_lagrangian"] = "penalty",
    ) -> NormalLawState:
        """Evaluate normal law state from canonical geometry and stiffness."""
        kp = float(k_penalty)
        lam = float(multiplier) if formulation == "augmented_lagrangian" else 0.0
        g = float(geometry.gap)

        if not np.isfinite(g):
            return cls(
                pressure=0.0,
                multiplier=lam,
                gap=g,
                is_active=False,
                potential_energy=0.0,
                first_derivative=0.0,
                multiplier_update=0.0,
                k_penalty=kp,
            )

        pressure = max(0.0, lam - kp * g)
        is_active = pressure > 0.0 or (formulation == "penalty" and g < 0.0)

        # Potential energy
        if formulation == "augmented_lagrangian":
            energy = (pressure**2 - lam**2) / (2.0 * kp)
        else:
            energy = 0.5 * kp * (min(0.0, g))**2

        first_derivative = -kp if is_active else 0.0
        multiplier_update = pressure

        return cls(
            pressure=pressure,
            multiplier=lam,
            gap=g,
            is_active=is_active,
            potential_energy=energy,
            first_derivative=first_derivative,
            multiplier_update=multiplier_update,
            k_penalty=kp,
        )


@dataclass(frozen=True)
class FrictionalConstraintState:
    """Candidate stick-slip frictional state.

    Attributes:
    -----------
    state : StickSlipState
        Current branch: INACTIVE (0), STICK (1), SLIP (2).
    anchor : np.ndarray
        Committed stick reference point x_rel^0 in R^3 [length].
    trial_anchor : np.ndarray
        Trial sliding reference point in R^3 [length].
    force : np.ndarray
        Tangential traction force f_T in R^3 [force].
    tangential_stiffness : np.ndarray
        Tangent block K_TT in R^{3 x 3} [force/length].
    normal_coupling_stiffness : np.ndarray
        Tangent cross-coupling block K_TN in R^{3 x 3} [force/length].
    slip_direction : np.ndarray
        Unit vector s in R^3 along tangential relative slip.
    slip_distance : float
        Accumulated slip distance [length].
    trial_slip_increment : float
        Relative slip increment ||Delta u_T|| in current step [length].
    multiplier : float
        Friction dual multiplier lambda_T [force].
    """

    state: StickSlipState
    anchor: np.ndarray
    trial_anchor: np.ndarray
    force: np.ndarray
    tangential_stiffness: np.ndarray
    normal_coupling_stiffness: np.ndarray
    slip_direction: np.ndarray
    slip_distance: float = 0.0
    trial_slip_increment: float = 0.0
    multiplier: float = 0.0

    def __post_init__(self) -> None:
        f = np.asarray(self.force, dtype=np.float64)
        if f.shape != (3,):
            raise ValueError(f"Friction force must be 3D, got {f.shape}")
        k_tt = np.asarray(self.tangential_stiffness, dtype=np.float64)
        if k_tt.shape != (3, 3):
            raise ValueError(f"K_TT must be 3x3, got {k_tt.shape}")
        k_tn = np.asarray(self.normal_coupling_stiffness, dtype=np.float64)
        if k_tn.shape != (3, 3):
            raise ValueError(f"K_TN must be 3x3, got {k_tn.shape}")


@dataclass(frozen=True)
class ContactAssemblyMap:
    """Global system degree-of-freedom scatter mapping for one contact candidate.

    Attributes:
    -----------
    slave_node : int
        Slave node index.
    slave_dofs : np.ndarray
        Global DOF indices for slave node (shape (3,)).
    master_nodes : np.ndarray
        Global master node indices (shape (N_master,)).
    master_dofs : np.ndarray
        Global DOF indices for all master nodes (shape (3 * N_master,)).
    stencil_dofs : np.ndarray
        Full combined global DOF indices for the contact stencil (shape (M,)).
    interpolation_weights : np.ndarray
        Partition of unity weights w (shape (1 + N_master,)) such that sum(w) = 0
        when mapped to relative displacement.
    slave_offset : int
        Base offset for the slave body in the global assembly vector.
    master_offset : int
        Base offset for the master body in the global assembly vector.
    """

    slave_node: int
    slave_dofs: np.ndarray
    master_nodes: np.ndarray
    master_dofs: np.ndarray
    stencil_dofs: np.ndarray
    interpolation_weights: np.ndarray
    slave_offset: int = 0
    master_offset: int = 0

    def __post_init__(self) -> None:
        s_dofs = np.asarray(self.slave_dofs, dtype=np.int64)
        if s_dofs.ndim != 1 or len(s_dofs) != 3:
            raise ValueError(f"slave_dofs must have length 3, got {s_dofs.shape}")
        tot_dofs = np.asarray(self.stencil_dofs, dtype=np.int64)
        if tot_dofs.ndim != 1 or len(tot_dofs) % 3 != 0:
            raise ValueError(f"stencil_dofs length must be a multiple of 3, got {tot_dofs.shape}")
        weights = np.asarray(self.interpolation_weights, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError(f"interpolation_weights must be 1D, got {weights.shape}")
        if len(weights) * 3 != len(tot_dofs):
            raise ValueError(
                f"Weights count ({len(weights)}) * 3 must match stencil DOFs length ({len(tot_dofs)})"
            )
