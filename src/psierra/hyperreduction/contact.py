"""
Hyper-Reduced Contact Mechanics via Sub-Basis Boundary Interface Projection (Method 1).

Enforces non-penetration contact constraints on ECSW hyper-reduced models without
evaluating full-order volumetric equations:
1. Reconstructs boundary surface nodal positions in microseconds using the pre-sliced
   surface sub-basis: x_surf = X_surf + Phi_surf @ u_r in R^{N_surf x 3}.
2. Evaluates analytical/mesh collision queries against rigid obstacles (sphere, plane, cylinder, skull).
3. Projects active contact forces f_{c, i} and rank-1 contact tangent stiffnesses K_{c, i} = k_p (n_i (x) n_i)
   directly onto the reduced generalized coordinates R^r:
       f_{r, c}(u_r) = sum_{i in active} Phi_{surf, i}^T f_{c, i} in R^r
       K_{r, c}(u_r) = sum_{i in active} k_p^* (v_i @ v_i^T) in R^{r x r}
   where v_i = Phi_{surf, i}^T n_i in R^r.
4. Strictly preserves tangent stiffness symmetry: K_{r, c} = K_{r, c}^T (positive semi-definite).
5. Supports both Penalty and Uzawa Augmented Lagrangian formulations, defaulting to
   Augmented Lagrangian for robust non-penetration without penalty ill-conditioning.
"""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np

from psierra.collision.shapes import CollisionShape
from psierra.collision.detector import CollisionDetector, ContactQueryResult
from psierra.collision.raytrace import RayMeshObstacle
from psierra.reduction.basis import SurfacePODBasis
from psierra.hyperreduction.model import HyperReducedNonlinearModel


@dataclass
class ContactTelemetry:
    """Real-time contact diagnostics for logging and interactive HUD display."""

    formulation: str
    num_active: int
    max_penetration: float
    max_multiplier: float
    max_contact_force: float
    total_contact_force_norm: float


class HyperReducedRigidContact:
    """
    Deformable-to-rigid contact operator for hyper-reduced finite-strain models.

    Parameters:
    -----------
    model : HyperReducedNonlinearModel
        Hyper-reduced physical domain containing surface modal sub-basis and reference coordinates.
    obstacle : CollisionShape, RayMeshObstacle, or Sequence[CollisionShape]
        Master obstacle (e.g. RigidPlane, RigidSphere, or RayMeshObstacle for skull).
    k_penalty : float, default=1.0
        Normal penalty stiffness [N/mm].
    c_damping : float, default=0.0
        Contact velocity damping coefficient [N*s/mm].
    formulation : {'augmented_lagrangian', 'penalty'}, default='augmented_lagrangian'
        Contact constraint enforcement scheme. Augmented Lagrangian is the default
        as it eliminates interface penetration without requiring massive penalty values.
    tolerance : float, default=0.0
        Proximity activation threshold (signed gap g <= tolerance).
    candidate_node_indices : Sequence[int] or np.ndarray, optional
        Subset of surface node indices to test for contact. If None, tests all surface nodes.
    """

    def __init__(
        self,
        model: HyperReducedNonlinearModel,
        obstacle: CollisionShape | Sequence[CollisionShape] | Any,
        k_penalty: float = 1.0,
        c_damping: float = 0.0,
        formulation: Literal["augmented_lagrangian", "penalty"] = "augmented_lagrangian",
        tolerance: float = 0.0,
        candidate_node_indices: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.r = model.r
        self.surface_basis: SurfacePODBasis = model.surface_basis

        # Resolve master collision detector
        if (
            hasattr(obstacle, "surface")
            or (hasattr(obstacle, "mesh") and hasattr(obstacle.mesh, "surface"))
            or hasattr(obstacle, "ugrid")
        ):
            master_obstacle = RayMeshObstacle(obstacle)
        else:
            master_obstacle = obstacle

        self.detector = (
            master_obstacle
            if isinstance(master_obstacle, CollisionDetector)
            else CollisionDetector(master_obstacle)
        )

        if not np.isfinite(k_penalty) or k_penalty <= 0.0:
            raise ValueError(f"k_penalty must be finite and positive, got {k_penalty}")
        self.k_penalty = float(k_penalty)
        self.c_damping = float(c_damping)

        if formulation not in ("augmented_lagrangian", "penalty"):
            raise ValueError(
                f"formulation must be 'augmented_lagrangian' or 'penalty', got {formulation}"
            )
        self.formulation = formulation
        self.tolerance = float(tolerance)

        # Reference surface coordinates
        self.reference_surface_nodes = self.model.domain.mesh.nodes[
            self.model.domain.mesh.surface.node_indices
        ].copy()  # (N_surf, 3)
        self.n_surface_nodes = len(self.reference_surface_nodes)

        # Pre-slice 3D modal sub-basis: (N_surf, 3, r)
        self.Phi_surf_3d = self.surface_basis.basis.reshape(self.n_surface_nodes, 3, self.r)

        # Candidate subset indexing (supports both global mesh node indices and surface-local indices)
        if candidate_node_indices is not None:
            cand_arr = np.asarray(candidate_node_indices, dtype=np.int64).ravel()
            surf_nodes = self.model.domain.mesh.surface.node_indices
            if np.any(cand_arr >= self.n_surface_nodes) or np.any(cand_arr < 0):
                # Map global mesh node IDs to surface sub-mesh indices
                global_to_surf = {int(gid): s_idx for s_idx, gid in enumerate(surf_nodes)}
                mapped = [global_to_surf[gid] for gid in cand_arr if gid in global_to_surf]
                self.candidate_indices = np.array(mapped, dtype=np.int64)
            else:
                self.candidate_indices = cand_arr
        else:
            self.candidate_indices = np.arange(self.n_surface_nodes, dtype=np.int64)

        self.n_candidates = len(self.candidate_indices)

        # Uzawa normal contact multipliers [N]
        self.multipliers = np.zeros(self.n_candidates, dtype=np.float64)

        # Cached telemetry from latest query
        self._last_telemetry = ContactTelemetry(
            formulation=self.formulation,
            num_active=0,
            max_penetration=0.0,
            max_multiplier=0.0,
            max_contact_force=0.0,
            total_contact_force_norm=0.0,
        )

    def toggle_formulation(self) -> str:
        """
        Toggles between 'augmented_lagrangian' and 'penalty' formulations at runtime.

        Returns:
        --------
        new_formulation : str
        """
        if self.formulation == "augmented_lagrangian":
            self.formulation = "penalty"
        else:
            self.formulation = "augmented_lagrangian"
            self.multipliers.fill(0.0)
        return self.formulation

    def reset_state(self) -> None:
        """Resets contact multipliers and cached telemetry."""
        self.multipliers.fill(0.0)
        self._last_telemetry = ContactTelemetry(
            formulation=self.formulation,
            num_active=0,
            max_penetration=0.0,
            max_multiplier=0.0,
            max_contact_force=0.0,
            total_contact_force_norm=0.0,
        )

    def evaluate_gaps(self, u_r: np.ndarray) -> np.ndarray:
        """
        Evaluates signed normal constraint gaps for candidate surface nodes:
        g < 0 indicates penetration into the obstacle.
        """
        u_surf = self.surface_basis.reconstruct_surface_displacements(u_r)
        x_surf = self.reference_surface_nodes + u_surf
        x_cand = x_surf[self.candidate_indices]
        query = self.detector.query(x_cand, tolerance=self.tolerance)
        return query.gaps

    def evaluate_reduced_forces_and_stiffness(
        self,
        u_r: np.ndarray,
        v_r: np.ndarray | None = None,
        a1: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates projected contact forces f_{r, c} and tangent stiffness K_{r, c}.

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,)
            Generalized reduced displacement coordinates.
        v_r : np.ndarray of shape (r,), optional
            Generalized reduced velocity coordinates (for contact velocity damping).
        a1 : float, default=0.0
            Dynamic integration parameter gamma / (beta * dt).

        Returns:
        --------
        f_r_c : np.ndarray of shape (r,)
            Projected contact repulsive force vector opposing penetration.
        K_r_c : np.ndarray of shape (r, r)
            Projected contact tangent stiffness matrix (strictly symmetric positive semi-definite).
        """
        u_r_vec = np.asarray(u_r, dtype=np.float64).ravel()

        # 1. Instantaneous surface reconstruction: x_surf in R^{N_surf x 3}
        u_surf = self.surface_basis.reconstruct_surface_displacements(u_r_vec)
        x_surf = self.reference_surface_nodes + u_surf
        x_cand = x_surf[self.candidate_indices]

        # 2. Collision proximity detection
        query: ContactQueryResult = self.detector.query(x_cand, tolerance=self.tolerance)
        gaps = query.gaps  # signed normal gap: g < 0 indicates penetration
        normals = query.normals  # unit outward normal pointing from obstacle into slave

        # 3. Active Contact Determination
        if self.formulation == "augmented_lagrangian":
            # Active if in proximity/penetration zone and effective pressure > 0
            active_mask = (gaps <= self.tolerance) & (
                (self.multipliers - self.k_penalty * gaps) > 0.0
            )
        else:
            # Penalty active if gap <= tolerance
            active_mask = gaps <= self.tolerance

        active_indices = np.flatnonzero(active_mask)
        num_active = len(active_indices)

        f_r_c = np.zeros(self.r, dtype=np.float64)
        K_r_c = np.zeros((self.r, self.r), dtype=np.float64)

        if num_active == 0:
            self._last_telemetry = ContactTelemetry(
                formulation=self.formulation,
                num_active=0,
                max_penetration=float(np.max(np.maximum(-gaps, 0.0), initial=0.0)),
                max_multiplier=float(np.max(self.multipliers, initial=0.0)),
                max_contact_force=0.0,
                total_contact_force_norm=0.0,
            )
            return f_r_c, K_r_c

        # 4. Normal contact scalar pressures p_i
        g_active = gaps[active_indices]
        n_active = normals[active_indices]

        if self.formulation == "augmented_lagrangian":
            lams = self.multipliers[active_indices]
            p_normal = np.maximum(0.0, lams - self.k_penalty * g_active)
        else:
            p_normal = -self.k_penalty * g_active  # since g <= 0, -kp*g > 0

        # Optional velocity damping: f_d = -c_p * min(v . n, 0) * n
        if self.c_damping > 0.0 and v_r is not None:
            v_surf = self.surface_basis.reconstruct_surface_displacements(v_r)
            v_cand = v_surf[self.candidate_indices]
            v_active = v_cand[active_indices]
            vn = np.einsum("ij,ij->i", v_active, n_active)
            vn_pen = np.minimum(vn, 0.0)
            p_normal += -self.c_damping * vn_pen

        # 5. Project onto reduced coordinates:
        # For each active node i:
        #   Phi_i is (3, r)
        #   v_i = Phi_i^T @ n_i in R^r
        #   f_{r, c} += p_normal[i] * v_i
        #   K_{r, c} += kp^* * (v_i @ v_i^T)
        kp_eff = self.k_penalty + a1 * self.c_damping
        cand_indices_active = self.candidate_indices[active_indices]
        Phi_active = self.Phi_surf_3d[cand_indices_active]  # shape: (num_active, 3, r)

        # Vectorized modal normal projection: v_proj has shape (num_active, r)
        # v_proj[i, k] = sum_d Phi_active[i, d, k] * n_active[i, d]
        v_proj = np.einsum("idr,id->ir", Phi_active, n_active)

        # Force accumulation: f_{r, c} = v_proj^T @ p_normal
        f_r_c = v_proj.T @ p_normal

        # Stiffness accumulation: K_{r, c} = kp_eff * (v_proj^T @ v_proj)
        K_r_c = kp_eff * (v_proj.T @ v_proj)

        # Enforce exact floating-point symmetry: K_{r, c} = 0.5 * (K_{r, c} + K_{r, c}^T)
        K_r_c = 0.5 * (K_r_c + K_r_c.T)

        # Update telemetry cache
        penetrations = np.maximum(-gaps, 0.0)
        self._last_telemetry = ContactTelemetry(
            formulation=self.formulation,
            num_active=num_active,
            max_penetration=float(np.max(penetrations, initial=0.0)),
            max_multiplier=float(np.max(self.multipliers, initial=0.0)),
            max_contact_force=float(np.max(p_normal, initial=0.0)),
            total_contact_force_norm=float(np.linalg.norm(f_r_c)),
        )

        return f_r_c, K_r_c

    def update_multipliers(self, u_r: np.ndarray | None = None) -> dict[str, float]:
        """
        Executes the Uzawa multiplier update at convergence:
            lambda_i <- max(0, lambda_i - kp * g_i)
        Penalty formulation is a no-op.

        Returns:
        --------
        stats : dict
            Diagnostic metrics (max_penetration, max_multiplier, num_active, etc.).
        """
        if self.formulation != "augmented_lagrangian":
            return {
                "max_penetration": self._last_telemetry.max_penetration,
                "max_multiplier": 0.0,
                "max_delta_multiplier": 0.0,
                "num_active": self._last_telemetry.num_active,
            }

        if u_r is not None:
            gaps = self.evaluate_gaps(u_r)
        else:
            gaps = self.evaluate_gaps(np.zeros(self.r))

        previous = self.multipliers.copy()
        # Enforce strict complementarity: zero multiplier on separated nodes (g > tolerance)
        in_contact = gaps <= self.tolerance
        self.multipliers[~in_contact] = 0.0
        self.multipliers[in_contact] = np.maximum(
            0.0, previous[in_contact] - self.k_penalty * gaps[in_contact]
        )
        max_delta = float(np.max(np.abs(self.multipliers - previous), initial=0.0))

        return {
            "max_penetration": float(np.max(np.maximum(-gaps, 0.0), initial=0.0)),
            "max_multiplier": float(np.max(self.multipliers, initial=0.0)),
            "max_delta_multiplier": max_delta,
            "num_active": int(np.count_nonzero(self.multipliers > 0.0)),
        }

    def get_telemetry(self) -> ContactTelemetry:
        """Returns the most recent contact diagnostics."""
        return self._last_telemetry
