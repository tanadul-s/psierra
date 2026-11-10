"""
Contact Mechanics & Penalty Boundary Formulations for Project Sierra.
====================================================================
Implements non-penetration penalty contact constraints, contact velocity damping,
tangent stiffness linearization, and potential energy tracking for
deformable-to-rigid (and multi-body) interactions.
"""

from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.sparse as spsp
import vtk

from psierra.domain.base import Domain
from psierra.collision.shapes import CollisionShape
from psierra.collision.raytrace import RayMeshObstacle
from psierra.collision.detector import CollisionDetector, ContactQueryResult
from psierra.collision.triangle import TriangleContactGeometry, TriangleGap


class ContactMultiplierState:
    """Explicit normal-multiplier state shared by contact formulations.

    Multipliers have force units and are aligned with ``evaluate_gaps()``.
    Candidate gaps may be +infinity when no self-contact feature is admissible;
    their projected multiplier is zero. State restoration invalidates derived
    contact data, so rejected solver increments cannot leak trial history.
    """

    multipliers: np.ndarray
    k_penalty: float
    formulation: str

    def evaluate_gaps(self, current_nodes: np.ndarray | None = None) -> np.ndarray:
        """Return candidate-aligned signed normal constraint gaps [length]."""
        raise NotImplementedError

    def snapshot_multipliers(self) -> np.ndarray:
        """Copy the current accepted normal multipliers [force]."""
        return self.multipliers.copy()

    def invalidate_state(self) -> None:
        """Invalidate derived queries after a multiplier/state transaction."""

    def set_multipliers(self, values: np.ndarray) -> None:
        """Set finite nonnegative multipliers [force] and invalidate queries."""
        values = np.asarray(values, dtype=np.float64)
        if (
            values.shape != self.multipliers.shape
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise ValueError(
                "Multipliers must be a finite nonnegative candidate-aligned vector"
            )
        self.multipliers[:] = values
        self.invalidate_state()

    def update_multipliers(self) -> dict[str, float]:
        """Project lambda<-max(0,lambda-kp*g); penalty formulation is a no-op.

        This updates dual state only; it does not establish equilibrium.
        Batch analyses solve equilibrium again after every committed update.
        """
        if self.formulation != "augmented_lagrangian":
            return {
                "max_penetration": 0.0,
                "max_multiplier": float(np.max(self.multipliers, initial=0.0)),
                "max_delta_multiplier": 0.0,
                "num_active": int(np.count_nonzero(self.multipliers > 0.0)),
            }
        gaps = self.evaluate_gaps()
        previous = self.snapshot_multipliers()
        self.set_multipliers(np.maximum(0.0, previous - self.k_penalty * gaps))
        return {
            "max_penetration": float(np.max(np.maximum(-gaps, 0.0), initial=0.0)),
            "max_multiplier": float(np.max(self.multipliers, initial=0.0)),
            "max_delta_multiplier": float(
                np.max(np.abs(self.multipliers - previous), initial=0.0)
            ),
            "num_active": int(np.count_nonzero(self.multipliers > 0.0)),
        }


class RigidContact(ContactMultiplierState):
    """
    Deformable-to-Rigid Contact Formulation for Project Sierra.

    Enforces the non-penetration condition between a deformable body (slave)
    and an analytical or 3D mesh rigid obstacle (master) via virtual elastic normal springs
    (penalty formulation) or adaptive Uzawa multipliers (augmented Lagrangian formulation):
        f_c = f_N * n
    where:
        f_N = -k_p * g_N (penalty formulation, active when signed gap g_N < tolerance)
        f_N = max(0, lambda_N - k_p * g_N) (augmented Lagrangian formulation)

    Also incorporates contact velocity damping:
        f_d = -c_p * (v . n) * n  (active when v . n < 0, penetrating velocity)
    to eliminate high-frequency contact chatter and control impact restitution.

    The planar normal tangent is K_c=kp*(n tensor n). Curved-obstacle normal
    variation and dissipative geometric derivatives remain approximate. AL
    activity follows max(0,lambda-kp*g)>0, including positive-gap opening while
    stored pressure remains. Fixed-lambda potential is (p^2-lambda^2)/(2*kp).
    Batch analyses enforce the outer multiplier/equilibrium lifecycle described
    in docs/06_solvers_and_analysis_pipelines.md.
    """

    def __init__(
        self,
        slave: Domain | CollisionShape | Any | None = None,
        master: CollisionShape | Sequence[CollisionShape] | Domain | Any | None = None,
        *,
        obstacle: CollisionShape | Sequence[CollisionShape] | Domain | Any | None = None,
        domain: Domain | None = None,
        body: int | str | Domain | None = None,
        slave_nodes: Sequence[int] | np.ndarray | None = None,
        k_penalty: float = 1e4,
        c_damping: float = 0.0,
        friction: float = 0.0,
        dof_offset: int = 0,
        dofs_per_node: int = 3,
        formulation: Literal["penalty", "augmented_lagrangian"] = "penalty",
        tolerance: float = 0.0,
    ) -> None:
        """
        Parameters:
        -----------
        slave : Domain, VolumetricMesh, or CollisionShape, optional
            Deformable domain undergoing contact (or legacy first positional obstacle).
        master : CollisionShape, Domain, VolumetricMesh, or Sequence[CollisionShape], optional
            Target obstacle or interacting body domain.
        obstacle : CollisionShape, Domain, VolumetricMesh, or Sequence[CollisionShape], optional
            Legacy alias for master obstacle.
        domain : Domain, optional
            Deformable domain or parent MultiBodySystem undergoing contact.
        body : int, str, or Domain, optional
            Specific constituent body within MultiBodySystem (when domain is MultiBodySystem).
        slave_nodes : Sequence[int] or np.ndarray, optional
            Candidate contact node indices on the slave body (defaults to surface nodes).
        k_penalty : float, default=1e4
            Normal penalty contact stiffness (N/mm).
        c_damping : float, default=0.0
            Contact velocity damping coefficient (N*s/mm).
        friction : float, default=0.0
            Tangential friction coefficient.
        dof_offset : int, default=0
            Global DOF offset when used within a MultiBodySystem assembly.
        dofs_per_node : int, default=3
            Degrees of freedom per node (3 for solid continuum, 6 for shell).
        formulation : {'penalty', 'augmented_lagrangian'}, default='penalty'
            Contact constraint enforcement scheme.
        tolerance : float, default=0.0
            Penetration activation threshold (gap <= tolerance).
        """
        # Resolve master / obstacle
        if master is None and obstacle is not None:
            master = obstacle
        # Resolve slave / domain
        if slave is None and domain is not None:
            slave = domain

        # Handle positional legacy RigidContact(obstacle, domain)
        if (
            isinstance(slave, (CollisionShape, RayMeshObstacle))
            and obstacle is None
            and not isinstance(master, (CollisionShape, RayMeshObstacle))
        ):
            slave, master = master, slave

        if master is None:
            raise ValueError("RigidContact requires a master obstacle or interacting body.")

        # Resolve target body within MultiBodySystem if slave is MultiBodySystem
        self._target_body: Domain | None = None
        if slave is not None and hasattr(slave, "bodies") and body is not None:
            b_idx = (
                slave.get_body_index(body)
                if hasattr(slave, "get_body_index")
                else (
                    body
                    if isinstance(body, int)
                    else slave.names.index(body)
                    if isinstance(body, str)
                    else slave.bodies.index(body)
                )
            )
            self._target_body = slave.bodies[b_idx]
            self.domain = slave
            dof_offset = int(slave.get_body_dof_slice(b_idx).start)
            dofs_per_node = getattr(self._target_body, "dofs_per_node", dofs_per_node)
        elif slave is not None:
            self._target_body = slave
            self.domain = domain if domain is not None else slave
        else:
            self.domain = domain

        self.slave = self._target_body
        self.master = master
        self.obstacle = master

        # Automatically wrap domain or mesh obstacles into RayMeshObstacle
        if (
            hasattr(master, "surface")
            or (hasattr(master, "mesh") and hasattr(master.mesh, "surface"))
            or hasattr(master, "ugrid")
        ):
            master_obstacle = RayMeshObstacle(master)
        else:
            master_obstacle = master

        self.detector = (
            master_obstacle
            if isinstance(master_obstacle, CollisionDetector)
            else CollisionDetector(master_obstacle)
        )

        if not np.isfinite(k_penalty) or k_penalty <= 0.0:
            raise ValueError("k_penalty must be finite and positive")
        self.k_penalty = float(k_penalty)
        self.c_damping = float(c_damping)
        self.friction = float(friction)
        self.dof_offset = int(dof_offset)
        self.dofs_per_node = int(dofs_per_node)
        self.formulation = formulation
        self.tolerance = float(tolerance)

        # Determine candidate slave nodes
        if slave_nodes is not None:
            self.slave_nodes = np.asarray(slave_nodes, dtype=np.int64)
            self.multipliers = np.zeros(len(self.slave_nodes), dtype=np.float64)
        elif self._target_body is not None:
            if (
                hasattr(self._target_body, "mesh")
                and hasattr(self._target_body.mesh, "surface")
                and self._target_body.mesh.surface.node_indices is not None
            ):
                self.slave_nodes = np.asarray(
                    self._target_body.mesh.surface.node_indices, dtype=np.int64
                )
            elif hasattr(self._target_body, "mesh"):
                self.slave_nodes = np.arange(self._target_body.mesh.n_nodes, dtype=np.int64)
            else:
                self.slave_nodes = np.arange(
                    self._target_body.n_dofs // self.dofs_per_node, dtype=np.int64
                )
            self.multipliers = np.zeros(len(self.slave_nodes), dtype=np.float64)
        elif self.domain is not None:
            if (
                hasattr(self.domain, "mesh")
                and hasattr(self.domain.mesh, "surface")
                and self.domain.mesh.surface.node_indices is not None
            ):
                self.slave_nodes = np.asarray(
                    self.domain.mesh.surface.node_indices, dtype=np.int64
                )
            elif hasattr(self.domain, "mesh"):
                self.slave_nodes = np.arange(self.domain.mesh.n_nodes, dtype=np.int64)
            else:
                self.slave_nodes = np.arange(
                    self.domain.n_dofs // self.dofs_per_node, dtype=np.int64
                )
            self.multipliers = np.zeros(len(self.slave_nodes), dtype=np.float64)
        else:
            self.slave_nodes = None
            self.multipliers = np.zeros(0, dtype=np.float64)

    # =========================================================================
    # Contact Evaluation
    # =========================================================================

    def _resolve_nodes(
        self, current_nodes: np.ndarray | None = None
    ) -> np.ndarray:
        """Resolves current 3D nodal coordinates from input or target domain."""
        if current_nodes is not None:
            c_nodes = np.asarray(current_nodes, dtype=np.float64)
            if self._target_body is not None:
                n_target = len(self._target_body.current_nodes)
                if len(c_nodes) == n_target:
                    return c_nodes
                # Multi-body stacked coordinates: extract target body slice
                if hasattr(self.domain, "bodies"):
                    node_start = 0
                    for b in self.domain.bodies:
                        if b is self._target_body:
                            break
                        node_start += len(b.current_nodes)
                    return c_nodes[node_start : node_start + n_target]
            return c_nodes
        if self._target_body is not None and hasattr(self._target_body, "current_nodes"):
            return self._target_body.current_nodes
        if self.domain is not None and hasattr(self.domain, "current_nodes"):
            return self.domain.current_nodes
        raise ValueError(
            "RigidContact requires current_nodes or a domain with current_nodes property."
        )

    def _resolve_slave_nodes(self, all_nodes: np.ndarray) -> np.ndarray:
        if self.slave_nodes is not None:
            return self.slave_nodes
        s_nodes = np.arange(len(all_nodes), dtype=np.int64)
        if len(self.multipliers) != len(s_nodes):
            self.multipliers = np.zeros(len(s_nodes), dtype=np.float64)
        return s_nodes

    def query_contact(
        self,
        current_nodes: np.ndarray | None = None,
        tolerance: float = 0.0,
    ) -> ContactQueryResult:
        """Queries the collision detector for all candidate slave nodes."""
        nodes = self._resolve_nodes(current_nodes)
        s_nodes = self._resolve_slave_nodes(nodes)
        slave_pts = nodes[s_nodes]
        return self.detector.query(slave_pts, tolerance=tolerance)

    def evaluate_gaps(self, current_nodes: np.ndarray | None = None) -> np.ndarray:
        """Return all candidate signed gaps [length], including opening nodes."""
        return self.query_contact(current_nodes).gaps.copy()

    def _active_indices(self, query: ContactQueryResult) -> np.ndarray:
        """AL activity uses projected pressure, including stored opening force."""
        if self.formulation == "augmented_lagrangian":
            return np.flatnonzero(self.multipliers - self.k_penalty * query.gaps > 0.0)
        return query.active_indices

    def evaluate_forces(
        self,
        current_nodes: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
        n_dofs: int | None = None,
    ) -> np.ndarray:
        """
        Evaluates the global contact force vector f_c.

        Parameters:
        -----------
        current_nodes : np.ndarray of shape (N_nodes, 3), optional
            Current deformed spatial positions of all domain nodes.
        velocities : np.ndarray, optional
            Current velocity vector of the domain (n_dofs,) or nodal velocities (N_nodes, 3).
        n_dofs : int, optional
            Total number of DOFs in the global system (defaults to domain.n_dofs).

        Returns:
        --------
        f_c : np.ndarray of shape (n_dofs,)
            Global contact repulsive force vector.
        """
        nodes = self._resolve_nodes(current_nodes)
        s_nodes = self._resolve_slave_nodes(nodes)

        if n_dofs is None:
            if self.domain is not None:
                n_dofs = self.domain.n_dofs
            else:
                n_dofs = len(nodes) * self.dofs_per_node + self.dof_offset

        f_c = np.zeros(n_dofs, dtype=np.float64)

        # Query proximity
        query_res = self.detector.query(nodes[s_nodes], tolerance=self.tolerance)
        active_idx = self._active_indices(query_res)
        if len(active_idx) == 0:
            return f_c

        active_nodes = s_nodes[active_idx]
        gaps = query_res.gaps[active_idx]  # strictly <= 0
        normals = query_res.normals[active_idx]

        if self.formulation == "augmented_lagrangian" and len(self.multipliers) >= len(s_nodes):
            lams = self.multipliers[active_idx]
            f_N = np.maximum(0.0, lams - self.k_penalty * gaps)
            spring_forces = f_N[:, None] * normals
        else:
            # Elastic penalty force: f_c = -k_p * g * n = k_p * |g| * n
            # Since g <= 0, -k_p * g is strictly non-negative
            spring_forces = -self.k_penalty * gaps[:, None] * normals  # (M, 3)

        # Contact damping force: f_d = -c_p * (v . n) * n (when v . n < 0)
        if self.c_damping > 0.0 and velocities is not None:
            vel = np.asarray(velocities, dtype=np.float64)
            if vel.ndim == 1:
                # Extract translational velocities for active nodes
                v_nodes = np.zeros((len(active_nodes), 3), dtype=np.float64)
                for i, node_id in enumerate(active_nodes):
                    dof_start = self.dof_offset + node_id * self.dofs_per_node
                    v_nodes[i] = vel[dof_start : dof_start + 3]
            else:
                v_nodes = vel[active_nodes, :3]

            vn = np.einsum("ij,ij->i", v_nodes, normals)  # normal velocity
            # Damping only resists penetration (vn < 0)
            penetrating_v = np.minimum(vn, 0.0)
            damp_forces = -self.c_damping * penetrating_v[:, None] * normals
            spring_forces += damp_forces

        # Scatter into global contact force vector
        for i, node_id in enumerate(active_nodes):
            dof_start = self.dof_offset + node_id * self.dofs_per_node
            f_c[dof_start : dof_start + 3] += spring_forces[i]

        return f_c

    def evaluate_stiffness(
        self,
        current_nodes: np.ndarray | None = None,
        n_dofs: int | None = None,
        a1: float = 0.0,
    ) -> spsp.csr_matrix:
        """
        Assembles the sparse tangent contact stiffness matrix K_c.
        K_c = sum_{active} (k_p + a_1 * c_p) * (n (x) n)

        Parameters:
        -----------
        current_nodes : np.ndarray, optional
            Deformed spatial coordinates of nodes.
        n_dofs : int, optional
            Global system DOF count.
        a1 : float, default=0.0
            Dynamic integration parameter gamma / (beta * dt). When c_damping > 0,
            incorporates the retained rank-one normal damping approximation.
        """
        nodes = self._resolve_nodes(current_nodes)
        s_nodes = self._resolve_slave_nodes(nodes)

        if n_dofs is None:
            if self.domain is not None:
                n_dofs = self.domain.n_dofs
            else:
                n_dofs = len(nodes) * self.dofs_per_node + self.dof_offset

        query_res = self.detector.query(nodes[s_nodes], tolerance=0.0)
        active_idx = self._active_indices(query_res)
        if len(active_idx) == 0:
            return spsp.csr_matrix((n_dofs, n_dofs), dtype=np.float64)

        active_nodes = s_nodes[active_idx]
        normals = query_res.normals[active_idx]

        num_active = len(active_nodes)
        # Each active node adds a 3x3 symmetric block (9 entries)
        rows = np.zeros(num_active * 9, dtype=np.int64)
        cols = np.zeros(num_active * 9, dtype=np.int64)
        vals = np.zeros(num_active * 9, dtype=np.float64)

        entry_idx = 0
        kp = self.k_penalty + a1 * self.c_damping

        for i, node_id in enumerate(active_nodes):
            dof_start = self.dof_offset + node_id * self.dofs_per_node
            n = normals[i]
            # Outer product n (x) n
            nn = np.outer(n, n)

            for a in range(3):
                for b in range(3):
                    rows[entry_idx] = dof_start + a
                    cols[entry_idx] = dof_start + b
                    vals[entry_idx] = kp * nn[a, b]
                    entry_idx += 1

        K_c = spsp.coo_matrix(
            (vals, (rows, cols)), shape=(n_dofs, n_dofs), dtype=np.float64
        ).tocsr()

        return K_c

    def evaluate_potential_energy(
        self, current_nodes: np.ndarray | None = None
    ) -> float:
        """
        Return normal potential [force*length]. For AL at fixed stored lambda,
        Pi_c=sum((max(0,lambda-kp*g)^2-lambda^2)/(2*kp)); penalty lambda=0
        gives 1/2*kp*min(g,0)^2. Curved normal tangent limitations remain.
        """
        nodes = self._resolve_nodes(current_nodes)
        s_nodes = self._resolve_slave_nodes(nodes)
        query_res = self.detector.query(nodes[s_nodes], tolerance=0.0)
        if self.formulation == "augmented_lagrangian":
            pressure = np.maximum(0.0, self.multipliers - self.k_penalty * query_res.gaps)
            return float(np.sum((pressure**2 - self.multipliers**2) / (2.0 * self.k_penalty)))
        if query_res.num_active == 0:
            return 0.0

        gaps = query_res.gaps[query_res.active_indices]
        return float(0.5 * self.k_penalty * np.sum(gaps**2))

    def get_contact_info(
        self, current_nodes: np.ndarray | None = None
    ) -> dict[str, float | int | np.ndarray]:
        """Returns structured summary of current contact status."""
        nodes = self._resolve_nodes(current_nodes)
        s_nodes = self._resolve_slave_nodes(nodes)
        query_res = self.detector.query(nodes[s_nodes], tolerance=self.tolerance)

        active_idx = self._active_indices(query_res)
        num_active = len(active_idx)
        max_pen = float(np.max(np.maximum(-query_res.gaps, 0.0), initial=0.0))
        active_nodes = s_nodes[active_idx]
        energy = self.evaluate_potential_energy(current_nodes)

        return {
            "num_active": num_active,
            "max_penetration": max_pen,
            "active_nodes": active_nodes,
            "potential_energy": energy,
            "max_multiplier": float(np.max(self.multipliers)) if len(self.multipliers) > 0 else 0.0,
        }

@dataclass(frozen=True)
class _DeformableContactQuery:
    """One slave node and its exact closest master feature in current geometry."""

    slave_index: int
    slave_node: int
    master_nodes: np.ndarray
    geometry: TriangleGap


class DeformableContact(ContactMultiplierState):
    """Conservative two-body node-to-triangle normal contact.

    For local coordinates q=[x_s,x_m1,x_m2,x_m3], let g be oriented signed
    closest-feature distance [length], b=grad(g), H=Hess(g), and
    p=max(0,lambda-kp*g) [force]. Normal forces and the RHS force tangent are
    f_c=p*b and K_c=-df_c/dq=kp*b*b.T-p*H. Differentiating feature projection
    weights and face normals yields a symmetric, possibly indefinite tangent.
    The fixed-multiplier potential is (p*p-lambda*lambda)/(2*kp).

    Triangles remain piecewise planar: face interiors use oriented face normals;
    edge/vertex pseudonormals determine only the signed Euclidean distance sign.
    Exact derivatives hold on a fixed smooth feature/sign branch. At zero
    edge/vertex distance, use the selected incident face's finite one-sided
    derivatives. Feature/facet/sign switches and activation can be nonsmooth.
    Global inside/outside requires consistently oriented closed manifold masters.

    The bound slave/master domains are the geometry authority for every method.
    The optional current_nodes argument is retained for the common contact API;
    callers must update the domains to evaluate alternative geometry. No private
    geometry or velocity checksum cache is used. Query geometry is rebuilt per
    call, with VTK proximity search and sparse COO matrix assembly.

    Existing damping/friction forces remain available. Their tangent retains
    only the normal velocity rank-one approximation; friction and dissipative
    geometry derivatives are omitted. Exact energy/force/tangent claims concern
    c_damping=friction=0. See
    docs/05_system_assembly_boundaries_and_domain_coupling.md and Wriggers,
    Computational Contact Mechanics, chs. 2 and 6.
    """

    def __init__(
        self,
        slave: Domain,
        master: Domain,
        slave_nodes: Sequence[int] | np.ndarray | None = None,
        k_penalty: float = 1e4,
        c_damping: float = 0.0,
        friction: float = 0.0,
        slave_offset: int = 0,
        master_offset: int = 0,
        domain: Domain | None = None,
        formulation: Literal["penalty", "augmented_lagrangian"] = "penalty",
        tolerance: float = 0.0,
        *,
        smoothed_normals: bool = False,
        smoothing_weight: Literal["angle", "area"] = "angle",
    ) -> None:
        """Bind body geometry and interface parameters.

        kp [force/length] must be positive. Damping [force*time/length] and
        Coulomb friction coefficient are nonnegative. Body DOF offsets use
        three translational DOFs per node. Tolerance [length] is retained
        interface metadata; signed gaps and force activation use physical zero.

        smoothed_normals : bool, default=False
            When True (Pillar 2: Spatial Regularization), contact normals are
            interpolated continuously across master facet surfaces using area/angle-weighted
            vertex pseudonormals: n(xi) = (sum xi_i * n_v_i) / ||sum xi_i * n_v_i||.
            Eliminates directional jumps Delta n = n_B - n_A as slave nodes cross facet edges.
        smoothing_weight : {'angle', 'area'}, default='angle'
            Weighting formulation for assembling vertex pseudonormals from incident facet normals.
        """
        if not np.isfinite(k_penalty) or k_penalty <= 0.0:
            raise ValueError("k_penalty must be finite and positive")
        if not np.isfinite(c_damping) or c_damping < 0.0:
            raise ValueError("c_damping must be finite and nonnegative")
        if not np.isfinite(friction) or friction < 0.0:
            raise ValueError("friction must be finite and nonnegative")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")
        if formulation not in ("penalty", "augmented_lagrangian"):
            raise ValueError("formulation must be 'penalty' or 'augmented_lagrangian'")
        if smoothing_weight not in ("angle", "area"):
            raise ValueError("smoothing_weight must be 'angle' or 'area'")
        self.slave = slave
        self.master = master
        self.k_penalty = float(k_penalty)
        self.c_damping = float(c_damping)
        self.friction = float(friction)
        self.slave_offset = int(slave_offset)
        self.master_offset = int(master_offset)
        self.domain = domain
        self.formulation = formulation
        self.tolerance = float(tolerance)
        self.smoothed_normals = bool(smoothed_normals)
        self.smoothing_weight = str(smoothing_weight)

        if slave_nodes is not None:
            self.slave_nodes = np.asarray(slave_nodes, dtype=np.int64)
        elif (
            hasattr(self.slave, "mesh")
            and hasattr(self.slave.mesh, "surface")
            and self.slave.mesh.surface.node_indices is not None
        ):
            self.slave_nodes = np.asarray(
                self.slave.mesh.surface.node_indices, dtype=np.int64
            )
        elif hasattr(self.slave, "mesh"):
            self.slave_nodes = np.arange(self.slave.mesh.n_nodes, dtype=np.int64)
        else:
            self.slave_nodes = np.arange(self.slave.n_dofs // 3, dtype=np.int64)

        if not hasattr(self.master, "mesh") or not hasattr(self.master.mesh, "surface"):
            raise TypeError("Master domain must possess a surface mesh representation.")
        surface = self.master.mesh.surface
        self._master_polydata = surface.polydata.copy(deep=True)
        self._master_surf_node_idx = np.asarray(surface.node_indices, dtype=np.int64)
        if not self._master_polydata.is_all_triangles:
            self._master_polydata = self._master_polydata.triangulate()
        self._master_faces = self._master_polydata.regular_faces
        if len(self._master_faces) == 0:
            raise ValueError("Master surface must contain at least one triangle")
        self._edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
        for face_index, face in enumerate(self._master_faces):
            for i, j in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(sorted((int(face[i]), int(face[j]))))
                self._edge_faces[edge].append(face_index)
        self._locator = vtk.vtkStaticCellLocator()
        self._locator.SetDataSet(self._master_polydata)
        self.multipliers = np.zeros(len(self.slave_nodes), dtype=np.float64)
        self._update_master_surface()

    @staticmethod
    def compute_barycentric(
        p: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray
    ) -> tuple[float, float, float]:
        """Return exact closest-feature barycentrics for point p [length].

        Face, edge, and vertex projection use Euclidean closest distance;
        outside-point coordinates are not independently clipped/renormalized.
        """
        weights = TriangleContactGeometry.closest_barycentric(p, np.array([v1, v2, v3]))
        return float(weights[0]), float(weights[1]), float(weights[2])

    def _update_master_surface(self) -> None:
        """Rebuild proximity geometry and outward feature sign pseudonormals.

        Vertex pseudonormals sum corner-angle-weighted unit face normals.
        Edge pseudonormals sum incident unit face normals. These vectors choose
        only distance signs; force directions come from exact gap gradients.
        """
        nodes = self.master.current_nodes[self._master_surf_node_idx]
        self._master_polydata.points = nodes
        self._face_normals = np.empty((len(self._master_faces), 3))
        self._vertex_pseudonormals = np.zeros_like(nodes)
        for face_index, face in enumerate(self._master_faces):
            vertices = nodes[face]
            normal, area = TriangleContactGeometry._face_normal(vertices)
            self._face_normals[face_index] = normal
            for i in range(3):
                if self.smoothing_weight == "area":
                    weight = area / 3.0
                else:
                    e1 = vertices[(i + 1) % 3] - vertices[i]
                    e2 = vertices[(i + 2) % 3] - vertices[i]
                    weight = np.arctan2(np.linalg.norm(np.cross(e1, e2)), e1 @ e2)
                self._vertex_pseudonormals[face[i]] += weight * normal
        norms = np.linalg.norm(self._vertex_pseudonormals, axis=1, keepdims=True)
        self._unit_vertex_pseudonormals = np.where(
            norms > 1e-14,
            self._vertex_pseudonormals / np.maximum(norms, 1e-14),
            np.array([0.0, 0.0, 1.0]),
        )
        self._locator.BuildLocator()

    def _queries(self) -> list[_DeformableContactQuery]:
        """Query all slave nodes using one consistent closest-feature definition."""
        self._update_master_surface()
        slave_positions = self.slave.current_nodes
        master_positions = self.master.current_nodes
        closest = np.zeros(3, dtype=np.float64)
        cell_id, sub_id, distance_squared = vtk.mutable(0), vtk.mutable(0), vtk.mutable(0.0)
        queries: list[_DeformableContactQuery] = []
        for index, node in enumerate(self.slave_nodes):
            point = slave_positions[node]
            self._locator.FindClosestPoint(point, closest, cell_id, sub_id, distance_squared)
            face_index = int(cell_id)
            face = self._master_faces[face_index]
            master_nodes = self._master_surf_node_idx[face]
            vertices = master_positions[master_nodes]
            weights = TriangleContactGeometry.closest_barycentric(point, vertices)
            active = np.flatnonzero(weights > 0.0)
            if len(active) == 3:
                pseudonormal = self._face_normals[face_index]
            elif len(active) == 2:
                edge = tuple(sorted((int(face[active[0]]), int(face[active[1]]))))
                pseudonormal = self._face_normals[self._edge_faces[edge]].sum(axis=0)
            else:
                pseudonormal = self._vertex_pseudonormals[face[active[0]]]
            if np.linalg.norm(pseudonormal) <= TriangleContactGeometry._relative_tolerance:
                raise ValueError("Undefined master feature pseudonormal: check surface orientation")
            v_normals = self._unit_vertex_pseudonormals[face] if self.smoothed_normals else None
            geometry = TriangleContactGeometry.evaluate(
                point, vertices, pseudonormal, vertex_normals=v_normals
            )
            queries.append(_DeformableContactQuery(index, int(node), master_nodes, geometry))
        return queries

    def _resolve_n_dofs(self, n_dofs: int | None) -> int:
        """Resolve global translational vector size from body offsets or assembly."""
        if n_dofs is not None:
            return n_dofs
        if self.domain is not None:
            return self.domain.n_dofs
        return max(
            self.slave_offset + self.slave.n_dofs,
            self.master_offset + self.master.n_dofs,
        )

    def evaluate_gaps(self, current_nodes: np.ndarray | None = None) -> np.ndarray:
        """Return candidate-aligned exact signed gaps from bound body geometry."""
        return np.array([query.geometry.gap for query in self._queries()])

    def _dof_indices(self, query: _DeformableContactQuery) -> np.ndarray:
        """Map local (slave, master triangle) coordinates into global DOFs."""
        return np.r_[
            self.slave_offset + 3 * query.slave_node + np.arange(3),
            (self.master_offset + 3 * query.master_nodes[:, None] + np.arange(3)).ravel(),
        ]

    def _pressure(self, query: _DeformableContactQuery) -> tuple[float, float]:
        """Return active p=max(0,lambda-kp*g) and fixed stored lambda [force]."""
        multiplier = (
            float(self.multipliers[query.slave_index])
            if self.formulation == "augmented_lagrangian" else 0.0
        )
        pressure = max(0.0, multiplier - self.k_penalty * query.geometry.gap)
        return pressure, multiplier

    def evaluate_forces(
        self,
        current_nodes: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
        n_dofs: int | None = None,
    ) -> np.ndarray:
        """Assemble f_c=p*grad(g) [force] using bound domain geometry.

        Equal-and-opposite frictionless force and moment follow from translation
        and rotation invariance of signed distance. Optional normal damping and
        regularized Coulomb friction preserve the existing dissipative laws;
        their full derivatives are outside this conservative validation stage.
        """
        forces = np.zeros(self._resolve_n_dofs(n_dofs), dtype=np.float64)
        velocity = (
            None if velocities is None
            else np.asarray(velocities, dtype=np.float64).reshape(-1)
        )
        for query in self._queries():
            pressure, _ = self._pressure(query)
            if pressure <= 0.0:
                continue
            geometry = query.geometry
            indices = self._dof_indices(query)
            local_force = pressure * geometry.gradient
            if velocity is not None:
                weights = np.r_[1.0, -geometry.barycentric]
                relative_velocity = weights @ velocity[indices].reshape(4, 3)
                normal_velocity = float(relative_velocity @ geometry.normal)
                if self.c_damping > 0.0 and normal_velocity < 0.0:
                    local_force += -self.c_damping * normal_velocity * geometry.gradient
                tangential_velocity = relative_velocity - normal_velocity * geometry.normal
                speed = float(np.linalg.norm(tangential_velocity))
                if self.friction > 0.0 and speed > 1e-9:
                    friction_force = (
                        -self.friction * pressure * np.tanh(speed / 1e-3)
                        * tangential_velocity / speed
                    )
                    local_force += np.kron(weights, friction_force)
            np.add.at(forces, indices, local_force)
        return forces

    def evaluate_stiffness(
        self,
        current_nodes: np.ndarray | None = None,
        n_dofs: int | None = None,
        a1: float = 0.0,
    ) -> spsp.csr_matrix:
        """Assemble K_c=kp*b*b.T-p*H=-df_c/dq [force/length].

        Sparse COO scatter retains all cross-body and geometric terms; the
        conservative tangent is symmetric but may be indefinite. With damping,
        add the retained approximate a1*c*b*b.T term (a1 [1/time]); damping
        geometry and friction derivatives are not included.
        """
        size = self._resolve_n_dofs(n_dofs)
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for query in self._queries():
            pressure, _ = self._pressure(query)
            if pressure <= 0.0:
                continue
            geometry = query.geometry
            gradient = geometry.gradient
            tangent = (
                (self.k_penalty + a1 * self.c_damping) * np.outer(gradient, gradient)
                - pressure * geometry.hessian
            )
            indices = self._dof_indices(query)
            rows.append(np.repeat(indices, 12))
            columns.append(np.tile(indices, 12))
            values.append(tangent.ravel())
        if not values:
            return spsp.csr_matrix((size, size), dtype=np.float64)
        return spsp.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(size, size),
        ).tocsr()


    def evaluate_potential_energy(
        self, current_nodes: np.ndarray | None = None
    ) -> float:
        """Return sum((p^2-lambda^2)/(2*kp)) [force*length] at fixed lambda.

        Geometry comes from bound domains. Dissipative forces have no stored
        potential and are excluded. Penalty energy is 0.5*kp*min(g,0)^2.
        """
        return float(self.get_contact_info(current_nodes)["potential_energy"])

    def get_contact_info(
        self, current_nodes: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Report active nodes, penetration [length], force multiplier, and energy.

        All diagnostics use the same domain-state geometry, gap, and pressure as
        force/tangent evaluation. Potential is the fixed-multiplier normal
        potential, including its constant inactive-branch offset.
        """
        active_nodes: list[int] = []
        energy = 0.0
        max_penetration = 0.0
        for query in self._queries():
            pressure, multiplier = self._pressure(query)
            energy += (pressure**2 - multiplier**2) / (2.0 * self.k_penalty)
            if pressure > 0.0 or query.geometry.gap < 0.0:
                active_nodes.append(query.slave_node)
                max_penetration = max(max_penetration, -query.geometry.gap)
        return {
            "num_active": len(active_nodes),
            "max_penetration": max_penetration,
            "active_nodes": np.asarray(active_nodes, dtype=np.int64),
            "potential_energy": energy,
            "max_multiplier": float(np.max(self.multipliers)) if self.multipliers.size else 0.0,
        }


@dataclass(frozen=True)
class _SelfContactQuery:
    """One admitted slave/triangle constraint with exact local gap derivatives."""

    slave_index: int
    slave_node: int
    master_nodes: np.ndarray
    geometry: TriangleGap


class SelfContact(ContactMultiplierState):
    """Conservative node-to-triangle mechanics on selected same-body candidates.

    For local q=[x_s,x_m1,x_m2,x_m3], use signed closest-feature distance g
    [length], b=grad(g), H=Hess(g), and p=max(0,lambda-kp*g) [force]. The
    normal force is f=p*b and its RHS tangent K=-df/dq=kp*b*b.T-p*H. Potential
    Pi=(p*p-lambda*lambda)/(2*kp) [force*length] matches the normal force at
    fixed lambda. Penalty uses lambda=0. The geometric tangent is symmetric
    on a smooth selected branch and may be indefinite. COO assembly scatters
    all 12 local DOFs into the same body with its global offset.

    ``TriangleContactGeometry`` supplies exact face/edge/vertex projection and
    analytical derivatives. Edge sign uses incident unit-face pseudonormals;
    vertex sign uses angle-weighted unit-face pseudonormals. These choose the
    sign only, while force directions come from the distance gradient. The
    shared finite selected-face convention applies at zero edge/vertex distance.
    Differentiability is not promised across feature/facet/sign/activation or
    candidate-filter switches. Exact conservative energy/derivative/moment
    claims concern damping=friction=0; dissipative tangents remain approximate.

    Detection retains BFS exclusions, area-weighted slave-normal opposition,
    axis-aligned proximity search, nearest admitted facet, and a distance cutoff.
    This is a local candidate policy, not a global self-intersection detector:
    an undeformed convex body's remote opposite wall can have negative local
    gap. Excluding own nearby faces invalidates global closest-surface signing.
    No CCD, edge-edge completeness, deep-penetration guarantee, persistent pair
    history or area-integrated pressure is provided. Multipliers remain per
    slave node, not per persistent node/facet pair. No trial-history cache,
    reference pair masks or example-specific detection rules are introduced.
    See docs/05_system_assembly_boundaries_and_domain_coupling.md; Wriggers,
    Computational Contact Mechanics, chs. 2 and 6.
    """

    def __init__(
        self,
        domain: Domain,
        contact_nodes: Sequence[int] | np.ndarray | None = None,
        *,
        k_penalty: float = 1e4,
        c_damping: float = 0.0,
        friction: float = 0.0,
        dof_offset: int = 0,
        formulation: Literal["penalty", "augmented_lagrangian"] = "penalty",
        tolerance: float = 0.0,
        search_radius: float | None = None,
        contact_thickness: float | None = None,
        normal_threshold: float = -0.2,
        exclusion_rings: int = 4,
        smoothed_normals: bool = False,
        smoothing_weight: Literal["angle", "area"] = "angle",
    ) -> None:
        """Bind surface candidates, material-independent contact law and search.

        kp [force/length] is nodal stiffness; damping [force*time/length] and
        friction are nonnegative. Offsets are nonnegative multiples of three.
        Candidate indices must be unique surface node indices. Rings (default4)
        count surface edge hops; even zero rings excludes incident facets.
        normal_threshold [-1,1] limits the slave/face normal dot product.

        search_radius [length] bounds the axis-aligned search (default
        max(5,0.2*reference bounding-box diagonal)). contact_thickness [length]
        is the maximum closest distance (default max(1,0.5*mean surface edge)),
        not an offset in g. These existing dimensional defaults and filters
        are retained, not validated as a universal collision detector.
        tolerance [length] is retained interface metadata; pressure activation
        uses physical zero. No AL or penalty stiffness is adjusted internally.

        smoothed_normals : bool, default=False
            When True (Pillar 2: Spatial Regularization), contact normals are
            interpolated continuously across contact facet surfaces using area/angle-weighted
            vertex pseudonormals: n(xi) = (sum xi_i * n_v_i) / ||sum xi_i * n_v_i||.
            Eliminates directional jumps Delta n = n_B - n_A as slave nodes cross facet edges.
        smoothing_weight : {'angle', 'area'}, default='angle'
            Weighting formulation for assembling vertex pseudonormals from incident facet normals.
        """
        for name, value in (
            ("k_penalty", k_penalty), ("search_radius", search_radius),
            ("contact_thickness", contact_thickness),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("c_damping", c_damping), ("friction", friction),
            ("tolerance", tolerance),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (("dof_offset", dof_offset), ("exclusion_rings", exclusion_rings)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
        if dof_offset % 3:
            raise ValueError("dof_offset must be a multiple of three")
        if not np.isfinite(normal_threshold) or not -1.0 <= normal_threshold <= 1.0:
            raise ValueError("normal_threshold must be finite and in [-1, 1]")
        if formulation not in ("penalty", "augmented_lagrangian"):
            raise ValueError("formulation must be 'penalty' or 'augmented_lagrangian'")
        if smoothing_weight not in ("angle", "area"):
            raise ValueError("smoothing_weight must be 'angle' or 'area'")

        self.domain = domain
        self.k_penalty = float(k_penalty)
        self.c_damping = float(c_damping)
        self.friction = float(friction)
        self.dof_offset = int(dof_offset)
        self.formulation = formulation
        self.tolerance = float(tolerance)
        self.normal_threshold = float(normal_threshold)
        self.exclusion_rings = int(exclusion_rings)
        self.smoothed_normals = bool(smoothed_normals)
        self.smoothing_weight = str(smoothing_weight)
        if not hasattr(domain, "mesh") or not hasattr(domain.mesh, "surface"):
            raise TypeError("Domain must possess a surface mesh representation for SelfContact")
        surface = domain.mesh.surface
        self._polydata = surface.polydata.copy(deep=True)
        self._surf_node_idx = np.asarray(surface.node_indices, dtype=np.int64)
        if not self._polydata.is_all_triangles:
            self._polydata = self._polydata.triangulate()
        self._faces = self._polydata.regular_faces
        self.n_surf_nodes = len(self._surf_node_idx)
        self.n_faces = len(self._faces)
        if self.n_faces == 0:
            raise ValueError("SelfContact surface must contain at least one triangle")
        self._vol_to_surf_idx = {
            int(node): index for index, node in enumerate(self._surf_node_idx)
        }
        raw_candidates = (
            self._surf_node_idx if contact_nodes is None else np.asarray(contact_nodes)
        )
        if raw_candidates.ndim != 1 or (
            raw_candidates.size and not np.issubdtype(raw_candidates.dtype, np.integer)
        ):
            raise ValueError("contact_nodes must be a one-dimensional integer vector")
        self.contact_nodes = np.asarray(raw_candidates, dtype=np.int64).copy()
        if len(np.unique(self.contact_nodes)) != len(self.contact_nodes):
            raise ValueError("contact_nodes must be unique")
        if not np.all(np.isin(self.contact_nodes, self._surf_node_idx)):
            raise ValueError("contact_nodes must be surface node indices")
        self.multipliers = np.zeros(len(self.contact_nodes), dtype=np.float64)

        self._edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
        for face_index, face in enumerate(self._faces):
            for i, j in ((0, 1), (1, 2), (2, 0)):
                self._edge_faces[tuple(sorted((int(face[i]), int(face[j]))))].append(face_index)
        self._excluded_faces = self._build_exclusion_graph(self.exclusion_rings)
        self._locator = vtk.vtkStaticCellLocator()
        self._locator.SetDataSet(self._polydata)
        self._update_surface()

        points = domain.mesh.nodes[self._surf_node_idx]
        diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
        self.search_radius = (
            max(5.0, 0.20 * diagonal) if search_radius is None else float(search_radius)
        )
        vertices = points[self._faces]
        self.mean_edge_length = float(np.mean([
            np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1),
            np.linalg.norm(vertices[:, 2] - vertices[:, 1], axis=1),
            np.linalg.norm(vertices[:, 0] - vertices[:, 2], axis=1),
        ]))
        self.contact_thickness = (
            max(1.0, 0.5 * self.mean_edge_length)
            if contact_thickness is None else float(contact_thickness)
        )

    def _build_exclusion_graph(self, rings: int) -> dict[int, set[int]]:
        """Exclude incident facets of nodes within the given surface BFS depth.

        This retains mesh-topology policy, not a proof that every remaining
        facet is physically eligible for self-contact. Depth zero excludes only
        incident facets; greater depth can also suppress legitimate close folds.
        """
        incident: dict[int, set[int]] = defaultdict(set)
        adjacency: dict[int, set[int]] = defaultdict(set)
        for face_index, face in enumerate(self._faces):
            for node in face:
                incident[int(node)].add(face_index)
                adjacency[int(node)].update(int(other) for other in face if other != node)
        excluded: dict[int, set[int]] = {}
        for node in range(self.n_surf_nodes):
            frontier, visited = {node}, {node}
            for _ in range(rings):
                neighbors: set[int] = set()
                for current in frontier:
                    neighbors.update(adjacency[current])
                frontier = neighbors - visited
                visited.update(frontier)
                if not frontier:
                    break
            excluded[node] = set().union(*(incident[current] for current in visited))
        return excluded

    def _resolve_nodes(self, current_nodes: np.ndarray | None) -> np.ndarray:
        """Resolve finite local geometry [length] from local or stacked input."""
        nodes = (
            self.domain.current_nodes if current_nodes is None
            else np.asarray(current_nodes, dtype=np.float64)
        )
        count = len(self.domain.current_nodes)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("SelfContact geometry must have shape (n_nodes, 3)")
        if len(nodes) != count:
            first = self.dof_offset // 3
            nodes = nodes[first:first + count]
        if nodes.shape != (count, 3) or not np.all(np.isfinite(nodes)):
            raise ValueError("SelfContact geometry must contain finite bound-body coordinates")
        return nodes

    def _compute_surface_normals(
        self, current_nodes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return opposition normals, unit face normals and vertex sign sums.

        Cross products/angle sums are vectorized over all facets. Area-weighted
        vertex normals are used only by the retained opposition filter. Sign
        pseudonormals sum corner-angle-weighted unit normals; they never define
        the gap-gradient direction. Relative-area rejection matches the shared
        triangular kernel, and degenerate geometry raises ValueError.
        """
        vertices = current_nodes[self._surf_node_idx[self._faces]]
        e1, e2 = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
        cross = np.cross(e1, e2)
        areas = np.linalg.norm(cross, axis=1)
        edge_scale = np.maximum.reduce((
            np.sum(e1*e1, axis=1), np.sum(e2*e2, axis=1),
            np.sum((e2-e1)**2, axis=1),
        ))
        if (
            not np.all(np.isfinite(areas))
            or not np.all(np.isfinite(edge_scale))
            or np.any(areas <= TriangleContactGeometry._relative_tolerance * edge_scale)
        ):
            raise ValueError("Degenerate self-contact triangle: nonzero relative area required")
        face_normals = cross / areas[:, None]
        nodal = np.zeros((self.n_surf_nodes, 3))
        vertex_sign = np.zeros_like(nodal)
        for corner in range(3):
            np.add.at(nodal, self._faces[:, corner], cross)
            a = vertices[:, (corner + 1) % 3] - vertices[:, corner]
            b = vertices[:, (corner + 2) % 3] - vertices[:, corner]
            angle = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))
            np.add.at(vertex_sign, self._faces[:, corner], angle[:, None] * face_normals)
        lengths = np.linalg.norm(nodal, axis=1)
        nodal = np.divide(
            nodal, lengths[:, None], out=np.zeros_like(nodal),
            where=lengths[:, None] > 0.0,
        )
        if self.smoothing_weight == "area":
            pseudonormals = np.zeros_like(nodal)
            for corner in range(3):
                np.add.at(pseudonormals, self._faces[:, corner], cross / 3.0)
        else:
            pseudonormals = vertex_sign
        p_norms = np.linalg.norm(pseudonormals, axis=1, keepdims=True)
        unit_vertex_pseudonormals = np.where(
            p_norms > 1e-14,
            pseudonormals / np.maximum(p_norms, 1e-14),
            np.array([0.0, 0.0, 1.0]),
        )
        return nodal, face_normals, vertex_sign, unit_vertex_pseudonormals

    def _update_surface(self, current_nodes: np.ndarray | None = None) -> None:
        """Rebuild current geometry and normals; no state checksum/history cache."""
        nodes = self._resolve_nodes(current_nodes)
        (
            self._nodal_normals,
            self._face_normals,
            self._vertex_pseudonormals,
            self._unit_vertex_pseudonormals,
        ) = self._compute_surface_normals(nodes)
        self._polydata.points = nodes[self._surf_node_idx]
        self._locator.BuildLocator()

    @staticmethod
    def closest_point_triangle(
        p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Return exact closest point [length] and dimensionless feature weights.

        Delegates face/edge/vertex mathematics and degeneracy checks to the same
        shared kernel used by gap, force, energy, and tangent evaluation.
        """
        vertices = np.array([a, b, c], dtype=np.float64)
        weights = TriangleContactGeometry.closest_barycentric(p, vertices)
        return weights @ vertices, (float(weights[0]), float(weights[1]), float(weights[2]))

    def _queries(self, current_nodes: np.ndarray | None = None) -> list[_SelfContactQuery]:
        """Select the nearest admitted feature per slave using retained filters.

        This shared query serves all mechanics/lifecycle/diagnostic APIs. Exact
        distance derivatives hold with candidate selection fixed. Equal squared
        distances choose the lower face index, without persistent facet history.
        """
        nodes = self._resolve_nodes(current_nodes)
        self._update_surface(nodes)
        surface_positions = nodes[self._surf_node_idx]
        vertices = surface_positions[self._faces]
        face_minimum = vertices.min(axis=1)
        face_maximum = vertices.max(axis=1)
        candidates = vtk.vtkIdList()
        queries: list[_SelfContactQuery] = []
        radius = self.search_radius
        for slave_index, slave_node in enumerate(self.contact_nodes):
            surface_index = self._vol_to_surf_idx[int(slave_node)]
            point = nodes[slave_node]
            slave_normal = self._nodal_normals[surface_index]
            excluded = self._excluded_faces[surface_index]
            bounds = [point[0]-radius, point[0]+radius, point[1]-radius,
                      point[1]+radius, point[2]-radius, point[2]+radius]
            self._locator.FindCellsWithinBounds(bounds, candidates)
            candidate_ids = np.fromiter(
                (candidates.GetId(index) for index in range(candidates.GetNumberOfIds())),
                dtype=np.int64, count=candidates.GetNumberOfIds(),
            )
            # Point-to-AABB distance is a rigorous lower bound on distance to
            # its enclosed triangle. Cull distant search-box hits in one vector
            # operation without changing any exact nearest-feature decision.
            box_delta = np.maximum(np.maximum(
                face_minimum[candidate_ids] - point,
                point - face_maximum[candidate_ids],
            ), 0.0)
            within_cutoff = np.einsum("ij,ij->i", box_delta, box_delta) <= self.contact_thickness**2
            best_squared = np.inf
            best_face: int | None = None
            for candidate_id in candidate_ids[within_cutoff]:
                face_index = int(candidate_id)
                if face_index in excluded:
                    continue
                if float(slave_normal @ self._face_normals[face_index]) >= self.normal_threshold:
                    continue
                weights = TriangleContactGeometry.closest_barycentric(point, vertices[face_index])
                residual = point - weights @ vertices[face_index]
                squared = float(residual @ residual)
                if squared > self.contact_thickness**2:
                    continue
                if squared < best_squared or (
                    squared == best_squared and best_face is not None
                    and face_index < best_face
                ):
                    best_squared, best_face = squared, face_index
            if best_face is None:
                continue
            face = self._faces[best_face]
            weights = TriangleContactGeometry.closest_barycentric(point, vertices[best_face])
            active = np.flatnonzero(weights > 0.0)
            if len(active) == 3:
                pseudonormal = self._face_normals[best_face]
            elif len(active) == 2:
                edge = tuple(sorted((int(face[active[0]]), int(face[active[1]]))))
                pseudonormal = self._face_normals[self._edge_faces[edge]].sum(axis=0)
            else:
                pseudonormal = self._vertex_pseudonormals[face[active[0]]]
            if np.linalg.norm(pseudonormal) <= TriangleContactGeometry._relative_tolerance:
                raise ValueError("Undefined self-contact feature pseudonormal: check surface orientation")
            v_normals = self._unit_vertex_pseudonormals[face] if self.smoothed_normals else None
            geometry = TriangleContactGeometry.evaluate(
                point, vertices[best_face], pseudonormal, vertex_normals=v_normals
            )
            queries.append(_SelfContactQuery(
                slave_index, int(slave_node), self._surf_node_idx[face], geometry
            ))
        return queries

    def _pressure(self, query: _SelfContactQuery) -> tuple[float, float]:
        """Return p=max(0,lambda-kp*g) and fixed per-slave lambda [force]."""
        multiplier = (
            float(self.multipliers[query.slave_index])
            if self.formulation == "augmented_lagrangian" else 0.0
        )
        return max(0.0, multiplier - self.k_penalty * query.geometry.gap), multiplier

    def _find_active_contacts(
        self, current_nodes: np.ndarray | None = None
    ) -> list[_SelfContactQuery]:
        """Return admitted typed constraints with strictly positive pressure."""
        return [
            query for query in self._queries(current_nodes)
            if self._pressure(query)[0] > 0.0
        ]

    def _dof_indices(self, query: _SelfContactQuery) -> np.ndarray:
        """Map the 12 local translational coordinates into global body DOFs."""
        nodes = np.r_[query.slave_node, query.master_nodes]
        return (self.dof_offset + 3 * nodes[:, None] + np.arange(3)).ravel()

    def evaluate_gaps(self, current_nodes: np.ndarray | None = None) -> np.ndarray:
        """Return candidate-aligned signed gaps [length], +inf if none admitted."""
        gaps = np.full(len(self.contact_nodes), np.inf)
        for query in self._queries(current_nodes):
            gaps[query.slave_index] = query.geometry.gap
        return gaps

    def evaluate_forces(
        self,
        current_nodes: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
        n_dofs: int | None = None,
    ) -> np.ndarray:
        """Assemble f=p*grad(g) [force], plus optional dissipative laws.

        Conservative normal forces have zero resultant and moment, following
        rigid-motion invariance of each admitted signed-distance potential.
        Normal damping and regularized friction remain available, with only
        their retained approximate dissipative tangent in evaluate_stiffness.
        Dissipative forces act only on strictly positive-pressure constraints;
        the p=0 activation boundary uses the open branch with no damping.
        """
        size = self.dof_offset + self.domain.n_dofs if n_dofs is None else n_dofs
        forces = np.zeros(size, dtype=np.float64)
        velocity = (
            None if velocities is None
            else np.asarray(velocities, dtype=np.float64).reshape(-1)
        )
        for query in self._find_active_contacts(current_nodes):
            pressure, _ = self._pressure(query)
            geometry = query.geometry
            indices = self._dof_indices(query)
            local_force = pressure * geometry.gradient
            if velocity is not None:
                weights = np.r_[1.0, -geometry.barycentric]
                relative_velocity = weights @ velocity[indices].reshape(4, 3)
                normal_velocity = float(relative_velocity @ geometry.normal)
                if self.c_damping > 0.0 and normal_velocity < 0.0:
                    local_force -= self.c_damping * normal_velocity * geometry.gradient
                tangential = relative_velocity - normal_velocity * geometry.normal
                speed = float(np.linalg.norm(tangential))
                if self.friction > 0.0 and speed > 1e-9:
                    friction_force = (
                        -self.friction * pressure * np.tanh(speed / 1e-3)
                        * tangential / speed
                    )
                    local_force += np.kron(weights, friction_force)
            np.add.at(forces, indices, local_force)
        return forces

    def evaluate_stiffness(
        self,
        current_nodes: np.ndarray | None = None,
        n_dofs: int | None = None,
        a1: float = 0.0,
    ) -> spsp.csr_matrix:
        """Assemble K=kp*b*b.T-p*H [force/length] on selected active branches.

        All analytical geometric terms enter symmetric 12x12 local blocks; the
        global tangent need not be PSD. Sparse COO assembly uses vectorized DOF
        indexing. Optional a1*c*b*b.T retains the approximate normal damping
        tangent; friction and full dissipative geometry derivatives are omitted.
        """
        size = self.dof_offset + self.domain.n_dofs if n_dofs is None else n_dofs
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for query in self._find_active_contacts(current_nodes):
            pressure, _ = self._pressure(query)
            geometry = query.geometry
            tangent = (
                (self.k_penalty + a1 * self.c_damping) * np.outer(geometry.gradient, geometry.gradient)
                - pressure * geometry.hessian
            )
            indices = self._dof_indices(query)
            rows.append(np.repeat(indices, 12))
            columns.append(np.tile(indices, 12))
            values.append(tangent.ravel())
        if not values:
            return spsp.csr_matrix((size, size), dtype=np.float64)
        return spsp.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(size, size),
        ).tocsr()

    def evaluate_potential_energy(
        self, current_nodes: np.ndarray | None = None
    ) -> float:
        """Return fixed-lambda normal potential sum((p^2-lambda^2)/(2*kp)).

        Potential [force*length] includes inactive AL constants; it has no
        dissipative component. Exact derivative claims hold with the admitted
        candidate, feature, and oriented sign branch fixed.
        """
        return float(self.get_contact_info(current_nodes)["potential_energy"])

    def get_contact_info(
        self, current_nodes: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Report selected-candidate normal-law penetration, activity and energy.

        All fields use the same candidate gaps and projected pressures. A local
        negative gap is not by itself proof of physical self-intersection.
        """
        gaps = self.evaluate_gaps(current_nodes)
        multipliers = (
            self.multipliers if self.formulation == "augmented_lagrangian"
            else np.zeros_like(gaps)
        )
        pressure = np.maximum(0.0, multipliers - self.k_penalty * gaps)
        active = (pressure > 0.0) | (gaps < 0.0)
        return {
            "num_active": int(np.count_nonzero(active)),
            "max_penetration": float(np.max(np.maximum(-gaps, 0.0), initial=0.0)),
            "active_nodes": self.contact_nodes[active],
            "potential_energy": float(np.sum((pressure**2 - multipliers**2) / (2.0 * self.k_penalty))),
            "max_multiplier": float(np.max(self.multipliers, initial=0.0)),
        }


