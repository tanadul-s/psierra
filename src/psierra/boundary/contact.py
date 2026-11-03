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
from typing import Any, Literal

import numpy as np
import scipy.sparse as spsp
import vtk

from psierra.domain.base import Domain
from psierra.collision.shapes import CollisionShape
from psierra.collision.raytrace import RayMeshObstacle
from psierra.collision.detector import CollisionDetector, ContactQueryResult


class RigidContact:
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

    Linearization yields a symmetric positive semi-definite tangent stiffness contribution:
        K_c = k_p * (n (x) n)
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
        if query_res.num_active == 0:
            return f_c

        active_idx = query_res.active_indices
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
            incorporates velocity damping linearization for exact Newton convergence.
        """
        nodes = self._resolve_nodes(current_nodes)
        s_nodes = self._resolve_slave_nodes(nodes)

        if n_dofs is None:
            if self.domain is not None:
                n_dofs = self.domain.n_dofs
            else:
                n_dofs = len(nodes) * self.dofs_per_node + self.dof_offset

        query_res = self.detector.query(nodes[s_nodes], tolerance=0.0)
        if query_res.num_active == 0:
            return spsp.csr_matrix((n_dofs, n_dofs), dtype=np.float64)

        active_idx = query_res.active_indices
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
        Calculates the stored contact elastic potential energy:
        Pi_c = 1/2 * k_p * sum_{active} g_N^2
        """
        nodes = self._resolve_nodes(current_nodes)
        s_nodes = self._resolve_slave_nodes(nodes)
        query_res = self.detector.query(nodes[s_nodes], tolerance=0.0)
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

        num_active = query_res.num_active
        max_pen = query_res.max_penetration
        active_nodes = s_nodes[query_res.active_indices]
        energy = (
            float(0.5 * self.k_penalty * np.sum(query_res.gaps[query_res.active_indices] ** 2))
            if num_active > 0
            else 0.0
        )

        return {
            "num_active": num_active,
            "max_penetration": max_pen,
            "active_nodes": active_nodes,
            "potential_energy": energy,
            "max_multiplier": float(np.max(self.multipliers)) if len(self.multipliers) > 0 else 0.0,
        }

    def update_multipliers(self) -> dict[str, float]:
        """
        Updates Augmented Lagrangian contact multipliers:
            lambda_s <- max(0, lambda_s - k_p * g_N)
        """
        nodes = self._resolve_nodes()
        s_nodes = self._resolve_slave_nodes(nodes)
        if len(self.multipliers) != len(s_nodes):
            self.multipliers = np.zeros(len(s_nodes), dtype=np.float64)

        query_res = self.detector.query(nodes[s_nodes], tolerance=self.tolerance)
        max_pen = 0.0
        max_delta_lam = 0.0
        num_active = 0

        active_set = (
            set(query_res.active_indices) if query_res.num_active > 0 else set()
        )
        for idx in range(len(s_nodes)):
            old_lam = self.multipliers[idx]
            if idx in active_set:
                gap = query_res.gaps[idx]
                new_lam = float(max(0.0, old_lam - self.k_penalty * gap))
                if gap < 0.0:
                    pen = -gap
                    if pen > max_pen:
                        max_pen = pen
            else:
                new_lam = 0.0
            self.multipliers[idx] = new_lam

            d_lam = abs(new_lam - old_lam)
            if d_lam > max_delta_lam:
                max_delta_lam = d_lam
            if new_lam > 0.0:
                num_active += 1

        return {
            "max_penetration": max_pen,
            "max_multiplier": float(np.max(self.multipliers)) if len(self.multipliers) > 0 else 0.0,
            "max_delta_multiplier": max_delta_lam,
            "num_active": num_active,
        }


class DeformableContact:
    """
    Two-Way Deformable-to-Deformable (Mesh-to-Mesh) Contact Formulation.

    Enforces non-penetration and contact equilibrium between two independent
    flexible continuum bodies (Slave Body A and Master Body B) with exact
    Newtonian action-reaction interface balance:
        f_s + sum_{i=1}^3 f_{m_i} = 0
    and symmetric cross-body tangent stiffness coupling:
        K_coupled = [ K_A + K_c,AA   K_c,AB ]
                    [ K_c,BA         K_B + K_c,BB ]
    where K_c,AB = K_c,BA^T.

    Each active slave node s penetrating master facet (m_1, m_2, m_3) with
    barycentric weights (xi_1, xi_2, xi_3) generates the 12x12 rank-1 contact element:
        K_c^(e) = k_p^* (w (x) w) (x) (n (x) n)
    where w = [1, -xi_1, -xi_2, -xi_3]^T.
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
    ) -> None:
        """
        Parameters:
        -----------
        slave : Domain
            Deformable slave continuum domain.
        master : Domain
            Deformable master continuum domain.
        slave_nodes : Sequence[int] or np.ndarray, optional
            Candidate contact node indices on the slave body (defaults to surface nodes).
        k_penalty : float, default=1e4
            Normal penalty contact stiffness in N/mm.
        c_damping : float, default=0.0
            Contact velocity damping coefficient in N*s/mm.
        friction : float, default=0.0
            Coulomb tangential friction coefficient.
        slave_offset : int, default=0
            Global DOF offset for slave body within multi-body system.
        master_offset : int, default=0
            Global DOF offset for master body within multi-body system.
        domain : Domain, optional
            Parent MultiBodySystem domain.
        formulation : {'penalty', 'augmented_lagrangian'}, default='penalty'
            Contact enforcement scheme.
        tolerance : float, default=0.0
            Penetration activation threshold (gap <= tolerance).
        """
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

        # 1. Determine candidate slave nodes
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

        # 2. Extract and configure master surface triangulation
        if hasattr(self.master, "mesh") and hasattr(self.master.mesh, "surface"):
            m_surf = self.master.mesh.surface
            self._master_polydata = m_surf.polydata.copy(deep=True)
            self._master_surf_node_idx = np.asarray(m_surf.node_indices, dtype=np.int64)
        else:
            raise TypeError("Master domain must possess a surface mesh representation.")

        if not self._master_polydata.is_all_triangles:
            self._master_polydata = self._master_polydata.triangulate()
        self._master_faces = self._master_polydata.regular_faces

        # VTK StaticCellLocator for master surface proximity
        self._locator = vtk.vtkStaticCellLocator()
        self._locator.SetDataSet(self._master_polydata)
        self._locator.BuildLocator()

        # Augmented Lagrangian multipliers (one per slave node)
        self.multipliers = np.zeros(len(self.slave_nodes), dtype=np.float64)
        self._master_nodal_normals = self._compute_master_nodal_normals()

    def _compute_master_nodal_normals(self) -> np.ndarray:
        """
        Computes area-weighted C0-continuous vertex normals for the master surface mesh.
        Interpolating contact normals from vertex normals prevents discontinuous force jumps
        and contact chatter across facet edges.
        """
        master_pos = self.master.current_nodes
        f_surf = self._master_faces
        m_vol = self._master_surf_node_idx[f_surf]
        v1 = master_pos[m_vol[:, 0]]
        v2 = master_pos[m_vol[:, 1]]
        v3 = master_pos[m_vol[:, 2]]
        face_normals = np.cross(v2 - v1, v3 - v1)

        n_surf_nodes = len(self._master_surf_node_idx)
        nodal_normals = np.zeros((n_surf_nodes, 3), dtype=np.float64)
        np.add.at(nodal_normals, f_surf[:, 0], face_normals)
        np.add.at(nodal_normals, f_surf[:, 1], face_normals)
        np.add.at(nodal_normals, f_surf[:, 2], face_normals)

        norms = np.linalg.norm(nodal_normals, axis=1, keepdims=True)
        norms[norms < 1e-14] = 1.0
        return nodal_normals / norms

    @staticmethod
    def compute_barycentric(
        p: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Computes normalized barycentric coordinates (xi1, xi2, xi3) of point p
        projected onto triangle (v1, v2, v3) such that:
            p = xi1 * v1 + xi2 * v2 + xi3 * v3, with sum(xi) = 1 and xi_i in [0, 1].
        """
        e1 = v2 - v1
        e2 = v3 - v1
        d = p - v1
        d00 = float(np.dot(e1, e1))
        d01 = float(np.dot(e1, e2))
        d11 = float(np.dot(e2, e2))
        d20 = float(np.dot(d, e1))
        d21 = float(np.dot(d, e2))
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-15:
            return 1.0, 0.0, 0.0

        xi2 = (d11 * d20 - d01 * d21) / denom
        xi3 = (d00 * d21 - d01 * d20) / denom
        xi1 = 1.0 - xi2 - xi3

        # Clamp to [0, 1] and renormalize
        xi1 = max(0.0, min(1.0, xi1))
        xi2 = max(0.0, min(1.0, xi2))
        xi3 = max(0.0, min(1.0, xi3))
        s = xi1 + xi2 + xi3
        if s > 0.0:
            return xi1 / s, xi2 / s, xi3 / s
        return 1.0, 0.0, 0.0

    def _update_master_surface(self) -> None:
        """Synchronizes master surface polydata coordinates with current deformed nodes."""
        current_m_nodes = self.master.current_nodes
        self._master_polydata.points = current_m_nodes[self._master_surf_node_idx]
        self._locator.BuildLocator()
        self._master_nodal_normals = self._compute_master_nodal_normals()

    def evaluate_forces(
        self,
        current_nodes: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
        n_dofs: int | None = None,
    ) -> np.ndarray:
        """
        Evaluates the global two-way contact force vector f_c.
        """
        self._update_master_surface()

        slave_nodes_pos = self.slave.current_nodes
        master_nodes_pos = self.master.current_nodes

        if n_dofs is None:
            if self.domain is not None:
                n_dofs = self.domain.n_dofs
            else:
                n_dofs = max(
                    self.slave_offset + self.slave.n_dofs,
                    self.master_offset + self.master.n_dofs,
                )

        f_c = np.zeros(n_dofs, dtype=np.float64)

        c_pt = np.zeros(3, dtype=np.float64)
        cid = vtk.mutable(0)
        subid = vtk.mutable(0)
        d2 = vtk.mutable(0.0)

        for s_idx, s_node in enumerate(self.slave_nodes):
            x_s = slave_nodes_pos[s_node]
            self._locator.FindClosestPoint(x_s, c_pt, cid, subid, d2)
            cell_id = int(cid)

            f_local = self._master_faces[cell_id]
            m1_vol = int(self._master_surf_node_idx[f_local[0]])
            m2_vol = int(self._master_surf_node_idx[f_local[1]])
            m3_vol = int(self._master_surf_node_idx[f_local[2]])

            v1 = master_nodes_pos[m1_vol]
            v2 = master_nodes_pos[m2_vol]
            v3 = master_nodes_pos[m3_vol]

            # Barycentric coordinates of c_pt on triangle
            xi1, xi2, xi3 = self.compute_barycentric(c_pt, v1, v2, v3)

            # C0-continuous contact normal interpolated from master vertex normals
            n_interp = (
                xi1 * self._master_nodal_normals[f_local[0]]
                + xi2 * self._master_nodal_normals[f_local[1]]
                + xi3 * self._master_nodal_normals[f_local[2]]
            )
            norm_mag = float(np.linalg.norm(n_interp))
            if norm_mag < 1e-14:
                continue
            n = n_interp / norm_mag

            # Signed normal gap (negative = penetration)
            gap = float(np.dot(x_s - c_pt, n))

            lam = (
                self.multipliers[s_idx]
                if self.formulation == "augmented_lagrangian"
                else 0.0
            )
            f_N = float(max(0.0, lam - self.k_penalty * gap))

            if f_N <= 0.0:
                continue

            f_damp = 0.0
            f_t_s = np.zeros(3, dtype=np.float64)
            f_t_m = np.zeros(3, dtype=np.float64)

            if velocities is not None:
                vel = np.asarray(velocities, dtype=np.float64)
                dof_s = self.slave_offset + 3 * s_node
                dof_m1 = self.master_offset + 3 * m1_vol
                dof_m2 = self.master_offset + 3 * m2_vol
                dof_m3 = self.master_offset + 3 * m3_vol

                v_s = vel[dof_s : dof_s + 3]
                v_m = (
                    xi1 * vel[dof_m1 : dof_m1 + 3]
                    + xi2 * vel[dof_m2 : dof_m2 + 3]
                    + xi3 * vel[dof_m3 : dof_m3 + 3]
                )
                v_rel = v_s - v_m
                vn = float(np.dot(v_rel, n))

                if self.c_damping > 0.0 and vn < 0.0:
                    f_damp = -self.c_damping * vn

                if self.friction > 0.0:
                    vt_vec = v_rel - vn * n
                    vt_mag = float(np.linalg.norm(vt_vec))
                    if vt_mag > 1e-9:
                        t_dir = vt_vec / vt_mag
                        ft_mag = self.friction * f_N * np.tanh(vt_mag / 1e-3)
                        f_t_s = -ft_mag * t_dir
                        f_t_m = ft_mag * t_dir

            f_total = f_N + f_damp

            # Action on slave node
            dof_s = self.slave_offset + 3 * s_node
            f_c[dof_s : dof_s + 3] += f_total * n + f_t_s

            # Reaction on master nodes (distributed via barycentric weights)
            dof_m1 = self.master_offset + 3 * m1_vol
            dof_m2 = self.master_offset + 3 * m2_vol
            dof_m3 = self.master_offset + 3 * m3_vol

            f_c[dof_m1 : dof_m1 + 3] += -xi1 * f_total * n + xi1 * f_t_m
            f_c[dof_m2 : dof_m2 + 3] += -xi2 * f_total * n + xi2 * f_t_m
            f_c[dof_m3 : dof_m3 + 3] += -xi3 * f_total * n + xi3 * f_t_m

        return f_c

    def evaluate_stiffness(
        self,
        current_nodes: np.ndarray | None = None,
        n_dofs: int | None = None,
        a1: float = 0.0,
    ) -> spsp.csr_matrix:
        """
        Assembles the sparse coupled symmetric tangent contact stiffness matrix K_c.
        """
        self._update_master_surface()

        slave_nodes_pos = self.slave.current_nodes
        master_nodes_pos = self.master.current_nodes

        if n_dofs is None:
            if self.domain is not None:
                n_dofs = self.domain.n_dofs
            else:
                n_dofs = max(
                    self.slave_offset + self.slave.n_dofs,
                    self.master_offset + self.master.n_dofs,
                )

        kp = self.k_penalty + a1 * self.c_damping

        c_pt = np.zeros(3, dtype=np.float64)
        cid = vtk.mutable(0)
        subid = vtk.mutable(0)
        d2 = vtk.mutable(0.0)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for s_idx, s_node in enumerate(self.slave_nodes):
            x_s = slave_nodes_pos[s_node]
            self._locator.FindClosestPoint(x_s, c_pt, cid, subid, d2)
            cell_id = int(cid)

            f_local = self._master_faces[cell_id]
            m1_vol = int(self._master_surf_node_idx[f_local[0]])
            m2_vol = int(self._master_surf_node_idx[f_local[1]])
            m3_vol = int(self._master_surf_node_idx[f_local[2]])

            v1 = master_nodes_pos[m1_vol]
            v2 = master_nodes_pos[m2_vol]
            v3 = master_nodes_pos[m3_vol]

            xi1, xi2, xi3 = self.compute_barycentric(c_pt, v1, v2, v3)

            # C0-continuous contact normal interpolated from master vertex normals
            n_interp = (
                xi1 * self._master_nodal_normals[f_local[0]]
                + xi2 * self._master_nodal_normals[f_local[1]]
                + xi3 * self._master_nodal_normals[f_local[2]]
            )
            norm_mag = float(np.linalg.norm(n_interp))
            if norm_mag < 1e-14:
                continue
            n = n_interp / norm_mag

            gap = float(np.dot(x_s - c_pt, n))

            lam = (
                self.multipliers[s_idx]
                if self.formulation == "augmented_lagrangian"
                else 0.0
            )
            f_N = float(max(0.0, lam - self.k_penalty * gap))

            if f_N <= 0.0:
                continue

            dof_starts = [
                self.slave_offset + 3 * s_node,
                self.master_offset + 3 * m1_vol,
                self.master_offset + 3 * m2_vol,
                self.master_offset + 3 * m3_vol,
            ]
            weights = [1.0, -xi1, -xi2, -xi3]
            nn = np.outer(n, n)

            # Assemble 16 blocks (4x4 node pairs)
            for a in range(4):
                w_a = weights[a]
                dof_a = dof_starts[a]
                for b in range(4):
                    w_b = weights[b]
                    dof_b = dof_starts[b]
                    block_scale = kp * w_a * w_b

                    for i in range(3):
                        for j in range(3):
                            rows.append(dof_a + i)
                            cols.append(dof_b + j)
                            vals.append(block_scale * nn[i, j])

        if not rows:
            return spsp.csr_matrix((n_dofs, n_dofs), dtype=np.float64)

        K_c = spsp.coo_matrix(
            (vals, (rows, cols)), shape=(n_dofs, n_dofs), dtype=np.float64
        ).tocsr()
        return K_c

    def update_multipliers(self) -> dict[str, float]:
        """
        Updates Augmented Lagrangian contact multipliers:
            lambda_s <- max(0, lambda_s - k_p * g_N)
        """
        self._update_master_surface()
        slave_nodes_pos = self.slave.current_nodes
        master_nodes_pos = self.master.current_nodes

        c_pt = np.zeros(3, dtype=np.float64)
        cid = vtk.mutable(0)
        subid = vtk.mutable(0)
        d2 = vtk.mutable(0.0)

        max_pen = 0.0
        num_active = 0
        max_delta_lam = 0.0

        for s_idx, s_node in enumerate(self.slave_nodes):
            x_s = slave_nodes_pos[s_node]
            self._locator.FindClosestPoint(x_s, c_pt, cid, subid, d2)
            cell_id = int(cid)

            f_local = self._master_faces[cell_id]
            m1_vol = int(self._master_surf_node_idx[f_local[0]])
            m2_vol = int(self._master_surf_node_idx[f_local[1]])
            m3_vol = int(self._master_surf_node_idx[f_local[2]])

            v1 = master_nodes_pos[m1_vol]
            v2 = master_nodes_pos[m2_vol]
            v3 = master_nodes_pos[m3_vol]

            xi1, xi2, xi3 = self.compute_barycentric(c_pt, v1, v2, v3)

            # C0-continuous contact normal interpolated from master vertex normals
            n_interp = (
                xi1 * self._master_nodal_normals[f_local[0]]
                + xi2 * self._master_nodal_normals[f_local[1]]
                + xi3 * self._master_nodal_normals[f_local[2]]
            )
            norm_mag = float(np.linalg.norm(n_interp))
            if norm_mag < 1e-14:
                continue
            n = n_interp / norm_mag

            gap = float(np.dot(x_s - c_pt, n))
            old_lam = self.multipliers[s_idx]
            new_lam = float(max(0.0, old_lam - self.k_penalty * gap))
            self.multipliers[s_idx] = new_lam

            delta_lam = abs(new_lam - old_lam)
            if delta_lam > max_delta_lam:
                max_delta_lam = delta_lam

            if gap < 0.0:
                pen = -gap
                if pen > max_pen:
                    max_pen = pen
            if new_lam > 0.0:
                num_active += 1

        return {
            "max_penetration": max_pen,
            "max_multiplier": float(np.max(self.multipliers)) if len(self.multipliers) > 0 else 0.0,
            "max_delta_multiplier": max_delta_lam,
            "num_active": num_active,
        }

    def evaluate_potential_energy(
        self, current_nodes: np.ndarray | None = None
    ) -> float:
        """
        Calculates stored contact potential energy:
        Pi_c = 1/2 * k_p * sum_{active} g_N^2
        """
        c_info = self.get_contact_info()
        return float(c_info["potential_energy"])

    def get_contact_info(
        self, current_nodes: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Returns structured summary of current contact status."""
        self._update_master_surface()
        slave_nodes_pos = self.slave.current_nodes
        master_nodes_pos = self.master.current_nodes

        c_pt = np.zeros(3, dtype=np.float64)
        cid = vtk.mutable(0)
        subid = vtk.mutable(0)
        d2 = vtk.mutable(0.0)

        num_active = 0
        max_pen = 0.0
        active_nodes = []
        contact_energy = 0.0

        for s_idx, s_node in enumerate(self.slave_nodes):
            x_s = slave_nodes_pos[s_node]
            self._locator.FindClosestPoint(x_s, c_pt, cid, subid, d2)
            cell_id = int(cid)

            f_local = self._master_faces[cell_id]
            m1_vol = int(self._master_surf_node_idx[f_local[0]])
            m2_vol = int(self._master_surf_node_idx[f_local[1]])
            m3_vol = int(self._master_surf_node_idx[f_local[2]])

            v1 = master_nodes_pos[m1_vol]
            v2 = master_nodes_pos[m2_vol]
            v3 = master_nodes_pos[m3_vol]

            xi1, xi2, xi3 = self.compute_barycentric(c_pt, v1, v2, v3)

            # C0-continuous contact normal interpolated from master vertex normals
            n_interp = (
                xi1 * self._master_nodal_normals[f_local[0]]
                + xi2 * self._master_nodal_normals[f_local[1]]
                + xi3 * self._master_nodal_normals[f_local[2]]
            )
            norm_mag = float(np.linalg.norm(n_interp))
            if norm_mag < 1e-14:
                continue
            n = n_interp / norm_mag

            gap = float(np.dot(x_s - c_pt, n))
            lam = (
                self.multipliers[s_idx]
                if self.formulation == "augmented_lagrangian"
                else 0.0
            )
            f_N = float(max(0.0, lam - self.k_penalty * gap))

            if f_N > 0.0 or gap < 0.0:
                num_active += 1
                active_nodes.append(s_node)
                if gap < 0.0:
                    pen = -gap
                    if pen > max_pen:
                        max_pen = pen
                    contact_energy += 0.5 * self.k_penalty * (gap**2)

        return {
            "num_active": num_active,
            "max_penetration": max_pen,
            "active_nodes": np.asarray(active_nodes, dtype=np.int64),
            "potential_energy": contact_energy,
            "max_multiplier": float(np.max(self.multipliers)) if len(self.multipliers) > 0 else 0.0,
        }


class SelfContact:
    """
    Self-Contact Boundary Formulation for Project Sierra.

    Prevents self-intersection and interpenetration of folds, opposing lobes,
    cavities, and complex geometry within a single deformable body:
        f_c = f_N * n
    where:
        f_N = -k_p * g_N (penalty formulation, active when signed gap g_N < tolerance)
        f_N = max(0, lambda_N - k_p * g_N) (augmented Lagrangian formulation)

    Key Features:
      - Topological Exclusion: Excludes 1-ring and 2-ring neighborhood facets
        from self-detection to eliminate spurious self-pinning and distance singularities.
      - Normal Direction Filtering: Skips non-opposing candidate facets
        (n_node . n_facet >= normal_threshold) to focus strictly on colliding surfaces.
      - Exact Voronoi Triangle Projection: Computes exact closest points and normalized
        barycentric coordinates on candidate master triangular facets.
      - C0-Continuous Normal Interpolation: Interpolates contact normals from vertex
        normals using barycentric weights to prevent lateral chatter across facet edges.
      - Action-Reaction Symmetry: Pushes slave node along +n and distributes reaction
        equally onto master facet nodes (-xi_i * n), strictly conserving linear and angular momentum.
      - Symmetric Tangent Stiffness: Assembles rank-1 block updates:
            K_c = kp * (w (x) w) (x) (n (x) n)
        with w = [1, -xi_1, -xi_2, -xi_3], ensuring exact symmetry K_c = K_c^T.
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
    ) -> None:
        """
        Parameters:
        -----------
        domain : Domain
            Deformable domain undergoing self-contact.
        contact_nodes : Sequence[int] or np.ndarray, optional
            Candidate contact node indices on the body (defaults to all surface nodes).
        k_penalty : float, default=1e4
            Normal penalty contact stiffness in N/mm.
        c_damping : float, default=0.0
            Contact velocity damping coefficient in N*s/mm.
        friction : float, default=0.0
            Tangential friction coefficient.
        dof_offset : int, default=0
            Global DOF offset when used within a MultiBodySystem assembly.
        formulation : {'penalty', 'augmented_lagrangian'}, default='penalty'
            Contact constraint enforcement scheme.
        tolerance : float, default=0.0
            Activation threshold (gap <= tolerance).
        search_radius : float, optional
            Bounding box search radius for proximity queries (mm).
            Defaults to 20% of the bounding box diagonal.
        contact_thickness : float, optional
            Maximum physical proximity threshold for active self-contact (mm).
            Prevents distant opposing body walls from triggering false interior penetrations.
            Defaults to 0.5 * mean surface edge length.
        normal_threshold : float, default=-0.2
            Maximum dot product (n_node . n_facet) for opposing contact detection.
        exclusion_rings : int, default=2
            Number of topological vertex-face neighbor rings to exclude from self-detection.
        """
        self.domain = domain
        self.k_penalty = float(k_penalty)
        self.c_damping = float(c_damping)
        self.friction = float(friction)
        self.dof_offset = int(dof_offset)
        self.formulation = formulation
        self.tolerance = float(tolerance)
        self.normal_threshold = float(normal_threshold)
        self.exclusion_rings = int(exclusion_rings)

        # 1. Extract surface triangulation
        if hasattr(self.domain, "mesh") and hasattr(self.domain.mesh, "surface"):
            surf = self.domain.mesh.surface
            self._polydata = surf.polydata.copy(deep=True)
            self._surf_node_idx = np.asarray(surf.node_indices, dtype=np.int64)
        else:
            raise TypeError("Domain must possess a surface mesh representation for SelfContact.")

        if not self._polydata.is_all_triangles:
            self._polydata = self._polydata.triangulate()
        self._faces = self._polydata.regular_faces
        self.n_surf_nodes = len(self._surf_node_idx)
        self.n_faces = len(self._faces)

        # Map from volume node index to surface node index
        self._vol_to_surf_idx: dict[int, int] = {
            int(vol_idx): surf_idx
            for surf_idx, vol_idx in enumerate(self._surf_node_idx)
        }

        # 2. Determine candidate contact nodes
        if contact_nodes is not None:
            self.contact_nodes = np.asarray(contact_nodes, dtype=np.int64)
        else:
            self.contact_nodes = self._surf_node_idx.copy()

        # Multipliers tracking
        self.multipliers = np.zeros(len(self.contact_nodes), dtype=np.float64)
        self._node_to_multiplier_idx: dict[int, int] = {
            int(node): idx for idx, node in enumerate(self.contact_nodes)
        }

        # 3. Determine search radius & contact thickness
        pts = self.domain.mesh.nodes[self._surf_node_idx]
        bbox_span = np.ptp(pts, axis=0)
        bbox_diag = float(np.linalg.norm(bbox_span))
        if search_radius is None:
            self.search_radius = max(5.0, 0.20 * bbox_diag)
        else:
            self.search_radius = float(search_radius)

        v1 = pts[self._faces[:, 0]]
        v2 = pts[self._faces[:, 1]]
        v3 = pts[self._faces[:, 2]]
        mean_edge = float(np.mean([
            np.linalg.norm(v2 - v1, axis=1),
            np.linalg.norm(v3 - v2, axis=1),
            np.linalg.norm(v1 - v3, axis=1),
        ]))
        self.mean_edge_length = mean_edge
        if contact_thickness is None:
            self.contact_thickness = max(1.0, 0.5 * mean_edge)
        else:
            self.contact_thickness = float(contact_thickness)

        # 4. Precompute topological exclusion graph
        self._excluded_faces = self._build_exclusion_graph(self.exclusion_rings)

        # 5. Build VTK StaticCellLocator for proximity queries
        self._locator = vtk.vtkStaticCellLocator()
        self._locator.SetDataSet(self._polydata)
        self._locator.BuildLocator()

        # 6. Precompute initial surface normals
        self._nodal_normals, self._face_normals = self._compute_surface_normals()

        # Cache for contact queries within the same state
        self._cached_contacts: list[dict[str, Any]] | None = None
        self._cached_key: tuple[float, float, float, int] | None = None

    def _build_exclusion_graph(self, rings: int) -> dict[int, set[int]]:
        """
        Precomputes the topological exclusion face set for each surface node using
        breadth-first search (BFS) up to the specified topological ring depth.
        For a node v, excludes all faces containing v or any topological neighbor
        within 'rings' edge hops, eliminating spurious self-detection across thin walls.
        """
        node_to_faces: dict[int, set[int]] = defaultdict(set)
        for f_idx, f in enumerate(self._faces):
            for v in f:
                node_to_faces[v].add(f_idx)

        node_adj: dict[int, set[int]] = defaultdict(set)
        for f in self._faces:
            u, v, w = f
            node_adj[u].update([v, w])
            node_adj[v].update([u, w])
            node_adj[w].update([u, v])

        excluded_faces: dict[int, set[int]] = {}
        for i in range(self.n_surf_nodes):
            current_ring = {i}
            all_visited = {i}
            for _ in range(rings):
                next_ring: set[int] = set()
                for v in current_ring:
                    next_ring.update(node_adj[v])
                current_ring = next_ring - all_visited
                all_visited.update(current_ring)
                if not current_ring:
                    break

            exc: set[int] = set()
            for v in all_visited:
                exc.update(node_to_faces[v])
            excluded_faces[i] = exc

        return excluded_faces

    def _compute_surface_normals(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes both face normals and area-weighted C0-continuous vertex normals
        for the current deformed surface mesh.
        """
        current_pos = self.domain.current_nodes
        f_surf = self._faces
        m_vol = self._surf_node_idx[f_surf]
        v1 = current_pos[m_vol[:, 0]]
        v2 = current_pos[m_vol[:, 1]]
        v3 = current_pos[m_vol[:, 2]]

        cross_prod = np.cross(v2 - v1, v3 - v1)
        f_norms = np.linalg.norm(cross_prod, axis=1, keepdims=True)
        f_norms[f_norms < 1e-14] = 1.0
        face_unit_normals = cross_prod / f_norms

        nodal_normals = np.zeros((self.n_surf_nodes, 3), dtype=np.float64)
        np.add.at(nodal_normals, f_surf[:, 0], cross_prod)
        np.add.at(nodal_normals, f_surf[:, 1], cross_prod)
        np.add.at(nodal_normals, f_surf[:, 2], cross_prod)

        n_norms = np.linalg.norm(nodal_normals, axis=1, keepdims=True)
        n_norms[n_norms < 1e-14] = 1.0
        nodal_unit_normals = nodal_normals / n_norms

        return nodal_unit_normals, face_unit_normals

    def _update_surface(self) -> None:
        """Synchronizes surface polydata points with current deformed domain coordinates."""
        current_pos = self.domain.current_nodes
        self._polydata.points = current_pos[self._surf_node_idx]
        self._locator.BuildLocator()
        self._nodal_normals, self._face_normals = self._compute_surface_normals()

    @staticmethod
    def closest_point_triangle(
        p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        """
        Computes exact closest point on triangle (a, b, c) to query point p
        and returns the point along with normalized barycentric coordinates (u, v, w).
        Uses Ericson's Voronoi region method (Real-Time Collision Detection, Sec. 5.1.5).
        """
        ab = b - a
        ac = c - a
        ap = p - a
        d1 = float(np.dot(ab, ap))
        d2 = float(np.dot(ac, ap))
        if d1 <= 0.0 and d2 <= 0.0:
            return a.copy(), (1.0, 0.0, 0.0)

        bp = p - b
        d3 = float(np.dot(ab, bp))
        d4 = float(np.dot(ac, bp))
        if d3 >= 0.0 and d4 <= d3:
            return b.copy(), (0.0, 1.0, 0.0)

        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return a + v * ab, (1.0 - v, v, 0.0)

        cp = p - c
        d5 = float(np.dot(ab, cp))
        d6 = float(np.dot(ac, cp))
        if d6 >= 0.0 and d5 <= d6:
            return c.copy(), (0.0, 0.0, 1.0)

        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return a + w * ac, (1.0 - w, 0.0, w)

        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return b + w * (c - b), (0.0, 1.0 - w, w)

        denom = va + vb + vc
        if abs(denom) < 1e-15:
            return a.copy(), (1.0, 0.0, 0.0)

        inv_denom = 1.0 / denom
        v = vb * inv_denom
        w = vc * inv_denom
        u = 1.0 - v - w
        return u * a + v * b + w * c, (u, v, w)

    def _find_active_contacts(
        self,
        current_nodes: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """
        Identifies all active self-contact pairs between candidate contact nodes
        and non-excluded opposing surface facets.
        """
        nodes_pos = self.domain.current_nodes if current_nodes is None else current_nodes

        # Check cache
        state_key = (
            float(np.sum(nodes_pos)),
            float(nodes_pos[0, 0]),
            float(nodes_pos[-1, -1]),
            id(velocities) if velocities is not None else 0,
        )
        if self._cached_key == state_key and self._cached_contacts is not None:
            return self._cached_contacts

        self._update_surface()
        r = self.search_radius
        id_list = vtk.vtkIdList()
        active_contacts: list[dict[str, Any]] = []

        surf_pos = nodes_pos[self._surf_node_idx]
        f_surf = self._faces
        m_v1_all = surf_pos[f_surf[:, 0]]
        m_v2_all = surf_pos[f_surf[:, 1]]
        m_v3_all = surf_pos[f_surf[:, 2]]

        for s_node in self.contact_nodes:
            s_mult_idx = self._node_to_multiplier_idx.get(int(s_node), -1)
            lam = self.multipliers[s_mult_idx] if (s_mult_idx >= 0 and self.formulation == "augmented_lagrangian") else 0.0

            s_surf_idx = self._vol_to_surf_idx.get(int(s_node), -1)
            x_s = nodes_pos[s_node]
            n_s = self._nodal_normals[s_surf_idx] if s_surf_idx >= 0 else np.array([0.0, 0.0, 1.0])
            exc = self._excluded_faces.get(s_surf_idx, set()) if s_surf_idx >= 0 else set()

            bbox = [
                x_s[0] - r, x_s[0] + r,
                x_s[1] - r, x_s[1] + r,
                x_s[2] - r, x_s[2] + r,
            ]
            self._locator.FindCellsWithinBounds(bbox, id_list)
            n_cands = id_list.GetNumberOfIds()
            if n_cands == 0:
                continue

            best_dist2 = float("inf")
            best_cand: tuple[int, np.ndarray, float, float, float] | None = None

            for c_i in range(n_cands):
                cid = id_list.GetId(c_i)
                if cid in exc:
                    continue

                # Filter non-opposing facets
                if s_surf_idx >= 0 and np.dot(n_s, self._face_normals[cid]) >= self.normal_threshold:
                    continue

                v1 = m_v1_all[cid]
                v2 = m_v2_all[cid]
                v3 = m_v3_all[cid]
                c_pt, (xi1, xi2, xi3) = self.closest_point_triangle(x_s, v1, v2, v3)
                diff = x_s - c_pt
                d2 = float(np.dot(diff, diff))

                if d2 > self.contact_thickness**2:
                    continue

                if d2 < best_dist2:
                    best_dist2 = d2
                    best_cand = (cid, c_pt, xi1, xi2, xi3)

            if best_cand is None:
                continue

            cid, c_pt, xi1, xi2, xi3 = best_cand
            f_local = self._faces[cid]
            m1_vol = int(self._surf_node_idx[f_local[0]])
            m2_vol = int(self._surf_node_idx[f_local[1]])
            m3_vol = int(self._surf_node_idx[f_local[2]])

            # C0-continuous contact normal interpolated from facet vertex normals
            n_interp = (
                xi1 * self._nodal_normals[f_local[0]]
                + xi2 * self._nodal_normals[f_local[1]]
                + xi3 * self._nodal_normals[f_local[2]]
            )
            norm_mag = float(np.linalg.norm(n_interp))
            if norm_mag < 1e-14:
                continue
            n = n_interp / norm_mag

            # Signed normal gap (negative = penetration)
            gap = float(np.dot(x_s - c_pt, n))

            # Alart-Curnier contact force operator
            f_N = float(max(0.0, lam - self.k_penalty * gap))

            if f_N <= 0.0 and gap > self.tolerance:
                continue

            # Velocity damping and friction
            f_damp = 0.0
            f_t_s = np.zeros(3, dtype=np.float64)
            f_t_m = np.zeros(3, dtype=np.float64)

            if velocities is not None:
                vel = np.asarray(velocities, dtype=np.float64)
                dof_s = self.dof_offset + 3 * s_node
                dof_m1 = self.dof_offset + 3 * m1_vol
                dof_m2 = self.dof_offset + 3 * m2_vol
                dof_m3 = self.dof_offset + 3 * m3_vol

                v_s = vel[dof_s : dof_s + 3]
                v_m = (
                    xi1 * vel[dof_m1 : dof_m1 + 3]
                    + xi2 * vel[dof_m2 : dof_m2 + 3]
                    + xi3 * vel[dof_m3 : dof_m3 + 3]
                )
                v_rel = v_s - v_m
                vn = float(np.dot(v_rel, n))

                if self.c_damping > 0.0 and vn < 0.0:
                    f_damp = -self.c_damping * vn

                if self.friction > 0.0:
                    vt_vec = v_rel - vn * n
                    vt_mag = float(np.linalg.norm(vt_vec))
                    if vt_mag > 1e-9:
                        t_dir = vt_vec / vt_mag
                        ft_mag = self.friction * f_N * np.tanh(vt_mag / 1e-3)
                        f_t_s = -ft_mag * t_dir
                        f_t_m = ft_mag * t_dir

            active_contacts.append({
                "slave_node": int(s_node),
                "master_nodes": (m1_vol, m2_vol, m3_vol),
                "barycentric": (xi1, xi2, xi3),
                "normal": n,
                "gap": gap,
                "f_N": f_N,
                "f_damp": f_damp,
                "f_t_s": f_t_s,
                "f_t_m": f_t_m,
                "mult_idx": s_mult_idx,
                "closest_point": c_pt,
            })

        self._cached_contacts = active_contacts
        self._cached_key = state_key
        return active_contacts

    def evaluate_forces(
        self,
        current_nodes: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
        n_dofs: int | None = None,
    ) -> np.ndarray:
        """
        Evaluates the global self-contact force vector f_c.
        Guarantees exact linear and angular momentum conservation: sum(f_c) == 0.
        """
        if n_dofs is None:
            n_dofs = self.dof_offset + self.domain.n_dofs

        f_c = np.zeros(n_dofs, dtype=np.float64)
        active_contacts = self._find_active_contacts(current_nodes, velocities)

        for c in active_contacts:
            s_node = c["slave_node"]
            m1, m2, m3 = c["master_nodes"]
            xi1, xi2, xi3 = c["barycentric"]
            n = c["normal"]
            f_total = c["f_N"] + c["f_damp"]
            f_t_s = c["f_t_s"]
            f_t_m = c["f_t_m"]

            # Action on slave node
            dof_s = self.dof_offset + 3 * s_node
            f_c[dof_s : dof_s + 3] += f_total * n + f_t_s

            # Reaction on master nodes
            dof_m1 = self.dof_offset + 3 * m1
            dof_m2 = self.dof_offset + 3 * m2
            dof_m3 = self.dof_offset + 3 * m3

            f_c[dof_m1 : dof_m1 + 3] += -xi1 * f_total * n + xi1 * f_t_m
            f_c[dof_m2 : dof_m2 + 3] += -xi2 * f_total * n + xi2 * f_t_m
            f_c[dof_m3 : dof_m3 + 3] += -xi3 * f_total * n + xi3 * f_t_m

        return f_c

    def evaluate_stiffness(
        self,
        current_nodes: np.ndarray | None = None,
        n_dofs: int | None = None,
        a1: float = 0.0,
    ) -> spsp.csr_matrix:
        """
        Assembles the sparse coupled symmetric tangent self-contact stiffness matrix K_c.
        K_c = sum_{active} kp * (w (x) w) (x) (n (x) n).
        """
        if n_dofs is None:
            n_dofs = self.dof_offset + self.domain.n_dofs

        active_contacts = self._find_active_contacts(current_nodes)
        if not active_contacts:
            return spsp.csr_matrix((n_dofs, n_dofs), dtype=np.float64)

        kp = self.k_penalty + a1 * self.c_damping
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for c in active_contacts:
            s_node = c["slave_node"]
            m1, m2, m3 = c["master_nodes"]
            xi1, xi2, xi3 = c["barycentric"]
            n = c["normal"]

            dof_starts = [
                self.dof_offset + 3 * s_node,
                self.dof_offset + 3 * m1,
                self.dof_offset + 3 * m2,
                self.dof_offset + 3 * m3,
            ]
            weights = [1.0, -xi1, -xi2, -xi3]
            nn = np.outer(n, n)

            # Assemble 16 blocks (4x4 node pairs)
            for a in range(4):
                w_a = weights[a]
                dof_a = dof_starts[a]
                for b in range(4):
                    w_b = weights[b]
                    dof_b = dof_starts[b]
                    block_scale = kp * w_a * w_b

                    for i in range(3):
                        for j in range(3):
                            rows.append(dof_a + i)
                            cols.append(dof_b + j)
                            vals.append(block_scale * nn[i, j])

        return spsp.coo_matrix(
            (vals, (rows, cols)), shape=(n_dofs, n_dofs), dtype=np.float64
        ).tocsr()

    def update_multipliers(self) -> dict[str, float]:
        """
        Updates Augmented Lagrangian contact multipliers:
            lambda_s <- max(0, lambda_s - k_p * g_N)
        """
        # Force re-query
        self._cached_contacts = None
        self._cached_key = None
        active_contacts = self._find_active_contacts()

        max_pen = 0.0
        num_active = 0
        max_delta_lam = 0.0

        for c in active_contacts:
            s_mult_idx = c["mult_idx"]
            if s_mult_idx < 0:
                continue

            gap = c["gap"]
            old_lam = self.multipliers[s_mult_idx]
            new_lam = float(max(0.0, old_lam - self.k_penalty * gap))
            self.multipliers[s_mult_idx] = new_lam

            delta_lam = abs(new_lam - old_lam)
            if delta_lam > max_delta_lam:
                max_delta_lam = delta_lam

            if gap < 0.0:
                pen = -gap
                if pen > max_pen:
                    max_pen = pen
            if new_lam > 0.0:
                num_active += 1

        # Clear cache so updated multipliers are used
        self._cached_contacts = None
        self._cached_key = None

        return {
            "max_penetration": max_pen,
            "max_multiplier": float(np.max(self.multipliers)) if len(self.multipliers) > 0 else 0.0,
            "max_delta_multiplier": max_delta_lam,
            "num_active": num_active,
        }

    def evaluate_potential_energy(
        self, current_nodes: np.ndarray | None = None
    ) -> float:
        """
        Calculates stored contact potential energy:
        Pi_c = 1/2 * k_p * sum_{active} g_N^2
        """
        c_info = self.get_contact_info(current_nodes)
        return float(c_info["potential_energy"])

    def get_contact_info(
        self, current_nodes: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Returns structured summary of current self-contact status."""
        active_contacts = self._find_active_contacts(current_nodes)

        num_active = 0
        max_pen = 0.0
        active_nodes = []
        contact_energy = 0.0

        for c in active_contacts:
            gap = c["gap"]
            f_N = c["f_N"]
            s_node = c["slave_node"]

            if f_N > 0.0 or gap < 0.0:
                num_active += 1
                active_nodes.append(s_node)
                if gap < 0.0:
                    pen = -gap
                    if pen > max_pen:
                        max_pen = pen
                    contact_energy += 0.5 * self.k_penalty * (gap**2)

        return {
            "num_active": num_active,
            "max_penetration": max_pen,
            "active_nodes": np.asarray(active_nodes, dtype=np.int64),
            "potential_energy": contact_energy,
            "max_multiplier": float(np.max(self.multipliers)) if len(self.multipliers) > 0 else 0.0,
        }


