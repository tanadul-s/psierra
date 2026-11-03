"""
Contact Mechanics & Penalty Boundary Formulations for Project Sierra.
====================================================================
Implements non-penetration penalty contact constraints, contact velocity damping,
tangent stiffness linearization, and potential energy tracking for
deformable-to-rigid (and multi-body) interactions.
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import scipy.sparse as spsp
import vtk

from psierra.domain.base import Domain
from psierra.collision.shapes import CollisionShape
from psierra.collision.raytrace import RayMeshObstacle
from psierra.collision.detector import CollisionDetector, ContactQueryResult


class PenaltyContact:
    """
    Penalty Contact Formulation for Deformable-to-Rigid and Multi-Body Contact.

    Enforces the non-penetration condition via a virtual elastic normal spring
    at active contacting nodes:
        f_c = -k_p * g_N * n   (active when signed gap g_N < 0)
    where:
        - g_N < 0 is the penetration depth into the obstacle.
        - n is the unit normal pointing outward from the obstacle into the deformable body.
        - -g_N * n is directed away from the obstacle (repulsive).

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
        """
        # Resolve master / obstacle
        if master is None and obstacle is not None:
            master = obstacle
        # Resolve slave / domain
        if slave is None and domain is not None:
            slave = domain

        # Handle positional legacy PenaltyContact(obstacle, domain)
        if (
            isinstance(slave, (CollisionShape, RayMeshObstacle))
            and obstacle is None
            and not isinstance(master, (CollisionShape, RayMeshObstacle))
        ):
            slave, master = master, slave

        if master is None:
            raise ValueError("PenaltyContact requires a master obstacle or interacting body.")

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

        # Determine candidate slave nodes
        if slave_nodes is not None:
            self.slave_nodes = np.asarray(slave_nodes, dtype=np.int64)
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
        else:
            self.slave_nodes = None

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
            "PenaltyContact requires current_nodes or a domain with current_nodes property."
        )

    def _resolve_slave_nodes(self, all_nodes: np.ndarray) -> np.ndarray:
        if self.slave_nodes is not None:
            return self.slave_nodes
        return np.arange(len(all_nodes), dtype=np.int64)

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
        query_res = self.detector.query(nodes[s_nodes], tolerance=0.0)
        if query_res.num_active == 0:
            return f_c

        active_idx = query_res.active_indices
        active_nodes = s_nodes[active_idx]
        gaps = query_res.gaps[active_idx]  # strictly <= 0
        normals = query_res.normals[active_idx]

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
        query_res = self.detector.query(nodes[s_nodes], tolerance=0.0)

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

            # Outward normal of master triangle
            e1 = v2 - v1
            e2 = v3 - v1
            n_unnorm = np.cross(e1, e2)
            norm_mag = np.linalg.norm(n_unnorm)
            if norm_mag < 1e-14:
                continue
            n = n_unnorm / norm_mag

            # Barycentric coordinates of c_pt on triangle
            xi1, xi2, xi3 = self.compute_barycentric(c_pt, v1, v2, v3)

            # Signed normal gap (negative = penetration)
            gap = float(np.dot(x_s - c_pt, n))

            lam = self.multipliers[s_idx]
            if self.formulation == "augmented_lagrangian":
                f_N = float(max(0.0, lam - self.k_penalty * gap))
            else:
                f_N = float(-self.k_penalty * gap) if gap <= self.tolerance else 0.0

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

            e1 = v2 - v1
            e2 = v3 - v1
            n_unnorm = np.cross(e1, e2)
            norm_mag = np.linalg.norm(n_unnorm)
            if norm_mag < 1e-14:
                continue
            n = n_unnorm / norm_mag

            xi1, xi2, xi3 = self.compute_barycentric(c_pt, v1, v2, v3)
            gap = float(np.dot(x_s - c_pt, n))

            lam = self.multipliers[s_idx]
            if self.formulation == "augmented_lagrangian":
                f_N = float(max(0.0, lam - self.k_penalty * gap))
            else:
                f_N = float(-self.k_penalty * gap) if gap <= self.tolerance else 0.0

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

            e1 = v2 - v1
            e2 = v3 - v1
            n_unnorm = np.cross(e1, e2)
            norm_mag = np.linalg.norm(n_unnorm)
            if norm_mag < 1e-14:
                continue
            n = n_unnorm / norm_mag

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

            e1 = v2 - v1
            e2 = v3 - v1
            n_unnorm = np.cross(e1, e2)
            norm_mag = np.linalg.norm(n_unnorm)
            if norm_mag < 1e-14:
                continue
            n = n_unnorm / norm_mag

            gap = float(np.dot(x_s - c_pt, n))
            lam = self.multipliers[s_idx]
            if self.formulation == "augmented_lagrangian":
                f_N = float(max(0.0, lam - self.k_penalty * gap))
            else:
                f_N = float(-self.k_penalty * gap) if gap <= self.tolerance else 0.0

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


class AugmentedLagrangianContact(DeformableContact):
    """
    Augmented Lagrangian Multiplier Contact Formulation.

    Decouples penetration error from penalty stiffness by introducing
    nodal contact Lagrange multipliers lambda_N:
        f_N = max(0, lambda_N - k_p * g_N)
    driving penetration to zero (g_N -> 0) without penalty-induced matrix ill-conditioning.
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
        tolerance: float = 0.0,
    ) -> None:
        super().__init__(
            slave=slave,
            master=master,
            slave_nodes=slave_nodes,
            k_penalty=k_penalty,
            c_damping=c_damping,
            friction=friction,
            slave_offset=slave_offset,
            master_offset=master_offset,
            domain=domain,
            formulation="augmented_lagrangian",
            tolerance=tolerance,
        )

