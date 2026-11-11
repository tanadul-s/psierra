"""Segment-to-Segment Mortar Contact Formulation for Project Sierra.

Implements weak-form interface contact mechanics with Wohlmuth biorthogonal
dual shape functions (Wohlmuth 2000, Popp et al. 2009, Puso & Laursen 2004).
Guarantees exact satisfaction of the contact patch test across non-conforming
mesh interfaces, exact action-reaction balance, and symmetric positive
semi-definite tangent stiffness coupling.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import scipy.sparse as spsp

from psierra.domain.base import Domain
from psierra.boundary.contact import ContactMultiplierState
from psierra.boundary.friction import (
    StickSlipState,
    CoulombFrictionState,
    CoulombFrictionModel,
)
from psierra.collision.mortar import (
    MortarInterfaceIntegrator,
    MortarSegmentData,
)


class MortarContact(ContactMultiplierState):
    """Segment-to-Segment Mortar Contact with Biorthogonal Dual Shape Functions.

    Enforces interface non-penetration constraints in a weak variational sense
    over interacting surface facet pairs:
        g_i = int_{Gamma_c} Phi_i(xi) * g_n(x) dGamma / D_{ii}
    where Phi_i are biorthogonal dual shape functions yielding a strictly diagonal
    slave mass matrix D.

    Supports both Penalty and Augmented Lagrangian formulations, two-way
    deformable-to-deformable coupling, and deformable-to-rigid interactions.
    Preserves exact action-reaction balance (sum f_s + sum f_m = 0) and
    symmetric tangent stiffness.
    """

    def __init__(
        self,
        slave: Domain,
        master: Domain | Any,
        slave_nodes: Sequence[int] | np.ndarray | None = None,
        k_penalty: float = 1e4,
        k_tangential: float | None = None,
        c_damping: float = 0.0,
        friction: float = 0.0,
        slave_offset: int = 0,
        master_offset: int = 0,
        dof_offset: int = 0,
        formulation: Literal["penalty", "augmented_lagrangian"] = "penalty",
        search_radius: float = 10.0,
        tolerance: float = 0.0,
        symmetric_tangent: bool = False,
        domain: Domain | None = None,
    ) -> None:
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

        self.slave = slave
        self.master = master
        self.k_penalty = float(k_penalty)
        self.k_tangential = float(k_tangential if k_tangential is not None else k_penalty)
        self.c_damping = float(c_damping)
        self.friction = float(friction)
        self.slave_offset = int(slave_offset or dof_offset)
        self.master_offset = int(master_offset)
        self.dof_offset = self.slave_offset
        self.formulation = formulation
        self.search_radius = float(search_radius)
        self.tolerance = float(tolerance)
        self.symmetric_tangent = bool(symmetric_tangent)
        self.domain = domain

        # Resolve slave surface geometry
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

        if not hasattr(self.slave, "mesh") or not hasattr(self.slave.mesh, "surface"):
            raise TypeError("Slave domain must possess a surface mesh representation.")

        slave_surf = self.slave.mesh.surface
        slave_pd = slave_surf.polydata
        if not slave_pd.is_all_triangles:
            slave_pd = slave_pd.triangulate()
        self._slave_faces = slave_surf.node_indices[slave_pd.regular_faces]

        # Resolve master surface geometry
        self.is_deformable_master = (
            hasattr(self.master, "mesh")
            and hasattr(self.master.mesh, "surface")
            and hasattr(self.master, "n_dofs")
        )

        if self.is_deformable_master:
            master_surf = self.master.mesh.surface
            master_pd = master_surf.polydata
            if not master_pd.is_all_triangles:
                master_pd = master_pd.triangulate()
            self._master_faces = master_surf.node_indices[master_pd.regular_faces]
            self.master_nodes = np.asarray(master_surf.node_indices, dtype=np.int64)
        elif hasattr(self.master, "surface"):
            master_pd = self.master.surface
            if not master_pd.is_all_triangles:
                master_pd = master_pd.triangulate()
            self._master_faces = master_pd.regular_faces
            self.master_nodes = np.arange(master_pd.n_points, dtype=np.int64)
        else:
            raise TypeError("Master domain must possess a surface mesh or polydata.")

        # Integrator initialization
        self.integrator = MortarInterfaceIntegrator(
            slave_nodes=self.slave.current_nodes,
            slave_faces=self._slave_faces,
            master_nodes=self.master.current_nodes,
            master_faces=self._master_faces,
            search_radius=self.search_radius,
        )

        # Multipliers aligned with slave_nodes
        self.multipliers = np.zeros(len(self.slave_nodes), dtype=np.float64)
        self.friction_state = CoulombFrictionState(len(self.slave_nodes))
        self._slave_node_to_cand = {int(node): idx for idx, node in enumerate(self.slave_nodes)}

        # Cached query state
        self._cached_D: np.ndarray | None = None
        self._cached_M: spsp.csr_matrix | None = None
        self._cached_weighted_gaps: np.ndarray | None = None
        self._cached_segments: list[MortarSegmentData] | None = None
        self._cached_average_gaps: np.ndarray | None = None

    def invalidate_state(self) -> None:
        """Clear cached integration matrices upon multiplier or state changes."""
        self._cached_D = None
        self._cached_M = None
        self._cached_weighted_gaps = None
        self._cached_segments = None
        self._cached_average_gaps = None

    def _integrate_interface(
        self,
        current_slave_coords: np.ndarray | None = None,
        current_master_coords: np.ndarray | None = None,
    ) -> tuple[np.ndarray, spsp.csr_matrix, np.ndarray, list[MortarSegmentData]]:
        """Run or return cached mortar interface integration."""
        s_coords = (
            self.slave.current_nodes
            if current_slave_coords is None
            else current_slave_coords
        )
        m_coords = (
            self.master.current_nodes
            if current_master_coords is None
            else current_master_coords
        )

        D_diag, M_csr, weighted_gaps, segments = self.integrator.integrate(
            current_slave_coords=s_coords,
            current_master_coords=m_coords,
        )
        self._cached_M = M_csr
        self._cached_weighted_gaps = weighted_gaps
        self._cached_segments = segments

        # Compute active contact area D_active from M_csr row sum
        D_active = np.array(M_csr.sum(axis=1)).ravel()
        active_mask = D_active > 1e-14
        self._cached_D = np.where(active_mask, D_active, D_diag)

        avg_gaps = np.full(len(D_active), np.inf, dtype=np.float64)
        avg_gaps[active_mask] = weighted_gaps[active_mask] / D_active[active_mask]
        self._cached_average_gaps = avg_gaps

        return self._cached_D, M_csr, weighted_gaps, segments

    def evaluate_gaps(self, current_nodes: np.ndarray | None = None) -> np.ndarray:
        """Return candidate-aligned signed normal gaps [length]."""
        self._integrate_interface()
        assert self._cached_average_gaps is not None
        # Return gaps aligned with self.slave_nodes
        return self._cached_average_gaps[self.slave_nodes].copy()

    def _resolve_n_dofs(self, n_dofs: int | None) -> int:
        """Resolve global translational vector size from body offsets or assembly."""
        if n_dofs is not None:
            return n_dofs
        if self.domain is not None:
            return self.domain.n_dofs
        if self.is_deformable_master:
            return max(
                self.slave_offset + self.slave.n_dofs,
                self.master_offset + self.master.n_dofs,
            )
        return self.slave_offset + self.slave.n_dofs

    def evaluate_forces(
        self,
        current_nodes: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
        n_dofs: int | None = None,
    ) -> np.ndarray:
        """Assemble conservative and weakly-coupled interface contact forces."""
        D_diag, M_csr, weighted_gaps, segments = self._integrate_interface()
        assert self._cached_average_gaps is not None
        avg_gaps = self._cached_average_gaps

        total_dofs = self._resolve_n_dofs(n_dofs)
        forces = np.zeros(total_dofs, dtype=np.float64)

        if len(segments) == 0:
            return forces

        # Build nodal multipliers map
        multipliers_by_node = np.zeros(len(D_diag), dtype=np.float64)
        if self.formulation == "augmented_lagrangian":
            multipliers_by_node[self.slave_nodes] = self.multipliers

        # Nodal contact pressure P_i = max(0, lambda_i - kp * bar_g_i) [force]
        # Active only where penetration occurs (bar_g < 0) or multiplier > 0
        pressures = np.zeros(len(D_diag), dtype=np.float64)
        active_nodes = np.flatnonzero(np.isfinite(avg_gaps))
        for node in active_nodes:
            g = avg_gaps[node]
            lam = multipliers_by_node[node]
            p = max(0.0, lam - self.k_penalty * g)
            pressures[node] = p

        # Velocity damping
        vel = (
            None
            if velocities is None
            else np.asarray(velocities, dtype=np.float64).reshape(-1)
        )

        # Average normal vector per slave node from incident active segments
        node_normals = np.zeros((len(D_diag), 3), dtype=np.float64)
        node_counts = np.zeros(len(D_diag), dtype=np.float64)
        for seg in segments:
            # seg.normal is master outward normal; contact force pushes slave along +n_m
            # Direction pushing slave out of master: n_push = seg.normal
            n_push = seg.normal
            for snode in seg.slave_nodes:
                node_normals[snode] += n_push
                node_counts[snode] += 1.0

        valid_norm_mask = node_counts > 0
        node_normals[valid_norm_mask] /= node_counts[valid_norm_mask, None]
        norm_len = np.linalg.norm(node_normals, axis=1, keepdims=True)
        node_normals = np.where(norm_len > 1e-14, node_normals / np.maximum(norm_len, 1e-14), 0.0)

        # 1. Slave forces: f_s_i = P_i * n_push + f_T_s_i
        active_p_nodes = np.flatnonzero(pressures > 0.0)
        slave_f_T = np.zeros((len(D_diag), 3), dtype=np.float64)

        if self.friction > 0.0:
            if self.friction_state is None or self.friction_state.n_candidates != len(self.slave_nodes):
                self.friction_state = CoulombFrictionState(len(self.slave_nodes))
                self._slave_node_to_cand = {int(node): idx for idx, node in enumerate(self.slave_nodes)}

            active_cand_set = set()
            for snode in active_p_nodes:
                cand_idx = self._slave_node_to_cand.get(int(snode))
                if cand_idx is not None:
                    active_cand_set.add(cand_idx)

            inact_mask = np.ones(len(self.slave_nodes), dtype=bool)
            if active_cand_set:
                inact_mask[list(active_cand_set)] = False
            self.friction_state.anchors[inact_mask] = np.nan
            self.friction_state.states[inact_mask] = StickSlipState.INACTIVE
            self.friction_state.forces[inact_mask] = 0.0
            self.friction_state.multipliers[inact_mask] = 0.0

            for snode in active_p_nodes:
                cand_idx = self._slave_node_to_cand.get(int(snode))
                if cand_idx is None:
                    continue
                p = pressures[snode]
                n_dir = node_normals[snode]

                x_s = self.slave.current_nodes[snode]
                x_m = np.zeros(3, dtype=np.float64)
                if self.is_deformable_master:
                    m_row = M_csr.getrow(snode)
                    if m_row.nnz > 0:
                        for col_idx, col in enumerate(m_row.indices):
                            w = m_row.data[col_idx] / D_diag[snode]
                            x_m += w * self.master.current_nodes[col]
                x_rel = x_s - x_m

                anchor = self.friction_state.anchors[cand_idx]
                if np.isnan(anchor[0]):
                    anchor = x_rel.copy()
                    self.friction_state.anchors[cand_idx] = anchor

                delta_u_rel = x_rel - anchor
                lam_T = self.friction_state.multipliers[cand_idx]

                f_T, _, _, state, s_dir = CoulombFrictionModel.return_mapping(
                    p_N=p,
                    normal=n_dir,
                    delta_u_rel=delta_u_rel,
                    mu=self.friction,
                    k_T=self.k_tangential,
                    k_N=self.k_penalty,
                    lambda_T=lam_T,
                    symmetric_tangent=self.symmetric_tangent,
                )

                self.friction_state.states[cand_idx] = state
                self.friction_state.forces[cand_idx] = f_T
                self.friction_state.slip_dirs[cand_idx] = s_dir
                slave_f_T[snode] = f_T

                if state == StickSlipState.SLIP:
                    elastic_slip = -(self.friction * p / self.k_tangential) * s_dir
                    self.friction_state.trial_anchors[cand_idx] = x_rel - elastic_slip
                    self.friction_state.trial_slip_increments[cand_idx] = float(np.linalg.norm(delta_u_rel - elastic_slip))

        for snode in active_p_nodes:
            p = pressures[snode]
            n_dir = node_normals[snode]
            dof_idx = self.slave_offset + 3 * snode
            f_local = p * n_dir + slave_f_T[snode]

            # Normal velocity damping
            if vel is not None and self.c_damping > 0.0:
                v_s = vel[dof_idx : dof_idx + 3]
                # Master velocity at snode via M
                v_m_avg = np.zeros(3, dtype=np.float64)
                if self.is_deformable_master:
                    m_row = M_csr.getrow(snode)
                    if m_row.nnz > 0:
                        m_cols = m_row.indices
                        m_data = m_row.data
                        for col_idx, col in enumerate(m_cols):
                            w = m_data[col_idx] / D_diag[snode]
                            m_dof = self.master_offset + 3 * col
                            v_m_avg += w * vel[m_dof : m_dof + 3]
                v_rel = v_s - v_m_avg
                v_n = float(v_rel @ n_dir)
                if v_n < 0.0:  # Moving into penetration
                    f_local += -self.c_damping * v_n * n_dir

            forces[dof_idx : dof_idx + 3] += f_local

        # 2. Master forces (two-way deformable): f_m_k = - sum_i (M_ik / D_ii) * (P_i * n_dir + f_T_s_i)
        if self.is_deformable_master and len(active_p_nodes) > 0:
            q = pressures / D_diag
            for c in range(3):
                # Vectorized normal + friction force transmission
                comp_vec = q * node_normals[:, c]
                if self.friction > 0.0:
                    comp_vec += slave_f_T[:, c] / D_diag
                f_m_c = -M_csr.T.dot(comp_vec)
                m_dofs = self.master_offset + 3 * np.arange(len(f_m_c)) + c
                valid_m_dofs = m_dofs[m_dofs < total_dofs]
                forces[valid_m_dofs] += f_m_c[m_dofs < total_dofs]

        return forces

    def evaluate_stiffness(
        self,
        current_nodes: np.ndarray | None = None,
        n_dofs: int | None = None,
        a1: float = 0.0,
    ) -> spsp.csr_matrix:
        """Assemble symmetric mortar contact tangent stiffness matrix K_c."""
        D_diag, M_csr, weighted_gaps, segments = self._integrate_interface()
        assert self._cached_average_gaps is not None
        avg_gaps = self._cached_average_gaps

        total_dofs = self._resolve_n_dofs(n_dofs)
        if len(segments) == 0:
            return spsp.csr_matrix((total_dofs, total_dofs), dtype=np.float64)

        # Identify active nodes
        multipliers_by_node = np.zeros(len(D_diag), dtype=np.float64)
        if self.formulation == "augmented_lagrangian":
            multipliers_by_node[self.slave_nodes] = self.multipliers

        pressures = np.zeros(len(D_diag), dtype=np.float64)
        active_nodes = np.flatnonzero(np.isfinite(avg_gaps))
        for node in active_nodes:
            g = avg_gaps[node]
            lam = multipliers_by_node[node]
            p = max(0.0, lam - self.k_penalty * g)
            pressures[node] = p

        active_p_nodes = np.flatnonzero(pressures > 0.0)
        if len(active_p_nodes) == 0:
            return spsp.csr_matrix((total_dofs, total_dofs), dtype=np.float64)

        # Average normals
        node_normals = np.zeros((len(D_diag), 3), dtype=np.float64)
        node_counts = np.zeros(len(D_diag), dtype=np.float64)
        for seg in segments:
            n_push = seg.normal
            for snode in seg.slave_nodes:
                node_normals[snode] += n_push
                node_counts[snode] += 1.0

        valid_norm_mask = node_counts > 0
        node_normals[valid_norm_mask] /= node_counts[valid_norm_mask, None]
        norm_len = np.linalg.norm(node_normals, axis=1, keepdims=True)
        node_normals = np.where(norm_len > 1e-14, node_normals / np.maximum(norm_len, 1e-14), 0.0)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        eff_kp = self.k_penalty + a1 * self.c_damping

        # Precompute per-node 3x3 stiffness blocks (normal + friction)
        node_blocks: dict[int, np.ndarray] = {}
        for snode in active_p_nodes:
            n_dir = node_normals[snode]
            nnT = np.outer(n_dir, n_dir) * eff_kp
            if self.friction > 0.0 and self.friction_state is not None:
                cand_idx = self._slave_node_to_cand.get(int(snode))
                if cand_idx is not None:
                    anchor = self.friction_state.anchors[cand_idx]
                    x_s = self.slave.current_nodes[snode]
                    x_m = np.zeros(3, dtype=np.float64)
                    if self.is_deformable_master:
                        m_row = M_csr.getrow(snode)
                        if m_row.nnz > 0:
                            for col_idx, col in enumerate(m_row.indices):
                                w = m_row.data[col_idx] / D_diag[snode]
                                x_m += w * self.master.current_nodes[col]
                    x_rel = x_s - x_m
                    delta_u_rel = (x_rel - anchor) if not np.isnan(anchor[0]) else np.zeros(3, dtype=np.float64)
                    lam_T = self.friction_state.multipliers[cand_idx]
                    p = pressures[snode]
                    _, K_TT, K_TN, _, _ = CoulombFrictionModel.return_mapping(
                        p_N=p,
                        normal=n_dir,
                        delta_u_rel=delta_u_rel,
                        mu=self.friction,
                        k_T=self.k_tangential,
                        k_N=self.k_penalty,
                        lambda_T=lam_T,
                        symmetric_tangent=self.symmetric_tangent,
                    )
                    k_block = nnT + K_TT + K_TN
                else:
                    k_block = nnT
            else:
                k_block = nnT
            node_blocks[int(snode)] = k_block

        # 1. Slave-Slave stiffness K_ss
        for snode in active_p_nodes:
            k_block = node_blocks[int(snode)]
            base_s = self.slave_offset + 3 * snode
            for r in range(3):
                for c in range(3):
                    rows.append(base_s + r)
                    cols.append(base_s + c)
                    vals.append(float(k_block[r, c]))

        # 2. Cross blocks K_sm, K_ms, K_mm (two-way deformable)
        if self.is_deformable_master:
            for snode in active_p_nodes:
                k_block = node_blocks[int(snode)]
                base_s = self.slave_offset + 3 * snode

                m_row = M_csr.getrow(snode)
                if m_row.nnz == 0:
                    continue

                m_cols = m_row.indices
                m_data = m_row.data
                d_inv = 1.0 / D_diag[snode]

                # K_sm and K_ms
                for idx_k, k in enumerate(m_cols):
                    w_k = m_data[idx_k] * d_inv
                    base_m = self.master_offset + 3 * k
                    block_sm = -w_k * k_block
                    for r in range(3):
                        for c in range(3):
                            # K_sm
                            rows.append(base_s + r)
                            cols.append(base_m + c)
                            vals.append(float(block_sm[r, c]))
                            # K_ms
                            rows.append(base_m + c)
                            cols.append(base_s + r)
                            vals.append(float(block_sm[r, c]))

                # K_mm: sum over slave nodes of (M_ik * M_il / D_ii^2) * k_block
                for idx_k, k in enumerate(m_cols):
                    w_k = m_data[idx_k] * d_inv
                    base_m_k = self.master_offset + 3 * k
                    for idx_l, l in enumerate(m_cols):
                        w_l = m_data[idx_l] * d_inv
                        base_m_l = self.master_offset + 3 * l
                        block_mm = (w_k * w_l) * k_block
                        for r in range(3):
                            for c in range(3):
                                rows.append(base_m_k + r)
                                cols.append(base_m_l + c)
                                vals.append(float(block_mm[r, c]))

        K_coo = spsp.coo_matrix(
            (vals, (rows, cols)), shape=(total_dofs, total_dofs), dtype=np.float64
        )
        return K_coo.tocsr()

    def evaluate_potential_energy(
        self, current_nodes: np.ndarray | None = None
    ) -> float:
        """Return scalar contact potential energy [force*length]."""
        return float(self.get_contact_info(current_nodes)["potential_energy"])

    def get_contact_info(
        self, current_nodes: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Report active nodes, maximum penetration, maximum force, and energy."""
        self._integrate_interface()
        assert self._cached_average_gaps is not None
        avg_gaps = self._cached_average_gaps[self.slave_nodes]

        multipliers = (
            self.multipliers
            if self.formulation == "augmented_lagrangian"
            else np.zeros_like(avg_gaps)
        )
        pressures = np.maximum(0.0, multipliers - self.k_penalty * avg_gaps)

        # Potential energy: sum((P^2 - lambda^2) / (2 * kp))
        if self.formulation == "augmented_lagrangian":
            energy = 0.5 / self.k_penalty * np.sum(pressures**2 - multipliers**2)
        else:
            energy = 0.5 * self.k_penalty * np.sum(np.maximum(-avg_gaps, 0.0)**2)

        info: dict[str, Any] = {
            "num_active": int(np.count_nonzero(pressures > 0.0)),
            "max_penetration": float(
                np.max(np.maximum(-avg_gaps, 0.0), initial=0.0)
            ),
            "max_force": float(np.max(pressures, initial=0.0)),
            "potential_energy": float(energy),
        }
        if self.friction_state is not None:
            info["num_stick"] = int(np.count_nonzero(self.friction_state.states == StickSlipState.STICK))
            info["num_slip"] = int(np.count_nonzero(self.friction_state.states == StickSlipState.SLIP))
            info["max_friction_force"] = float(np.max(np.linalg.norm(self.friction_state.forces, axis=1), initial=0.0))
            info["max_slip_distance"] = float(np.max(self.friction_state.slip_distances, initial=0.0))
        return info

    def commit_state(self) -> None:
        """Commits converged contact friction state at the end of a load/time step."""
        if (
            hasattr(self, "friction")
            and self.friction > 0.0
            and hasattr(self, "friction_state")
            and self.friction_state is not None
        ):
            self.friction_state.commit_step()
