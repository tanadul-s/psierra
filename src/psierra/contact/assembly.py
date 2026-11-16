"""Centralized Contact Assembly Operators for Project Sierra.

Milestone 1.0 Stage C of Loop 3 Contact Mechanics Overhaul.
-----------------------------------------------------------
This module implements the unified assembly operators for one-body and two-body
contact interactions, ensuring that:
1. Degree-of-freedom mapping is defined once and shared by all contact classes.
2. Sparse COO coordinate triplet collection and CSR conversion are centralized.
3. Action-reaction linear momentum balance (sum f_s + sum f_m = 0) and net moment
   balance (sum x (x) f = 0) are strictly preserved across multi-body and self-contact.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import scipy.sparse as spsp

from psierra.contact.data import ContactAssemblyMap


class ContactAssembler:
    """Centralized assembly engine for one-body, two-body, and self-contact stencils."""

    # =========================================================================
    # 1. Assembly Map Factory Helpers
    # =========================================================================

    @staticmethod
    def build_one_body_map(
        slave_node: int,
        dof_offset: int = 0,
        dofs_per_node: int = 3,
        contact_dofs_per_node: int = 3,
    ) -> ContactAssemblyMap:
        """Construct ContactAssemblyMap for a single-body interaction (e.g. RigidContact)."""
        sn = int(slave_node)
        s_dofs = int(dof_offset) + sn * int(dofs_per_node) + np.arange(contact_dofs_per_node, dtype=np.int64)
        return ContactAssemblyMap(
            slave_node=sn,
            slave_dofs=s_dofs,
            master_nodes=np.zeros(0, dtype=np.int64),
            master_dofs=np.zeros(0, dtype=np.int64),
            stencil_dofs=s_dofs,
            interpolation_weights=np.array([1.0], dtype=np.float64),
            slave_offset=int(dof_offset),
            master_offset=0,
        )

    @staticmethod
    def build_two_body_map(
        slave_node: int,
        master_nodes: Sequence[int] | np.ndarray,
        barycentric: Sequence[float] | np.ndarray,
        slave_offset: int = 0,
        master_offset: int = 0,
    ) -> ContactAssemblyMap:
        """Construct ContactAssemblyMap for two-body interaction (e.g. DeformableContact)."""
        sn = int(slave_node)
        mn = np.asarray(master_nodes, dtype=np.int64)
        bary = np.asarray(barycentric, dtype=np.float64)

        s_dofs = int(slave_offset) + 3 * sn + np.arange(3, dtype=np.int64)
        m_dofs = (int(master_offset) + 3 * mn[:, None] + np.arange(3, dtype=np.int64)).ravel()
        stencil_dofs = np.r_[s_dofs, m_dofs]
        weights = np.r_[1.0, -bary]

        return ContactAssemblyMap(
            slave_node=sn,
            slave_dofs=s_dofs,
            master_nodes=mn,
            master_dofs=m_dofs,
            stencil_dofs=stencil_dofs,
            interpolation_weights=weights,
            slave_offset=int(slave_offset),
            master_offset=int(master_offset),
        )

    @staticmethod
    def build_self_contact_map(
        slave_node: int,
        master_nodes: Sequence[int] | np.ndarray,
        barycentric: Sequence[float] | np.ndarray,
        dof_offset: int = 0,
    ) -> ContactAssemblyMap:
        """Construct ContactAssemblyMap for single-body self-contact interactions."""
        return ContactAssembler.build_two_body_map(
            slave_node=slave_node,
            master_nodes=master_nodes,
            barycentric=barycentric,
            slave_offset=dof_offset,
            master_offset=dof_offset,
        )

    # =========================================================================
    # 2. Vector & Sparse Matrix Global Assembly
    # =========================================================================

    @staticmethod
    def assemble_forces(
        stencils: Sequence[tuple[ContactAssemblyMap, np.ndarray]],
        total_dofs: int,
    ) -> np.ndarray:
        """Assemble global contact force vector from local stencil forces.

        Parameters:
        -----------
        stencils : Sequence[tuple[ContactAssemblyMap, np.ndarray]]
            Pairs of (assembly_map, local_force_vector).
        total_dofs : int
            Total degree-of-freedom size of the global system.

        Returns:
        --------
        global_forces : np.ndarray of shape (total_dofs,)
            Global contact force vector.
        """
        global_forces = np.zeros(total_dofs, dtype=np.float64)
        for asm_map, local_force in stencils:
            f = np.asarray(local_force, dtype=np.float64).ravel()
            if len(f) != len(asm_map.stencil_dofs):
                raise ValueError(
                    f"Local force size ({len(f)}) must match stencil DOFs count ({len(asm_map.stencil_dofs)})"
                )
            np.add.at(global_forces, asm_map.stencil_dofs, f)
        return global_forces

    @staticmethod
    def assemble_stiffness(
        stencils: Sequence[tuple[ContactAssemblyMap, np.ndarray]],
        total_dofs: int,
    ) -> spsp.csr_matrix:
        """Assemble global sparse contact tangent stiffness matrix in CSR format.

        Parameters:
        -----------
        stencils : Sequence[tuple[ContactAssemblyMap, np.ndarray]]
            Pairs of (assembly_map, local_stiffness_block).
        total_dofs : int
            Total degree-of-freedom size of the global system.

        Returns:
        --------
        K_c : spsp.csr_matrix of shape (total_dofs, total_dofs)
            Global sparse tangent stiffness matrix.
        """
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        data: list[np.ndarray] = []

        for asm_map, local_k in stencils:
            K = np.asarray(local_k, dtype=np.float64)
            dofs = asm_map.stencil_dofs
            M = len(dofs)
            if K.shape != (M, M):
                raise ValueError(f"Local stiffness shape {K.shape} must match stencil size ({M}, {M})")

            # Create grid of row and col indices for this block
            row_grid = np.repeat(dofs, M)
            col_grid = np.tile(dofs, M)

            rows.append(row_grid)
            cols.append(col_grid)
            data.append(K.ravel())

        if not rows:
            return spsp.csr_matrix((total_dofs, total_dofs), dtype=np.float64)

        all_rows = np.concatenate(rows)
        all_cols = np.concatenate(cols)
        all_data = np.concatenate(data)

        K_coo = spsp.coo_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(total_dofs, total_dofs),
            dtype=np.float64,
        )
        return K_coo.tocsr()

    # =========================================================================
    # 3. Conservation Diagnostics
    # =========================================================================

    @staticmethod
    def check_linear_momentum_balance(
        global_forces: np.ndarray,
        slave_dofs: Sequence[int] | np.ndarray,
        master_dofs: Sequence[int] | np.ndarray,
        atol: float = 1e-10,
    ) -> bool:
        """Verify action-reaction linear momentum balance: sum f_s + sum f_m = 0."""
        f = np.asarray(global_forces, dtype=np.float64)
        s_idx = np.asarray(slave_dofs, dtype=np.int64)
        m_idx = np.asarray(master_dofs, dtype=np.int64)

        f_slave_sum = np.sum(f[s_idx])
        f_master_sum = np.sum(f[m_idx])
        net_force = f_slave_sum + f_master_sum
        return bool(abs(net_force) <= atol)

    @staticmethod
    def check_angular_momentum_balance(
        global_forces: np.ndarray,
        coordinates: np.ndarray,
        dof_offset: int = 0,
        atol: float = 1e-10,
    ) -> bool:
        """Verify that interface forces exert zero net torque: sum x (x) f = 0."""
        f = np.asarray(global_forces, dtype=np.float64)
        coords = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
        n_nodes = len(coords)

        net_moment = np.zeros(3, dtype=np.float64)
        for i in range(n_nodes):
            f_i = f[dof_offset + 3 * i : dof_offset + 3 * i + 3]
            x_i = coords[i]
            net_moment += np.cross(x_i, f_i)

        return bool(np.linalg.norm(net_moment) <= atol)
