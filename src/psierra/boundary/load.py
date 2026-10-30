from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import scipy.sparse as spsp

from psierra.boundary.base import NeumannBoundaryCondition


class NodalForce(NeumannBoundaryCondition):
    """
    Concentrated point force applied to specific nodes.
    """

    def __init__(
        self,
        node_indices: Sequence[int] | np.ndarray,
        force_vector: Sequence[float] | np.ndarray,
        dofs_per_node: int = 3,
    ) -> None:
        """
        Parameters:
        -----------
        node_indices : array-like
            Nodes where the force is applied.
        force_vector : (3,) array-like
            Force components [Fx, Fy, Fz] in Newtons applied to EACH selected node.
        """
        self.node_indices = np.asarray(node_indices, dtype=np.int64)
        self.force_vector = np.asarray(force_vector, dtype=np.float64)
        self.dofs_per_node = dofs_per_node

        if self.force_vector.shape != (self.dofs_per_node,):
            raise ValueError(
                f"force_vector must have shape ({self.dofs_per_node},), got {self.force_vector.shape}"
            )

    def compute_force_vector(self, total_dofs: int) -> np.ndarray:
        """Computes and returns the assembled concentrated nodal force vector."""
        f_global = np.zeros(total_dofs, dtype=np.float64)
        for i, node in enumerate(self.node_indices):
            start_dof = node * self.dofs_per_node
            f_global[start_dof : start_dof + self.dofs_per_node] += self.force_vector
        return f_global


class GravityLoad(NeumannBoundaryCondition):
    """
    Uniform volumetric body force (Gravity) computed via global mass matrix M:
    F_gravity = M @ g_global
    """

    def __init__(
        self,
        acceleration: Sequence[float] | np.ndarray = (0.0, 0.0, -9.81),
        dofs_per_node: int = 3,
    ) -> None:
        """
        Parameters:
        -----------
        acceleration : (3,) array-like, default=(0, 0, -9.81)
            Gravitational acceleration vector [gx, gy, gz] in m/s^2.
        """
        self.acceleration = np.asarray(acceleration, dtype=np.float64)
        self.dofs_per_node = dofs_per_node

    def compute_force_vector(self, mass_matrix: spsp.csr_matrix) -> np.ndarray:
        """
        Computes F_gravity = M @ g_global.

        Parameters:
        -----------
        mass_matrix : spsp.csr_matrix
            Assembled global consistent mass matrix.
        """
        total_dofs = mass_matrix.shape[0]
        n_nodes = total_dofs // self.dofs_per_node

        # Broadcast [gx, gy, gz] to all nodes
        g_global = np.tile(self.acceleration, n_nodes)
        return mass_matrix @ g_global
