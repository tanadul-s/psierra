from __future__ import annotations
from collections.abc import Sequence
import numpy as np

from psierra.boundary.base import DirichletBoundaryCondition


class DirichletConstraint(DirichletBoundaryCondition):
    """
    Represents a prescribed displacement constraint on a set of nodes.

    Example:
    --------
    # Clamp nodes in all 3 directions (ux = uy = uz = 0)
    fixed = DirichletConstraint(node_indices, components=(0, 1, 2), value=0.0)

    # Prescribe displacement in Z only (uz = -0.05)
    disp_z = DirichletConstraint(node_indices, components=(2,), value=-0.05)
    """

    def __init__(
        self,
        node_indices: Sequence[int] | np.ndarray,
        components: Sequence[int] = (0, 1, 2),
        value: float | np.ndarray = 0.0,
        dofs_per_node: int = 3,
    ) -> None:
        """
        Parameters:
        -----------
        node_indices : array-like of shape (N,)
            Array of node indices where the constraint is applied.
        components : tuple of int, default=(0, 1, 2)
            Local coordinate axes to constrain (0=X, 1=Y, 2=Z).
        value : float or array-like
            Prescribed displacement value(s).
        dofs_per_node : int, default=3
            Degrees of freedom per node.
        """
        self.node_indices = np.asarray(node_indices, dtype=np.int64)
        self.components = tuple(components)
        self.dofs_per_node = dofs_per_node

        # Compute the global DOF indices for this constraint
        dofs_list: list[np.ndarray] = []
        for comp in self.components:
            dofs_list.append(self.node_indices * self.dofs_per_node + comp)

        self.dofs = np.sort(np.concatenate(dofs_list))

        # Prescribed values
        if np.isscalar(value):
            self.values = np.full(len(self.dofs), fill_value=value, dtype=np.float64)
        else:
            self.values = np.asarray(value, dtype=np.float64)
            if len(self.values) != len(self.dofs):
                raise ValueError(
                    f"Length of values ({len(self.values)}) does not match number of constrained DOFs ({len(self.dofs)})."
                )


class DirichletManager:
    """
    Manages and partitions multiple Dirichlet constraints for a global system.
    """

    def __init__(self, total_dofs: int) -> None:
        self.total_dofs = total_dofs
        self._constraints: list[DirichletConstraint] = []

    def add_constraint(self, constraint: DirichletConstraint) -> None:
        """Adds a Dirichlet constraint to the manager."""
        self._constraints.append(constraint)

    def remove_constraint(self, constraint: DirichletConstraint) -> None:
        """Removes a Dirichlet constraint from the manager."""
        self._constraints.remove(constraint)

    def clear(self) -> None:
        """Clears all Dirichlet constraints from the manager."""
        self._constraints.clear()

    @property
    def prescribed_dofs(self) -> np.ndarray:
        """Sorted array of unique constrained global DOF indices."""
        if not self._constraints:
            return np.array([], dtype=np.int64)
        all_dofs = np.concatenate([c.dofs for c in self._constraints])
        return np.unique(all_dofs)

    @property
    def free_dofs(self) -> np.ndarray:
        """Sorted array of unconstrained (free) global DOF indices."""
        p_dofs = self.prescribed_dofs
        if len(p_dofs) == 0:
            return np.arange(self.total_dofs, dtype=np.int64)
        mask = np.ones(self.total_dofs, dtype=bool)
        mask[p_dofs] = False
        return np.where(mask)[0]

    @property
    def prescribed_values(self) -> np.ndarray:
        """Array of prescribed displacement values aligned with prescribed_dofs."""
        p_dofs = self.prescribed_dofs
        values = np.zeros(len(p_dofs), dtype=np.float64)

        for c in self._constraints:
            # Map constraint DOFs to the unique prescribed_dofs array
            indices = np.searchsorted(p_dofs, c.dofs)
            values[indices] = c.values

        return values
