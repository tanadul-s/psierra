from __future__ import annotations


class BoundaryCondition:
    """Base class for all boundary conditions in psierra."""

    pass


class DirichletBoundaryCondition(BoundaryCondition):
    """Base class for essential/Dirichlet boundary conditions (prescribed displacements)."""

    pass


class NeumannBoundaryCondition(BoundaryCondition):
    """Base class for natural/Neumann boundary conditions (applied forces/tractions)."""

    pass
