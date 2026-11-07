"""
Project Sierra - High-Performance Finite Element Structural Analysis Package.
"""

from psierra.multibody.system import MultiBodySystem
from psierra.multibody import MultiDomain
from psierra.boundary.contact import (
    RigidContact,
    DeformableContact,
    SelfContact,
)
from psierra.collision import (
    CollisionShape,
    RigidPlane,
    RigidSphere,
    RigidCylinder,
    RayMeshObstacle,
    CollisionDetector,
)
from psierra.reduction import (
    PODBasis,
    SurfacePODBasis,
    SnapshotManager,
    ReducedLinearModel,
    ReducedLinearDynamicSolver,
    ReducedLinearStaticSolver,
    ReducedSolverStats,
)

__version__ = "26.10"
__all__ = [
    "MultiBodySystem",
    "MultiDomain",
    "RigidContact",
    "DeformableContact",
    "SelfContact",
    "CollisionShape",
    "RigidPlane",
    "RigidSphere",
    "RigidCylinder",
    "RayMeshObstacle",
    "CollisionDetector",
    "PODBasis",
    "SurfacePODBasis",
    "SnapshotManager",
    "ReducedLinearModel",
    "ReducedLinearDynamicSolver",
    "ReducedLinearStaticSolver",
    "ReducedSolverStats",
]
