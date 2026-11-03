"""
Project Sierra - High-Performance Finite Element Structural Analysis Package.
"""

from psierra.multibody.system import MultiBodySystem
from psierra.multibody import MultiDomain
from psierra.boundary.contact import (
    RigidContact,
    DeformableContact,
)
from psierra.collision import (
    CollisionShape,
    RigidPlane,
    RigidSphere,
    RigidCylinder,
    RayMeshObstacle,
    CollisionDetector,
)

__version__ = "26.10"
__all__ = [
    "MultiBodySystem",
    "MultiDomain",
    "RigidContact",
    "DeformableContact",
    "CollisionShape",
    "RigidPlane",
    "RigidSphere",
    "RigidCylinder",
    "RayMeshObstacle",
    "CollisionDetector",
]
