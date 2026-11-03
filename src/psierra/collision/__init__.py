"""
Collision Detection & Proximity Architecture for Project Sierra.
===============================================================
"""

from psierra.collision.shapes import (
    CollisionShape,
    RigidPlane,
    RigidSphere,
    RigidCylinder,
)
from psierra.collision.raytrace import RayMeshObstacle
from psierra.collision.detector import CollisionDetector, ContactQueryResult

__all__ = [
    "CollisionShape",
    "RigidPlane",
    "RigidSphere",
    "RigidCylinder",
    "RayMeshObstacle",
    "CollisionDetector",
    "ContactQueryResult",
]

