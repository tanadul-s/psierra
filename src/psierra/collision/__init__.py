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
from psierra.collision.mortar import (
    sutherland_hodgman_clip_2d,
    triangulate_convex_polygon_2d,
    polygon_area_2d,
    dunavant_triangle_quadrature_3pt,
    eval_dual_shape_functions,
    MortarSegmentData,
    MortarInterfaceIntegrator,
)

__all__ = [
    "CollisionShape",
    "RigidPlane",
    "RigidSphere",
    "RigidCylinder",
    "RayMeshObstacle",
    "CollisionDetector",
    "ContactQueryResult",
    "sutherland_hodgman_clip_2d",
    "triangulate_convex_polygon_2d",
    "polygon_area_2d",
    "dunavant_triangle_quadrature_3pt",
    "eval_dual_shape_functions",
    "MortarSegmentData",
    "MortarInterfaceIntegrator",
]

