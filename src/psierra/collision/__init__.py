"""
Collision Detection & Proximity Architecture for Project Sierra.
===============================================================
"""

from psierra.collision.shapes import (
    CollisionShape,
    RigidPlane,
    RigidSphere,
    RigidCylinder,
    InfiniteCylinder,
    CappedCylinder,
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
from psierra.collision.manifold import (
    ensure_consistent_surface_orientation,
    is_inside_closed_surface,
    compute_angle_weighted_pseudonormals,
    query_closed_surface_signed_distance,
)
from psierra.collision.ccd import (
    ContinuousCollisionResult,
    ray_triangle_intersection,
    ray_plane_intersection,
    ray_sphere_intersection,
    ray_cylinder_intersection,
    compute_time_of_impact,
    conservative_advancement_step,
)

__all__ = [
    "CollisionShape",
    "RigidPlane",
    "RigidSphere",
    "RigidCylinder",
    "InfiniteCylinder",
    "CappedCylinder",
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
    "ensure_consistent_surface_orientation",
    "is_inside_closed_surface",
    "compute_angle_weighted_pseudonormals",
    "query_closed_surface_signed_distance",
    "ContinuousCollisionResult",
    "ray_triangle_intersection",
    "ray_plane_intersection",
    "ray_sphere_intersection",
    "ray_cylinder_intersection",
    "compute_time_of_impact",
    "conservative_advancement_step",
]

