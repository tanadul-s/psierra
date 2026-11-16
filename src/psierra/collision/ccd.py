"""
Continuous Collision Detection (CCD) & Time-of-Impact for Project Sierra.
========================================================================
Implements exact Time-of-Impact (TOI) and conservative advancement for moving
slave nodes against analytical obstacles (planes, spheres, cylinders) and triangular
surface meshes (Möller-Trumbore ray-triangle intersection) to prevent high-speed tunneling.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyvista as pv

from psierra.collision.shapes import (
    CollisionShape,
    RigidPlane,
    RigidSphere,
    RigidCylinder,
)
from psierra.collision.raytrace import RayMeshObstacle


@dataclass(frozen=True)
class ContinuousCollisionResult:
    """Result of a continuous collision detection query along a linear trajectory segment."""

    has_impact: bool
    time_of_impact: float  # Parameter t in [0.0, 1.0]; 1.0 if no impact
    impact_point: np.ndarray  # Coordinates at impact time t
    normal: np.ndarray  # Outward surface unit normal at impact
    node_index: int | None = None
    feature_id: str | None = None
    feature_index: int | None = None
    barycentric: np.ndarray | None = None


def ray_triangle_intersection(
    ray_origin: np.ndarray,
    ray_delta: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    eps: float = 1e-12,
) -> tuple[bool, float, float, float]:
    """
    Evaluates Möller-Trumbore ray-triangle intersection for trajectory x(t) = origin + t * delta (t in [0, 1]).

    Parameters:
    -----------
    ray_origin : np.ndarray of shape (3,)
        Starting position at t = 0.
    ray_delta : np.ndarray of shape (3,)
        Trajectory displacement vector (x_end - x_start).
    v0, v1, v2 : np.ndarray of shape (3,)
        Triangle vertex coordinates.
    eps : float, default=1e-12
        Tolerance for coplanar / parallel singularity checks.

    Returns:
    --------
    hit : bool
        True if the trajectory intersects the closed triangle for t in [0, 1].
    t : float
        Time-of-Impact fraction in [0, 1].
    u, v : float
        Barycentric coordinates at intersection point: x(t) = (1 - u - v)*v0 + u*v1 + v*v2.
    """
    orig = np.asarray(ray_origin, dtype=np.float64).flatten()
    delta = np.asarray(ray_delta, dtype=np.float64).flatten()
    p0 = np.asarray(v0, dtype=np.float64).flatten()
    p1 = np.asarray(v1, dtype=np.float64).flatten()
    p2 = np.asarray(v2, dtype=np.float64).flatten()

    e1 = p1 - p0
    e2 = p2 - p0
    pvec = np.cross(delta, e2)
    det = float(np.dot(e1, pvec))

    if abs(det) < eps:
        # Ray is parallel to triangle
        return False, 1.0, 0.0, 0.0

    inv_det = 1.0 / det
    tvec = orig - p0
    u = float(np.dot(tvec, pvec)) * inv_det
    if u < -eps or u > 1.0 + eps:
        return False, 1.0, 0.0, 0.0

    qvec = np.cross(tvec, e1)
    v = float(np.dot(delta, qvec)) * inv_det
    if v < -eps or u + v > 1.0 + eps:
        return False, 1.0, 0.0, 0.0

    t = float(np.dot(e2, qvec)) * inv_det
    if t < -eps or t > 1.0 + eps:
        return False, 1.0, 0.0, 0.0

    t_clamped = float(np.clip(t, 0.0, 1.0))
    u_clamped = float(np.clip(u, 0.0, 1.0))
    v_clamped = float(np.clip(v, 0.0, 1.0))
    return True, t_clamped, u_clamped, v_clamped


def ray_plane_intersection(
    ray_origin: np.ndarray,
    ray_delta: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    eps: float = 1e-12,
) -> tuple[bool, float]:
    """
    Evaluates linear trajectory intersection against an infinite rigid plane.

    Parameters:
    -----------
    ray_origin : np.ndarray of shape (3,)
        Starting point at t = 0.
    ray_delta : np.ndarray of shape (3,)
        Displacement vector.
    plane_origin : np.ndarray of shape (3,)
        Reference point on plane.
    plane_normal : np.ndarray of shape (3,)
        Outward plane unit normal vector.
    """
    orig = np.asarray(ray_origin, dtype=np.float64).flatten()
    delta = np.asarray(ray_delta, dtype=np.float64).flatten()
    p0 = np.asarray(plane_origin, dtype=np.float64).flatten()
    n = np.asarray(plane_normal, dtype=np.float64).flatten()
    n = n / np.linalg.norm(n)

    denom = float(np.dot(delta, n))
    g0 = float(np.dot(orig - p0, n))

    if abs(denom) < eps:
        # Trajectory is parallel to plane
        if abs(g0) < eps:
            return True, 0.0
        return False, 1.0

    t = -g0 / denom
    if -eps <= t <= 1.0 + eps:
        return True, float(np.clip(t, 0.0, 1.0))
    return False, 1.0


def ray_sphere_intersection(
    ray_origin: np.ndarray,
    ray_delta: np.ndarray,
    center: np.ndarray,
    radius: float,
    eps: float = 1e-12,
) -> tuple[bool, float]:
    """
    Evaluates linear trajectory intersection against a rigid sphere.
    """
    orig = np.asarray(ray_origin, dtype=np.float64).flatten()
    delta = np.asarray(ray_delta, dtype=np.float64).flatten()
    c = np.asarray(center, dtype=np.float64).flatten()
    r = float(radius)

    m = orig - c
    dist_m = float(np.linalg.norm(m))
    if dist_m <= r + eps:
        # Already inside or on boundary
        return True, 0.0

    a = float(np.dot(delta, delta))
    if a < eps:
        # Zero motion
        return False, 1.0

    b = 2.0 * float(np.dot(m, delta))
    c_val = float(np.dot(m, m)) - r * r

    disc = b * b - 4.0 * a * c_val
    if disc < 0.0:
        return False, 1.0

    sqrt_disc = float(np.sqrt(max(0.0, disc)))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # First entry root
    if -eps <= t1 <= 1.0 + eps:
        return True, float(np.clip(t1, 0.0, 1.0))
    if -eps <= t2 <= 1.0 + eps and t1 < -eps:
        # Exiting
        return True, float(np.clip(t2, 0.0, 1.0))

    return False, 1.0


def ray_cylinder_intersection(
    ray_origin: np.ndarray,
    ray_delta: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    height: float | None = None,
    capped: bool = False,
    eps: float = 1e-12,
) -> tuple[bool, float]:
    """
    Evaluates linear trajectory intersection against a cylinder obstacle.
    """
    orig = np.asarray(ray_origin, dtype=np.float64).flatten()
    delta = np.asarray(ray_delta, dtype=np.float64).flatten()
    c = np.asarray(center, dtype=np.float64).flatten()
    a_vec = np.asarray(axis, dtype=np.float64).flatten()
    a_vec = a_vec / np.linalg.norm(a_vec)
    r = float(radius)

    m = orig - c
    m_axis = float(np.dot(m, a_vec))
    d_axis = float(np.dot(delta, a_vec))

    m_perp = m - m_axis * a_vec
    d_perp = delta - d_axis * a_vec

    valid_hits: list[float] = []

    # Check barrel intersection
    A = float(np.dot(d_perp, d_perp))
    B = 2.0 * float(np.dot(m_perp, d_perp))
    C = float(np.dot(m_perp, m_perp)) - r * r

    if float(np.linalg.norm(m_perp)) <= r + eps:
        if not capped or height is None:
            return True, 0.0
        elif abs(m_axis) <= 0.5 * height + eps:
            return True, 0.0

    if A > eps:
        disc = B * B - 4.0 * A * C
        if disc >= 0.0:
            sqrt_disc = float(np.sqrt(max(0.0, disc)))
            t_barrel1 = (-B - sqrt_disc) / (2.0 * A)
            t_barrel2 = (-B + sqrt_disc) / (2.0 * A)

            for t_candidate in (t_barrel1, t_barrel2):
                if -eps <= t_candidate <= 1.0 + eps:
                    t_clamped = float(np.clip(t_candidate, 0.0, 1.0))
                    if not capped or height is None:
                        valid_hits.append(t_clamped)
                    else:
                        h = 0.5 * height
                        z_at_t = m_axis + t_clamped * d_axis
                        if abs(z_at_t) <= h + eps:
                            valid_hits.append(t_clamped)

    # Check end caps if capped
    if capped and height is not None and abs(d_axis) > eps:
        h = 0.5 * height
        for z_cap in (-h, h):
            t_cap = (z_cap - m_axis) / d_axis
            if -eps <= t_cap <= 1.0 + eps:
                t_clamped = float(np.clip(t_cap, 0.0, 1.0))
                pt_at_t = orig + t_clamped * delta
                rad_dist = float(np.linalg.norm(pt_at_t - c - float(np.dot(pt_at_t - c, a_vec)) * a_vec))
                if rad_dist <= r + eps:
                    valid_hits.append(t_clamped)

    if len(valid_hits) > 0:
        return True, min(valid_hits)
    return False, 1.0


def compute_time_of_impact(
    x_start: np.ndarray,
    x_end: np.ndarray,
    obstacle: CollisionShape | RayMeshObstacle | pv.PolyData,
) -> ContinuousCollisionResult:
    """
    Computes earliest Time-of-Impact (TOI) across all moving slave nodes against an obstacle.

    Parameters:
    -----------
    x_start : np.ndarray of shape (N, 3)
        Positions of slave nodes at beginning of step (t = 0).
    x_end : np.ndarray of shape (N, 3)
        Proposed positions of slave nodes at end of step (t = 1).
    obstacle : CollisionShape, RayMeshObstacle, or pv.PolyData
        Obstacle boundary geometry.

    Returns:
    --------
    result : ContinuousCollisionResult
        Contains earliest TOI in [0, 1], impact coordinates, and outward normal.
    """
    x0 = np.asarray(x_start, dtype=np.float64)
    x1 = np.asarray(x_end, dtype=np.float64)
    if x0.ndim == 1:
        x0 = x0.reshape(1, 3)
    if x1.ndim == 1:
        x1 = x1.reshape(1, 3)

    n_nodes = x0.shape[0]
    deltas = x1 - x0

    min_toi = 1.0
    hit_node = None
    hit_pt = np.zeros(3, dtype=np.float64)
    hit_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    has_impact = False

    if isinstance(obstacle, RigidPlane):
        for i in range(n_nodes):
            hit, t = ray_plane_intersection(x0[i], deltas[i], obstacle.origin, obstacle.normal)
            if hit and t < min_toi:
                has_impact = True
                min_toi = t
                hit_node = i
                hit_pt = x0[i] + t * deltas[i]
                hit_normal = obstacle.normal.copy()

    elif isinstance(obstacle, RigidSphere):
        for i in range(n_nodes):
            hit, t = ray_sphere_intersection(x0[i], deltas[i], obstacle.center, obstacle.radius)
            if hit and t < min_toi:
                has_impact = True
                min_toi = t
                hit_node = i
                hit_pt = x0[i] + t * deltas[i]
                norm_vec = hit_pt - obstacle.center
                norm_d = np.linalg.norm(norm_vec)
                hit_normal = norm_vec / norm_d if norm_d > 1e-12 else np.array([0.0, 1.0, 0.0])

    elif isinstance(obstacle, RigidCylinder):
        for i in range(n_nodes):
            hit, t = ray_cylinder_intersection(
                x0[i],
                deltas[i],
                obstacle.center,
                obstacle.axis,
                obstacle.radius,
                obstacle.height,
                obstacle.capped,
            )
            if hit and t < min_toi:
                has_impact = True
                min_toi = t
                hit_node = i
                hit_pt = x0[i] + t * deltas[i]
                # Outward normal at hit point
                _, norm_arr, _ = obstacle.query_points(hit_pt.reshape(1, 3))
                hit_normal = norm_arr[0]

    elif isinstance(obstacle, (RayMeshObstacle, pv.PolyData)):
        pdata = obstacle.polydata if isinstance(obstacle, RayMeshObstacle) else obstacle
        if not pdata.is_all_triangles:
            pdata = pdata.triangulate()

        pts = np.asarray(pdata.points, dtype=np.float64)
        faces = pdata.regular_faces  # (N_faces, 3)

        c_normals = pdata.cell_normals
        if c_normals is None:
            pdata = pdata.compute_normals(cell_normals=True, point_normals=False)
            c_normals = pdata.cell_normals
        cell_norms = np.asarray(c_normals, dtype=np.float64)

        for i in range(n_nodes):
            ray_d = deltas[i]
            ray_o = x0[i]
            for f_idx, face in enumerate(faces):
                hit, t, u, v = ray_triangle_intersection(
                    ray_o, ray_d, pts[face[0]], pts[face[1]], pts[face[2]]
                )
                if hit and t < min_toi:
                    has_impact = True
                    min_toi = t
                    hit_node = i
                    hit_pt = ray_o + t * ray_d
                    hit_normal = cell_norms[f_idx].copy()

    return ContinuousCollisionResult(
        has_impact=has_impact,
        time_of_impact=min_toi,
        impact_point=hit_pt,
        normal=hit_normal,
        node_index=hit_node,
    )


def conservative_advancement_step(
    x_start: np.ndarray,
    x_end: np.ndarray,
    obstacle: Any,
    safety_factor: float = 0.9,
) -> tuple[float, np.ndarray, ContinuousCollisionResult]:
    """
    Calculates the safe advance fraction alpha in (0, 1] and updated node coordinates
    to prevent surface penetration and high-speed tunneling.

    Parameters:
    -----------
    x_start : np.ndarray of shape (N, 3)
        Node coordinates at step start.
    x_end : np.ndarray of shape (N, 3)
        Proposed node coordinates at step end.
    obstacle : CollisionShape, RayMeshObstacle, or pv.PolyData
        Collision obstacle.
    safety_factor : float, default=0.9
        Safety scale applied to Time-of-Impact (e.g. 0.9 advances to 90% of distance to impact).

    Returns:
    --------
    step_fraction : float
        Advance fraction alpha in (0, 1].
    x_advanced : np.ndarray of shape (N, 3)
        Advanced node coordinates: x_start + alpha * (x_end - x_start).
    result : ContinuousCollisionResult
        Detailed collision query result.
    """
    res = compute_time_of_impact(x_start, x_end, obstacle)

    if res.has_impact:
        alpha = float(np.clip(safety_factor * res.time_of_impact, 0.0, 1.0))
    else:
        alpha = 1.0

    x0 = np.asarray(x_start, dtype=np.float64)
    x1 = np.asarray(x_end, dtype=np.float64)
    x_advanced = x0 + alpha * (x1 - x0)

    return alpha, x_advanced, res
