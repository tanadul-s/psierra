"""
Analytical Collision Shapes for Project Sierra.
==============================================
Provides geometric primitives (planes, spheres, cylinders) for fast, exact
deformable-to-rigid contact and collision detection.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pyvista as pv


class CollisionShape(ABC):
    """
    Abstract base class for analytical and mesh-based collision obstacles.
    """

    @abstractmethod
    def query_points(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates signed normal gaps, surface outward normals, and closest points
        for a cloud of candidate query points.

        Parameters:
        -----------
        points : np.ndarray of shape (N, 3)
            Spatial coordinates of query points.

        Returns:
        --------
        gaps : np.ndarray of shape (N,)
            Signed distance from obstacle surface.
            g < 0 indicates penetration into the obstacle.
            g >= 0 indicates separation (outside the obstacle).
        normals : np.ndarray of shape (N, 3)
            Outward unit normal vectors pointing from obstacle into the deformable body.
        closest_points : np.ndarray of shape (N, 3)
            Orthogonal projection of each query point onto the obstacle boundary surface.
        """
        pass

    @abstractmethod
    def to_pyvista(self, **kwargs) -> pv.PolyData:
        """Generates a PyVista PolyData surface representation for visualization."""
        pass


class RigidPlane(CollisionShape):
    """
    Rigid Infinite Plane Obstacle.

    Defined by a point on the plane (origin) and an outward normal vector.
    A point x has signed gap g = (x - origin) . normal.
    Points on the side of normal have g > 0. Points behind the plane (penetrating)
    have g < 0.
    """

    def __init__(
        self,
        origin: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
        normal: Sequence[float] | np.ndarray = (0.0, 1.0, 0.0),
    ) -> None:
        self.origin = np.asarray(origin, dtype=np.float64).flatten()
        n = np.asarray(normal, dtype=np.float64).flatten()
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            raise ValueError("Normal vector must have non-zero length.")
        self.normal = n / norm

    def query_points(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected points of shape (N, 3), got {pts.shape}")

        n_pts = pts.shape[0]
        # g = (x - x0) . n
        diff = pts - self.origin
        gaps = np.dot(diff, self.normal)  # shape (N,)
        normals = np.tile(self.normal, (n_pts, 1))  # shape (N, 3)
        closest_points = pts - gaps[:, None] * normals
        return gaps, normals, closest_points

    def to_pyvista(
        self,
        x_range: tuple[float, float] = (-100.0, 100.0),
        z_range: tuple[float, float] = (-100.0, 100.0),
        **kwargs,
    ) -> pv.PolyData:
        """Generates a planar rectangular patch oriented along plane normal."""
        # Standard horizontal plane if normal is aligned with Y
        plane = pv.Plane(
            center=self.origin,
            direction=self.normal,
            i_size=x_range[1] - x_range[0],
            j_size=z_range[1] - z_range[0],
            i_resolution=10,
            j_resolution=10,
        )
        return plane


class RigidSphere(CollisionShape):
    """
    Rigid Sphere Obstacle.

    Defined by center c and radius R.
    For any query point x, distance d = ||x - c||.
    Gap g = d - R. When d < R, g < 0 (penetration inside sphere).
    Outward unit normal n = (x - c) / d points outward from sphere center.
    """

    def __init__(
        self,
        center: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
        radius: float = 1.0,
    ) -> None:
        if radius <= 0.0:
            raise ValueError(f"Sphere radius must be positive, got {radius}")
        self.center = np.asarray(center, dtype=np.float64).flatten()
        self.radius = float(radius)

    def query_points(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected points of shape (N, 3), got {pts.shape}")

        diff = pts - self.center
        dists = np.linalg.norm(diff, axis=1)

        # Avoid zero division for point exactly at center
        safe_dists = np.where(dists < 1e-14, 1.0, dists)
        normals = diff / safe_dists[:, None]
        # For points at center, assign arbitrary outward normal
        center_mask = dists < 1e-14
        if np.any(center_mask):
            normals[center_mask] = np.array([0.0, 1.0, 0.0])

        gaps = dists - self.radius
        closest_points = self.center + self.radius * normals
        return gaps, normals, closest_points

    def to_pyvista(self, **kwargs) -> pv.PolyData:
        return pv.Sphere(
            center=self.center,
            radius=self.radius,
            theta_resolution=30,
            phi_resolution=30,
        )


class RigidCylinder(CollisionShape):
    """
    Rigid Cylinder Obstacle.

    Defined by a center point c, axis unit vector a, and radius R.
    Radial distance d = ||(x - c) - ((x - c) . a) a||.
    Gap g = d - R.
    """

    def __init__(
        self,
        center: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
        axis: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0),
        radius: float = 1.0,
        height: float = 100.0,
    ) -> None:
        if radius <= 0.0:
            raise ValueError(f"Cylinder radius must be positive, got {radius}")
        self.center = np.asarray(center, dtype=np.float64).flatten()
        a = np.asarray(axis, dtype=np.float64).flatten()
        norm = np.linalg.norm(a)
        if norm < 1e-14:
            raise ValueError("Axis vector must have non-zero length.")
        self.axis = a / norm
        self.radius = float(radius)
        self.height = float(height)

    def query_points(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = np.asarray(points, dtype=np.float64)
        diff = pts - self.center
        proj_length = np.dot(diff, self.axis)
        proj_points = self.center + proj_length[:, None] * self.axis

        rad_vec = pts - proj_points
        dists = np.linalg.norm(rad_vec, axis=1)

        safe_dists = np.where(dists < 1e-14, 1.0, dists)
        normals = rad_vec / safe_dists[:, None]
        center_mask = dists < 1e-14
        if np.any(center_mask):
            normals[center_mask] = np.array([0.0, 1.0, 0.0])

        gaps = dists - self.radius
        closest_points = proj_points + self.radius * normals
        return gaps, normals, closest_points

    def to_pyvista(self, **kwargs) -> pv.PolyData:
        return pv.Cylinder(
            center=self.center,
            direction=self.axis,
            radius=self.radius,
            height=self.height,
            resolution=30,
        )

