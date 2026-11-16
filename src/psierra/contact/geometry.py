"""Canonical Contact Geometry Providers for Project Sierra.

Milestone 1.0 Stage B of Loop 3 Contact Mechanics Overhaul.
-----------------------------------------------------------
This module defines the abstract ContactGeometryProvider interface and canonical
providers that evaluate the differentiable contact geometry:
    G(q) = (g_N, grad g_N, Hess g_N, x_closest, n, feature_id)

Canonical Providers:
1. PlanarGeometryProvider: Exact analytical infinite plane obstacle.
2. SphereGeometryProvider: Exact analytical sphere obstacle with curvature Hessian.
3. CylinderGeometryProvider: Exact analytical cylinder obstacle with circumferential Hessian.
4. TriangleGeometryProvider: Exact closest-feature node-to-triangle geometry with analytical 12-DOF Hessian.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

from psierra.collision.triangle import TriangleContactGeometry
from psierra.contact.data import ContactCandidate, DifferentiableContactGeometry


class ContactGeometryProvider(ABC):
    """Abstract base provider for differentiable contact geometry G(q).

    Every geometry provider must declare:
    - Physical length units (e.g. 'mm').
    - Outward normal orientation convention (e.g. 'master_outward').
    - Smooth-branch validity criteria (is_smooth_branch).
    - Evaluation of canonical DifferentiableContactGeometry.
    """

    @property
    @abstractmethod
    def units(self) -> str:
        """Physical length units, e.g. 'mm'."""

    @property
    @abstractmethod
    def orientation_convention(self) -> str:
        """Surface outward normal convention, e.g. 'master_outward'."""

    @abstractmethod
    def is_smooth_branch(
        self, candidate: ContactCandidate, coordinates: np.ndarray
    ) -> bool:
        """Return True if coordinates lie strictly within a smooth differentiable branch."""

    @abstractmethod
    def evaluate_geometry(
        self, candidate: ContactCandidate, coordinates: np.ndarray
    ) -> DifferentiableContactGeometry:
        """Evaluate canonical differentiable contact geometry G(q)."""


class PlanarGeometryProvider(ContactGeometryProvider):
    """Exact analytical planar obstacle geometry provider.

    Governing law:
        g_N = (x_s - x_0) . n
        grad_{x_s}(g_N) = n
        Hess_{x_s}(g_N) = 0_{3 x 3}
    """

    def __init__(
        self,
        origin: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
        normal: Sequence[float] | np.ndarray = (0.0, 1.0, 0.0),
        units: str = "mm",
    ) -> None:
        self._origin = np.asarray(origin, dtype=np.float64).flatten()
        n = np.asarray(normal, dtype=np.float64).flatten()
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            raise ValueError("Planar normal must have non-zero length")
        self._normal = n / norm
        self._units = str(units)

    @property
    def units(self) -> str:
        return self._units

    @property
    def orientation_convention(self) -> str:
        return "master_outward"

    @property
    def origin(self) -> np.ndarray:
        return self._origin.copy()

    @property
    def normal(self) -> np.ndarray:
        return self._normal.copy()

    def is_smooth_branch(
        self, candidate: ContactCandidate, coordinates: np.ndarray
    ) -> bool:
        """A plane is C^inf smooth everywhere."""
        return True

    def evaluate_geometry(
        self, candidate: ContactCandidate, coordinates: np.ndarray
    ) -> DifferentiableContactGeometry:
        """Evaluate planar gap, unit normal, and vanishing Hessian."""
        x_s = np.asarray(coordinates, dtype=np.float64).flatten()
        if x_s.shape != (3,):
            raise ValueError(f"Slave point must have shape (3,), got {x_s.shape}")

        diff = x_s - self._origin
        gap = float(np.dot(diff, self._normal))
        closest_point = x_s - gap * self._normal
        gradient = self._normal.copy()
        hessian = np.zeros((3, 3), dtype=np.float64)

        return DifferentiableContactGeometry(
            gap=gap,
            normal=self._normal.copy(),
            closest_point=closest_point,
            gradient=gradient,
            hessian=hessian,
            feature_id="plane",
            feature_index=0,
            barycentric=None,
            is_smooth_branch=True,
            units=self._units,
            orientation_convention=self.orientation_convention,
        )


class SphereGeometryProvider(ContactGeometryProvider):
    """Exact analytical spherical obstacle geometry provider.

    Governing law:
        d = ||x_s - c||
        g_N = d - R
        n = (x_s - c) / d
        grad_{x_s}(g_N) = n
        Hess_{x_s}(g_N) = (1 / d) * (I - n (x) n)
    """

    def __init__(
        self,
        center: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
        radius: float = 1.0,
        units: str = "mm",
    ) -> None:
        if radius <= 0.0 or not np.isfinite(radius):
            raise ValueError(f"Sphere radius must be finite and positive, got {radius}")
        self._center = np.asarray(center, dtype=np.float64).flatten()
        self._radius = float(radius)
        self._units = str(units)

    @property
    def units(self) -> str:
        return self._units

    @property
    def orientation_convention(self) -> str:
        return "master_outward"

    @property
    def center(self) -> np.ndarray:
        return self._center.copy()

    @property
    def radius(self) -> float:
        return self._radius

    def is_smooth_branch(
        self, candidate: ContactCandidate, coordinates: np.ndarray
    ) -> bool:
        """Smooth everywhere except at the exact center singularity d = 0."""
        x_s = np.asarray(coordinates, dtype=np.float64).flatten()
        return float(np.linalg.norm(x_s - self._center)) > 1e-12 * self._radius

    def evaluate_geometry(
        self, candidate: ContactCandidate, coordinates: np.ndarray
    ) -> DifferentiableContactGeometry:
        """Evaluate spherical gap, normal, and exact curvature Hessian."""
        x_s = np.asarray(coordinates, dtype=np.float64).flatten()
        if x_s.shape != (3,):
            raise ValueError(f"Slave point must have shape (3,), got {x_s.shape}")

        diff = x_s - self._center
        dist = float(np.linalg.norm(diff))

        if dist < 1e-14:
            # Singularity at center: default to +y normal
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            gap = -self._radius
            closest_point = self._center + self._radius * normal
            gradient = normal.copy()
            hessian = np.zeros((3, 3), dtype=np.float64)
            smooth = False
        else:
            normal = diff / dist
            gap = dist - self._radius
            closest_point = self._center + self._radius * normal
            gradient = normal.copy()
            hessian = (np.eye(3) - np.outer(normal, normal)) / dist
            smooth = True

        return DifferentiableContactGeometry(
            gap=gap,
            normal=normal,
            closest_point=closest_point,
            gradient=gradient,
            hessian=hessian,
            feature_id="sphere",
            feature_index=0,
            barycentric=None,
            is_smooth_branch=smooth,
            units=self._units,
            orientation_convention=self.orientation_convention,
        )


class CylinderGeometryProvider(ContactGeometryProvider):
    """Exact analytical cylinder obstacle geometry provider.

    Governing law:
        r = (I - a (x) a) (x_s - c)
        d = ||r||
        g_N = d - R
        n = r / d
        grad_{x_s}(g_N) = n
        Hess_{x_s}(g_N) = (1 / d) * (I - a (x) a - n (x) n)
    """

    def __init__(
        self,
        center: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
        axis: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0),
        radius: float = 1.0,
        height: float | None = None,
        units: str = "mm",
    ) -> None:
        if radius <= 0.0 or not np.isfinite(radius):
            raise ValueError(f"Cylinder radius must be finite and positive, got {radius}")
        self._center = np.asarray(center, dtype=np.float64).flatten()
        a = np.asarray(axis, dtype=np.float64).flatten()
        norm = np.linalg.norm(a)
        if norm < 1e-14:
            raise ValueError("Cylinder axis vector must have non-zero length")
        self._axis = a / norm
        self._radius = float(radius)
        self._height = float(height) if height is not None else None
        self._units = str(units)

    @property
    def units(self) -> str:
        return self._units

    @property
    def orientation_convention(self) -> str:
        return "master_outward"

    @property
    def center(self) -> np.ndarray:
        return self._center.copy()

    @property
    def axis(self) -> np.ndarray:
        return self._axis.copy()

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def height(self) -> float | None:
        return self._height

    def is_smooth_branch(
        self, candidate: ContactCandidate, coordinates: np.ndarray
    ) -> bool:
        """Smooth everywhere except along the central axis line d = 0."""
        x_s = np.asarray(coordinates, dtype=np.float64).flatten()
        diff = x_s - self._center
        proj = np.dot(diff, self._axis) * self._axis
        rad_dist = float(np.linalg.norm(diff - proj))
        return rad_dist > 1e-12 * self._radius

    def evaluate_geometry(
        self, candidate: ContactCandidate, coordinates: np.ndarray
    ) -> DifferentiableContactGeometry:
        """Evaluate cylinder gap, normal, and circumferential curvature Hessian."""
        x_s = np.asarray(coordinates, dtype=np.float64).flatten()
        if x_s.shape != (3,):
            raise ValueError(f"Slave point must have shape (3,), got {x_s.shape}")

        diff = x_s - self._center
        proj_length = float(np.dot(diff, self._axis))
        proj_axis_pt = self._center + proj_length * self._axis
        rad_vec = x_s - proj_axis_pt
        dist = float(np.linalg.norm(rad_vec))

        if dist < 1e-14:
            # Degenerate point on axis: pick arbitrary normal perpendicular to axis
            arbitrary = np.array([1.0, 0.0, 0.0]) if abs(self._axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            normal = arbitrary - np.dot(arbitrary, self._axis) * self._axis
            normal /= np.linalg.norm(normal)
            gap = -self._radius
            closest_point = proj_axis_pt + self._radius * normal
            gradient = normal.copy()
            hessian = np.zeros((3, 3), dtype=np.float64)
            smooth = False
        else:
            normal = rad_vec / dist
            gap = dist - self._radius
            closest_point = proj_axis_pt + self._radius * normal
            gradient = normal.copy()
            # Hessian = (1/d) * (I - a (x) a - n (x) n)
            P_transverse = np.eye(3) - np.outer(self._axis, self._axis) - np.outer(normal, normal)
            hessian = P_transverse / dist
            smooth = True

        return DifferentiableContactGeometry(
            gap=gap,
            normal=normal,
            closest_point=closest_point,
            gradient=gradient,
            hessian=hessian,
            feature_id="cylinder",
            feature_index=0,
            barycentric=None,
            is_smooth_branch=smooth,
            units=self._units,
            orientation_convention=self.orientation_convention,
        )


class TriangleGeometryProvider(ContactGeometryProvider):
    """Exact closest-feature node-to-triangle geometry provider.

    Wraps TriangleContactGeometry to evaluate Euclidean closest-point projections,
    closest-feature barycentrics, and exact 12-DOF gap gradients and Hessians.
    """

    def __init__(self, units: str = "mm") -> None:
        self._units = str(units)

    @property
    def units(self) -> str:
        return self._units

    @property
    def orientation_convention(self) -> str:
        return "master_outward"

    def is_smooth_branch(
        self, candidate: ContactCandidate, coordinates: np.ndarray
    ) -> bool:
        """True when the projection is strictly in face interior; False at ridges/edges."""
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.shape != (4, 3):
            return False
        point = coords[0]
        vertices = coords[1:]
        weights = TriangleContactGeometry.closest_barycentric(point, vertices)
        # Strictly interior face projection has all weights > 1e-6
        return bool(np.all(weights > 1e-6))

    def evaluate_geometry(
        self, candidate: ContactCandidate, coordinates: np.ndarray
    ) -> DifferentiableContactGeometry:
        """Evaluate 12-DOF gap, normal, gradient, and Hessian for node-triangle stencil.

        coordinates : np.ndarray of shape (4, 3)
            Row 0: slave node spatial position x_s.
            Rows 1-3: master triangle vertex positions [v0, v1, v2].
        """
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.shape != (4, 3):
            raise ValueError(f"Triangle geometry requires shape (4, 3), got {coords.shape}")

        point = coords[0]
        vertices = coords[1:]

        face_normal, _ = TriangleContactGeometry._face_normal(vertices)
        weights = TriangleContactGeometry.closest_barycentric(point, vertices)
        active = np.flatnonzero(weights > 0.0)

        if len(active) == 3:
            pseudonormal = face_normal
            feature_str = "face"
        elif len(active) == 2:
            pseudonormal = face_normal
            feature_str = "edge"
        else:
            pseudonormal = face_normal
            feature_str = "vertex"

        geom = TriangleContactGeometry.evaluate(
            point=point,
            vertices=vertices,
            pseudonormal=pseudonormal,
            vertex_normals=None,
        )

        closest_point = weights @ vertices
        is_smooth = len(active) == 3 and bool(np.all(weights > 1e-6))

        return DifferentiableContactGeometry(
            gap=geom.gap,
            normal=geom.normal,
            closest_point=closest_point,
            gradient=geom.gradient,
            hessian=geom.hessian,
            feature_id=feature_str,
            feature_index=int(candidate.candidate_index),
            barycentric=geom.barycentric,
            is_smooth_branch=is_smooth,
            units=self._units,
            orientation_convention=self.orientation_convention,
        )
