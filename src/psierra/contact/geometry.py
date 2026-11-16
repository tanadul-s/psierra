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
        capped: bool = False,
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
        self._capped = bool(capped)
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

    @property
    def capped(self) -> bool:
        return self._capped

    def is_smooth_branch(
        self, candidate: ContactCandidate, coordinates: np.ndarray
    ) -> bool:
        """Smooth everywhere except along the central axis line d = 0 or corner rims."""
        x_s = np.asarray(coordinates, dtype=np.float64).flatten()
        diff = x_s - self._center
        proj = np.dot(diff, self._axis) * self._axis
        rad_dist = float(np.linalg.norm(diff - proj))
        if rad_dist <= 1e-12 * self._radius:
            return False
        if self._capped and self._height is not None:
            z = float(np.dot(diff, self._axis))
            h = 0.5 * self._height
            # Corner rim singularity check
            if abs(abs(z) - h) < 1e-12 * h and abs(rad_dist - self._radius) < 1e-12 * self._radius:
                return False
        return True

    def evaluate_geometry(
        self, candidate: ContactCandidate, coordinates: np.ndarray
    ) -> DifferentiableContactGeometry:
        """Evaluate cylinder gap, normal, and exact curvature Hessian."""
        x_s = np.asarray(coordinates, dtype=np.float64).flatten()
        if x_s.shape != (3,):
            raise ValueError(f"Slave point must have shape (3,), got {x_s.shape}")

        diff = x_s - self._center
        proj_length = float(np.dot(diff, self._axis))
        proj_axis_pt = self._center + proj_length * self._axis
        rad_vec = x_s - proj_axis_pt
        dist = float(np.linalg.norm(rad_vec))

        if dist < 1e-14:
            arbitrary = np.array([1.0, 0.0, 0.0]) if abs(self._axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            radial_normal = arbitrary - np.dot(arbitrary, self._axis) * self._axis
            radial_normal /= np.linalg.norm(radial_normal)
        else:
            radial_normal = rad_vec / dist

        if not self._capped or self._height is None:
            # Infinite cylinder
            if dist < 1e-14:
                normal = radial_normal
                gap = -self._radius
                closest_point = proj_axis_pt + self._radius * normal
                gradient = normal.copy()
                hessian = np.zeros((3, 3), dtype=np.float64)
                smooth = False
            else:
                normal = radial_normal
                gap = dist - self._radius
                closest_point = proj_axis_pt + self._radius * normal
                gradient = normal.copy()
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

        # Capped finite cylinder
        h = 0.5 * self._height
        abs_z = abs(proj_length)
        sgn_z = 1.0 if proj_length >= 0.0 else -1.0

        if abs_z < h and dist >= self._radius:
            # Outside barrel
            normal = radial_normal
            gap = dist - self._radius
            closest_point = proj_axis_pt + self._radius * normal
            gradient = normal.copy()
            P_transverse = np.eye(3) - np.outer(self._axis, self._axis) - np.outer(normal, normal)
            hessian = P_transverse / max(dist, 1e-14)
            feature_id = "cylinder_barrel"
            smooth = dist > 1e-12 * self._radius

        elif abs_z >= h and dist < self._radius:
            # Outside cap
            normal = sgn_z * self._axis
            gap = abs_z - h
            cap_center = self._center + (sgn_z * h) * self._axis
            closest_point = cap_center + rad_vec
            gradient = normal.copy()
            hessian = np.zeros((3, 3), dtype=np.float64)
            feature_id = "cylinder_cap"
            smooth = True

        elif abs_z >= h and dist >= self._radius:
            # Outside rim corner
            dr = dist - self._radius
            dz = abs_z - h
            gap = float(np.sqrt(dr**2 + dz**2))
            rim_center = self._center + (sgn_z * h) * self._axis + self._radius * radial_normal
            closest_point = rim_center
            safe_gap = max(gap, 1e-14)
            if gap < 1e-14:
                normal = (radial_normal + sgn_z * self._axis) / np.sqrt(2.0)
            else:
                normal = (dr * radial_normal + (dz * sgn_z) * self._axis) / safe_gap
            gradient = normal.copy()
            P_theta = np.eye(3) - np.outer(self._axis, self._axis) - np.outer(radial_normal, radial_normal)
            hessian = (np.eye(3) - np.outer(normal, normal)) / safe_gap - (self._radius / (safe_gap * max(dist, 1e-14))) * P_theta
            feature_id = "cylinder_rim"
            smooth = gap > 1e-12 * self._radius

        else:
            # Inside cylinder
            d_barrel = self._radius - dist
            d_cap = h - abs_z
            if d_barrel <= d_cap:
                normal = radial_normal
                gap = -(self._radius - dist)
                closest_point = proj_axis_pt + self._radius * normal
                gradient = normal.copy()
                P_transverse = np.eye(3) - np.outer(self._axis, self._axis) - np.outer(normal, normal)
                hessian = P_transverse / max(dist, 1e-14)
                feature_id = "cylinder_barrel"
                smooth = dist > 1e-12 * self._radius
            else:
                normal = sgn_z * self._axis
                gap = -(h - abs_z)
                cap_center = self._center + (sgn_z * h) * self._axis
                closest_point = cap_center + rad_vec
                gradient = normal.copy()
                hessian = np.zeros((3, 3), dtype=np.float64)
                feature_id = "cylinder_cap"
                smooth = True

        return DifferentiableContactGeometry(
            gap=gap,
            normal=normal,
            closest_point=closest_point,
            gradient=gradient,
            hessian=hessian,
            feature_id=feature_id,
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

    def __init__(self, units: str = "mm", energy_consistent_smoothing: bool = False) -> None:
        self._units = str(units)
        self._energy_consistent_smoothing = bool(energy_consistent_smoothing)

    @property
    def units(self) -> str:
        return self._units

    @property
    def energy_consistent_smoothing(self) -> bool:
        return self._energy_consistent_smoothing

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
        self,
        candidate: ContactCandidate,
        coordinates: np.ndarray,
        pseudonormal: np.ndarray | None = None,
        vertex_normals: np.ndarray | None = None,
    ) -> DifferentiableContactGeometry:
        """Evaluate 12-DOF gap, normal, gradient, and Hessian for node-triangle stencil.

        coordinates : np.ndarray of shape (4, 3)
            Row 0: slave node spatial position x_s.
            Rows 1-3: master triangle vertex positions [v0, v1, v2].
        pseudonormal : np.ndarray, optional
            Outward facet/edge/vertex pseudonormal for distance signing.
            If None, defaults to candidate.normal_hint or face normal.
        vertex_normals : np.ndarray of shape (3, 3), optional
            Master vertex pseudonormals for smoothed normal interpolation (Pillar 2).
        """
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.shape != (4, 3):
            raise ValueError(f"Triangle geometry requires shape (4, 3), got {coords.shape}")

        point = coords[0]
        vertices = coords[1:]

        face_normal, _ = TriangleContactGeometry._face_normal(vertices)
        weights = TriangleContactGeometry.closest_barycentric(point, vertices)
        active = np.flatnonzero(weights > 0.0)

        feature_str: str = "face" if len(active) == 3 else "edge" if len(active) == 2 else "vertex"

        if pseudonormal is None:
            if "pseudonormal" in candidate.metadata and candidate.metadata["pseudonormal"] is not None:
                pseudonormal = np.asarray(candidate.metadata["pseudonormal"], dtype=np.float64)
            else:
                pseudonormal = face_normal

        if vertex_normals is not None and self._energy_consistent_smoothing:
            return evaluate_smoothed_triangle_geometry(
                point=point,
                vertices=vertices,
                vertex_normals=vertex_normals,
                candidate_index=int(candidate.candidate_index),
                units=self._units,
            )

        geom = TriangleContactGeometry.evaluate(
            point=point,
            vertices=vertices,
            pseudonormal=pseudonormal,
            vertex_normals=vertex_normals,
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


def evaluate_smoothed_triangle_geometry(
    point: np.ndarray,
    vertices: np.ndarray,
    vertex_normals: np.ndarray,
    candidate_index: int = 0,
    units: str = "mm",
) -> DifferentiableContactGeometry:
    """Evaluate energy-consistent smoothed-normal contact geometry (resolves LEGACY-DISC-03).

    Differentiates the complete spatial projection and normal variation:
        g_N = r . n(xi)
        grad_q(g_N) = W^T n - sum( (x_mi . n) * dxi_i/dq ) + (dn/dq)^T r
        Hess_q(g_N) = d(grad_q(g_N))/dq

    Parameters:
    -----------
    point : np.ndarray of shape (3,)
        Spatial coordinates of the slave node.
    vertices : np.ndarray of shape (3, 3)
        Spatial coordinates of master triangle vertices [v0, v1, v2].
    vertex_normals : np.ndarray of shape (3, 3)
        Master vertex pseudonormals [n_v0, n_v1, n_v2].
    candidate_index : int, default=0
        Index of the candidate interaction.
    units : str, default='mm'
        Length units.

    Returns:
    --------
    geometry : DifferentiableContactGeometry
        Canonical contact geometry with exact negative energy gradient and Jacobian tangent.
    """
    p = np.asarray(point, dtype=np.float64).flatten()
    v = np.asarray(vertices, dtype=np.float64).reshape(3, 3)
    vn = np.asarray(vertex_normals, dtype=np.float64).reshape(3, 3)

    weights = TriangleContactGeometry.closest_barycentric(p, v)
    tilde_n = weights @ vn
    norm_n = float(np.linalg.norm(tilde_n))
    if norm_n < 1e-14:
        normal = np.array([0.0, 0.0, 1.0])
        norm_n = 1.0
    else:
        normal = tilde_n / norm_n

    r = p - weights @ v
    gap = float(r @ normal)
    closest_point = p - r

    def _eval_grad(q: np.ndarray) -> np.ndarray:
        pt = q[:3]
        vt = q[3:].reshape(3, 3)
        w = TriangleContactGeometry.closest_barycentric(pt, vt)
        tn = w @ vn
        ntn = float(np.linalg.norm(tn))
        n_curr = tn / ntn if ntn > 1e-14 else np.array([0.0, 0.0, 1.0])
        res = pt - w @ vt

        w4 = np.r_[1.0, -w]
        W = np.kron(w4.reshape(1, 4), np.eye(3))
        edges = (vt[1:] - vt[0]).T
        dw = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        B = np.column_stack([np.kron(dw[:, j], res) - W.T @ edges[:, j] for j in range(2)])
        dz = -np.linalg.solve(edges.T @ edges, B.T)
        dweights = dw @ dz
        dbary = -dweights[1:]

        P_n = np.eye(3) - np.outer(n_curr, n_curr)
        D_tilde = sum(np.outer(vn[i], dbary[i]) for i in range(3))
        dnormal = P_n @ D_tilde / ntn

        g = np.kron(w4, n_curr) - sum((vt[i] @ n_curr) * dbary[i] for i in range(3)) + dnormal.T @ res
        return g

    q0 = np.r_[p, v.ravel()]
    gradient = _eval_grad(q0)

    # Centered finite differences on 12-DOF stencil for exact consistent Hessian
    eps = 1e-6
    hessian = np.zeros((12, 12), dtype=np.float64)
    for i in range(12):
        qp = q0.copy()
        qp[i] += eps
        qm = q0.copy()
        qm[i] -= eps
        hessian[:, i] = (_eval_grad(qp) - _eval_grad(qm)) / (2.0 * eps)
    hessian = 0.5 * (hessian + hessian.T)

    active = np.flatnonzero(weights > 0.0)
    is_smooth = len(active) == 3 and bool(np.all(weights > 1e-6))
    feature_str = "face" if len(active) == 3 else "edge" if len(active) == 2 else "vertex"

    return DifferentiableContactGeometry(
        gap=gap,
        normal=normal,
        closest_point=closest_point,
        gradient=gradient,
        hessian=hessian,
        feature_id=feature_str,
        feature_index=candidate_index,
        barycentric=weights,
        is_smooth_branch=is_smooth,
        units=units,
        orientation_convention="master_outward",
    )


class SmoothedTriangleGeometryProvider(ContactGeometryProvider):
    """Energy-consistent smoothed-normal geometry provider (resolves LEGACY-DISC-03).

    Evaluates complete master shape and normal variation across smoothed triangular facets:
        g_N = r . n(xi)
        grad_q(g_N) = W^T n - sum( (x_mi . n) * dxi_i/dq ) + (dn/dq)^T r
        Hess_q(g_N) = d(grad_q(g_N))/dq
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
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.shape != (4, 3):
            return False
        point = coords[0]
        vertices = coords[1:]
        weights = TriangleContactGeometry.closest_barycentric(point, vertices)
        return bool(np.all(weights > 1e-6))

    def evaluate_geometry(
        self,
        candidate: ContactCandidate,
        coordinates: np.ndarray,
        pseudonormal: np.ndarray | None = None,
        vertex_normals: np.ndarray | None = None,
    ) -> DifferentiableContactGeometry:
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.shape != (4, 3):
            raise ValueError(f"Triangle geometry requires shape (4, 3), got {coords.shape}")

        point = coords[0]
        vertices = coords[1:]

        if vertex_normals is None:
            # Fallback to triangle face normal if vertex normals not provided
            face_normal, _ = TriangleContactGeometry._face_normal(vertices)
            vertex_normals = np.tile(face_normal, (3, 1))

        return evaluate_smoothed_triangle_geometry(
            point=point,
            vertices=vertices,
            vertex_normals=vertex_normals,
            candidate_index=int(candidate.candidate_index),
            units=self._units,
        )
