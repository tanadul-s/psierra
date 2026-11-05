"""Exact closest triangular features and their branchwise gap derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class TriangleGap:
    """Signed gap [length], unit normal, and local 12-DOF derivatives.

    Local coordinates are ``q=[x_slave,x_m1,x_m2,x_m3]``. The gradient is
    dimensionless and the Hessian has units of inverse length. Derivatives
    hold the nearest feature and its oriented sign branch fixed.
    """

    barycentric: np.ndarray
    gap: float
    normal: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray
    feature: Literal["face", "edge", "vertex"]


class TriangleContactGeometry:
    """Closest-point signed distance on an oriented piecewise planar triangle.

    Projection is minimized over the face and its three closed edges. For a
    fixed closest feature, stationary projection coordinates are differentiated
    analytically using a 1x1 or 2x2 solve. Both normal and shape-weight variation
    enter ``Hess(g)``; no finite differences or full mesh Jacobians are used.

    Edge/vertex pseudonormals determine only the distance sign. The gradient
    direction is the signed Euclidean distance direction. Sign validity for a
    whole surface assumes a consistently outward-oriented closed manifold;
    open triangles supply only a local oriented-side convention.

    At a zero-distance edge/vertex, use the incident face's finite one-sided
    plane derivatives. Feature switches, sign switches, and nearest-facet ties
    need not have a unique classical Hessian. See
    docs/05_system_assembly_boundaries_and_domain_coupling.md;
    Wriggers, Computational Contact Mechanics, chs. 2 and 6.
    """

    _relative_tolerance = 64.0 * np.finfo(np.float64).eps

    @classmethod
    def _face_normal(cls, vertices: np.ndarray) -> tuple[np.ndarray, float]:
        """Return oriented unit normal and twice-area; reject degenerate facets."""
        if vertices.shape != (3, 3) or not np.all(np.isfinite(vertices)):
            raise ValueError("Triangle vertices must be finite with shape (3, 3)")
        e1, e2 = vertices[1:] - vertices[0]
        cross = np.cross(e1, e2)
        area = float(np.linalg.norm(cross))
        scale_squared = max(
            float(e1 @ e1), float(e2 @ e2),
            float((e2 - e1) @ (e2 - e1)),
        )
        if area <= cls._relative_tolerance * scale_squared or area == 0.0:
            raise ValueError("Degenerate master triangle: nonzero relative area required")
        return cross / area, area

    @staticmethod
    def _skew(vector: np.ndarray) -> np.ndarray:
        """Cross-product matrix S(v), defined by S(v)w=v cross w."""
        x, y, z = vector
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

    @classmethod
    def closest_barycentric(cls, point: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """Minimize Euclidean distance over the face, edges, and vertices.

        Inputs are coordinates [length]; output weights are dimensionless.
        Unlike componentwise barycentric clipping, this also gives the exact
        closest feature for points outside the triangle (Ericson, Real-Time
        Collision Detection, sec. 5.1.5).
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        point = np.asarray(point, dtype=np.float64)
        normal, area = cls._face_normal(vertices)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("Slave point must be finite with shape (3,)")
        e1, e2 = vertices[1:] - vertices[0]
        d = point - vertices[0]
        z1 = float(np.cross(d, e2) @ normal) / area
        z2 = float(np.cross(e1, d) @ normal) / area
        weights = np.array([1.0 - z1 - z2, z1, z2])
        if np.all(weights >= -cls._relative_tolerance):
            # Snap only roundoff at a feature boundary, not an outside point.
            weights[weights < cls._relative_tolerance] = 0.0
            return weights / weights.sum()

        best_squared = np.inf
        closest_weights = np.zeros(3)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = vertices[j] - vertices[i]
            t = float(np.clip((point - vertices[i]) @ edge / (edge @ edge), 0.0, 1.0))
            residual = point - vertices[i] - t * edge
            squared_distance = float(residual @ residual)
            if squared_distance < best_squared:
                best_squared = squared_distance
                closest_weights.fill(0.0)
                closest_weights[i] = 1.0 - t
                closest_weights[j] = t
        closest_weights[closest_weights < cls._relative_tolerance] = 0.0
        return closest_weights / closest_weights.sum()

    @classmethod
    def evaluate(
        cls,
        point: np.ndarray,
        vertices: np.ndarray,
        pseudonormal: np.ndarray | None = None,
    ) -> TriangleGap:
        """Return g, grad(g), Hess(g) on the selected smooth feature branch.

        For feature coordinates z, let r=x_s-sum(xi_i*x_mi), w=[1,-xi],
        W=w tensor I, E=[feature edges], and
        B_j=(dw/dz_j) tensor r - W.T E_j. Stationarity gives
        dz/dq=-(E.T E)^-1 B.T. Differentiating grad(g)=w tensor n then gives
        H_i=n tensor (dw_i/dq)+w_i*(dn/dq). Face dn/dq comes directly from
        the normalized cross product, remaining finite at g=0. For nonzero
        edge/vertex distance, dn/dq=(I-n*n.T)*(dr/dq)/g.
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        point = np.asarray(point, dtype=np.float64)
        face_normal, area = cls._face_normal(vertices)
        barycentric = cls.closest_barycentric(point, vertices)
        active = np.flatnonzero(barycentric > 0.0)
        feature: Literal["face", "edge", "vertex"] = (
            "face" if len(active) == 3 else "edge" if len(active) == 2 else "vertex"
        )
        residual = point - barycentric @ vertices
        distance = float(np.linalg.norm(residual))
        length_scale = max(float(np.linalg.norm(v - vertices[0])) for v in vertices[1:])
        plane_branch = feature == "face" or distance <= cls._relative_tolerance * length_scale
        if plane_branch:
            normal = face_normal
            gap = float(residual @ normal)
            active = np.arange(3)
        else:
            sign_normal = face_normal if pseudonormal is None else np.asarray(pseudonormal)
            sign = -1.0 if float(residual @ sign_normal) < 0.0 else 1.0
            gap = sign * distance
            normal = residual / gap

        weights = np.r_[1.0, -barycentric]
        W = np.kron(weights.reshape(1, 4), np.eye(3))
        edges = (vertices[active[1:]] - vertices[active[0]]).T
        n_coordinates = len(active) - 1
        dw = np.zeros((4, n_coordinates))
        edge_maps = np.zeros((n_coordinates, 3, 12))
        for j, node in enumerate(active[1:]):
            dw[1 + active[0], j] = 1.0
            dw[1 + node, j] = -1.0
            edge_maps[j, :, 3 * (1 + node):3 * (2 + node)] = np.eye(3)
            edge_maps[j, :, 3 * (1 + active[0]):3 * (2 + active[0])] = -np.eye(3)
        if n_coordinates:
            B = np.column_stack([
                np.kron(dw[:, j], residual) - W.T @ edges[:, j]
                for j in range(n_coordinates)
            ])
            dz = -np.linalg.solve(edges.T @ edges, B.T)
        else:
            dz = np.zeros((0, 12))
        dweights = dw @ dz
        projector = np.eye(3) - np.outer(normal, normal)
        if plane_branch:
            d_cross = (
                -cls._skew(edges[:, 1]) @ edge_maps[0]
                + cls._skew(edges[:, 0]) @ edge_maps[1]
            )
            dnormal = projector @ d_cross / area
        else:
            dnormal = projector @ (W - edges @ dz) / gap
        gradient = np.kron(weights, normal)
        hessian = np.vstack([
            np.outer(normal, dweights[i]) + weights[i] * dnormal
            for i in range(4)
        ])
        # Analytical symmetry is exact; remove floating-point roundoff only.
        hessian = 0.5 * (hessian + hessian.T)
        return TriangleGap(barycentric, gap, normal, gradient, hessian, feature)
