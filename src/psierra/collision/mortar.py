"""Segment-to-Segment Mortar Contact Interface Integration & Geometry Kernels.

Implements Sutherland-Hodgman 2D polygon clipping, sub-triangulation,
biorthogonal dual shape functions (Wohlmuth 2000, Popp et al. 2009),
Dunavant 3-point quadrature, and Mortar coupling matrix assembly (D, M).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as spsp


_CLIP_TOL = 1e-12
_AREA_TOL = 1e-14


def polygon_area_2d(polygon: np.ndarray) -> float:
    """Compute unsigned 2D polygon area using the Shoelace formula."""
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def is_counter_clockwise_2d(polygon: np.ndarray) -> bool:
    """Check whether a 2D polygon has counter-clockwise vertex ordering."""
    if len(polygon) < 3:
        return True
    x = polygon[:, 0]
    y = polygon[:, 1]
    signed_area = 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return signed_area < 0.0  # With roll(y, 1), x_i * y_{i-1} - y_i * x_{i-1} is positive for CCW when negated


def ensure_counter_clockwise_2d(polygon: np.ndarray) -> np.ndarray:
    """Return polygon vertices in counter-clockwise order."""
    if len(polygon) < 3:
        return polygon
    # Cross product of edges (p1 - p0) x (p2 - p0)
    p0, p1, p2 = polygon[0], polygon[1], polygon[2]
    cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
    if cross < 0.0:
        return polygon[::-1].copy()
    return polygon


def sutherland_hodgman_clip_2d(
    subject_poly: np.ndarray, clip_poly: np.ndarray
) -> np.ndarray:
    """Clip a 2D subject polygon against a convex 2D clip polygon.

    Both polygons are 2D arrays of shape (N, 2). The clip polygon must be convex
    and counter-clockwise oriented.
    Returns the clipped convex polygon vertices (P, 2).
    """
    subject_poly = np.asarray(subject_poly, dtype=np.float64)
    clip_poly = np.asarray(clip_poly, dtype=np.float64)
    if len(subject_poly) < 3 or len(clip_poly) < 3:
        return np.empty((0, 2), dtype=np.float64)

    clip_poly = ensure_counter_clockwise_2d(clip_poly)
    output = subject_poly.copy()
    n_clip = len(clip_poly)

    for i in range(n_clip):
        cp1 = clip_poly[i]
        cp2 = clip_poly[(i + 1) % n_clip]
        edge = cp2 - cp1

        input_list = output
        output_list = []
        if len(input_list) == 0:
            break

        s = input_list[-1]
        for e in input_list:
            # Point inside half-plane: cross product of directed edge with (pt - cp1) >= -tol
            e_in = bool((edge[0] * (e[1] - cp1[1]) - edge[1] * (e[0] - cp1[0])) >= -_CLIP_TOL)
            s_in = bool((edge[0] * (s[1] - cp1[1]) - edge[1] * (s[0] - cp1[0])) >= -_CLIP_TOL)

            if e_in:
                if not s_in:
                    # Line intersection
                    dc = edge
                    ds = e - s
                    denom = dc[0] * ds[1] - dc[1] * ds[0]
                    if abs(denom) > 1e-14:
                        dp = s - cp1
                        u = float((dc[0] * dp[1] - dc[1] * dp[0]) / denom)
                        output_list.append(s - u * ds)
                    else:
                        output_list.append(s)
                output_list.append(e)
            elif s_in:
                dc = edge
                ds = e - s
                denom = dc[0] * ds[1] - dc[1] * ds[0]
                if abs(denom) > 1e-14:
                    dp = s - cp1
                    u = float((dc[0] * dp[1] - dc[1] * dp[0]) / denom)
                    output_list.append(s - u * ds)
                else:
                    output_list.append(e)
            s = e

        # Filter contiguous duplicate vertices
        filtered: list[np.ndarray] = []
        for pt in output_list:
            if not filtered or np.linalg.norm(pt - filtered[-1]) > 1e-11:
                filtered.append(pt)
        if len(filtered) > 1 and np.linalg.norm(filtered[0] - filtered[-1]) <= 1e-11:
            filtered.pop()

        output = np.array(filtered, dtype=np.float64) if len(filtered) >= 3 else np.empty((0, 2), dtype=np.float64)

    if len(output) >= 3 and polygon_area_2d(output) > _AREA_TOL:
        return output
    return np.empty((0, 2), dtype=np.float64)


def triangulate_convex_polygon_2d(polygon: np.ndarray) -> list[np.ndarray]:
    """Partition a convex polygon (P, 2) into P - 2 triangles."""
    p = np.asarray(polygon, dtype=np.float64)
    n = len(p)
    if n < 3:
        return []
    p0 = p[0]
    return [np.array([p0, p[i], p[i + 1]], dtype=np.float64) for i in range(1, n - 1)]


def dunavant_triangle_quadrature_3pt() -> tuple[np.ndarray, np.ndarray]:
    """Return 3-point Dunavant Gauss quadrature barycentric coordinates and weights.

    Exact for polynomials up to degree 2 (sufficient for linear-linear mortar products).
    """
    points = np.array(
        [
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        ],
        dtype=np.float64,
    )
    weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    return points, weights


def eval_dual_shape_functions(xi: np.ndarray) -> np.ndarray:
    """Evaluate Wohlmuth biorthogonal dual shape functions on a linear triangle.

    Phi_i(xi) = 4 * xi_i - 1.
    Satisfies int_{tau} Phi_i * N_j dGamma = (A / 3) * delta_{ij}.
    """
    return 4.0 * np.asarray(xi, dtype=np.float64) - 1.0


@dataclass
class MortarSegmentData:
    """Computed mortar coupling data for an interacting slave-master facet pair."""

    slave_facet_idx: int
    master_facet_idx: int
    slave_nodes: np.ndarray  # (3,) global node indices
    master_nodes: np.ndarray  # (3,) global node indices
    overlap_area: float
    D_local: np.ndarray  # (3, 3) diagonal
    M_local: np.ndarray  # (3, 3)
    weighted_gaps: np.ndarray  # (3,)
    normal: np.ndarray  # (3,) master outward unit normal


class MortarInterfaceIntegrator:
    """Performs spatial polygon clipping and mortar matrix assembly across interface surfaces."""

    def __init__(
        self,
        slave_nodes: np.ndarray,
        slave_faces: np.ndarray,
        master_nodes: np.ndarray,
        master_faces: np.ndarray,
        search_radius: float = 10.0,
    ) -> None:
        """Initialize the integrator with surface mesh topology.

        slave_nodes: (N_s, 3) coordinates
        slave_faces: (F_s, 3) triangle vertex node indices
        master_nodes: (N_m, 3) coordinates
        master_faces: (F_m, 3) triangle vertex node indices
        search_radius: distance tolerance for candidate AABB overlap [length]
        """
        self.slave_nodes = np.asarray(slave_nodes, dtype=np.float64)
        self.slave_faces = np.asarray(slave_faces, dtype=np.int64)
        self.master_nodes = np.asarray(master_nodes, dtype=np.float64)
        self.master_faces = np.asarray(master_faces, dtype=np.int64)
        self.search_radius = float(search_radius)

    @staticmethod
    def _facet_normal(vertices: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute outward unit normal and area of a 3D triangle."""
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        cross = np.cross(v1, v2)
        norm = float(np.linalg.norm(cross))
        if norm < 1e-14:
            return np.array([0.0, 0.0, 1.0]), 0.0
        return cross / norm, 0.5 * norm

    def integrate(
        self,
        current_slave_coords: np.ndarray | None = None,
        current_master_coords: np.ndarray | None = None,
    ) -> tuple[np.ndarray, spsp.csr_matrix, np.ndarray, list[MortarSegmentData]]:
        """Integrate interface mortar matrices and weighted gaps.

        Returns:
            D_diag: (N_slave_nodes,) diagonal slave mortar matrix D_{ii}
            M_csr: (N_slave_nodes, N_master_nodes) sparse cross-mortar matrix M_{ik}
            weighted_gaps: (N_slave_nodes,) integrated weighted gap vector \\tilde{g}_i
            segments: list of active MortarSegmentData records
        """
        s_coords = (
            self.slave_nodes
            if current_slave_coords is None
            else np.asarray(current_slave_coords, dtype=np.float64)
        )
        m_coords = (
            self.master_nodes
            if current_master_coords is None
            else np.asarray(current_master_coords, dtype=np.float64)
        )

        n_s_nodes = len(s_coords)
        n_m_nodes = len(m_coords)

        # 1. Compute slave facet normals, areas, and bounding boxes
        s_faces = self.slave_faces
        m_faces = self.master_faces

        s_face_normals = np.empty((len(s_faces), 3), dtype=np.float64)
        s_face_areas = np.empty(len(s_faces), dtype=np.float64)
        s_bboxes_min = np.empty((len(s_faces), 3), dtype=np.float64)
        s_bboxes_max = np.empty((len(s_faces), 3), dtype=np.float64)

        for i, face in enumerate(s_faces):
            verts = s_coords[face]
            n, a = self._facet_normal(verts)
            s_face_normals[i] = n
            s_face_areas[i] = a
            s_bboxes_min[i] = np.min(verts, axis=0) - self.search_radius
            s_bboxes_max[i] = np.max(verts, axis=0) + self.search_radius

        m_face_normals = np.empty((len(m_faces), 3), dtype=np.float64)
        m_face_areas = np.empty(len(m_faces), dtype=np.float64)
        m_bboxes_min = np.empty((len(m_faces), 3), dtype=np.float64)
        m_bboxes_max = np.empty((len(m_faces), 3), dtype=np.float64)

        for j, face in enumerate(m_faces):
            verts = m_coords[face]
            n, a = self._facet_normal(verts)
            m_face_normals[j] = n
            m_face_areas[j] = a
            m_bboxes_min[j] = np.min(verts, axis=0)
            m_bboxes_max[j] = np.max(verts, axis=0)

        # Initialize global accumulators
        D_diag = np.zeros(n_s_nodes, dtype=np.float64)
        # Pre-accumulate D_diag for all slave facets (each facet gives A/3 to its 3 nodes)
        for i, face in enumerate(s_faces):
            area = s_face_areas[i]
            D_diag[face] += area / 3.0

        # Protect against unreferenced or isolated slave nodes: minimum epsilon area
        D_diag = np.maximum(D_diag, 1e-14)

        M_rows: list[int] = []
        M_cols: list[int] = []
        M_vals: list[float] = []
        weighted_gaps = np.zeros(n_s_nodes, dtype=np.float64)
        segments: list[MortarSegmentData] = []

        quad_pts, quad_weights = dunavant_triangle_quadrature_3pt()

        # Broad-phase + Narrow-phase pairing
        for s_idx, s_face in enumerate(s_faces):
            s_verts = s_coords[s_face]
            n_s = s_face_normals[s_idx]
            if s_face_areas[s_idx] < _AREA_TOL:
                continue

            s_min = s_bboxes_min[s_idx]
            s_max = s_bboxes_max[s_idx]

            # Vectorized AABB candidate check against all master facets
            aabb_mask = np.all(s_max >= m_bboxes_min, axis=1) & np.all(s_min <= m_bboxes_max, axis=1)
            candidate_m_indices = np.flatnonzero(aabb_mask)

            if len(candidate_m_indices) == 0:
                continue

            # Setup local 2D coordinate frame on slave facet
            v1 = s_verts[1] - s_verts[0]
            norm_v1 = float(np.linalg.norm(v1))
            if norm_v1 < 1e-14:
                continue
            t1 = v1 / norm_v1
            t2 = np.cross(n_s, t1)
            o_s = s_verts[0]

            def to_2d(pt: np.ndarray) -> np.ndarray:
                d = pt - o_s
                return np.array([float(d @ t1), float(d @ t2)], dtype=np.float64)

            u_s = np.array([to_2d(v) for v in s_verts], dtype=np.float64)
            u_s = ensure_counter_clockwise_2d(u_s)

            for m_idx in candidate_m_indices:
                m_verts = m_coords[m_faces[m_idx]]
                n_m = m_face_normals[m_idx]

                # Opposing normal filter: master and slave must face each other (n_s . n_m < -0.2)
                normal_alignment = float(n_s @ n_m)
                if normal_alignment > -0.2:
                    continue

                # Normal separation filter: facet centroid distance along master normal must be within search_radius
                c_s = np.mean(s_verts, axis=0)
                normal_dist = float((c_s - m_verts[0]) @ n_m)
                if abs(normal_dist) > self.search_radius:
                    continue

                # Project master triangle vertices to slave tangent plane
                u_m = np.array([to_2d(v) for v in m_verts], dtype=np.float64)
                u_m = ensure_counter_clockwise_2d(u_m)

                # Sutherland-Hodgman polygon clipping
                clipped_poly = sutherland_hodgman_clip_2d(u_m, u_s)
                if len(clipped_poly) < 3:
                    continue

                poly_area = polygon_area_2d(clipped_poly)
                if poly_area < _AREA_TOL:
                    continue

                # Partition clipped polygon into sub-triangles
                sub_triangles = triangulate_convex_polygon_2d(clipped_poly)
                if not sub_triangles:
                    continue

                M_local = np.zeros((3, 3), dtype=np.float64)
                D_local = np.diag([s_face_areas[s_idx] / 3.0] * 3)
                gaps_local = np.zeros(3, dtype=np.float64)

                denom = float(n_s @ n_m)
                if abs(denom) < 1e-12:
                    continue

                for tri in sub_triangles:
                    tri_area = polygon_area_2d(tri)
                    if tri_area < _AREA_TOL:
                        continue

                    for g_idx in range(len(quad_pts)):
                        bary_tri = quad_pts[g_idx]
                        w_g = quad_weights[g_idx]
                        dGamma = tri_area * w_g

                        # 2D point on sub-triangle
                        pt_2d = bary_tri[0] * tri[0] + bary_tri[1] * tri[1] + bary_tri[2] * tri[2]

                        # 3D point on slave facet
                        x_g = o_s + pt_2d[0] * t1 + pt_2d[1] * t2

                        # Slave barycentric coordinates on s_verts
                        # Solve [v1, v2] * [xi1, xi2]^T = x_g - s_verts[0]
                        A_mat = np.column_stack([s_verts[1] - s_verts[0], s_verts[2] - s_verts[0]])
                        rhs = x_g - s_verts[0]
                        xi_sub, _, _, _ = np.linalg.lstsq(A_mat, rhs, rcond=None)
                        xi_s = np.array([1.0 - xi_sub[0] - xi_sub[1], xi_sub[0], xi_sub[1]], dtype=np.float64)
                        xi_s = np.clip(xi_s, 0.0, 1.0)
                        sum_xi = float(np.sum(xi_s))
                        if sum_xi > 1e-14:
                            xi_s /= sum_xi

                        # Dual shape functions: Phi_i(xi) = 4*xi_i - 1
                        Phi_s = eval_dual_shape_functions(xi_s)

                        # Ray projection along slave normal onto master plane
                        # (x_g + t * n_s - m_verts[0]) @ n_m = 0
                        # gap_g = t = (m_verts[0] - x_g) @ n_m / (n_s @ n_m)
                        t = float((m_verts[0] - x_g) @ n_m / denom)
                        gap_g = t
                        x_m_proj = x_g + t * n_s

                        # Master barycentric coordinates on m_verts
                        Am_mat = np.column_stack([m_verts[1] - m_verts[0], m_verts[2] - m_verts[0]])
                        rhsm = x_m_proj - m_verts[0]
                        xi_m_sub, _, _, _ = np.linalg.lstsq(Am_mat, rhsm, rcond=None)
                        xi_m = np.array([1.0 - xi_m_sub[0] - xi_m_sub[1], xi_m_sub[0], xi_m_sub[1]], dtype=np.float64)
                        xi_m = np.clip(xi_m, 0.0, 1.0)
                        sum_m = float(np.sum(xi_m))
                        if sum_m > 1e-14:
                            xi_m /= sum_m

                        N_m = xi_m

                        # Accumulate M_local and weighted gap
                        M_local += dGamma * np.outer(Phi_s, N_m)
                        gaps_local += dGamma * Phi_s * gap_g

                # Record segment
                seg = MortarSegmentData(
                    slave_facet_idx=s_idx,
                    master_facet_idx=m_idx,
                    slave_nodes=s_face.copy(),
                    master_nodes=m_faces[m_idx].copy(),
                    overlap_area=poly_area,
                    D_local=D_local,
                    M_local=M_local,
                    weighted_gaps=gaps_local,
                    normal=n_m.copy(),
                )
                segments.append(seg)

                # Accumulate into global sparse M and weighted_gaps
                for a in range(3):
                    s_node = int(s_face[a])
                    weighted_gaps[s_node] += gaps_local[a]
                    for b in range(3):
                        m_node = int(m_faces[m_idx][b])
                        val = float(M_local[a, b])
                        if abs(val) > 1e-15:
                            M_rows.append(s_node)
                            M_cols.append(m_node)
                            M_vals.append(val)

        if M_rows:
            M_coo = spsp.coo_matrix(
                (M_vals, (M_rows, M_cols)), shape=(n_s_nodes, n_m_nodes), dtype=np.float64
            )
            M_csr = M_coo.tocsr()
        else:
            M_csr = spsp.csr_matrix((n_s_nodes, n_m_nodes), dtype=np.float64)

        return D_diag, M_csr, weighted_gaps, segments
