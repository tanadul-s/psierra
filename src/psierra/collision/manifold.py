"""
Surface Manifold Orientation, Pseudonormals, and Inside-Outside Classification.
==============================================================================
Provides robust topology and orientation contracts for master triangular surfaces:
1. Watertight manifold repair and consistent outward orientation validation.
2. Topological inside-outside classification for closed manifolds via ray-winding.
3. Angle-weighted pseudonormal computation (Bærentzen & Aanæs 2005) for smooth
   edge and vertex sign evaluation.
"""

from __future__ import annotations
from typing import Sequence

import numpy as np
import pyvista as pv
import vtk


def ensure_consistent_surface_orientation(
    polydata: pv.PolyData,
    force_outward: bool = True,
    clean_mesh: bool = True,
) -> pv.PolyData:
    """
    Validates and repairs triangular surface mesh orientation.

    Ensures that adjacent triangles have consistent facet winding order and outward
    normals. For closed watertight surfaces, validates that normals point outward
    into the exterior (away from the enclosed volume).

    Parameters:
    -----------
    polydata : pv.PolyData
        Input triangular surface mesh.
    force_outward : bool, default=True
        Whether to enforce outward-pointing normals on closed surfaces.
    clean_mesh : bool, default=True
        Whether to merge duplicate vertices and remove degenerate cells first.

    Returns:
    --------
    oriented_polydata : pv.PolyData
        Surface mesh with consistently oriented cells and unit normals.
    """
    if not isinstance(polydata, pv.PolyData):
        raise TypeError(f"Expected pyvista.PolyData, got {type(polydata)}")

    pdata = polydata.copy()

    # 1. Triangulate if not all triangles
    if not pdata.is_all_triangles:
        pdata = pdata.triangulate()

    # 2. Clean degenerate vertices / duplicate points
    if clean_mesh:
        pdata = pdata.clean(tolerance=1e-8, absolute=False)

    # 3. VTK PolyDataNormals for consistent facet winding
    norm_filter = vtk.vtkPolyDataNormals()
    norm_filter.SetInputData(pdata)
    norm_filter.SetAutoOrientNormals(True)
    norm_filter.SetConsistency(True)
    norm_filter.SetSplitting(False)
    norm_filter.SetComputePointNormals(True)
    norm_filter.SetComputeCellNormals(True)
    norm_filter.Update()

    oriented = pv.wrap(norm_filter.GetOutput())

    # 4. Check outward orientation for closed watertight surfaces via Gauss Divergence Theorem
    # Volume = 1/3 \sum (x_centroid \cdot n_cell) * Area
    if force_outward and oriented.n_open_edges == 0 and oriented.n_cells > 0:
        pts = np.asarray(oriented.points, dtype=np.float64)
        faces = oriented.regular_faces  # shape (N_cells, 3)
        tri_pts = pts[faces]  # shape (N_cells, 3, 3)

        e1 = tri_pts[:, 1] - tri_pts[:, 0]
        e2 = tri_pts[:, 2] - tri_pts[:, 0]
        cross = np.cross(e1, e2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        centroids = np.mean(tri_pts, axis=1)

        c_normals = np.asarray(oriented.cell_normals, dtype=np.float64)
        vol_approx = (1.0 / 3.0) * np.sum(np.sum(centroids * c_normals, axis=1) * areas)

        if vol_approx < -1e-12:
            # Normals are pointing inward! Flip faces
            oriented = oriented.flip_faces(inplace=False)
            norm_filter2 = vtk.vtkPolyDataNormals()
            norm_filter2.SetInputData(oriented)
            norm_filter2.SetComputePointNormals(True)
            norm_filter2.SetComputeCellNormals(True)
            norm_filter2.Update()
            oriented = pv.wrap(norm_filter2.GetOutput())

    return oriented


def is_inside_closed_surface(
    points: np.ndarray | Sequence[Sequence[float]],
    surface: pv.PolyData,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Evaluates topological inside-outside classification for query points against
    a closed watertight triangular surface using VTK ray-winding (vtkSelectEnclosedPoints).

    Points inside the enclosed volume return True; points outside return False.
    Immune to concave pockets and local facet winding variations.

    Parameters:
    -----------
    points : np.ndarray of shape (N, 3) or (3,)
        Spatial coordinates of query points.
    surface : pv.PolyData
        Closed watertight surface mesh.
    tolerance : float, default=1e-6
        Geometric tolerance for boundary surface coincidence.

    Returns:
    --------
    inside : np.ndarray of shape (N,) bool
        True if the point lies strictly inside the surface volume.
    """
    pts = np.asarray(points, dtype=np.float64)
    single_point = pts.ndim == 1
    if single_point:
        pts = pts.reshape(1, 3)

    if pts.shape[1] != 3:
        raise ValueError(f"Expected points of shape (N, 3), got {pts.shape}")

    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetSurfaceData(surface)
    enclosed.SetTolerance(float(tolerance))
    enclosed.Initialize(surface)

    n_pts = pts.shape[0]
    inside = np.zeros(n_pts, dtype=bool)
    for i in range(n_pts):
        inside[i] = bool(enclosed.IsInsideSurface(pts[i, 0], pts[i, 1], pts[i, 2]))

    enclosed.Complete()

    if single_point:
        return inside[0]
    return inside


def compute_angle_weighted_pseudonormals(
    polydata: pv.PolyData,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    r"""
    Computes angle-weighted pseudonormals for all vertices and edges of a triangular mesh
    following Bærentzen & Aanæs (2005).

    The vertex pseudonormal is the sum of incident facet normals weighted by the
    facet's interior angle at that vertex:
        n_v = \sum \theta_{f, v} n_f / ||\sum \theta_{f, v} n_f||

    The edge pseudonormal is the normalized sum of adjacent face normals:
        n_e = (n_f1 + n_f2) / ||n_f1 + n_f2||

    Parameters:
    -----------
    polydata : pv.PolyData
        Triangular surface mesh.

    Returns:
    --------
    vertex_pseudonormals : np.ndarray of shape (N_vertices, 3)
        Angle-weighted outward unit normal for each mesh vertex.
    edge_pseudonormals : dict mapping (min_idx, max_idx) -> np.ndarray of shape (3,)
        Unit normal for each manifold edge.
    """
    pdata = polydata
    if not pdata.is_all_triangles:
        pdata = pdata.triangulate()

    pts = np.asarray(pdata.points, dtype=np.float64)
    faces = pdata.regular_faces  # shape (N_cells, 3)
    n_vertices = pts.shape[0]

    vertex_normals_acc = np.zeros((n_vertices, 3), dtype=np.float64)
    edge_face_map: dict[tuple[int, int], list[np.ndarray]] = {}

    for face in faces:
        i, j, k = int(face[0]), int(face[1]), int(face[2])
        v_i, v_j, v_k = pts[i], pts[j], pts[k]

        e_ij = v_j - v_i
        e_ik = v_k - v_i
        e_ji = -e_ij
        e_jk = v_k - v_j
        e_ki = -e_ik
        e_kj = -e_jk

        cross = np.cross(e_ij, e_ik)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-14:
            continue
        n_face = cross / norm_cross

        # Interior angles at each vertex
        # angle at i
        cos_i = np.clip(np.dot(e_ij, e_ik) / (np.linalg.norm(e_ij) * np.linalg.norm(e_ik)), -1.0, 1.0)
        theta_i = float(np.arccos(cos_i))

        # angle at j
        cos_j = np.clip(np.dot(e_ji, e_jk) / (np.linalg.norm(e_ji) * np.linalg.norm(e_jk)), -1.0, 1.0)
        theta_j = float(np.arccos(cos_j))

        # angle at k
        cos_k = np.clip(np.dot(e_ki, e_kj) / (np.linalg.norm(e_ki) * np.linalg.norm(e_kj)), -1.0, 1.0)
        theta_k = float(np.arccos(cos_k))

        vertex_normals_acc[i] += theta_i * n_face
        vertex_normals_acc[j] += theta_j * n_face
        vertex_normals_acc[k] += theta_k * n_face

        # Collect incident faces for each edge
        for u, v in ((i, j), (j, k), (k, i)):
            edge_key = (min(u, v), max(u, v))
            edge_face_map.setdefault(edge_key, []).append(n_face)

    # Normalize vertex pseudonormals
    v_norms = np.linalg.norm(vertex_normals_acc, axis=1, keepdims=True)
    vertex_pseudonormals = np.where(v_norms > 1e-14, vertex_normals_acc / v_norms, 0.0)

    # Normalize edge pseudonormals
    edge_pseudonormals: dict[tuple[int, int], np.ndarray] = {}
    for edge_key, face_norms in edge_face_map.items():
        n_acc = np.sum(face_norms, axis=0)
        norm_acc = np.linalg.norm(n_acc)
        if norm_acc > 1e-14:
            edge_pseudonormals[edge_key] = n_acc / norm_acc
        else:
            edge_pseudonormals[edge_key] = face_norms[0]

    return vertex_pseudonormals, edge_pseudonormals


def query_closed_surface_signed_distance(
    points: np.ndarray,
    surface: pv.PolyData,
    locator: vtk.vtkStaticCellLocator | None = None,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates exact signed distance gaps, outward normals, and closest points
    against a closed surface mesh with topological inside-outside classification.

    Parameters:
    -----------
    points : np.ndarray of shape (N, 3)
        Spatial coordinates of query points.
    surface : pv.PolyData
        Closed watertight surface mesh.
    locator : vtk.vtkStaticCellLocator, optional
        Pre-built cell locator for closest-point queries.
    tolerance : float, default=1e-6
        Inside-outside test tolerance.

    Returns:
    --------
    gaps : np.ndarray of shape (N,)
        Signed distance (g < 0 inside, g >= 0 outside).
    normals : np.ndarray of shape (N, 3)
        Outward unit normals pointing away from the obstacle.
    closest_points : np.ndarray of shape (N, 3)
        Closest boundary points on the mesh surface.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points of shape (N, 3), got {pts.shape}")

    n_pts = pts.shape[0]
    gaps = np.zeros(n_pts, dtype=np.float64)
    normals = np.zeros((n_pts, 3), dtype=np.float64)
    closest_points = np.zeros((n_pts, 3), dtype=np.float64)

    if locator is None:
        loc = vtk.vtkStaticCellLocator()
        loc.SetDataSet(surface)
        loc.BuildLocator()
    else:
        loc = locator

    inside_mask = is_inside_closed_surface(pts, surface, tolerance=tolerance)

    closest_pt = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)

    cell_normals = surface.cell_normals
    if cell_normals is None:
        surface = surface.compute_normals(cell_normals=True, point_normals=False)
        cell_normals = surface.cell_normals
    c_normals = np.asarray(cell_normals, dtype=np.float64)

    for i in range(n_pts):
        pt = pts[i]
        loc.FindClosestPoint(pt, closest_pt, cell_id, sub_id, dist2)
        c_id = cell_id.get()
        cp = np.array(closest_pt, dtype=np.float64)
        closest_points[i] = cp

        dist = float(np.sqrt(dist2.get()))
        is_inside = inside_mask[i]

        diff = pt - cp
        diff_norm = np.linalg.norm(diff)

        if diff_norm > 1e-12:
            # Outward normal points from cp to pt if outside, or opposite if inside
            n = diff / diff_norm
            if is_inside:
                n = -n
        else:
            # Point is on surface: use facet normal
            n = c_normals[c_id]
            norm_n = np.linalg.norm(n)
            n = n / norm_n if norm_n > 1e-12 else np.array([0.0, 1.0, 0.0])

        normals[i] = n
        gaps[i] = -dist if is_inside else dist

    return gaps, normals, closest_points
