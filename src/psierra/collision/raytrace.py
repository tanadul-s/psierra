"""
Ray-Tracing & BVH-Accelerated Mesh Obstacle for Project Sierra.
==============================================================
Provides high-performance collision detection and ray-triangle intersection
using Intel Embree (via embreex) and PyVista / VTK StaticCellLocator for
imported 3D surface meshes (e.g. VTK floor, skull, surgical tools).
"""

from __future__ import annotations
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pyvista as pv
import vtk

from psierra.collision.shapes import CollisionShape
from psierra.mesh.surface import SurfaceMesh

try:
    from embreex.rtcore_scene import EmbreeScene
    from embreex.mesh_construction import TriangleMesh
    _EMBREE_AVAILABLE = True
except ImportError:
    _EMBREE_AVAILABLE = False


class RayMeshObstacle(CollisionShape):
    """
    3D Surface Mesh Obstacle with BVH Ray-Tracing & Closest-Point Locators.

    Supports imported VTK, STL, PLY, and OBJ surfaces (e.g. floor1.vtk, skull.vtk).
    Integrates Intel Embree BVH ray-triangle traversal for surgical probes / optical rays
    and VTK StaticCellLocator for exact orthogonal facet projection and signed normal gap evaluation.
    """

    def __init__(
        self,
        source: Path | str | SurfaceMesh | pv.PolyData,
        use_embree: bool = True,
    ) -> None:
        """
        Parameters:
        -----------
        source : Path, str, SurfaceMesh, or pv.PolyData
            Path to 3D surface mesh file (VTK format preferred) or an in-memory surface.
        use_embree : bool, default=True
            Whether to initialize the Intel Embree BVH ray-tracing engine (if available).
        """
        if isinstance(source, (str, Path)):
            smesh = SurfaceMesh(Path(source))
            pdata = smesh.polydata
        elif isinstance(source, SurfaceMesh):
            pdata = source.polydata
        elif isinstance(source, pv.PolyData):
            pdata = source
        elif isinstance(source, pv.UnstructuredGrid):
            pdata = source.extract_surface(algorithm=None)
        elif hasattr(source, "surface"):
            # VolumetricMesh
            pdata = source.surface.polydata
        elif hasattr(source, "mesh") and hasattr(source.mesh, "surface"):
            # SolidDomain
            pdata = source.mesh.surface.polydata
        elif hasattr(source, "ugrid"):
            pdata = source.ugrid.extract_surface(algorithm=None)
        else:
            raise TypeError(
                f"Unsupported source type for RayMeshObstacle: {type(source)}"
            )

        if not pdata.is_all_triangles:
            pdata = pdata.triangulate()

        # Compute cell and point normals
        pdata = pdata.compute_normals(
            cell_normals=True,
            point_normals=True,
            split_vertices=False,
            consistent_normals=True,
            auto_orient_normals=True,
        )
        self._polydata = pdata

        # 1. Build VTK StaticCellLocator for fast 3D proximity / closest point
        self._locator = vtk.vtkStaticCellLocator()
        self._locator.SetDataSet(self._polydata)
        self._locator.BuildLocator()

        # Extract cell normals
        cell_normals = self._polydata.cell_normals
        if cell_normals is None:
            cell_normals = self._polydata.compute_normals(
                cell_normals=True, point_normals=False
            ).cell_normals
        self._cell_normals = np.asarray(cell_normals, dtype=np.float64)

        # 2. Build Intel Embree BVH Scene
        self._embree_scene: EmbreeScene | None = None
        if use_embree and _EMBREE_AVAILABLE:
            try:
                verts = np.ascontiguousarray(self._polydata.points, dtype=np.float32)
                faces = np.ascontiguousarray(
                    self._polydata.regular_faces, dtype=np.int32
                )
                scene = EmbreeScene()
                TriangleMesh(scene, verts, faces)
                self._embree_scene = scene
            except Exception:
                self._embree_scene = None

    @property
    def polydata(self) -> pv.PolyData:
        """PyVista PolyData triangular surface representation."""
        return self._polydata

    @property
    def has_embree(self) -> bool:
        """Whether Intel Embree BVH ray-tracer is active."""
        return self._embree_scene is not None

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """AABB bounding box [xmin, xmax, ymin, ymax, zmin, zmax]."""
        return self._polydata.bounds

    # =========================================================================
    # Closest-Point & Gap Queries (CollisionShape Interface)
    # =========================================================================

    def query_points(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Projects query points onto the triangular mesh facets to calculate signed normal gaps,
        facet outward normals, and closest boundary points.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected points of shape (N, 3), got {pts.shape}")

        n_pts = pts.shape[0]
        gaps = np.zeros(n_pts, dtype=np.float64)
        normals = np.zeros((n_pts, 3), dtype=np.float64)
        closest_points = np.zeros((n_pts, 3), dtype=np.float64)

        closest_pt = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)

        locator = self._locator
        c_normals = self._cell_normals

        for i in range(n_pts):
            pt = pts[i]
            locator.FindClosestPoint(pt, closest_pt, cell_id, sub_id, dist2)
            c_id = cell_id.get()
            cp = np.array(closest_pt, dtype=np.float64)
            closest_points[i] = cp

            # Normal of the closest facet
            n = c_normals[c_id]
            norm_n = np.linalg.norm(n)
            if norm_n > 1e-12:
                n = n / norm_n
            else:
                n = np.array([0.0, 1.0, 0.0])
            normals[i] = n

            # Signed gap: projection along outward normal
            diff = pt - cp
            gaps[i] = np.dot(diff, n)

        return gaps, normals, closest_points

    # =========================================================================
    # BVH Ray-Tracing
    # =========================================================================

    def ray_intersect(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Traces rays against the triangular mesh surface using Intel Embree BVH
        (or PyVista native ray-tracing fallback).

        Parameters:
        -----------
        origins : np.ndarray of shape (N, 3)
            Starting coordinates of rays.
        directions : np.ndarray of shape (N, 3)
            Direction vectors of rays (need not be pre-normalized).

        Returns:
        --------
        result : dict with keys:
            - 'hit' : np.ndarray of shape (N,) bool, True where ray hit the surface.
            - 'distances' : np.ndarray of shape (N,) float, distance to intersection point.
            - 'hit_points' : np.ndarray of shape (N, 3) float, coordinates of intersection.
            - 'normals' : np.ndarray of shape (N, 3) float, surface normal at intersection.
            - 'prim_ids' : np.ndarray of shape (N,) int, triangle index of hit facet.
        """
        orig = np.asarray(origins, dtype=np.float64)
        dirs = np.asarray(directions, dtype=np.float64)
        if orig.ndim == 1:
            orig = orig[None, :]
        if dirs.ndim == 1:
            dirs = dirs[None, :]

        n_rays = len(orig)
        # Normalize directions
        dir_norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        unit_dirs = np.where(dir_norms > 1e-14, dirs / dir_norms, dirs)

        if self._embree_scene is not None:
            # High-performance Intel Embree BVH path
            orig_f32 = orig.astype(np.float32)
            dirs_f32 = unit_dirs.astype(np.float32)
            res = self._embree_scene.run(orig_f32, dirs_f32, output=True)

            prim_ids = np.asarray(res["primID"], dtype=np.int64)
            hits = prim_ids >= 0
            tfar = np.asarray(res["tfar"], dtype=np.float64)
            tfar[~hits] = np.inf

            # Hit normals
            raw_ng = np.asarray(res["Ng"], dtype=np.float64)
            ng_norms = np.linalg.norm(raw_ng, axis=1, keepdims=True)
            normals = np.where(ng_norms > 1e-14, raw_ng / ng_norms, 0.0)

            hit_points = orig + unit_dirs * np.where(hits[:, None], tfar[:, None], 0.0)

            return {
                "hit": hits,
                "distances": tfar,
                "hit_points": hit_points,
                "normals": normals,
                "prim_ids": prim_ids,
            }
        else:
            # PyVista native fallback
            hits = np.zeros(n_rays, dtype=bool)
            tfar = np.full(n_rays, np.inf, dtype=np.float64)
            hit_points = np.zeros((n_rays, 3), dtype=np.float64)
            normals = np.zeros((n_rays, 3), dtype=np.float64)
            prim_ids = np.full(n_rays, -1, dtype=np.int64)

            for i in range(n_rays):
                ray_end = orig[i] + unit_dirs[i] * 1e6
                pts, cells = self._polydata.ray_trace(orig[i], ray_end)
                if len(pts) > 0:
                    hits[i] = True
                    hit_points[i] = pts[0]
                    prim_ids[i] = cells[0]
                    tfar[i] = np.linalg.norm(pts[0] - orig[i])
                    normals[i] = self._cell_normals[cells[0]]

            return {
                "hit": hits,
                "distances": tfar,
                "hit_points": hit_points,
                "normals": normals,
                "prim_ids": prim_ids,
            }

    def to_pyvista(self, **kwargs) -> pv.PolyData:
        return self._polydata.copy()

