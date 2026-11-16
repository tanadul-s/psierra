"""
Unit and Characterization Tests: Collision Geometry, Sign Contract, and CCD.
===========================================================================
Verifies:
1. RigidCylinder collision contract (infinite vs capped cylinder, LEGACY-DISC-05).
2. Closed manifold orientation repair and topological inside-outside classification.
3. Angle-weighted pseudonormal computation (Bærentzen & Aanæs 2005).
4. Continuous Collision Detection (CCD) & Time-of-Impact (TOI) against high-speed tunneling.
5. Conservative advancement step control.
"""

from __future__ import annotations
import numpy as np
import pytest
import pyvista as pv

from psierra.collision import (
    RigidPlane,
    RigidSphere,
    RigidCylinder,
    InfiniteCylinder,
    CappedCylinder,
    RayMeshObstacle,
    ensure_consistent_surface_orientation,
    is_inside_closed_surface,
    compute_angle_weighted_pseudonormals,
    query_closed_surface_signed_distance,
    ContinuousCollisionResult,
    ray_triangle_intersection,
    ray_plane_intersection,
    ray_sphere_intersection,
    ray_cylinder_intersection,
    compute_time_of_impact,
    conservative_advancement_step,
)
from psierra.contact.geometry import (
    CylinderGeometryProvider,
    ContactCandidate,
)


# =============================================================================
# 1. RigidCylinder Collision Contract (LEGACY-DISC-05)
# =============================================================================

class TestRigidCylinderContract:
    """Verifies infinite circular cylinder vs finite capped cylinder semantics."""

    def test_legacy_disc_05_preserved_default_infinite(self) -> None:
        """RigidCylinder defaults to capped=False (infinite cylinder), preserving LEGACY-DISC-05."""
        cylinder = RigidCylinder(center=(0, 0, 0), axis=(0, 0, 1), radius=1.0, height=10.0)
        assert cylinder.capped is False

        # Midpoint inside height span: z = 0.0, r = 1.2
        pt_mid = np.array([[1.2, 0.0, 0.0]])
        gap_mid, norm_mid, cp_mid = cylinder.query_points(pt_mid)

        # Far point far beyond height span: z = 500.0, r = 1.2
        pt_far = np.array([[1.2, 0.0, 500.0]])
        gap_far, norm_far, cp_far = cylinder.query_points(pt_far)

        # In infinite cylinder, radial gap and normal are identical regardless of z
        assert gap_mid[0] == pytest.approx(0.2, rel=1e-6)
        assert gap_far[0] == pytest.approx(gap_mid[0], rel=1e-6)
        np.testing.assert_allclose(norm_mid, norm_far, atol=1e-12)
        np.testing.assert_allclose(norm_mid[0], [1.0, 0.0, 0.0], atol=1e-12)
        assert cp_mid[0, 2] == pytest.approx(0.0)
        assert cp_far[0, 2] == pytest.approx(500.0)

    def test_infinite_cylinder_subclass(self) -> None:
        """InfiniteCylinder explicitly instantiates capped=False."""
        inf_cyl = InfiniteCylinder(center=(0, 0, 0), axis=(0, 1, 0), radius=2.0)
        assert inf_cyl.capped is False
        assert inf_cyl.radius == 2.0

    def test_capped_cylinder_barrel_outside(self) -> None:
        """Capped cylinder queries outside barrel region (|z| <= h, r >= R)."""
        # Height = 10 -> h = 5.0, Radius = 2.0 along z-axis
        cyl = CappedCylinder(center=(0, 0, 0), axis=(0, 0, 1), radius=2.0, height=10.0)
        assert cyl.capped is True

        # Point at z = 3.0 (|z| < 5), r = 3.5 (x = 3.5, y = 0.0)
        pts = np.array([[3.5, 0.0, 3.0]])
        gaps, normals, cps = cyl.query_points(pts)

        assert gaps[0] == pytest.approx(1.5, rel=1e-6)
        np.testing.assert_allclose(normals[0], [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(cps[0], [2.0, 0.0, 3.0], atol=1e-12)

    def test_capped_cylinder_caps_outside(self) -> None:
        """Capped cylinder queries outside end cap region (|z| > h, r <= R)."""
        cyl = CappedCylinder(center=(0, 0, 0), axis=(0, 0, 1), radius=2.0, height=10.0)
        h = 5.0

        # Point above top cap: z = 8.0, x = 1.0, y = 0.0 (r = 1.0 <= R)
        pt_top = np.array([[1.0, 0.0, 8.0]])
        gap_top, norm_top, cp_top = cyl.query_points(pt_top)

        assert gap_top[0] == pytest.approx(3.0, rel=1e-6)  # 8.0 - 5.0
        np.testing.assert_allclose(norm_top[0], [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(cp_top[0], [1.0, 0.0, 5.0], atol=1e-12)

        # Point below bottom cap: z = -7.5, x = 0.5, y = 0.0 (r = 0.5 <= R)
        pt_bot = np.array([[0.5, 0.0, -7.5]])
        gap_bot, norm_bot, cp_bot = cyl.query_points(pt_bot)

        assert gap_bot[0] == pytest.approx(2.5, rel=1e-6)  # |-7.5| - 5.0
        np.testing.assert_allclose(norm_bot[0], [0.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(cp_bot[0], [0.5, 0.0, -5.0], atol=1e-12)

    def test_capped_cylinder_rim_outside(self) -> None:
        """Capped cylinder queries outside corner rim circle (|z| > h, r > R)."""
        cyl = CappedCylinder(center=(0, 0, 0), axis=(0, 0, 1), radius=2.0, height=10.0)
        # h = 5.0, R = 2.0
        # Point at z = 8.0 (dz = 3.0), x = 6.0 (r = 6.0, dr = 4.0), y = 0.0
        pt = np.array([[6.0, 0.0, 8.0]])
        gaps, normals, cps = cyl.query_points(pt)

        # Expected gap = sqrt(dr^2 + dz^2) = sqrt(4^2 + 3^2) = 5.0
        assert gaps[0] == pytest.approx(5.0, rel=1e-6)
        # Closest point on rim circle at z = 5.0, r = 2.0
        np.testing.assert_allclose(cps[0], [2.0, 0.0, 5.0], atol=1e-12)
        # Normal = (pt - cp) / gap = [4/5, 0, 3/5] = [0.8, 0.0, 0.6]
        np.testing.assert_allclose(normals[0], [0.8, 0.0, 0.6], atol=1e-12)

    def test_capped_cylinder_inside_penetration(self) -> None:
        """Capped cylinder queries strictly inside volume (g < 0)."""
        cyl = CappedCylinder(center=(0, 0, 0), axis=(0, 0, 1), radius=2.0, height=10.0)
        # h = 5.0, R = 2.0

        # Point closer to barrel: z = 0.0, x = 1.7 (d_barrel = 0.3, d_cap = 5.0)
        pt_barrel = np.array([[1.7, 0.0, 0.0]])
        gap_b, norm_b, cp_b = cyl.query_points(pt_barrel)
        assert gap_b[0] == pytest.approx(-0.3, rel=1e-6)
        np.testing.assert_allclose(norm_b[0], [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(cp_b[0], [2.0, 0.0, 0.0], atol=1e-12)

        # Point closer to top cap: z = 4.8, x = 0.5 (d_barrel = 1.5, d_cap = 0.2)
        pt_cap = np.array([[0.5, 0.0, 4.8]])
        gap_c, norm_c, cp_c = cyl.query_points(pt_cap)
        assert gap_c[0] == pytest.approx(-0.2, rel=1e-6)
        np.testing.assert_allclose(norm_c[0], [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(cp_c[0], [0.5, 0.0, 5.0], atol=1e-12)

    def test_cylinder_geometry_provider_capped_derivatives(self) -> None:
        """Verifies CylinderGeometryProvider exact differentiation under capped=True."""
        provider = CylinderGeometryProvider(
            center=(0, 0, 0), axis=(0, 0, 1), radius=2.0, height=10.0, capped=True
        )
        assert provider.capped is True
        cand = ContactCandidate(candidate_index=0, slave_node=0, master_feature_id="cylinder")

        # 1. Barrel evaluation
        geom_b = provider.evaluate_geometry(cand, np.array([3.0, 0.0, 1.0]))
        assert geom_b.feature_id == "cylinder_barrel"
        assert geom_b.gap == pytest.approx(1.0, rel=1e-6)
        np.testing.assert_allclose(geom_b.normal, [1.0, 0.0, 0.0], atol=1e-12)
        # Barrel Hessian has curvature only along transverse direction (y-axis)
        assert geom_b.hessian[1, 1] == pytest.approx(1.0 / 3.0, rel=1e-6)
        assert geom_b.hessian[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert geom_b.hessian[2, 2] == pytest.approx(0.0, abs=1e-12)

        # 2. Cap evaluation
        geom_c = provider.evaluate_geometry(cand, np.array([1.0, 0.0, 7.0]))
        assert geom_c.feature_id == "cylinder_cap"
        assert geom_c.gap == pytest.approx(2.0, rel=1e-6)
        np.testing.assert_allclose(geom_c.normal, [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(geom_c.hessian, 0.0, atol=1e-12)


# =============================================================================
# 2. Mesh Orientation Validation & Inside-Outside Classification
# =============================================================================

class TestManifoldOrientationAndInsideOutside:
    """Verifies consistent orientation repair and topological inside-outside classification."""

    def test_ensure_consistent_surface_orientation_sphere(self) -> None:
        """Verifies orientation repair turns inverted closed spheres outward."""
        sphere = pv.Sphere(radius=2.0, theta_resolution=20, phi_resolution=20)
        # Deliberately invert faces
        inverted_sphere = sphere.flip_faces(inplace=False)

        # Repair
        repaired = ensure_consistent_surface_orientation(inverted_sphere, force_outward=True)

        assert repaired.n_cells == sphere.n_cells
        assert repaired.n_open_edges == 0

        # Points at center should be strictly inside
        inside = is_inside_closed_surface(np.array([[0.0, 0.0, 0.0]]), repaired)
        assert bool(inside[0]) is True

    def test_inside_outside_classification_cube(self) -> None:
        """Topological inside-outside classification for closed cube."""
        cube = pv.Cube(center=(0.0, 0.0, 0.0), x_length=2.0, y_length=2.0, z_length=2.0).triangulate()
        cube = ensure_consistent_surface_orientation(cube)

        pts = np.array([
            [0.0, 0.0, 0.0],    # Center (inside)
            [0.9, 0.9, 0.9],    # Near corner (inside)
            [1.5, 0.0, 0.0],    # Outside +x
            [0.0, -2.0, 0.0],   # Outside -y
            [0.0, 0.0, 3.0],    # Outside +z
        ])

        inside = is_inside_closed_surface(pts, cube)
        np.testing.assert_array_equal(inside, [True, True, False, False, False])

        # Test query_closed_surface_signed_distance
        gaps, normals, cps = query_closed_surface_signed_distance(pts, cube)
        assert gaps[0] < 0.0  # Inside
        assert gaps[1] < 0.0  # Inside
        assert gaps[2] == pytest.approx(0.5, rel=1e-3)  # 1.5 - 1.0 = 0.5
        assert gaps[3] == pytest.approx(1.0, rel=1e-3)  # |-2.0| - 1.0 = 1.0
        assert gaps[4] == pytest.approx(2.0, rel=1e-3)  # 3.0 - 1.0 = 2.0

    def test_angle_weighted_pseudonormals_cube(self) -> None:
        """Angle-weighted vertex and edge pseudonormals on a closed cube."""
        cube = pv.Cube(center=(0.0, 0.0, 0.0), x_length=2.0, y_length=2.0, z_length=2.0).triangulate()
        cube = ensure_consistent_surface_orientation(cube)

        v_pseudonormals, edge_pseudonormals = compute_angle_weighted_pseudonormals(cube)

        assert v_pseudonormals.shape == (cube.n_points, 3)
        # All vertex pseudonormals must be unit vectors pointing outward
        norms = np.linalg.norm(v_pseudonormals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

        # For points in the octant x > 0, y > 0, z > 0, outward normal must have positive components
        pts = np.asarray(cube.points)
        corner_mask = (pts[:, 0] > 0.9) & (pts[:, 1] > 0.9) & (pts[:, 2] > 0.9)
        if np.any(corner_mask):
            corner_n = v_pseudonormals[corner_mask][0]
            assert corner_n[0] > 0.0 and corner_n[1] > 0.0 and corner_n[2] > 0.0

    def test_ray_mesh_obstacle_watertight_sign(self) -> None:
        """RayMeshObstacle with use_watertight_sign=True."""
        sphere = pv.Sphere(center=(0, 0, 0), radius=2.0).triangulate()
        obs = RayMeshObstacle(sphere, use_embree=False, use_watertight_sign=True)

        pts = np.array([
            [0.0, 0.0, 0.0],  # Inside: gap = -2.0
            [3.0, 0.0, 0.0],  # Outside: gap = +1.0
        ])
        gaps, normals, cps = obs.query_points(pts)

        assert gaps[0] == pytest.approx(-2.0, rel=1e-2)
        assert gaps[1] == pytest.approx(1.0, rel=1e-2)


# =============================================================================
# 3. Continuous Collision Detection (CCD) & High-Speed Tunneling
# =============================================================================

class TestContinuousCollisionDetection:
    """Verifies Time-of-Impact (TOI) and anti-tunneling conservative advancement."""

    def test_ray_triangle_intersection_moller_trumbore(self) -> None:
        """Möller-Trumbore ray-triangle intersection accuracy."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([2.0, 0.0, 0.0])
        v2 = np.array([0.0, 2.0, 0.0])

        # 1. Direct hit at centroid: ray moves from z = 1.0 to z = -1.0
        orig = np.array([0.5, 0.5, 1.0])
        delta = np.array([0.0, 0.0, -2.0])
        hit, t, u, v = ray_triangle_intersection(orig, delta, v0, v1, v2)

        assert hit is True
        assert t == pytest.approx(0.5, rel=1e-6)  # Impact at z = 0 (t = 0.5)
        assert u == pytest.approx(0.25, rel=1e-6)
        assert v == pytest.approx(0.25, rel=1e-6)

        # 2. Miss outside hypotenuse
        orig_miss = np.array([1.5, 1.5, 1.0])
        hit_m, t_m, _, _ = ray_triangle_intersection(orig_miss, delta, v0, v1, v2)
        assert hit_m is False
        assert t_m == 1.0

        # 3. Parallel ray (no intersection)
        delta_parallel = np.array([1.0, 0.0, 0.0])
        hit_p, t_p, _, _ = ray_triangle_intersection(orig, delta_parallel, v0, v1, v2)
        assert hit_p is False
        assert t_p == 1.0

        # 4. Backward ray (moving away from triangle)
        delta_backward = np.array([0.0, 0.0, 2.0])
        hit_b, t_b, _, _ = ray_triangle_intersection(orig, delta_backward, v0, v1, v2)
        assert hit_b is False
        assert t_b == 1.0

    def test_high_speed_tunneling_planar_obstacle(self) -> None:
        """High-speed tunneling through a thin planar obstacle."""
        plane = RigidPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))

        # Node starts at z = 0.5 (outside, g = 0.5)
        # Node moves with large displacement delta_z = -5.0 to z = -4.5 (passed completely through)
        x_start = np.array([[1.0, 2.0, 0.5]])
        x_end = np.array([[1.0, 2.0, -4.5]])

        # CCD detection
        res = compute_time_of_impact(x_start, x_end, plane)
        assert res.has_impact is True
        # Impact occurs at z = 0: 0.5 + t * (-5.0) = 0 => t = 0.1
        assert res.time_of_impact == pytest.approx(0.1, rel=1e-6)
        np.testing.assert_allclose(res.impact_point, [1.0, 2.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(res.normal, [0.0, 0.0, 1.0], atol=1e-12)

        # Conservative advancement with safety factor 0.8
        alpha, x_adv, _ = conservative_advancement_step(x_start, x_end, plane, safety_factor=0.8)
        assert alpha == pytest.approx(0.08, rel=1e-6)  # 0.8 * 0.1
        # Advanced position: z = 0.5 + 0.08 * (-5.0) = 0.1 > 0 (strictly separated!)
        assert x_adv[0, 2] == pytest.approx(0.1, rel=1e-6)

    def test_high_speed_tunneling_sphere_obstacle(self) -> None:
        """High-speed tunneling through a rigid sphere."""
        sphere = RigidSphere(center=(0.0, 0.0, 0.0), radius=2.0)

        # Node flies directly through sphere: from x = -10.0 to x = +10.0
        x_start = np.array([[-10.0, 0.0, 0.0]])
        x_end = np.array([[+10.0, 0.0, 0.0]])

        res = compute_time_of_impact(x_start, x_end, sphere)
        assert res.has_impact is True
        # Impact at sphere entry x = -2.0: -10 + t * 20 = -2 => 20 t = 8 => t = 0.4
        assert res.time_of_impact == pytest.approx(0.4, rel=1e-6)
        np.testing.assert_allclose(res.impact_point, [-2.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(res.normal, [-1.0, 0.0, 0.0], atol=1e-12)

    def test_high_speed_tunneling_cylinder_obstacle(self) -> None:
        """High-speed tunneling through capped cylinder cap and barrel."""
        # Cylinder aligned with z-axis, R = 2.0, height = 6.0 (h = 3.0)
        cyl = CappedCylinder(center=(0, 0, 0), axis=(0, 0, 1), radius=2.0, height=6.0)

        # 1. Flying down through top cap from z = 8.0 to z = -8.0 along axis
        x_start = np.array([[0.5, 0.0, 8.0]])
        x_end = np.array([[0.5, 0.0, -8.0]])

        res_cap = compute_time_of_impact(x_start, x_end, cyl)
        assert res_cap.has_impact is True
        # Cap at z = 3.0: 8.0 + t * (-16.0) = 3.0 => 16 t = 5 => t = 5/16 = 0.3125
        assert res_cap.time_of_impact == pytest.approx(0.3125, rel=1e-6)
        np.testing.assert_allclose(res_cap.impact_point, [0.5, 0.0, 3.0], atol=1e-12)
        np.testing.assert_allclose(res_cap.normal, [0.0, 0.0, 1.0], atol=1e-12)

        # 2. Flying horizontally through barrel from x = -10.0 to x = +10.0 at z = 0.0
        x_start_b = np.array([[-10.0, 0.0, 0.0]])
        x_end_b = np.array([[+10.0, 0.0, 0.0]])

        res_bar = compute_time_of_impact(x_start_b, x_end_b, cyl)
        assert res_bar.has_impact is True
        # Barrel hit at x = -2.0: -10 + t * 20 = -2 => t = 0.4
        assert res_bar.time_of_impact == pytest.approx(0.4, rel=1e-6)
        np.testing.assert_allclose(res_bar.impact_point, [-2.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(res_bar.normal, [-1.0, 0.0, 0.0], atol=1e-12)

    def test_high_speed_tunneling_mesh_surface(self) -> None:
        """High-speed tunneling through a triangular mesh surface."""
        # Simple planar square mesh of 2 triangles in xy plane at z = 0
        v = np.array([
            [-5.0, -5.0, 0.0],
            [5.0, -5.0, 0.0],
            [5.0, 5.0, 0.0],
            [-5.0, 5.0, 0.0],
        ])
        faces = np.array([
            [3, 0, 1, 2],
            [3, 0, 2, 3],
        ])
        mesh = pv.PolyData(v, faces)

        # Node moving rapidly through mesh: z = 2.0 to z = -3.0
        x_start = np.array([[1.0, 1.0, 2.0]])
        x_end = np.array([[1.0, 1.0, -3.0]])

        res = compute_time_of_impact(x_start, x_end, mesh)
        assert res.has_impact is True
        # Impact at z = 0: 2.0 + t * (-5.0) = 0 => t = 0.4
        assert res.time_of_impact == pytest.approx(0.4, rel=1e-6)
        np.testing.assert_allclose(res.impact_point, [1.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(res.normal, [0.0, 0.0, 1.0], atol=1e-12)

    def test_multi_node_ccd_and_conservative_advancement(self) -> None:
        """Verifies multi-node simultaneous advance selects earliest TOI across all nodes."""
        plane = RigidPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))

        # 5 moving slave nodes
        # Node 0: stays far away (y = 5.0 -> 4.0)
        # Node 1: moving away (y = 2.0 -> 3.0)
        # Node 2: hits at y = 0 with t = 0.25 (y = 1.0 -> -3.0)
        # Node 3: hits at y = 0 with t = 0.50 (y = 2.0 -> -2.0)
        # Node 4: parallel motion at y = 3.0
        x_start = np.array([
            [0.0, 5.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 2.0, 0.0],
            [4.0, 3.0, 0.0],
        ])
        x_end = np.array([
            [0.0, 4.0, 0.0],
            [1.0, 3.0, 0.0],
            [2.0, -3.0, 0.0],
            [3.0, -2.0, 0.0],
            [4.0, 3.0, 1.0],
        ])

        res = compute_time_of_impact(x_start, x_end, plane)
        assert res.has_impact is True
        assert res.node_index == 2
        assert res.time_of_impact == pytest.approx(0.25, rel=1e-6)
        np.testing.assert_allclose(res.impact_point, [2.0, 0.0, 0.0], atol=1e-12)

        # Conservative advancement with safety_factor = 0.9
        alpha, x_adv, _ = conservative_advancement_step(x_start, x_end, plane, safety_factor=0.9)
        assert alpha == pytest.approx(0.9 * 0.25, rel=1e-6)
        # Node 2 should be at y = 1.0 + 0.225 * (-4.0) = 0.1 > 0
        assert x_adv[2, 1] == pytest.approx(0.1, rel=1e-6)
        # Node 3 should be at y = 2.0 + 0.225 * (-4.0) = 1.1 > 0
        assert x_adv[3, 1] == pytest.approx(1.1, rel=1e-6)
        # All nodes remain on positive side of plane
        assert np.all(x_adv[:, 1] > 0.0)

    def test_concave_pocket_non_convex_manifold(self) -> None:
        """Verifies topological sign classification in concave exterior pockets of non-convex meshes."""
        # Create an L-shaped concave manifold by extruding a 2D L-shape polygon
        # Points: (0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)
        poly_pts = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ])
        poly = pv.PolyData(poly_pts, [6, 0, 1, 2, 3, 4, 5])
        l_shape = poly.triangulate().extrude((0.0, 0.0, 1.0), capping=True).triangulate().clean()
        l_shape = ensure_consistent_surface_orientation(l_shape)

        assert l_shape.n_open_edges == 0

        # Query points:
        # Point A: (0.5, 0.5, 0.5) inside the solid L body
        # Point B: (1.5, 1.5, 0.5) inside the concave pocket (exterior, facing two inner walls!)
        # Point C: (-1.0, -1.0, 0.5) far outside
        pts = np.array([
            [0.5, 0.5, 0.5],
            [1.5, 1.5, 0.5],
            [-1.0, -1.0, 0.5],
        ])

        inside = is_inside_closed_surface(pts, l_shape)
        assert bool(inside[0]) is True   # Inside solid
        assert bool(inside[1]) is False  # In concave pocket -> OUTSIDE!
        assert bool(inside[2]) is False  # Far exterior -> OUTSIDE!

        gaps, normals, cps = query_closed_surface_signed_distance(pts, l_shape)
        assert gaps[0] < 0.0  # Penetrating inside solid
        assert gaps[1] > 0.0  # Positive separation in concave exterior pocket!
        assert gaps[2] > 0.0  # Positive separation far outside

        # In the concave pocket at (1.5, 1.5, 0.5), closest walls are at x = 1.0 or y = 1.0 (dist = 0.5)
        assert gaps[1] == pytest.approx(0.5, rel=1e-2)
