from __future__ import annotations
from pathlib import Path
import tempfile
import numpy as np
import pytest
import pyvista as pv
import meshio

from psierra.mesher import (
    TetrahedralMesher,
    TetMesher,
    generate_tetrahedra,
    upgrade_tet4_to_tet10,
    downgrade_tet10_to_tet4,
    compute_tet10_connectivity_and_points,
    read_surface,
    export_mesh,
    import_mesh,
    is_gmsh_available,
    read_step,
)
from psierra.mesh.volumetric import VolumetricMesh


class TestTetrahedralMesher:
    """Verifies tetrahedral mesher, Tet4-to-Tet10 upgrading, and IO."""

    @pytest.fixture
    def test_sphere_surface(self) -> pv.PolyData:
        """Generates a closed, manifold sphere surface."""
        return pv.Sphere(radius=10.0, theta_resolution=16, phi_resolution=16)

    @pytest.fixture
    def single_tet4_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """A canonical single linear tetrahedron."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.8660254, 0.0],
            [0.5, 0.2886751, 0.8164966],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        return nodes, elements

    def test_single_tet_upgrade_to_tet10(self, single_tet4_arrays: tuple[np.ndarray, np.ndarray]) -> None:
        nodes, elements = single_tet4_arrays
        nodes10, elements10 = compute_tet10_connectivity_and_points(nodes, elements)

        assert elements10.shape == (1, 10)
        assert len(nodes10) == 10  # 4 corners + 6 mid-edge nodes

        # Check corner nodes preserved
        assert np.allclose(nodes10[:4], nodes)
        assert np.array_equal(elements10[0, :4], [0, 1, 2, 3])

        # Check canonical VTK mid-edge geometric positions
        p0, p1, p2, p3 = nodes[0], nodes[1], nodes[2], nodes[3]
        n4, n5, n6, n7, n8, n9 = [nodes10[idx] for idx in elements10[0, 4:]]

        assert np.allclose(n4, 0.5 * (p0 + p1), atol=1e-12)  # Edge (0, 1)
        assert np.allclose(n5, 0.5 * (p1 + p2), atol=1e-12)  # Edge (1, 2)
        assert np.allclose(n6, 0.5 * (p2 + p0), atol=1e-12)  # Edge (2, 0)
        assert np.allclose(n7, 0.5 * (p0 + p3), atol=1e-12)  # Edge (0, 3)
        assert np.allclose(n8, 0.5 * (p1 + p3), atol=1e-12)  # Edge (1, 3)
        assert np.allclose(n9, 0.5 * (p2 + p3), atol=1e-12)  # Edge (2, 3)

    def test_point_and_cell_data_preservation_on_upgrade(
        self, single_tet4_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        nodes, elements = single_tet4_arrays
        grid = pv.UnstructuredGrid({pv.CellType.TETRA: elements}, nodes)

        # Attach test nodal field: f(x, y, z) = x + 2*y + 3*z
        test_field = nodes[:, 0] + 2.0 * nodes[:, 1] + 3.0 * nodes[:, 2]
        grid.point_data["TestLinearField"] = test_field
        grid.cell_data["ElementTag"] = np.array([42])

        grid10 = upgrade_tet4_to_tet10(grid)
        assert isinstance(grid10, pv.UnstructuredGrid)
        assert grid10.distinct_cell_types == {pv.CellType.QUADRATIC_TETRA}
        assert grid10.cell_data["ElementTag"][0] == 42

        # Verify mid-edge nodal values match analytical linear interpolation
        coords10 = grid10.points
        expected_field = coords10[:, 0] + 2.0 * coords10[:, 1] + 3.0 * coords10[:, 2]
        assert np.allclose(grid10.point_data["TestLinearField"], expected_field, atol=1e-12)

    def test_pipeline_generate_tet4(self, test_sphere_surface: pv.PolyData) -> None:
        mesher = TetrahedralMesher(order=1, edge_length_fac=0.2, quiet=True)
        vmesh = mesher.generate(test_sphere_surface)

        assert isinstance(vmesh, VolumetricMesh)
        assert vmesh.order == 1
        assert vmesh.cell_type == pv.CellType.TETRA
        assert vmesh.nodes_per_element == 4
        assert vmesh.n_elements > 0
        assert vmesh.n_nodes > 0

        # Sphere of radius 10: volume = 4/3 * pi * 10^3 ~ 4188.79
        expected_vol = (4.0 / 3.0) * np.pi * (10.0**3)
        assert np.isclose(vmesh.volume, expected_vol, rtol=0.15)

        # Verify all scaled Jacobians are positive
        qualities = vmesh.compute_qualities("scaled_jacobian")
        assert np.all(qualities > 0.0)

    def test_pipeline_generate_tet10(self, test_sphere_surface: pv.PolyData) -> None:
        vmesh10 = generate_tetrahedra(test_sphere_surface, order=2, edge_length_fac=0.2, quiet=True)

        assert isinstance(vmesh10, VolumetricMesh)
        assert vmesh10.order == 2
        assert vmesh10.cell_type == pv.CellType.QUADRATIC_TETRA
        assert vmesh10.nodes_per_element == 10
        assert vmesh10.elements.shape[1] == 10

        # Base geometric volume must be close to sphere volume
        expected_vol = (4.0 / 3.0) * np.pi * (10.0**3)
        assert np.isclose(vmesh10.volume, expected_vol, rtol=0.15)

        # Scaled Jacobians evaluated on element corners must be positive
        qualities = vmesh10.compute_qualities("scaled_jacobian")
        assert np.all(qualities > 0.0)

    def test_vmesh_to_quadratic_preserves_corners_and_midpoints(
        self, test_sphere_surface: pv.PolyData
    ) -> None:
        # Also test TetMesher alias
        mesher = TetMesher(order=1, edge_length_fac=0.2, quiet=True)
        vmesh4 = mesher.generate(test_sphere_surface)

        # Upgrade using to_quadratic()
        vmesh10 = vmesh4.to_quadratic()
        assert vmesh10.n_elements == vmesh4.n_elements
        assert np.allclose(vmesh10.corner_nodes, vmesh4.nodes)
        assert np.array_equal(vmesh10.elements[:, :4], vmesh4.elements)
        assert len(vmesh10.corner_nodes) == vmesh4.n_nodes
        assert len(vmesh10.midedge_nodes) == vmesh10.n_nodes - vmesh4.n_nodes

        # Check mid-edge coordinates for all elements
        c10 = vmesh10.elements
        pts = vmesh10.nodes
        for i in range(len(c10)):
            n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = c10[i]
            p0, p1, p2, p3 = pts[n0], pts[n1], pts[n2], pts[n3]
            assert np.allclose(pts[n4], 0.5 * (p0 + p1), atol=1e-12)
            assert np.allclose(pts[n5], 0.5 * (p1 + p2), atol=1e-12)
            assert np.allclose(pts[n6], 0.5 * (p2 + p0), atol=1e-12)
            assert np.allclose(pts[n7], 0.5 * (p0 + p3), atol=1e-12)
            assert np.allclose(pts[n8], 0.5 * (p1 + p3), atol=1e-12)
            assert np.allclose(pts[n9], 0.5 * (p2 + p3), atol=1e-12)

        # Downgrade back to linear
        vmesh_back = vmesh10.to_linear()
        assert vmesh_back.n_elements == vmesh4.n_elements
        assert vmesh_back.n_nodes == vmesh4.n_nodes
        assert np.array_equal(vmesh_back.elements, vmesh4.elements)

    def test_pipeline_generate_from_existing_vmesh_does_not_remesh(
        self, test_sphere_surface: pv.PolyData
    ) -> None:
        vmesh4 = TetrahedralMesher(order=1, edge_length_fac=0.2, quiet=True).generate(test_sphere_surface)

        # Pass existing VolumetricMesh to mesher with order=2: must upgrade, NOT re-mesh!
        mesher10 = TetrahedralMesher(order=2, quiet=True)
        vmesh10 = mesher10.generate(vmesh4)

        assert vmesh10.n_elements == vmesh4.n_elements
        assert np.allclose(vmesh10.corner_nodes, vmesh4.nodes)
        assert np.array_equal(vmesh10.elements[:, :4], vmesh4.elements)

    def test_stl_file_pipeline_and_meshio_roundtrip(self, test_sphere_surface: pv.PolyData) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = Path(tmpdir) / "sphere.stl"
            vtu_path = Path(tmpdir) / "mesh_tet10.vtu"
            msh_path = Path(tmpdir) / "mesh_tet10.msh"

            # 1. Save STL surface
            test_sphere_surface.save(str(stl_path))

            # 2. Run pipeline directly from STL file path
            vmesh10 = generate_tetrahedra(stl_path, order=2, edge_length_fac=0.2, quiet=True)
            assert vmesh10.order == 2

            # 3. Export to VTU with PyVista backend
            export_mesh(vmesh10, vtu_path, backend="pyvista")
            assert vtu_path.exists()

            # 4. Read back VTU
            imported_vtu = import_mesh(vtu_path)
            assert imported_vtu.n_cells == vmesh10.n_elements
            assert imported_vtu.n_points == vmesh10.n_nodes
            assert pv.CellType.QUADRATIC_TETRA in imported_vtu.distinct_cell_types

            # 5. Export to MSH with Meshio backend
            export_mesh(vmesh10, msh_path, backend="meshio")
            assert msh_path.exists()

            # 6. Read back MSH with Meshio
            m = meshio.read(str(msh_path))
            assert any(cell_block.type == "tetra10" for cell_block in m.cells)

    def test_step_reader_error_or_import(self) -> None:
        # If gmsh is not installed, attempting to read a step file must raise an informative ImportError
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_step = Path(tmpdir) / "model.step"
            dummy_step.write_text("ISO-10303-21; HEADER; ENDSEC; DATA; ENDSEC; END-ISO-10303-21;")

            if not is_gmsh_available():
                with pytest.raises(ImportError, match="requires the 'gmsh' library"):
                    read_step(dummy_step)

    def test_pipeline_invalid_order_raises(self) -> None:
        with pytest.raises(ValueError, match="Order must be 1"):
            TetrahedralMesher(order=3)  # type: ignore[arg-type]

    def test_rcm_reordering_reduces_bandwidth(self, test_sphere_surface: pv.PolyData) -> None:
        # Generate Tet10 mesh with appended mid-edge nodes
        vmesh10 = TetrahedralMesher(order=2, edge_length_fac=0.2, quiet=True).generate(test_sphere_surface)
        vmesh10["OriginalX"] = vmesh10.nodes[:, 0]

        bw_orig = vmesh10.bandwidth
        assert bw_orig > 0

        # Apply RCM renumbering
        vmesh_rcm = vmesh10.reorder_nodes("rcm")
        bw_rcm = vmesh_rcm.bandwidth

        # Bandwidth must be significantly reduced
        assert bw_rcm < bw_orig
        assert bw_orig / bw_rcm > 1.5

        # Volume and centroids must be identical
        assert np.isclose(vmesh_rcm.volume, vmesh10.volume, atol=1e-12)
        assert np.allclose(vmesh_rcm.centroid, vmesh10.centroid, atol=1e-12)

        # Point data array must be permuted consistently with coordinates
        assert np.allclose(vmesh_rcm["OriginalX"], vmesh_rcm.nodes[:, 0], atol=1e-12)

    def test_mesher_auto_reorder_flag(self, test_sphere_surface: pv.PolyData) -> None:
        vmesh_unreordered = TetrahedralMesher(order=2, edge_length_fac=0.2, reorder=False, quiet=True).generate(test_sphere_surface)
        vmesh_reordered = TetrahedralMesher(order=2, edge_length_fac=0.2, reorder=True, quiet=True).generate(test_sphere_surface)

        assert vmesh_reordered.bandwidth < vmesh_unreordered.bandwidth

