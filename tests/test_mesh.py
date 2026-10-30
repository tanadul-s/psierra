from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from psierra.mesh.volumetric import VolumetricMesh
from psierra.mesh.surface import SurfaceMesh


class TestMeshModule:
    """Verifies volumetric and surface mesh representations, properties, and geometric extraction."""

    def test_five_tet_cube_mesh_properties(self, five_tet_cube_mesh: VolumetricMesh) -> None:
        mesh = five_tet_cube_mesh

        assert mesh.n_nodes == 8
        assert mesh.n_elements == 5
        assert mesh.n_dofs == 24

        # Total volume of [0, 1]^3 cube must be exactly 1.0
        assert np.isclose(mesh.volume, 1.0, atol=1e-12)

        # Centroid of unit cube is (0.5, 0.5, 0.5)
        assert np.allclose(mesh.centroid, np.array([0.5, 0.5, 0.5]), atol=1e-12)

    def test_surface_extraction_from_volumetric(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        mesh = five_tet_cube_mesh
        surface = mesh.surface

        assert isinstance(surface, SurfaceMesh)
        # Unit cube has 6 square faces, each split into 2 triangles = 12 boundary triangles
        assert surface.n_elements == 12

        # Surface area of unit cube = 6 * 1.0 = 6.0
        assert np.isclose(surface.area, 6.0, atol=1e-12)

        # Mapping test: surface nodes must map identically to volumetric nodes
        assert surface.node_indices is not None
        mapped_nodes = mesh.nodes[surface.node_indices]
        assert np.allclose(mapped_nodes, surface.nodes, atol=1e-14)

    def test_mesh_transform(self, five_tet_cube_mesh: VolumetricMesh) -> None:
        mesh = five_tet_cube_mesh.copy()
        orig_volume = mesh.volume

        # Rotate by 90 degrees about Z and translate by [10, 20, 30]
        rot = Rotation.from_euler("z", 90, degrees=True)
        mesh.transform(orientation=rot, translation=np.array([10.0, 20.0, 30.0]))

        # Volume must be invariant under rigid transformation
        assert np.isclose(mesh.volume, orig_volume, atol=1e-12)

        # New centroid should be rotated [0.5, 0.5, 0.5] + translation
        expected_centroid = rot.apply(np.array([0.5, 0.5, 0.5])) + np.array([10.0, 20.0, 30.0])
        assert np.allclose(mesh.centroid, expected_centroid, atol=1e-12)

    def test_real_vtk_mesh_load(self, beam_vtk_path: Path) -> None:
        if not beam_vtk_path.exists():
            pytest.skip("beam1.vtk not found")

        mesh = VolumetricMesh(beam_vtk_path)
        assert mesh.n_nodes > 0
        assert mesh.n_elements > 0
        assert mesh.volume > 0.0

        surface = mesh.surface
        assert surface.n_elements > 0
        assert surface.area > 0.0
