from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
import pyvista as pv

from psierra.material import IsotropicElasticParameters, Hookean, NeoHookean
from psierra.element import Tet4Linear
from psierra.mesh import VolumetricMesh, SurfaceMesh
from psierra.domain import SolidDomain


@pytest.fixture
def regular_tet_coords() -> np.ndarray:
    """
    Returns reference coordinates of a standard unit right tetrahedron:
    Nodes: (0,0,0), (1,0,0), (0,1,0), (0,0,1).
    Volume = 1/6.
    """
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def inverted_tet_coords() -> np.ndarray:
    """
    Returns coordinates with inverted node ordering (negative Jacobian).
    """
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def hookean_material() -> Hookean:
    """Steel-like isotropic linear elastic material (E = 200 GPa, nu = 0.3, rho = 7850 kg/m^3)."""
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=200e9,
        poissons_ratio=0.3,
        mass_density=7850.0,
    )
    return Hookean(params)


@pytest.fixture
def neohookean_material() -> NeoHookean:
    """Soft tissue-like Neo-Hookean material (E = 10 kPa, nu = 0.45, rho = 1000 kg/m^3)."""
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=10e3,
        poissons_ratio=0.45,
        mass_density=1000.0,
    )
    return NeoHookean(params)


@pytest.fixture
def single_element_mesh(regular_tet_coords: np.ndarray) -> VolumetricMesh:
    """Creates a procedural single-tetrahedron VolumetricMesh."""
    cells = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    cell_types = np.array([pv.CellType.TETRA], dtype=np.uint8)
    ugrid = pv.UnstructuredGrid(cells, cell_types, regular_tet_coords)
    return VolumetricMesh(ugrid)


@pytest.fixture
def five_tet_cube_mesh() -> VolumetricMesh:
    """
    Creates a unit cube [0, 1]^3 discretized into 5 tetrahedra.
    Total Volume = 1.0. 8 vertices.
    """
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
        ],
        dtype=np.float64,
    )
    tets = [
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
    ]
    cells = []
    for tet in tets:
        cells.extend([4] + tet)
    cells_array = np.array(cells, dtype=np.int64)
    cell_types = np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8)
    ugrid = pv.UnstructuredGrid(cells_array, cell_types, nodes)
    return VolumetricMesh(ugrid)


@pytest.fixture
def beam_vtk_path() -> Path:
    """Path to the benchmark cantilever beam mesh."""
    path = Path(__file__).resolve().parent.parent / "data" / "beam1" / "mesh" / "beam1.vtk"
    return path


@pytest.fixture
def regular_tet10_coords(regular_tet_coords: np.ndarray) -> np.ndarray:
    """
    Returns reference coordinates of a standard unit right quadratic tetrahedron (10 nodes).
    Corner nodes 0-3: (0,0,0), (1,0,0), (0,1,0), (0,0,1).
    Mid-edge nodes 4-9:
      - 4: (0.5, 0, 0)   [edge 0-1]
      - 5: (0.5, 0.5, 0) [edge 1-2]
      - 6: (0, 0.5, 0)   [edge 2-0]
      - 7: (0, 0, 0.5)   [edge 0-3]
      - 8: (0.5, 0, 0.5) [edge 1-3]
      - 9: (0, 0.5, 0.5) [edge 2-3]
    """
    v = regular_tet_coords
    coords = np.zeros((10, 3), dtype=np.float64)
    coords[0:4] = v
    coords[4] = 0.5 * (v[0] + v[1])
    coords[5] = 0.5 * (v[1] + v[2])
    coords[6] = 0.5 * (v[2] + v[0])
    coords[7] = 0.5 * (v[0] + v[3])
    coords[8] = 0.5 * (v[1] + v[3])
    coords[9] = 0.5 * (v[2] + v[3])
    return coords


@pytest.fixture
def inverted_tet10_coords(inverted_tet_coords: np.ndarray) -> np.ndarray:
    """Returns coordinates of an inverted 10-node tetrahedron."""
    v = inverted_tet_coords
    coords = np.zeros((10, 3), dtype=np.float64)
    coords[0:4] = v
    coords[4] = 0.5 * (v[0] + v[1])
    coords[5] = 0.5 * (v[1] + v[2])
    coords[6] = 0.5 * (v[2] + v[0])
    coords[7] = 0.5 * (v[0] + v[3])
    coords[8] = 0.5 * (v[1] + v[3])
    coords[9] = 0.5 * (v[2] + v[3])
    return coords


@pytest.fixture
def single_element_tet10_mesh(regular_tet10_coords: np.ndarray) -> VolumetricMesh:
    """Creates a procedural single-element quadratic tetrahedron VolumetricMesh."""
    cells = np.array([10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    cell_types = np.array([pv.CellType.QUADRATIC_TETRA], dtype=np.uint8)
    ugrid = pv.UnstructuredGrid(cells, cell_types, regular_tet10_coords)
    return VolumetricMesh(ugrid)

