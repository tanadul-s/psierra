from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
import pyvista as pv

from psierra.material import IsotropicElasticParameters, Hookean, NeoHookean
from psierra.element import Tet4
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
