from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
import pyvista as pv

from psierra.domain.solid import SolidDomain
from psierra.domain.shell import ShellDomain
from psierra.domain.coupled import CoupledSolidShellDomain
from psierra.element.tet4 import Tet4
from psierra.element.shell3 import Shell3
from psierra.material.hookean import Hookean
from psierra.material.parameters import IsotropicElasticParameters
from psierra.mesh.volumetric import VolumetricMesh
from psierra.mesh.surface import SurfaceMesh
from psierra.boundary.dirichlet import DirichletConstraint, DirichletManager
from psierra.analysis.linear_static import LinearStaticAnalysis, LinearStaticResult


@pytest.fixture
def plate_surface_mesh() -> SurfaceMesh:
    # 2-triangle rectangular plate in XY plane: corners (0,0), (1,0), (1,0.5), (0,0.5)
    points = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 0.5, 0.0],  # 2
            [0.0, 0.5, 0.0],  # 3
        ],
        dtype=np.float64,
    )
    faces = np.array([3, 0, 1, 2, 3, 0, 2, 3])
    poly = pv.PolyData(points, faces)
    return SurfaceMesh(poly)


def test_shell_domain(plate_surface_mesh: SurfaceMesh) -> None:
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(youngs_modulus=1e7, poissons_ratio=0.3, mass_density=1000.0)
    mat = Hookean(params)
    elem = Shell3(mat, thickness=0.01)

    domain = ShellDomain(mesh=plate_surface_mesh, material=mat, element=elem)
    assert domain.n_dofs == 4 * 6  # 4 nodes * 6 DOFs = 24 DOFs
    assert domain.displacement.shape == (4, 3)
    assert domain.rotation.shape == (4, 3)

    K = domain.assemble_stiffness_matrix()
    assert K.shape == (24, 24)
    assert np.allclose(K.toarray(), K.toarray().T, atol=1e-10)

    M = domain.assemble_mass_matrix()
    assert M.shape == (24, 24)
    assert np.allclose(M.toarray(), M.toarray().T, atol=1e-10)

    # Solve with LinearStaticAnalysis
    # Clamp root edge: nodes 0 and 3 (all 6 DOFs)
    dm = DirichletManager(domain.n_dofs)
    dm.add_constraint(
        DirichletConstraint([0, 3], components=(0, 1, 2, 3, 4, 5), value=0.0, dofs_per_node=6)
    )

    # Apply downward tip load at nodes 1 and 2
    f_ext = np.zeros(domain.n_dofs, dtype=np.float64)
    f_ext[1 * 6 + 2] = -10.0  # Fz on node 1
    f_ext[2 * 6 + 2] = -10.0  # Fz on node 2

    analysis = LinearStaticAnalysis(domain)
    res = analysis.solve(dm, f_ext, verbose=False)

    assert isinstance(res, LinearStaticResult)
    assert res.strain_energy > 0.0
    # Tip nodes should deflect downwards
    assert res.displacement[1 * 6 + 2] < 0.0
    assert res.displacement[2 * 6 + 2] < 0.0


def test_coupled_solid_shell_domain(beam_vtk_path: Path) -> None:
    if not beam_vtk_path.exists():
        pytest.skip(f"Beam mesh file not found: {beam_vtk_path}")

    vmesh = VolumetricMesh(beam_vtk_path)
    smesh = vmesh.surface

    params_solid = IsotropicElasticParameters()
    params_solid.set_from_youngs_poisson(youngs_modulus=50000.0, poissons_ratio=0.4, mass_density=1e-9)
    mat_solid = Hookean(params_solid)
    elem_solid = Tet4(mat_solid)
    solid_domain = SolidDomain(vmesh, mat_solid, elem_solid)

    params_shell = IsotropicElasticParameters()
    params_shell.set_from_youngs_poisson(youngs_modulus=10.0e6, poissons_ratio=0.3, mass_density=1.1e-9)
    mat_shell = Hookean(params_shell)
    elem_shell = Shell3(mat_shell, thickness=0.5)
    shell_domain = ShellDomain(smesh, mat_shell, elem_shell)

    coupled_domain = CoupledSolidShellDomain(solid_domain, shell_domain)

    expected_dofs = 3 * vmesh.n_nodes + 3 * smesh.n_nodes
    assert coupled_domain.n_dofs == expected_dofs
    assert coupled_domain.displacement.shape == (vmesh.n_nodes, 3)

    K = coupled_domain.assemble_stiffness_matrix()
    assert K.shape == (expected_dofs, expected_dofs)
    # Check symmetry
    diff = K - K.T
    diff.eliminate_zeros()
    assert (diff.nnz == 0) or (np.max(np.abs(diff.data)) < 1e-8)

    M = coupled_domain.assemble_mass_matrix()
    assert M.shape == (expected_dofs, expected_dofs)

    # Solve with LinearStaticAnalysis
    from psierra.boundary.selector import select_nodes_by_plane
    root_nodes = select_nodes_by_plane(vmesh, plane="xy", bound="min")
    dm = DirichletManager(coupled_domain.n_dofs)
    # Clamp solid translations at root face
    dm.add_constraint(DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0, dofs_per_node=3))
    # Clamp shell rotations at root face
    surf_root_mask = np.isin(smesh.node_indices, root_nodes)
    surf_root_idx = np.where(surf_root_mask)[0]
    rot_dofs = [3 * vmesh.n_nodes + 3 * s + c for s in surf_root_idx for c in (0, 1, 2)]
    dm.prescribe_dofs(rot_dofs, value=0.0)

    f_ext = np.zeros(coupled_domain.n_dofs, dtype=np.float64)
    tip_nodes = select_nodes_by_plane(vmesh, plane="xy", bound="max")
    for tn in tip_nodes:
        f_ext[tn * 3 + 1] = -10.0  # Downward load along Y

    analysis = LinearStaticAnalysis(coupled_domain)
    res = analysis.solve(dm, f_ext, verbose=False)
    assert isinstance(res, LinearStaticResult)
    assert res.strain_energy > 0.0
    # Tip deflection along Y should be negative
    assert np.mean(res.displacement[tip_nodes * 3 + 1]) < 0.0


def test_domain_to_pyvista_and_save_state(tmp_path: Path, beam_vtk_path: Path) -> None:
    if not beam_vtk_path.exists():
        pytest.skip(f"Beam mesh file not found: {beam_vtk_path}")

    vmesh = VolumetricMesh(beam_vtk_path)
    smesh = vmesh.surface

    params_solid = IsotropicElasticParameters()
    params_solid.set_from_youngs_poisson(youngs_modulus=50000.0, poissons_ratio=0.4, mass_density=1e-9)
    mat_solid = Hookean(params_solid)
    elem_solid = Tet4(mat_solid)
    solid_domain = SolidDomain(vmesh, mat_solid, elem_solid)

    # Set mock displacement on solid
    mock_u = np.zeros(solid_domain.n_dofs, dtype=np.float64)
    mock_u[1::3] = -5.0  # -5 mm along Y
    solid_domain.update_state(u=mock_u)

    # Test SolidDomain to_pyvista
    grid_warped = solid_domain.to_pyvista(factor=1.0)
    assert np.allclose(grid_warped.points[:, 1], vmesh.nodes[:, 1] - 5.0)
    assert "Displacement" in grid_warped.point_data
    assert np.allclose(grid_warped.point_data["Displacement"][:, 1], -5.0)

    grid_unwarped = solid_domain.to_pyvista(factor=0.0)
    assert np.allclose(grid_unwarped.points, vmesh.nodes)

    # Test SolidDomain save_state (defaults to factor=0.0: reference initial geometry with displacement field)
    solid_file = tmp_path / "solid_deformed.vtu"
    solid_domain.save_state(solid_file)
    assert solid_file.exists()
    read_grid = pv.read(str(solid_file))
    assert np.allclose(read_grid.points, vmesh.nodes)
    assert np.allclose(read_grid.point_data["Displacement"][:, 1], -5.0)
    warped_from_vtu = read_grid.warp_by_vector("Displacement")
    assert np.allclose(warped_from_vtu.points[:, 1], vmesh.nodes[:, 1] - 5.0)

    # Test explicit factor=1.0 saving
    solid_file_warped = tmp_path / "solid_prewarped.vtu"
    solid_domain.save_state(solid_file_warped, factor=1.0)
    assert np.allclose(pv.read(str(solid_file_warped)).points[:, 1], vmesh.nodes[:, 1] - 5.0)

    # Test ShellDomain to_pyvista & save_state
    params_shell = IsotropicElasticParameters()
    params_shell.set_from_youngs_poisson(youngs_modulus=10.0e6, poissons_ratio=0.3, mass_density=1.1e-9)
    mat_shell = Hookean(params_shell)
    elem_shell = Shell3(mat_shell, thickness=0.5)
    shell_domain = ShellDomain(smesh, mat_shell, elem_shell)

    mock_u_shell = np.zeros(shell_domain.n_dofs, dtype=np.float64)
    mock_u_shell[1::6] = -3.0  # -3 mm along Y
    shell_domain.update_state(u=mock_u_shell)

    shell_warped = shell_domain.to_pyvista(factor=1.0)
    assert isinstance(shell_warped, pv.UnstructuredGrid)
    assert np.allclose(shell_warped.points[:, 1], smesh.nodes[:, 1] - 3.0)
    assert "Displacement" in shell_warped.point_data
    assert "Rotation" in shell_warped.point_data
    assert np.allclose(shell_warped.point_data["Displacement"][:, 1], -3.0)

    # ShellDomain must save as .vtu (no .vtp), defaults to factor=0.0
    shell_file = tmp_path / "shell_deformed.vtu"
    shell_domain.save_state(shell_file)
    assert shell_file.exists()
    read_shell = pv.read(str(shell_file))
    assert np.allclose(read_shell.points, smesh.nodes)
    assert np.allclose(read_shell.point_data["Displacement"][:, 1], -3.0)
    shell_warped_from_vtu = read_shell.warp_by_vector("Displacement")
    assert np.allclose(shell_warped_from_vtu.points[:, 1], smesh.nodes[:, 1] - 3.0)

    # Test CoupledSolidShellDomain unified save_state (domain labeling for ParaView threshold)
    coupled_domain = CoupledSolidShellDomain(solid_domain, shell_domain)
    coupled_unified_file = tmp_path / "coupled_unified.vtu"
    coupled_domain.save_state(coupled_unified_file)
    assert coupled_unified_file.exists()

    read_unified = pv.read(str(coupled_unified_file))
    assert read_unified.n_cells == vmesh.n_elements + smesh.n_elements
    assert "Domain ID" in read_unified.cell_data
    assert "Domain Name" in read_unified.cell_data
    assert "Displacement" in read_unified.point_data
    assert "Rotation" in read_unified.point_data

    # Test thresholding in ParaView workflow
    solid_thresh = read_unified.threshold((0, 0), scalars="Domain ID")
    shell_thresh = read_unified.threshold((1, 1), scalars="Domain ID")
    assert solid_thresh.n_cells == vmesh.n_elements
    assert shell_thresh.n_cells == smesh.n_elements

    # Test Coupled internal force and tangent stiffness assemblies
    f_int_coupled = coupled_domain.assemble_internal_force_vector()
    K_tan_coupled = coupled_domain.assemble_tangent_stiffness_matrix()
    assert f_int_coupled.shape == (coupled_domain.n_dofs,)
    assert K_tan_coupled.shape == (coupled_domain.n_dofs, coupled_domain.n_dofs)

    # Test legacy CoupledSolidShellDomain save_solid_state, save_shell_state, and save_pvd
    coupled_solid_file = tmp_path / "coupled_solid.vtu"
    coupled_shell_file = tmp_path / "coupled_shell.vtu"
    coupled_pvd_file = tmp_path / "coupled.pvd"

    coupled_domain.save_solid_state(coupled_solid_file)
    coupled_domain.save_shell_state(coupled_shell_file)
    coupled_domain.save_pvd(coupled_pvd_file)

    assert coupled_solid_file.exists()
    assert coupled_shell_file.exists()
    assert coupled_pvd_file.exists()
    assert "coupled_solid.vtu" in coupled_pvd_file.read_text()
    assert "coupled_shell.vtu" in coupled_pvd_file.read_text()
