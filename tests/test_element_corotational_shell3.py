from __future__ import annotations
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Shell3, CorotationalShell3


@pytest.fixture
def test_triangle_coords() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [30.0, 80.0, 0.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def hookean_material() -> Hookean:
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(youngs_modulus=210000.0, poissons_ratio=0.3, mass_density=7.85e-9)
    return Hookean(params)


def test_corotational_shell3_initialization(hookean_material: Hookean) -> None:
    elem = CorotationalShell3(hookean_material, thickness=1.5, drilling_factor=1e-3)
    assert elem.thickness == 1.5
    assert elem.drilling_factor == 1e-3
    assert elem.nodes_per_element == 3
    assert elem.dofs_per_node == 6
    assert elem.dofs_per_element == 18


def test_corotational_shell3_rigid_translation_invariance(
    hookean_material: Hookean, test_triangle_coords: np.ndarray
) -> None:
    elem = CorotationalShell3(hookean_material, thickness=1.0)

    # Pure 3D rigid translation
    u_rigid = np.zeros((3, 6), dtype=np.float64)
    u_rigid[:, :3] = [25.0, -15.0, 42.0]

    K_tan, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(test_triangle_coords, u_rigid)

    assert np.allclose(f_int, 0.0, atol=1e-8)
    assert np.allclose(K_tan, K_tan.T, atol=1e-8)


def test_corotational_shell3_rigid_rotation_invariance(
    hookean_material: Hookean, test_triangle_coords: np.ndarray
) -> None:
    elem = CorotationalShell3(hookean_material, thickness=1.0)

    # Large finite 3D rigid rotation (35 degrees around axis [1, 2, -1])
    rot_axis = np.array([1.0, 2.0, -1.0])
    rot_axis /= np.linalg.norm(rot_axis)
    rot_angle = np.radians(35.0)
    rot_vec = rot_axis * rot_angle
    R_mat = Rotation.from_rotvec(rot_vec).as_matrix()

    x_rigid = (R_mat @ test_triangle_coords.T).T
    u_trans = x_rigid - test_triangle_coords

    u_rigid = np.zeros((3, 6), dtype=np.float64)
    u_rigid[:, :3] = u_trans
    for i in range(3):
        u_rigid[i, 3:] = rot_vec

    K_tan, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(test_triangle_coords, u_rigid)

    # 1. Spurious internal forces must be identically zero
    assert np.allclose(f_int, 0.0, atol=1e-7)

    # 2. Tangent stiffness matrix must remain symmetric
    assert np.allclose(K_tan, K_tan.T, atol=1e-8)

    # 3. Exactly 6 zero eigenvalues corresponding to rigid body modes
    eigs = np.linalg.eigvalsh(K_tan)
    zero_eigs = eigs[:6]
    positive_eigs = eigs[6:]
    assert np.all(np.abs(zero_eigs) < 1e-4)
    assert np.all(positive_eigs > 0.0)


def test_corotational_shell3_small_strain_consistency(
    hookean_material: Hookean, test_triangle_coords: np.ndarray
) -> None:
    elem_corot = CorotationalShell3(hookean_material, thickness=1.0)
    elem_linear = Shell3(hookean_material, thickness=1.0)

    K_0 = elem_linear.compute_linear_stiffness_matrix(test_triangle_coords)

    # At zero displacement, K_tan must equal linear stiffness K_0
    u_zero = np.zeros((3, 6), dtype=np.float64)
    K_tan_0, f_int_0 = elem_corot.compute_nonlinear_force_and_tangent_stiffness(
        test_triangle_coords, u_zero
    )

    assert np.allclose(f_int_0, 0.0, atol=1e-14)
    assert np.allclose(K_tan_0, K_0, atol=1e-8)

    # For pure rotational perturbation, f_int matches K_0 @ u to machine precision
    u_rot = np.zeros((3, 6), dtype=np.float64)
    u_rot[0, 3] = 1e-6
    u_rot[1, 4] = -1e-6
    _, f_int_rot = elem_corot.compute_nonlinear_force_and_tangent_stiffness(
        test_triangle_coords, u_rot
    )
    assert np.allclose(f_int_rot, K_0 @ u_rot.ravel(), atol=1e-10)

    # For pure membrane stretching, f_int matches linear CST membrane forces to machine precision
    u_stretch = np.zeros((3, 6), dtype=np.float64)
    u_stretch[1, 0] = 1e-6  # stretch along edge 1-2
    _, f_int_stretch = elem_corot.compute_nonlinear_force_and_tangent_stiffness(
        test_triangle_coords, u_stretch
    )
    assert np.allclose(f_int_stretch, K_0 @ u_stretch.ravel(), atol=1e-9)


def test_corotational_shell3_extract_element_rotation(
    hookean_material: Hookean, test_triangle_coords: np.ndarray
) -> None:
    elem = CorotationalShell3(hookean_material, thickness=1.0)

    rot_vec = np.array([0.2, -0.4, 0.3])
    R_expected = Rotation.from_rotvec(rot_vec).as_matrix()

    x_rigid = (R_expected @ test_triangle_coords.T).T
    u_nodes = np.zeros((3, 6), dtype=np.float64)
    u_nodes[:, :3] = x_rigid - test_triangle_coords
    for i in range(3):
        u_nodes[i, 3:] = rot_vec

    R_extracted, Lambda_0, Lambda_def = elem.extract_element_rotation(
        test_triangle_coords, u_nodes
    )

    assert np.allclose(R_extracted, R_expected, atol=1e-12)
    assert np.isclose(np.linalg.det(R_extracted), 1.0)


def test_corotational_shell3_nonlinear_analysis_rotated_strip(hookean_material: Hookean) -> None:
    import pyvista as pv
    from psierra.mesh import SurfaceMesh
    from psierra.domain import ShellDomain
    from psierra.boundary import DirichletConstraint, DirichletManager
    from psierra.analysis import NonlinearStaticAnalysis

    L, W = 100.0, 10.0
    Nx, Ny = 10, 2
    xs = np.linspace(0.0, L, Nx + 1)
    ys = np.linspace(-W / 2.0, W / 2.0, Ny + 1)

    points = []
    node_map = {}
    idx = 0
    for iy in range(Ny + 1):
        for ix in range(Nx + 1):
            points.append([xs[ix], ys[iy], 0.0])
            node_map[(ix, iy)] = idx
            idx += 1

    faces = []
    for iy in range(Ny):
        for ix in range(Nx):
            n0 = node_map[(ix, iy)]
            n1 = node_map[(ix + 1, iy)]
            n2 = node_map[(ix + 1, iy + 1)]
            n3 = node_map[(ix, iy + 1)]
            faces.extend([3, n0, n1, n2, 3, n0, n2, n3])

    # Rotate mesh by 30 deg in X and 45 deg in Y in 3D space
    rot = Rotation.from_euler("xy", [30.0, 45.0], degrees=True)
    R_space = rot.as_matrix()
    pts_rot = (R_space @ np.array(points, dtype=np.float64).T).T

    poly = pv.PolyData(pts_rot, np.array(faces, dtype=np.int64))
    smesh = SurfaceMesh(poly)

    elem = CorotationalShell3(hookean_material, thickness=1.0)
    domain = ShellDomain(smesh, hookean_material, elem)

    root_nodes = np.array([node_map[(0, iy)] for iy in range(Ny + 1)])
    tip_nodes = np.array([node_map[(Nx, iy)] for iy in range(Ny + 1)])

    dm = DirichletManager(domain.n_dofs)
    dm.add_constraint(DirichletConstraint(root_nodes, components=(0, 1, 2, 3, 4, 5), value=0.0))

    # Pull along rotated axis
    pull_dir = R_space @ np.array([1.0, 0.0, 0.0])
    f_ext = np.zeros(domain.n_dofs, dtype=np.float64)
    F_total = 100.0
    for tn in tip_nodes:
        f_ext[tn * 6 : tn * 6 + 3] = (F_total / len(tip_nodes)) * pull_dir

    solver = NonlinearStaticAnalysis(domain)
    res = solver.solve(dm, f_ext, n_steps=3, max_iterations=15, tolerance=1e-4, verbose=False)

    assert res.converged
    assert res.steps_completed == 3
