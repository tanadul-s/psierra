from __future__ import annotations
import numpy as np
import pytest

from psierra.element.shell3 import Shell3
from psierra.material.hookean import Hookean
from psierra.material.parameters import IsotropicElasticParameters


@pytest.fixture
def linear_elastic_material() -> Hookean:
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=1.0e7,  # 10 MPa (representative of stiff tissue / membrane)
        poissons_ratio=0.3,
        mass_density=1050.0,   # 1050 kg/m^3
    )
    return Hookean(params)


@pytest.fixture
def flat_triangle_coords() -> np.ndarray:
    # Right-angled triangle in the XY plane
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def arbitrary_3d_triangle_coords() -> np.ndarray:
    # Triangle oriented arbitrarily in 3D space
    coords = np.array(
        [
            [0.2, 0.1, 0.5],
            [1.1, 0.4, 0.9],
            [0.3, 1.2, 0.7],
        ],
        dtype=np.float64,
    )
    return coords


def test_shell3_initialization(linear_elastic_material: Hookean) -> None:
    thickness = 0.002
    elem = Shell3(material=linear_elastic_material, thickness=thickness, drilling_factor=1e-3)
    assert elem.thickness == pytest.approx(0.002)
    assert elem.drilling_factor == pytest.approx(1e-3)
    assert elem.nodes_per_element == 3
    assert elem.dofs_per_node == 6
    assert elem.dofs_per_element == 18

    # Thickness property setter validation
    elem.thickness = 0.001
    assert elem.thickness == pytest.approx(0.001)

    with pytest.raises(ValueError, match="strictly positive"):
        Shell3(material=linear_elastic_material, thickness=-0.001)

    with pytest.raises(ValueError, match="strictly positive"):
        Shell3(material=linear_elastic_material, thickness=0.0)


def test_shell3_local_coordinate_system(flat_triangle_coords: np.ndarray) -> None:
    Lambda, X_loc, area = Shell3.compute_local_coordinate_system(flat_triangle_coords)
    assert area == pytest.approx(0.5)
    # Check orthonormality of Lambda
    assert np.allclose(Lambda @ Lambda.T, np.eye(3), atol=1e-12)
    assert np.allclose(Lambda.T @ Lambda, np.eye(3), atol=1e-12)

    # Local coordinates of node 0 should be at origin
    assert np.allclose(X_loc[0], [0.0, 0.0], atol=1e-12)
    # Node 1 along local x-axis
    assert X_loc[1, 1] == pytest.approx(0.0, abs=1e-12)
    assert X_loc[1, 0] > 0.0

    # Degenerate collinear triangle should raise ValueError
    collinear_coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float64
    )
    with pytest.raises(ValueError, match="Collinear or degenerate"):
        Shell3.compute_local_coordinate_system(collinear_coords)


def test_shell3_stiffness_symmetry_and_rank(
    linear_elastic_material: Hookean,
    arbitrary_3d_triangle_coords: np.ndarray,
) -> None:
    elem = Shell3(material=linear_elastic_material, thickness=0.005)
    K = elem.compute_linear_stiffness_matrix(arbitrary_3d_triangle_coords)

    assert K.shape == (18, 18)
    # Exact symmetry
    assert np.allclose(K, K.T, atol=1e-10)

    # Eigenvalues: exactly 6 rigid body modes (3 translations, 3 rotations)
    eigenvalues = np.linalg.eigvalsh(K)
    zero_eigs = eigenvalues[:6]
    positive_eigs = eigenvalues[6:]

    # Zero eigenvalues should be practically zero relative to the positive ones
    assert np.all(np.abs(zero_eigs) < 1e-8)
    assert np.all(positive_eigs > 1e-3)
    assert len(positive_eigs) == 12
    # Ensure clear separation between zero rigid modes and elastic deformation modes
    assert np.max(np.abs(zero_eigs)) / np.min(positive_eigs) < 1e-8


def test_shell3_rigid_body_modes(
    linear_elastic_material: Hookean,
    arbitrary_3d_triangle_coords: np.ndarray,
) -> None:
    """
    Verifies that applying pure rigid body translation or pure rigid body rotation
    yields zero strain energy: u^T * K * u ~ 0.
    """
    elem = Shell3(material=linear_elastic_material, thickness=0.002)
    K = elem.compute_linear_stiffness_matrix(arbitrary_3d_triangle_coords)

    # 1. Pure rigid translation: [tx, ty, tz, 0, 0, 0] repeated for 3 nodes
    translations = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.5, -0.3, 0.8]),
    ]
    for t in translations:
        u_trans = np.zeros(18, dtype=np.float64)
        for i in range(3):
            u_trans[i * 6 : i * 6 + 3] = t
        energy = float(u_trans.T @ K @ u_trans)
        assert abs(energy) < 1e-7

    # 2. Pure rigid infinitesimal rotation around center of triangle
    c = np.mean(arbitrary_3d_triangle_coords, axis=0)
    omega = 1e-4 * np.array([0.02, -0.015, 0.03])  # infinitesimal rotation vector (radians)

    u_rot = np.zeros(18, dtype=np.float64)
    for i in range(3):
        r_i = arbitrary_3d_triangle_coords[i] - c
        # Translation induced by rotation: delta = omega x r
        delta = np.cross(omega, r_i)
        u_rot[i * 6 : i * 6 + 3] = delta
        # Rotation vector at node
        u_rot[i * 6 + 3 : i * 6 + 6] = omega

    energy_rot = float(u_rot.T @ K @ u_rot)
    assert abs(energy_rot) < 1e-8


def test_shell3_membrane_tension(linear_elastic_material: Hookean) -> None:
    """
    Uniform in-plane tension test on a right-triangle shell element in XY plane.
    """
    elem = Shell3(material=linear_elastic_material, thickness=0.001)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    K = elem.compute_linear_stiffness_matrix(coords)

    # Stretch in X direction: u_x = eps0 * x, v = -nu * eps0 * y
    eps0 = 1e-4
    nu = linear_elastic_material.parameters.poissons_ratio
    u_disp = np.zeros(18, dtype=np.float64)
    for i in range(3):
        x, y = coords[i, 0], coords[i, 1]
        u_disp[i * 6] = eps0 * x
        u_disp[i * 6 + 1] = -nu * eps0 * y

    forces = K @ u_disp
    # In-plane forces should be non-zero on stretching DOFs and zero on out-of-plane DOFs
    for i in range(3):
        # Out-of-plane z force and bending rotations th_x, th_y should be zero
        assert abs(forces[i * 6 + 2]) < 1e-8
        assert abs(forces[i * 6 + 3]) < 1e-8
        assert abs(forces[i * 6 + 4]) < 1e-8


def test_shell3_mass_matrix(
    linear_elastic_material: Hookean,
    arbitrary_3d_triangle_coords: np.ndarray,
) -> None:
    thickness = 0.002
    elem = Shell3(material=linear_elastic_material, thickness=thickness)
    M = elem.compute_mass_matrix(node_coords=arbitrary_3d_triangle_coords)

    assert M.shape == (18, 18)
    assert np.allclose(M, M.T, atol=1e-12)

    # Check total translational mass
    _, _, area = Shell3.compute_local_coordinate_system(arbitrary_3d_triangle_coords)
    expected_total_mass = linear_elastic_material.mass_density * thickness * area

    # The sum of translational masses along X, Y, Z for rigid body translation
    u_x = np.zeros(18, dtype=np.float64)
    u_x[0::6] = 1.0  # u_x = 1 at all 3 nodes
    computed_mass_x = float(u_x.T @ M @ u_x)
    assert computed_mass_x == pytest.approx(expected_total_mass, rel=1e-6)


def test_shell3_cantilever_plate_bending() -> None:
    """
    Verifies DKT bending accuracy on a cantilever plate strip against Euler-Bernoulli beam theory:
    w_tip = P * L^3 / (3 * E * I).
    Even with an ultra-coarse mesh of just 2 elements, DKT captures >90% of analytical flexibility.
    """
    params = IsotropicElasticParameters()
    E = 1.0e7
    nu = 0.0  # Poisson ratio = 0 to isolate pure 1D beam bending
    rho = 1000.0
    params.set_from_youngs_poisson(E, nu, rho)
    mat = Hookean(params)

    L = 1.0
    b = 0.1
    t = 0.01
    elem = Shell3(material=mat, thickness=t)

    # 4 nodes defining a rectangle split into 2 triangles
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],  # 0: root bottom
            [L, 0.0, 0.0],    # 1: tip bottom
            [L, b, 0.0],      # 2: tip top
            [0.0, b, 0.0],    # 3: root top
        ],
        dtype=np.float64,
    )

    K1 = elem.compute_linear_stiffness_matrix(nodes[[0, 1, 2]])
    K2 = elem.compute_linear_stiffness_matrix(nodes[[0, 2, 3]])

    K_global = np.zeros((24, 24), dtype=np.float64)
    for conn, Ke in [([0, 1, 2], K1), ([0, 2, 3], K2)]:
        dofs = []
        for n in conn:
            dofs.extend(range(n * 6, (n + 1) * 6))
        K_global[np.ix_(dofs, dofs)] += Ke

    # Root nodes 0 and 3 clamped
    fixed = list(range(0, 6)) + list(range(18, 24))
    free = [i for i in range(24) if i not in fixed]

    # Tip downward point load P = -1.0 N split across nodes 1 and 2
    P = -1.0
    F = np.zeros(24, dtype=np.float64)
    F[1 * 6 + 2] = P / 2.0
    F[2 * 6 + 2] = P / 2.0

    K_ff = K_global[np.ix_(free, free)]
    F_f = F[free]
    u_f = np.linalg.solve(K_ff, F_f)

    tip_w = 0.5 * (u_f[free.index(1 * 6 + 2)] + u_f[free.index(2 * 6 + 2)])

    # Beam theory: w = P L^3 / (3 E I), I = b t^3 / 12
    I = b * t**3 / 12.0
    w_analytical = P * L**3 / (3.0 * E * I)

    # With only 2 elements, deflection should be within 10% of analytical solution (>90%)
    assert tip_w / w_analytical > 0.90
    assert tip_w / w_analytical < 1.05

