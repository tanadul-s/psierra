from __future__ import annotations
import numpy as np
import pytest

from psierra.element.shell6 import Shell6
from psierra.element.tet10 import Tet10
from psierra.material.hookean import Hookean
from psierra.material.parameters import IsotropicElasticParameters


@pytest.fixture
def linear_elastic_material() -> Hookean:
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=1.0e7,  # 10 MPa
        poissons_ratio=0.3,
        mass_density=1050.0,   # 1050 kg/m^3
    )
    return Hookean(params)


@pytest.fixture
def flat_shell6_coords() -> np.ndarray:
    # Right-angled triangle in XY plane:
    # 3 corners: (0,0), (1,0), (0,1)
    # 3 mid-edges: (0.5,0), (0.5,0.5), (0,0.5)
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    mids = np.array(
        [
            0.5 * (corners[0] + corners[1]),
            0.5 * (corners[1] + corners[2]),
            0.5 * (corners[2] + corners[0]),
        ],
        dtype=np.float64,
    )
    return np.vstack([corners, mids])


@pytest.fixture
def arbitrary_3d_shell6_coords() -> np.ndarray:
    # Arbitrarily oriented 6-node triangle in 3D
    corners = np.array(
        [
            [0.2, 0.1, 0.5],
            [1.1, 0.4, 0.9],
            [0.3, 1.2, 0.7],
        ],
        dtype=np.float64,
    )
    mids = np.array(
        [
            0.5 * (corners[0] + corners[1]),
            0.5 * (corners[1] + corners[2]),
            0.5 * (corners[2] + corners[0]),
        ],
        dtype=np.float64,
    )
    return np.vstack([corners, mids])


def test_shell6_initialization(linear_elastic_material: Hookean) -> None:
    thickness = 0.003
    elem = Shell6(
        material=linear_elastic_material,
        thickness=thickness,
        drilling_factor=1e-3,
        shear_correction_factor=5.0 / 6.0,
    )
    assert elem.thickness == pytest.approx(0.003)
    assert elem.drilling_factor == pytest.approx(1e-3)
    assert elem.shear_correction_factor == pytest.approx(5.0 / 6.0)
    assert elem.nodes_per_element == 6
    assert elem.dofs_per_node == 6
    assert elem.dofs_per_element == 36

    elem.thickness = 0.001
    assert elem.thickness == pytest.approx(0.001)

    with pytest.raises(ValueError, match="strictly positive"):
        Shell6(material=linear_elastic_material, thickness=-0.002)

    with pytest.raises(ValueError, match="strictly positive"):
        Shell6(material=linear_elastic_material, thickness=0.0)


def test_shell6_shape_functions() -> None:
    # Reference coordinates of the 6 nodes
    node_pts = [
        (0.0, 0.0),      # Node 0
        (1.0, 0.0),      # Node 1
        (0.0, 1.0),      # Node 2
        (0.5, 0.0),      # Node 3
        (0.5, 0.5),      # Node 4
        (0.0, 0.5),      # Node 5
    ]

    # Kronecker delta property: N_i(node_j) == delta_ij
    for j, (xi, eta) in enumerate(node_pts):
        N = Shell6.evaluate_shape_functions(xi, eta)
        expected = np.zeros(6)
        expected[j] = 1.0
        assert np.allclose(N, expected, atol=1e-14)

    # Partition of unity and gradient sum at sample interior points
    test_pts = [(0.1, 0.2), (0.3, 0.4), (0.2, 0.6), (1.0 / 3.0, 1.0 / 3.0)]
    for xi, eta in test_pts:
        N = Shell6.evaluate_shape_functions(xi, eta)
        assert np.isclose(np.sum(N), 1.0, atol=1e-14)

        dxi, deta = Shell6.evaluate_shape_gradients_ref(xi, eta)
        assert np.isclose(np.sum(dxi), 0.0, atol=1e-14)
        assert np.isclose(np.sum(deta), 0.0, atol=1e-14)


def test_shell6_stiffness_symmetry_and_rank(
    linear_elastic_material: Hookean,
    arbitrary_3d_shell6_coords: np.ndarray,
) -> None:
    elem = Shell6(material=linear_elastic_material, thickness=0.004)
    K = elem.compute_linear_stiffness_matrix(arbitrary_3d_shell6_coords)

    assert K.shape == (36, 36)
    # Exact symmetry
    assert np.allclose(K, K.T, atol=1e-10)

    # Eigenvalues: exactly 6 rigid body modes (3 translations + 3 rotations)
    eigenvalues = np.linalg.eigvalsh(K)
    zero_eigs = eigenvalues[:6]
    positive_eigs = eigenvalues[6:]

    assert np.all(np.abs(zero_eigs) < 1e-8)
    assert np.all(positive_eigs > 1e-3)
    assert len(positive_eigs) == 30

    # Strict spectral gap between zero rigid modes and physical deformation modes
    assert np.max(np.abs(zero_eigs)) / np.min(positive_eigs) < 1e-8


def test_shell6_rigid_body_modes(
    linear_elastic_material: Hookean,
    arbitrary_3d_shell6_coords: np.ndarray,
) -> None:
    elem = Shell6(material=linear_elastic_material, thickness=0.002)
    K = elem.compute_linear_stiffness_matrix(arbitrary_3d_shell6_coords)

    # 1. Pure rigid translation along arbitrary directions
    translations = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.5, -0.3, 0.8]),
    ]
    for t in translations:
        u_trans = np.zeros(36, dtype=np.float64)
        for i in range(6):
            u_trans[i * 6 : i * 6 + 3] = t
        energy = float(u_trans.T @ K @ u_trans)
        assert abs(energy) < 1e-8

    # 2. Pure rigid infinitesimal rotation around centroid
    c = np.mean(arbitrary_3d_shell6_coords, axis=0)
    omega = 1e-4 * np.array([0.02, -0.015, 0.03])

    u_rot = np.zeros(36, dtype=np.float64)
    for i in range(6):
        r_i = arbitrary_3d_shell6_coords[i] - c
        delta = np.cross(omega, r_i)
        u_rot[i * 6 : i * 6 + 3] = delta
        u_rot[i * 6 + 3 : i * 6 + 6] = omega

    energy_rot = float(u_rot.T @ K @ u_rot)
    assert abs(energy_rot) < 1e-8


def test_shell6_conforming_coupling_with_tet10(
    linear_elastic_material: Hookean,
) -> None:
    """
    Verifies that Shell6 shares exact C^0 displacement continuity with Tet10 boundary face.
    For any quadratic displacement field, evaluating u(x) on the interface produces
    identical values between the solid Tet10 element and the boundary Shell6 element.
    """
    # Define a Tet10 element
    tet_corners = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
        ],
        dtype=np.float64,
    )
    # 6 mid-edges of Tet10: 4(0-1), 5(1-2), 6(2-0), 7(0-3), 8(1-3), 9(2-3)
    tet_mids = np.array(
        [
            0.5 * (tet_corners[0] + tet_corners[1]),  # 4
            0.5 * (tet_corners[1] + tet_corners[2]),  # 5
            0.5 * (tet_corners[2] + tet_corners[0]),  # 6
            0.5 * (tet_corners[0] + tet_corners[3]),  # 7
            0.5 * (tet_corners[1] + tet_corners[3]),  # 8
            0.5 * (tet_corners[2] + tet_corners[3]),  # 9
        ],
        dtype=np.float64,
    )
    tet_nodes = np.vstack([tet_corners, tet_mids])

    # The bottom face in XY plane (zeta = 0) is formed by nodes:
    # Corners: 0, 1, 2
    # Mid-edges: 4 (0-1), 5 (1-2), 6 (2-0)
    face_node_indices = [0, 1, 2, 4, 5, 6]
    shell_nodes = tet_nodes[face_node_indices]

    # Create Shell6 element
    shell = Shell6(material=linear_elastic_material, thickness=0.001)

    # Prescribe arbitrary quadratic nodal displacements at the 6 shared boundary nodes
    np.random.seed(42)
    nodal_disp_face = np.random.randn(6, 3)

    # Sample random points on the triangular face: (xi, eta) with xi + eta <= 1
    sample_coords = [
        (0.2, 0.3),
        (0.4, 0.1),
        (0.15, 0.7),
        (0.33, 0.33),
    ]

    for xi, eta in sample_coords:
        # 1. Evaluate displacement via Shell6 shape functions:
        N_shell = Shell6.evaluate_shape_functions(xi, eta)
        u_shell = N_shell @ nodal_disp_face

        # 2. Evaluate displacement via Tet10 shape functions on face (zeta = 0):
        # Master tetrahedron coordinates: (xi, eta, zeta=0)
        zeta = 0.0
        N_tet = Tet10.evaluate_shape_functions(xi, eta, zeta)

        # Full 10-node Tet displacement array (zero on non-face nodes)
        nodal_disp_tet = np.zeros((10, 3), dtype=np.float64)
        for local_idx, global_tet_idx in enumerate(face_node_indices):
            nodal_disp_tet[global_tet_idx] = nodal_disp_face[local_idx]

        u_tet = N_tet @ nodal_disp_tet

        # Solid-shell interface continuity must hold to machine precision
        assert np.allclose(u_shell, u_tet, atol=1e-14)


def test_shell6_cantilever_plate_bending() -> None:
    """
    Verifies quadratic Reissner-Mindlin plate bending on a cantilever strip against
    Euler-Bernoulli beam theory: w_tip = P * L^3 / (3 * E * I).
    Tests monotonic mesh convergence:
      - Nx=1 (2 elements): captures >70% of analytical flexibility.
      - Nx=2 (4 elements): captures >93% of analytical flexibility.
      - Nx=4 (8 elements): captures >98% of analytical flexibility.
    """
    params = IsotropicElasticParameters()
    E = 1.0e7
    nu = 0.0
    rho = 1000.0
    params.set_from_youngs_poisson(E, nu, rho)
    mat = Hookean(params)

    L = 1.0
    b = 0.1
    t = 0.01
    elem = Shell6(material=mat, thickness=t)

    def solve_cantilever(Nx: int) -> float:
        nx_pts = 2 * Nx + 1
        ny_pts = 3
        xs = np.linspace(0, L, nx_pts)
        ys = np.linspace(0, b, ny_pts)

        node_map = {}
        node_coords = []
        idx = 0
        for ix in range(nx_pts):
            for iy in range(ny_pts):
                node_map[(ix, iy)] = idx
                node_coords.append([xs[ix], ys[iy], 0.0])
                idx += 1
        node_coords_arr = np.array(node_coords, dtype=np.float64)
        n_nodes = len(node_coords_arr)

        K_global = np.zeros((n_nodes * 6, n_nodes * 6), dtype=np.float64)
        for ex in range(Nx):
            ix0 = 2 * ex
            ix1 = 2 * ex + 1
            ix2 = 2 * ex + 2
            conn1 = [
                node_map[(ix0, 0)], node_map[(ix2, 0)], node_map[(ix2, 2)],
                node_map[(ix1, 0)], node_map[(ix2, 1)], node_map[(ix1, 1)]
            ]
            conn2 = [
                node_map[(ix0, 0)], node_map[(ix2, 2)], node_map[(ix0, 2)],
                node_map[(ix1, 1)], node_map[(ix1, 2)], node_map[(ix0, 1)]
            ]

            K1 = elem.compute_linear_stiffness_matrix(node_coords_arr[conn1])
            K2 = elem.compute_linear_stiffness_matrix(node_coords_arr[conn2])
            for conn, Ke in [(conn1, K1), (conn2, K2)]:
                dofs = []
                for n in conn:
                    dofs.extend(range(n * 6, (n + 1) * 6))
                K_global[np.ix_(dofs, dofs)] += Ke

        # Clamped at x = 0
        fixed = []
        for iy in range(3):
            nid = node_map[(0, iy)]
            fixed.extend(range(nid * 6, (nid + 1) * 6))
        free = [i for i in range(n_nodes * 6) if i not in fixed]

        # Tip downward load P = -1.0 N split across tip nodes at x = L
        P = -1.0
        F = np.zeros(n_nodes * 6, dtype=np.float64)
        F[node_map[(2 * Nx, 0)] * 6 + 2] = P * (1.0 / 6.0)
        F[node_map[(2 * Nx, 1)] * 6 + 2] = P * (4.0 / 6.0)
        F[node_map[(2 * Nx, 2)] * 6 + 2] = P * (1.0 / 6.0)

        K_ff = K_global[np.ix_(free, free)]
        u_f = np.linalg.solve(K_ff, F[free])
        return float(u_f[free.index(node_map[(2 * Nx, 1)] * 6 + 2)])

    I = b * t**3 / 12.0
    w_analytical = -1.0 * L**3 / (3.0 * E * I)

    w_1 = solve_cantilever(1)
    w_2 = solve_cantilever(2)
    w_4 = solve_cantilever(4)

    # Monotonic convergence towards analytical solution
    assert w_1 / w_analytical > 0.70
    assert w_2 / w_analytical > 0.93
    assert w_4 / w_analytical > 0.98
    assert w_4 / w_analytical < 1.02


def test_shell6_mass_matrix(
    linear_elastic_material: Hookean,
    arbitrary_3d_shell6_coords: np.ndarray,
) -> None:
    thickness = 0.002
    elem = Shell6(material=linear_elastic_material, thickness=thickness)
    M = elem.compute_mass_matrix(node_coords=arbitrary_3d_shell6_coords)

    assert M.shape == (36, 36)
    assert np.allclose(M, M.T, atol=1e-12)

    _, _, area = Shell6.compute_local_coordinate_system(arbitrary_3d_shell6_coords)
    expected_total_mass = linear_elastic_material.mass_density * thickness * area

    # Total translational mass in X direction
    u_x = np.zeros(36, dtype=np.float64)
    u_x[0::6] = 1.0
    computed_mass_x = float(u_x.T @ M @ u_x)
    assert computed_mass_x == pytest.approx(expected_total_mass, rel=1e-6)
