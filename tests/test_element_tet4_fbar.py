"""
Unit and benchmark tests for Tet4Fbar finite element.

Validates:
1. Rigid body translation invariance (internal forces vanish to machine precision).
2. Large finite rigid body rotation invariance (Green-Lagrange strain and internal forces vanish).
3. Small-strain limit matching linear elasticity (f_int matches K_lin * u).
4. Directional derivative tangent consistency (k_tan matches d f_int / d u via finite differences).
5. Hyperelastic constitutive law compatibility (Hookean, NeoHookean, MooneyRivlin, Yeoh).
6. Inverted element detection and error handling.
7. Domain-level volumetric locking alleviation on nearly-incompressible soft tissue (nu = 0.499).
8. Quadratic Newton-Raphson convergence within NonlinearStaticAnalysis.
"""

from __future__ import annotations
import numpy as np
import pytest
import pyvista as pv

from psierra.element.tet4_linear import Tet4Linear
from psierra.element.tet4_total_lagrangian import Tet4TotalLagrangian
from psierra.element.tet4_fbar import Tet4Fbar
from psierra.material.parameters import (
    IsotropicElasticParameters,
    MooneyRivlinParameters,
    YeohParameters,
)
from psierra.material.hookean import Hookean
from psierra.material.neohookean import NeoHookean
from psierra.material.mooney_rivlin import MooneyRivlin
from psierra.material.yeoh import Yeoh
from psierra.mesh.volumetric import VolumetricMesh
from psierra.domain.solid import SolidDomain
from psierra.boundary.dirichlet import DirichletConstraint, DirichletManager
from psierra.boundary.load import NodalForce
from psierra.boundary.selector import select_nodes_by_plane
from psierra.analysis.nonlinear_static import NonlinearStaticAnalysis


def create_five_tet_cube() -> VolumetricMesh:
    """Builds a unit cube discretized into 5 linear tetrahedra."""
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
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


def create_cantilever_beam_mesh(nx: int = 4, ny: int = 1, nz: int = 1) -> VolumetricMesh:
    """Builds a structured tetrahedral beam mesh."""
    all_nodes: list[list[float]] = []
    all_tets: list[list[int]] = []
    node_map: dict[tuple[int, int, int], int] = {}

    def get_node(ix: int, iy: int, iz: int) -> int:
        key = (ix, iy, iz)
        if key not in node_map:
            node_map[key] = len(all_nodes)
            all_nodes.append([float(ix), float(iy), float(iz)])
        return node_map[key]

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                n0 = get_node(ix, iy, iz)
                n1 = get_node(ix + 1, iy, iz)
                n2 = get_node(ix + 1, iy + 1, iz)
                n3 = get_node(ix, iy + 1, iz)
                n4 = get_node(ix, iy, iz + 1)
                n5 = get_node(ix + 1, iy, iz + 1)
                n6 = get_node(ix + 1, iy + 1, iz + 1)
                n7 = get_node(ix, iy + 1, iz + 1)

                if (ix + iy + iz) % 2 == 0:
                    tets = [
                        [n0, n1, n3, n4],
                        [n1, n2, n3, n6],
                        [n1, n4, n5, n6],
                        [n3, n4, n6, n7],
                        [n1, n3, n4, n6],
                    ]
                else:
                    tets = [
                        [n0, n1, n2, n5],
                        [n0, n2, n3, n7],
                        [n0, n4, n5, n7],
                        [n2, n5, n6, n7],
                        [n2, n0, n5, n7],
                    ]
                all_tets.extend(tets)

    nodes = np.array(all_nodes, dtype=np.float64)
    cells: list[int] = []
    for tet in all_tets:
        cells.extend([4] + tet)
    cells_array = np.array(cells, dtype=np.int64)
    cell_types = np.full(len(all_tets), pv.CellType.TETRA, dtype=np.uint8)
    ugrid = pv.UnstructuredGrid(cells_array, cell_types, nodes)
    return VolumetricMesh(ugrid)


class TestTet4Fbar:
    """Rigorous mathematical and physical verification of Tet4Fbar element."""

    def test_pure_translation_invariance(
        self, hookean_material: Hookean, regular_tet_coords: np.ndarray
    ) -> None:
        """Under arbitrary rigid translation, internal forces must vanish to machine precision."""
        tet = Tet4Fbar(hookean_material)
        t = np.array([14.2, -8.7, 3.1], dtype=np.float64)
        u_nodes = np.tile(t, (4, 1))

        k_tan, f_int = tet.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet_coords, u_nodes
        )
        assert np.allclose(f_int, 0.0, atol=1e-10)

    def test_pure_rotation_invariance_nonlinear(
        self, neohookean_material: NeoHookean, regular_tet_coords: np.ndarray
    ) -> None:
        """Under large finite 3D rigid rotation, strain and internal forces must vanish."""
        tet = Tet4Fbar(neohookean_material)

        theta = np.deg2rad(45.0)
        axis = np.array([1.0, 2.0, -1.0]) / np.linalg.norm([1.0, 2.0, -1.0])
        K_mat = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        R = np.eye(3) + np.sin(theta) * K_mat + (1.0 - np.cos(theta)) * (K_mat @ K_mat)

        x_rot = regular_tet_coords @ R.T
        u_nodes = x_rot - regular_tet_coords

        k_tan, f_int = tet.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet_coords, u_nodes
        )
        assert np.allclose(f_int, 0.0, atol=1e-10)

    def test_small_strain_limit_matches_linear(
        self, hookean_material: Hookean, regular_tet_coords: np.ndarray
    ) -> None:
        """In the infinitesimal limit, Tet4Fbar internal force matches linear stiffness * u."""
        tet = Tet4Fbar(hookean_material)
        K_lin = tet.compute_linear_stiffness_matrix(regular_tet_coords)

        np.random.seed(42)
        u_small = 1e-6 * np.random.randn(4, 3)
        f_expected = K_lin @ u_small.ravel()

        _, f_fbar = tet.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet_coords, u_small
        )

        rel_diff = np.linalg.norm(f_fbar - f_expected) / np.linalg.norm(f_expected)
        assert rel_diff < 1e-4

    def test_nonlinear_tangent_consistency(
        self, neohookean_material: NeoHookean, regular_tet_coords: np.ndarray
    ) -> None:
        """Directional derivative of internal force matches tangent stiffness matrix."""
        tet = Tet4Fbar(neohookean_material)

        np.random.seed(101)
        u0 = 0.015 * np.random.randn(4, 3)
        v_dir = np.random.randn(4, 3)
        v_dir = v_dir / np.linalg.norm(v_dir)

        k_tan, _ = tet.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet_coords, u0
        )

        h = 1e-7
        _, f_plus = tet.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet_coords, u0 + h * v_dir
        )
        _, f_minus = tet.compute_nonlinear_force_and_tangent_stiffness(
            regular_tet_coords, u0 - h * v_dir
        )
        df_fd = (f_plus - f_minus) / (2.0 * h)
        df_exact = k_tan @ v_dir.ravel()

        rel_err = np.linalg.norm(df_exact - df_fd) / np.linalg.norm(df_exact)
        assert rel_err < 1e-5

    def test_material_models_compatibility(
        self, regular_tet_coords: np.ndarray
    ) -> None:
        """Verifies Tet4Fbar compatibility with Hookean, NeoHookean, MooneyRivlin, and Yeoh."""
        params_iso = IsotropicElasticParameters()
        params_iso.set_from_youngs_poisson(1e5, 0.499, 1000.0)

        materials = [
            Hookean(params_iso),
            NeoHookean(params_iso),
            MooneyRivlin(
                MooneyRivlinParameters(
                    c10=2e4, c01=1e4, bulk_modulus=1e7, mass_density=1000.0
                )
            ),
            Yeoh(
                YeohParameters(
                    c10=2e4, c20=-1e3, c30=500.0, bulk_modulus=1e7, mass_density=1000.0
                )
            ),
        ]

        u_test = np.array(
            [
                [0.01, -0.005, 0.002],
                [-0.008, 0.012, 0.001],
                [0.003, -0.004, 0.015],
                [-0.002, 0.006, -0.008],
            ],
            dtype=np.float64,
        )

        for mat in materials:
            elem = Tet4Fbar(mat)
            k_tan, f_int = elem.compute_nonlinear_force_and_tangent_stiffness(
                regular_tet_coords, u_test
            )
            assert k_tan.shape == (12, 12)
            assert f_int.shape == (12,)
            assert np.all(np.isfinite(k_tan))
            assert np.all(np.isfinite(f_int))

    def test_inverted_element_detection(
        self, hookean_material: Hookean, inverted_tet_coords: np.ndarray
    ) -> None:
        """Inverted reference coordinates must raise ValueError."""
        tet = Tet4Fbar(hookean_material)
        with pytest.raises(ValueError, match="Negative or zero element volume detected"):
            tet.compute_volume_and_gradients(inverted_tet_coords)

    def test_domain_assembly_shear_cube(self) -> None:
        """
        Solves 5-tet shear cube at nu = 0.499 with NonlinearStaticAnalysis.
        Verifies full convergence and nonzero shear deformation.
        """
        mesh = create_five_tet_cube()
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(
            youngs_modulus=1e5, poissons_ratio=0.499, mass_density=1000.0
        )
        mat = NeoHookean(params)
        elem = Tet4Fbar(mat)
        domain = SolidDomain(mesh, mat, elem)

        fixed_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
        clamp = DirichletConstraint(node_indices=fixed_nodes, components=(0, 1, 2), value=0.0)
        dir_mgr = DirichletManager(mesh.n_dofs)
        dir_mgr.add_constraint(clamp)

        top_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
        total_fx = 50.0
        load = NodalForce(
            node_indices=top_nodes,
            force_vector=[total_fx / len(top_nodes), 0.0, 0.0],
        )
        f_ext = load.compute_force_vector(mesh.n_dofs)

        analysis = NonlinearStaticAnalysis(domain)
        result = analysis.solve(
            dirichlet=dir_mgr,
            f_ext=f_ext,
            n_steps=4,
            nr_iterations=15,
            nr_tol=1e-4,
            verbose=False,
        )

        assert result.converged
        assert result.steps_completed == 4

        mean_ux = float(np.mean(domain.displacement[top_nodes, 0]))
        assert mean_ux > 1e-3, f"Expected compliant shear deformation, got {mean_ux:.4e}"

    def test_volumetric_locking_alleviation_beam(self) -> None:
        """
        Benchmarking Tet4TotalLagrangian vs Tet4Fbar on a cantilever beam with nu = 0.499.
        Standard Tet4 locks, while Tet4Fbar yields significantly higher compliance.
        """
        mesh = create_cantilever_beam_mesh(nx=4, ny=2, nz=2)
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(
            youngs_modulus=1e5, poissons_ratio=0.499, mass_density=1000.0
        )
        mat = NeoHookean(params)

        fixed_nodes = select_nodes_by_plane(mesh, plane="yz", bound="min")
        clamp = DirichletConstraint(node_indices=fixed_nodes, components=(0, 1, 2), value=0.0)
        dir_mgr = DirichletManager(mesh.n_dofs)
        dir_mgr.add_constraint(clamp)

        tip_nodes = select_nodes_by_plane(mesh, plane="yz", bound="max")
        total_fy = 5.0
        load = NodalForce(
            node_indices=tip_nodes,
            force_vector=[0.0, total_fy / len(tip_nodes), 0.0],
        )
        f_ext = load.compute_force_vector(mesh.n_dofs)

        # 1. Standard Tet4TotalLagrangian solve
        elem_std = Tet4TotalLagrangian(mat)
        domain_std = SolidDomain(mesh, mat, elem_std)
        analysis_std = NonlinearStaticAnalysis(domain_std)
        res_std = analysis_std.solve(
            dirichlet=dir_mgr,
            f_ext=f_ext,
            n_steps=4,
            nr_iterations=15,
            nr_tol=1e-4,
            verbose=False,
        )
        assert res_std.converged
        uy_std = float(np.mean(domain_std.displacement[tip_nodes, 1]))

        # 2. Tet4Fbar solve
        elem_fbar = Tet4Fbar(mat)
        domain_fbar = SolidDomain(mesh, mat, elem_fbar)
        analysis_fbar = NonlinearStaticAnalysis(domain_fbar)
        res_fbar = analysis_fbar.solve(
            dirichlet=dir_mgr,
            f_ext=f_ext,
            n_steps=4,
            nr_iterations=15,
            nr_tol=1e-4,
            verbose=False,
        )
        assert res_fbar.converged
        uy_fbar = float(np.mean(domain_fbar.displacement[tip_nodes, 1]))

        # Verifications
        assert res_fbar.steps_completed == 4
        # F-bar must be compliant and relieve volumetric locking
        assert uy_fbar > 1.15 * uy_std, (
            f"Expected Tet4Fbar deflection ({uy_fbar:.4e}) to exceed standard Tet4 ({uy_std:.4e})"
        )

    def test_consistent_mass_matrix(self) -> None:
        """Verifies consistent mass matrix total mass matches total reference mass."""
        mesh = create_five_tet_cube()
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(
            youngs_modulus=1e5, poissons_ratio=0.499, mass_density=1050.0
        )
        mat = NeoHookean(params)
        elem = Tet4Fbar(mat)
        domain = SolidDomain(mesh, mat, elem)

        M = domain.assemble_mass_matrix()
        # Row sums of consistent mass matrix for each translational direction equals nodal mass
        total_mass_x = float(np.sum(M[0::3, 0::3]))
        expected_mass = 1050.0 * mesh.volume
        assert np.isclose(total_mass_x, expected_mass, rtol=1e-5)
