import numpy as np
import pytest
from pathlib import Path

from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear
from psierra.boundary import (
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
    HydrostaticPressureLoad,
    BuoyancyLoad,
    select_nodes_by_plane,
)
from psierra.analysis import LinearStaticAnalysis


@pytest.fixture
def beam_mesh() -> VolumetricMesh:
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        mesh_path = Path("psierra/data/beam1/mesh/beam1.vtk")
    return VolumetricMesh(mesh_path)


class TestHydrostaticAndBuoyancyLoad:
    def test_hydrostatic_pressure_cube_archimedes(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """Verifies that surface hydrostatic pressure integration recovers Archimedes buoyancy to machine precision."""
        mesh = five_tet_cube_mesh
        volume = mesh.volume  # 1.0 mm^3 for unit cube
        rho_fluid = 1000e-12  # 1000 kg/m^3 in tonne/mm^3
        g_accel = 9810.0  # mm/s^2

        hydro = HydrostaticPressureLoad(
            fluid_density=rho_fluid,
            acceleration=(0.0, 0.0, -g_accel),
            fluid_level=2.0,  # fully submerged (cube z in [0, 1])
        )

        f_hydro = hydro.compute_force_vector(mesh)
        assert f_hydro.shape == (mesh.n_dofs,)

        res_3d, buoyant_mag = hydro.compute_buoyant_resultant(mesh)
        expected_buoyancy = rho_fluid * volume * g_accel

        # Net lateral forces must be zero by symmetry
        assert abs(res_3d[0]) < 1e-14
        assert abs(res_3d[1]) < 1e-14

        # Net vertical force must match Archimedes' buoyancy
        np.testing.assert_allclose(res_3d[2], expected_buoyancy, rtol=1e-12)
        np.testing.assert_allclose(buoyant_mag, expected_buoyancy, rtol=1e-12)

    def test_buoyancy_load_cube_archimedes(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """Verifies volumetric BuoyancyLoad matches Archimedes buoyancy."""
        mesh = five_tet_cube_mesh
        volume = mesh.volume
        rho_fluid = 1000e-12
        g_accel = 9810.0

        buoyancy = BuoyancyLoad(
            fluid_density=rho_fluid,
            acceleration=(0.0, 0.0, -g_accel),
        )

        f_buoyancy = buoyancy.compute_force_vector(mesh)
        assert f_buoyancy.shape == (mesh.n_dofs,)

        res_3d, buoyant_mag = buoyancy.compute_buoyant_resultant(mesh)
        expected_buoyancy = rho_fluid * volume * g_accel

        assert abs(res_3d[0]) < 1e-14
        assert abs(res_3d[1]) < 1e-14
        np.testing.assert_allclose(res_3d[2], expected_buoyancy, rtol=1e-12)
        np.testing.assert_allclose(buoyant_mag, expected_buoyancy, rtol=1e-12)

    def test_hydrostatic_and_buoyancy_equivalence_on_beam(
        self, beam_mesh: VolumetricMesh
    ) -> None:
        """Verifies that surface HydrostaticPressureLoad and volumetric BuoyancyLoad produce identical net upward force."""
        mesh = beam_mesh
        rho_fluid = 1000e-12  # Water / CSF density
        g_accel = 9810.0

        # High fluid level (fully submerged)
        h_fluid = float(mesh.bounds[5] + 50.0)

        hydro = HydrostaticPressureLoad(
            fluid_density=rho_fluid,
            acceleration=(0.0, 0.0, -g_accel),
            fluid_level=h_fluid,
        )
        buoyancy = BuoyancyLoad(
            fluid_density=rho_fluid,
            acceleration=(0.0, 0.0, -g_accel),
        )

        res_hydro, mag_hydro = hydro.compute_buoyant_resultant(mesh)
        res_buoy, mag_buoy = buoyancy.compute_buoyant_resultant(mesh)
        expected_buoyancy = rho_fluid * mesh.volume * g_accel

        # Both match theoretical Archimedes buoyancy
        np.testing.assert_allclose(mag_hydro, expected_buoyancy, rtol=1e-10)
        np.testing.assert_allclose(mag_buoy, expected_buoyancy, rtol=1e-12)
        np.testing.assert_allclose(mag_hydro, mag_buoy, rtol=1e-10)

        # Net horizontal forces on closed mesh cancel
        assert abs(res_hydro[0]) < 1e-10
        assert abs(res_hydro[1]) < 1e-10

    def test_pressure_field_depth_gradient(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """Verifies facet pressures increase with depth and are zero above fluid level."""
        mesh = five_tet_cube_mesh
        rho_fluid = 1000e-12
        g_accel = 9810.0
        h_fluid = 0.5  # Halfway up the cube [0, 1]

        hydro = HydrostaticPressureLoad(
            fluid_density=rho_fluid,
            acceleration=(0.0, 0.0, -g_accel),
            fluid_level=h_fluid,
        )

        pressures = hydro.evaluate_facet_pressures(mesh)
        assert len(pressures) == mesh.surface.n_elements

        smesh = mesh.surface
        centroids = smesh.centroids
        for p, c in zip(pressures, centroids):
            z = c[2]
            if z >= h_fluid:
                assert p == 0.0
            else:
                expected_p = rho_fluid * g_accel * (h_fluid - z)
                np.testing.assert_allclose(p, expected_p, rtol=1e-6)

    def test_arbitrary_gravity_orientations(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """Verifies buoyancy direction dynamically tracks arbitrary 3D gravity vectors."""
        mesh = five_tet_cube_mesh
        rho_fluid = 1000e-12

        # 1. Supine posture: g along -Y
        hydro_supine = HydrostaticPressureLoad(
            fluid_density=rho_fluid,
            acceleration=(0.0, -9810.0, 0.0),
            fluid_level=2.0,
        )
        res_y, _ = hydro_supine.compute_buoyant_resultant(mesh)
        assert res_y[1] > 0.0  # Upward along +Y
        assert abs(res_y[0]) < 1e-14
        assert abs(res_y[2]) < 1e-14

        # 2. Tilted posture: g along (-3000, 0, -4000)
        g_tilted = np.array([-3000.0, 0.0, -4000.0])
        buoyancy_tilted = BuoyancyLoad(
            fluid_density=rho_fluid,
            acceleration=g_tilted,
        )
        res_tilted, mag = buoyancy_tilted.compute_buoyant_resultant(mesh)
        # Unit upward direction is -g / ||g||
        u_expected = -g_tilted / np.linalg.norm(g_tilted)
        unit_res = res_tilted / np.linalg.norm(res_tilted)
        np.testing.assert_allclose(unit_res, u_expected, atol=1e-12)

    def test_cantilever_beam_buoyancy_counter_deflection(
        self, beam_mesh: VolumetricMesh
    ) -> None:
        """Verifies that buoyant upward forces physically reduce cantilever deflection caused by gravity."""
        mesh = beam_mesh
        params = IsotropicElasticParameters()
        # Solid density 2000 kg/m^3 (sinks in water 1000 kg/m^3)
        rho_solid = 2000e-12
        rho_fluid = 1000e-12
        params.set_from_youngs_poisson(
            youngs_modulus=10.0,  # 10 MPa (flexible for visible deflection)
            poissons_ratio=0.3,
            mass_density=rho_solid,
        )
        material = Hookean(params)
        element = Tet4Linear(material)

        # Clamped at root face (X minimum)
        root_nodes = select_nodes_by_plane(mesh, plane="yz", bound="min")
        clamp = DirichletConstraint(root_nodes, (0, 1, 2), 0.0)
        dirichlet = DirichletManager(mesh.n_dofs)
        dirichlet.add_constraint(clamp)

        # 1. Case A: Gravity only in air (sagging down in -Z)
        dom_air = SolidDomain(mesh.copy(), material, element)
        gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
        M = dom_air.assemble_mass_matrix()
        f_grav = gravity.compute_force_vector(M)

        solver_air = LinearStaticAnalysis(dom_air)
        res_air = solver_air.solve(dirichlet=dirichlet, f_ext=f_grav, verbose=False)
        disp_air = np.linalg.norm(res_air.displacement.reshape(-1, 3), axis=1).max()

        # 2. Case B: Gravity + Buoyancy in water (Archimedean buoyant relief)
        dom_water = SolidDomain(mesh.copy(), material, element)
        buoyancy = BuoyancyLoad(
            fluid_density=rho_fluid,
            acceleration=(0.0, 0.0, -9810.0),
        )
        f_buoy = buoyancy.compute_force_vector(dom_water)
        f_net = f_grav + f_buoy

        solver_water = LinearStaticAnalysis(dom_water)
        res_water = solver_water.solve(dirichlet=dirichlet, f_ext=f_net, verbose=False)
        disp_water = np.linalg.norm(res_water.displacement.reshape(-1, 3), axis=1).max()

        # Since rho_solid = 2000 and rho_fluid = 1000, net weight is exactly halved (50% reduction)
        np.testing.assert_allclose(disp_water, 0.5 * disp_air, rtol=1e-3)
        assert disp_water < disp_air

        # 3. Case C: Surface Hydrostatic Pressure (surface traction)
        dom_hydro = SolidDomain(mesh.copy(), material, element)
        hydro = HydrostaticPressureLoad(
            fluid_density=rho_fluid,
            acceleration=(0.0, 0.0, -9810.0),
        )
        f_hydro = hydro.compute_force_vector(dom_hydro)
        f_net_hydro = f_grav + f_hydro

        solver_hydro = LinearStaticAnalysis(dom_hydro)
        res_hydro = solver_hydro.solve(dirichlet=dirichlet, f_ext=f_net_hydro, verbose=False)
        disp_hydro = np.linalg.norm(res_hydro.displacement.reshape(-1, 3), axis=1).max()

        # Surface pressure traction and volumetric buoyancy agree to < 1% deflection
        np.testing.assert_allclose(disp_hydro, disp_water, rtol=0.01)

    def test_parameter_validation(self, five_tet_cube_mesh: VolumetricMesh) -> None:
        """Verifies input validation and exception handling."""
        with pytest.raises(ValueError, match="acceleration norm must be strictly positive"):
            HydrostaticPressureLoad(acceleration=(0.0, 0.0, 0.0))

        with pytest.raises(ValueError, match="acceleration must have shape"):
            HydrostaticPressureLoad(acceleration=(0.0, -9810.0))  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="up_direction norm must be strictly positive"):
            HydrostaticPressureLoad(up_direction=(0.0, 0.0, 0.0))

        with pytest.raises(ValueError, match="acceleration norm must be strictly positive"):
            BuoyancyLoad(acceleration=(0.0, 0.0, 0.0))
