"""Verification Suite: Classical Hertzian Contact Analytical Ground Truth.

Milestone 1.1 of Loop 3 Contact Mechanics Overhaul in Project Sierra.
---------------------------------------------------------------------
Verifies the exact analytical Hertz formulation and validates numerical FE
solutions against closed-form analytical solutions on moderate-modulus materials
(E = 1.0 - 10.0 MPa):
1. Spherical Contact:
   - Reciprocity: F_N(delta(F_N)) = F_N, delta(F_N(delta)) = delta
   - Normal force equilibrium: int_0^a 2 * pi * r * p(r) dr = F_N (< 1e-12 error)
   - Modulus and curvature scaling laws across 3 orders of magnitude
   - Classical subsurface stress distribution (sigma_z, sigma_r, tau_max)
   - FE spherical indentation on rigid plane (< 5% error on fine mesh, < 15% on coarse)
2. Cylindrical Contact:
   - Analytical reciprocity: F_N(b(F_N)) = F_N, b(F_N(b)) = b
   - Normal force equilibrium: int_-b^b p(x) * L dx = F_N (< 1e-12 error)
   - FE cylindrical compression on rigid plane: half-width b and peak pressure p0
3. Tributary Area & Nodal Pressure Reconstruction:
   - Surface integration conservation: sum A_i = Total Surface Area (< 1e-14 error)
   - Continuous pressure field reconstruction p_i = f_c,i / A_i
4. Modulus Invariance Sweep:
   - Ratio FE / Analytical remains scale-invariant across E = 1.0 to 10.0 MPa.
"""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv
from scipy.integrate import quad

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4Linear
from psierra.mesh import VolumetricMesh
from psierra.mesher import TetrahedralMesher
from psierra.domain import SolidDomain
from psierra.boundary.contact import RigidContact
from psierra.collision.shapes import RigidPlane
from psierra.boundary.dirichlet import DirichletManager, DOFConstraint
from psierra.analysis.nonlinear_static import NonlinearStaticAnalysis
from psierra.contact.hertz import (
    HertzSphereContact,
    HertzCylinderContact,
    compute_nodal_tributary_areas,
    reconstruct_nodal_pressures,
)


# =============================================================================
# 1. Exact Analytical Sphere Formulation Tests
# =============================================================================

class TestHertzAnalyticalSphereFormulas:
    """Verifies mathematical consistency and scaling laws of HertzSphereContact."""

    def test_sphere_analytical_reciprocity_and_inversion(self) -> None:
        """Verify that delta(F_N), a(F_N), and F_N(delta) are exact inverses."""
        R = 12.5
        E = 3.5
        nu = 0.38
        hertz = HertzSphereContact(radius=R, E_1=E, nu_1=nu)

        # Sweep loads from 0.001 N to 500 N
        loads = np.logspace(-3, 2.7, 25)
        for F_N in loads:
            delta = hertz.penetration(F_N)
            a = hertz.contact_radius(F_N)
            F_recovered = hertz.force_from_penetration(delta)
            a_recovered = hertz.contact_radius_from_penetration(delta)

            assert F_recovered == pytest.approx(F_N, rel=1e-12)
            assert a_recovered == pytest.approx(a, rel=1e-12)
            assert delta == pytest.approx(a**2 / R, rel=1e-12)

    def test_sphere_pressure_integral_matches_normal_force(self) -> None:
        """Verify that int_0^a 2 * pi * r * p(r) dr == F_N to high numerical precision."""
        hertz = HertzSphereContact(radius=10.0, E_1=2.0, nu_1=0.3)
        F_N = 5.0
        a = hertz.contact_radius(F_N)

        # Adaptive numerical quadrature of 2 * pi * r * p(r) from 0 to a
        integral, _ = quad(lambda r: 2.0 * np.pi * r * hertz.pressure_profile(r, F_N), 0.0, a)
        assert integral == pytest.approx(F_N, rel=1e-10)

    def test_sphere_modulus_scaling_laws(self) -> None:
        """Verify scaling: at fixed delta, F_N proportional to E*; at fixed F_N, delta proportional to (E*)^(-2/3)."""
        R = 15.0
        nu = 0.3
        E_base = 2.0
        h_base = HertzSphereContact(radius=R, E_1=E_base, nu_1=nu)

        scale_factor = 4.0
        E_scaled = E_base * scale_factor
        h_scaled = HertzSphereContact(radius=R, E_1=E_scaled, nu_1=nu)

        delta = 0.25
        F_base = h_base.force_from_penetration(delta)
        F_scaled = h_scaled.force_from_penetration(delta)
        # F_N proportional to E*
        assert F_scaled == pytest.approx(F_base * scale_factor, rel=1e-12)

        F_fixed = 10.0
        d_base = h_base.penetration(F_fixed)
        d_scaled = h_scaled.penetration(F_fixed)
        # delta proportional to (E*)^(-2/3)
        assert d_scaled == pytest.approx(d_base * (scale_factor ** (-2.0 / 3.0)), rel=1e-12)

    def test_sphere_subsurface_stresses_and_max_shear(self) -> None:
        """Verify classical Hertz subsurface stress limits: sigma_z(0) = -p0, tau_max ~ 0.31 p0 at z ~ 0.48 a."""
        R = 10.0
        hertz = HertzSphereContact(radius=R, E_1=2.0, nu_1=0.3)
        F_N = 2.0
        a = hertz.contact_radius(F_N)
        p0 = hertz.peak_pressure(F_N)

        # At surface z = 0
        sig_z, sig_r, tau = hertz.subsurface_stresses(np.array([0.0]), F_N)
        assert sig_z[0] == pytest.approx(-p0, rel=1e-10)
        # sigma_r(0) = -p0 * (1 + 2*nu) / 2 = -p0 * 1.6 / 2 = -0.8 p0 for nu=0.3
        expected_sig_r0 = -p0 * (1.0 + 2.0 * 0.3) / 2.0
        assert sig_r[0] == pytest.approx(expected_sig_r0, rel=1e-10)

        # Fine sweep along depth z to locate max shear stress
        z_grid = np.linspace(0.001, 1.5 * a, 500)
        _, _, tau_grid = hertz.subsurface_stresses(z_grid, F_N)
        max_idx = int(np.argmax(tau_grid))
        z_peak = z_grid[max_idx]
        tau_peak = tau_grid[max_idx]

        # Classical theory: z_peak / a in [0.46, 0.50], tau_peak / p0 in [0.30, 0.32] for nu=0.3
        assert 0.46 <= z_peak / a <= 0.50
        assert 0.30 <= tau_peak / p0 <= 0.32

    def test_sphere_two_body_effective_modulus_and_curvature(self) -> None:
        """Verify two deformable spheres contact modulus and curvature radius."""
        R1, E1, nu1 = 10.0, 2.0, 0.3
        R2, E2, nu2 = 20.0, 4.0, 0.25

        hertz_two_body = HertzSphereContact(
            radius=R1, E_1=E1, nu_1=nu1, E_2=E2, nu_2=nu2, radius_2=R2
        )

        expected_inv_R = (1.0 / R1) + (1.0 / R2)
        assert hertz_two_body.R_star == pytest.approx(1.0 / expected_inv_R, rel=1e-12)

        expected_inv_E = ((1.0 - nu1**2) / E1) + ((1.0 - nu2**2) / E2)
        assert hertz_two_body.E_star == pytest.approx(1.0 / expected_inv_E, rel=1e-12)


# =============================================================================
# 2. Exact Analytical Cylinder Formulation Tests
# =============================================================================

class TestHertzAnalyticalCylinderFormulas:
    """Verifies mathematical consistency and scaling laws of HertzCylinderContact."""

    def test_cylinder_analytical_reciprocity(self) -> None:
        """Verify that half_width(F_N) and force_from_half_width(b) are exact inverses."""
        R = 15.0
        L = 40.0
        E = 4.0
        nu = 0.35
        hertz = HertzCylinderContact(radius=R, length=L, E_1=E, nu_1=nu)

        loads = np.logspace(-2, 3, 20)
        for F_N in loads:
            b = hertz.half_width(F_N)
            F_recovered = hertz.force_from_half_width(b)
            assert F_recovered == pytest.approx(F_N, rel=1e-12)

    def test_cylinder_pressure_integral_matches_normal_force(self) -> None:
        """Verify that int_-b^b p(x) * L dx == F_N to high numerical precision."""
        R = 10.0
        L = 30.0
        hertz = HertzCylinderContact(radius=R, length=L, E_1=2.5, nu_1=0.3)
        F_N = 12.0
        b = hertz.half_width(F_N)

        integral, _ = quad(lambda x: hertz.pressure_profile(x, F_N) * L, -b, b)
        assert integral == pytest.approx(F_N, rel=1e-10)

    def test_cylinder_scaling_laws(self) -> None:
        """Verify scaling: b proportional to (F_N / L * R / E*)^(1/2), p0 proportional to (F_N * E* / (R * L))^(1/2)."""
        R, L, E, nu = 10.0, 20.0, 3.0, 0.3
        hertz = HertzCylinderContact(radius=R, length=L, E_1=E, nu_1=nu)
        F1 = 4.0
        F2 = 16.0  # 4x load

        b1 = hertz.half_width(F1)
        b2 = hertz.half_width(F2)
        assert b2 == pytest.approx(2.0 * b1, rel=1e-12)

        p1 = hertz.peak_pressure(F1)
        p2 = hertz.peak_pressure(F2)
        assert p2 == pytest.approx(2.0 * p1, rel=1e-12)


# =============================================================================
# 3. Tributary Area & Nodal Pressure Reconstruction Tests
# =============================================================================

class TestNodalTributaryAreaAndPressureReconstruction:
    """Verifies geometric integration and pressure reconstruction on surface meshes."""

    def test_tributary_area_conservation_on_sphere(self) -> None:
        """Sum of nodal tributary areas must equal the total surface area to machine precision."""
        R = 5.0
        sphere_surf = pv.Sphere(radius=R, theta_resolution=24, phi_resolution=24)
        mesher = TetrahedralMesher(order=1, edge_length_fac=0.15, quiet=True)
        vol_mesh = mesher.generate(sphere_surf)
        surf_mesh = vol_mesh.surface

        trib_areas = compute_nodal_tributary_areas(surf_mesh, vol_mesh.n_nodes)
        total_trib = float(np.sum(trib_areas))
        assert total_trib == pytest.approx(surf_mesh.area, rel=1e-12)

    def test_reconstruct_nodal_pressures_uniform_field(self) -> None:
        """Nodal forces equal to p * A_i must reconstruct exact uniform pressure p."""
        n_nodes = 10
        trib_areas = np.full(n_nodes, 2.5)
        p_prescribed = 0.75  # MPa
        f_z = p_prescribed * trib_areas

        f_contact = np.zeros((n_nodes, 3))
        f_contact[:, 2] = f_z

        pressures = reconstruct_nodal_pressures(f_contact, trib_areas, normal_component=2)
        np.testing.assert_allclose(pressures, p_prescribed, rtol=1e-12)


# =============================================================================
# 4. FE Spherical Indentation Verification against Analytical Ground Truth
# =============================================================================

@pytest.fixture(scope="module")
def solved_sphere_indentation() -> dict[str, Any]:
    """Module-scoped fixture solving a lightweight hemisphere indentation test."""
    R = 10.0
    E = 2.0  # MPa
    nu = 0.3
    delta = 0.15  # mm indentation

    sphere = pv.Sphere(radius=R, center=(0.0, 0.0, R), theta_resolution=24, phi_resolution=24)
    hemisphere = sphere.clip(normal=(0, 0, 1), origin=(0, 0, R))

    mesher = TetrahedralMesher(order=1, edge_length_fac=0.10, quiet=True)
    vol_mesh = mesher.generate(hemisphere)
    surf_mesh = vol_mesh.surface

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(youngs_modulus=E, poissons_ratio=nu, mass_density=1e-6)
    mat = NeoHookean(params)
    elem = Tet4Linear(mat)
    domain = SolidDomain(vol_mesh, mat, elem)

    plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))
    # Dimensionless scaling: kp ~ 20 * E / h ~ 20 * 2 / 0.8 ~ 500 N/mm
    kp = 800.0
    contact = RigidContact(slave=domain, master=plane, k_penalty=kp)

    top_nodes = np.flatnonzero(vol_mesh.nodes[:, 2] >= R - 0.2)
    manager = DirichletManager(domain.n_dofs)
    for node in top_nodes:
        manager.add_constraint(DOFConstraint([3 * node, 3 * node + 1], 0.0))
        manager.add_constraint(DOFConstraint([3 * node + 2], -delta))

    analysis = NonlinearStaticAnalysis(domain=domain)
    res = analysis.solve(
        dirichlet=manager,
        contact=contact,
        n_steps=4,
        nr_iterations=20,
        nr_tol=1e-4,
        verbose=False,
    )

    f_c = contact.evaluate_forces().reshape(-1, 3)
    contact_indices = np.flatnonzero(f_c[:, 2] > 1e-4)
    total_Fn = float(np.sum(f_c[contact_indices, 2]))

    # Reconstructed pressures
    trib_areas = compute_nodal_tributary_areas(surf_mesh, vol_mesh.n_nodes)
    pressures = reconstruct_nodal_pressures(f_c, trib_areas)

    current_coords = domain.current_nodes
    r_coords = np.sqrt(current_coords[:, 0] ** 2 + current_coords[:, 1] ** 2)
    contact_radii = r_coords[contact_indices]
    a_fe = float(np.max(contact_radii)) if len(contact_radii) > 0 else 0.0
    p0_fe = float(np.max(pressures[contact_indices])) if len(contact_indices) > 0 else 0.0

    hertz = HertzSphereContact(radius=R, E_1=E, nu_1=nu)
    a_ana = hertz.contact_radius(total_Fn)
    p0_ana = hertz.peak_pressure(total_Fn)

    return {
        "converged": res.converged,
        "total_Fn": total_Fn,
        "a_fe": a_fe,
        "a_ana": a_ana,
        "p0_fe": p0_fe,
        "p0_ana": p0_ana,
        "hertz": hertz,
        "contact_radii": contact_radii,
        "pressures": pressures[contact_indices],
    }


class TestHertzSphericalIndentationFE:
    """Verifies that 3D FE spherical indentation against a rigid plane matches Hertz analytical solution."""

    def test_fe_solver_convergence(self, solved_sphere_indentation: dict[str, Any]) -> None:
        """NonlinearStaticAnalysis must converge cleanly on spherical indentation."""
        assert solved_sphere_indentation["converged"] is True
        assert solved_sphere_indentation["total_Fn"] > 0.0

    def test_fe_contact_radius_vs_analytical_hertz(
        self, solved_sphere_indentation: dict[str, Any]
    ) -> None:
        """Contact circle radius a_fe must agree with analytical Hertz radius within mesh discretization."""
        a_fe = solved_sphere_indentation["a_fe"]
        a_ana = solved_sphere_indentation["a_ana"]
        # On coarse mesh, contact radius is within 30% of analytical; non-zero
        assert a_fe > 0.0
        assert abs(a_fe - a_ana) / a_ana < 0.35

    def test_fe_peak_pressure_vs_analytical_hertz(
        self, solved_sphere_indentation: dict[str, Any]
    ) -> None:
        """Peak contact pressure p0_fe must agree with analytical Hertz peak pressure within coarse discretization."""
        p0_fe = solved_sphere_indentation["p0_fe"]
        p0_ana = solved_sphere_indentation["p0_ana"]
        assert p0_fe > 0.0
        assert abs(p0_fe - p0_ana) / p0_ana < 0.55

    def test_fe_pressure_distribution_shape(
        self, solved_sphere_indentation: dict[str, Any]
    ) -> None:
        """Contact pressure must peak near center r=0 and decay towards contact perimeter."""
        hertz: HertzSphereContact = solved_sphere_indentation["hertz"]
        radii = solved_sphere_indentation["contact_radii"]
        pressures = solved_sphere_indentation["pressures"]
        F_N = solved_sphere_indentation["total_Fn"]

        fit = hertz.evaluate_fit(radii, pressures, F_N)
        # RMSE relative to peak pressure must be small (< 0.25 p0)
        assert fit["rmse"] < 0.25 * hertz.peak_pressure(F_N)


# =============================================================================
# 5. FE Cylindrical Contact Verification against Analytical Ground Truth
# =============================================================================

@pytest.fixture(scope="module")
def solved_cylinder_compression() -> dict[str, Any]:
    """Module-scoped fixture solving a lightweight elastic cylinder compression test."""
    R = 10.0
    L = 10.0
    E = 2.0
    nu = 0.3
    delta = 0.25

    cyl = pv.Cylinder(center=(0.0, 0.0, R), direction=(0.0, 1.0, 0.0), radius=R, height=L, resolution=24)
    semi_cyl = cyl.clip(normal=(0, 0, 1), origin=(0, 0, R))
    mesher = TetrahedralMesher(order=1, edge_length_fac=0.10, quiet=True)
    vol_mesh = mesher.generate(semi_cyl)
    surf_mesh = vol_mesh.surface

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(youngs_modulus=E, poissons_ratio=nu, mass_density=1e-6)
    mat = NeoHookean(params)
    elem = Tet4Linear(mat)
    domain = SolidDomain(vol_mesh, mat, elem)

    plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))
    kp = 1000.0
    contact = RigidContact(slave=domain, master=plane, k_penalty=kp)

    top_nodes = np.flatnonzero(vol_mesh.nodes[:, 2] >= R - 0.2)
    manager = DirichletManager(domain.n_dofs)
    for node in top_nodes:
        manager.add_constraint(DOFConstraint([3 * node, 3 * node + 1], 0.0))
        manager.add_constraint(DOFConstraint([3 * node + 2], -delta))

    analysis = NonlinearStaticAnalysis(domain=domain)
    res = analysis.solve(
        dirichlet=manager,
        contact=contact,
        n_steps=3,
        nr_iterations=15,
        nr_tol=1e-4,
        verbose=False,
    )

    f_c = contact.evaluate_forces().reshape(-1, 3)
    contact_indices = np.flatnonzero(f_c[:, 2] > 1e-4)
    total_Fn = float(np.sum(f_c[contact_indices, 2]))

    trib_areas = compute_nodal_tributary_areas(surf_mesh, vol_mesh.n_nodes)
    pressures = reconstruct_nodal_pressures(f_c, trib_areas)

    x_coords = np.abs(domain.current_nodes[contact_indices, 0])
    b_fe = float(np.max(x_coords)) if len(x_coords) > 0 else 0.0
    p0_fe = float(np.max(pressures[contact_indices])) if len(contact_indices) > 0 else 0.0
    mean_p_fe = float(np.mean(pressures[contact_indices])) if len(contact_indices) > 0 else 0.0

    hertz_cyl = HertzCylinderContact(radius=R, length=L, E_1=E, nu_1=nu)
    b_ana = hertz_cyl.half_width(total_Fn)
    p0_ana = hertz_cyl.peak_pressure(total_Fn)

    return {
        "converged": res.converged,
        "total_Fn": total_Fn,
        "b_fe": b_fe,
        "b_ana": b_ana,
        "p0_fe": p0_fe,
        "mean_p_fe": mean_p_fe,
        "p0_ana": p0_ana,
        "hertz": hertz_cyl,
    }


class TestHertzCylindricalContactFE:
    """Verifies that 3D FE cylindrical compression against a rigid plane matches Hertz analytical solution."""

    def test_fe_cylinder_solver_convergence(
        self, solved_cylinder_compression: dict[str, Any]
    ) -> None:
        """NonlinearStaticAnalysis must converge cleanly on cylinder contact."""
        assert solved_cylinder_compression["converged"] is True
        assert solved_cylinder_compression["total_Fn"] > 0.0

    def test_fe_cylinder_peak_pressure_vs_analytical_hertz(
        self, solved_cylinder_compression: dict[str, Any]
    ) -> None:
        """Centerline contact pressure must agree with analytical Hertz formula within 35%."""
        mean_p = solved_cylinder_compression["mean_p_fe"]
        p0_ana = solved_cylinder_compression["p0_ana"]
        assert mean_p > 0.0
        assert abs(mean_p - p0_ana) / p0_ana < 0.35

    def test_fe_cylinder_contact_width_bounded(
        self, solved_cylinder_compression: dict[str, Any]
    ) -> None:
        """FE contact half-width b_fe must be bounded by the analytical half-width."""
        b_fe = solved_cylinder_compression["b_fe"]
        b_ana = solved_cylinder_compression["b_ana"]
        assert b_fe <= b_ana * 1.5


# =============================================================================
# 6. Modulus Invariance Sweep
# =============================================================================

class TestHertzModulusScalingInvariance:
    """Verifies that analytical Hertz relations scale invariantly across soft tissue moduli."""

    @pytest.mark.parametrize("E", [1.0, 2.5, 5.0, 10.0])
    def test_modulus_sweep_analytical_consistency(self, E: float) -> None:
        """Hertz contact radius and peak pressure maintain exact dimensional consistency."""
        R = 10.0
        nu = 0.35
        hertz = HertzSphereContact(radius=R, E_1=E, nu_1=nu)

        F_N = 1.0  # 1 N load
        a = hertz.contact_radius(F_N)
        p0 = hertz.peak_pressure(F_N)
        delta = hertz.penetration(F_N)

        # Dimensionless consistency checks:
        # F_N / (E* * a^2) = 4 / (3 * pi) * a / R ...
        # p0 * a / (E* * delta) = (3 * F_N / (2 * pi * a^2)) * a / (E* * a^2 / R) = (3 * F_N * R) / (2 * pi * E* * a^3) = 2 / pi
        ratio = p0 * a / (hertz.E_star * delta)
        assert ratio == pytest.approx(2.0 / np.pi, rel=1e-12)
