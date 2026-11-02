from __future__ import annotations
import pytest
import numpy as np

from psierra.mesh import VolumetricMesh
from psierra.boundary.selector import select_nodes_by_plane
from psierra.boundary import DirichletConstraint, DirichletManager, GravityLoad
from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear
from psierra.domain import SolidDomain
from psierra.damping import RayleighDamping
from psierra.analysis import InteractiveDynamicSimulation, SimulationStats


class TestInteractiveDynamicSimulation:
    """Verifies online/interactive dynamic simulation framework."""

    @pytest.fixture
    def beam_domain(self, five_tet_cube_mesh: VolumetricMesh) -> SolidDomain:
        mesh = five_tet_cube_mesh
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(
            youngs_modulus=10000.0, poissons_ratio=0.3, mass_density=1000e-6
        )
        mat = Hookean(params)
        elem = Tet4Linear(mat)
        return SolidDomain(mesh, mat, elem)

    def test_interactive_initialization_and_validation(
        self, beam_domain: SolidDomain
    ) -> None:
        """Verifies parameter bounds checking and pre-factorization."""
        domain = beam_domain
        fixed_nodes = select_nodes_by_plane(domain.mesh, plane="xy", bound="min")
        clamp = DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)

        # Negative or zero dt validation
        with pytest.raises(ValueError, match="Time step dt must be positive"):
            InteractiveDynamicSimulation(domain, dirichlet=[clamp], dt=-0.01)
        with pytest.raises(ValueError, match="Time step dt must be positive"):
            InteractiveDynamicSimulation(domain, dirichlet=[clamp], dt=0.0)

        # Invalid spectral_radius validation
        with pytest.raises(ValueError, match="spectral_radius must be in"):
            InteractiveDynamicSimulation(domain, dirichlet=[clamp], spectral_radius=-0.5)
        with pytest.raises(ValueError, match="spectral_radius must be in"):
            InteractiveDynamicSimulation(domain, dirichlet=[clamp], spectral_radius=1.5)

        # Successful creation with default rho_inf = 0.8
        sim = InteractiveDynamicSimulation(domain, dirichlet=[clamp], dt=0.005)
        assert sim.dt == 0.005
        assert sim.spectral_radius == 0.8
        assert np.isclose(sim.alpha_m, 1.0 / 3.0)
        assert np.isclose(sim.alpha_f, 4.0 / 9.0)
        assert sim.step_count == 0
        assert sim.current_time == 0.0
        assert hasattr(sim, "_solve_step")

    def test_interactive_stepping_and_time_advancement(
        self, beam_domain: SolidDomain
    ) -> None:
        """Verifies step() and advance_time() execution."""
        domain = beam_domain
        fixed_nodes = select_nodes_by_plane(domain.mesh, plane="xy", bound="min")
        clamp = DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)

        sim = InteractiveDynamicSimulation(domain, dirichlet=[clamp], dt=0.005)

        # Single step
        sim.step()
        assert sim.step_count == 1
        assert np.isclose(sim.current_time, 0.005)

        # Advance time by 0.05 seconds (10 sub-steps)
        substeps = sim.advance_time(0.05)
        assert substeps == 10
        assert sim.step_count == 11
        assert np.isclose(sim.current_time, 0.055)

        # Advance wall clock by 0.01s (2 sub-steps)
        clock_steps = sim.advance_wall_clock(0.01)
        assert clock_steps == 2
        assert sim.step_count == 13

    def test_interactive_perturbation_forces(
        self, beam_domain: SolidDomain
    ) -> None:
        """Verifies point force perturbations, excitations, and clear operations."""
        domain = beam_domain
        fixed_nodes = select_nodes_by_plane(domain.mesh, plane="xy", bound="min")
        top_nodes = select_nodes_by_plane(domain.mesh, plane="xy", bound="max")
        clamp = DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)

        sim = InteractiveDynamicSimulation(domain, dirichlet=[clamp], dt=0.005)

        # Apply perturbation force on top node
        target_node = int(top_nodes[0])
        sim.apply_perturbation_force(target_node, [100.0, 0.0, -50.0])
        assert sim.f_ext_dynamic[target_node * 3] == 100.0
        assert sim.f_ext_dynamic[target_node * 3 + 2] == -50.0

        # Step forward with perturbation
        sim.step()
        assert np.max(np.abs(sim.displacement)) > 0.0

        # Clear perturbation
        sim.clear_perturbation_forces()
        assert np.all(sim.f_ext_dynamic == 0.0)

        # Harmonic excitation
        sim.add_harmonic_excitation(target_node, component=2, amplitude=50.0, frequency_hz=10.0)
        assert len(sim._harmonic_sources) == 1
        sim.step()
        sim.clear_harmonic_excitations()
        assert len(sim._harmonic_sources) == 0

    def test_interactive_energy_and_performance_telemetry(
        self, beam_domain: SolidDomain
    ) -> None:
        """Verifies kinetic energy, strain energy, and speed ratio tracking."""
        domain = beam_domain
        fixed_nodes = select_nodes_by_plane(domain.mesh, plane="xy", bound="min")
        top_nodes = select_nodes_by_plane(domain.mesh, plane="xy", bound="max")
        clamp = DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        damping = RayleighDamping(alpha=1.0, beta=0.001)

        sim = InteractiveDynamicSimulation(
            domain, damping=damping, dirichlet=[clamp], dt=0.005
        )

        # Distort structure with force pulse
        sim.apply_perturbation_force(int(top_nodes[0]), [0.0, 0.0, -200.0])
        sim.advance_time(0.05)
        sim.clear_perturbation_forces()
        sim.advance_time(0.05)

        # Energy checks
        ke = sim.kinetic_energy
        se = sim.strain_energy
        te = sim.total_energy
        assert ke >= 0.0
        assert se >= 0.0
        assert np.isclose(te, ke + se)

        # Telemetry stats
        stats = sim.get_performance_stats()
        assert isinstance(stats, SimulationStats)
        assert stats.steps_completed == 20
        assert np.isclose(stats.sim_time, 0.10)
        assert stats.wall_time > 0.0
        assert stats.average_step_ms > 0.0
        assert stats.speed_ratio > 0.0

    def test_interactive_reset(self, beam_domain: SolidDomain) -> None:
        """Verifies that reset() restores initial state."""
        domain = beam_domain
        fixed_nodes = select_nodes_by_plane(domain.mesh, plane="xy", bound="min")
        top_nodes = select_nodes_by_plane(domain.mesh, plane="xy", bound="max")
        clamp = DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)

        sim = InteractiveDynamicSimulation(domain, dirichlet=[clamp], dt=0.005)
        sim.apply_perturbation_force(int(top_nodes[0]), [50.0, 50.0, 50.0])
        sim.advance_time(0.1)

        assert sim.step_count > 0
        assert sim.current_time > 0.0
        assert np.max(np.abs(sim.displacement)) > 0.0

        sim.reset()
        assert sim.step_count == 0
        assert sim.current_time == 0.0
        assert np.allclose(sim.displacement, 0.0)
        assert np.allclose(sim.velocity, 0.0)

    def test_interactive_solve_method(self, beam_domain: SolidDomain) -> None:
        """Verifies Analysis base class solve() execution."""
        domain = beam_domain
        fixed_nodes = select_nodes_by_plane(domain.mesh, plane="xy", bound="min")
        clamp = DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)

        sim = InteractiveDynamicSimulation(domain, dirichlet=[clamp], dt=0.01)
        stats = sim.solve(duration=0.05)
        assert isinstance(stats, SimulationStats)
        assert stats.steps_completed == 5
        assert np.isclose(stats.sim_time, 0.05)

    def test_interactive_closest_node_and_pyvista_update(
        self, beam_domain: SolidDomain
    ) -> None:
        """Verifies find_closest_node and update_pyvista_grid."""
        domain = beam_domain
        fixed_nodes = select_nodes_by_plane(domain.mesh, plane="xy", bound="min")
        clamp = DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)

        sim = InteractiveDynamicSimulation(domain, dirichlet=[clamp], dt=0.005)

        # Closest node test
        target_pt = domain.mesh.nodes[0]
        closest = sim.find_closest_node(target_pt, deformed=False)
        assert closest == 0

        # PyVista grid update test
        grid = domain.to_pyvista(factor=0.0)
        sim.apply_perturbation_force(0, [10.0, 0.0, 0.0])
        sim.step()
        sim.update_pyvista_grid(grid)
        assert np.allclose(grid.points, sim.deformed_nodes)
        assert np.allclose(grid.point_data["Displacement"], sim.displacement)
        assert np.allclose(grid.point_data["Displacement"][:, 1], sim.displacement[:, 1])

    def test_sim01_headless_cli_execution(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verifies sim01_online_dynamic_cantilever runs end-to-end via CLI in headless mode."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "examples/06_interactive/sim01_online_dynamic_cantilever.py",
            "--headless",
            "--duration",
            "0.02",
            "--dt",
            "0.005",
            "--save-snapshots",
            "--output-dir",
            str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Script failed with output:\n{result.stderr}"
        assert "Interactive Dynamic Simulation Pipeline completed successfully!" in result.stdout
        assert (tmp_path / "sim01_online_dynamic.pvd").exists()

    def test_interactive_pyvista_gui_loop(self, beam_domain: SolidDomain) -> None:
        """Verifies PyVista interactive loop and window active query without closed attribute errors."""
        import pyvista as pv

        domain = beam_domain
        fixed_nodes = select_nodes_by_plane(domain.mesh, plane="xy", bound="min")
        top_nodes = select_nodes_by_plane(domain.mesh, plane="xy", bound="max")
        clamp = DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)
        sim = InteractiveDynamicSimulation(domain, dirichlet=[clamp], dt=0.005)

        plotter = pv.Plotter(off_screen=True)
        grid = domain.to_pyvista(factor=1.0)
        actor = plotter.add_mesh(
            grid,
            scalars="Displacement",
            component=1,
            clim=[-10.0, 10.0],
            cmap="coolwarm",
        )
        plotter.show(interactive_update=True)
        img_initial = plotter.screenshot()

        state = {"exit_requested": False}

        def is_window_active() -> bool:
            if state["exit_requested"]:
                return False
            if getattr(plotter, "_closed", False):
                return False
            if getattr(plotter, "render_window", None) is None:
                return False
            return True

        assert is_window_active()
        sim.apply_perturbation_force(int(top_nodes[0]), [0.0, -100.0, 0.0])
        for _ in range(5):
            sim.step()
            sim.update_pyvista_grid(
                grid,
                plotter=plotter,
                field="Displacement",
                symmetric_clim=True,
                min_clim=10.0,
            )
            plotter.update()

        img_deformed = plotter.screenshot()
        pixel_diff = np.abs(img_initial.astype(float) - img_deformed.astype(float)).sum()
        assert pixel_diff > 0, "Colormap failed to update dynamically during interactive loop!"

        plotter.close()
        assert not is_window_active()

    def test_interactive_nonlinear_tet10_stepping(
        self, five_tet_cube_mesh: VolumetricMesh
    ) -> None:
        """Verifies interactive dynamic simulation in nonlinear mode with Tet10."""
        from psierra.material import NeoHookean
        from psierra.element import Tet10TotalLagrangian

        mesh10 = five_tet_cube_mesh.to_quadratic()
        params = IsotropicElasticParameters()
        params.set_from_youngs_poisson(10000.0, 0.3, 1000e-6)
        mat = NeoHookean(params)
        elem = Tet10TotalLagrangian(mat)
        domain = SolidDomain(mesh10, mat, elem)

        fixed_nodes = select_nodes_by_plane(mesh10, plane="xy", bound="min")
        top_nodes = select_nodes_by_plane(mesh10, plane="xy", bound="max")
        clamp = DirichletConstraint(fixed_nodes, components=(0, 1, 2), value=0.0)

        sim = InteractiveDynamicSimulation(
            domain,
            dirichlet=[clamp],
            dt=0.005,
            spectral_radius=0.8,
            nonlinear=True,
            nr_max_iter=2,
            modified_newton=True,
        )

        sim.apply_perturbation_force(int(top_nodes[0]), [0.0, 0.0, -100.0])
        sim.step()
        assert sim.step_count == 1
        assert np.max(np.abs(sim.displacement)) > 0.0
        assert sim.strain_energy >= 0.0

    def test_sim02_headless_cli_execution(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verifies sim02_online_dynamic_axial_torsion runs end-to-end via CLI in headless mode."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "examples/06_interactive/sim02_online_dynamic_axial_torsion.py",
            "--headless",
            "--duration",
            "0.02",
            "--dt",
            "0.005",
            "--formulation",
            "linear",
            "--save-snapshots",
            "--output-dir",
            str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Script failed with output:\n{result.stderr}"
        assert "Interactive Dynamic Axial Torsion Pipeline completed successfully!" in result.stdout
        assert (tmp_path / "sim02_online_dynamic_torsion.pvd").exists()
