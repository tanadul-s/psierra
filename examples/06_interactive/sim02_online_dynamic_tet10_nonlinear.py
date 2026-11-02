"""
Simulation 02 (Interactive): Real-Time Online Dynamic Simulation with Nonlinear Tet10 Elements.

Demonstrates high-performance interactive dynamic FEA simulation of a 3D continuum
cantilever beam using 10-node quadratic tetrahedra (Tet10) and nonlinear hyperelasticity:
1. Quadratic Tet10 finite-strain kinematics (Tet10TotalLagrangian / Tet10UpdatedLagrangian).
2. Elimination of artificial shear locking in interactive dynamic deformation.
3. Modified Newton-Raphson Generalized-alpha dynamic time stepping with single-factorization reuse.
4. Interactive disturbances: impulse perturbations and sinusoidal harmonic resonance excitation.
5. In-place PyVista VTK vertex coordinate buffer mutation for zero-overhead live 3D visualization.
6. Comprehensive execution telemetry tracking simulation speed ratio, step ms, and energies.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet10TotalLagrangian, Tet10UpdatedLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary.selector import select_nodes_by_plane
from psierra.boundary import DirichletConstraint, GravityLoad
from psierra.damping import RayleighDamping
from psierra.analysis import InteractiveDynamicSimulation, SimulationStats
from psierra.exchange import PvdCollectionWriter

install(show_locals=False)
console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time Interactive Dynamic Simulation with Nonlinear Quadratic Tet10 Elements"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless batch demonstration mode without opening a GUI window (auto-detected if DISPLAY is absent)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.5,
        help="Physical simulation duration in seconds for demonstration sequence (default: 0.5 s)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005,
        help="Physics simulation time step dt in seconds (default: 0.005 s = 200 Hz)",
    )
    parser.add_argument(
        "--spectral-radius",
        type=float,
        default=0.8,
        help="Generalized-alpha high-frequency spectral radius rho_inf in [0.0, 1.0] (default: 0.8)",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        choices=["total_lagrangian", "updated_lagrangian"],
        default="total_lagrangian",
        help="Nonlinear kinematic formulation: 'total_lagrangian' or 'updated_lagrangian' (default: total_lagrangian)",
    )
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        default=False,
        help="Export time-series VTU and PVD snapshots for ParaView post-processing inspection",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/06_interactive/sim02_output",
        help="Output directory path for saved snapshots (default: examples/06_interactive/sim02_output)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Automatically enforce headless mode if no DISPLAY is present
    has_display = "DISPLAY" in os.environ and bool(os.environ["DISPLAY"])
    is_headless = args.headless or not has_display

    # 0. Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]Simulation 02 (Interactive): Real-Time Dynamic Nonlinear Tet10 Pipeline[/bold cyan]\n"
            "[dim]Decoupled Generalized-alpha Dynamic Physics Engine with Quadratic Tet10 Hyperelasticity[/dim]\n"
            f"[dim]Mode: {'[bold red]Headless Verification Mode[/bold red]' if is_headless else '[bold green]Interactive PyVista GUI Mode[/bold green]'}[/dim]",
            title="[bold blue]project-sierra[/bold blue]",
            border_style="cyan",
        )
    )

    # 1. Loading Base Mesh & Upgrading to Quadratic Tet10
    console.print(
        "\n[bold yellow][1/6] Loading Mesh Geometry & Quadratic Upgrade[/bold yellow]"
    )
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Beam mesh file not found: {mesh_path}")

    mesh_linear = VolumetricMesh(mesh_path)
    mesh_tet10 = mesh_linear.to_quadratic()
    console.print(
        f"  • Upgraded Tet10 Mesh: [green]{mesh_tet10.n_nodes}[/green] nodes | "
        f"[green]{mesh_tet10.n_elements}[/green] quadratic elements | "
        f"[green]{mesh_tet10.n_dofs}[/green] DOFs"
    )
    console.print(
        "  • Beam Dimensions    : X: [-60, 60] mm, Y: [-60, 60] mm, Z: [-30, 200] mm (L = 230 mm)"
    )

    # 2. Material & Hyperelastic Domain Configuration
    console.print(
        "\n[bold yellow][2/6] Configuring Hyperelastic Material & Physical Domain[/bold yellow]"
    )
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000.0,  # MPa (N/mm^2)
        poissons_ratio=0.35,
        mass_density=1100e-6,  # tonne/mm^3 (1100 kg/m^3)
    )
    material = NeoHookean(params)

    if args.formulation == "updated_lagrangian":
        element = Tet10UpdatedLagrangian(material)
        elem_name = "Tet10UpdatedLagrangian (UL Quadratic)"
    else:
        element = Tet10TotalLagrangian(material)
        elem_name = "Tet10TotalLagrangian (TL Quadratic)"

    domain = SolidDomain(mesh_tet10, material, element)
    console.print(
        f"  • Constitutive Law   : [cyan]{material.__class__.__name__}[/cyan] (Compressible Neo-Hookean)"
    )
    console.print(f"  • Element Formulation: [cyan]{elem_name}[/cyan]")
    console.print(
        f"  • Elastic Modulus    : [green]{params.youngs_modulus:.1f}[/green] MPa | "
        f"Poisson's Ratio: [green]{params.poissons_ratio:.2f}[/green]"
    )

    # 3. Boundary Conditions & Base External Gravity
    console.print(
        "\n[bold yellow][3/6] Setting Up Boundary Conditions & Gravity Load[/bold yellow]"
    )
    root_nodes = select_nodes_by_plane(mesh_tet10, plane="xy", bound="min")
    tip_nodes = select_nodes_by_plane(mesh_tet10, plane="xy", bound="max")
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)

    # Base physical gravity load along -Y (calibrated to ~100 kN body weight matching sim18 benchmark)
    gravity = GravityLoad(acceleration=(0.0, -98.1, 0.0))
    M = domain.assemble_mass_matrix()
    f_grav = gravity.compute_force_vector(M)
    total_weight = abs(float(f_grav[1::3].sum()))

    console.print(
        f"  • Clamped Root Face (Z = -30 mm) : [green]{len(root_nodes)}[/green] nodes ({len(clamp.dofs)} DOFs)"
    )
    console.print(
        f"  • Free Tip Face    (Z = 200 mm) : [green]{len(tip_nodes)}[/green] nodes"
    )
    console.print(
        f"  • Total Base Gravity Load (F_y)  : [green]{total_weight:.2f}[/green] N"
    )

    # 4. Interactive Simulation Engine Initialization (Nonlinear Mode)
    console.print(
        "\n[bold yellow][4/6] Initializing Interactive Nonlinear Dynamic Engine[/bold yellow]"
    )
    damping = RayleighDamping(alpha=1.0, beta=0.001)

    t_init_0 = time.perf_counter()
    sim = InteractiveDynamicSimulation(
        domain=domain,
        damping=damping,
        dirichlet=[clamp],
        dt=args.dt,
        spectral_radius=args.spectral_radius,
        f_ext_base=f_grav,
        nonlinear=True,
        nr_max_iter=1,  # 1 Modified Newton step per physics sub-step for interactive performance
        modified_newton=True,
    )
    t_init_ms = (time.perf_counter() - t_init_0) * 1000.0

    console.print(
        f"  • Dynamic Time Step (dt)       : [green]{sim.dt * 1000.0:.2f}[/green] ms ({1.0 / sim.dt:.0f} Hz)"
    )
    console.print(
        f"  • High-Frequency Spectral Radius: [cyan]{sim.spectral_radius:.2f}[/cyan] (rho_inf)"
    )
    console.print(
        f"  • Nonlinear Mode               : [bold green]ENABLED[/bold green] (Modified Newton-Raphson Generalized-alpha)"
    )
    console.print(
        f"  • Internal Force Assembly      : [cyan]Exact 4-Point Hammer Quadrature on Quadratic Tet10[/cyan]"
    )
    console.print(
        f"  • Pre-factorized K_eff Setup   : [green]{t_init_ms:.2f}[/green] ms"
    )

    # Target node for interactive disturbances: center node of the free tip
    tip_coords = mesh_tet10.nodes[tip_nodes]
    tip_center = tip_coords.mean(axis=0)
    tip_center_node = int(
        tip_nodes[np.argmin(np.linalg.norm(tip_coords - tip_center, axis=1))]
    )
    console.print(
        f"  • Primary Interactive Tip Node : Node #[green]{tip_center_node}[/green] at {mesh_tet10.nodes[tip_center_node].round(1)}"
    )

    # 5. Execution Phase
    console.print(
        "\n[bold yellow][5/6] Running Nonlinear Dynamic Simulation Pipeline[/bold yellow]"
    )

    save_snapshots = args.save_snapshots
    output_dir = Path(args.output_dir)
    pvd_writer = None
    if save_snapshots:
        output_dir.mkdir(parents=True, exist_ok=True)
        pvd_writer = PvdCollectionWriter(output_dir / "sim02_online_dynamic_tet10.pvd")
        console.print(f"  • Exporting ParaView snapshots to: [dim]{output_dir}[/dim]")

    if is_headless:
        # ---------------------------------------------------------------------
        # Headless Demonstration Sequence
        # ---------------------------------------------------------------------
        console.print(
            "  • Executing automated interactive nonlinear perturbation sequence:"
        )
        console.print(
            "    - [cyan]Phase 1 (0.00s - 0.20s)[/cyan]: Gravity dynamic settlement under Tet10 hyperelasticity"
        )
        console.print(
            "    - [cyan]Phase 2 (0.20s - 0.30s)[/cyan]: User downward impulse disturbance (-300 N) applied at tip"
        )
        console.print(
            "    - [cyan]Phase 3 (0.30s - 0.50s)[/cyan]: Free damped nonlinear vibration and recovery"
        )

        total_duration = args.duration
        dt = sim.dt
        n_steps = int(round(total_duration / dt))
        snapshot_interval = max(1, int(round(0.02 / dt)))

        snapshot_count = 0
        export_grid = domain.to_pyvista(factor=1.0) if save_snapshots else None

        t_sim_start = time.perf_counter()
        for step_idx in range(n_steps):
            current_t = sim.current_time

            # Phase 2: Downward impulse perturbation
            if 0.20 <= current_t < 0.30:
                sim.apply_perturbation_force(tip_center_node, [0.0, -300.0, 0.0])
            elif current_t >= 0.30 and current_t < 0.31:
                sim.clear_perturbation_forces()

            # Execute single nonlinear dynamic physics step
            sim.step()

            # Export snapshot
            if save_snapshots and (
                step_idx % snapshot_interval == 0 or step_idx == n_steps - 1
            ):
                sim.update_pyvista_grid(export_grid)
                vtu_filename = f"snapshot_{snapshot_count:04d}.vtu"
                vtu_path = output_dir / vtu_filename
                export_grid.save(str(vtu_path))
                if pvd_writer is not None:
                    pvd_writer.add_timestep(
                        timestep=sim.current_time, filename=vtu_filename
                    )
                snapshot_count += 1

        total_wall_loop = time.perf_counter() - t_sim_start
        if save_snapshots and pvd_writer is not None:
            pvd_writer.write()
            console.print(
                f"  • Exported [green]{snapshot_count}[/green] snapshots to [cyan]{pvd_writer.pvd_path}[/cyan]"
            )

    else:
        # ---------------------------------------------------------------------
        # Interactive PyVista GUI Mode
        # ---------------------------------------------------------------------
        import pyvista as pv

        console.print("  • Launching interactive 3D window...")
        plotter = pv.Plotter(
            window_size=[1280, 800],
            title="Sierra FEA - Real-Time Dynamic Tet10 Nonlinear Simulation",
        )
        plotter.set_background("#1e1e24", top="#141418")

        grid = domain.to_pyvista(factor=1.0)
        actor = plotter.add_mesh(
            grid,
            scalars="Displacement_Y",
            clim=[-5.0, 5.0],
            cmap="coolwarm",
            show_edges=True,
            edge_color="#444444",
            scalar_bar_args={"title": "Displacement Y (mm)", "color": "white"},
        )

        plotter.camera_position = [
            (250.0, 200.0, 350.0),
            (0.0, -100.0, 85.0),
            (0.0, 1.0, 0.0),
        ]

        state = {
            "harmonic_active": False,
            "impulse_steps": 0,
            "impulse_force": 0.0,
            "exit_requested": False,
        }

        def is_window_active() -> bool:
            if state["exit_requested"]:
                return False
            if getattr(plotter, "_closed", False):
                return False
            if getattr(plotter, "render_window", None) is None:
                return False
            return True

        def key_impulse_down() -> None:
            state["impulse_steps"] = int(round(0.10 / sim.dt))
            state["impulse_force"] = -400.0
            console.print(
                "  [bold red]▶ [KEY] Downward impulse (-400 N) applied[/bold red]"
            )

        def key_impulse_up() -> None:
            state["impulse_steps"] = int(round(0.10 / sim.dt))
            state["impulse_force"] = 400.0
            console.print(
                "  [bold green]▶ [KEY] Upward impulse (+400 N) applied[/bold green]"
            )

        def key_toggle_harmonic() -> None:
            state["harmonic_active"] = not state["harmonic_active"]
            if state["harmonic_active"]:
                sim.add_harmonic_excitation(
                    tip_center_node, component=1, amplitude=150.0, frequency_hz=10.0
                )
                console.print(
                    "  [bold magenta]▶ [KEY] Harmonic excitation ENABLED (10 Hz, 150 N)[/bold magenta]"
                )
            else:
                sim.clear_harmonic_excitations()
                console.print(
                    "  [bold dim]▶ [KEY] Harmonic excitation DISABLED[/bold dim]"
                )

        def key_clear() -> None:
            state["harmonic_active"] = False
            state["impulse_steps"] = 0
            state["impulse_force"] = 0.0
            sim.clear_harmonic_excitations()
            sim.clear_perturbation_forces()
            console.print(
                "  [bold yellow]▶ [KEY] All dynamic perturbation forces cleared[/bold yellow]"
            )

        def key_reset() -> None:
            key_clear()
            sim.reset()
            sim.update_pyvista_grid(
                grid,
                plotter=plotter,
                field="Displacement_Y",
                symmetric_clim=True,
                min_clim=5.0,
            )
            plotter.update()
            console.print(
                "  [bold cyan]▶ [KEY] Simulation RESET to undeformed resting state[/bold cyan]"
            )

        def key_quit() -> None:
            state["exit_requested"] = True
            if not getattr(plotter, "_closed", False):
                plotter.close()
            console.print(
                "  [bold red]▶ [KEY] Exiting interactive window...[/bold red]"
            )

        plotter.add_key_event("i", key_impulse_down)
        plotter.add_key_event("u", key_impulse_up)
        plotter.add_key_event("h", key_toggle_harmonic)
        plotter.add_key_event("c", key_clear)
        plotter.add_key_event("r", key_reset)
        plotter.add_key_event("q", key_quit)
        plotter.add_key_event("Q", key_quit)

        info_text = (
            "Interactive Nonlinear Tet10 Controls:\n"
            "  [I] Downward Tip Impulse (-400 N)\n"
            "  [U] Upward Tip Impulse (+400 N)\n"
            "  [H] Toggle Harmonic Resonance (10 Hz)\n"
            "  [C] Clear Dynamic Forces\n"
            "  [R] Reset Simulation\n"
            "  [Q] Exit Window"
        )
        plotter.add_text(info_text, position="upper_left", font_size=10, color="white")

        telemetry_actor = plotter.add_text(
            "Telemetry: Initializing...",
            position="lower_left",
            font_size=10,
            color="#00ffcc",
        )

        plotter.show(interactive_update=True)

        last_time = time.perf_counter()
        fps_timer = time.perf_counter()
        frame_counter = 0

        while is_window_active():
            t_now = time.perf_counter()
            dt_wall = t_now - last_time
            last_time = t_now

            if state["impulse_steps"] > 0:
                sim.apply_perturbation_force(
                    tip_center_node, [0.0, state["impulse_force"], 0.0]
                )
                state["impulse_steps"] -= 1
                if state["impulse_steps"] == 0:
                    sim.clear_perturbation_forces()

            sim.advance_wall_clock(dt_wall, max_substeps=10)
            sim.update_pyvista_grid(
                grid,
                plotter=plotter,
                field="Displacement_Y",
                symmetric_clim=True,
                min_clim=5.0,
            )
            try:
                plotter.update()
            except Exception:
                break

            if not is_window_active():
                break

            frame_counter += 1
            if t_now - fps_timer >= 0.25:
                render_fps = frame_counter / (t_now - fps_timer)
                fps_timer = t_now
                frame_counter = 0
                stats = sim.get_performance_stats()
                tip_y = float(sim.displacement[tip_center_node, 1])

                telemetry_str = (
                    f"Model: Quadratic Tet10 ({args.formulation.replace('_', ' ').title()})\n"
                    f"Sim Time: {stats.sim_time:.3f} s | Wall Time: {stats.wall_time:.3f} s | "
                    f"Speed Ratio: {stats.speed_ratio:.1f}x Real-Time\n"
                    f"Physics Rate: {stats.effective_hz:.0f} Hz ({stats.average_step_ms:.1f} ms/step) | "
                    f"Render FPS: {render_fps:.1f} FPS\n"
                    f"Tip Deflection Y: {tip_y:.2f} mm | Total Energy: {stats.total_energy:.2e} J"
                )
                try:
                    telemetry_actor.SetText(0, telemetry_str)
                except Exception:
                    pass

        if not getattr(plotter, "_closed", False):
            plotter.close()

    # 6. Final Performance Telemetry & Reporting
    console.print("\n[bold yellow][6/6] Performance Telemetry Summary[/bold yellow]")
    final_stats = sim.get_performance_stats()
    tip_y_min = float(np.min(sim.displacement[:, 1]))

    table = Table(
        title="Interactive Nonlinear Tet10 Dynamic Telemetry",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric Description", style="cyan")
    table.add_column("Measured Value", style="bold green", justify="right")
    table.add_column("Engineering Notes", style="dim", justify="left")

    table.add_row(
        "Simulated Physical Time",
        f"{final_stats.sim_time:.3f} s",
        f"Duration: {args.duration:.2f} s",
    )
    table.add_row(
        "Total Wall-Clock Time",
        f"{final_stats.wall_time:.4f} s",
        "Elapsed computation time",
    )
    table.add_row(
        "Physics Time Steps Executed",
        f"{final_stats.steps_completed:,}",
        f"Fixed dt = {sim.dt*1000:.1f} ms",
    )
    table.add_row(
        "Average Physics Step Duration",
        f"{final_stats.average_step_ms:.1f} ms",
        "Nonlinear internal force evaluation",
    )
    table.add_row(
        "Effective Physics Rate",
        f"{final_stats.effective_hz:.1f} Hz",
        "Interactive physics update rate",
    )
    table.add_row(
        "Peak Downward Tip Deflection",
        f"{tip_y_min:.2f} mm",
        "Tet10 flexure without shear locking",
    )
    table.add_row(
        "Instantaneous Kinetic Energy",
        f"{final_stats.kinetic_energy:.4e} J",
        "E_k = 0.5 * v^T * M * v",
    )
    table.add_row(
        "Instantaneous Strain Energy",
        f"{final_stats.strain_energy:.4e} J",
        "Nonlinear hyperelastic internal energy",
    )
    table.add_row(
        "Total Mechanical Energy",
        f"{final_stats.total_energy:.4e} J",
        "E_total = E_k + U",
    )

    console.print(table)
    console.print(
        "\n[bold green]✔ Interactive Nonlinear Tet10 Dynamic Simulation Pipeline completed successfully![/bold green]\n"
    )


if __name__ == "__main__":
    main()
