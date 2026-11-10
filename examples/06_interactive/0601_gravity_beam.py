"""
Simulation 0601 (Interactive): Real-Time Online Dynamic Simulation Pipeline.

Demonstrates high-performance interactive dynamic FEA simulation using
Generalized-alpha implicit time integration with pre-factorized SuperLU reuse.

Features demonstrated:
1. Real-time dynamic loop operating with fixed time step dt decoupled from rendering frame rate.
2. High-performance sparse factorization reuse achieving >10,000 Hz physics rates (>20x real-time).
3. Controllable numerical high-frequency damping parameterized by spectral radius rho_inf.
4. Interactive user disturbances: point impulse perturbations and sinusoidal harmonic resonance excitation.
5. In-place PyVista VTK vertex coordinate buffer mutation for zero-overhead live 3D visualization.
6. Execution telemetry tracking simulation speed ratio, step ms, and kinetic/strain energies.
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

from psierra.material import IsotropicElasticParameters, Hookean, NeoHookean
from psierra.element import (
    Tet4Linear,
    Tet10Linear,
    Tet4Corotational,
    Tet4TotalLagrangian,
    Tet4UpdatedLagrangian,
    Tet10TotalLagrangian,
    Tet10UpdatedLagrangian,
)
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary.selector import select_nodes_by_plane
from psierra.boundary import DirichletConstraint, GravityLoad
from psierra.damping import RayleighDamping
from psierra.analysis import InteractiveDynamicSimulation, SimulationStats
from psierra.exchange import SimulationWriter

install(show_locals=False)
console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 0601 (Interactive): Real-Time Online Dynamic Simulation Pipeline (Cantilever Beam Demonstration)"
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
        default=2.0,
        help="Physical simulation duration in seconds for demonstration sequence (default: 2.0 s)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="Physics simulation time step dt in seconds (default: 0.002 s = 500 Hz)",
    )
    parser.add_argument(
        "--spectral-radius",
        type=float,
        default=0.8,
        help="Generalized-alpha high-frequency spectral radius rho_inf in [0.0, 1.0] (default: 0.8)",
    )
    parser.add_argument(
        "--element",
        type=str,
        choices=["tet4", "tet10"],
        default="tet4",
        help="Element order: 'tet4' (linear, ultra-fast) or 'tet10' (quadratic, shear-locking free) (default: tet4)",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        choices=["linear", "corotational", "total_lagrangian", "updated_lagrangian"],
        default=None,
        help="Kinematic formulation: 'linear', 'corotational', 'total_lagrangian', or 'updated_lagrangian' (default: linear)",
    )
    parser.add_argument(
        "--nonlinear",
        action="store_true",
        default=False,
        help="Enable nonlinear dynamic time integration (internal force evaluation)",
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
        default="results/06_interactive/0601_gravity_beam",
        help="Output directory path for saved snapshots (default: results/06_interactive/0601_gravity_beam)",
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
            "[bold cyan]Simulation 0601 (Interactive): Real-Time Online Dynamic Simulation Pipeline[/bold cyan]\n"
            "[dim]Decoupled Generalized-alpha Dynamic Physics Engine with Live In-Place 3D Rendering[/dim]\n"
            f"[dim]Mode: {'[bold red]Headless Verification Mode[/bold red]' if is_headless else '[bold green]Interactive PyVista GUI Mode[/bold green]'}[/dim]",
            title="[bold blue]project-sierra[/bold blue]",
            border_style="cyan",
        )
    )

    # 1. Loading Mesh Geometry
    console.print("\n[bold yellow][1/6] Loading Mesh Geometry[/bold yellow]")
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Beam mesh file not found: {mesh_path}")

    mesh = VolumetricMesh(mesh_path)
    console.print(
        f"  • Volumetric Mesh : [green]{mesh.n_nodes}[/green] nodes | "
        f"[green]{mesh.n_elements}[/green] Tet4 elements | "
        f"Volume: [green]{mesh.volume:.1f}[/green] mm³"
    )
    console.print(
        "  • Beam Dimensions : X: [-60, 60] mm, Y: [-60, 60] mm, Z: [-30, 200] mm (Length L = 230 mm)"
    )

    # Determine kinematic formulation and nonlinearity
    formulation = args.formulation
    if formulation is None:
        formulation = "total_lagrangian" if args.nonlinear else "linear"

    is_nonlinear = formulation in (
        "corotational",
        "total_lagrangian",
        "updated_lagrangian",
    )

    # Upgrade to Quadratic Tet10 if requested
    if args.element == "tet10":
        if formulation == "corotational":
            console.print(
                "  [yellow]Note: Corotational formulation is available for Tet4. Reverting to Tet4.[/yellow]"
            )
            args.element = "tet4"
        else:
            mesh = mesh.to_quadratic()
            console.print(
                f"  • Upgraded to Quadratic Tet10: [green]{mesh.n_nodes}[/green] nodes | "
                f"[green]{mesh.n_elements}[/green] elements | [green]{mesh.n_dofs}[/green] DOFs"
            )

    # 2. Material & Physical Continuum Configuration
    console.print(
        "\n[bold yellow][2/6] Configuring Material & Physical Domain[/bold yellow]"
    )
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000.0,  # MPa (N/mm^2)
        poissons_ratio=0.35,
        mass_density=1100e-6,  # tonne/mm^3 (1100 kg/m^3)
    )

    if formulation == "corotational":
        material = Hookean(params)
        element = Tet4Corotational(material)
        mode_desc = "Corotational Dynamic (Rigid-Frame Filtered Generalized-alpha)"
    elif formulation == "total_lagrangian":
        material = NeoHookean(params)
        element = (
            Tet10TotalLagrangian(material)
            if args.element == "tet10"
            else Tet4TotalLagrangian(material)
        )
        mode_desc = f"Nonlinear Hyperelastic (TL {element.__class__.__name__})"
    elif formulation == "updated_lagrangian":
        material = NeoHookean(params)
        element = (
            Tet10UpdatedLagrangian(material)
            if args.element == "tet10"
            else Tet4UpdatedLagrangian(material)
        )
        mode_desc = f"Nonlinear Hyperelastic (UL {element.__class__.__name__})"
    else:  # linear
        material = Hookean(params)
        element = (
            Tet10Linear(material) if args.element == "tet10" else Tet4Linear(material)
        )
        mode_desc = f"Linear Dynamic ({element.__class__.__name__} Pre-factorized)"

    domain = SolidDomain(mesh, material, element)

    console.print(f"  • Constitutive Law : [cyan]{material.__class__.__name__}[/cyan]")
    console.print(f"  • Element Type     : [cyan]{element.__class__.__name__}[/cyan]")
    console.print(f"  • Dynamic Engine   : [cyan]{mode_desc}[/cyan]")
    console.print(
        f"  • Elastic Modulus  : [green]{params.youngs_modulus:.1f}[/green] MPa | "
        f"Poisson's Ratio: [green]{params.poissons_ratio:.2f}[/green] | "
        f"Density: [green]{params.mass_density * 1e9:.1f}[/green] kg/m³"
    )

    # 3. Boundary Conditions & External Base Gravity
    console.print(
        "\n[bold yellow][3/6] Setting Up Boundary Conditions & Gravity Load[/bold yellow]"
    )
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
    tip_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)

    # Base gravity body load along -Y (calibrated to ~100 kN body weight matching sim02 and sim18)
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

    # 4. Interactive Simulation Engine Initialization
    console.print(
        "\n[bold yellow][4/6] Initializing Real-Time Interactive Dynamic Engine[/bold yellow]"
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
        nonlinear=is_nonlinear,
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
        f"  • Algorithmic Parameters       : alpha_m = {sim.alpha_m:.4f}, alpha_f = {sim.alpha_f:.4f}"
    )
    console.print(
        f"  • Rayleigh Physical Damping    : alpha = 1.0 (mass), beta = 0.001 (stiffness)"
    )
    console.print(
        f"  • Pre-factorized K_eff Setup   : [green]{t_init_ms:.2f}[/green] ms (One-time SuperLU factorization)"
    )

    # Target node for interactive disturbances: center node of the free tip
    tip_coords = mesh.nodes[tip_nodes]
    tip_center = tip_coords.mean(axis=0)
    tip_center_node = int(
        tip_nodes[np.argmin(np.linalg.norm(tip_coords - tip_center, axis=1))]
    )
    console.print(
        f"  • Primary Interactive Tip Node : Node #[green]{tip_center_node}[/green] at {mesh.nodes[tip_center_node].round(1)}"
    )

    # 5. Execution Phase
    console.print(
        "\n[bold yellow][5/6] Running Dynamic Simulation Pipeline[/bold yellow]"
    )

    # Snapshot management
    save_snapshots = args.save_snapshots
    output_dir = Path(args.output_dir)
    writer = None
    if save_snapshots:
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SimulationWriter(output_dir / "result.pvd", mesh=domain.mesh, format="pvd")
        console.print(f"  • Exporting ParaView snapshots to: [dim]{output_dir}[/dim]")

    if is_headless:
        # ---------------------------------------------------------------------
        # Headless Demonstration Sequence
        # ---------------------------------------------------------------------
        console.print("  • Executing automated interactive perturbation sequence:")
        console.print(
            "    - [cyan]Phase 1 (0.00s - 0.40s)[/cyan]: Gravity dynamic settlement under Generalized-alpha dissipation"
        )
        console.print(
            "    - [cyan]Phase 2 (0.40s - 0.50s)[/cyan]: User downward impulse disturbance (-300 N) applied at tip"
        )
        console.print(
            "    - [cyan]Phase 3 (0.50s - 1.00s)[/cyan]: Free damped vibration and recovery"
        )
        console.print(
            "    - [cyan]Phase 4 (1.00s - 1.60s)[/cyan]: Dynamic sinusoidal resonance excitation (F = 150*sin(2*pi*15*t) N)"
        )
        console.print(
            "    - [cyan]Phase 5 (1.60s - 2.00s)[/cyan]: Release and numerical damping settling back to equilibrium"
        )

        total_duration = args.duration
        dt = sim.dt
        n_steps = int(round(total_duration / dt))
        snapshot_interval = max(
            1, int(round(0.02 / dt))
        )  # Snapshot every 20 ms (50 FPS)

        tip_y_history: list[float] = []
        time_history: list[float] = []
        snapshot_count = 0

        # Create initial PyVista grid for snapshot export if requested
        export_grid = domain.to_pyvista(factor=1.0) if save_snapshots else None

        t_sim_start = time.perf_counter()
        for step_idx in range(n_steps):
            current_t = sim.current_time

            # Phase 2: Downward impulse perturbation
            if 0.40 <= current_t < 0.50:
                sim.apply_perturbation_force(tip_center_node, [0.0, -300.0, 0.0])
            elif current_t >= 0.50 and 0.50 <= current_t < 0.51:
                sim.clear_perturbation_forces()

            # Phase 4: Dynamic sinusoidal harmonic excitation
            if 1.00 <= current_t < 1.60:
                # 15 Hz driving frequency
                f_harm = 150.0 * np.sin(2.0 * np.pi * 15.0 * (current_t - 1.00))
                sim.clear_perturbation_forces()
                sim.apply_perturbation_force(tip_center_node, [0.0, f_harm, 0.0])
            elif current_t >= 1.60 and current_t < 1.61:
                sim.clear_perturbation_forces()

            # Execute single O(N) back-substitution physics step
            sim.step()

            # Record metrics
            tip_disp_y = float(sim.displacement[tip_center_node, 1])
            tip_y_history.append(tip_disp_y)
            time_history.append(sim.current_time)

            # Export snapshot
            if save_snapshots and (
                step_idx % snapshot_interval == 0 or step_idx == n_steps - 1
            ):
                if writer is not None:
                    writer.add_timestep(
                        time=sim.current_time,
                        domain=domain,
                        step_index=snapshot_count,
                    )
                snapshot_count += 1

        total_wall_loop = time.perf_counter() - t_sim_start
        if save_snapshots and writer is not None:
            writer.close()
            console.print(
                f"  • Exported [green]{snapshot_count}[/green] snapshots to [cyan]{output_dir}[/cyan]"
            )

    else:
        # ---------------------------------------------------------------------
        # Interactive PyVista GUI Mode
        # ---------------------------------------------------------------------
        import pyvista as pv

        console.print("  • Launching interactive 3D window...")
        plotter = pv.Plotter(
            window_size=[1280, 800], title="Sierra FEA - Real-Time Dynamic Simulation"
        )
        plotter.set_background("#1e1e24", top="#141418")

        # PyVista mesh setup
        grid = domain.to_pyvista(factor=1.0)
        actor = plotter.add_mesh(
            grid,
            scalars="Displacement_Y",
            clim=[-15.0, 15.0],
            cmap="coolwarm",
            show_edges=True,
            edge_color="#444444",
            scalar_bar_args={"title": "Displacement Y (mm)", "color": "white"},
        )

        # Camera and lighting
        plotter.camera_position = [
            (250.0, 200.0, 350.0),
            (0.0, -100.0, 85.0),
            (0.0, 1.0, 0.0),
        ]

        # Keybindings and interactive disturbance state
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
            state["impulse_steps"] = int(round(0.10 / sim.dt))  # 100 ms pulse
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
                    tip_center_node, component=1, amplitude=150.0, frequency_hz=15.0
                )
                console.print(
                    "  [bold magenta]▶ [KEY] Harmonic excitation ENABLED (15 Hz, 150 N)[/bold magenta]"
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
                min_clim=15.0,
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

        # On-screen instructions
        info_text = (
            "Interactive Controls:\n"
            "  [I] Downward Tip Impulse (-400 N)\n"
            "  [U] Upward Tip Impulse (+400 N)\n"
            "  [H] Toggle Harmonic Resonance (15 Hz)\n"
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

            # Handle active impulse pulse
            if state["impulse_steps"] > 0:
                sim.apply_perturbation_force(
                    tip_center_node, [0.0, state["impulse_force"], 0.0]
                )
                state["impulse_steps"] -= 1
                if state["impulse_steps"] == 0:
                    sim.clear_perturbation_forces()

            # Advance physics synchronized to elapsed real wall-clock time
            sim.advance_wall_clock(dt_wall, max_substeps=20)

            # In-place VTK vertex buffer mutation (zero-overhead)
            sim.update_pyvista_grid(
                grid,
                plotter=plotter,
                field="Displacement_Y",
                symmetric_clim=True,
                min_clim=15.0,
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
                    f"Model: {element.__class__.__name__} ({formulation.replace('_', ' ').title()})\n"
                    f"Sim Time: {stats.sim_time:.3f} s | Wall Time: {stats.wall_time:.3f} s | "
                    f"Speed Ratio: {stats.speed_ratio:.1f}x Real-Time\n"
                    f"Physics Rate: {stats.effective_hz:.0f} Hz ({stats.average_step_ms:.3f} ms/step) | "
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
    tip_y_final = float(sim.displacement[tip_center_node, 1])
    tip_y_min = float(np.min(sim.displacement[:, 1]))

    table = Table(
        title="Simulation 0601 (Interactive): Telemetry Summary",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric Description", style="cyan")
    table.add_column("Measured Value", style="bold green", justify="right")
    table.add_column("Engineering Benchmark / Target", style="dim", justify="left")

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
        f"{final_stats.average_step_ms * 1000.0:.1f} µs ({final_stats.average_step_ms:.4f} ms)",
        "Target: < 1.0 ms (>1,000 Hz)",
    )
    table.add_row(
        "Effective Physics Rate",
        f"{final_stats.effective_hz:,.0f} Hz",
        "Target: > 1,000 Hz (Haptic standard)",
    )
    table.add_row(
        "Real-Time Speed Ratio",
        f"{final_stats.speed_ratio:.1f}x Real-Time",
        "Target: > 1.0x (Real-time speedup)",
    )
    table.add_row(
        "Peak Downward Tip Deflection",
        f"{tip_y_min:.2f} mm",
        "Under gravity + dynamic perturbations",
    )
    table.add_row(
        "Instantaneous Kinetic Energy",
        f"{final_stats.kinetic_energy:.4e} J",
        "E_k = 0.5 * v^T * M * v",
    )
    table.add_row(
        "Instantaneous Strain Energy",
        f"{final_stats.strain_energy:.4e} J",
        "U = 0.5 * u^T * K * u",
    )
    table.add_row(
        "Total Mechanical Energy",
        f"{final_stats.total_energy:.4e} J",
        "E_total = E_k + U",
    )

    console.print(table)
    console.print(
        "\n[bold green]✔ Interactive Dynamic Simulation Pipeline completed successfully![/bold green]\n"
    )


if __name__ == "__main__":
    main()
