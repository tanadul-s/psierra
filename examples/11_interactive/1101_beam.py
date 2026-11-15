"""
Example 1101: Real-Time Interactive Dynamic Cantilever Beam Simulation.
========================================================================
Demonstrates high-performance interactive 3D dynamic continuum FEA simulation using
Generalized-alpha implicit time integration with pre-factorized SuperLU reuse:
1. Multi-Modal Disturbance Modes:
   - Bending Mode (--mode bending): Dynamic gravity settlement, transverse impulse perturbations, and harmonic excitation.
   - Torsion Mode (--mode torsion): Exact pure torsional force couple producing dynamic twisting moment M_z and coolwarm twist angle fields.
2. Ultra-High Physics Rates:
   - Factorization reuse achieving > 10,000 Hz physics rates (> 20x real-time).
   - In-place vertex coordinate mutation for zero-overhead live 3D PyVista visualization.
3. Dual-Mode Execution:
   - Automated non-GUI headless benchmark (--headless) streaming time-series frames to XDMF.
   - Full interactive 3D GUI with keyboard controls ([Up]/[Down]/[Left]/[Right] for bending, [T]/[Y] for torsion).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Sequence
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


def _resolve_mesh_path(rel_path: str) -> Path:
    p = Path(rel_path)
    if p.exists():
        return p
    alt = Path("psierra") / rel_path
    if alt.exists():
        return alt
    return p


def compute_twist_angles(
    nodes_ref: np.ndarray, disp: np.ndarray, center_xy: Sequence[float] = (0.0, 0.0)
) -> np.ndarray:
    Xc = nodes_ref[:, 0] - center_xy[0]
    Yc = nodes_ref[:, 1] - center_xy[1]
    r = np.hypot(Xc, Yc)
    valid = r > 1e-2

    x_def = Xc + disp[:, 0]
    y_def = Yc + disp[:, 1]

    theta_rad = np.zeros(len(nodes_ref), dtype=np.float64)
    theta_rad[valid] = np.arctan2(y_def[valid], x_def[valid]) - np.arctan2(
        Yc[valid], Xc[valid]
    )
    theta_rad = (theta_rad + np.pi) % (2.0 * np.pi) - np.pi
    if not np.all(valid):
        mean_valid = np.mean(theta_rad[valid]) if np.any(valid) else 0.0
        theta_rad[~valid] = mean_valid
    return np.rad2deg(theta_rad)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 1101: Real-Time Interactive Dynamic Cantilever Beam Simulation"
    )
    parser.add_argument(
        "--mode",
        choices=["bending", "torsion"],
        default="bending",
        help="Interactive disturbance mode: 'bending' (default) or 'torsion'",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Execute both bending and torsion validation sequences sequentially",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless batch demonstration mode without opening a GUI window",
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
        help="Element order: 'tet4' (linear) or 'tet10' (quadratic) (default: tet4)",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        choices=["linear", "corotational", "total_lagrangian", "updated_lagrangian"],
        default=None,
        help="Kinematic formulation: 'linear', 'corotational', 'total_lagrangian', or 'updated_lagrangian'",
    )
    parser.add_argument(
        "--nonlinear",
        action="store_true",
        default=False,
        help="Enable nonlinear dynamic time integration",
    )
    parser.add_argument(
        "--torque",
        type=float,
        default=1.5e7,
        help="Baseline twisting torque M_z in N*mm (default: 1.5e7 N*mm = 15.0 kN*m)",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="data/beam1/mesh/beam1.vtk",
        help="Path to volumetric mesh (default: data/beam1/mesh/beam1.vtk)",
    )
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        default=False,
        help="Export time-series XDMF snapshots for ParaView post-processing inspection",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/11_interactive/1101_beam",
        help="Output directory path (default: results/11_interactive/1101_beam)",
    )
    return parser.parse_args()


def run_bending_simulation(args: argparse.Namespace, is_headless: bool, out_dir: Path) -> dict[str, float]:
    console.print(
        Panel.fit(
            "[bold cyan]Bending Mode: Dynamic Gravity Settlement & Impulse/Harmonic Perturbation[/bold cyan]",
            border_style="cyan",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = _resolve_mesh_path(args.mesh)
    mesh = VolumetricMesh(mesh_path)
    if args.element == "tet10":
        mesh = mesh.to_quadratic()

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1100e-6)
    formulation = args.formulation or ("total_lagrangian" if args.nonlinear else "linear")
    is_nonlinear = formulation in ("corotational", "total_lagrangian", "updated_lagrangian")

    if formulation == "linear":
        material = Hookean(params)
        element = Tet10Linear(material) if args.element == "tet10" else Tet4Linear(material)
    elif formulation == "corotational":
        material = Hookean(params)
        element = Tet4Corotational(material)
    elif formulation == "total_lagrangian":
        material = NeoHookean(params)
        element = Tet10TotalLagrangian(material) if args.element == "tet10" else Tet4TotalLagrangian(material)
    elif formulation == "updated_lagrangian":
        material = NeoHookean(params)
        element = Tet10UpdatedLagrangian(material) if args.element == "tet10" else Tet4UpdatedLagrangian(material)
    else:
        material = Hookean(params)
        element = Tet4Linear(material)

    domain = SolidDomain(mesh=mesh, material=material, element=element)
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    gravity = GravityLoad(acceleration=(0.0, -9.81e3, 0.0))
    M = domain.assemble_mass_matrix()
    f_grav = gravity.compute_force_vector(mass_matrix=M)

    damping = RayleighDamping(alpha=1.0, beta=0.002)
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

    tip_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
    tip_coords = mesh.nodes[tip_nodes]
    tip_center = tip_coords.mean(axis=0)
    tip_center_node = int(tip_nodes[np.argmin(np.linalg.norm(tip_coords - tip_center, axis=1))])

    if is_headless:
        n_steps = int(round(args.duration / args.dt))
        writer = None
        if args.save_snapshots:
            writer = SimulationWriter(out_dir / "result.xdmf", mesh=domain.mesh, format="xdmf")

        tip_y_history: list[float] = []
        t0 = time.perf_counter()
        for step in range(n_steps):
            current_t = sim.current_time
            if 0.40 <= current_t < 0.50:
                sim.apply_perturbation_force(tip_center_node, [0.0, -3.0e5, 0.0])
            elif 0.50 <= current_t < 0.51:
                sim.clear_perturbation_forces()
            if 1.00 <= current_t < 1.60:
                f_harm = 1.5e5 * np.sin(2.0 * np.pi * 15.0 * (current_t - 1.00))
                sim.clear_perturbation_forces()
                sim.apply_perturbation_force(tip_center_node, [0.0, f_harm, 0.0])
            elif 1.60 <= current_t < 1.61:
                sim.clear_perturbation_forces()

            sim.step()
            tip_disp_y = float(sim.displacement[tip_center_node, 1])
            tip_y_history.append(tip_disp_y)
            if writer is not None and (step % 10 == 0 or step == n_steps - 1):
                writer.add_timestep(time=current_t, displacement=sim.displacement, step_index=step)

        t_sim = time.perf_counter() - t0
        rate_hz = n_steps / t_sim
        table = Table(title="Bending Simulation Performance Summary", border_style="cyan")
        table.add_column("Metric", style="bold white")
        table.add_column("Value", style="cyan")
        table.add_row("Total Steps", f"{n_steps}")
        table.add_row("Wall-Clock Time", f"{t_sim:.3f} s")
        table.add_row("Physics Rate", f"[bold green]{rate_hz:,.0f} Hz[/bold green]")
        table.add_row("Real-Time Speedup", f"{rate_hz * args.dt:.1f}x real-time")
        table.add_row("Peak Tip Sag", f"{min(tip_y_history):.2f} mm")
        console.print(table)
        console.print("\n[bold green]✔ Interactive Dynamic Simulation Pipeline completed successfully![/bold green]\n")
        return {"rate_hz": rate_hz, "speedup": rate_hz * args.dt}
    else:
        console.print("[green]Opening interactive PyVista bending window...[/green]")
        import pyvista as pv
        plotter = pv.Plotter(title="Example 1101: Interactive Dynamic Beam Bending")
        grid = domain.to_pyvista(factor=1.0)
        grid.point_data["Displacement_Y"] = np.zeros(mesh.n_nodes, dtype=np.float32)
        actor = plotter.add_mesh(
            grid,
            scalars="Displacement_Y",
            cmap="viridis",
            show_edges=True,
            clim=[-40.0, 5.0],
            scalar_bar_args={"title": "Tip Deflection Y (mm)", "vertical": True},
        )

        initial_points = mesh.nodes.copy()
        f_manual_mag = 3.0e5  # 0.3 N = 300,000 g*mm/s^2, causes ~28 mm tip deflection
        gui_state = {
            "exit_requested": False,
            "harmonic_active": False,
            "current_force_val": 0.0,
        }

        hud_text = (
            "Interactive Beam Bending Controls:\n"
            "  [I / Up]   Push Downward Force (-0.3 N)\n"
            "  [K / Down] Pull Upward Force (+0.3 N)\n"
            "  [Space]    Toggle Harmonic Excitation (10 Hz)\n"
            "  [C]        Clear Applied Perturbations\n"
            "  [R]        Reset Dynamic Beam State\n"
            "  [Q / Esc]  Exit Interactive Simulation"
        )
        plotter.add_text(
            hud_text,
            position="lower_left",
            font_size=9,
            color="#cfd8dc",
            font="courier",
        )

        telemetry_actor = plotter.add_text(
            "Initializing...",
            position="upper_left",
            font_size=10,
            color="#00e676",
            font="courier",
        )

        def key_push_down() -> None:
            gui_state["current_force_val"] = -f_manual_mag * 1e-6
            sim.apply_perturbation_force(tip_center_node, [0.0, -f_manual_mag, 0.0])
            console.print("[red]▶ [KEY] Downward tip force applied (-0.3 N)[/red]")

        def key_pull_up() -> None:
            gui_state["current_force_val"] = f_manual_mag * 1e-6
            sim.apply_perturbation_force(tip_center_node, [0.0, f_manual_mag, 0.0])
            console.print("[green]▶ [KEY] Upward tip force applied (+0.3 N)[/green]")

        def key_toggle_harmonic() -> None:
            gui_state["harmonic_active"] = not gui_state["harmonic_active"]
            if gui_state["harmonic_active"]:
                sim.add_harmonic_excitation(
                    node_idx=tip_center_node,
                    amplitude=2.0e5,
                    frequency_hz=10.0,
                    direction=(0.0, 1.0, 0.0),
                )
                console.print("[yellow]▶ [KEY] Harmonic excitation ON (10 Hz, 0.2 N)[/yellow]")
            else:
                sim.clear_harmonic_excitations()
                console.print("[dim]▶ [KEY] Harmonic excitation OFF[/dim]")

        def key_clear() -> None:
            gui_state["harmonic_active"] = False
            gui_state["current_force_val"] = 0.0
            sim.clear_perturbation_forces()
            sim.clear_harmonic_excitations()
            console.print("[yellow]▶ [KEY] All perturbations cleared[/yellow]")

        def key_reset() -> None:
            key_clear()
            sim.reset()
            grid.points = initial_points
            grid.point_data["Displacement_Y"] = np.zeros(mesh.n_nodes, dtype=np.float32)
            plotter.update()
            console.print("[cyan]▶ [KEY] Dynamic simulation state reset[/cyan]")

        def key_quit() -> None:
            gui_state["exit_requested"] = True
            if not getattr(plotter, "_closed", False):
                plotter.close()

        # Non-conflicting keys (avoiding VTK defaults)
        for k in ["i", "I", "Up"]:
            plotter.add_key_event(k, key_push_down)
        for k in ["k", "K", "Down"]:
            plotter.add_key_event(k, key_pull_up)
        for k in ["space", "Space"]:
            plotter.add_key_event(k, key_toggle_harmonic)
        for k in ["c", "C"]:
            plotter.add_key_event(k, key_clear)
        for k in ["r", "R"]:
            plotter.add_key_event(k, key_reset)
        for k in ["q", "Q", "Escape", "x", "X"]:
            plotter.add_key_event(k, key_quit)

        # Neutralize default VTK keys
        if hasattr(plotter, "iren") and plotter.iren is not None:
            plotter.iren.interactor.RemoveObservers("CharEvent")
            if plotter.iren.get_interactor_style() is not None:
                plotter.iren.get_interactor_style().RemoveObservers("CharEvent")

        plotter.show(interactive_update=True)

        fps_timer = time.perf_counter()
        fps_count = 0
        current_fps = 0.0

        try:
            while not getattr(plotter, "_closed", False) and plotter.render_window is not None:
                if gui_state["exit_requested"]:
                    break

                sim.step()

                disp = sim.displacement
                grid.points = initial_points + disp
                grid.point_data["Displacement_Y"] = disp[:, 1].astype(np.float32)

                fps_count += 1
                t_now = time.perf_counter()
                if t_now - fps_timer >= 0.5:
                    current_fps = fps_count / (t_now - fps_timer)
                    fps_count = 0
                    fps_timer = t_now

                tip_y = float(disp[tip_center_node, 1])
                telemetry = (
                    f"Time: {sim.current_time:.3f} s | Rate: {1.0/args.dt:.0f} Hz (dt={args.dt*1e3:.1f}ms)\n"
                    f"Tip Deflection Y: {tip_y:+.2f} mm | FPS: {current_fps:.1f}\n"
                    f"Harmonic: {'ACTIVE (10 Hz)' if gui_state['harmonic_active'] else 'OFF'} | "
                    f"Manual Force: {gui_state['current_force_val']:+.2f} N"
                )
                telemetry_actor.SetText(2, telemetry)

                plotter.update()
        except Exception:
            pass

        return {}


def run_torsion_simulation(args: argparse.Namespace, is_headless: bool, out_dir: Path) -> dict[str, float]:
    console.print(
        Panel.fit(
            "[bold cyan]Torsion Mode: Dynamic Twisting Moment M_z & Spring-back Vibration[/bold cyan]",
            border_style="cyan",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = _resolve_mesh_path(args.mesh)
    mesh = VolumetricMesh(mesh_path)
    if args.element == "tet10":
        mesh = mesh.to_quadratic()

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1100e-6)
    formulation = args.formulation or ("total_lagrangian" if args.nonlinear else "linear")
    is_nonlinear = formulation in ("corotational", "total_lagrangian", "updated_lagrangian")

    if formulation == "linear":
        material = Hookean(params)
        element = Tet10Linear(material) if args.element == "tet10" else Tet4Linear(material)
    elif formulation == "corotational":
        material = Hookean(params)
        element = Tet4Corotational(material)
    elif formulation == "total_lagrangian":
        material = NeoHookean(params)
        element = Tet10TotalLagrangian(material) if args.element == "tet10" else Tet4TotalLagrangian(material)
    else:
        material = Hookean(params)
        element = Tet4Linear(material)

    domain = SolidDomain(mesh=mesh, material=material, element=element)
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)

    tip_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
    tip_coords = mesh.nodes[tip_nodes]
    tip_Xc = tip_coords[:, 0]
    tip_Yc = tip_coords[:, 1]
    J_tip = float(np.sum(tip_Xc**2 + tip_Yc**2))
    unit_torque_forces = np.zeros((len(tip_nodes), 3), dtype=np.float64)
    unit_torque_forces[:, 0] = -tip_Yc / J_tip
    unit_torque_forces[:, 1] = tip_Xc / J_tip

    damping = RayleighDamping(alpha=1.0, beta=0.002)
    sim = InteractiveDynamicSimulation(
        domain=domain,
        damping=damping,
        dirichlet=[clamp],
        dt=args.dt,
        spectral_radius=args.spectral_radius,
        nonlinear=is_nonlinear,
        modified_newton=True,
    )

    def set_active_torque(moment_z: float) -> None:
        sim.clear_perturbation_forces()
        if abs(moment_z) > 1e-6:
            sim.apply_perturbation_forces(tip_nodes, unit_torque_forces * moment_z)

    if is_headless:
        n_steps = int(round(args.duration / args.dt))
        writer = None
        if args.save_snapshots:
            writer = SimulationWriter(out_dir / "result.xdmf", mesh=domain.mesh, format="xdmf")

        tip_twist_history: list[float] = []
        t0 = time.perf_counter()
        for step in range(n_steps):
            current_t = sim.current_time
            if 0.30 <= current_t < 0.50:
                set_active_torque(args.torque)
            elif 0.50 <= current_t < 1.00:
                set_active_torque(0.0)
            elif 1.00 <= current_t < 1.60:
                m_harm = args.torque * np.sin(2.0 * np.pi * 20.0 * (current_t - 1.00))
                set_active_torque(m_harm)
            elif 1.60 <= current_t:
                set_active_torque(0.0)

            sim.step()
            thetas = compute_twist_angles(mesh.nodes[tip_nodes], sim.displacement[tip_nodes])
            tip_twist_history.append(float(np.mean(thetas)))
            if writer is not None and (step % 10 == 0 or step == n_steps - 1):
                writer.add_timestep(time=current_t, displacement=sim.displacement, step_index=step)

        t_sim = time.perf_counter() - t0
        rate_hz = n_steps / t_sim
        table = Table(title="Torsion Simulation Performance Summary", border_style="cyan")
        table.add_column("Metric", style="bold white")
        table.add_column("Value", style="cyan")
        table.add_row("Total Steps", f"{n_steps}")
        table.add_row("Wall-Clock Time", f"{t_sim:.3f} s")
        table.add_row("Physics Rate", f"[bold green]{rate_hz:,.0f} Hz[/bold green]")
        table.add_row("Real-Time Speedup", f"{rate_hz * args.dt:.1f}x real-time")
        table.add_row("Max Tip Twist", f"{max(tip_twist_history):.2f}°")
        console.print(table)
        console.print("\n[bold green]✔ Interactive Dynamic Axial Torsion Pipeline completed successfully![/bold green]\n")
        return {"rate_hz": rate_hz, "speedup": rate_hz * args.dt}
    else:
        console.print("[green]Opening interactive PyVista torsion window...[/green]")
        import pyvista as pv
        plotter = pv.Plotter(title="Example 1101: Interactive Dynamic Beam Torsion")
        grid = domain.to_pyvista(factor=1.0)
        grid.point_data["Twist_deg"] = np.zeros(mesh.n_nodes, dtype=np.float32)
        plotter.add_mesh(
            grid,
            scalars="Twist_deg",
            cmap="coolwarm",
            show_edges=True,
            clim=[-30.0, 30.0],
            scalar_bar_args={"title": "Twist Angle (deg)", "vertical": True},
        )

        initial_points = mesh.nodes.copy()
        tau_manual_mag = args.torque if abs(args.torque) > 1e-3 else 5.0e6  # 5 N*mm
        gui_state = {
            "exit_requested": False,
            "harmonic_active": False,
            "current_torque_val": 0.0,
        }

        hud_text = (
            "Interactive Beam Torsion Controls:\n"
            "  [T / Left]   Apply Counter-Clockwise Torque (+M_z)\n"
            "  [Y / Right]  Apply Clockwise Torque (-M_z)\n"
            "  [Space]      Toggle Harmonic Torsion (15 Hz)\n"
            "  [C]          Clear Applied Perturbations\n"
            "  [R]          Reset Dynamic Beam State\n"
            "  [Q / Esc]    Exit Interactive Simulation"
        )
        plotter.add_text(
            hud_text,
            position="lower_left",
            font_size=9,
            color="#cfd8dc",
            font="courier",
        )

        telemetry_actor = plotter.add_text(
            "Initializing...",
            position="upper_left",
            font_size=10,
            color="#00e676",
            font="courier",
        )

        def key_twist_ccw() -> None:
            gui_state["current_torque_val"] = tau_manual_mag
            set_active_torque(tau_manual_mag)
            console.print("[cyan]▶ [KEY] Counter-Clockwise torque applied (+M_z)[/cyan]")

        def key_twist_cw() -> None:
            gui_state["current_torque_val"] = -tau_manual_mag
            set_active_torque(-tau_manual_mag)
            console.print("[magenta]▶ [KEY] Clockwise torque applied (-M_z)[/magenta]")

        def key_toggle_harmonic() -> None:
            gui_state["harmonic_active"] = not gui_state["harmonic_active"]
            if gui_state["harmonic_active"]:
                console.print("[yellow]▶ [KEY] Harmonic torsion ON (15 Hz)[/yellow]")
            else:
                set_active_torque(0.0)
                gui_state["current_torque_val"] = 0.0
                console.print("[dim]▶ [KEY] Harmonic torsion OFF[/dim]")

        def key_clear() -> None:
            gui_state["harmonic_active"] = False
            gui_state["current_torque_val"] = 0.0
            set_active_torque(0.0)
            console.print("[yellow]▶ [KEY] All torques cleared[/yellow]")

        def key_reset() -> None:
            key_clear()
            sim.reset()
            grid.points = initial_points
            grid.point_data["Twist_deg"] = np.zeros(mesh.n_nodes, dtype=np.float32)
            plotter.update()
            console.print("[cyan]▶ [KEY] Dynamic simulation state reset[/cyan]")

        def key_quit() -> None:
            gui_state["exit_requested"] = True
            if not getattr(plotter, "_closed", False):
                plotter.close()

        # Non-conflicting keys (avoiding VTK defaults)
        for k in ["t", "T", "Left"]:
            plotter.add_key_event(k, key_twist_ccw)
        for k in ["y", "Y", "Right"]:
            plotter.add_key_event(k, key_twist_cw)
        for k in ["space", "Space"]:
            plotter.add_key_event(k, key_toggle_harmonic)
        for k in ["c", "C"]:
            plotter.add_key_event(k, key_clear)
        for k in ["r", "R"]:
            plotter.add_key_event(k, key_reset)
        for k in ["q", "Q", "Escape", "x", "X"]:
            plotter.add_key_event(k, key_quit)

        # Neutralize default VTK keys
        if hasattr(plotter, "iren") and plotter.iren is not None:
            plotter.iren.interactor.RemoveObservers("CharEvent")
            if plotter.iren.get_interactor_style() is not None:
                plotter.iren.get_interactor_style().RemoveObservers("CharEvent")

        plotter.show(interactive_update=True)

        fps_timer = time.perf_counter()
        fps_count = 0
        current_fps = 0.0

        try:
            while not getattr(plotter, "_closed", False) and plotter.render_window is not None:
                if gui_state["exit_requested"]:
                    break

                if gui_state["harmonic_active"]:
                    m_harm = tau_manual_mag * np.sin(2.0 * np.pi * 15.0 * sim.current_time)
                    set_active_torque(m_harm)
                    gui_state["current_torque_val"] = m_harm

                sim.step()

                disp = sim.displacement
                grid.points = initial_points + disp
                thetas = compute_twist_angles(mesh.nodes, disp)
                grid.point_data["Twist_deg"] = thetas.astype(np.float32)

                fps_count += 1
                t_now = time.perf_counter()
                if t_now - fps_timer >= 0.5:
                    current_fps = fps_count / (t_now - fps_timer)
                    fps_count = 0
                    fps_timer = t_now

                tip_twist = float(np.mean(thetas[tip_nodes]))
                telemetry = (
                    f"Time: {sim.current_time:.3f} s | Rate: {1.0/args.dt:.0f} Hz (dt={args.dt*1e3:.1f}ms)\n"
                    f"Tip Twist: {tip_twist:+.2f}° | FPS: {current_fps:.1f}\n"
                    f"Harmonic: {'ACTIVE (15 Hz)' if gui_state['harmonic_active'] else 'OFF'} | "
                    f"Torque: {gui_state['current_torque_val']:.2e} N*mm"
                )
                telemetry_actor.SetText(2, telemetry)

                plotter.update()
        except Exception:
            pass

        return {}


def main() -> None:
    args = parse_arguments()
    is_headless = args.headless or (
        not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
    )
    out_dir = Path(args.output_dir)

    console.print(
        Panel.fit(
            "[bold cyan]Example 1101: Real-Time Interactive Dynamic Cantilever Beam Simulation[/bold cyan]\n"
            f"[white]Decoupled Generalized-alpha Dynamic Physics Engine • dt = {args.dt*1e3:.1f} ms ({1/args.dt:.0f} Hz)[/white]\n"
            f"[dim]Mode: {'All (Bending + Torsion)' if args.all else args.mode} • {'Headless' if is_headless else 'Interactive GUI'}[/dim]",
            title="[bold blue]project-sierra / interactive[/bold blue]",
            border_style="cyan",
        )
    )

    if args.all:
        run_bending_simulation(args, is_headless=True, out_dir=out_dir / "bending")
        run_torsion_simulation(args, is_headless=True, out_dir=out_dir / "torsion")
    elif args.mode == "bending":
        run_bending_simulation(args, is_headless=is_headless, out_dir=out_dir)
    else:
        run_torsion_simulation(args, is_headless=is_headless, out_dir=out_dir)

    console.print(
        "\n[bold green]✔ Interactive Dynamic Simulation Pipeline completed successfully![/bold green]\n"
    )


if __name__ == "__main__":
    main()
