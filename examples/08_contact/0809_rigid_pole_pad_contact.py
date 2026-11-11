"""
Simulation 0809: Deformable Pad Impact and Stick-Slip Sliding on Rigid Pole.
=============================================================================
Demonstrates:
1. Contact Showcase with newly introduced geometries:
   - Floating annular pad: psierra/data/pad1/mesh/pad1.vtk
   - Stationary center pole: psierra/data/pole1/mesh/pole1.vtk (Rigid Obstacle)
2. Deformable-to-Rigid 3D Contact Mechanics:
   - RayMeshObstacle representation of the triangulated cylindrical pole surface.
   - Total Lagrangian hyperelastic formulation (Tet4TotalLagrangian, Neo-Hookean).
   - Candidate slave nodes constrained to inner pad hole for optimal efficiency.
3. 3D Active-Set Coulomb Stick-Slip Friction:
   - Lateral gravity (g_x = -9810 mm/s^2) closes 10 mm annular clearance and establishes normal contact.
   - Axial gravity (g_y = -2000 mm/s^2) drives axial sliding along the pole cylinder.
   - Elastoplastic return mapping enforces stick-slip transition:
     Phi_trial = ||f_T^trial|| - mu * f_N <= 0.
4. Consistent Algorithmic Tangent and High-Fidelity Output:
   - Generalized-alpha integration (rho_inf = 0.8) with controllable dissipation.
   - Unified XDMF/HDF5 and PVD ParaView export in results/08_contact/0809_rigid_pole_pad_contact.
   - Rich telemetry reporting normal forces, friction traction, and stick/slip states.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import (
    DirichletManager,
    GravityLoad,
    RigidContact,
    StickSlipState,
)
from psierra.collision import RayMeshObstacle
from psierra.damping import RayleighDamping
from psierra.analysis import NonlinearDynamicAnalysis

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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 0809: Deformable Pad Impact and Stick-Slip Sliding on Rigid Pole"
    )
    parser.add_argument(
        "--pad-mesh",
        type=str,
        default="data/pad1/mesh/pad1.vtk",
        help="Path to volumetric pad mesh (default: data/pad1/mesh/pad1.vtk)",
    )
    parser.add_argument(
        "--pole-mesh",
        type=str,
        default="data/pole1/mesh/pole1.vtk",
        help="Path to volumetric pole mesh (default: data/pole1/mesh/pole1.vtk)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Dynamic time increment in seconds (default: 0.001 s = 1000 Hz)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=60,
        help="Number of dynamic time steps (default: 60 steps = 0.060 s, ensuring sustained contact)",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.35,
        help="Coulomb friction coefficient mu (default: 0.35)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=5000.0,
        help="Normal penalty contact stiffness in N/mm (default: 5000.0 N/mm)",
    )
    parser.add_argument(
        "--kt",
        type=float,
        default=None,
        help="Tangential stick penalty stiffness in N/mm (default: same as kp)",
    )
    parser.add_argument(
        "--cdamp",
        type=float,
        default=0.005,
        help="Contact normal velocity damping in N*s/mm (default: 0.005 N*s/mm)",
    )
    parser.add_argument(
        "--gx",
        type=float,
        default=-9810.0,
        help="Lateral gravity acceleration in mm/s^2 closing gap (default: -9810.0 mm/s^2)",
    )
    parser.add_argument(
        "--gy",
        type=float,
        default=-2000.0,
        help="Axial gravity acceleration in mm/s^2 driving sliding (default: -2000.0 mm/s^2)",
    )
    parser.add_argument(
        "--initial-offset-x",
        type=float,
        default=0.0,
        help="Initial X offset of pad in mm (default: 0.0 mm, e.g. -9.5 mm to start near contact)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="xdmf",
        choices=["xdmf", "pvd"],
        help="Export format for ParaView visualization (default: xdmf)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Output streaming stride (default: 1 step)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/08_contact/0809_rigid_pole_pad_contact",
        help="Output directory (default: results/08_contact/0809_rigid_pole_pad_contact)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    console.print(
        Panel.fit(
            f"[bold cyan]Simulation 0809: Deformable Pad on Rigid Pole Contact[/bold cyan]\n"
            f"[dim]Deformable-to-Rigid Contact with Stick-Slip Coulomb Friction Return Mapping[/dim]\n\n"
            f"Pad Mesh                 : [white]{args.pad_mesh}[/white]\n"
            f"Pole Mesh (Obstacle)     : [white]{args.pole_mesh}[/white]\n"
            f"Friction Coefficient mu  : [yellow]{args.friction}[/yellow]\n"
            f"Normal Penalty kp        : [green]{args.kp:.1e} N/mm[/green]\n"
            f"Tangential Penalty kt    : [green]{args.kt or args.kp:.1e} N/mm[/green]\n"
            f"Contact Damping cdamp    : [cyan]{args.cdamp:.3e} N*s/mm[/cyan]\n"
            f"Gravity Vector (gx, gy)  : [yellow]({args.gx:.1f}, {args.gy:.1f}, 0.0) mm/s^2[/yellow]\n"
            f"Time Increment dt        : [magenta]{args.dt * 1000:.2f} ms[/magenta]\n"
            f"Total Steps              : [cyan]{args.steps}[/cyan] (Total time: {args.steps * args.dt:.3f} s)\n"
            f"Export Output            : [white]{args.output_dir}[/white] ([bold]{args.format}[/bold])",
            border_style="cyan",
        )
    )

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Meshes
    console.print("\n[bold yellow][1/5] Loading Mesh Geometries...[/bold yellow]")
    pad_file = _resolve_mesh_path(args.pad_mesh)
    pole_file = _resolve_mesh_path(args.pole_mesh)
    if not pad_file.exists():
        raise FileNotFoundError(f"Pad mesh not found: {pad_file}")
    if not pole_file.exists():
        raise FileNotFoundError(f"Pole mesh not found: {pole_file}")

    pad_mesh = VolumetricMesh(pad_file)
    console.print(
        f"  Pad Mesh: {pad_mesh.n_nodes} nodes, {pad_mesh.n_elements} elements, "
        f"X bounds: [{pad_mesh.bounds[0]:.1f}, {pad_mesh.bounds[1]:.1f}] mm, "
        f"Y bounds: [{pad_mesh.bounds[2]:.1f}, {pad_mesh.bounds[3]:.1f}] mm"
    )

    # Pole as RayMeshObstacle
    pole_obstacle = RayMeshObstacle(pole_file)
    console.print(
        f"  Pole Obstacle: Wrapped via RayMeshObstacle with {pole_obstacle.polydata.n_cells} surface facets"
    )

    # 2. Material & Deformable Domain Setup for Pad
    console.print("\n[bold yellow][2/5] Setting Up Deformable Pad Domain...[/bold yellow]")
    params_pad = IsotropicElasticParameters()
    params_pad.set_from_youngs_poisson(
        youngs_modulus=1000.0,  # 1000 MPa (compliant elastomer/polymer)
        poissons_ratio=0.30,
        mass_density=1.2e-6,   # 1200 kg/m^3 = 1.2e-6 kg/mm^3
    )
    mat_pad = NeoHookean(params_pad)
    elem_pad = Tet4TotalLagrangian(mat_pad)
    pad_domain = SolidDomain(pad_mesh, mat_pad, elem_pad)

    if args.initial_offset_x != 0.0:
        pad_domain.update_state(u=np.tile([args.initial_offset_x, 0.0, 0.0], pad_mesh.n_nodes))
        console.print(f"  Applied Initial Displacement: X = {args.initial_offset_x:.2f} mm")

    # Candidate contact nodes: inner hole of pad (r < 65 mm)
    surf_nodes = pad_mesh.surface.node_indices
    coords = pad_mesh.nodes[surf_nodes]
    r_surf = np.sqrt(coords[:, 0] ** 2 + coords[:, 2] ** 2)
    inner_nodes = surf_nodes[r_surf < 65.0]
    console.print(
        f"  Pad Inner Hole Contact Candidates: {len(inner_nodes)} nodes (out of {len(surf_nodes)} surface nodes)"
    )

    # 3. Contact Pair Setup (Deformable-to-Rigid with Stick-Slip Friction)
    console.print("\n[bold yellow][3/5] Configuring Rigid Contact with Coulomb Friction...[/bold yellow]")
    contact = RigidContact(
        slave=pad_domain,
        master=pole_obstacle,
        slave_nodes=inner_nodes,
        k_penalty=args.kp,
        k_tangential=args.kt or args.kp,
        c_damping=args.cdamp,
        friction=args.friction,
        formulation="penalty",
    )

    # 4. External Forces & Damping
    console.print("\n[bold yellow][4/5] Assembling Loads and Generalized-alpha Solver...[/bold yellow]")
    gravity = GravityLoad(acceleration=(args.gx, args.gy, 0.0))
    M = pad_domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    total_pad_mass = float(np.sum(M) / 3.0)
    console.print(f"  Pad Total Mass: {total_pad_mass:.3f} kg")
    console.print(f"  Total Gravity Force Vector: Fx = {f_ext[0::3].sum():.2f} N, Fy = {f_ext[1::3].sum():.2f} N")

    dirichlet = DirichletManager(pad_domain.n_dofs)
    damping = RayleighDamping(alpha=1.0, beta=0.001)

    solver = NonlinearDynamicAnalysis(
        domain=pad_domain,
        damping=damping,
        spectral_radius=0.8,
    )

    # 5. Execute Simulation
    console.print("\n[bold yellow][5/5] Launching Dynamic Contact Simulation...[/bold yellow]")
    start_time = time.perf_counter()
    sol = solver.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        dt=args.dt,
        n_steps=args.steps,
        contact=contact,
        nr_iterations=20,
        nr_tol=1e-4,
        save_directory=out_path,
        output_name="rigid_pole_pad",
        format=args.format,
        save_interval=args.save_interval,
        include_stresses=True,
        smooth_to_nodes=True,
        verbose=True,
    )
    elapsed = time.perf_counter() - start_time

    # 6. Diagnostics & Telemetry
    contact_info = contact.get_contact_info()
    f_contact = contact.evaluate_forces(n_dofs=pad_domain.n_dofs)
    f_c_norm = float(np.linalg.norm(f_contact))

    u_pad = pad_domain.u.reshape(-1, 3)
    max_ux = float(np.max(np.abs(u_pad[:, 0])))
    max_uy = float(np.max(np.abs(u_pad[:, 1])))

    v_pad = pad_domain.v.reshape(-1, 3)
    max_vx = float(np.max(np.abs(v_pad[:, 0])))
    max_vy = float(np.max(np.abs(v_pad[:, 1])))

    summary_table = Table(
        title="Simulation 0809: Telemetry & Contact Resolution Summary",
        border_style="green",
    )
    summary_table.add_column("Metric / Field", style="bold cyan")
    summary_table.add_column("Result Value", style="bold white")

    summary_table.add_row(
        "Solver Status",
        "[bold green]CONVERGED (All Steps Resolved)[/bold green]"
        if sol.converged
        else "[bold red]FAILED[/bold red]",
    )
    summary_table.add_row("Wall Clock Time", f"{elapsed:.2f} s")
    summary_table.add_row("Steps Completed", f"{sol.steps_completed} / {args.steps}")
    summary_table.add_row("Total Time Simulated", f"{sol.steps_completed * args.dt * 1000:.1f} ms")
    n_active = len(np.atleast_1d(contact_info.get("active_nodes", []))) if "active_nodes" in contact_info else 0
    summary_table.add_row("Active Contact Nodes", f"{n_active} nodes")
    summary_table.add_row("Max Penetration Depth", f"{contact_info.get('max_penetration', 0.0):.4f} mm")
    summary_table.add_row("Total Contact Force ||F_c||", f"{f_c_norm:.2f} N")
    summary_table.add_row("Max Lateral Displacement (X)", f"{max_ux:.3f} mm")
    summary_table.add_row("Max Axial Displacement (Y)", f"{max_uy:.3f} mm")
    summary_table.add_row("Final Peak Velocity (X)", f"{max_vx:.2f} mm/s")
    summary_table.add_row("Final Peak Velocity (Y)", f"{max_vy:.2f} mm/s")
    summary_table.add_row("Stick Candidates", f"{contact_info.get('num_stick', 0)}")
    summary_table.add_row("Slip Candidates", f"{contact_info.get('num_slip', 0)}")
    summary_table.add_row("Max Friction Force", f"{contact_info.get('max_friction_force', 0.0):.2f} N")
    summary_table.add_row("Max Slip Distance", f"{contact_info.get('max_slip_distance', 0.0):.4f} mm")
    summary_table.add_row("Exported Frames", f"{len(sol.time_series_files)} frame(s)")

    console.print()
    console.print(summary_table)

    if sol.time_series_files:
        console.print(
            f"\n[bold green]✓ Visualization files successfully saved to:[/bold green] [white]{out_path}[/white]"
        )


if __name__ == "__main__":
    main()
