"""
Simulation 0810: Two-Way Deformable Pad Impact and Stick-Slip Sliding on Deformable Pole.
========================================================================================
Demonstrates:
1. Contact Showcase with newly introduced geometries:
   - Floating annular pad: psierra/data/pad1/mesh/pad1.vtk (Flexible Impactor)
   - Stationary center pole: psierra/data/pole1/mesh/pole1.vtk (Flexible Target with Fixed Shaft)
2. Two-Way Deformable-to-Deformable 3D Contact Mechanics:
   - MultiBodySystem assembly incorporating BOTH Pole and Pad continuum domains.
   - Total Lagrangian hyperelastic formulation (Tet4TotalLagrangian, Neo-Hookean).
   - Exact action-reaction momentum conservation (f_slave + sum f_master = 0).
   - Off-diagonal symmetric tangent stiffness coupling between Pole and Pad.
3. Pillar 2 Spatial Regularization (Smoothed Normals):
   - Continuous angle-weighted vertex pseudonormal field on the cylindrical master pole.
   - Eliminates directional facet-edge chattering and ensures smooth quadratic Newton convergence.
4. 3D Active-Set Coulomb Stick-Slip Friction:
   - Lateral gravity (g_x = -9810 mm/s^2) closes 10 mm annular clearance and establishes normal contact.
   - Axial gravity (g_y = -2000 mm/s^2) drives axial sliding along the pole cylinder.
   - Elastoplastic return mapping enforces stick-slip transition:
     Phi_trial = ||f_T^trial|| - mu * f_N <= 0.
5. Consistent Algorithmic Tangent and High-Fidelity Output:
   - Generalized-alpha integration (rho_inf = 0.8) with controllable dissipation.
   - Unified XDMF/HDF5 and PVD ParaView export via system.unified_mesh in results/08_contact/0810_deformable_pole_pad_contact.
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
from psierra.multibody import MultiBodySystem
from psierra.boundary import (
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
    StickSlipState,
)
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
        description="Simulation 0810: Two-Way Deformable Pad on Deformable Pole Dynamic Contact Mechanics"
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
        help="Number of dynamic time steps (default: 60 steps = 0.060 s, through gap closure and sustained contact)",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.30,
        help="Coulomb friction coefficient mu (default: 0.30)",
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
        "--no-smoothed-normals",
        action="store_true",
        help="Disable Pillar 2 vertex pseudonormal smoothing on cylindrical master facet surface",
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
        default="results/08_contact/0810_deformable_pole_pad_contact",
        help="Output directory (default: results/08_contact/0810_deformable_pole_pad_contact)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    console.print(
        Panel.fit(
            f"[bold cyan]Simulation 0810: Two-Way Deformable Pad on Deformable Pole Contact[/bold cyan]\n"
            f"[dim]Two-Way Continuum Contact • Pillar 2 Smoothed Normals • Stick-Slip Coulomb Friction[/dim]\n\n"
            f"Pad Mesh (Impactor)      : [white]{args.pad_mesh}[/white]\n"
            f"Pole Mesh (Flexible)     : [white]{args.pole_mesh}[/white]\n"
            f"Friction Coefficient mu  : [yellow]{args.friction}[/yellow]\n"
            f"Normal Penalty kp        : [green]{args.kp:.1e} N/mm[/green]\n"
            f"Tangential Penalty kt    : [green]{args.kt or args.kp:.1e} N/mm[/green]\n"
            f"Contact Damping cdamp    : [cyan]{args.cdamp:.3e} N*s/mm[/cyan]\n"
            f"Smoothed Normals         : [green]{not args.no_smoothed_normals}[/green]\n"
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
    console.print("\n[bold yellow][1/5] Loading Volumetric Mesh Geometries...[/bold yellow]")
    pad_file = _resolve_mesh_path(args.pad_mesh)
    pole_file = _resolve_mesh_path(args.pole_mesh)
    if not pad_file.exists():
        raise FileNotFoundError(f"Pad mesh not found: {pad_file}")
    if not pole_file.exists():
        raise FileNotFoundError(f"Pole mesh not found: {pole_file}")

    pole_mesh = VolumetricMesh(pole_file)
    pad_mesh = VolumetricMesh(pad_file)

    console.print(
        f"  Pole Mesh: {pole_mesh.n_nodes} nodes, {pole_mesh.n_elements} elements, "
        f"r in [{np.sqrt(pole_mesh.nodes[:, 0]**2 + pole_mesh.nodes[:, 2]**2).min():.1f}, "
        f"{np.sqrt(pole_mesh.nodes[:, 0]**2 + pole_mesh.nodes[:, 2]**2).max():.1f}] mm, "
        f"Y in [{pole_mesh.nodes[:, 1].min():.1f}, {pole_mesh.nodes[:, 1].max():.1f}] mm"
    )
    console.print(
        f"  Pad Mesh: {pad_mesh.n_nodes} nodes, {pad_mesh.n_elements} elements, "
        f"hole r = {np.sqrt(pad_mesh.nodes[:, 0]**2 + pad_mesh.nodes[:, 2]**2).min():.1f} mm, "
        f"Y in [{pad_mesh.nodes[:, 1].min():.1f}, {pad_mesh.nodes[:, 1].max():.1f}] mm"
    )

    # 2. Material & Deformable Continuum Domains
    console.print("\n[bold yellow][2/5] Initializing Deformable Continuum Domains...[/bold yellow]")
    # Pole: Stiff structural alloy (E = 50 GPa = 50,000 MPa, nu = 0.25, rho = 2.7e-6 kg/mm^3)
    params_pole = IsotropicElasticParameters()
    params_pole.set_from_youngs_poisson(
        youngs_modulus=50000.0,
        poissons_ratio=0.25,
        mass_density=2.7e-6,
    )
    mat_pole = NeoHookean(params_pole)
    elem_pole = Tet4TotalLagrangian(mat_pole)
    pole_domain = SolidDomain(pole_mesh, mat_pole, elem_pole)

    # Pad: Compliant polymer/elastomer (E = 1000 MPa, nu = 0.30, rho = 1.2e-6 kg/mm^3)
    params_pad = IsotropicElasticParameters()
    params_pad.set_from_youngs_poisson(
        youngs_modulus=1000.0,
        poissons_ratio=0.30,
        mass_density=1.2e-6,
    )
    mat_pad = NeoHookean(params_pad)
    elem_pad = Tet4TotalLagrangian(mat_pad)
    pad_domain = SolidDomain(pad_mesh, mat_pad, elem_pad)

    # 3. Assemble MultiBodySystem & Register Two-Way Contact
    console.print("\n[bold yellow][3/5] Assembling MultiBodySystem and Registering Contact Pair...[/bold yellow]")
    system = MultiBodySystem(
        {
            "Pole": pole_domain,
            "Pad": pad_domain,
        }
    )

    if args.initial_offset_x != 0.0:
        u_init = np.zeros(system.n_dofs, dtype=np.float64)
        pad_sl = system.get_body_dof_slice("Pad")
        u_init[pad_sl] = np.tile([args.initial_offset_x, 0.0, 0.0], pad_mesh.n_nodes)
        system.update_state(u=u_init)
        console.print(f"  Applied Initial Displacement to Pad: X = {args.initial_offset_x:.2f} mm")

    # Candidate contact nodes on Pad: inner hole (r < 65 mm)
    surf_nodes = pad_mesh.surface.node_indices
    coords = pad_mesh.nodes[surf_nodes]
    r_surf = np.sqrt(coords[:, 0] ** 2 + coords[:, 2] ** 2)
    inner_nodes = surf_nodes[r_surf < 65.0]
    console.print(
        f"  Pad Inner Hole Contact Candidates: {len(inner_nodes)} nodes (out of {len(surf_nodes)} surface nodes)"
    )

    contact = system.add_contact_pair(
        slave="Pad",
        master="Pole",
        slave_nodes=inner_nodes,
        k_penalty=args.kp,
        k_tangential=args.kt or args.kp,
        c_damping=args.cdamp,
        friction=args.friction,
        formulation="penalty",
        smoothed_normals=not args.no_smoothed_normals,
        smoothing_weight="angle",
    )

    # Fixed boundary conditions on Pole: inner cylindrical face (r < 20 mm)
    pole_inner_nodes = np.flatnonzero(
        np.sqrt(pole_mesh.nodes[:, 0] ** 2 + pole_mesh.nodes[:, 2] ** 2) < 20.0
    )
    pole_offset = int(system._dof_offsets[0])
    dirichlet = DirichletManager(system.n_dofs)
    dirichlet.add_constraint(
        DirichletConstraint(pole_inner_nodes, components=(0, 1, 2), dof_offset=pole_offset)
    )
    console.print(
        f"  Pole Dirichlet Boundary Condition: {len(pole_inner_nodes)} nodes fixed at inner cylindrical mounting shaft"
    )

    # 4. External Loads & Generalized-alpha Dynamic Integration
    console.print("\n[bold yellow][4/5] Assembling System Dynamics & Generalized-alpha Solver...[/bold yellow]")
    gravity = GravityLoad(acceleration=(args.gx, args.gy, 0.0))
    M = system.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    mass_pole = float(np.sum(M[: pole_domain.n_dofs, : pole_domain.n_dofs]) / 3.0)
    mass_pad = float(np.sum(M[pole_domain.n_dofs :, pole_domain.n_dofs :]) / 3.0)
    console.print(f"  System Masses: Pole = {mass_pole:.3f} kg, Pad = {mass_pad:.3f} kg")
    console.print(f"  Total System DOFs: {system.n_dofs} (Pole: {pole_domain.n_dofs}, Pad: {pad_domain.n_dofs})")

    damping = RayleighDamping(alpha=1.0, beta=0.001)
    solver = NonlinearDynamicAnalysis(
        domain=system,
        damping=damping,
        spectral_radius=0.8,
    )

    # 5. Execute Simulation
    console.print("\n[bold yellow][5/5] Launching Two-Way Dynamic Contact Simulation...[/bold yellow]")
    start_time = time.perf_counter()
    sol = solver.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        dt=args.dt,
        n_steps=args.steps,
        contact=system.contacts,
        nr_iterations=20,
        nr_tol=1e-4,
        save_directory=out_path,
        output_name="deformable_pole_pad",
        format=args.format,
        save_interval=args.save_interval,
        include_stresses=True,
        smooth_to_nodes=True,
        verbose=True,
    )
    elapsed = time.perf_counter() - start_time

    # 6. Diagnostics & Telemetry
    contact_info = contact.get_contact_info()
    f_contact = contact.evaluate_forces(n_dofs=system.n_dofs)
    f_c_norm = float(np.linalg.norm(f_contact))

    u_pad = pad_domain.u.reshape(-1, 3)
    max_ux_pad = float(np.max(np.abs(u_pad[:, 0])))
    max_uy_pad = float(np.max(np.abs(u_pad[:, 1])))

    u_pole = pole_domain.u.reshape(-1, 3)
    max_ux_pole = float(np.max(np.abs(u_pole[:, 0])))
    max_uy_pole = float(np.max(np.abs(u_pole[:, 1])))

    summary_table = Table(
        title="Simulation 0810: Two-Way Deformable Contact & Telemetry Summary",
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
    summary_table.add_row("Pad Max Lateral Disp (X)", f"{max_ux_pad:.3f} mm")
    summary_table.add_row("Pad Max Axial Disp (Y)", f"{max_uy_pad:.3f} mm")
    summary_table.add_row("Pole Max Lateral Disp (X)", f"{max_ux_pole:.3f} mm")
    summary_table.add_row("Pole Max Axial Disp (Y)", f"{max_uy_pole:.3f} mm")
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
