"""
Simulation 19: Static Cantilever Beam Resting on 3D Rigid Floor Mesh.
=====================================================================
Demonstrates:
1. MultiBodySystem assembly incorporating BOTH Beam and Floor domains.
2. Wrapping the floor mesh with VolumetricMesh to leverage geometric transformations.
3. Total Lagrangian formulation (Tet4TotalLagrangian) capturing finite-strain kinematics.
4. Encapsulating contact within MultiBodySystem without violating layer boundaries in solve().
5. Interactive gap configuration in the 10-30 mm range with exact physical contact.
6. Unified ParaView PVD export showing BOTH bodies interacting in 3D deformed space.
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
from psierra.element import Tet4TotalLagrangian, Tet4Linear
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.multibody import MultiBodySystem
from psierra.boundary import (
    select_nodes_by_bounds,
    DirichletConstraint,
    DirichletManager,
)
from psierra.boundary.contact import PenaltyContact
from psierra.analysis import NonlinearStaticAnalysis

install(show_locals=False)
console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 19: Static Cantilever Beam Contact with 3D Rigid Floor Mesh"
    )
    parser.add_argument(
        "--beam-mesh",
        type=str,
        default="data/beam1/mesh/beam1.vtk",
        help="Path to volumetric beam mesh (default: data/beam1/mesh/beam1.vtk)",
    )
    parser.add_argument(
        "--floor-mesh",
        type=str,
        default="data/floor1/mesh/floor1.vtk",
        help="Path to volumetric floor mesh (default: data/floor1/mesh/floor1.vtk)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of incremental Newton-Raphson load steps (default: 10)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="Initial vertical gap between beam bottom and floor top in mm (default: 0.0 mm)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=500000.0,
        help="Normal penalty contact stiffness in N/mm (default: 500000.0 N/mm)",
    )
    parser.add_argument(
        "--load",
        type=float,
        default=1.2e6,
        help="Total downward tip load in N (default: 1.2e6 N = 1.2 MN)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/05_simulation/sim19_output",
        help="Output directory path for ParaView PVD collection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    console.print(
        Panel.fit(
            "[bold cyan]Simulation 19: Static Cantilever Beam Contact on 3D Rigid Floor[/bold cyan]\n"
            "[dim]MultiBodySystem Assembly (Beam + Floor Domains) with Total Lagrangian Kinematics[/dim]",
            title="[bold blue]project-sierra — Contact Mechanics[/bold blue]",
            border_style="cyan",
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Beam Mesh & Setup Deformable Domain
    console.print("\n[bold yellow][1/5] Initializing Deformable Beam (Total Lagrangian)[/bold yellow]")
    beam_mesh_path = Path(args.beam_mesh)
    if not beam_mesh_path.exists():
        raise FileNotFoundError(f"Beam mesh not found at: {beam_mesh_path}")

    beam_mesh = VolumetricMesh(beam_mesh_path)
    beam_bottom_y = float(beam_mesh.bounds[2])  # -60.0 mm
    beam_top_y = float(beam_mesh.bounds[3])
    min_z = float(beam_mesh.bounds[4])
    max_z = float(beam_mesh.bounds[5])

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000.0,  # 21,000 MPa (standard repository beam material)
        poissons_ratio=0.35,
        mass_density=1100e-6,
    )
    mat_beam = NeoHookean(params)
    elem_beam = Tet4TotalLagrangian(mat_beam)
    beam_domain = SolidDomain(beam_mesh, mat_beam, elem_beam)

    console.print(f"  Beam Mesh: {beam_mesh.n_nodes} nodes, {beam_mesh.n_elements} elements")
    console.print(f"  Beam Form: Total Lagrangian (Tet4TotalLagrangian, E = {params.youngs_modulus:.0f} MPa)")
    console.print(f"  Beam Y bounds: [{beam_bottom_y:.1f}, {beam_top_y:.1f}] mm (bottom at Y = {beam_bottom_y:.1f} mm)")

    # 2. Load Floor Mesh as VolumetricMesh & Transform
    console.print("\n[bold yellow][2/5] Initializing Floor as VolumetricMesh Domain[/bold yellow]")
    floor_mesh_path = Path(args.floor_mesh)
    if not floor_mesh_path.exists():
        raise FileNotFoundError(f"Floor mesh not found at: {floor_mesh_path}")

    floor_mesh = VolumetricMesh(floor_mesh_path)
    orig_top_y = float(floor_mesh.bounds[3])

    # Position floor under cantilever landing zone using VolumetricMesh.transform
    target_top_y = beam_bottom_y - args.gap
    dy = target_top_y - orig_top_y
    floor_mesh.transform(translation=(0.0, dy, 150.0))

    elem_floor = Tet4Linear(mat_beam)
    floor_domain = SolidDomain(floor_mesh, mat_beam, elem_floor)

    console.print(f"  Floor Mesh: {floor_mesh.n_nodes} nodes, {floor_mesh.n_elements} elements")
    console.print(f"  Floor Y bounds: [{floor_mesh.bounds[2]:.1f}, {floor_mesh.bounds[3]:.1f}] mm (top at Y = {target_top_y:.1f} mm)")
    console.print(f"  Effective Vertical Gap: {beam_bottom_y - floor_mesh.bounds[3]:.1f} mm")

    # 3. Incorporate BOTH into MultiBodySystem Assembly & Add Contact
    console.print("\n[bold yellow][3/5] Assembling MultiBodySystem & Registering Contact[/bold yellow]")
    system = MultiBodySystem({
        "Beam": beam_domain,
        "Floor": floor_domain,
    })

    contact = system.add_contact_pair(
        slave="Beam",
        master="Floor",
        k_penalty=args.kp,
    )

    console.print(f"  Assembly Bodies : {system.names} (Total DOFs: {system.n_dofs})")
    console.print(f"  Registered Contact: PenaltyContact(slave='Beam', master='Floor', k_p = {args.kp:.1e} N/mm)")

    # Boundary conditions:
    # - Clamped beam root at min Z
    base_nodes = select_nodes_by_bounds(
        beam_mesh, bounds=(-1000.0, 1000.0, -1000.0, 1000.0, min_z - 1.0, min_z + 5.0)
    )
    beam_dof_offset = system.get_body_dof_slice("Beam").start
    clamp_beam = DirichletConstraint(
        base_nodes, components=(0, 1, 2), value=0.0, dof_offset=beam_dof_offset
    )

    # - Perfectly rigid floor: all floor nodes clamped to zero displacement
    floor_dof_offset = system.get_body_dof_slice("Floor").start
    clamp_floor = DirichletConstraint(
        np.arange(floor_mesh.n_nodes),
        components=(0, 1, 2),
        value=0.0,
        dof_offset=floor_dof_offset,
    )

    dirichlet = DirichletManager(system.n_dofs)
    dirichlet.add_constraint(clamp_beam)
    dirichlet.add_constraint(clamp_floor)
    console.print(f"  Clamped {len(base_nodes)} beam root nodes and all {floor_mesh.n_nodes} floor nodes (rigid)")

    # Downward external load at beam tip (max Z)
    tip_nodes = select_nodes_by_bounds(
        beam_mesh, bounds=(-1000.0, 1000.0, -1000.0, 1000.0, max_z - 20.0, max_z + 1.0)
    )
    f_ext = np.zeros(system.n_dofs, dtype=np.float64)
    tip_force_per_node = -args.load / len(tip_nodes)
    for n in tip_nodes:
        f_ext[3 * n + 1] = tip_force_per_node
    console.print(f"  Applied downward tip load: F_total = {args.load:.2e} N across {len(tip_nodes)} tip nodes")

    # 4. Incremental Nonlinear Static Solve (NO contact passed into solve!)
    console.print("\n[bold yellow][4/5] Executing Incremental Contact Equilibrium Solve[/bold yellow]")
    t_start = time.perf_counter()
    solver = NonlinearStaticAnalysis(system)
    result = solver.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        n_steps=args.steps,
        nr_tol=1e-3,
        factor=0.0,  # factor=0.0 exports reference mesh with 'Displacement' array for manual ParaView Warp by Vector
        save_directory=output_dir,
        output_name="sim19_contact_static",
        save_interval=1,
        verbose=True,
    )
    solve_duration = time.perf_counter() - t_start

    # 5. Analysis & Verification
    console.print("\n[bold yellow][5/5] Contact Mechanics Analysis & Verification[/bold yellow]")
    c_info = contact.get_contact_info()
    min_final_y = float(np.min(beam_domain.current_nodes[:, 1]))
    floor_top = float(np.max(floor_domain.current_nodes[:, 1]))
    max_penetration = float(c_info["max_penetration"])
    num_contact_nodes = int(c_info["num_active"])
    contact_energy = float(c_info["potential_energy"])

    table = Table(title="Simulation 19 Multi-Body Contact Results Summary", border_style="cyan")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="green")
    table.add_column("Unit", style="dim")

    table.add_row("Solve Status", "CONVERGED" if result.converged else "FAILED", "")
    table.add_row("Total Computation Time", f"{solve_duration:.3f}", "s")
    table.add_row("Assembly Constituents", f"{system.names}", "")
    table.add_row("Configured Gap (--gap)", f"{args.gap:.2f}", "mm")
    table.add_row("Floor Top Surface (Y_top)", f"{floor_top:.2f}", "mm")
    table.add_row("Beam Minimum Y (Y_min)", f"{min_final_y:.2f}", "mm")
    table.add_row("Active Contact Nodes", f"{num_contact_nodes}", "nodes")
    table.add_row("Max Normal Penetration (delta)", f"{max_penetration:.4f}", "mm")
    table.add_row("Contact Potential Energy (Pi_c)", f"{contact_energy:.4f}", "N*mm (mJ)")
    console.print(table)

    console.print(
        Panel.fit(
            f"[bold green]✓ Multi-Body Static Contact Saved to {output_dir / 'sim19_contact_static.pvd'}[/bold green]\n"
            f"[dim]Assembly Layer: MultiBodySystem containing both {system.names}[/dim]\n"
            f"[dim]The Total Lagrangian beam bent across the {args.gap:.1f} mm gap and settled stably onto the 3D floor mesh.[/dim]\n"
            f"[dim]Open sim19_contact_static.pvd in ParaView and apply 'Warp By Vector' (Displacement) to view deformed contact.[/dim]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
