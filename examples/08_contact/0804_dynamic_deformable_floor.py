"""
Simulation 22: Two-Way Deformable-to-Deformable Dynamic Contact Mechanics.
==========================================================================
Demonstrates:
1. MultiBodySystem assembly incorporating BOTH Beam (Impactor) and Floor (Flexible Target) domains.
2. Two-way deformable continuum mechanics with Total Lagrangian finite-strain formulation (Tet4TotalLagrangian).
3. Deformable-to-deformable mesh-to-mesh contact mechanics with exact action-reaction balance (f_s + sum(f_m) = 0).
4. Off-diagonal symmetric tangent stiffness coupling:
       K_coupled = [ K_A + K_c,AA   K_c,AB ]
                   [ K_c,BA         K_B + K_c,BB ]
5. Augmented Lagrangian Multiplier Method driving interface penetration error to near zero.
6. Rayleigh damping (alpha=2.0, beta=0.001) and Generalized-alpha integration (rho_inf=0.8).
7. Unified ParaView PVD export (factor=0.0) recording simultaneous deformation of BOTH flexible bodies.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import pyvista as pv
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
    select_nodes_by_bounds,
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
)
from psierra.damping import RayleighDamping
from psierra.boundary.contact import DeformableContact
from psierra.analysis import NonlinearDynamicAnalysis

install(show_locals=False)
console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 22: Two-Way Deformable-to-Deformable Dynamic Contact Mechanics"
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
        "--dt",
        type=float,
        default=0.002,
        help="Time step in seconds (default: 0.002 s = 500 Hz)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Number of dynamic time steps (default: 15 steps = 0.03 s, through impact & contact)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=1.0,
        help="Initial vertical gap between cantilever beam bottom and floor top in mm (default: 1.0 mm)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=200000.0,
        help="Normal penalty contact stiffness in N/mm (default: 200000.0 N/mm)",
    )
    parser.add_argument(
        "--al-gap-tol",
        type=float,
        default=1e-3,
        help="Maximum allowable penetration in mm under Augmented Lagrangian (default: 1e-3 mm = 1 um)",
    )
    parser.add_argument(
        "--al-tol",
        type=float,
        default=1e-3,
        help="Relative projected multiplier convergence tolerance under Augmented Lagrangian (default: 1e-3)",
    )
    parser.add_argument(
        "--al-max-iter",
        type=int,
        default=50,
        help="Maximum Augmented Lagrangian Uzawa iterations per step (default: 50)",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        choices=["augmented_lagrangian", "penalty"],
        default="augmented_lagrangian",
        help="Contact enforcement scheme: 'augmented_lagrangian' or 'penalty' (default: augmented_lagrangian)",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.0,
        help="Coulomb friction coefficient (default: 0.0)",
    )
    parser.add_argument(
        "--spectral-radius",
        type=float,
        default=0.8,
        help="Generalized-alpha spectral radius rho_inf in [0, 1] (default: 0.8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/08_contact/0804_dynamic_deformable_floor",
        help="Output directory path for ParaView PVD collection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    console.print(
        Panel.fit(
            "[bold cyan]Simulation 22: Two-Way Deformable-to-Deformable Dynamic Contact Mechanics[/bold cyan]\n"
            "[dim]MultiBodySystem Assembly (Beam + Flexible Floor) with Action-Reaction Coupling & Augmented Lagrangian[/dim]",
            title="[bold blue]project-sierra — Phase 3 Loop 2 Deliverable[/bold blue]",
            border_style="cyan",
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Upper Deformable Beam Mesh
    console.print(
        "\n[bold yellow][1/5] Initializing Upper Deformable Beam (Total Lagrangian)[/bold yellow]"
    )
    beam_path = Path(args.beam_mesh)
    if not beam_path.exists():
        raise FileNotFoundError(f"Beam mesh file not found: {beam_path}")

    beam_mesh = VolumetricMesh(beam_path)
    # The beam mesh consists of a fixed mounting block at Z in [-30, 0] with Y in [-60, 60] mm
    # and a cantilever arm at Z in [0, 200] with Y in [-25, 25] mm.
    # The floor is positioned under the cantilever landing zone (Z = 150 mm).
    cantilever_mask = beam_mesh.nodes[:, 2] > 0.0
    beam_bottom_y = float(beam_mesh.nodes[cantilever_mask, 1].min())  # -25.0 mm
    beam_top_y = float(beam_mesh.bounds[3])
    min_z = float(beam_mesh.bounds[4])
    max_z = float(beam_mesh.bounds[5])

    params_beam = IsotropicElasticParameters()
    params_beam.set_from_youngs_poisson(
        youngs_modulus=21000.0,  # 21,000 MPa (standard cantilever beam)
        poissons_ratio=0.35,
        mass_density=1100e-6,  # 1100e-6 tonne/mm^3
    )
    mat_beam = NeoHookean(params_beam)
    elem_beam = Tet4TotalLagrangian(mat_beam)
    beam_domain = SolidDomain(beam_mesh, mat_beam, elem_beam)

    console.print(
        f"  Beam Mesh: {beam_mesh.n_nodes} nodes, {beam_mesh.n_elements} elements"
    )
    console.print(
        f"  Beam Formulation: Total Lagrangian (Tet4TotalLagrangian, E = {params_beam.youngs_modulus:.0f} MPa)"
    )
    console.print(
        f"  Beam Y bounds: [{beam_bottom_y:.1f}, {beam_top_y:.1f}] mm (bottom at Y = {beam_bottom_y:.1f} mm)"
    )

    # 2. Load Lower Flexible Floor as Deformable VolumetricMesh
    console.print(
        "\n[bold yellow][2/5] Initializing Lower Flexible Floor as Deformable Continuum Domain[/bold yellow]"
    )
    floor_path = Path(args.floor_mesh)
    if not floor_path.exists():
        raise FileNotFoundError(f"Floor mesh file not found: {floor_path}")

    floor_mesh = VolumetricMesh(floor_path)
    orig_top_y = float(floor_mesh.bounds[3])

    # Position floor under cantilever landing zone (Z = 150 mm) with specified gap
    target_top_y = beam_bottom_y - args.gap
    dy = target_top_y - orig_top_y
    floor_mesh.transform(translation=(0.0, dy, 150.0))

    params_floor = IsotropicElasticParameters()
    params_floor.set_from_youngs_poisson(
        youngs_modulus=5000.0,  # 5,000 MPa (compliant flexible foundation)
        poissons_ratio=0.35,
        mass_density=1100e-6,
    )
    mat_floor = NeoHookean(params_floor)
    elem_floor = Tet4TotalLagrangian(mat_floor)
    floor_domain = SolidDomain(floor_mesh, mat_floor, elem_floor)

    console.print(
        f"  Floor Mesh: {floor_mesh.n_nodes} nodes, {floor_mesh.n_elements} elements"
    )
    console.print(
        f"  Floor Formulation: Total Lagrangian (Tet4TotalLagrangian, E = {params_floor.youngs_modulus:.0f} MPa)"
    )
    console.print(
        f"  Floor Y bounds: [{floor_mesh.bounds[2]:.1f}, {floor_mesh.bounds[3]:.1f}] mm (top at Y = {target_top_y:.1f} mm)"
    )
    console.print(f"  Configured Gap: {args.gap:.2f} mm")

    # 3. Assemble MultiBodySystem & Register Two-Way Deformable Contact
    console.print(
        "\n[bold yellow][3/5] Assembling MultiBodySystem & Registering Deformable Contact[/bold yellow]"
    )
    system = MultiBodySystem(
        {
            "Beam": beam_domain,
            "Floor": floor_domain,
        }
    )

    contact = system.add_contact_pair(
        slave="Beam",
        master="Floor",
        k_penalty=args.kp,
        formulation=args.formulation,
        friction=args.friction,
    )

    console.print(f"  Assembly Bodies : {system.names} (Total DOFs: {system.n_dofs})")
    console.print(
        f"  Contact Formulation: {type(contact).__name__}(slave='Beam', master='Floor', k_p = {args.kp:.1e} N/mm, mode={args.formulation})"
    )

    # Boundary conditions:
    # - Clamped beam root at min Z
    base_nodes = select_nodes_by_bounds(
        beam_mesh, bounds=(-1000.0, 1000.0, -1000.0, 1000.0, min_z - 1.0, min_z + 5.0)
    )
    beam_dof_offset = system.get_body_dof_slice("Beam").start
    clamp_beam = DirichletConstraint(
        base_nodes, components=(0, 1, 2), value=0.0, dof_offset=beam_dof_offset
    )

    # - Clamped bottom surface of flexible floor plate (leaving top free to flex)
    bot_nodes = np.where(
        np.isclose(floor_mesh.nodes[:, 1], floor_mesh.bounds[2], atol=1.0)
    )[0]
    floor_dof_offset = system.get_body_dof_slice("Floor").start
    clamp_floor = DirichletConstraint(
        bot_nodes,
        components=(0, 1, 2),
        value=0.0,
        dof_offset=floor_dof_offset,
    )

    dirichlet = DirichletManager(system.n_dofs)
    dirichlet.add_constraint(clamp_beam)
    dirichlet.add_constraint(clamp_floor)
    console.print(
        f"  Boundary Conditions: Clamped {len(base_nodes)} beam root nodes and {len(bot_nodes)} floor base nodes"
    )

    # Rayleigh Damping: alpha = 2.0, beta = 0.001
    damping = RayleighDamping(alpha=2.0, beta=0.001)
    console.print(
        f"  Rayleigh Damping   : alpha = {damping.alpha:.1f}, beta = {damping.beta:.4f}"
    )

    # Dynamic gravity load applied to the cantilever beam
    grav = GravityLoad(acceleration=(0.0, -9810.0, 0.0))
    M = system.assemble_mass_matrix()
    f_grav = grav.compute_force_vector(M)
    beam_mass = float(np.sum(M.diagonal()[: beam_domain.n_dofs]) / 3.0)
    console.print(
        f"  Dynamic Gravity    : g = -9810 mm/s^2 (Beam Mass = {beam_mass:.1f} tonne)"
    )

    # 4. Implicit Dynamic Solve
    console.print(
        "\n[bold yellow][4/5] Executing Generalized-alpha Dynamic Time Stepping[/bold yellow]"
    )
    solver = NonlinearDynamicAnalysis(
        system, damping=damping, spectral_radius=args.spectral_radius
    )
    t_start = time.perf_counter()
    result = solver.solve(
        dirichlet=dirichlet,
        f_ext=f_grav,
        dt=args.dt,
        n_steps=args.steps,
        nr_tol=1e-3,
        al_gap_tol=args.al_gap_tol,
        al_tol=args.al_tol,
        al_max_iterations=args.al_max_iter,
        factor=0.0,  # factor=0.0 exports reference mesh with 'Displacement' array for manual ParaView Warp by Vector
        save_directory=output_dir,
        output_name="result",
        save_interval=1,
        verbose=True,
    )
    solve_duration = time.perf_counter() - t_start

    # 5. Two-Way Contact Kinematics & Energy Analysis
    console.print(
        "\n[bold yellow][5/5] Two-Way Contact Verification & Deformation Analysis[/bold yellow]"
    )
    c_info = contact.get_contact_info()
    contact_energy = float(c_info["potential_energy"])
    max_pen = float(c_info["max_penetration"])
    num_active = int(c_info["num_active"])

    beam_disp_y_min = float(np.min(beam_domain.displacement[:, 1]))
    floor_disp_y_min = float(np.min(floor_domain.displacement[:, 1]))

    # Analyze time series for lowest point reached
    contact_occurred = False
    if result.time_series_files:
        for p in result.time_series_files:
            try:
                g = pv.read(p)
                beam_cells = np.where(g.cell_data["Body ID"] == 0)[0]
                if len(beam_cells) > 0:
                    beam_sub = g.extract_cells(beam_cells)
                    disp_y = beam_sub.point_data["Displacement"][:, 1]
                    y_min_step = float((beam_sub.points[:, 1] + disp_y).min())
                else:
                    disp_y = g.point_data["Displacement"][:, 1]
                    y_min_step = float((g.points[:, 1] + disp_y).min())
                if y_min_step <= target_top_y + 1e-3:
                    contact_occurred = True
            except Exception:
                pass

    table = Table(
        title="Simulation 22 Two-Way Deformable Contact Results Summary",
        border_style="cyan",
    )
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="green")
    table.add_column("Unit", style="dim")

    table.add_row("Solve Status", "CONVERGED" if result.converged else "FAILED", "")
    table.add_row("Total Computation Time", f"{solve_duration:.3f}", "s")
    table.add_row("Total Physical Duration", f"{args.steps * args.dt:.4f}", "s")
    table.add_row("Time Step (dt)", f"{args.dt:.4f}", "s")
    table.add_row("Assembly Constituents", f"{system.names}", "")
    table.add_row("Contact Formulation", f"{args.formulation}", "")
    table.add_row("Configured Gap (--gap)", f"{args.gap:.2f}", "mm")
    table.add_row(
        "Contact Interaction Active",
        "YES (Two-Way Dynamic Impact)" if contact_occurred else "NO",
        "",
    )
    table.add_row("Active Contacting Nodes", f"{num_active}", "")
    table.add_row("Max Compliance Penetration (g_N)", f"{max_pen:.4f}", "mm")
    table.add_row("Max Beam Vertical Deflection", f"{beam_disp_y_min:.3f}", "mm")
    table.add_row("Max Flexible Floor Deflection", f"{floor_disp_y_min:.3f}", "mm")
    table.add_row(
        "Stored Contact Potential Energy (Pi_c)", f"{contact_energy:.4f}", "N*mm (mJ)"
    )
    if getattr(contact, "formulation", "") == "augmented_lagrangian":
        max_lam = float(np.max(contact.multipliers))
        table.add_row("Max Multiplier (lambda_N)", f"{max_lam:.2f}", "N")

    console.print(table)

    console.print(
        Panel.fit(
            f"[bold green]✓ Two-Way Deformable Contact Saved to {output_dir / 'result.pvd'}[/bold green]\n"
            f"[dim]Assembly Layer: MultiBodySystem containing both {system.names}[/dim]\n"
            f"[dim]Upper Beam and Lower Flexible Floor deformed simultaneously under two-way interface action-reaction forces.[/dim]\n"
            f"[dim]Open result.pvd in ParaView and apply 'Warp By Vector' (Displacement) with scale factor 1.0.[/dim]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
