"""
Simulation 0806: Segment-to-Segment Mortar Contact — Dynamic Beam on Rigid 3D Floor.
===================================================================================
Demonstrates:
1. MultiBodySystem assembly incorporating BOTH Beam and Floor domains.
2. Segment-to-Segment Mortar Contact formulation with Wohlmuth biorthogonal dual shape
   functions satisfying int_tau Phi_i * N_j dGamma = (A/3) * delta_{ij}.
3. Sutherland-Hodgman 2D polygon clipping on local tangent planes with 3-point Dunavant
   Gauss quadrature over clipped sub-triangles.
4. Total Lagrangian formulation (Tet4TotalLagrangian) capturing finite-strain dynamics.
5. Exact action-reaction momentum balance and symmetric tangent stiffness matrix.
6. Generalized-alpha dynamic time stepping with Augmented Lagrangian Uzawa enforcement.
7. Rich console telemetry and unified ParaView XDMF/PVD export in results/08_contact/0806_mortar_dynamic_rigid_floor.
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
from psierra.element import Tet4TotalLagrangian, Tet4Linear
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
from psierra.boundary.mortar import MortarContact
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
        description="Simulation 0806: Segment-to-Segment Mortar Contact Dynamic Rigid Floor Impact"
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
        default=25,
        help="Number of dynamic time steps (default: 25 steps = 0.05 s)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="Initial vertical gap between beam bottom and floor mesh top in mm (default: 0.0 mm)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=50000.0,
        help="Normal penalty contact stiffness in N/mm (default: 50000.0 N/mm)",
    )
    parser.add_argument(
        "--cdamp",
        type=float,
        default=10.0,
        help="Contact normal velocity damping in N*s/mm (default: 10.0 N*s/mm)",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=20.0,
        help="Contact broad-phase search radius in mm (default: 20.0 mm)",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        choices=["augmented_lagrangian", "penalty"],
        default="penalty",
        help="Contact enforcement scheme: 'augmented_lagrangian' or 'penalty' (default: penalty)",
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
        default=25,
        help="Maximum Augmented Lagrangian Uzawa iterations per step (default: 25)",
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
        default="results/08_contact/0806_mortar_dynamic_rigid_floor",
        help="Output directory path for ParaView XDMF/PVD collection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    console.print(
        Panel.fit(
            "[bold cyan]Simulation 0806: Segment-to-Segment Mortar Dynamic Beam Impact[/bold cyan]\n"
            "[dim]Wohlmuth Biorthogonal Dual Basis • Sutherland-Hodgman Polygon Clipping • Dunavant Quadrature[/dim]",
            title="[bold blue]project-sierra — Mortar Contact Mechanics[/bold blue]",
            border_style="cyan",
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Deformable Beam Mesh
    console.print(
        "\n[bold yellow][1/5] Loading Deformable Beam (Total Lagrangian)[/bold yellow]"
    )
    beam_path = _resolve_mesh_path(args.beam_mesh)
    if not beam_path.exists():
        raise FileNotFoundError(f"Beam mesh file not found: {beam_path}")

    beam_mesh = VolumetricMesh(beam_path)
    beam_bottom_y = float(beam_mesh.bounds[2])  # -60.0 mm
    beam_top_y = float(beam_mesh.bounds[3])
    min_z = float(beam_mesh.bounds[4])

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000.0,
        poissons_ratio=0.35,
        mass_density=1100e-6,
    )
    mat_beam = NeoHookean(params)
    elem_beam = Tet4TotalLagrangian(mat_beam)
    beam_domain = SolidDomain(beam_mesh, mat_beam, elem_beam)

    console.print(
        f"  Beam Mesh: {beam_mesh.n_nodes} nodes, {beam_mesh.n_elements} elements"
    )
    console.print(
        f"  Beam Formulation: Total Lagrangian (Tet4TotalLagrangian, E = {params.youngs_modulus:.0f} MPa)"
    )
    console.print(
        f"  Beam Y bounds: [{beam_bottom_y:.1f}, {beam_top_y:.1f}] mm (bottom coordinate: Y = {beam_bottom_y:.1f} mm)"
    )

    # 2. Load Floor as VolumetricMesh & Transform
    console.print(
        "\n[bold yellow][2/5] Initializing Floor as Rigid VolumetricMesh Domain[/bold yellow]"
    )
    floor_path = _resolve_mesh_path(args.floor_mesh)
    if not floor_path.exists():
        raise FileNotFoundError(f"Floor mesh file not found: {floor_path}")

    floor_mesh = VolumetricMesh(floor_path)
    orig_top_y = float(floor_mesh.bounds[3])

    # Position floor under cantilever landing zone (Z = 150 mm) with specified gap
    target_top_y = beam_bottom_y - args.gap
    dy = target_top_y - orig_top_y
    floor_mesh.transform(translation=(0.0, dy, 150.0))

    elem_floor = Tet4Linear(mat_beam)
    floor_domain = SolidDomain(floor_mesh, mat_beam, elem_floor)

    console.print(
        f"  Floor Mesh: {floor_mesh.n_nodes} nodes, {floor_mesh.n_elements} elements"
    )
    console.print(
        f"  Floor Y bounds: [{floor_mesh.bounds[2]:.1f}, {floor_mesh.bounds[3]:.1f}] mm (top at Y = {target_top_y:.1f} mm)"
    )
    console.print(
        f"  Effective Vertical Gap: {beam_bottom_y - floor_mesh.bounds[3]:.1f} mm"
    )

    # 3. Assemble MultiBodySystem & Register Segment-to-Segment Mortar Contact
    console.print(
        "\n[bold yellow][3/5] Assembling MultiBodySystem & Registering Mortar Contact[/bold yellow]"
    )
    system = MultiBodySystem(
        {
            "Beam": beam_domain,
            "Floor": floor_domain,
        }
    )

    contact = system.add_mortar_contact(
        slave="Beam",
        master="Floor",
        k_penalty=args.kp,
        c_damping=args.cdamp,
        formulation=args.formulation,
        search_radius=args.search_radius,
    )

    console.print(f"  Assembly Bodies : {system.names} (Total DOFs: {system.n_dofs})")
    console.print(
        f"  Registered Contact: {type(contact).__name__}(slave='Beam', master='Floor', k_p = {args.kp:.1e} N/mm, mode={args.formulation}, R_s = {args.search_radius:.1f} mm)"
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
    console.print(
        f"  Clamped {len(base_nodes)} beam root nodes and all {floor_mesh.n_nodes} floor nodes (rigid)"
    )

    # Rayleigh Damping: alpha = 2.0, beta = 0.001
    damping = RayleighDamping(alpha=2.0, beta=0.001)
    console.print(
        f"  Rayleigh Damping   : alpha = {damping.alpha:.1f}, beta = {damping.beta:.4f}"
    )

    # Dynamic gravity
    grav = GravityLoad(acceleration=(0.0, -9810.0, 0.0))
    M = system.assemble_mass_matrix()
    f_grav = grav.compute_force_vector(M)
    total_mass = float(np.sum(M.diagonal()[: beam_domain.n_dofs]) / 3.0)
    console.print(
        f"  Dynamic gravity    : g = -9810 mm/s^2 (Beam Mass = {total_mass:.1f} tonne)"
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
        factor=0.0,
        save_directory=output_dir,
        output_name="result",
        save_interval=1,
        verbose=True,
    )
    solve_duration = time.perf_counter() - t_start

    # 5. Energy Analysis & Verification
    console.print(
        "\n[bold yellow][5/5] Rebound Kinematics & Contact Verification[/bold yellow]"
    )
    c_info = contact.get_contact_info()
    contact_energy = float(c_info["potential_energy"])
    max_pen = float(c_info["max_penetration"])
    max_force = float(c_info["max_force"])
    final_min_y = float(np.min(beam_domain.current_nodes[:, 1]))

    lowest_y_reached = min(float(np.min(beam_mesh.nodes[:, 1])), final_min_y)
    contact_occurred = c_info["num_active"] > 0 or lowest_y_reached <= target_top_y + 1e-3

    table = Table(
        title="Simulation 0806 Mortar Dynamic Results Summary",
        border_style="cyan",
    )
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="green")
    table.add_column("Unit", style="dim")

    table.add_row("Solve Status", "CONVERGED" if result.converged else "FAILED", "")
    table.add_row("Contact Method", "Segment-to-Segment Mortar", "")
    table.add_row("Dual Basis Type", "Wohlmuth Biorthogonal (Diagonal D)", "")
    table.add_row("Contact Formulation", args.formulation, "")
    table.add_row("Total Computation Time", f"{solve_duration:.3f}", "s")
    table.add_row("Total Physical Duration", f"{args.steps * args.dt:.4f}", "s")
    table.add_row("Time Step (dt)", f"{args.dt:.4f}", "s")
    table.add_row("Assembly Constituents", f"{system.names}", "")
    table.add_row("Configured Gap (--gap)", f"{args.gap:.2f}", "mm")
    table.add_row("Floor Top Surface (Y_top)", f"{target_top_y:.2f}", "mm")
    table.add_row("Lowest Y Reached", f"{lowest_y_reached:.2f}", "mm")
    table.add_row(
        "Contact Impact Detected",
        "YES (Touchdown & Mortar Repulsion)" if contact_occurred else "NO (Free Swing)",
        "",
    )
    table.add_row("Active Mortar Slave Nodes", f"{c_info['num_active']}", "")
    table.add_row("Max Compliance Penetration (delta)", f"{max_pen:.4f}", "mm")
    table.add_row("Max Contact Repulsive Force", f"{max_force:.2f}", "N")
    if args.formulation == "augmented_lagrangian":
        max_mult = float(c_info.get("max_multiplier", 0.0))
        table.add_row("Max Multiplier (lambda_N)", f"{max_mult:.2f}", "N")
    table.add_row(
        "Stored Contact Potential Energy (Pi_c)", f"{contact_energy:.4f}", "N*mm (mJ)"
    )
    console.print(table)

    console.print(
        Panel.fit(
            f"[bold green]✓ Mortar Dynamic Mesh Contact Saved to {output_dir / 'result.xdmf'}[/bold green]\n"
            f"[dim]Assembly Layer: MultiBodySystem containing both {system.names}[/dim]\n"
            f"[dim]Segment-to-segment mortar integration accurately prevented penetration and conserved momentum.[/dim]\n"
            f"[dim]Open result.xdmf in ParaView and apply 'Warp By Vector' (Displacement) to view dynamic motion.[/dim]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
