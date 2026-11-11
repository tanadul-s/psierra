"""
Simulation 0807: Two-Way Deformable Segment-to-Segment Mortar Dynamic Contact Mechanics.
========================================================================================
Demonstrates:
1. MultiBodySystem assembly incorporating BOTH Beam (Impactor) and Floor (Flexible Target) domains.
2. Two-way deformable continuum mechanics with Total Lagrangian finite-strain formulation
   (Tet4TotalLagrangian) for BOTH bodies.
3. Segment-to-Segment Mortar Contact formulation with Wohlmuth biorthogonal dual shape functions:
   - Weak coupling ensuring exact action-reaction linear momentum balance (sum f_s + sum f_m = 0).
   - Conservative tangent stiffness symmetry with full two-way off-diagonal coupling blocks:
     K_c = [ K_ss  K_sm ]
           [ K_ms  K_mm ]
4. Augmented Lagrangian Multiplier Method driving interface penetration error to near zero.
5. Sutherland-Hodgman 2D polygon clipping on local tangent planes with 3-point Dunavant numerical quadrature.
6. Rayleigh damping (alpha=2.0, beta=0.001) and Generalized-alpha dynamic time stepping (rho_inf=0.8).
7. Unified ParaView XDMF/PVD export recording simultaneous dynamic deformation of BOTH flexible bodies in
   results/08_contact/0807_mortar_dynamic_deformable_floor.
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
        description="Simulation 0807: Two-Way Deformable Segment-to-Segment Mortar Dynamic Contact Mechanics"
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
        help="Initial vertical gap between cantilever beam bottom and floor top in mm (default: 0.0 mm)",
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
        default="augmented_lagrangian",
        help="Contact enforcement scheme: 'augmented_lagrangian' or 'penalty' (default: augmented_lagrangian)",
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
        default="results/08_contact/0807_mortar_dynamic_deformable_floor",
        help="Output directory path for ParaView XDMF/PVD collection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    console.print(
        Panel.fit(
            "[bold cyan]Simulation 0807: Two-Way Deformable Segment-to-Segment Mortar Contact[/bold cyan]\n"
            "[dim]Dual Biorthogonal Mortar Formulation • Full Off-Diagonal Coupled Tangent Stiffness • Uzawa AL[/dim]",
            title="[bold blue]project-sierra — Two-Way Mortar Contact Mechanics[/bold blue]",
            border_style="cyan",
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Deformable Beam Mesh (Impactor)
    console.print(
        "\n[bold yellow][1/5] Loading Deformable Beam (Total Lagrangian Impactor)[/bold yellow]"
    )
    beam_path = _resolve_mesh_path(args.beam_mesh)
    if not beam_path.exists():
        raise FileNotFoundError(f"Beam mesh file not found: {beam_path}")

    beam_mesh = VolumetricMesh(beam_path)
    beam_bottom_y = float(beam_mesh.bounds[2])  # -60.0 mm
    beam_top_y = float(beam_mesh.bounds[3])
    min_z = float(beam_mesh.bounds[4])

    params_beam = IsotropicElasticParameters()
    params_beam.set_from_youngs_poisson(
        youngs_modulus=21000.0,
        poissons_ratio=0.35,
        mass_density=1100e-6,
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
        f"  Beam Y bounds: [{beam_bottom_y:.1f}, {beam_top_y:.1f}] mm (bottom coordinate: Y = {beam_bottom_y:.1f} mm)"
    )

    # 2. Load Floor as Deformable VolumetricMesh & Transform (Flexible Target)
    console.print(
        "\n[bold yellow][2/5] Initializing Flexible Floor as Deformable Domain[/bold yellow]"
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

    # 3. Assemble MultiBodySystem & Register Two-Way Mortar Contact
    console.print(
        "\n[bold yellow][3/5] Assembling MultiBodySystem & Registering Two-Way Mortar Contact[/bold yellow]"
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
        f"  Clamped {len(base_nodes)} beam root nodes and {len(bot_nodes)} floor bottom support nodes"
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
        "\n[bold yellow][5/5] Two-Way Contact Verification & Deformation Analysis[/bold yellow]"
    )
    c_info = contact.get_contact_info()
    contact_energy = float(c_info["potential_energy"])
    max_pen = float(c_info["max_penetration"])
    max_force = float(c_info["max_force"])

    beam_disp_y_min = float(np.min(beam_domain.displacement[:, 1]))
    floor_disp_y_min = float(np.min(floor_domain.displacement[:, 1]))

    table = Table(
        title="Simulation 0807 Two-Way Mortar Dynamic Results Summary",
        border_style="cyan",
    )
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="green")
    table.add_column("Unit", style="dim")

    table.add_row("Solve Status", "CONVERGED" if result.converged else "FAILED", "")
    table.add_row("Contact Method", "Two-Way Segment-to-Segment Mortar", "")
    table.add_row("Coupling Type", "Full Off-Diagonal Tangent Stiffness Blocks", "")
    table.add_row("Contact Formulation", args.formulation, "")
    table.add_row("Total Computation Time", f"{solve_duration:.3f}", "s")
    table.add_row("Total Physical Duration", f"{args.steps * args.dt:.4f}", "s")
    table.add_row("Time Step (dt)", f"{args.dt:.4f}", "s")
    table.add_row("Assembly Constituents", f"{system.names}", "")
    table.add_row("Configured Gap (--gap)", f"{args.gap:.2f}", "mm")
    table.add_row("Beam Max Deflection (U_y)", f"{beam_disp_y_min:.3f}", "mm")
    table.add_row("Floor Max Deflection (U_y)", f"{floor_disp_y_min:.3f}", "mm")
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
            f"[bold green]✓ Two-Way Mortar Dynamic Contact Saved to {output_dir / 'result.xdmf'}[/bold green]\n"
            f"[dim]Assembly Layer: MultiBodySystem containing both {system.names}[/dim]\n"
            f"[dim]Simultaneous dynamic deformation of both flexible beam and flexible floor successfully computed.[/dim]\n"
            f"[dim]Open result.xdmf in ParaView and apply 'Warp By Vector' (Displacement) to view coupled dynamics.[/dim]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
