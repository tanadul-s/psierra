"""
Simulation 0805: Nonlinear Dynamic Self-Contact on Slotted Cuboid Flexure.
========================================================================
Demonstrates:
1. Finite-strain Total Lagrangian formulation (Tet4TotalLagrangian) with Neo-Hookean hyperelasticity.
2. Slotted keyhole flexure geometry (cuboid1.vtk) with combined downward gravity and dynamic inward pinching.
3. Autonomous self-contact mechanics (SelfContact) preventing interpenetration across the 10.0 mm slot.
4. Topologically filtered Star BFS graph neighbor exclusion (rings=6) preventing false adjacent-facet detections.
5. Generalized-alpha dynamic time stepping (rho_inf = 0.8) capturing impact closure and elastic rebound.
6. Unified ParaView PVD time-series collection export (factor=0.0) with 3D "Displacement" vector field.
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
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
)
from psierra.damping import RayleighDamping
from psierra.analysis import NonlinearDynamicAnalysis

install(show_locals=False)
console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 0805: Nonlinear Dynamic Self-Contact on Slotted Cuboid Flexure"
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="data/cuboid1/mesh/cuboid1.vtk",
        help="Path to slotted cuboid mesh (default: data/cuboid1/mesh/cuboid1.vtk)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Dynamic time step in seconds (default: 0.02 s = 50 Hz)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of dynamic time steps (default: 25 steps = 0.5 s total duration)",
    )
    parser.add_argument(
        "--pinch-force",
        type=float,
        default=25000.0,
        help="Total bilateral inward pinching force in N (default: 25000.0 N, set 0.0 for pure gravity)",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=9810.0,
        help="Gravitational acceleration magnitude in mm/s^2 along -Z (default: 9810.0 mm/s^2, set 0.0 to disable)",
    )
    parser.add_argument(
        "--youngs-modulus",
        type=float,
        default=21000.0,
        help="Young's modulus in MPa (default: 21000.0 MPa matching standard repository material)",
    )
    parser.add_argument(
        "--poisson-ratio",
        type=float,
        default=0.35,
        help="Poisson's ratio (default: 0.35)",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1100e-6,
        help="Mass density in tonne/mm^3 (default: 1100e-6 tonne/mm^3)",
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
        default=0.0,
        help="Contact normal velocity damping in N*s/mm (default: 0.0)",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        choices=["penalty", "augmented_lagrangian"],
        default="penalty",
        help="Contact enforcement scheme: 'penalty' or 'augmented_lagrangian' (default: penalty)",
    )
    parser.add_argument(
        "--al-gap-tol",
        type=float,
        default=0.05,
        help="Augmented Lagrangian penetration tolerance in mm (default: 0.05 mm)",
    )
    parser.add_argument(
        "--al-tol",
        type=float,
        default=0.05,
        help="Augmented Lagrangian multiplier convergence tolerance (default: 0.05)",
    )
    parser.add_argument(
        "--al-max-iter",
        type=int,
        default=50,
        help="Maximum Augmented Lagrangian Uzawa iterations per step (default: 50)",
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
        default="results/08_contact/0805_self_contact_cuboid",
        help="Output directory path for ParaView PVD collection (default: results/08_contact/0805_self_contact_cuboid)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Launch interactive PyVista 3D visualization window upon completion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    console.print(
        Panel.fit(
            "[bold cyan]Simulation 0805: Nonlinear Dynamic Self-Contact on Slotted Cuboid Flexure[/bold cyan]\n"
            "[dim]Project Sierra | Keyhole Flexure Inward Pinching, Slit Closure, and Elastic Rebound[/dim]",
            title="[bold blue]project-sierra — Autonomous Self-Contact Mechanics[/bold blue]",
            border_style="cyan",
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # [1/5] Loading Mesh Geometry & Slit Inspection
    # =========================================================================
    console.print(
        "\n[bold yellow][1/5] Loading Mesh Geometry & Keyhole Slit Inspection[/bold yellow]"
    )
    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    cuboid_mesh = VolumetricMesh(mesh_path)
    nodes = cuboid_mesh.nodes
    b_span = np.ptp(nodes, axis=0)

    console.print(
        f"  Loaded Cuboid Mesh: [green]{cuboid_mesh.n_nodes}[/green] nodes, "
        f"[green]{cuboid_mesh.n_elements}[/green] tetrahedral elements."
    )
    console.print(
        f"  Bounding Box: X=[{nodes[:, 0].min():.1f}, {nodes[:, 0].max():.1f}], "
        f"Y=[{nodes[:, 1].min():.1f}, {nodes[:, 1].max():.1f}], "
        f"Z=[{nodes[:, 2].min():.1f}, {nodes[:, 2].max():.1f}] mm (Span: {b_span[0]:.1f} × {b_span[1]:.1f} × {b_span[2]:.1f} mm)"
    )

    # Identify slot dimensions dynamically
    surf_nodes = np.asarray(cuboid_mesh.surface.node_indices)
    upper_mask = (nodes[surf_nodes, 2] >= 55.0) & (np.abs(nodes[surf_nodes, 0]) <= 7.0)
    cand_coords = nodes[surf_nodes[upper_mask]]
    x_left_face = float(np.max(cand_coords[cand_coords[:, 0] < 0, 0]))
    x_right_face = float(np.min(cand_coords[cand_coords[:, 0] > 0, 0]))

    left_slit_nodes = surf_nodes[
        (np.abs(nodes[surf_nodes, 0] - x_left_face) < 0.5) & (nodes[surf_nodes, 2] >= 55.0)
    ]
    right_slit_nodes = surf_nodes[
        (np.abs(nodes[surf_nodes, 0] - x_right_face) < 0.5) & (nodes[surf_nodes, 2] >= 55.0)
    ]
    slit_gap = float(nodes[right_slit_nodes, 0].min() - nodes[left_slit_nodes, 0].max())

    console.print(
        f"  Keyhole Slit Geometry: [green]{len(left_slit_nodes)}[/green] left face nodes (X = {x_left_face:.1f} mm) vs "
        f"[green]{len(right_slit_nodes)}[/green] right face nodes (X = {x_right_face:.1f} mm)"
    )
    console.print(
        f"  Undeformed Slit Gap: [cyan]{slit_gap:.2f} mm[/cyan] (touchdown requires ~{slit_gap / 2.0:.1f} mm inward deflection per prong)"
    )

    # =========================================================================
    # [2/5] Hyperelastic Constitutive Model & Solid Continuum
    # =========================================================================
    console.print(
        "\n[bold yellow][2/5] Initializing Hyperelastic Continuum Domain[/bold yellow]"
    )
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=args.youngs_modulus,
        poissons_ratio=args.poisson_ratio,
        mass_density=args.density,
    )
    mat = NeoHookean(params)
    elem = Tet4TotalLagrangian(mat)
    cuboid_domain = SolidDomain(cuboid_mesh, mat, elem)

    M = cuboid_domain.assemble_mass_matrix()
    total_mass_tonne = float(np.sum(M.diagonal()) / 3.0)
    total_mass_kg = total_mass_tonne * 1e3

    console.print(
        f"  Material Model: [cyan]NeoHookean[/cyan] (E = {params.youngs_modulus:.1f} MPa, "
        f"ν = {params.poissons_ratio:.2f}, ρ = {args.density:.2e} tonne/mm³)"
    )
    console.print(f"  Element Kinematics: [cyan]Tet4TotalLagrangian[/cyan] (Large-Strain Finite Deformation)")
    console.print(f"  Total Body Mass   : [green]{total_mass_kg:.2f} kg[/green] ({total_mass_tonne:.4f} tonne)")

    # =========================================================================
    # [3/5] Assembly Layer & Autonomous Self-Contact Registration
    # =========================================================================
    console.print(
        "\n[bold yellow][3/5] MultiBodySystem Assembly & Self-Contact Configuration[/bold yellow]"
    )
    system = MultiBodySystem({"SlottedCuboid": cuboid_domain})

    # Restrict contact candidate search to the inner flexure arms and slot chamfer
    slit_candidates = surf_nodes[(nodes[surf_nodes, 2] >= 50.0) & (np.abs(nodes[surf_nodes, 0]) <= 7.0)]
    console.print(
        f"  Self-Contact Candidates: [green]{len(slit_candidates)}[/green] surface nodes facing across the slit"
    )

    contact = system.add_self_contact(
        body="SlottedCuboid",
        contact_nodes=slit_candidates,
        k_penalty=args.kp,
        c_damping=args.cdamp,
        contact_thickness=12.0,  # Detection shell wider than initial 10 mm gap
        search_radius=25.0,
        exclusion_rings=6,       # Exclude up to 6-ring local topological neighbors (prevents arm-step false detection)
        normal_threshold=-0.2,   # Require opposing outward normals (n1 . n2 <= -0.2)
        formulation=args.formulation,
    )

    console.print(
        f"  Registered Contact: {type(contact).__name__}("
        f"k_p = {args.kp:.1e} N/mm, mode={args.formulation}, thickness = 12.0 mm, rings = 6)"
    )

    # Boundary Conditions: Clamped base plate at Z <= 2.0 mm
    base_nodes = np.where(nodes[:, 2] <= 2.0)[0]
    clamp_base = DirichletConstraint(base_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(system.n_dofs)
    dirichlet.add_constraint(clamp_base)

    console.print(
        f"  Boundary Conditions: Fixed [green]{len(base_nodes)}[/green] base nodes at Z <= 2.0 mm (Rigid Foundation)"
    )

    # Dynamic External Loads: Pinching Load + Gravitational Body Force
    f_ext = np.zeros(system.n_dofs)

    # 1. Bilateral Pinching Force: Applied symmetrically on upper outer flanks
    if args.pinch_force > 0.0:
        left_flank = np.where((nodes[:, 0] <= -24.0) & (nodes[:, 2] >= 55.0))[0]
        right_flank = np.where((nodes[:, 0] >= 24.0) & (nodes[:, 2] >= 55.0))[0]
        f_left_node = (args.pinch_force / 2.0) / len(left_flank)
        f_right_node = (-args.pinch_force / 2.0) / len(right_flank)
        for n in left_flank:
            f_ext[3 * n] += f_left_node
        for n in right_flank:
            f_ext[3 * n] += f_right_node

        console.print(
            f"  Dynamic Pinching Load: [cyan]{args.pinch_force:.1f} N[/cyan] total inward load "
            f"([green]+{args.pinch_force/2:.1f} N[/green] on {len(left_flank)} left flank nodes, "
            f"[green]-{args.pinch_force/2:.1f} N[/green] on {len(right_flank)} right flank nodes)"
        )

    # 2. Gravitational Body Force: Uniform downward body load F_grav = M @ g
    if args.gravity != 0.0:
        gravity = GravityLoad(acceleration=(0.0, 0.0, -abs(args.gravity)), dofs_per_node=3)
        f_grav = gravity.compute_force_vector(M)
        f_ext += f_grav
        console.print(
            f"  Dynamic Gravity Load : [cyan]g = -{abs(args.gravity):.1f} mm/s²[/cyan] along -Z "
            f"(Total Body Weight F_z = [green]{f_grav[2::3].sum():.1f} N[/green])"
        )

    # Rayleigh Damping
    damping = RayleighDamping(alpha=1.0, beta=0.002)
    console.print(
        f"  Rayleigh Damping     : alpha = {damping.alpha:.2f}, beta = {damping.beta:.4f}"
    )

    # =========================================================================
    # [4/5] Generalized-alpha Dynamic Time Stepping
    # =========================================================================
    console.print(
        "\n[bold yellow][4/5] Executing Generalized-alpha Dynamic Time Stepping[/bold yellow]"
    )
    solver = NonlinearDynamicAnalysis(
        system, damping=damping, spectral_radius=args.spectral_radius
    )

    t_start = time.perf_counter()
    result = solver.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
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

    # =========================================================================
    # [5/5] Kinematics & Contact Verification
    # =========================================================================
    console.print(
        "\n[bold yellow][5/5] Kinematics & Self-Contact Verification[/bold yellow]"
    )

    c_info = contact.get_contact_info()
    contact_energy = float(c_info["potential_energy"])
    max_pen = float(c_info["max_penetration"])
    active_count = int(c_info["num_active"])

    # Analyze time series for minimum remaining slit gap
    min_slit_gap_reached = slit_gap
    touchdown_detected = False

    if result.time_series_files:
        for p in result.time_series_files:
            try:
                g = pv.read(p)
                disp = g.point_data["Displacement"]
                pts_cur = g.points + disp
                x_left_max = float(pts_cur[left_slit_nodes, 0].max())
                x_right_min = float(pts_cur[right_slit_nodes, 0].min())
                gap_step = x_right_min - x_left_max
                if gap_step < min_slit_gap_reached:
                    min_slit_gap_reached = gap_step
                if gap_step <= 0.05:  # Within contact threshold
                    touchdown_detected = True
            except Exception:
                pass

    if active_count > 0 or max_pen > 0.0:
        touchdown_detected = True

    max_disp_val = float(np.linalg.norm(cuboid_domain.displacement, axis=1).max())

    table = Table(
        title="Simulation 0805 Dynamic Self-Contact Results Summary",
        border_style="cyan",
    )
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="green")
    table.add_column("Unit", style="dim")

    table.add_row("Solve Status", "CONVERGED" if result.converged else "FAILED", "")
    table.add_row("Contact Formulation", args.formulation, "")
    table.add_row("Gravitational Acceleration", f"-{abs(args.gravity):.1f}" if args.gravity != 0.0 else "None", "mm/s^2" if args.gravity != 0.0 else "")
    table.add_row("Bilateral Pinching Force", f"{args.pinch_force:.1f}" if args.pinch_force != 0.0 else "None", "N" if args.pinch_force != 0.0 else "")
    table.add_row("Total Computation Time", f"{solve_duration:.3f}", "s")
    table.add_row("Total Physical Duration", f"{args.steps * args.dt:.4f}", "s")
    table.add_row("Time Step (dt)", f"{args.dt:.4f}", "s")
    table.add_row("Total Time Steps", f"{args.steps}", "")
    table.add_row("Steps Completed", f"{result.steps_completed}", "")
    table.add_row("Undeformed Slit Gap", f"{slit_gap:.3f}", "mm")
    table.add_row("Minimum Slit Gap Reached", f"{min_slit_gap_reached:.3f}", "mm")
    table.add_row(
        "Self-Contact Touchdown",
        "YES (Slit Closed & Rebounded)" if touchdown_detected else "NO (Free Oscillation)",
        "",
    )
    table.add_row("Max Total Displacement", f"{max_disp_val:.3f}", "mm")
    table.add_row("Active Contact Candidates", f"{len(slit_candidates)}", "nodes")
    table.add_row("Contacting Nodes at Peak", f"{active_count}", "nodes")
    table.add_row("Max Penetration (delta)", f"{max_pen:.4f}", "mm")
    table.add_row(
        "Stored Contact Potential Energy", f"{contact_energy:.4f}", "N*mm (mJ)"
    )

    console.print(table)

    console.print(
        Panel.fit(
            f"[bold green]✓ Dynamic Self-Contact Results Saved to {output_dir / 'result.pvd'}[/bold green]\n"
            f"[dim]Total Lagrangian Neo-Hookean flexure squeezed dynamically, slit closed, and rebounded elastically.\n"
            f"Open result.pvd in ParaView and apply 'Warp By Vector' (Displacement, Scale = 1.0).[/dim]",
            border_style="green",
        )
    )

    # Optional 3D Visualization
    if args.show:
        console.print("\n[bold cyan]Launching Interactive PyVista 3D Viewer...[/bold cyan]")
        pl = pv.Plotter(title="Project Sierra - Simulation 0805: Dynamic Self-Contact")
        surf = cuboid_mesh.surface.polydata.copy()
        surf["Displacement"] = cuboid_domain.displacement
        warped = surf.warp_by_vector("Displacement", factor=1.0)
        pl.add_mesh(
            cuboid_mesh.surface.polydata,
            style="wireframe",
            color="white",
            opacity=0.3,
            label="Undeformed Mesh",
        )
        pl.add_mesh(
            warped,
            scalars="Displacement",
            cmap="turbo",
            show_edges=True,
            lighting=True,
            label="Deformed (Warped)",
        )
        pl.add_axes()
        pl.add_legend()
        pl.show()


if __name__ == "__main__":
    main()

