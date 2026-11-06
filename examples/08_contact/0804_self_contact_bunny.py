"""
Simulation 0804: Deformable Self-Contact Mechanics on Stanford Bunny with Inclined Gravity Sag.
=============================================================================================
Demonstrates:
1. Volumetric continuum self-contact mechanics driven purely by self-weight gravity loading
   (GravityLoad) on the Stanford Bunny geometry without artificial ear-pushing forces or
   prescribed ear displacements.
2. Compound body inclination (Pitch and Roll):
   - Pitch angle (theta_y): tilts the bunny front up/backwards (positive) so gravity pulls
     ears backwards onto the nape/back, or nose down (negative) so ears drape over the forehead.
   - Roll angle (theta_x): banks the bunny sideways (positive banks right) so gravity has a
     transverse g_x component, forcing the uphill ear to collapse sideways toward the downhill ear.
3. Configurable clamping height threshold (--clamp-height): allows clamping the bottom pedestal (1.0 mm)
   or the lower haunches/torso (~34.0 mm) to keep the body firm while the head and ears flex freely.
4. Topological Star Exclusion (BFS graph expansion, rings=5) eliminating spurious self-detection across
   thin ear membranes and the inter-ear saddle in both undeformed and deformed states.
5. C0-continuous vertex normal interpolation preventing facet edge chatter and lateral discontinuity jumps.
6. Exact Voronoi triangle closest-point projection (Ericson formulation) for master surface facet queries.
7. Strict Newton action-reaction balance (f_s + sum(f_m) = 0) and symmetric tangent stiffness assembly (K_c = K_c^T).
8. Uzawa Augmented Lagrangian multiplier updates driving penetration error to near zero.
9. Support for both Tet4Linear and Tet4TotalLagrangian element formulations with Hookean or NeoHookean material.
10. Incremental Newton-Raphson nonlinear static solve with rich telemetry.
11. ParaView PVD collection export with factor=0.0 and attached 3D 'Displacement' vector array.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import pyvista as pv
from scipy.spatial.distance import cdist
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

from psierra.material import IsotropicElasticParameters, Hookean, NeoHookean
from psierra.element import Tet4Linear, Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import (
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
)
from psierra.boundary.contact import SelfContact
from psierra.analysis import NonlinearStaticAnalysis

install(show_locals=False)
console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 0804: Deformable Self-Contact Mechanics on Stanford Bunny with Inclined Gravity Sag"
    )
    parser.add_argument(
        "--bunny-mesh",
        type=str,
        default="data/bunny1/mesh/bunny1.vtk",
        help="Path to volumetric Stanford Bunny mesh (default: data/bunny1/mesh/bunny1.vtk)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of incremental load steps (default: 5)",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=9810.0,
        help="Gravity acceleration magnitude in mm/s^2 (default: 9810.0 mm/s^2)",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.0,
        help="Pitch inclination angle in degrees: positive tilts nose up / backwards (default: 20.0°)",
    )
    parser.add_argument(
        "--roll",
        type=float,
        default=0.0,
        help="Roll inclination angle in degrees: positive banks right / left ear uphill (default: 15.0°)",
    )
    parser.add_argument(
        "--clamp-height",
        type=float,
        default=34.0,
        help="Height threshold Z_clamp in mm below which nodes are clamped (default: 34.0 mm for lower body/haunches; 1.0 mm for pedestal only)",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1.0e-6,
        help="Material density in tonne/mm^3 (default: 1.0e-6 tonne/mm^3 = 1000 kg/m^3)",
    )
    parser.add_argument(
        "--youngs-modulus",
        type=float,
        default=2.0,
        help="Young's modulus in N/mm^2 (default: 2.0 N/mm^2 = 2.0 MPa for soft elastomer)",
    )
    parser.add_argument(
        "--poisson-ratio",
        type=float,
        default=0.35,
        help="Poisson's ratio (default: 0.35)",
    )
    parser.add_argument(
        "--material",
        type=str,
        choices=["hookean", "neohookean"],
        default="neohookean",
        help="Constitutive material model (default: neohookean)",
    )
    parser.add_argument(
        "--element",
        type=str,
        choices=["linear", "total_lagrangian"],
        default="total_lagrangian",
        help="Kinematic element formulation (default: total_lagrangian)",
    )
    parser.add_argument(
        "--k-penalty",
        type=float,
        default=20.0,
        help="Normal self-contact penalty stiffness in N/mm (default: 20.0)",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        choices=["penalty", "augmented_lagrangian"],
        default="augmented_lagrangian",
        help="Contact enforcement scheme (default: augmented_lagrangian)",
    )
    parser.add_argument(
        "--exclusion-rings",
        type=int,
        default=5,
        help="Topological BFS exclusion ring depth (default: 5)",
    )
    parser.add_argument(
        "--contact-thickness",
        type=float,
        default=5.0,
        help="Maximum contact interaction proximity threshold in mm (default: 5.0)",
    )
    parser.add_argument(
        "--normal-threshold",
        type=float,
        default=-0.2,
        help="Maximum dot product (n_node . n_facet) for opposing contact detection (default: -0.2)",
    )
    parser.add_argument(
        "--all-surface-nodes",
        action="store_true",
        help="Evaluate all unconstrained surface nodes as contact candidates instead of upper head/ear nodes (default: False)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Newton-Raphson relative convergence tolerance (default: 1e-3)",
    )
    parser.add_argument(
        "--al-gap-tol",
        type=float,
        default=1e-3,
        help="Augmented Lagrangian penetration tolerance in mm (default: 1e-3 mm = 1 um)",
    )
    parser.add_argument(
        "--al-tol",
        type=float,
        default=1e-3,
        help="Augmented Lagrangian multiplier convergence tolerance (default: 1e-3)",
    )
    parser.add_argument(
        "--nr-max",
        type=int,
        default=25,
        help="Maximum Newton iterations per load step (default: 25)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="results/08_contact/0804_self_contact_bunny",
        help="Directory to save ParaView .pvd and .vtu time series (default: results/08_contact/0805_self_contact_bunny)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Disable ParaView PVD / VTU file export",
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
            "[bold cyan]Simulation 0804: Deformable Self-Contact Mechanics on Stanford Bunny[/bold cyan]\n"
            "[dim]Project Sierra | Inclined Gravity Ear-Droop & Torso Sag Self-Contact[/dim]",
            border_style="cyan",
        )
    )

    # =========================================================================
    # [1/6] Loading Mesh Geometry & Verifying Base Location
    # =========================================================================
    console.print(
        "[bold yellow][1/6] Loading Mesh Geometry & Verifying Base Location...[/bold yellow]"
    )
    bunny_path = Path(args.bunny_mesh)
    if not bunny_path.exists():
        raise FileNotFoundError(f"Bunny mesh not found: {bunny_path}")

    bunny_mesh = VolumetricMesh(bunny_path)
    nodes = bunny_mesh.nodes

    # Verify base sits exactly on the XY plane (Z = 0.0)
    z_min = float(nodes[:, 2].min())
    if abs(z_min) > 1e-6:
        console.print(
            f"  Transforming mesh: translating by ΔZ = {-z_min:.6f} mm to seat base on XY plane (Z=0.0)..."
        )
        bunny_mesh.transform(translation=(0.0, 0.0, -z_min))
        nodes = bunny_mesh.nodes

    b_span = np.ptp(nodes, axis=0)
    console.print(
        f"  Loaded Stanford Bunny mesh: [green]{bunny_mesh.n_nodes}[/green] nodes, "
        f"[green]{bunny_mesh.n_elements}[/green] elements, "
        f"[green]{len(bunny_mesh.surface.node_indices)}[/green] surface nodes."
    )
    console.print(
        f"  Bounding Box: X=[{nodes[:, 0].min():.2f}, {nodes[:, 0].max():.2f}], "
        f"Y=[{nodes[:, 1].min():.2f}, {nodes[:, 1].max():.2f}], "
        f"Z=[{nodes[:, 2].min():.2f}, {nodes[:, 2].max():.2f}] mm (Height: {b_span[2]:.2f} mm)"
    )

    # =========================================================================
    # [2/6] Configuring Material & Element Formulation
    # =========================================================================
    console.print(
        "[bold yellow][2/6] Configuring Material & Element Formulation...[/bold yellow]"
    )
    mat_params = IsotropicElasticParameters()
    mat_params.set_from_youngs_poisson(
        args.youngs_modulus, args.poisson_ratio, args.density
    )
    if args.material.lower() == "neohookean":
        material = NeoHookean(mat_params)
        mat_name = "NeoHookean"
    else:
        material = Hookean(mat_params)
        mat_name = "Hookean"

    if args.element.lower() == "total_lagrangian":
        element = Tet4TotalLagrangian(material)
        elem_name = "Tet4TotalLagrangian"
    else:
        element = Tet4Linear(material)
        elem_name = "Tet4Linear"

    console.print(
        f"  Material: [cyan]{mat_name}[/cyan] (E = {args.youngs_modulus:.2f} N/mm², "
        f"ν = {args.poisson_ratio:.2f}, ρ = {args.density:.2e} tonne/mm³)"
    )
    console.print(f"  Element:  [cyan]{elem_name}[/cyan]")

    # =========================================================================
    # [3/6] Setting Up Physical Domain & Inclined Gravity Body Load
    # =========================================================================
    console.print(
        "[bold yellow][3/6] Setting Up Physical Domain & Inclined Gravity Body Load...[/bold yellow]"
    )
    domain = SolidDomain(bunny_mesh, material, element)
    console.print(
        f"  Total Domain DOFs: [cyan]{domain.n_dofs}[/cyan] ({bunny_mesh.n_nodes} nodes × 3)"
    )

    # Compute mass matrix and 3D inclined gravity vector
    M = domain.assemble_mass_matrix()
    total_mass_tonne = float(np.sum(M.diagonal()) / 3.0)
    total_mass_kg = total_mass_tonne * 1e3

    pitch_rad = np.radians(args.pitch)
    roll_rad = np.radians(args.roll)
    gx = args.gravity * np.sin(roll_rad)
    gy = args.gravity * np.sin(pitch_rad)
    gz = -args.gravity * np.cos(pitch_rad) * np.cos(roll_rad)
    gravity_vec = (gx, gy, gz)

    gravity = GravityLoad(acceleration=gravity_vec)
    f_ext = gravity.compute_force_vector(M)
    total_weight_N = float(
        np.linalg.norm([np.sum(f_ext[0::3]), np.sum(f_ext[1::3]), np.sum(f_ext[2::3])])
    )

    console.print(
        f"  Total Bunny Mass : [green]{total_mass_kg:.2f} kg[/green] ({total_mass_tonne:.4f} tonne)"
    )
    console.print(
        f"  Gravity Vector   : [cyan]g = [{gx:+.1f}, {gy:+.1f}, {gz:+.1f}] mm/s²[/cyan] "
        f"(|g| = {args.gravity:.1f} mm/s², Pitch = {args.pitch:+.1f}°, Roll = {args.roll:+.1f}°)"
    )
    console.print(f"  Resultant Weight : [green]{total_weight_N:.1f} N[/green]")

    # =========================================================================
    # [4/6] Applying Boundary Conditions & Self-Contact Constraints
    # =========================================================================
    console.print(
        "[bold yellow][4/6] Applying Boundary Conditions & Self-Contact Constraints...[/bold yellow]"
    )

    # 1. Base clamp: Fix all nodes below the specified height threshold
    base_nodes = np.where(nodes[:, 2] <= args.clamp_height)[0]
    base_center = nodes[base_nodes].mean(axis=0)
    console.print(
        f"  Fixed Clamped Region: [green]{len(base_nodes)}[/green] nodes clamped at Z <= {args.clamp_height:.1f} mm "
        f"(Footprint: X=[{nodes[base_nodes, 0].min():.1f}, {nodes[base_nodes, 0].max():.1f}], "
        f"Y=[{nodes[base_nodes, 1].min():.1f}, {nodes[base_nodes, 1].max():.1f}], "
        f"Center=[{base_center[0]:.1f}, {base_center[1]:.1f}])"
    )

    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(
        DirichletConstraint(base_nodes, components=(0, 1, 2), value=0.0)
    )

    # 2. Candidate self-contact slave nodes
    surf_nodes = np.array(list(bunny_mesh.surface.node_indices))
    if args.all_surface_nodes:
        cand_nodes = surf_nodes[nodes[surf_nodes, 2] > args.clamp_height]
        cand_desc = (
            f"all {len(cand_nodes)} free surface nodes (Z > {args.clamp_height:.1f} mm)"
        )
    else:
        z_thresh = max(50.0, args.clamp_height)
        cand_nodes = surf_nodes[nodes[surf_nodes, 2] > z_thresh]
        cand_desc = (
            f"{len(cand_nodes)} upper head & ear surface nodes (Z > {z_thresh:.1f} mm)"
        )

    console.print(
        f"  Contact Candidate Nodes: [green]{len(cand_nodes)}[/green] ({cand_desc})"
    )
    console.print(
        f"  Master Facets: [green]{bunny_mesh.surface.n_elements}[/green] surface triangles across entire bunny body"
    )

    # 3. Configure SelfContact boundary formulation
    contact = SelfContact(
        domain=domain,
        contact_nodes=cand_nodes,
        k_penalty=args.k_penalty,
        exclusion_rings=args.exclusion_rings,
        contact_thickness=args.contact_thickness,
        normal_threshold=args.normal_threshold,
        formulation=args.formulation,
    )
    console.print(
        f"  SelfContact Formulation: [cyan]{args.formulation.upper()}[/cyan] "
        f"(k_p = {args.k_penalty:.1e} N/mm, BFS rings = {args.exclusion_rings}, thickness = {args.contact_thickness:.1f} mm)"
    )

    # Initial ear-to-ear minimum distance
    e1_mask = (nodes[:, 2] > 65.0) & (nodes[:, 0] < -10.0)
    e2_mask = (nodes[:, 2] > 65.0) & (nodes[:, 0] >= -10.0)
    d_ee_orig = float(np.min(cdist(nodes[e1_mask], nodes[e2_mask])))
    console.print(
        f"  Undeformed Ear1-to-Ear2 Distance: [cyan]{d_ee_orig:.2f} mm[/cyan]"
    )

    # =========================================================================
    # [5/6] Executing Simulation Solver (Incremental Newton-Raphson)
    # =========================================================================
    console.print(
        "[bold yellow][5/6] Executing Simulation Solver (Incremental Newton-Raphson)...[/bold yellow]"
    )
    save_dir = None if args.no_export else Path(args.export_dir)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        console.print(
            f"  Export directory: [dim]{save_dir}[/dim] (factor=0.0, WarpByVector compliant)"
        )

    solver = NonlinearStaticAnalysis(domain)

    t_start = time.time()
    result = solver.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        contact=contact,
        n_steps=args.steps,
        nr_iterations=args.nr_max,
        nr_tol=args.tol,
        al_gap_tol=args.al_gap_tol,
        al_tol=args.al_tol,
        save_directory=save_dir,
        output_name="result",
        factor=0.0,  # Strictly factor=0.0 for ParaView WarpByVector
        verbose=True,
    )
    t_solve = time.time() - t_start

    # =========================================================================
    # [6/6] Verifying Results & Post-Processing
    # =========================================================================
    console.print(
        "\n[bold yellow][6/6] Verifying Results & Post-Processing...[/bold yellow]"
    )

    # Extract final deformation and contact metrics
    u_final = domain.u.reshape(-1, 3)
    curr_nodes = domain.current_nodes
    u_magnitudes = np.linalg.norm(u_final, axis=1)
    max_u = float(np.max(u_magnitudes))
    z_disp = u_final[:, 2]
    max_sag = float(np.min(z_disp))

    # Regional sag metrics
    belly_mask = (nodes[:, 2] >= 20.0) & (nodes[:, 2] < 50.0)
    torso_sag = float(np.mean(z_disp[belly_mask])) if np.any(belly_mask) else 0.0
    head_mask = (nodes[:, 2] >= 55.0) & (nodes[:, 2] < 70.0)
    head_sag = float(np.mean(z_disp[head_mask])) if np.any(head_mask) else 0.0
    ears_mask = nodes[:, 2] >= 70.0
    ears_sag = float(np.mean(z_disp[ears_mask])) if np.any(ears_mask) else 0.0

    # Deformed Ear-to-Ear distance
    d_ee_def = float(np.min(cdist(curr_nodes[e1_mask], curr_nodes[e2_mask])))

    contact_info = contact.get_contact_info()
    f_contact = contact.evaluate_forces(n_dofs=domain.n_dofs)
    total_contact_force = float(np.sum(np.abs(f_contact)))

    table = Table(
        title="Simulation 0804: Stanford Bunny Inclined Gravity Sag & Self-Contact Summary",
        border_style="cyan",
    )
    table.add_column("Parameter / Metric", style="bold white")
    table.add_column("Value", style="green")

    table.add_row("Mesh Elements", f"{bunny_mesh.n_elements} linear tetrahedra (Tet4)")
    table.add_row("Total Degrees of Freedom", f"{domain.n_dofs:,}")
    table.add_row(
        "Free DOFs / Prescribed DOFs",
        f"{len(dirichlet.free_dofs):,} / {len(dirichlet.prescribed_dofs):,}",
    )
    table.add_row("Kinematic Formulation", f"{elem_name}")
    table.add_row("Material Model", f"{mat_name}")
    table.add_row(
        "Young's Modulus (E)",
        f"{args.youngs_modulus:.2f} N/mm² ({args.youngs_modulus:.2f} MPa)",
    )
    table.add_row(
        "Clamped Boundary Threshold",
        f"Z <= {args.clamp_height:.1f} mm ({len(base_nodes)} nodes)",
    )
    table.add_row("Inclination Pitch (Nose Up/Down)", f"{args.pitch:+.1f}°")
    table.add_row("Inclination Roll (Bank Right/Left)", f"{args.roll:+.1f}°")
    table.add_row(
        "Gravity Acceleration Vector", f"[{gx:+.1f}, {gy:+.1f}, {gz:+.1f}] mm/s²"
    )
    table.add_row(
        "Bunny Mass & Self-Weight", f"{total_mass_kg:.2f} kg ({total_weight_N:.1f} N)"
    )
    table.add_row("Contact Formulation", f"{args.formulation.upper()}")
    table.add_row("Penalty Stiffness (k_p)", f"{args.k_penalty:.1e} N/mm")
    table.add_row("Topological BFS Rings", f"{args.exclusion_rings}")
    table.add_row("Load Steps Completed", f"{result.steps_completed} / {args.steps}")
    table.add_row("Total Newton Iterations", f"{sum(result.iteration_history)}")
    table.add_row(
        "Convergence Status",
        (
            "[bold green]CONVERGED[/bold green]"
            if result.converged
            else "[bold red]FAILED[/bold red]"
        ),
    )
    table.add_row("Total Solve Time", f"{t_solve:.2f} s")
    table.add_row("Max Resultant Displacement", f"{max_u:.3f} mm")
    table.add_row("Peak Downward Sag (min ΔZ)", f"{max_sag:.3f} mm")
    table.add_row("Mean Ear Droop (Z >= 70 mm)", f"{ears_sag:.3f} mm")
    table.add_row("Torso / Belly Mean Sag", f"{torso_sag:.3f} mm")
    table.add_row("Head / Skull Mean Sag", f"{head_sag:.3f} mm")
    table.add_row(
        "Ear1-to-Ear2 Distance",
        f"{d_ee_orig:.2f} mm -> {d_ee_def:.2f} mm (Δ = {d_ee_def - d_ee_orig:+.2f} mm)",
    )
    table.add_row("Active Self-Contact Nodes", f"{contact_info['num_active']}")
    table.add_row(
        "Interface Penetration Depth", f"{contact_info['max_penetration']:.4f} mm"
    )
    table.add_row("Peak Uzawa Multiplier", f"{contact_info['max_multiplier']:.2f} N")
    table.add_row(
        "Stored Contact Potential Energy",
        f"{contact_info['potential_energy']:.2e} N·mm",
    )

    console.print(table)

    # Print active contact node kinematics
    if contact_info["num_active"] > 0:
        act_nodes = contact_info["active_nodes"]
        act_table = Table(
            title="Active Self-Contact Node Kinematics (Ear-to-Body Contact Engagement)",
            border_style="magenta",
        )
        act_table.add_column("Node ID", style="bold white")
        act_table.add_column("Undeformed [X, Y, Z] (mm)", style="dim")
        act_table.add_column("Deformed [X, Y, Z] (mm)", style="green")
        act_table.add_column("Total Droop ΔZ (mm)", style="cyan")
        act_table.add_column("Resultant |u| (mm)", style="yellow")

        for n_id in act_nodes:
            p0 = nodes[n_id]
            p1 = curr_nodes[n_id]
            dz = p1[2] - p0[2]
            mag = float(np.linalg.norm(p1 - p0))
            act_table.add_row(
                f"{n_id}",
                f"[{p0[0]:6.1f}, {p0[1]:6.1f}, {p0[2]:6.1f}]",
                f"[{p1[0]:6.1f}, {p1[1]:6.1f}, {p1[2]:6.1f}]",
                f"{dz:+.2f} mm",
                f"{mag:.2f} mm",
            )
        console.print(act_table)

    # Detailed height profile
    console.print("\n[bold cyan]Height-wise Sag & Deformation Profile:[/bold cyan]")
    profile_table = Table(border_style="dim")
    profile_table.add_column("Region", style="bold white")
    profile_table.add_column("Z Range (mm)", style="dim")
    profile_table.add_column("Mean |u| (mm)", style="green")
    profile_table.add_column("Mean Z-Sag (mm)", style="cyan")
    profile_table.add_column("Max |u| (mm)", style="yellow")

    for z_low, z_high, label in [
        (0.0, 5.0, "Base Pedestal"),
        (5.0, 25.0, "Lower Haunches"),
        (25.0, 45.0, "Mid Torso / Belly"),
        (45.0, 60.0, "Upper Chest / Neck"),
        (60.0, 70.0, "Skull / Saddle"),
        (70.0, 75.0, "Lower Ears"),
        (75.0, 85.0, "Upper Ears & Tips"),
    ]:
        mask = (nodes[:, 2] >= z_low) & (nodes[:, 2] < z_high)
        if np.any(mask):
            profile_table.add_row(
                label,
                f"[{z_low:4.1f}, {z_high:4.1f}]",
                f"{np.mean(u_magnitudes[mask]):.3f}",
                f"{np.mean(z_disp[mask]):.3f}",
                f"{np.max(u_magnitudes[mask]):.3f}",
            )
    console.print(profile_table)

    if save_dir is not None:
        pvd_path = save_dir / "result.pvd"
        console.print(
            f"\n[bold green]✓ ParaView PVD collection successfully exported to:[/bold green] [cyan]{pvd_path}[/cyan]"
        )
        console.print(
            "[dim]  To visualize in ParaView: Open .pvd -> Apply 'Warp By Vector' (Vectors: Displacement, Scale: 1.0).[/dim]"
        )

    # Optional 3D Visualization
    if args.show:
        console.print(
            "\n[bold cyan]Opening Interactive PyVista 3D Visualizer...[/bold cyan]"
        )
        p = pv.Plotter(
            title="Stanford Bunny Gravity Sag & Self-Contact - Final Deformed State"
        )
        p.set_background("#1e1e24")

        # Deformed mesh with Warp By Vector
        vmesh_pv = domain.mesh.grid.copy()
        vmesh_pv.point_data["Displacement"] = u_final
        vmesh_pv.point_data["Disp_Mag"] = u_magnitudes
        vmesh_pv.point_data["Z_Sag"] = z_disp
        vmesh_pv.set_active_scalars("Z_Sag")
        warped = vmesh_pv.warp_by_vector("Displacement", factor=1.0)

        # Wireframe of undeformed reference state
        p.add_mesh(
            domain.mesh.grid,
            color="gray",
            opacity=0.25,
            style="wireframe",
            label="Undeformed Bunny",
        )

        # Shaded deformed solid colored by Z Sag
        p.add_mesh(
            warped,
            cmap="coolwarm",
            show_edges=True,
            edge_color="black",
            opacity=0.9,
            scalar_bar_args={"title": "Vertical Sag ΔZ (mm)"},
        )

        # Highlight active self-contact nodes
        if contact_info["num_active"] > 0:
            act_nodes = contact_info["active_nodes"]
            act_pts = domain.current_nodes[act_nodes]
            p.add_points(
                act_pts,
                color="red",
                point_size=12.0,
                render_points_as_spheres=True,
                label="Self-Contact Nodes",
            )

        p.add_legend()
        p.add_axes()
        p.camera_position = "iso"
        p.show()


if __name__ == "__main__":
    main()
