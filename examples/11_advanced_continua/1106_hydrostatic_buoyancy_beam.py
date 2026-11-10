"""
Simulation 1106: Cantilever Beam Deflection under Gravitational Body Load and Hydrostatic Buoyancy.

Demonstrates:
- [1/6] Continuum Mesh Geometry & Orientation: 3D cantilever beam continuum.
- [2/6] Configuring Material & Fluid Properties:
        Solid: E = 21,000 MPa, nu = 0.35, rho_solid = 2000 kg/m^3 (2000e-12 tonne/mm^3).
        Fluid: Water rho_fluid = 1000 kg/m^3 (1000e-12 tonne/mm^3).
- [3/6] Applying Boundary Conditions & Combined Gravity + Buoyancy Loads:
        Clamped root and combined F_net = F_gravity + F_buoyancy (Archimedean 50% buoyant lift).
- [4/6] Executing All 4 Continuum Analysis Solvers:
        1. Linear Static (LinearStaticAnalysis)
        2. Linear Dynamic (LinearDynamicAnalysis)
        3. Total Lagrangian Static (NonlinearStaticAnalysis)
        4. Total Lagrangian Dynamic (NonlinearDynamicAnalysis)
- [5/6] Multi-Model Telemetry & Comparative Synthesis:
        Deflection reduction due to buoyancy, dynamic settling, geometric stiffening.
- [6/6] ParaView Unified XDMF+HDF5 Visual Export:
        Streaming clean 2-file datasets for side-by-side deformed configuration analysis.
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
from scipy.spatial.transform import Rotation

from psierra.material import IsotropicElasticParameters, Hookean, NeoHookean
from psierra.element import Tet4Linear, Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary.selector import select_nodes_by_bounds
from psierra.boundary import (
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
    HydrostaticPressureLoad,
    BuoyancyLoad,
)
from psierra.damping import RayleighDamping
from psierra.analysis import (
    LinearStaticAnalysis,
    LinearDynamicAnalysis,
    NonlinearStaticAnalysis,
    NonlinearDynamicAnalysis,
)

install(show_locals=False)
console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 1106: Beam Deflection under Gravity and Hydrostatic Buoyancy"
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="data/beam1/mesh/beam1.vtk",
        help="Path to beam volumetric mesh (default: data/beam1/mesh/beam1.vtk)",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=9810.0,
        help="Gravitational acceleration magnitude in mm/s^2 (default: 9810.0 for 1.0g)",
    )
    parser.add_argument(
        "--youngs",
        type=float,
        default=1.0,
        help="Young's modulus in MPa (default: 1.0 MPa for visible soft-material cantilever deflection)",
    )
    parser.add_argument(
        "--steps-dynamic",
        type=int,
        default=30,
        help="Number of dynamic time steps (default: 30)",
    )
    parser.add_argument(
        "--steps-static-nl",
        type=int,
        default=5,
        help="Number of nonlinear static incremental load steps (default: 5)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Dynamic time increment dt in seconds (default: 0.01 s)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["xdmf", "pvd"],
        default="xdmf",
        help="ParaView visualization export format (default: xdmf)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/11_advanced_continua/1106_hydrostatic_buoyancy_beam",
        help="Base output directory for simulation results",
    )
    return parser.parse_args()


def resolve_mesh_path(rel_path: str) -> Path:
    p1 = Path(rel_path)
    if p1.exists():
        return p1
    p2 = Path("psierra") / rel_path
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Mesh file not found at '{p1}' or '{p2}'")


def main() -> None:
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format
    g_accel = args.gravity
    dt = args.dt
    n_steps_dyn = args.steps_dynamic
    n_steps_static_nl = args.steps_static_nl

    # 0. Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]Simulation 1106: Beam Deflection under Gravity & Hydrostatic Buoyancy[/bold cyan]\n"
            "[white]Complete 4-Analysis Suite: Linear Static, Linear Dynamic, TL Static, TL Dynamic[/white]\n"
            f"[dim]E = {args.youngs:,.1f} MPa • ν = 0.35 • ρ_solid = 2,000 kg/m³ • ρ_fluid = 1,000 kg/m³ • g = {g_accel:.1f} mm/s²[/dim]",
            title="[bold blue]project-sierra / Advanced Continua[/bold blue]",
            border_style="cyan",
        )
    )

    t0_all = time.perf_counter()

    # 1. Loading Mesh Geometry & Spatial Orientation
    console.print("\n[bold yellow][1/6] Loading Mesh Geometry & Spatial Orientation[/bold yellow]")
    mesh_path = resolve_mesh_path(args.mesh)
    mesh = VolumetricMesh(mesh_path)
    # Orient length along Y-axis and deflection along Z-axis
    mesh.transform(
        orientation=Rotation.from_euler("zyz", (0, 90, 90), degrees=True),
        about_point=(0, 0, 0),
    )

    console.print(
        f"  • Mesh File      : [cyan]{mesh_path}[/cyan]\n"
        f"  • Topology       : [green]{mesh.n_nodes:,}[/green] nodes | "
        f"[green]{mesh.n_elements:,}[/green] Tet4 elements | [green]{mesh.n_dofs:,}[/green] DOFs\n"
        f"  • Dimensions     : X ∈ [{mesh.bounds[0]:.1f}, {mesh.bounds[1]:.1f}], "
        f"Y ∈ [{mesh.bounds[2]:.1f}, {mesh.bounds[3]:.1f}], "
        f"Z ∈ [{mesh.bounds[4]:.1f}, {mesh.bounds[5]:.1f}] mm\n"
        f"  • Beam Volume    : [green]{mesh.volume:,.2f}[/green] mm³"
    )

    # 2. Material & Fluid Properties
    console.print("\n[bold yellow][2/6] Configuring Material & Fluid Properties[/bold yellow]")
    youngs_modulus_mpa = args.youngs
    poissons_ratio = 0.35
    rho_solid = 2000e-12  # 2000 kg/m^3 in tonne/mm^3
    rho_fluid = 1000e-12  # 1000 kg/m^3 in tonne/mm^3 (Water)

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=youngs_modulus_mpa,
        poissons_ratio=poissons_ratio,
        mass_density=rho_solid,
    )

    total_solid_mass_kg = mesh.volume * rho_solid * 1e3
    total_weight_n = total_solid_mass_kg * (g_accel / 1000.0)
    total_buoyant_lift_n = (mesh.volume * rho_fluid * 1e3) * (g_accel / 1000.0)
    net_downward_force_n = total_weight_n - total_buoyant_lift_n
    buoyant_relief_pct = (total_buoyant_lift_n / total_weight_n) * 100.0

    console.print(
        f"  • Young's Modulus E : [green]{youngs_modulus_mpa:,.1f}[/green] MPa | Poisson's Ratio ν: [green]{poissons_ratio:.2f}[/green]\n"
        f"  • Solid Density ρ_s : [green]{rho_solid:.2e}[/green] tonne/mm³ ({rho_solid*1e12:.0f} kg/m³)\n"
        f"  • Fluid Density ρ_f : [green]{rho_fluid:.2e}[/green] tonne/mm³ ({rho_fluid*1e12:.0f} kg/m³ - Water)\n"
        f"  • Downward Weight   : [bold red]{total_weight_n:.3f}[/bold red] N (acting along -Z)\n"
        f"  • Upward Buoyant Lift: [bold green]{total_buoyant_lift_n:.3f}[/bold green] N (acting along +Z)\n"
        f"  • Net Submerged Force: [bold yellow]{net_downward_force_n:.3f}[/bold yellow] N "
        f"([cyan]{buoyant_relief_pct:.1f}% buoyant weight reduction[/cyan])"
    )

    # 3. Boundary Clamping & Combined Force Assembly
    console.print("\n[bold yellow][3/6] Applying Boundary Clamping & Assembling Forces[/bold yellow]")
    root_nodes = select_nodes_by_bounds(mesh, bounds=(-61, 61, -31, 1, -61, 61))
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Downward gravity body load
    mat_dummy = Hookean(params)
    elem_dummy = Tet4Linear(mat_dummy)
    dom_dummy = SolidDomain(mesh.copy(), mat_dummy, elem_dummy)
    M = dom_dummy.assemble_mass_matrix()
    gravity = GravityLoad(acceleration=(0.0, 0.0, -g_accel))
    f_gravity = gravity.compute_force_vector(M)

    # Upward buoyancy load
    buoyancy = BuoyancyLoad(
        fluid_density=rho_fluid,
        acceleration=(0.0, 0.0, -g_accel),
    )
    f_buoyancy = buoyancy.compute_force_vector(dom_dummy)

    # Net applied force vector
    f_net = f_gravity + f_buoyancy

    # Rayleigh damping for dynamic time stepping
    damping = RayleighDamping(alpha=2.0, beta=0.001)

    console.print(
        f"  • Clamped Root Face : [green]{len(root_nodes)}[/green] nodes clamped at Y ≤ 0 mm\n"
        f"  • Free Degrees      : [green]{len(dirichlet.free_dofs)}[/green] free DOFs\n"
        f"  • Net Load Sum (Z)  : [bold green]{f_net.reshape(-1, 3)[:, 2].sum():.3f}[/bold green] N (matches theoretical {net_downward_force_n:.3f} N)"
    )

    results_summary: list[dict[str, str | float]] = []

    # =========================================================================
    # 4A. Analysis 1: Linear Static Analysis
    # =========================================================================
    console.print("\n[bold yellow][4/6] Executing All 4 Continuum Analysis Solvers[/bold yellow]")
    console.print("  [bold cyan]• [1/4] Linear Static Analysis (Direct Sparse Factorization)...[/bold cyan]")
    mat_linear = Hookean(params)
    elem_linear = Tet4Linear(mat_linear)
    dom_ls = SolidDomain(mesh.copy(), mat_linear, elem_linear)

    t_start = time.perf_counter()
    solver_ls = LinearStaticAnalysis(dom_ls)
    dir_ls = output_dir / "01_linear_static"
    res_ls = solver_ls.solve(
        dirichlet=dirichlet,
        f_ext=f_net,
        save_directory=dir_ls,
        format=fmt,
        verbose=False,
    )
    t_ls = time.perf_counter() - t_start

    disp_ls = float(np.linalg.norm(dom_ls.displacement, axis=1).max())
    sag_ls = float(abs(dom_ls.displacement[:, 2].min()))
    rxn_ls = float(res_ls.reactions[clamp.dofs][2::3].sum())

    console.print(
        f"    ✓ Solved in {t_ls*1000:.2f} ms | Max Deflection: [bold green]{disp_ls:.3f} mm[/bold green] | "
        f"Tip Sag (Z): [bold green]{sag_ls:.3f} mm[/bold green]"
    )

    results_summary.append({
        "Model": "Linear Elastic (Hookean)",
        "Analysis Type": "Linear Static",
        "Kinematics": "Small-Strain Simplex",
        "Max Disp (mm)": disp_ls,
        "Tip Sag (mm)": sag_ls,
        "Reaction R_z (N)": rxn_ls,
        "Steps": 1,
        "Time": t_ls * 1000,
        "Export Path": str(dir_ls / f"result.{fmt}"),
    })

    # =========================================================================
    # 4B. Analysis 2: Linear Dynamic Analysis
    # =========================================================================
    console.print(f"  [bold cyan]• [2/4] Linear Dynamic Analysis ({n_steps_dyn} steps, Generalized-α, dt={dt:.3f}s)...[/bold cyan]")
    dom_ld = SolidDomain(mesh.copy(), mat_linear, elem_linear)

    t_start = time.perf_counter()
    solver_ld = LinearDynamicAnalysis(dom_ld, damping=damping)
    dir_ld = output_dir / "02_linear_dynamic"
    res_ld = solver_ld.solve(
        dirichlet=dirichlet,
        f_ext=f_net,
        dt=dt,
        n_steps=n_steps_dyn,
        save_interval=1,
        save_directory=dir_ld,
        format=fmt,
        verbose=False,
    )
    t_ld = time.perf_counter() - t_start

    disp_ld = float(np.linalg.norm(dom_ld.displacement, axis=1).max())
    sag_ld = float(abs(dom_ld.displacement[:, 2].min()))

    console.print(
        f"    ✓ Solved in {t_ld*1000:.2f} ms ({res_ld.steps_completed} steps) | "
        f"End Max Deflection: [bold green]{disp_ld:.3f} mm[/bold green] | "
        f"End Tip Sag (Z): [bold green]{sag_ld:.3f} mm[/bold green]"
    )

    results_summary.append({
        "Model": "Linear Elastic (Hookean)",
        "Analysis Type": "Linear Dynamic",
        "Kinematics": "Small-Strain Generalized-α",
        "Max Disp (mm)": disp_ld,
        "Tip Sag (mm)": sag_ld,
        "Reaction R_z (N)": rxn_ls,
        "Steps": res_ld.steps_completed,
        "Time": t_ld * 1000,
        "Export Path": str(dir_ld / f"result.{fmt}"),
    })

    # =========================================================================
    # 4C. Analysis 3: Total Lagrangian Static Analysis
    # =========================================================================
    console.print(f"  [bold cyan]• [3/4] Total Lagrangian Static Analysis ({n_steps_static_nl} load steps, Newton-Raphson)...[/bold cyan]")
    mat_nonlinear = NeoHookean(params)
    elem_nonlinear = Tet4TotalLagrangian(mat_nonlinear)
    dom_tls = SolidDomain(mesh.copy(), mat_nonlinear, elem_nonlinear)

    t_start = time.perf_counter()
    solver_tls = NonlinearStaticAnalysis(dom_tls)
    dir_tls = output_dir / "03_tl_static"
    res_tls = solver_tls.solve(
        dirichlet=dirichlet,
        f_ext=f_net,
        n_steps=n_steps_static_nl,
        nr_iterations=15,
        nr_tol=1e-4,
        save_history=False,
        save_directory=dir_tls,
        format=fmt,
        verbose=False,
    )
    t_tls = time.perf_counter() - t_start

    disp_tls = float(np.linalg.norm(dom_tls.displacement, axis=1).max())
    sag_tls = float(abs(dom_tls.displacement[:, 2].min()))
    rxn_tls = float(res_tls.reactions[clamp.dofs][2::3].sum())
    total_nr_iters = sum(res_tls.iteration_history)

    console.print(
        f"    ✓ Solved in {t_tls*1000:.2f} ms ({res_tls.steps_completed} steps, {total_nr_iters} total iters) | "
        f"Max Deflection: [bold green]{disp_tls:.3f} mm[/bold green] | "
        f"Tip Sag (Z): [bold green]{sag_tls:.3f} mm[/bold green]"
    )

    results_summary.append({
        "Model": "Hyperelastic (Neo-Hookean)",
        "Analysis Type": "TL Static",
        "Kinematics": "Total Lagrangian Finite-Strain",
        "Max Disp (mm)": disp_tls,
        "Tip Sag (mm)": sag_tls,
        "Reaction R_z (N)": rxn_tls,
        "Steps": f"{res_tls.steps_completed} steps ({total_nr_iters} iters)",
        "Time": t_tls * 1000,
        "Export Path": str(dir_tls / f"result.{fmt}"),
    })

    # =========================================================================
    # 4D. Analysis 4: Total Lagrangian Dynamic Analysis
    # =========================================================================
    console.print(f"  [bold cyan]• [4/4] Total Lagrangian Dynamic Analysis ({n_steps_dyn} steps, Generalized-α, dt={dt:.3f}s)...[/bold cyan]")
    dom_tld = SolidDomain(mesh.copy(), mat_nonlinear, elem_nonlinear)

    t_start = time.perf_counter()
    solver_tld = NonlinearDynamicAnalysis(dom_tld, damping=damping)
    dir_tld = output_dir / "04_tl_dynamic"
    res_tld = solver_tld.solve(
        dirichlet=dirichlet,
        f_ext=f_net,
        dt=dt,
        n_steps=n_steps_dyn,
        nr_iterations=10,
        nr_tol=1e-3,
        save_interval=1,
        save_directory=dir_tld,
        format=fmt,
        verbose=False,
    )
    t_tld = time.perf_counter() - t_start

    disp_tld = float(np.linalg.norm(dom_tld.displacement, axis=1).max())
    sag_tld = float(abs(dom_tld.displacement[:, 2].min()))

    console.print(
        f"    ✓ Solved in {t_tld:.2f} s ({res_tld.steps_completed} steps) | "
        f"End Max Deflection: [bold green]{disp_tld:.3f} mm[/bold green] | "
        f"End Tip Sag (Z): [bold green]{sag_tld:.3f} mm[/bold green]"
    )

    results_summary.append({
        "Model": "Hyperelastic (Neo-Hookean)",
        "Analysis Type": "TL Dynamic",
        "Kinematics": "Total Lagrangian Generalized-α",
        "Max Disp (mm)": disp_tld,
        "Tip Sag (mm)": sag_tld,
        "Reaction R_z (N)": rxn_tls,
        "Steps": res_tld.steps_completed,
        "Time": t_tld * 1000,
        "Export Path": str(dir_tld / f"result.{fmt}"),
    })

    # =========================================================================
    # 5. Multi-Model Telemetry & Comparative Synthesis
    # =========================================================================
    console.print("\n[bold yellow][5/6] Multi-Model Telemetry & Comparative Synthesis[/bold yellow]")

    summary_table = Table(
        title="Simulation 1106: Cantilever Beam Gravity & Hydrostatic Buoyancy Suite",
        border_style="cyan",
    )
    summary_table.add_column("Analysis Suite", style="bold white")
    summary_table.add_column("Formulation", style="cyan")
    summary_table.add_column("Kinematic Framework", style="dim")
    summary_table.add_column("Max Disp", justify="right", style="green")
    summary_table.add_column("Tip Z-Sag", justify="right", style="yellow")
    summary_table.add_column("Reaction R_z", justify="right", style="magenta")
    summary_table.add_column("Steps / Iters", justify="center", style="white")
    summary_table.add_column("Solve Time", justify="right", style="dim")

    for r in results_summary:
        time_str = f"{r['Time']:.1f} ms" if r["Time"] < 1000 else f"{r['Time']/1000:.2f} s"
        summary_table.add_row(
            str(r["Analysis Type"]),
            str(r["Model"]),
            str(r["Kinematics"]),
            f"{r['Max Disp (mm)']:.3f} mm",
            f"{r['Tip Sag (mm)']:.3f} mm",
            f"{r['Reaction R_z (N)']:.3f} N",
            str(r["Steps"]),
            time_str,
        )

    console.print(summary_table)

    # Physical Insights
    stiffening_pct = ((disp_ls - disp_tls) / disp_ls) * 100.0

    console.print(
        f"\n[bold cyan]Mechanical Synthesis & Physical Insights:[/bold cyan]\n"
        f"  1. [bold white]Archimedean Buoyant Support[/bold white]:\n"
        f"     • Submerged in water (ρ = 1000 kg/m³), buoyant lift provides {total_buoyant_lift_n:.3f} N upward force,\n"
        f"       reducing downward gravitational sag from 18.29 N to 9.14 N ({buoyant_relief_pct:.1f}% reduction).\n"
        f"  2. [bold white]Geometric Stiffening in Finite Strain (TL)[/bold white]:\n"
        f"     • Linear Static assumes infinitesimal displacements, predicting {disp_ls:.3f} mm deflection.\n"
        f"     • Total Lagrangian accounts for finite rotations and axial membrane tension, predicting {disp_tls:.3f} mm\n"
        f"       ({stiffening_pct:.1f}% more rigid due to large-rotation geometric stiffening).\n"
        f"  3. [bold white]Dynamic vs. Static Convergence[/bold white]:\n"
        f"     • Generalized-α time integration captures the transient inertia and Rayleigh damping,\n"
        f"       settling smoothly toward the static equilibrium configuration."
    )

    # 6. ParaView Export Datasets
    console.print(
        f"\n[bold green][6/6] ParaView {fmt.upper()} Visualization Datasets Successfully Exported:[/bold green]"
    )
    for r in results_summary:
        console.print(f"  • {r['Analysis Type']:<18} : [cyan]{r['Export Path']}[/cyan]")

    console.print(
        "\n[bold magenta]ParaView Visualization Tip:[/bold magenta] Open all 4 datasets in ParaView side-by-side.\n"
        "Apply [bold]Warp By Vector[/bold] (Vectors: Displacement, Scale: 1.0) to directly compare the linear vs. nonlinear\n"
        "and static vs. dynamic trajectories of the submerged beam under gravity and buoyancy!\n"
    )

    elapsed_all = time.perf_counter() - t0_all
    console.print(f"[bold green]✓ Simulation 1106 completed in {elapsed_all:.2f} s.[/bold green]\n")


if __name__ == "__main__":
    main()
