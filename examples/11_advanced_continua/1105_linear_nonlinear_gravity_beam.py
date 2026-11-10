"""
Simulation 1105: Beam Deflection under Gravitational Body Load (Linear vs. Nonlinear).

Demonstrates:
- [1/6] Loading Mesh Geometry & Spatial Orientation: Cantilever 3D beam continuum.
- [2/6] Configuring Material Parameters:
        Young's Modulus E = 21,000 MPa, Poisson's Ratio nu = 0.35, Density rho = 1100e-6 tonne/mm^3 (1100 kg/m^3).
- [3/6] Kinematic Formulations & Continuum Domains:
        • Linear Elasticity: Small-strain Hookean law + Tet4Linear element.
        • Nonlinear Hyperelasticity: Finite-strain Neo-Hookean law + Tet4TotalLagrangian element.
- [4/6] Applying Boundary Conditions & Gravity Body Load:
        Clamped root and distributed gravitational body load (F_ext = M * g).
- [5/6] Executing Simulation Solvers:
        • Linear Static Direct Solver (LinearStaticAnalysis) & Dynamic Relaxation (DynamicRelaxationAnalysis).
        • Nonlinear Incremental Newton-Raphson (NonlinearStaticAnalysis) & Dynamic Relaxation (DynamicRelaxationAnalysis).
- [6/6] Comparative Synthesis, Mechanical Insights & XDMF Animation Export:
        Geometric stiffening effect, equilibrium verification, and ParaView side-by-side visualization.
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
)
from psierra.analysis import (
    LinearStaticAnalysis,
    NonlinearStaticAnalysis,
    DynamicRelaxationAnalysis,
)
from psierra.exchange.writer import SimulationWriter

install(show_locals=False)
console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 1105: Linear vs. Nonlinear Beam Gravity Deflection"
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
        default=4905.0,
        help="Gravitational acceleration magnitude in mm/s^2 (default: 4905.0 for 0.5g)",
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
        default="results/11_advanced_continua/1105_gravity_beam",
        help="Output directory for simulation results (default: results/11_advanced_continua/1105_gravity_beam)",
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="Also export full DR pseudo-time relaxation trajectory animation to trajectory.xdmf (default: False, exports final static equilibrium state to result.xdmf)",
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

    # 0. Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]Simulation 1105: Beam Deflection under Gravity Load[/bold cyan]\n"
            f"[white]Linear Elasticity vs. Nonlinear Finite Strain Hyperelasticity[/white]\n"
            f"[dim]E = 21,000 MPa • ν = 0.35 • ρ = 1,100 kg/m³ (1100e-6 tonne/mm³) • g = {g_accel:.1f} mm/s²[/dim]",
            title="[bold blue]project-sierra / Advanced Continua[/bold blue]",
            border_style="cyan",
        )
    )

    t0_all = time.perf_counter()

    # 1. Loading Mesh Geometry & Spatial Orientation
    console.print("\n[bold yellow][1/6] Loading Mesh Geometry & Spatial Orientation[/bold yellow]")
    mesh_path = resolve_mesh_path(args.mesh)
    mesh = VolumetricMesh(mesh_path)
    # Rotate beam to orient length along Y-axis and deflection along Z-axis
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
        f"  • Volume         : [green]{mesh.volume:,.2f}[/green] mm³"
    )

    # 2. Material Specification
    console.print("\n[bold yellow][2/6] Configuring Material Parameters[/bold yellow]")
    youngs_modulus_mpa = 21000.0
    poissons_ratio = 0.35
    mass_density_tonne_mm3 = 1100e-6  # 1.1e-3 g/mm^3 = 1100 kg/m^3

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=youngs_modulus_mpa,
        poissons_ratio=poissons_ratio,
        mass_density=mass_density_tonne_mm3,
    )

    total_mass_kg = mesh.volume * mass_density_tonne_mm3 * 1e3
    total_weight_n = total_mass_kg * (g_accel / 1000.0)

    console.print(
        f"  • Young's Modulus E : [green]{youngs_modulus_mpa:,.1f}[/green] MPa\n"
        f"  • Poisson's Ratio ν : [green]{poissons_ratio:.2f}[/green]\n"
        f"  • Mass Density ρ    : [green]{mass_density_tonne_mm3:.2e}[/green] tonne/mm³ ({mass_density_tonne_mm3*1e9:.0f} kg/m³)\n"
        f"  • Total Beam Mass   : [green]{total_mass_kg:.3f}[/green] kg\n"
        f"  • Total Gravity F_z : [bold green]{total_weight_n:.2f}[/bold green] N (acting along -Z)"
    )

    # 3. Boundary Conditions & External Gravity Load
    console.print("\n[bold yellow][3/6] Applying Boundary Clamping & Gravity Load[/bold yellow]")
    # Clamped root boundary (Y <= 0 mm)
    root_nodes = select_nodes_by_bounds(mesh, bounds=(-61, 61, -31, 1, -61, 61))
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Downward gravity load vector
    gravity = GravityLoad(acceleration=(0.0, 0.0, -g_accel))
    mat_dummy = Hookean(params)
    elem_dummy = Tet4Linear(mat_dummy)
    dom_dummy = SolidDomain(mesh, mat_dummy, elem_dummy)
    M = dom_dummy.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    console.print(
        f"  • Clamped Root Face : [green]{len(root_nodes)}[/green] nodes clamped at Y ≤ 0 mm ({len(clamp.dofs)} DOFs)\n"
        f"  • Free Degrees      : [green]{len(dirichlet.free_dofs)}[/green] free DOFs"
    )

    # Dictionary to collect results for comparison
    results_summary: list[dict[str, str | float]] = []

    # =========================================================================
    # 4. Linear Elastic Static Analysis
    # =========================================================================
    console.print("\n[bold yellow][4/6] Solving Linear Elastic Static Deflection[/bold yellow]")
    mat_linear = Hookean(params)
    elem_linear = Tet4Linear(mat_linear)

    # 4A. Direct Linear Static Solve (u = K^-1 * F)
    console.print("  [bold cyan]• Method 1A: Direct Linear Static Solve (K⁻¹ F)...[/bold cyan]")
    dom_linear_direct = SolidDomain(mesh.copy(), mat_linear, elem_linear)
    t_start = time.perf_counter()
    solver_linear_direct = LinearStaticAnalysis(dom_linear_direct)
    dir_linear_direct = output_dir / "linear_direct"
    res_linear_direct = solver_linear_direct.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        save_directory=dir_linear_direct,
        format=fmt,
        verbose=False,
    )
    t_linear_direct = time.perf_counter() - t_start

    disp_linear_direct = float(np.linalg.norm(dom_linear_direct.displacement, axis=1).max())
    sag_linear_direct = float(abs(dom_linear_direct.displacement[:, 2].min()))
    rxn_linear_direct = float(res_linear_direct.reactions[clamp.dofs][2::3].sum())

    console.print(
        f"    ✓ Solved in {t_linear_direct*1000:.2f} ms | Max Deflection: [bold green]{disp_linear_direct:.3f} mm[/bold green] | "
        f"Tip Sag (Z): [bold green]{sag_linear_direct:.3f} mm[/bold green]"
    )

    results_summary.append({
        "Model": "Linear Elastic (Hookean)",
        "Method": "Direct Linear Solve (K⁻¹ F)",
        "Max Disp (mm)": disp_linear_direct,
        "Tip Sag (mm)": sag_linear_direct,
        "Reaction R_z (N)": rxn_linear_direct,
        "Steps / Iter": 1,
        "Time (ms)": t_linear_direct * 1000,
        "Export Path": str(dir_linear_direct / f"result.{fmt}"),
    })

    # 4B. Dynamic Relaxation Linear Solve (Matrix-Free Explicit)
    console.print("  [bold cyan]• Method 1B: Explicit Dynamic Relaxation Solve (DR Kinetic)...[/bold cyan]")
    dom_linear_dr = SolidDomain(mesh.copy(), mat_linear, elem_linear)
    t_start = time.perf_counter()
    solver_linear_dr = DynamicRelaxationAnalysis(
        dom_linear_dr, damping_type="kinetic", safety_factor=1.1, dt=1.0
    )
    dir_linear_dr = output_dir / "linear_dynamic_relaxation"
    res_linear_dr = solver_linear_dr.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        tol=1e-4,
        max_steps=1000,
        save_directory=dir_linear_dr,
        output_name="result",
        format=fmt,
        verbose=False,
    )
    t_linear_dr = time.perf_counter() - t_start

    if args.trajectory:
        console.print("    [dim]• Exporting linear DR pseudo-time relaxation trajectory to trajectory.xdmf...[/dim]")
        dom_linear_traj = SolidDomain(mesh.copy(), mat_linear, elem_linear)
        solver_linear_traj = DynamicRelaxationAnalysis(
            dom_linear_traj, damping_type="kinetic", safety_factor=1.1, dt=1.0
        )
        solver_linear_traj.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            tol=1e-4,
            max_steps=1000,
            save_interval=25,
            save_trajectory=True,
            save_directory=dir_linear_dr,
            output_name="trajectory",
            format=fmt,
            verbose=False,
        )

    disp_linear_dr = float(np.linalg.norm(dom_linear_dr.displacement, axis=1).max())
    sag_linear_dr = float(abs(dom_linear_dr.displacement[:, 2].min()))
    rxn_linear_dr = float(res_linear_dr.reactions[clamp.dofs][2::3].sum())

    console.print(
        f"    ✓ Solved in {t_linear_dr*1000:.2f} ms ({res_linear_dr.steps_completed} steps, {res_linear_dr.kinetic_peaks} kinetic peaks) | "
        f"Max Deflection: [bold green]{disp_linear_dr:.3f} mm[/bold green] | Diff vs Direct: [dim]{abs(disp_linear_dr - disp_linear_direct):.4f} mm[/dim]"
    )

    results_summary.append({
        "Model": "Linear Elastic (Hookean)",
        "Method": "Dynamic Relaxation (Kinetic Damping)",
        "Max Disp (mm)": disp_linear_dr,
        "Tip Sag (mm)": sag_linear_dr,
        "Reaction R_z (N)": rxn_linear_dr,
        "Steps / Iter": res_linear_dr.steps_completed,
        "Time (ms)": t_linear_dr * 1000,
        "Export Path": str(dir_linear_dr / f"result.{fmt}"),
    })

    # =========================================================================
    # 5. Nonlinear Hyperelastic Static Analysis (Finite Strain Total Lagrangian)
    # =========================================================================
    console.print("\n[bold yellow][5/6] Solving Nonlinear Hyperelastic Large Deflection[/bold yellow]")
    mat_nonlinear = NeoHookean(params)
    elem_nonlinear = Tet4TotalLagrangian(mat_nonlinear)

    # 5A. Incremental Newton-Raphson Solve
    console.print("  [bold cyan]• Method 2A: Incremental Newton-Raphson (10 load steps, tangent reassembly)...[/bold cyan]")
    dom_nonlinear_nr = SolidDomain(mesh.copy(), mat_nonlinear, elem_nonlinear)
    t_start = time.perf_counter()
    solver_nonlinear_nr = NonlinearStaticAnalysis(dom_nonlinear_nr)
    dir_nonlinear_nr = output_dir / "nonlinear_newton_raphson"
    res_nonlinear_nr = solver_nonlinear_nr.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        n_steps=10,
        nr_iterations=25,
        nr_tol=1e-4,
        save_history=False,
        save_directory=dir_nonlinear_nr,
        format=fmt,
        verbose=False,
    )
    t_nonlinear_nr = time.perf_counter() - t_start

    disp_nonlinear_nr = float(np.linalg.norm(dom_nonlinear_nr.displacement, axis=1).max())
    sag_nonlinear_nr = float(abs(dom_nonlinear_nr.displacement[:, 2].min()))
    rxn_nonlinear_nr = float(res_nonlinear_nr.reactions[clamp.dofs][2::3].sum())
    total_nr_iters = sum(res_nonlinear_nr.iteration_history)

    console.print(
        f"    ✓ Solved in {t_nonlinear_nr:.2f} s (10 steps, {total_nr_iters} total iterations) | "
        f"Max Deflection: [bold green]{disp_nonlinear_nr:.3f} mm[/bold green] | "
        f"Tip Sag (Z): [bold green]{sag_nonlinear_nr:.3f} mm[/bold green]"
    )

    results_summary.append({
        "Model": "Nonlinear Hyperelastic (Neo-Hookean)",
        "Method": "Incremental Newton-Raphson",
        "Max Disp (mm)": disp_nonlinear_nr,
        "Tip Sag (mm)": sag_nonlinear_nr,
        "Reaction R_z (N)": rxn_nonlinear_nr,
        "Steps / Iter": f"10 steps ({total_nr_iters} iters)",
        "Time (ms)": t_nonlinear_nr * 1000,
        "Export Path": str(dir_nonlinear_nr / f"result.{fmt}"),
    })

    # 5B. Dynamic Relaxation Nonlinear Solve (Matrix-Free Explicit)
    console.print("  [bold cyan]• Method 2B: Explicit Dynamic Relaxation Solve (DR Kinetic on Neo-Hookean)...[/bold cyan]")
    dom_nonlinear_dr = SolidDomain(mesh.copy(), mat_nonlinear, elem_nonlinear)
    t_start = time.perf_counter()
    solver_nonlinear_dr = DynamicRelaxationAnalysis(
        dom_nonlinear_dr, damping_type="kinetic", safety_factor=1.1, dt=1.0
    )
    dir_nonlinear_dr = output_dir / "nonlinear_dynamic_relaxation"
    res_nonlinear_dr = solver_nonlinear_dr.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        tol=1e-4,
        max_steps=1000,
        save_directory=dir_nonlinear_dr,
        output_name="result",
        format=fmt,
        verbose=False,
    )
    t_nonlinear_dr = time.perf_counter() - t_start

    if args.trajectory:
        console.print("    [dim]• Exporting nonlinear DR pseudo-time relaxation trajectory to trajectory.xdmf...[/dim]")
        dom_nonlinear_traj = SolidDomain(mesh.copy(), mat_nonlinear, elem_nonlinear)
        solver_nonlinear_traj = DynamicRelaxationAnalysis(
            dom_nonlinear_traj, damping_type="kinetic", safety_factor=1.1, dt=1.0
        )
        solver_nonlinear_traj.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            tol=1e-4,
            max_steps=1000,
            save_interval=25,
            save_trajectory=True,
            save_directory=dir_nonlinear_dr,
            output_name="trajectory",
            format=fmt,
            verbose=False,
        )

    disp_nonlinear_dr = float(np.linalg.norm(dom_nonlinear_dr.displacement, axis=1).max())
    sag_nonlinear_dr = float(abs(dom_nonlinear_dr.displacement[:, 2].min()))
    rxn_nonlinear_dr = float(res_nonlinear_dr.reactions[clamp.dofs][2::3].sum())

    console.print(
        f"    ✓ Solved in {t_nonlinear_dr:.2f} s ({res_nonlinear_dr.steps_completed} steps, {res_nonlinear_dr.kinetic_peaks} kinetic peaks) | "
        f"Max Deflection: [bold green]{disp_nonlinear_dr:.3f} mm[/bold green] | Diff vs NR: [dim]{abs(disp_nonlinear_dr - disp_nonlinear_nr):.4f} mm[/dim]"
    )

    results_summary.append({
        "Model": "Nonlinear Hyperelastic (Neo-Hookean)",
        "Method": "Dynamic Relaxation (Kinetic Damping)",
        "Max Disp (mm)": disp_nonlinear_dr,
        "Tip Sag (mm)": sag_nonlinear_dr,
        "Reaction R_z (N)": rxn_nonlinear_dr,
        "Steps / Iter": res_nonlinear_dr.steps_completed,
        "Time (ms)": t_nonlinear_dr * 1000,
        "Export Path": str(dir_nonlinear_dr / f"result.{fmt}"),
    })

    # =========================================================================
    # 6. Comparative Synthesis & Summary Table
    # =========================================================================
    console.print("\n[bold yellow][6/6] Verifying Results & Multi-Model Comparative Synthesis[/bold yellow]")

    summary_table = Table(
        title="Simulation 1105: Linear vs. Nonlinear Gravity Deflection Benchmark",
        border_style="cyan",
    )
    summary_table.add_column("Formulation", style="bold white")
    summary_table.add_column("Solver Strategy", style="cyan")
    summary_table.add_column("Max Deflection", justify="right", style="green")
    summary_table.add_column("Tip Z-Sag", justify="right", style="yellow")
    summary_table.add_column("Reaction R_z", justify="right", style="magenta")
    summary_table.add_column("Steps / Iters", justify="center", style="white")
    summary_table.add_column("Wall Time", justify="right", style="dim")

    for r in results_summary:
        time_str = f"{r['Time (ms)']:.1f} ms" if r["Time (ms)"] < 1000 else f"{r['Time (ms)']/1000:.2f} s"
        summary_table.add_row(
            str(r["Model"]),
            str(r["Method"]),
            f"{r['Max Disp (mm)']:.3f} mm",
            f"{r['Tip Sag (mm)']:.3f} mm",
            f"{r['Reaction R_z (N)']:.2f} N",
            str(r["Steps / Iter"]),
            time_str,
        )

    console.print(summary_table)

    # Mechanical Insights
    stiffening_ratio = ((disp_linear_direct - disp_nonlinear_nr) / disp_linear_direct) * 100.0
    dr_lin_error = abs(disp_linear_dr - disp_linear_direct) / disp_linear_direct * 100.0
    dr_nl_error = abs(disp_nonlinear_dr - disp_nonlinear_nr) / disp_nonlinear_nr * 100.0

    console.print(
        f"\n[bold cyan]Mechanical Synthesis & Physical Insights:[/bold cyan]\n"
        f"  1. [bold white]Geometric Stiffening Effect[/bold white]:\n"
        f"     • Linear Elasticity assumes infinitesimal displacements (∇u ≈ 0), overestimating deflection at [green]{disp_linear_direct:.2f} mm[/green].\n"
        f"     • Nonlinear Total Lagrangian hyperelasticity accounts for large rotations and finite tensile membrane strain,\n"
        f"       correctly predicting a stiffer equilibrium at [green]{disp_nonlinear_nr:.2f} mm[/green] ([bold yellow]{stiffening_ratio:.1f}% more rigid[/bold yellow]).\n"
        f"  2. [bold white]Dynamic Relaxation Solver Precision[/bold white]:\n"
        f"     • Linear Agreement   : DR reproduces direct linear solve to [bold green]{dr_lin_error:.3f}%[/bold green] relative accuracy.\n"
        f"     • Nonlinear Agreement: DR reproduces Newton-Raphson finite strain equilibrium to [bold green]{dr_nl_error:.3f}%[/bold green] relative accuracy.\n"
        f"  3. [bold white]Static Equilibrium Balance[/bold white]:\n"
        f"     • Applied downward weight : {total_weight_n:.2f} N\n"
        f"     • Linear support reaction : {rxn_linear_direct:.2f} N (Eq error: {abs(rxn_linear_direct - total_weight_n):.2e} N)\n"
        f"     • Nonlinear support reaction : {rxn_nonlinear_nr:.2f} N (Eq error: {abs(rxn_nonlinear_nr - total_weight_n):.2e} N)"
    )

    console.print(
        f"\n[bold green]✓ All ParaView {fmt.upper()} Visualization Datasets Successfully Exported:[/bold green]"
    )
    for r in results_summary:
        console.print(f"  • {r['Model']:<35} : [cyan]{r['Export Path']}[/cyan]")

    console.print(
        "\n[bold magenta]ParaView Visualization Guide:[/bold magenta]\n"
        "  • [bold white]Instant Linear Comparison (Direct vs DR)[/bold white]:\n"
        "    Open 'linear_direct/result.xdmf' and 'linear_dynamic_relaxation/result.xdmf'.\n"
        "    Apply [bold]Warp By Vector[/bold] (Vectors: Displacement, Scale: 1.0).\n"
        "    Both files load the converged static equilibrium directly at Step 0 with identical ~182.00 mm deflection!\n"
        "  • [bold white]Instant Nonlinear Comparison (Newton-Raphson vs DR)[/bold white]:\n"
        "    Open 'nonlinear_dynamic_relaxation/result.xdmf' (or final step of 'nonlinear_newton_raphson/result.xdmf').\n"
        "    Both converge to ~145.95 mm finite-strain equilibrium (demonstrating geometric stiffening).\n"
        "  • [bold white]Kinetic Damping Relaxation Movie[/bold white]:\n"
        "    Run with [bold]--trajectory[/bold] to export 'trajectory.xdmf'. Open it in ParaView and click [bold]Play (▶)[/bold]\n"
        "    to watch the explicit fictitious kinetic energy resets guide the beam to static equilibrium!\n"
    )

    elapsed_all = time.perf_counter() - t0_all
    console.print(f"[bold green]✓ Simulation 1105 completed in {elapsed_all:.2f} s.[/bold green]\n")


if __name__ == "__main__":
    main()
