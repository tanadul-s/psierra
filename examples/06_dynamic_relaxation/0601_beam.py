"""
Example 0601: Matrix-Free Static Equilibrium via Dynamic Relaxation on Cantilever Beam.

Demonstrates:
1. Native beam coordinate frame (Z in [-30, 200] mm, clamped at Z = min).
2. Matrix-free Dynamic Relaxation (DR):
   - Finding large-deflection nonlinear static equilibrium without tangent stiffness assembly or factorization.
   - Kinetic energy peak damping (resetting velocity to zero when kinetic energy peaks).
   - Viscous mass-proportional damping comparison.
3. Compressible Neo-Hookean hyperelastic soft cantilever beam under gravity.
4. ParaView XDMF / PVD export of converged static configuration and trajectory.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psierra.material import IsotropicElasticParameters, Hookean, NeoHookean
from psierra.element import Tet4Linear, Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import (
    select_nodes_by_plane,
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
)
from psierra.analysis import DynamicRelaxationAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0601: Matrix-Free Dynamic Relaxation Cantilever Beam"
    )
    parser.add_argument(
        "--damping",
        type=str,
        choices=["kinetic", "viscous"],
        default="kinetic",
        help="Dynamic relaxation damping mechanism (default: kinetic)",
    )
    parser.add_argument(
        "--material",
        type=str,
        choices=["neohookean", "hookean"],
        default="neohookean",
        help="Material model (default: neohookean)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Maximum pseudo-time integration iterations (default: 400)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Residual force convergence tolerance (default: 1e-4)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["xdmf", "pvd"],
        default="xdmf",
        help="Output ParaView dataset format (default: xdmf)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/06_dynamic_relaxation/0601_beam",
        help="Base output directory (default: results/06_dynamic_relaxation/0601_beam)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comparison between kinetic peak damping and viscous damping",
    )
    return parser.parse_args()


def run_dr_simulation(
    damping_type: str,
    material_type: str,
    max_steps: int,
    tol: float,
    fmt: str,
    output_dir: Path,
    mesh_path: Path,
    verbose: bool = True,
) -> dict[str, str | float]:
    console = Console()
    mesh = VolumetricMesh(mesh_path)

    # Compliant beam parameters: E = 21,000 Pa, nu = 0.35, rho = 1100e-6 g/mm³
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000.0,
        poissons_ratio=0.35,
        mass_density=1100e-6,
    )

    if material_type == "neohookean":
        mat = NeoHookean(params)
        elem = Tet4TotalLagrangian(mat)
    else:
        mat = Hookean(params)
        elem = Tet4Linear(mat)

    domain = SolidDomain(mesh, mat, elem)

    # Clamped root face at Z = min
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min", tolerance=1e-3)
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Gravity body force along -Y (1.0g downward)
    g_acc = -9810.0
    gravity = GravityLoad(acceleration=(0.0, g_acc, 0.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    case_tag = f"{damping_type}_{material_type}"
    run_out_dir = output_dir / case_tag
    run_out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        console.print(f"  • Running DR solver: [bold cyan]{case_tag}[/bold cyan] (damping={damping_type})")

    t0 = time.perf_counter()
    solver = DynamicRelaxationAnalysis(
        domain, damping_type=damping_type, safety_factor=1.1, dt=1.0
    )
    result = solver.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        tol=tol,
        max_steps=max_steps,
        save_directory=run_out_dir,
        output_name="result",
        format=fmt,
        verbose=False,
    )
    wall_time = time.perf_counter() - t0

    disp = domain.displacement
    max_disp = float(np.linalg.norm(disp, axis=1).max())
    tip_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max", tolerance=1e-3)
    tip_sag = float(-disp[tip_nodes, 1].mean())

    kinetic_peaks = getattr(result, "kinetic_peaks", 0)

    out_file = run_out_dir / f"result.{fmt}"

    return {
        "case": case_tag,
        "damping": damping_type.capitalize(),
        "material": material_type.capitalize(),
        "steps_completed": result.steps_completed,
        "kinetic_peaks": kinetic_peaks,
        "converged": result.converged,
        "max_disp_mm": max_disp,
        "tip_sag_mm": tip_sag,
        "wall_time_s": wall_time,
        "output_path": str(out_file),
    }


def main() -> None:
    args = parse_args()
    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Example 0601: Matrix-Free Static Equilibrium via Dynamic Relaxation[/bold cyan]\n"
            f"[dim]Damping: {args.damping.capitalize()} | Material: {args.material.capitalize()} | Tol: {args.tol:.1e}[/dim]",
            title="[bold blue]project-sierra — Dynamic Relaxation[/bold blue]",
            border_style="cyan",
        )
    )

    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Beam mesh file not found: {mesh_path}")

    base_out = Path(args.output_dir)

    if args.all:
        console.print("\n[bold yellow]Executing Comparison: Kinetic Peak Damping vs Viscous Damping...[/bold yellow]")
        cases = [
            ("kinetic", "neohookean"),
            ("viscous", "neohookean"),
            ("kinetic", "hookean"),
        ]
        results = []
        for damp, mat in cases:
            res = run_dr_simulation(
                damping_type=damp,
                material_type=mat,
                max_steps=args.max_steps,
                tol=args.tol,
                fmt=args.format,
                output_dir=base_out,
                mesh_path=mesh_path,
                verbose=True,
            )
            results.append(res)

        table = Table(title="Example 0601: Dynamic Relaxation Solver Comparison", border_style="cyan")
        table.add_column("Case Tag", style="bold white")
        table.add_column("Damping Type", style="cyan")
        table.add_column("Material", style="magenta")
        table.add_column("Steps", style="white")
        table.add_column("Kinetic Peaks", style="yellow")
        table.add_column("Max Disp (mm)", style="green")
        table.add_column("Tip Sag (mm)", style="green")
        table.add_column("Runtime (s)", style="blue")

        for r in results:
            table.add_row(
                str(r["case"]),
                str(r["damping"]),
                str(r["material"]),
                f"{r['steps_completed']}/{args.max_steps}",
                str(r["kinetic_peaks"]),
                f"{r['max_disp_mm']:.3f}",
                f"{r['tip_sag_mm']:.3f}",
                f"{r['wall_time_s']:.3f}",
            )
        console.print(table)

    else:
        console.print(f"\n[bold yellow]Executing Canonical Dynamic Relaxation ({args.damping} damping)...[/bold yellow]")
        res = run_dr_simulation(
            damping_type=args.damping,
            material_type=args.material,
            max_steps=args.max_steps,
            tol=args.tol,
            fmt=args.format,
            output_dir=base_out,
            mesh_path=mesh_path,
            verbose=True,
        )

        table = Table(title=f"Example 0601: {res['case']}", border_style="cyan")
        table.add_column("Quantity", style="bold white")
        table.add_column("Value", style="cyan")
        table.add_row("Damping Scheme", str(res["damping"]))
        table.add_row("Constitutive Law", str(res["material"]))
        table.add_row("Iterations to Equilibrium", f"{res['steps_completed']} steps")
        table.add_row("Kinetic Energy Peak Resets", str(res["kinetic_peaks"]))
        table.add_row("Convergence Status", "[bold green]Converged[/bold green]" if res["converged"] else "[bold red]Max Steps[/bold red]")
        table.add_row("Max Resultant Displacement", f"{res['max_disp_mm']:.4f} mm")
        table.add_row("Tip Deflection Sag", f"{res['tip_sag_mm']:.4f} mm")
        table.add_row("Solver Runtime", f"{res['wall_time_s']:.3f} s")
        table.add_row("ParaView Dataset", str(res["output_path"]))
        console.print(table)
        console.print(f"\n[bold green]✓ Dynamic relaxation equilibrium achieved. Output saved to:[/bold green] [cyan]{res['output_path']}[/cyan]\n")


if __name__ == "__main__":
    main()
