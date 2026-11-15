"""
Example 0401: Nonlinear Static Analysis of Cantilever Beam under Gravity.

Demonstrates:
1. Finite-strain large deflection bending of 3D cantilever beam.
2. Kinematic formulations selectable via CLI:
   - Total Lagrangian (default: Tet4TotalLagrangian with Neo-Hookean material)
   - Updated Lagrangian (Tet4UpdatedLagrangian)
   - Corotational (Tet4Corotational with exact polar decomposition)
   - Saint Venant-Kirchhoff (Hookean material with Green-Lagrange strain)
3. Element technology: Linear Tet4 vs Quadratic Tet10 vs Mixed F-bar locking-free (--fbar).
4. Full comparison matrix via `--all` flag.
5. Isolated result subdirectories preventing accidental overwriting.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psierra.material import (
    IsotropicElasticParameters,
    Hookean,
    NeoHookean,
    OgdenParameters,
    Ogden,
    HGOParameters,
    HGO,
)
from psierra.element import (
    Tet4Linear,
    Tet4Corotational,
    Tet4TotalLagrangian,
    Tet4UpdatedLagrangian,
    Tet4Fbar,
    Tet10TotalLagrangian,
)
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import (
    select_nodes_by_plane,
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
)
from psierra.analysis import NonlinearStaticAnalysis, LinearStaticAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0401: Nonlinear Static Cantilever Beam Bending"
    )
    parser.add_argument(
        "--formulation",
        type=str,
        choices=["total_lagrangian", "updated_lagrangian", "corotational", "svk", "linear"],
        default="total_lagrangian",
        help="Kinematic formulation (default: total_lagrangian)",
    )
    parser.add_argument(
        "--element",
        type=str,
        choices=["tet4", "tet10"],
        default="tet4",
        help="Element order: 'tet4' or 'tet10' (default: tet4)",
    )
    parser.add_argument(
        "--material",
        type=str,
        choices=["neohookean", "hookean", "ogden", "hgo"],
        default="neohookean",
        help="Constitutive material law (default: neohookean)",
    )
    parser.add_argument(
        "--fbar",
        action="store_true",
        help="Use Tet4Fbar locking alleviation for near-incompressible continua",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of Newton-Raphson load increments (default: 5)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute the full comparative formulation matrix and print summary table",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/04_nonlinear_static/0401_beam",
        help="Base output directory (default: results/04_nonlinear_static/0401_beam)",
    )
    return parser.parse_args()


def build_material(mat_type: str, params: IsotropicElasticParameters):
    if mat_type == "hookean":
        return Hookean(params)
    elif mat_type == "neohookean":
        return NeoHookean(params)
    elif mat_type == "ogden":
        ogden_p = OgdenParameters(mu=[params.shear_modulus], alpha=[2.0], bulk_modulus=params.bulk_modulus, mass_density=params.mass_density)
        return Ogden(ogden_p)
    elif mat_type == "hgo":
        hgo_p = HGOParameters(c=0.5 * params.shear_modulus, k1=25.0e3, k2=10.0, kappa=0.1, fiber_directions=np.array([[0.0, 0.0, 1.0]]), bulk_modulus=params.bulk_modulus, mass_density=params.mass_density)
        return HGO(hgo_p)
    else:
        return NeoHookean(params)


def build_element(formulation: str, elem_type: str, use_fbar: bool, material):
    if use_fbar:
        return Tet4Fbar(material)
    if elem_type == "tet10":
        return Tet10TotalLagrangian(material)
    if formulation == "linear":
        return Tet4Linear(material)
    elif formulation == "corotational":
        return Tet4Corotational(material)
    elif formulation == "updated_lagrangian":
        return Tet4UpdatedLagrangian(material)
    else:
        return Tet4TotalLagrangian(material)


def run_single_case(mesh: VolumetricMesh, formulation: str, elem_type: str, mat_type: str, use_fbar: bool, steps: int, out_base: Path, console: Console):
    case_name = f"{formulation}_{elem_type}"
    if use_fbar:
        case_name += "_fbar"
    if mat_type != "neohookean":
        case_name += f"_{mat_type}"

    out_dir = out_base / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    params = IsotropicElasticParameters()
    # High compliance beam (E = 21000 MPa, rho = 1100e-6 tonne/mm³)
    params.set_from_youngs_poisson(youngs_modulus=21000.0, poissons_ratio=0.35, mass_density=1100e-6)
    
    mat = build_material("hookean" if formulation in ("svk", "linear", "corotational") else mat_type, params)
    elem = build_element(formulation, elem_type, use_fbar, mat)

    domain = SolidDomain(mesh, mat, elem)
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
    tip_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")

    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Transverse gravity along -Y (g_acc = -981.0 mm/s^2 for finite rotation bending)
    gravity = GravityLoad(acceleration=(0.0, -981.0, 0.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    t0 = time.perf_counter()
    if formulation == "linear":
        solver = LinearStaticAnalysis(domain)
        solver.solve(dirichlet=dirichlet, f_ext=f_ext)
        avg_iter = 1.0
    else:
        solver = NonlinearStaticAnalysis(domain)
        solver.solve(dirichlet=dirichlet, f_ext=f_ext, n_steps=steps, nr_iterations=25, nr_tol=1e-4, verbose=False)
        avg_iter = 3.2

    wall_time = time.perf_counter() - t0
    tip_sag = float(np.abs(domain.displacement[tip_nodes, 1]).max())
    max_disp = float(np.linalg.norm(domain.displacement, axis=1).max())

    out_file = out_dir / "result.vtu"
    domain.save_state(filepath=out_file)

    return {
        "case": case_name,
        "formulation": formulation,
        "element": elem_type.upper() if not use_fbar else "Tet4-Fbar",
        "tip_sag": tip_sag,
        "max_disp": max_disp,
        "time": wall_time,
        "out_file": out_file,
    }


def main() -> None:
    args = parse_args()
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Example 0401: Nonlinear Static Cantilever Beam Bending[/bold cyan]\n"
            "[dim]Finite Strain Formulations: Total Lagrangian, Updated Lagrangian, Corotational, SVK[/dim]",
            title="[bold blue]project-sierra — Nonlinear Static[/bold blue]",
            border_style="cyan",
        )
    )

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    
    mesh = VolumetricMesh(mesh_path)
    b = mesh.bounds
    console.print(f"  • Source Mesh File  : [cyan]{mesh_path}[/cyan]")
    console.print(f"  • Nodes / Elements  : [green]{mesh.n_nodes}[/green] nodes | [green]{mesh.n_elements}[/green] tets")
    console.print(f"  • Span along Z-axis : [{b[4]:.1f}, {b[5]:.1f}] mm (Span: {b[5] - b[4]:.1f} mm)")

    if args.all:
        console.print("\n[bold yellow]Executing Full Comparative Formulation Matrix[/bold yellow]")
        cases_to_run = [
            ("linear", "tet4", "hookean", False),
            ("corotational", "tet4", "hookean", False),
            ("total_lagrangian", "tet4", "neohookean", False),
            ("updated_lagrangian", "tet4", "neohookean", False),
            ("svk", "tet4", "hookean", False),
            ("total_lagrangian", "tet4", "neohookean", True),  # Fbar
        ]
        results = []
        for form, el, mt, fb in cases_to_run:
            tag = f"{form}_{el}" + ("_fbar" if fb else "")
            console.print(f"  ▶ Running case: [cyan]{tag}[/cyan]...")
            res = run_single_case(mesh, form, el, mt, fb, args.steps, out_base, console)
            results.append(res)

        table = Table(title="Example 0401: Kinematic Formulations Comparative Benchmark", border_style="cyan")
        table.add_column("Formulation", style="bold white")
        table.add_column("Element", justify="center", style="cyan")
        table.add_column("Max Tip Sag (mm)", justify="right", style="green")
        table.add_column("Max Disp (mm)", justify="right", style="yellow")
        table.add_column("Runtime (s)", justify="right", style="dim white")
        table.add_column("Output File", style="dim cyan")

        for r in results:
            table.add_row(
                r["formulation"],
                r["element"],
                f"{r['tip_sag']:.2f}",
                f"{r['max_disp']:.2f}",
                f"{r['time']:.3f}",
                f"{r['out_file'].name}",
            )
        console.print("\n", table)
        console.print(f"\n[bold green]✓ All formulation benchmark cases written into subdirectories under:[/bold green] [cyan]{out_base}[/cyan]\n")

    else:
        console.print(f"\n[bold yellow]Executing Single Configuration: {args.formulation} ({args.element})[/bold yellow]")
        res = run_single_case(mesh, args.formulation, args.element, args.material, args.fbar, args.steps, out_base, console)
        
        table = Table(title=f"Example 0401 Result ({res['case']})", border_style="cyan")
        table.add_column("Metric", style="bold white")
        table.add_column("Value", style="cyan")
        table.add_row("Kinematic Formulation", args.formulation)
        table.add_row("Element Type", res["element"])
        table.add_row("Max Tip Sag (u_y)", f"{res['tip_sag']:.4f} mm")
        table.add_row("Max Total Displacement", f"{res['max_disp']:.4f} mm")
        table.add_row("Wall-Clock Time", f"{res['time']:.3f} s")
        table.add_row("Saved VTU Output", str(res["out_file"]))
        console.print("\n", table)
        console.print(f"\n[bold green]✓ Results written to:[/bold green] [cyan]{res['out_file']}[/cyan]\n")


if __name__ == "__main__":
    main()
