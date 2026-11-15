"""
Example 0201: Linear Static Analysis of 3D Cantilever Beam.

Demonstrates:
1. Ingesting volumetric beam in native Z-axis coordinates (no arbitrary Euler rotation).
2. Clamping root plane (Z_min) using semantic plane selector.
3. Applying loading conditions via CLI flag:
   - `--load gravity`: Body load self-weight along -Y (consistent mass integration)
   - `--load tip`: Distributed traction/nodal load at tip (Z_max) along -Y
   - `--load both`: Coupled gravity and tip load
4. Direct sparse SuperLU factorization (LinearStaticAnalysis).
5. Result isolation into parameter-tagged subdirectories.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import (
    select_nodes_by_plane,
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
    NodalForce,
)
from psierra.analysis import LinearStaticAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0201: Linear Static Beam Bending"
    )
    parser.add_argument(
        "--load",
        type=str,
        choices=["gravity", "tip", "both"],
        default="gravity",
        help="Loading mode: 'gravity' (self-weight), 'tip' (end load), or 'both' (default: gravity)",
    )
    parser.add_argument(
        "--fine",
        action="store_true",
        help="Use fine mesh (beam1_fine.vtk) instead of standard (beam1.vtk)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/02_linear_static/0201_beam",
        help="Base output directory (default: results/02_linear_static/0201_beam)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    console.print(
        Panel.fit(
            f"[bold cyan]Example 0201: Linear Static Cantilever Beam ({args.load.upper()})[/bold cyan]\n"
            "[dim]Small-Strain Infinitesimal Elasticity & Direct Sparse Factorization[/dim]",
            title="[bold blue]project-sierra — Linear Static[/bold blue]",
            border_style="cyan",
        )
    )

    case_tag = f"{args.load}_{'fine' if args.fine else 'standard'}"
    out_dir = Path(args.output_dir) / case_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loading Mesh in Native Coordinates
    console.print("\n[bold yellow][1/6] Loading Mesh Geometry[/bold yellow]")
    mesh_file = "data/beam1/mesh/beam1_fine.vtk" if args.fine else "data/beam1/mesh/beam1.vtk"
    mesh_path = Path(mesh_file)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found at: {mesh_path}")
    
    mesh = VolumetricMesh(mesh_path)
    b = mesh.bounds
    console.print(f"  • Source Mesh File  : [cyan]{mesh_path}[/cyan]")
    console.print(f"  • Nodes / DOFs      : [green]{mesh.n_nodes}[/green] ({mesh.n_dofs} DOFs)")
    console.print(f"  • Elements (Tet4)   : [green]{mesh.n_elements}[/green]")
    console.print(f"  • Length along Z    : [{b[4]:.1f}, {b[5]:.1f}] mm (Span: {b[5] - b[4]:.1f} mm)")

    # 2. Material & Element
    console.print("\n[bold yellow][2/6] Configuring Material & Element Formulation[/bold yellow]")
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(youngs_modulus=68.0e3, poissons_ratio=0.33, mass_density=2700e-6)  # MPa & tonne/mm³
    material = Hookean(params)
    element = Tet4Linear(material)
    console.print("  • Material : [cyan]Hookean Linear Elasticity[/cyan] (E = 68.0 GPa, nu = 0.33)")
    console.print("  • Element  : [cyan]Tet4Linear[/cyan]")

    # 3. Domain
    console.print("\n[bold yellow][3/6] Setting Up Physical Domain Container[/bold yellow]")
    domain = SolidDomain(mesh, material, element)

    # 4. Boundary Conditions & Loads
    console.print("\n[bold yellow][4/6] Applying Boundary Conditions & External Loads[/bold yellow]")
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
    tip_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
    
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(clamp)
    console.print(f"  • Clamped Root Nodes (Z_min): [green]{len(root_nodes)}[/green] nodes fully fixed (u_x = u_y = u_z = 0)")

    f_ext = np.zeros(domain.n_dofs, dtype=np.float64)

    if args.load in ("gravity", "both"):
        gravity = GravityLoad(acceleration=(0.0, -9810.0, 0.0))  # mm/s² along -Y
        M = domain.assemble_mass_matrix()
        f_ext += gravity.compute_force_vector(M)
        console.print("  • Body Load Applied         : Self-weight gravity (|g| = 9.81 m/s² along -Y)")

    if args.load in ("tip", "both"):
        tip_force_total = -500.0  # N along Y
        tip_force_per_node = tip_force_total / len(tip_nodes)
        tip_load = NodalForce(tip_nodes, component=1, value=tip_force_per_node)
        f_ext += tip_load.compute_force_vector(domain.n_dofs)
        console.print(f"  • Tip Load Applied          : Total Fy = {tip_force_total:.1f} N across [green]{len(tip_nodes)}[/green] tip nodes")

    # 5. Solver
    console.print("\n[bold yellow][5/6] Executing Linear Static Solver[/bold yellow]")
    analysis = LinearStaticAnalysis(domain)
    u_sol = analysis.solve(dirichlet=dirichlet, f_ext=f_ext)
    console.print("  [green]✓ Sparse SuperLU direct solve converged successfully.[/green]")

    # 6. Results & Verification
    console.print("\n[bold yellow][6/6] Verifying Results & State Export[/bold yellow]")
    u_reshaped = domain.displacement
    max_tip_sag = np.abs(u_reshaped[tip_nodes, 1]).max()
    max_disp = np.linalg.norm(u_reshaped, axis=1).max()
    
    table = Table(title=f"Simulation 0201 Verification Summary ({case_tag})", border_style="cyan")
    table.add_column("Quantity", style="bold white")
    table.add_column("Value", justify="right", style="cyan")
    table.add_row("Total System DOFs", f"{domain.n_dofs}")
    table.add_row("Fixed Dirichlet DOFs", f"{len(root_nodes) * 3}")
    table.add_row("Free Unconstrained DOFs", f"{domain.n_dofs - len(root_nodes) * 3}")
    table.add_row("Max Tip Sag (u_y)", f"{max_tip_sag:.4f} mm")
    table.add_row("Max Total Displacement", f"{max_disp:.4f} mm")
    console.print(table)

    out_file = out_dir / "result.vtu"
    domain.save_state(filepath=out_file)
    console.print(f"\n[bold green]✓ Output successfully exported to:[/bold green] [cyan]{out_file}[/cyan]\n")


if __name__ == "__main__":
    main()
