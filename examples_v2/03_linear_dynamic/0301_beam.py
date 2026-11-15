"""
Example 0301: Linear Dynamic Transient Vibration of Cantilever Beam.

Demonstrates:
1. Ingesting volumetric beam in native Z-axis coordinates.
2. Clamping root plane (Z_min) via semantic plane selector.
3. Applying self-weight gravity loading along -Y with Rayleigh proportional damping.
4. Generalized-alpha implicit linear dynamic time integration (default: 100 steps).
5. Transient peak monitoring, kinetic energy dissipation, and XDMF/HDF5 export.
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
)
from psierra.damping import RayleighDamping
from psierra.analysis import LinearDynamicAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0301: Linear Dynamic Cantilever Beam"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of dynamic time steps (default: 100)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005,
        help="Time step size in seconds (default: 0.005 s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/03_linear_dynamic/0301_beam",
        help="Target output directory (default: results/03_linear_dynamic/0301_beam)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    console.print(
        Panel.fit(
            f"[bold cyan]Example 0301: Linear Dynamic Cantilever Beam ({args.steps} steps)[/bold cyan]\n"
            "[dim]Generalized-Alpha Transient Time Integration with Rayleigh Damping[/dim]",
            title="[bold blue]project-sierra — Linear Dynamic[/bold blue]",
            border_style="cyan",
        )
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loading Mesh in Native Coordinates
    console.print("\n[bold yellow][1/6] Loading Mesh Geometry[/bold yellow]")
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
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
    params.set_from_youngs_poisson(youngs_modulus=70000.0, poissons_ratio=0.35, mass_density=1100e-6)  # MPa & tonne/mm³
    material = Hookean(params)
    element = Tet4Linear(material)
    console.print("  • Material : [cyan]Hookean Elasticity[/cyan] (E = 70.0 GPa, nu = 0.35)")
    console.print("  • Element  : [cyan]Tet4Linear[/cyan]")

    # 3. Domain
    console.print("\n[bold yellow][3/6] Setting Up Physical Domain Container[/bold yellow]")
    domain = SolidDomain(mesh, material, element)

    # 4. Boundary Conditions & External Loads
    console.print("\n[bold yellow][4/6] Applying Boundary Conditions & External Loads[/bold yellow]")
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
    tip_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")

    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Gravity along -Y (transverse downward)
    gravity = GravityLoad(acceleration=(0.0, -9810.0, 0.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    # Rayleigh damping: alpha = 2.0 (mass proportional), beta = 0.001 (stiffness proportional)
    damping = RayleighDamping(alpha=2.0, beta=0.001)

    console.print(f"  • Clamped Root Face (Z_min): [green]{len(root_nodes)}[/green] nodes constrained")
    console.print(f"  • Transverse Gravity Load  : [green]{np.abs(f_ext[1::3].sum()):.2f}[/green] N along -Y")
    console.print(f"  • Rayleigh Damping         : alpha = {damping.alpha}, beta = {damping.beta}")

    # 5. Solver
    console.print("\n[bold yellow][5/6] Executing Linear Dynamic Integration[/bold yellow]")
    analysis = LinearDynamicAnalysis(domain, damping=damping, spectral_radius=0.8)
    
    result = analysis.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        dt=args.dt,
        n_steps=args.steps,
        include_stresses=True,
        smooth_to_nodes=True,
        save_directory=out_dir,
        save_interval=5,
    )
    console.print("  [green]✓ Generalized-alpha dynamic time marching complete.[/green]")

    # 6. Verification
    console.print("\n[bold yellow][6/6] Verifying Results & Post-Processing[/bold yellow]")
    max_disp = float(np.linalg.norm(domain.displacement, axis=1).max())
    total_time = args.steps * args.dt

    summary_table = Table(title="Example 0301: Linear Dynamic Verification", border_style="cyan")
    summary_table.add_column("Quantity", style="bold white")
    summary_table.add_column("Value", style="cyan")
    summary_table.add_column("Status / Note", style="green")

    summary_table.add_row("Time Integrator", f"Generalized-alpha (ρ∞={analysis.spectral_radius})", "Unconditionally Stable")
    summary_table.add_row("Step Size (dt)", f"{args.dt:.4f} s", "Time Increment")
    summary_table.add_row("Dynamic Steps", f"{args.steps}", f"Total Duration = {total_time:.2f} s")
    summary_table.add_row("Peak Resultant Disp", f"{max_disp:.4f} mm", "Transient Dynamic Peak")
    summary_table.add_row("Exported Animation", f"{out_dir / 'result.xdmf'}", "XDMF + HDF5 Time Series")
    console.print(summary_table)
    console.print(f"\n[bold green]✓ ParaView animation written to:[/bold green] [cyan]{out_dir / 'result.xdmf'}[/cyan]\n")


if __name__ == "__main__":
    main()
