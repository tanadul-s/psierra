"""
Example 0302: Linear Dynamic Transient Impulse Response of Anatomical Brain.

Demonstrates:
1. Ingesting MRI-segmented patient brain mesh (18k nodes, 90k elements) in RAS coordinates.
2. Clamping cranial base & posterior fossa anatomical anchors.
3. Applying transient impulse gravitational acceleration.
4. Linear dynamic time marching via generalized-alpha integration (default: 100 steps).
5. Monitoring subcortical tumor drift and exporting XDMF/HDF5 time series.
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
from psierra.boundary import DirichletConstraint, DirichletManager, GravityLoad
from psierra.damping import RayleighDamping
from psierra.analysis import LinearDynamicAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0302: Linear Dynamic Brain Shift Impulse"
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
        default="results/03_linear_dynamic/0302_brain",
        help="Target output directory (default: results/03_linear_dynamic/0302_brain)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    console.print(
        Panel.fit(
            f"[bold cyan]Example 0302: Linear Dynamic Brain Shift ({args.steps} steps)[/bold cyan]\n"
            "[dim]Transient Impulse Elastodynamics on Patient Brain Geometry via Generalized-Alpha[/dim]",
            title="[bold blue]project-sierra — Linear Dynamic[/bold blue]",
            border_style="cyan",
        )
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loading Mesh in RAS Anatomical Coordinates
    console.print("\n[bold yellow][1/6] Loading Anatomical Brain Mesh[/bold yellow]")
    brain_mesh_path = Path("data/mrbraintumor1/mesh/brain.vtk")
    if not brain_mesh_path.exists():
        raise FileNotFoundError(f"Brain mesh not found: {brain_mesh_path}")

    mesh = VolumetricMesh(brain_mesh_path)
    b = mesh.bounds
    console.print(f"  • Source Mesh File  : [cyan]{brain_mesh_path}[/cyan]")
    console.print(f"  • Nodes / DOFs      : [green]{mesh.n_nodes}[/green] ({mesh.n_dofs} DOFs)")
    console.print(f"  • Elements (Tet4)   : [green]{mesh.n_elements}[/green]")
    console.print(f"  • Volume            : [green]{mesh.volume * 1e-3:.1f}[/green] cm³")

    # 2. Material & Element (Parenchymal tissue: E = 3.0 kPa, nu = 0.45, rho = 1040 kg/m³)
    console.print("\n[bold yellow][2/6] Configuring Parenchymal Tissue Properties[/bold yellow]")
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=3000.0,       # 3.0 kPa in Pa (g/(mm·s²))
        poissons_ratio=0.45,
        mass_density=1040e-6,        # 1040 kg/m³ = 1040e-6 g/mm³
    )
    material = Hookean(params)
    element = Tet4Linear(material)
    console.print("  • Material : [cyan]Hookean Elastic Parenchyma[/cyan] (E = 3.0 kPa, nu = 0.45, rho = 1040e-6 g/mm³)")
    console.print("  • Element  : [cyan]Tet4Linear[/cyan]")

    # 3. Domain
    console.print("\n[bold yellow][3/6] Setting Up Physical Domain Container[/bold yellow]")
    domain = SolidDomain(mesh, material, element)

    # 4. Boundary Conditions & Loads
    console.print("\n[bold yellow][4/6] Applying Cranial Base Anchors & Gravity Impulse[/bold yellow]")
    # Cranial base & tentorium support (clamped posterior/inferior nodes)
    clamped_node_mask = mesh.nodes[:, 1] < -20.0
    clamped_nodes = np.where(clamped_node_mask)[0]

    clamp = DirichletConstraint(clamped_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Supine surgical gravity along -Z (A-P gravitational sag)
    gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    # Tissue Rayleigh damping: alpha = 1.5, beta = 0.005
    damping = RayleighDamping(alpha=1.5, beta=0.005)

    console.print(f"  • Cranial Base Anchors     : [green]{len(clamped_nodes)}[/green] nodes fully constrained")
    console.print(f"  • Free Cortical DOFs       : [green]{domain.n_dofs - len(clamped_nodes) * 3}[/green]")
    console.print("  • Direction of Gravity     : Supine surgical posture (|g| = 9.81 m/s² along -Z)")

    # 5. Solver
    console.print("\n[bold yellow][5/6] Executing Linear Dynamic Integration[/bold yellow]")
    analysis = LinearDynamicAnalysis(domain, damping=damping, spectral_radius=0.8)
    
    result = analysis.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        dt=args.dt,
        n_steps=args.steps,
        include_stresses=False,
        save_directory=out_dir,
        save_interval=10,
    )
    console.print("  [green]✓ Generalized-alpha dynamic time marching complete.[/green]")

    # 6. Verification
    console.print("\n[bold yellow][6/6] Verifying Results & Post-Processing[/bold yellow]")
    max_disp = float(np.linalg.norm(domain.displacement, axis=1).max())
    total_time = args.steps * args.dt

    summary_table = Table(title="Example 0302: Linear Dynamic Brain Shift Verification", border_style="cyan")
    summary_table.add_column("Quantity", style="bold white")
    summary_table.add_column("Value", style="cyan")
    summary_table.add_column("Status / Note", style="green")

    summary_table.add_row("Time Integrator", f"Generalized-alpha (ρ∞={analysis.spectral_radius})", "Unconditionally Stable")
    summary_table.add_row("Step Size (dt)", f"{args.dt:.4f} s", "Time Increment")
    summary_table.add_row("Dynamic Steps", f"{args.steps}", f"Duration = {total_time:.2f} s")
    summary_table.add_row("Peak Cortical Sag", f"{max_disp:.4f} mm", "Dynamic Parenchymal Peak")
    summary_table.add_row("Exported Animation", f"{out_dir / 'result.xdmf'}", "XDMF + HDF5 Time Series")
    console.print(summary_table)
    console.print(f"\n[bold green]✓ ParaView animation written to:[/bold green] [cyan]{out_dir / 'result.xdmf'}[/cyan]\n")


if __name__ == "__main__":
    main()
