"""
Example 0402: Static Nonlinear Neurosurgical Brain Shift under Gravity.

Demonstrates:
1. Ingesting MRI-segmented patient brain mesh (18,397 nodes, 90,586 elements) in native RAS coordinates.
2. Clamping cranial base and posterior fossa support.
3. Simulating static anatomical brain shift across surgical patient postures:
   - Supine: Gravity pulling posteriorly along -Z
   - Prone: Gravity pulling anteriorly along +Z
   - Left Lateral: Gravity pulling along -X
   - Right Lateral: Gravity pulling along +X
4. Compressible Neo-Hookean hyperelastic parenchymal tissue model.
5. Incremental Newton-Raphson nonlinear static solve with backtracking line search.
6. Exporting deformed anatomical state to ParaView VTU.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import DirichletConstraint, DirichletManager, GravityLoad
from psierra.analysis import NonlinearStaticAnalysis

SURGICAL_POSTURES = {
    "supine": (0.0, -9810.0, 0.0, "Supine (Posterior sag along -Y)"),
    "prone": (0.0, 9810.0, 0.0, "Prone (Anterior sag along +Y)"),
    "left_lateral": (-9810.0, 0.0, 0.0, "Left Lateral (Sag along -X)"),
    "right_lateral": (9810.0, 0.0, 0.0, "Right Lateral (Sag along +X)"),
    "upright": (0.0, 0.0, -9810.0, "Sitting / Upright (Caudal sag along -Z)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0402: Static Nonlinear Brain Shift"
    )
    parser.add_argument(
        "--posture",
        type=str,
        choices=list(SURGICAL_POSTURES.keys()),
        default="supine",
        help="Surgical patient posture controlling gravity vector direction (default: supine)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of incremental load steps (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/04_nonlinear_static/0402_brain",
        help="Base output directory (default: results/04_nonlinear_static/0402_brain)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    gx, gy, gz, posture_desc = SURGICAL_POSTURES[args.posture]

    console.print(
        Panel.fit(
            f"[bold cyan]Example 0402: Static Brain Shift ({args.posture.upper()})[/bold cyan]\n"
            f"[white]{posture_desc}[/white]\n"
            "[dim]Finite-Deformation Hyperelastic Parenchyma & Incremental Newton-Raphson[/dim]",
            title="[bold blue]project-sierra — Nonlinear Static[/bold blue]",
            border_style="cyan",
        )
    )

    out_dir = Path(args.output_dir) / args.posture
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loading Anatomical Brain Mesh
    console.print("\n[bold yellow][1/6] Loading Anatomical Brain Mesh[/bold yellow]")
    brain_mesh_path = Path("data/mrbraintumor1/mesh/brain.vtk")
    if not brain_mesh_path.exists():
        raise FileNotFoundError(f"Brain mesh not found: {brain_mesh_path}")

    mesh = VolumetricMesh(brain_mesh_path)
    b = mesh.bounds
    console.print(f"  • Source Mesh File  : [cyan]{brain_mesh_path}[/cyan]")
    console.print(
        f"  • Nodes / Elements  : [green]{mesh.n_nodes}[/green] nodes | [green]{mesh.n_elements}[/green] Tet4 elements"
    )
    console.print(
        f"  • Volume            : [green]{mesh.volume * 1e-3:.1f}[/green] cm³"
    )

    # 2. Material & Element (E = 3.0 kPa, nu = 0.45, rho = 1040 kg/m^3)
    console.print(
        "\n[bold yellow][2/6] Configuring Hyperelastic Parenchyma Properties[/bold yellow]"
    )
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=3.0e-3,  # 3.0 kPa in MPa
        poissons_ratio=0.45,
        mass_density=1040.0 * 1e-12,  # tonne/mm³
    )
    material = NeoHookean(params)
    element = Tet4TotalLagrangian(material)
    console.print(
        "  • Material : [cyan]Compressible Neo-Hookean[/cyan] (E = 3.0 kPa, nu = 0.45)"
    )
    console.print("  • Element  : [cyan]Tet4TotalLagrangian[/cyan]")

    # 3. Domain
    console.print(
        "\n[bold yellow][3/6] Setting Up Physical Domain Container[/bold yellow]"
    )
    domain = SolidDomain(mesh, material, element)

    # 4. Boundary Conditions & Gravitational Vector
    console.print(
        "\n[bold yellow][4/6] Applying Cranial Base Anchors & Posture Gravity[/bold yellow]"
    )
    clamped_node_mask = mesh.nodes[:, 1] < -20.0
    clamped_nodes = np.where(clamped_node_mask)[0]

    clamp = DirichletConstraint(clamped_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    gravity = GravityLoad(acceleration=(gx, gy, gz))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    console.print(f"  • Cranial Base Fixed Nodes : [green]{len(clamped_nodes)}[/green]")
    console.print(
        f"  • Posture Gravity Vector   : gx={gx:.0f}, gy={gy:.0f}, gz={gz:.0f} mm/s²"
    )

    # 5. Solver
    console.print(
        "\n[bold yellow][5/6] Executing Incremental Newton-Raphson Solver[/bold yellow]"
    )
    solver = NonlinearStaticAnalysis(domain)

    t0 = time.perf_counter()
    solver.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        n_steps=args.steps,
        nr_iterations=20,
        nr_tol=1e-3,
        verbose=True,
    )
    wall_time = time.perf_counter() - t0
    console.print(
        f"  [green]✓ Static nonlinear equilibrium reached in {wall_time:.2f} s.[/green]"
    )

    # 6. Verification
    console.print("\n[bold yellow][6/6] Verifying Results & State Export[/bold yellow]")
    disp = domain.displacement
    max_disp = float(np.linalg.norm(disp, axis=1).max())

    table = Table(
        title=f"Example 0402: Static Brain Shift ({args.posture.upper()})",
        border_style="cyan",
    )
    table.add_column("Quantity", style="bold white")
    table.add_column("Value", style="cyan")
    table.add_row("Surgical Posture", args.posture.upper())
    table.add_row("Gravity Vector (mm/s²)", f"({gx:.0f}, {gy:.0f}, {gz:.0f})")
    table.add_row("Load Increments", f"{args.steps} steps")
    table.add_row("Peak Parenchymal Sag", f"{max_disp:.4f} mm")
    table.add_row("Solver Runtime", f"{wall_time:.2f} s")
    console.print(table)

    out_file = out_dir / "result.vtu"
    domain.save_state(filepath=out_file)
    console.print(
        f"\n[bold green]✓ Deformed anatomical state saved to:[/bold green] [cyan]{out_file}[/cyan]\n"
    )


if __name__ == "__main__":
    main()
