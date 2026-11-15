"""
Example 0105: SolidDomain Lifecycle, Assembly, State Update, & VTU Export.

Demonstrates the canonical 6-stage lifecycle for continuum mechanics containers:
1. Ingesting native volumetric mesh (raw Z-axis orientation).
2. Configuring material parameters (Hookean / Neo-Hookean).
3. Instantiating physical SolidDomain container.
4. One-line global stiffness & mass matrix assembly.
5. Updating domain state (nodal displacements & element von Mises stress field).
6. Exporting deformed state directly to ParaView VTU.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear
from psierra.domain import SolidDomain
from psierra.mesh import VolumetricMesh


def main() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Example 0105: SolidDomain Container Lifecycle[/bold cyan]\n"
            "[dim]Composability Chain: Parameters -> Material -> Element -> Domain -> State Export[/dim]",
            title="[bold blue]project-sierra — Basics[/bold blue]",
            border_style="cyan",
        )
    )

    out_dir = Path("results/01_basics/0105_solid_domain")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loading Mesh in Native Coordinates
    console.print("\n[bold yellow][1/6] Loading Mesh Geometry[/bold yellow]")
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found at: {mesh_path}")
    
    mesh = VolumetricMesh(mesh_path)
    b = mesh.bounds
    console.print(f"  • Ingested Native Mesh : [green]{mesh.n_nodes}[/green] nodes | [green]{mesh.n_elements}[/green] elements")
    console.print(f"  • Length along Z-axis  : [{b[4]:.1f}, {b[5]:.1f}] mm (Total span: {b[5] - b[4]:.1f} mm)")

    # 2. Material & Element
    console.print("\n[bold yellow][2/6] Configuring Material & Element Formulation[/bold yellow]")
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(youngs_modulus=68.0e3, poissons_ratio=0.33, mass_density=2700e-6)  # MPa & tonne/mm³
    material = Hookean(params)
    element = Tet4Linear(material)
    console.print("  • Material : [cyan]Hookean Linear Elasticity[/cyan] (E = 68.0 GPa, nu = 0.33)")
    console.print("  • Element  : [cyan]Tet4Linear[/cyan] (Constant Strain Tetrahedron)")

    # 3. Domain
    console.print("\n[bold yellow][3/6] Setting Up Physical Domain Container[/bold yellow]")
    domain = SolidDomain(mesh, material, element)
    console.print(f"  • Domain DOFs          : [green]{domain.n_dofs}[/green]")

    # 4. Assembly in one line
    console.print("\n[bold yellow][4/6] One-Line Global Matrix Assembly[/bold yellow]")
    K = domain.assemble_stiffness_matrix()
    M = domain.assemble_mass_matrix()
    console.print(f"  • Assembled K matrix   : {K.shape} ({K.nnz} non-zeros)")
    console.print(f"  • Assembled M matrix   : {M.shape} ({M.nnz} non-zeros)")

    # 5. State update (Parabolic bending along -Y as function of native Z^2)
    console.print("\n[bold yellow][5/6] Updating Physical Kinematic State[/bold yellow]")
    z_span = b[5] - b[4]
    z_norm = (mesh.nodes[:, 2] - b[4]) / z_span  # 0 at root, 1 at tip
    
    synthetic_disp = np.zeros((mesh.n_nodes, 3))
    synthetic_disp[:, 1] = -15.0 * (z_norm ** 2)  # -15 mm tip sag along Y
    
    # Synthetic element von Mises stress
    synthetic_vonmises = 120.0 * np.abs(mesh.centroids[:, 2] / b[5])  # MPa
    
    domain.update_state(u=synthetic_disp.ravel())
    console.print(f"  • Maximum Deflection   : [green]{np.abs(synthetic_disp[:, 1]).max():.2f}[/green] mm along -Y")

    # 6. State export
    console.print("\n[bold yellow][6/6] Verifying Results & State Export[/bold yellow]")
    out_file = out_dir / "result.vtu"
    domain.save_state(filepath=out_file, cell_data={"von_mises_stress": synthetic_vonmises})
    console.print(f"  [bold green]✓ Deformed state written to:[/bold green] [cyan]{out_file}[/cyan]\n")


if __name__ == "__main__":
    main()
