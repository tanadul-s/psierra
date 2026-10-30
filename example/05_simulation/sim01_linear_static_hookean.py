from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install
from scipy.spatial.transform import Rotation

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import (
    select_nodes_by_plane,
    DirichletConstraint,
    DirichletManager,
    NodalForce,
)
from psierra.analysis import LinearStaticAnalysis

install(show_locals=False)
console = Console()


def main() -> None:
    # 0. Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]Simulation 01: Linear Static Analysis (Hookean)[/bold cyan]\n"
            "[dim]3D Cantilever Beam Static Bending under Distributed Tip Load[/dim]",
            title="[bold blue]project-sierra[/bold blue]",
            border_style="cyan",
        )
    )

    # 1. Loading Mesh Geometry
    console.print("\n[bold yellow][1/6] Loading Mesh Geometry[/bold yellow]")
    mesh_path = Path("data/beam1/mesh/beam1_fine.vtk")
    mesh = VolumetricMesh(mesh_path)
    mesh.transform(
        orientation=Rotation.from_euler("zyz", (0, 90, 90), degrees=True),
        about_point=(0, 0, 0),
    )
    console.print(
        f"  • Nodes: [green]{mesh.n_nodes}[/green] | DoFs: [green]{mesh.n_dofs}[/green] | Elements: [green]{mesh.n_elements}[/green]"
    )
    console.print(
        f"  • Volume: [green]{mesh.volume:.2f}[/green] mm³ | Centroid: [dim]{tuple(float(c) for c in np.round(mesh.centroid, 2))}[/dim]"
    )

    # 2. Configuring Material & Element Formulation
    console.print("\n[bold yellow][2/6] Configuring Material & Element Formulation[/bold yellow]")
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000.0,
        poissons_ratio=0.35,
        mass_density=1100e-6,
    )
    material = Hookean(params)
    element = Tet4(material)
    console.print(f"  • Constitutive Law : [cyan]{material.__class__.__name__}[/cyan]")
    console.print(f"  • Element Type     : [cyan]{element.__class__.__name__}[/cyan] (Linear Tetrahedron)")
    console.print(f"  • Young's Modulus  : [green]{params.youngs_modulus:.1f}[/green] MPa | Poisson's Ratio: [green]{params.poissons_ratio:.2f}[/green]")

    # 3. Setting Up Physical Domain
    console.print("\n[bold yellow][3/6] Setting Up Physical Continuum Domain[/bold yellow]")
    domain = SolidDomain(mesh, material, element)

    # 4. Applying Boundary Conditions & External Loads
    console.print("\n[bold yellow][4/6] Applying Boundary Conditions & External Loads[/bold yellow]")
    root_nodes = select_nodes_by_plane(mesh, plane="xz", bound="min")
    clamp = DirichletConstraint(
        node_indices=root_nodes, components=(0, 1, 2), value=0.0
    )
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    tip_nodes = select_nodes_by_plane(mesh, plane="xz", bound="max")
    total_force_z = -150e+3  # -150 kN in -Z
    force_per_node = total_force_z / len(tip_nodes)
    tip_load = NodalForce(
        node_indices=tip_nodes,
        force_vector=[0.0, 0.0, force_per_node],
    )
    f_ext = tip_load.compute_force_vector(total_dofs=domain.mesh.n_dofs)

    console.print(f"  • Clamped Root     : [green]{len(root_nodes)}[/green] nodes ({len(clamp.dofs)} DOFs)")
    console.print(f"  • Distributed Tip  : [green]{len(tip_nodes)}[/green] nodes | Total F_z = [green]{total_force_z:.1f}[/green] N")

    # 5. Executing Simulation Solver
    console.print("\n[bold yellow][5/6] Executing Simulation Solver[/bold yellow]")
    analysis = LinearStaticAnalysis(domain)
    result = analysis.solve(dirichlet=dirichlet, f_ext=f_ext)

    # 6. Verifying Results & Post-Processing
    console.print("\n[bold yellow][6/6] Verifying Results & Post-Processing[/bold yellow]")
    max_disp = float(np.linalg.norm(domain.displacement, axis=1).max())
    tip_disp_z = float(domain.displacement[tip_nodes, 2].mean())
    total_reaction_z = float(result.reactions[clamp.dofs][2::3].sum())
    equilibrium_error = abs(total_reaction_z + total_force_z)
    eq_pass = equilibrium_error < 1e-5
    status_style = "green" if eq_pass else "red"

    summary_table = Table(title="Simulation 01 Summary & Verification", border_style="cyan")
    summary_table.add_column("Quantity", style="bold white")
    summary_table.add_column("Value", style="cyan")
    summary_table.add_column("Status / Unit", style="green")

    summary_table.add_row("Max Resultant Displacement", f"{max_disp:.4f} mm", "Peak Deflection")
    summary_table.add_row("Mean Tip Z-Deflection", f"{tip_disp_z:.4f} mm", "Downward")
    summary_table.add_row("Total Elastic Strain Energy", f"{result.strain_energy:.4e} J", "Strain Energy")
    summary_table.add_row("Total Reaction Force (R_z)", f"{total_reaction_z:.2f} N", "Support Reaction")
    summary_table.add_row("Total Applied Force (F_z)", f"{total_force_z:.2f} N", "Applied Load")
    summary_table.add_row("Static Equilibrium Error", f"{equilibrium_error:.2e} N", f"[bold {status_style}]PASS: {eq_pass}[/bold {status_style}]")

    console.print(summary_table)

    output_dir = Path("example/05_simulation/sim01_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.vtu"
    domain.save_state(output_path)
    console.print(f"[bold green]✓ Output successfully exported to:[/bold green] [cyan]{output_path}[/cyan]\n")


if __name__ == "__main__":
    main()

