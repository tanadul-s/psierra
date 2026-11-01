from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install
from scipy.spatial.transform import Rotation

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary.selector import select_nodes_by_bounds
from psierra.boundary import (
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
)
from psierra.analysis import NonlinearStaticAnalysis

install(show_locals=False)
console = Console()


def main() -> None:
    # 0. Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]Simulation 06: Nonlinear Static Analysis (Neo-Hookean)[/bold cyan]\n"
            "[dim]Finite Strain Hyperelastic Beam Bending under Gravitational Body Load[/dim]",
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
    material = NeoHookean(params)
    element = Tet4TotalLagrangian(material)
    console.print(f"  • Constitutive Law : [cyan]{material.__class__.__name__}[/cyan] (Compressible Neo-Hookean)")
    console.print(f"  • Element Type     : [cyan]{element.__class__.__name__}[/cyan] (Total Lagrangian Tetrahedron)")
    console.print(f"  • Young's Modulus  : [green]{params.youngs_modulus:.1f}[/green] MPa | Poisson's Ratio: [green]{params.poissons_ratio:.2f}[/green]")

    # 3. Setting Up Physical Domain
    console.print("\n[bold yellow][3/6] Setting Up Physical Continuum Domain[/bold yellow]")
    domain = SolidDomain(mesh, material, element)

    # 4. Applying Boundary Conditions & External Loads
    console.print("\n[bold yellow][4/6] Applying Boundary Conditions & External Loads[/bold yellow]")
    root_nodes = select_nodes_by_bounds(mesh, bounds=(-61, 61, -31, 1, -61, 61))
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    total_weight = float(f_ext[2::3].sum())
    console.print(f"  • Clamped Root     : [green]{len(root_nodes)}[/green] nodes ({len(clamp.dofs)} DOFs)")
    console.print(f"  • Gravity Load F_z : [green]{total_weight:.2f}[/green] N")

    # 5. Executing Simulation Solver
    console.print("\n[bold yellow][5/6] Executing Simulation Solver[/bold yellow]")
    analysis = NonlinearStaticAnalysis(domain)
    result = analysis.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        n_steps=10,
        max_iterations=20,
        tolerance=1e-4,
    )

    # 6. Verifying Results & Post-Processing
    console.print("\n[bold yellow][6/6] Verifying Results & Post-Processing[/bold yellow]")
    disp = domain.displacement
    max_disp = float(np.linalg.norm(disp, axis=1).max())
    tip_z = float(disp[:, 2].min())
    tip_y = float(disp[:, 1].min())

    status_style = "green" if result.converged else "red"

    summary_table = Table(title="Simulation 06 Summary & Verification", border_style="cyan")
    summary_table.add_column("Quantity", style="bold white")
    summary_table.add_column("Value", style="cyan")
    summary_table.add_column("Status / Unit", style="green")

    summary_table.add_row("Convergence Status", f"{result.converged}", f"[bold {status_style}]{result.steps_completed}/10 Steps[/bold {status_style}]")
    summary_table.add_row("Max Resultant Displacement", f"{max_disp:.2f} mm", "Peak Deflection")
    summary_table.add_row("Downward Deflection (Z)", f"{tip_z:.2f} mm", "Bending Deflection")
    summary_table.add_row("Axial Shortening (Y)", f"{tip_y:.2f} mm", "Geometric Inward Pull")
    summary_table.add_row("Total External Load (F_z)", f"{total_weight:.2f} N", "Applied Load")

    console.print(summary_table)

    output_dir = Path("examples/05_simulation/sim06_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.vtu"
    domain.save_state(filepath=output_path, include_stresses=True, smooth_to_nodes=True)
    console.print(f"[bold green]✓ Output successfully exported to:[/bold green] [cyan]{output_path}[/cyan]\n")


if __name__ == "__main__":
    main()

