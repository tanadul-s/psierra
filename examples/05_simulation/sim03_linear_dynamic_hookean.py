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
from psierra.boundary.selector import select_nodes_by_bounds
from psierra.boundary import (
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
)
from psierra.damping import RayleighDamping
from psierra.analysis import LinearDynamicAnalysis

install(show_locals=False)
console = Console()


def main() -> None:
    # 0. Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]Simulation 03: Linear Dynamic Analysis (Hookean)[/bold cyan]\n"
            "[dim]Dynamic Cantilever Beam Transient Vibration with Rayleigh Damping[/dim]",
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
        youngs_modulus=70000.0,
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
    root_nodes = select_nodes_by_bounds(mesh, bounds=(-61, 61, -31, 1, -61, 61))
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    # Rayleigh Damping: alpha = 2.0 (mass damping), beta = 0.001 (stiffness damping)
    damping = RayleighDamping(alpha=2.0, beta=0.001)

    console.print(f"  • Clamped Root     : [green]{len(root_nodes)}[/green] nodes ({len(clamp.dofs)} DOFs)")
    console.print(f"  • Gravity Load F_z : [green]{f_ext[2::3].sum():.2f}[/green] N")
    console.print(f"  • Rayleigh Damping : alpha = [green]{damping.alpha}[/green] | beta = [green]{damping.beta}[/green]")

    # 5. Executing Simulation Solver
    console.print("\n[bold yellow][5/6] Executing Simulation Solver[/bold yellow]")
    output_dir = Path("examples/05_simulation/sim03_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    dt = 0.005
    n_steps = 100
    analysis = LinearDynamicAnalysis(domain, damping=damping)
    result = analysis.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        dt=dt,
        n_steps=n_steps,
        include_stresses=True,
        smooth_to_nodes=True,
        save_directory=output_dir,
        save_interval=5,
    )

    # 6. Verifying Results & Post-Processing
    console.print("\n[bold yellow][6/6] Verifying Results & Post-Processing[/bold yellow]")
    max_disp = float(np.linalg.norm(domain.displacement, axis=1).max())
    total_time = n_steps * dt

    summary_table = Table(title="Simulation 03 Summary & Verification", border_style="cyan")
    summary_table.add_column("Quantity", style="bold white")
    summary_table.add_column("Value", style="cyan")
    summary_table.add_column("Status / Unit", style="green")

    summary_table.add_row("Time Step Size (dt)", f"{dt:.4f} s", "Newmark Time Increment")
    summary_table.add_row("Total Time Steps", f"{n_steps}", f"Duration = {total_time:.2f} s")
    summary_table.add_row("Max Resultant Displacement", f"{max_disp:.4f} mm", "Transient Dynamic Peak")
    summary_table.add_row("Exported VTU Snapshots", f"{len(result.time_series_files)}", "PVD Collection Items")

    console.print(summary_table)
    console.print(f"[bold green]✓ ParaView animation written to:[/bold green] [cyan]{output_dir / "output.pvd"}[/cyan]\n")


if __name__ == "__main__":
    main()

