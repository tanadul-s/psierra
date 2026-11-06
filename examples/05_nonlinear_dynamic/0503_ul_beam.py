from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install
from scipy.spatial.transform import Rotation

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4UpdatedLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary.selector import select_nodes_by_bounds
from psierra.boundary import (
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
)
from psierra.damping import RayleighDamping
from psierra.analysis import NonlinearDynamicAnalysis

install(show_locals=False)
console = Console()


def main() -> None:
    # 0. Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]Simulation 13: Nonlinear Dynamic Analysis (Updated Lagrangian + Neo-Hookean)[/bold cyan]\n"
            "[dim]Transient Gravity-Induced Vibration of Hyperelastic Beam via Updated Lagrangian[/dim]",
            title="[bold blue]project-sierra[/bold blue]",
            border_style="cyan",
        )
    )

    # 1. Loading Mesh Geometry
    console.print("\n[bold yellow][1/6] Loading Mesh Geometry[/bold yellow]")
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
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
    console.print(
        "\n[bold yellow][2/6] Configuring Material & Element Formulation[/bold yellow]"
    )
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000.0,
        poissons_ratio=0.35,
        mass_density=1100e-6,
    )
    material = NeoHookean(params)
    element = Tet4UpdatedLagrangian(material)
    console.print(
        f"  • Constitutive Law : [cyan]{material.__class__.__name__}[/cyan] (Compressible Neo-Hookean)"
    )
    console.print(
        f"  • Element Type     : [cyan]{element.__class__.__name__}[/cyan] (Updated Lagrangian Tetrahedron)"
    )
    console.print(
        f"  • Young's Modulus  : [green]{params.youngs_modulus:.1f}[/green] MPa | Poisson's Ratio: [green]{params.poissons_ratio:.2f}[/green]"
    )

    # 3. Setting Up Physical Domain
    console.print(
        "\n[bold yellow][3/6] Setting Up Physical Continuum Domain[/bold yellow]"
    )
    domain = SolidDomain(mesh, material, element)

    # 4. Applying Boundary Conditions & External Loads
    console.print(
        "\n[bold yellow][4/6] Applying Boundary Conditions & External Loads[/bold yellow]"
    )
    root_nodes = select_nodes_by_bounds(mesh, bounds=(-61, 61, -31, 1, -61, 61))
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    total_weight = float(f_ext[2::3].sum())
    console.print(
        f"  • Clamped Root     : [green]{len(root_nodes)}[/green] nodes ({len(clamp.dofs)} DOFs)"
    )
    console.print(f"  • Gravity Load F_z : [green]{total_weight:.2f}[/green] N")

    # 5. Executing Simulation Solver
    console.print("\n[bold yellow][5/6] Executing Simulation Solver[/bold yellow]")
    damping = RayleighDamping(alpha=1.0, beta=0.001)
    analysis = NonlinearDynamicAnalysis(domain, damping=damping, spectral_radius=0.8)
    console.print(
        f"  • Time Integrator  : [cyan]Generalized-alpha[/cyan] (spectral_radius = [green]{analysis.spectral_radius}[/green])"
    )

    output_dir = Path("results/05_nonlinear_dynamic/0503_ul_beam")
    dt = 0.005
    n_steps = 1000
    result = analysis.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        dt=dt,
        n_steps=n_steps,
        nr_iterations=15,
        nr_tol=1e-4,
        steady_state_tol=1e-4,
        kinetic_energy_tol=1e-4,
        consecutive_hold_steps=3,
        include_stresses=True,
        smooth_to_nodes=True,
        save_interval=3,
        save_directory=output_dir,
    )

    # 6. Verifying Results & Post-Processing
    console.print(
        "\n[bold yellow][6/6] Verifying Results & Post-Processing[/bold yellow]"
    )
    disp = domain.displacement
    max_disp = float(np.linalg.norm(disp, axis=1).max())
    tip_z = float(disp[:, 2].min())
    total_time = result.steps_completed * dt

    status_style = "green" if result.converged else "red"

    summary_table = Table(
        title="Simulation 13 Summary & Verification", border_style="cyan"
    )
    summary_table.add_column("Quantity", style="bold white")
    summary_table.add_column("Value", style="cyan")
    summary_table.add_column("Status / Unit", style="green")

    summary_table.add_row(
        "Time Integration Scheme", f"Generalized-alpha (ρ∞={analysis.spectral_radius})", "Optimal Damping"
    )
    summary_table.add_row(
        "Convergence Status",
        f"{result.converged}",
        f"[bold {status_style}]{result.steps_completed}/{n_steps} Steps[/bold {status_style}]",
    )
    summary_table.add_row(
        "Duration (Total Time)", f"{total_time:.2f} s", f"{result.steps_completed} Steps @ dt = {dt:.4f}s"
    )
    summary_table.add_row(
        "Max Resultant Displacement", f"{max_disp:.2f} mm", "Dynamic Peak"
    )
    summary_table.add_row("Downward Deflection (Z)", f"{tip_z:.2f} mm", "Transient Sag")
    summary_table.add_row(
        "Exported VTU Snapshots",
        f"{len(result.time_series_files)}",
        "PVD Collection Items",
    )
    summary_table.add_row(
        "Total External Load (F_z)", f"{total_weight:.2f} N", "Applied Load"
    )

    console.print(summary_table)
    console.print(
        f"[bold green]✓ Transient time-series PVD exported to:[/bold green] [cyan]{output_dir / "result.pvd"}[/cyan]\n"
    )


if __name__ == "__main__":
    main()
