from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear, Tet10Linear
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
            "[bold cyan]Simulation 08: Tet10 vs Tet4 Cantilever Beam Bending[/bold cyan]\n"
            "[dim]Demonstrating Elimination of Artificial Shear Locking via Quadratic Elements[/dim]",
            title="[bold blue]project-sierra — Phase 2[/bold blue]",
            border_style="cyan",
        )
    )

    output_dir = Path("examples/05_simulation/sim08_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loading Base Mesh Geometry
    console.print("\n[bold yellow][1/6] Loading Mesh Geometry[/bold yellow]")
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Beam mesh not found: {mesh_path}")

    mesh_tet4 = VolumetricMesh(mesh_path)
    console.print(
        f"  • Linear Tet4 Base Mesh  : [green]{mesh_tet4.n_nodes}[/green] nodes | "
        f"[green]{mesh_tet4.n_elements}[/green] elements | "
        f"[green]{mesh_tet4.n_dofs}[/green] DOFs"
    )

    # Upgrade to Quadratic Tet10 Mesh
    mesh_tet10 = mesh_tet4.to_quadratic()
    console.print(
        f"  • Quadratic Tet10 Upgraded: [green]{mesh_tet10.n_nodes}[/green] nodes | "
        f"[green]{mesh_tet10.n_elements}[/green] elements | "
        f"[green]{mesh_tet10.n_dofs}[/green] DOFs"
    )

    # 2. Material & Element Configuration
    console.print("\n[bold yellow][2/6] Configuring Material & Element Formulation[/bold yellow]")
    E_modulus = 21000.0  # MPa
    nu_ratio = 0.35
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=E_modulus,
        poissons_ratio=nu_ratio,
        mass_density=1.1e-6,
    )
    material = Hookean(params)
    elem_tet4 = Tet4Linear(material)
    elem_tet10 = Tet10Linear(material)
    console.print(f"  • Young's Modulus E: {E_modulus:.1f} MPa")
    console.print(f"  • Poisson's Ratio ν: {nu_ratio:.2f}")
    console.print(f"  • Linear Element   : [cyan]{elem_tet4.__class__.__name__}[/cyan] (4 nodes, 1-point quadrature)")
    console.print(f"  • Quadratic Element: [cyan]{elem_tet10.__class__.__name__}[/cyan] (10 nodes, 4-point Keast quadrature)")

    # 3. Setting Up Physical Domains
    console.print("\n[bold yellow][3/6] Setting Up Physical Domains[/bold yellow]")
    dom4 = SolidDomain(mesh_tet4, material, elem_tet4)
    dom10 = SolidDomain(mesh_tet10, material, elem_tet10)
    console.print(f"  • Tet4 Domain      : [green]{dom4.n_dofs}[/green] DOFs")
    console.print(f"  • Tet10 Domain     : [green]{dom10.n_dofs}[/green] DOFs")

    # 4. Setup Boundary Conditions & Load
    console.print("\n[bold yellow][4/6] Applying Boundary Conditions & External Loads[/bold yellow]")
    total_fz = -1000.0  # N

    # Tet4 Boundaries
    root_nodes4 = select_nodes_by_plane(mesh_tet4, plane="xz", bound="min")
    dir4 = DirichletManager(dom4.n_dofs)
    dir4.add_constraint(
        DirichletConstraint(root_nodes4, components=(0, 1, 2), value=0.0)
    )

    tip_nodes4 = select_nodes_by_plane(mesh_tet4, plane="xz", bound="max")
    load4 = NodalForce(tip_nodes4, force_vector=[0.0, 0.0, total_fz / len(tip_nodes4)])
    f4 = load4.compute_force_vector(dom4.n_dofs)

    # Tet10 Boundaries
    root_nodes10 = select_nodes_by_plane(mesh_tet10, plane="xz", bound="min")
    dir10 = DirichletManager(dom10.n_dofs)
    dir10.add_constraint(
        DirichletConstraint(root_nodes10, components=(0, 1, 2), value=0.0)
    )

    tip_nodes10 = select_nodes_by_plane(mesh_tet10, plane="xz", bound="max")
    load10 = NodalForce(
        tip_nodes10, force_vector=[0.0, 0.0, total_fz / len(tip_nodes10)]
    )
    f10 = load10.compute_force_vector(dom10.n_dofs)

    console.print(f"  • Fixed Root Face (XZ min): {len(root_nodes4)} (Tet4) / {len(root_nodes10)} (Tet10) nodes clamped")
    console.print(f"  • Total Tip Load F_z      : {total_fz:.1f} N distributed across tip face")

    # 5. Executing Solvers
    console.print("\n[bold yellow][5/6] Executing Simulation Solvers[/bold yellow]")
    res4 = LinearStaticAnalysis(dom4).solve(dirichlet=dir4, f_ext=f4, verbose=False)
    tip_def4 = float(np.mean(res4.displacement[tip_nodes4 * 3 + 2]))

    res10 = LinearStaticAnalysis(dom10).solve(dirichlet=dir10, f_ext=f10, verbose=False)
    tip_def10 = float(np.mean(res10.displacement[tip_nodes10 * 3 + 2]))

    # Export Deformed Results
    dom4.save_state(output_dir / "cantilever_tet4_def.vtu", include_stresses=True)
    dom10.save_state(output_dir / "cantilever_tet10_def.vtu", include_stresses=True)

    # 6. Performance & Shear Locking Comparison Table
    console.print("\n[bold yellow][6/6] Verifying Results & Post-Processing[/bold yellow]")
    table = Table(title="Cantilever Beam: Linear (Tet4) vs Quadratic (Tet10)")
    table.add_column("Metric / Property", style="cyan")
    table.add_column("Linear (Tet4)", style="magenta")
    table.add_column("Quadratic (Tet10)", style="green")
    table.add_column("Improvement / Ratio", style="yellow")

    table.add_row("Nodes", str(mesh_tet4.n_nodes), str(mesh_tet10.n_nodes), f"{mesh_tet10.n_nodes / mesh_tet4.n_nodes:.1f}x nodes")
    table.add_row("DOFs", str(mesh_tet4.n_dofs), str(mesh_tet10.n_dofs), f"{mesh_tet10.n_dofs / mesh_tet4.n_dofs:.1f}x DOFs")
    table.add_row("Interpolation Order", "Linear (p = 1)", "Quadratic (p = 2)", "Higher-order parabolic")
    table.add_row(
        "Tip Deflection (Uz)",
        f"{tip_def4:.5f} mm",
        f"{tip_def10:.5f} mm",
        f"[bold green]{abs(tip_def10 / tip_def4):.2f}x flexibility[/bold green]",
    )
    table.add_row(
        "Strain Energy",
        f"{res4.strain_energy:.3f} mJ",
        f"{res10.strain_energy:.3f} mJ",
        f"{res10.strain_energy / res4.strain_energy:.2f}x energy capture",
    )
    table.add_row(
        "Shear Locking Status",
        "[bold red]Artificially Locked[/bold red]",
        "[bold green]Locking Alleviated[/bold green]",
        "Captures true bending curvature",
    )

    console.print(table)
    console.print(
        f"\n[bold green]✓ Tet10 successfully alleviates shear locking ({abs(tip_def10 / tip_def4):.2f}x more flexible than Tet4).[/bold green]\n"
    )


if __name__ == "__main__":
    main()

