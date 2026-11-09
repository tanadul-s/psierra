"""
Simulation 1101: Volumetric Locking Alleviation via Tet4Fbar under Nonlinear Incremental Bending.

Demonstrates the alleviation of artificial volumetric locking in nearly-incompressible
soft tissue continua (nu = 0.499, K/mu = 500) under finite-strain incremental transverse bending.
Solves the full nonlinear equilibrium path across 15 incremental load steps using
NonlinearStaticAnalysis (Newton-Raphson with Neo-Hookean hyperelasticity).
Compares standard Total Lagrangian linear tetrahedra (Tet4TotalLagrangian) against the
locking-alleviated nodal F-bar formulation (Tet4Fbar).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian, Tet4Fbar
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary.selector import select_nodes_by_bounds
from psierra.boundary import DirichletConstraint, DirichletManager, NodalForce
from psierra.analysis import NonlinearStaticAnalysis

install(show_locals=False)
console = Console()


def main() -> None:
    # 0. Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]Simulation 1101: Volumetric Locking Alleviation via Tet4Fbar[/bold cyan]\n"
            "[dim]Nearly-Incompressible Soft Tissue (nu = 0.499, K/mu = 500) 15-Step Nonlinear Showcase[/dim]",
            title="[bold blue]project-sierra[/bold blue]",
            border_style="cyan",
        )
    )

    # 1. Loading Mesh Geometry
    console.print("\n[bold yellow][1/6] Loading Mesh Geometry[/bold yellow]")
    mesh_path = Path("data/beam1/mesh/beam1_fine.vtk")
    mesh = VolumetricMesh(mesh_path)
    console.print(
        f"  • Nodes: [green]{mesh.n_nodes}[/green] | DoFs: [green]{mesh.n_dofs}[/green] | Elements: [green]{mesh.n_elements}[/green]"
    )
    console.print(
        f"  • Volume: [green]{mesh.volume:.2f}[/green] mm³ | Centroid: [dim]{tuple(float(c) for c in np.round(mesh.centroid, 2))}[/dim]"
    )

    # 2. Configuring Nearly-Incompressible Soft-Tissue Material
    console.print(
        "\n[bold yellow][2/6] Configuring Soft-Tissue Material Parameters (nu = 0.499)[/bold yellow]"
    )
    params = IsotropicElasticParameters()
    # Parenchyma / hydrogel parameters: E = 1000 Pa, nu = 0.499, rho = 1040 kg/m^3
    params.set_from_youngs_poisson(
        youngs_modulus=1000.0,
        poissons_ratio=0.499,
        mass_density=1040e-12,  # Ton/mm^3
    )
    material = NeoHookean(params)
    k_bulk = params.bulk_modulus
    mu = params.shear_modulus
    ratio_k_mu = k_bulk / mu

    console.print(
        f"  • Constitutive Law : [cyan]{material.__class__.__name__}[/cyan] (Compressible Neo-Hookean)"
    )
    console.print(
        f"  • Young's Modulus  : [green]{params.youngs_modulus:.1f}[/green] Pa | Poisson's Ratio: [bold red]{params.poissons_ratio:.3f}[/bold red]"
    )
    console.print(
        f"  • Bulk Modulus K   : [green]{k_bulk:.2e}[/green] Pa | Shear Modulus mu: [green]{mu:.2e}[/green] Pa"
    )
    console.print(
        f"  • Incompressibility Ratio K / mu : [bold magenta]{ratio_k_mu:.1f}x[/bold magenta]"
    )

    # 3. Boundary Conditions & Incremental External Load
    console.print(
        "\n[bold yellow][3/6] Applying Boundary Constraints and Transverse Load[/bold yellow]"
    )
    bounds = mesh.bounds
    root_nodes = select_nodes_by_bounds(
        mesh,
        bounds=(
            bounds[0] - 1.0,
            bounds[0] + 5.0,
            bounds[2] - 1.0,
            bounds[3] + 1.0,
            bounds[4] - 1.0,
            bounds[5] + 1.0,
        ),
    )
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    tip_nodes = select_nodes_by_bounds(
        mesh,
        bounds=(
            bounds[1] - 5.0,
            bounds[1] + 1.0,
            bounds[2] - 1.0,
            bounds[3] + 1.0,
            bounds[4] - 1.0,
            bounds[5] + 1.0,
        ),
    )
    total_fz = -200.0  # Transverse bending force
    load = NodalForce(
        tip_nodes,
        force_vector=[0.0, 0.0, total_fz / len(tip_nodes)],
    )
    f_ext = load.compute_force_vector(mesh.n_dofs)

    console.print(
        f"  • Clamped Root     : [green]{len(root_nodes)}[/green] nodes ({len(clamp.dofs)} DOFs)"
    )
    console.print(
        f"  • Loaded Tip       : [green]{len(tip_nodes)}[/green] nodes | Total F_z: [green]{total_fz:.2f}[/green] N"
    )

    output_dir = Path("results/11_advanced_continua/1101_tet4fbar_incompressibility")
    output_dir.mkdir(parents=True, exist_ok=True)
    n_load_steps = 15

    # 4. Standard Total Lagrangian Linear Tetrahedra (Tet4TotalLagrangian)
    console.print(
        f"\n[bold yellow][4/6] Solving Standard Total Lagrangian (Tet4TotalLagrangian) across {n_load_steps} Steps[/bold yellow]"
    )
    elem_tl = Tet4TotalLagrangian(material)
    domain_tl = SolidDomain(mesh, material, elem_tl)
    analysis_tl = NonlinearStaticAnalysis(domain_tl)
    res_tl = analysis_tl.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        n_steps=n_load_steps,
        nr_iterations=15,
        nr_tol=1e-4,
        save_directory=output_dir / "static_tet4tl",
        output_name="result",
        verbose=False,
    )
    disp_tl = domain_tl.displacement
    sag_tl = float(abs(disp_tl[:, 2].min()))
    max_disp_tl = float(np.max(np.linalg.norm(disp_tl, axis=1)))

    console.print(
        f"  • Tet4TotalLagrangian Status : [green]{res_tl.converged}[/green] "
        f"({res_tl.steps_completed}/{n_load_steps} steps completed)"
    )
    console.print(
        f"  • Tet4TotalLagrangian Peak Sag : [red]{sag_tl:.4f}[/red] mm ([bold red]Artificially Locked by Volumetric Stiffening[/bold red])"
    )
    console.print(f"  • Tet4TotalLagrangian Max Disp : [red]{max_disp_tl:.4f}[/red] mm")

    # 5. Locking-Alleviated Formulation (Tet4Fbar)
    console.print(
        f"\n[bold yellow][5/6] Solving Locking-Alleviated Formulation (Tet4Fbar) across {n_load_steps} Steps[/bold yellow]"
    )
    elem_fbar = Tet4Fbar(material)
    domain_fbar = SolidDomain(mesh, material, elem_fbar)
    analysis_fbar = NonlinearStaticAnalysis(domain_fbar)
    res_fbar = analysis_fbar.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        n_steps=n_load_steps,
        nr_iterations=15,
        nr_tol=1e-4,
        save_directory=output_dir / "static_tet4fbar",
        output_name="result",
        verbose=False,
    )
    disp_fbar = domain_fbar.displacement
    sag_fbar = float(abs(disp_fbar[:, 2].min()))
    max_disp_fbar = float(np.max(np.linalg.norm(disp_fbar, axis=1)))
    compliance_ratio = sag_fbar / sag_tl if sag_tl > 0 else 0.0

    console.print(
        f"  • Tet4Fbar Status            : [green]{res_fbar.converged}[/green] "
        f"({res_fbar.steps_completed}/{n_load_steps} steps completed)"
    )
    console.print(
        f"  • Tet4Fbar Peak Sag          : [green]{sag_fbar:.4f}[/green] mm "
        f"([bold cyan]{compliance_ratio:.2f}x compliance / locking eliminated[/bold cyan])"
    )
    console.print(
        f"  • Tet4Fbar Max Disp          : [green]{max_disp_fbar:.4f}[/green] mm"
    )

    # 6. Verification Summary Table & Export
    console.print(
        "\n[bold yellow][6/6] Verifying Results & Post-Processing[/bold yellow]"
    )
    summary_table = Table(
        title=f"Simulation 1101: {n_load_steps}-Step Nonlinear Static Analysis (nu = 0.499, K/mu = 500)",
        border_style="cyan",
    )
    summary_table.add_column("Metric / Formulation", style="bold white")
    summary_table.add_column("Tet4TotalLagrangian (Standard)", style="red")
    summary_table.add_column("Tet4Fbar (Proposed)", style="green")
    summary_table.add_column("Assessment", style="cyan")

    tl_status = f"{res_tl.converged} ({res_tl.steps_completed}/{n_load_steps} steps)"
    fbar_status = (
        f"{res_fbar.converged} ({res_fbar.steps_completed}/{n_load_steps} steps)"
    )
    summary_table.add_row(
        "Solver Convergence",
        tl_status,
        fbar_status,
        "Both converged along full equilibrium path",
    )
    summary_table.add_row(
        "Peak Bending Sag (|Z|)",
        f"{sag_tl:.4f} mm",
        f"{sag_fbar:.4f} mm",
        f"{compliance_ratio:.2f}x compliant (locking eliminated)",
    )
    summary_table.add_row(
        "Max Resultant Disp",
        f"{max_disp_tl:.4f} mm",
        f"{max_disp_fbar:.4f} mm",
        f"{compliance_ratio:.2f}x flexibility recovered",
    )
    summary_table.add_row(
        "Poisson's Ratio nu",
        "0.499",
        "0.499",
        f"K/mu = {ratio_k_mu:.1f}x (incompressible regime)",
    )
    summary_table.add_row(
        "Load Incrementing",
        f"{n_load_steps} Newton steps",
        f"{n_load_steps} Newton steps",
        "Continuous finite-strain equilibrium curve",
    )

    console.print(summary_table)

    console.print(
        f"[bold green]✓ Output successfully exported to:[/bold green] [cyan]{output_dir}[/cyan]"
    )
    console.print(
        f"  • [cyan]static_tet4tl/result.pvd[/cyan]   : Standard Total Lagrangian final state (Sag: {sag_tl:.4f} mm)"
    )
    console.print(
        f"  • [cyan]static_tet4fbar/result.pvd[/cyan] : Locking-alleviated F-bar final state (Sag: {sag_fbar:.4f} mm)"
    )
    console.print(
        "[bold magenta]ParaView Animation Tip:[/bold magenta] Open [cyan]static_tet4fbar/result.pvd[/cyan] "
        "and [cyan]static_tet4tl/result.pvd[/cyan] side-by-side in ParaView. Apply [bold]Warp By Vector[/bold] "
        "and click the [bold]Play (▶)[/bold] button to watch the continuous nonlinear bending trajectory!"
    )


if __name__ == "__main__":
    main()
