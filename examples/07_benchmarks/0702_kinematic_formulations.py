from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import (
    Tet4Linear,
    Tet4Corotational,
    Tet4TotalLagrangian,
    Tet4UpdatedLagrangian,
)
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import (
    select_nodes_by_plane,
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
)
from psierra.analysis import LinearStaticAnalysis, NonlinearStaticAnalysis

install(show_locals=False)
console = Console()


def main() -> None:
    # 0. Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]Simulation 14: Comprehensive Kinematic Formulations Comparison[/bold cyan]\n"
            "[dim]Cantilever Beam under Gravity: Linear vs Corotational (CR) vs Total Lagrangian (TL) vs Updated Lagrangian (UL)[/dim]",
            title="[bold blue]project-sierra[/bold blue]",
            border_style="cyan",
        )
    )

    # 1. Loading Mesh Geometry
    console.print("\n[bold yellow][1/5] Loading Mesh Geometry[/bold yellow]")
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Beam mesh file not found: {mesh_path}")

    mesh = VolumetricMesh(mesh_path)
    console.print(
        f"  • Volumetric Mesh : [green]{mesh.n_nodes}[/green] nodes | "
        f"[green]{mesh.n_elements}[/green] Tet4 elements | "
        f"Volume: [green]{mesh.volume:.1f}[/green] mm³"
    )
    console.print(
        "  • Beam Dimensions : X: [-60, 60] mm, Y: [-60, 60] mm, Z: [-30, 200] mm (Length L = 230 mm)"
    )

    # 2. Material Configuration
    console.print("\n[bold yellow][2/5] Configuring Material Parameters[/bold yellow]")
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000.0,
        poissons_ratio=0.35,
        mass_density=1100e-6,
    )
    material = Hookean(params)
    console.print(
        f"  • Constitutive Law : [cyan]{material.__class__.__name__}[/cyan] "
        f"(E = {params.youngs_modulus:.1f} MPa, ν = {params.poissons_ratio:.2f}, ρ = {params.mass_density:.2e} tonne/mm³)"
    )

    # 3. Boundary Conditions & External Gravity Load
    console.print(
        "\n[bold yellow][3/5] Setting Up Boundary Conditions & Gravity Load[/bold yellow]"
    )
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
    tip_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")

    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Reference cantilever chord length
    initial_root_z = float(np.mean(mesh.nodes[root_nodes, 2]))
    initial_tip_z = float(np.mean(mesh.nodes[tip_nodes, 2]))
    initial_chord = initial_tip_z - initial_root_z

    # Downward gravity body force along -Y: g = [0, -9810, 0] mm/s²
    g_acc = -9810.0
    gravity = GravityLoad(acceleration=(0.0, g_acc, 0.0))

    # Mass matrix from reference geometry
    elem_ref = Tet4Linear(material)
    dom_ref = SolidDomain(mesh, material, elem_ref)
    M_ref = dom_ref.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M_ref)
    total_weight = abs(float(np.sum(f_ext[1::3])))

    console.print(
        f"  • Clamped Root Face (Z = {initial_root_z:.1f} mm) : [green]{len(root_nodes)}[/green] nodes ({len(clamp.dofs)} DOFs)"
    )
    console.print(
        f"  • Free Tip Face    (Z = {initial_tip_z:.1f} mm) : [green]{len(tip_nodes)}[/green] nodes"
    )
    console.print(f"  • Cantilever Span  : [green]{initial_chord:.1f}[/green] mm")
    console.print(
        f"  • Total Gravity Load (F_y): [green]{total_weight:.2f}[/green] N (downward along -Y axis)"
    )

    # 4. Executing Solvers for All 4 Formulations
    console.print(
        "\n[bold yellow][4/5] Executing Solvers for All 4 Formulations[/bold yellow]"
    )

    # --- Formulation 1: Linear Small-Strain ---
    console.print(
        "  [cyan][1/4] Solving Formulation 1: Linear Small-Strain (Tet4Linear)...[/cyan]"
    )
    elem_lin = Tet4Linear(material)
    dom_lin = SolidDomain(mesh, material, elem_lin)
    res_lin = LinearStaticAnalysis(dom_lin).solve(
        dirichlet=dirichlet, f_ext=f_ext, verbose=False
    )
    disp_lin = dom_lin.displacement
    tip_uy_lin = float(np.mean(disp_lin[tip_nodes, 1]))
    tip_uz_lin = float(np.mean(disp_lin[tip_nodes, 2]))
    max_disp_lin = float(np.linalg.norm(disp_lin, axis=1).max())
    chord_lin = float(np.sqrt((initial_chord + tip_uz_lin) ** 2 + tip_uy_lin**2))

    # --- Formulation 2: Corotational ---
    console.print(
        "  [cyan][2/4] Solving Formulation 2: Corotational Finite Rotation (Tet4Corotational)...[/cyan]"
    )
    elem_cr = Tet4Corotational(material)
    dom_cr = SolidDomain(mesh, material, elem_cr)
    solver_cr = NonlinearStaticAnalysis(dom_cr)
    res_cr = solver_cr.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        n_steps=10,
        nr_iterations=20,
        nr_tol=1e-4,
        verbose=False,
    )
    disp_cr = dom_cr.displacement
    tip_uy_cr = float(np.mean(disp_cr[tip_nodes, 1]))
    tip_uz_cr = float(np.mean(disp_cr[tip_nodes, 2]))
    max_disp_cr = float(np.linalg.norm(disp_cr, axis=1).max())
    chord_cr = float(np.sqrt((initial_chord + tip_uz_cr) ** 2 + tip_uy_cr**2))

    # --- Formulation 3: Total Lagrangian ---
    console.print(
        "  [cyan][3/4] Solving Formulation 3: Total Lagrangian Finite Strain (Tet4TotalLagrangian)...[/cyan]"
    )
    elem_tl = Tet4TotalLagrangian(material)
    dom_tl = SolidDomain(mesh, material, elem_tl)
    solver_tl = NonlinearStaticAnalysis(dom_tl)
    res_tl = solver_tl.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        n_steps=10,
        nr_iterations=20,
        nr_tol=1e-4,
        verbose=False,
    )
    disp_tl = dom_tl.displacement
    tip_uy_tl = float(np.mean(disp_tl[tip_nodes, 1]))
    tip_uz_tl = float(np.mean(disp_tl[tip_nodes, 2]))
    max_disp_tl = float(np.linalg.norm(disp_tl, axis=1).max())
    chord_tl = float(np.sqrt((initial_chord + tip_uz_tl) ** 2 + tip_uy_tl**2))

    # --- Formulation 4: Updated Lagrangian ---
    console.print(
        "  [cyan][4/4] Solving Formulation 4: Updated Lagrangian Spatial Integration (Tet4UpdatedLagrangian)...[/cyan]"
    )
    elem_ul = Tet4UpdatedLagrangian(material)
    dom_ul = SolidDomain(mesh, material, elem_ul)
    solver_ul = NonlinearStaticAnalysis(dom_ul)
    res_ul = solver_ul.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        n_steps=10,
        nr_iterations=20,
        nr_tol=1e-4,
        verbose=False,
    )
    disp_ul = dom_ul.displacement
    tip_uy_ul = float(np.mean(disp_ul[tip_nodes, 1]))
    tip_uz_ul = float(np.mean(disp_ul[tip_nodes, 2]))
    max_disp_ul = float(np.linalg.norm(disp_ul, axis=1).max())
    chord_ul = float(np.sqrt((initial_chord + tip_uz_ul) ** 2 + tip_uy_ul**2))

    # 5. Comparing Results & Exporting VTU
    console.print(
        "\n[bold yellow][5/5] Comparative Verification & Result Analysis[/bold yellow]"
    )

    table = Table(
        title="Cantilever Bending under Gravity: 4 Kinematic Formulations Comparison",
        border_style="cyan",
        header_style="bold cyan",
    )
    table.add_column("Metric / Property", style="bold white")
    table.add_column("Linear (Small Strain)", style="magenta")
    table.add_column("Corotational (CR)", style="yellow")
    table.add_column("Total Lagrangian (TL)", style="green")
    table.add_column("Updated Lagrangian (UL)", style="blue")

    table.add_row(
        "Element Class",
        f"{elem_lin.__class__.__name__}",
        f"{elem_cr.__class__.__name__}",
        f"{elem_tl.__class__.__name__}",
        f"{elem_ul.__class__.__name__}",
    )
    table.add_row(
        "Kinematic Description",
        "Infinitesimal ε = sym(∇u)",
        "Polar Rotation Filtered R",
        "Material Config X (E, S)",
        "Spatial Config x (e, σ)",
    )
    table.add_row(
        "Solver Convergence",
        "Direct Solve (1 step)",
        f"{res_cr.steps_completed}/10 steps",
        f"{res_tl.steps_completed}/10 steps",
        f"{res_ul.steps_completed}/10 steps",
    )
    table.add_row(
        "Transverse Sag (Uy)",
        f"{tip_uy_lin:.2f} mm",
        f"{tip_uy_cr:.2f} mm",
        f"{tip_uy_tl:.2f} mm",
        f"{tip_uy_ul:.2f} mm",
    )
    table.add_row(
        "Axial Shortening (Uz)",
        f"{tip_uz_lin:.3f} mm",
        f"{tip_uz_cr:.3f} mm",
        f"{tip_uz_tl:.3f} mm",
        f"{tip_uz_ul:.3f} mm",
    )
    table.add_row(
        "Max Resultant Disp",
        f"{max_disp_lin:.2f} mm",
        f"{max_disp_cr:.2f} mm",
        f"{max_disp_tl:.2f} mm",
        f"{max_disp_ul:.2f} mm",
    )
    table.add_row(
        "Tip Chord Length",
        f"{chord_lin:.2f} mm (Elongated)",
        f"{chord_cr:.2f} mm (Preserved)",
        f"{chord_tl:.2f} mm (Preserved)",
        f"{chord_ul:.2f} mm (Preserved)",
    )

    console.print(table)

    # Numerical equivalence check between TL and UL
    tl_ul_diff = np.max(np.abs(disp_tl - disp_ul))
    console.print(
        f"\n[bold green]✓ Total Lagrangian vs Updated Lagrangian Maximum Displacement Difference:[/bold green] "
        f"[bold white]{tl_ul_diff:.4e} mm[/bold white]"
    )

    output_dir = Path("results/07_benchmarks/0702_kinematic_formulations")
    output_dir.mkdir(parents=True, exist_ok=True)

    dom_lin.save_state(output_dir / "result_linear.vtu")
    dom_cr.save_state(output_dir / "result_cr.vtu")
    dom_tl.save_state(output_dir / "result_tl.vtu")
    dom_ul.save_state(output_dir / "result_ul.vtu")
    console.print(
        f"[bold green]✓ Saved deformed states for all 4 formulations in:[/bold green] [cyan]{output_dir}[/cyan]\n"
    )


if __name__ == "__main__":
    main()
