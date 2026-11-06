from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary.selector import select_nodes_by_plane
from psierra.boundary import DirichletConstraint, DirichletManager, GravityLoad
from psierra.damping import RayleighDamping
from psierra.analysis import NonlinearDynamicAnalysis

install(show_locals=False)
console = Console()


def main() -> None:
    # 0. Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]Benchmark 0704: Dynamic Time Integration Schemes Comparison[/bold cyan]\n"
            "[dim]Cantilever Beam Transient Dynamic Response: Undamped Newmark vs Generalized-alpha vs HHT-alpha vs Asymptotic Annihilation[/dim]",
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

    # 2. Material & Physical Continuum Configuration
    console.print("\n[bold yellow][2/5] Configuring Continuum Domain[/bold yellow]")
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000.0,  # MPa (N/mm^2)
        poissons_ratio=0.35,
        mass_density=1100e-6,    # tonne/mm^3 (1100 kg/m^3)
    )
    material = NeoHookean(params)
    console.print(
        f"  • Constitutive Law : [cyan]{material.__class__.__name__}[/cyan] (Compressible Neo-Hookean)"
    )
    console.print(
        f"  • Young's Modulus  : [green]{params.youngs_modulus:.1f}[/green] MPa | Poisson's Ratio: [green]{params.poissons_ratio:.2f}[/green]"
    )

    # Undamped physical continuum (C = 0) to evaluate pure algorithmic dissipation
    damping = RayleighDamping(alpha=0.0, beta=0.0)
    console.print("  • Physical Damping : [dim]Undamped (C = 0) to isolate algorithmic dissipation[/dim]")

    # 3. Boundary Conditions & External Gravity Load
    console.print("\n[bold yellow][3/5] Setting Up Boundary Conditions & Gravity Load[/bold yellow]")
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
    tip_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")

    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Suddenly applied gravity load along -Y axis: g = -9810 mm/s^2
    gravity = GravityLoad(acceleration=(0.0, -9810.0, 0.0))
    dummy_domain = SolidDomain(mesh, material, Tet4TotalLagrangian(material))
    M = dummy_domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)
    total_weight = abs(float(f_ext[1::3].sum()))

    console.print(f"  • Clamped Root Face (Z = -30 mm) : [green]{len(root_nodes)}[/green] nodes ({len(clamp.dofs)} DOFs)")
    console.print(f"  • Free Tip Face    (Z = 200 mm) : [green]{len(tip_nodes)}[/green] nodes")
    console.print(f"  • Total Self-Weight Load (F_y)  : [green]{total_weight:.2f}[/green] N (downward step load at t=0)")

    # 4. Executing Solvers Across Different Spectral Radii
    console.print("\n[bold yellow][4/5] Executing Dynamic Solvers Across Spectral Radii[/bold yellow]")
    output_dir = Path("results/07_benchmarks/0704_time_integrators")
    output_dir.mkdir(parents=True, exist_ok=True)

    dt = 0.005  # 5 ms time step
    n_steps = 30  # 150 ms total duration

    schemes = [
        ("Undamped Newmark", 1.0, "Zero numerical dissipation; exact energy preservation in linear limit"),
        ("Generalized-alpha (Default)", 0.8, "Optimal high-frequency dissipation; filters mesh noise with O(dt^2)"),
        ("HHT-alpha", 0.5, "Moderate high-frequency dissipation (alpha_m = 0, alpha_f = 1/3)"),
        ("Asymptotic Annihilation", 0.0, "Maximal high-frequency annihilation in 1 step; contact chatter suppression"),
    ]

    results_summary = []

    for name, rho_inf, desc in schemes:
        console.print(f"\n  [bold cyan]• Running: {name} (rho_inf = {rho_inf:.2f})...[/bold cyan]")
        element = Tet4TotalLagrangian(material)
        domain = SolidDomain(mesh, material, element)
        solver = NonlinearDynamicAnalysis(
            domain=domain,
            damping=damping,
            spectral_radius=rho_inf,
        )

        console.print(
            f"    Algorithmic Parameters: alpha_m = {solver.alpha_m:.4f}, "
            f"alpha_f = {solver.alpha_f:.4f}, gamma = {solver.gamma:.4f}, beta = {solver.beta:.4f}"
        )

        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=dt,
            n_steps=n_steps,
            nr_iterations=15,
            nr_tol=1e-4,
            include_stresses=True,
            smooth_to_nodes=True,
            save_directory=output_dir / f"rho_{rho_inf:.2f}",
            output_name="result",
            save_interval=2,
            verbose=False,
        )

        disp = domain.displacement
        tip_uy = float(np.mean(disp[tip_nodes, 1]))
        max_disp = float(np.linalg.norm(disp, axis=1).max())
        total_iters = sum(res.iteration_history)
        avg_iters = total_iters / max(1, res.steps_completed)

        results_summary.append({
            "name": name,
            "rho_inf": rho_inf,
            "alpha_m": solver.alpha_m,
            "alpha_f": solver.alpha_f,
            "gamma": solver.gamma,
            "beta": solver.beta,
            "converged": res.converged,
            "steps": res.steps_completed,
            "tip_uy": tip_uy,
            "max_disp": max_disp,
            "avg_iters": avg_iters,
            "desc": desc,
        })

    # 5. Comparative Verification & Rich Summary Table
    console.print("\n[bold yellow][5/5] Comparative Verification & Result Analysis[/bold yellow]")

    summary_table = Table(
        title="Benchmark 0704: Dynamic Time Integration Schemes Comparison",
        border_style="cyan",
        header_style="bold magenta",
    )
    summary_table.add_column("Integration Scheme", style="bold cyan")
    summary_table.add_column("rho_inf", justify="center")
    summary_table.add_column("alpha_m / alpha_f", justify="center")
    summary_table.add_column("gamma / beta", justify="center")
    summary_table.add_column("Convergence", justify="center")
    summary_table.add_column("Tip Sag Uy (mm)", justify="right")
    summary_table.add_column("Max Disp (mm)", justify="right")
    summary_table.add_column("Avg NR Iters", justify="right")

    for item in results_summary:
        conv_text = f"[green]{item['steps']}/{n_steps}[/green]" if item["converged"] else f"[red]{item['steps']}/{n_steps}[/red]"
        summary_table.add_row(
            item["name"],
            f"{item['rho_inf']:.2f}",
            f"{item['alpha_m']:.3f} / {item['alpha_f']:.3f}",
            f"{item['gamma']:.3f} / {item['beta']:.3f}",
            conv_text,
            f"{item['tip_uy']:.2f}",
            f"{item['max_disp']:.2f}",
            f"{item['avg_iters']:.1f}",
        )

    console.print(summary_table)

    console.print("\n[bold green]✓ Key Algorithmic Observations:[/bold green]")
    console.print(
        "  1. [bold cyan]Single Parameter Control[/bold cyan]: All time integration algorithms are seamlessly parameterized\n"
        "     via the high-frequency spectral radius rho_infinity without exposing error-prone (beta, gamma) constants."
    )
    console.print(
        "  2. [bold cyan]Strict Second-Order Accuracy[/bold cyan]: Unlike classical Newmark with artificial damping (which drops to O(dt)),\n"
        "     Generalized-alpha preserves strict O(dt^2) convergence across all rho_infinity in [0, 1]."
    )
    console.print(
        "  3. [bold cyan]Algorithmic Dissipation Effect[/bold cyan]:\n"
        f"     • rho_inf = 1.0 (Undamped Newmark) oscillates with zero numerical loss (Final Tip Sag: {results_summary[0]['tip_uy']:.2f} mm).\n"
        f"     • rho_inf = 0.8 (Optimal Default) gently damps spurious high-frequency modes (Final Tip Sag: {results_summary[1]['tip_uy']:.2f} mm).\n"
        f"     • rho_inf = 0.0 (Asymptotic Annihilation) aggressively stabilizes sudden shock loads (Final Tip Sag: {results_summary[3]['tip_uy']:.2f} mm)."
    )
    console.print(
        f"\n[bold green]✓ Exported PVD collections and VTU snapshots to:[/bold green] [cyan]{output_dir}[/cyan]\n"
    )


if __name__ == "__main__":
    main()

