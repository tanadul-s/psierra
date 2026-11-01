from __future__ import annotations
from pathlib import Path
import numpy as np
import pyvista as pv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4, Shell3
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain, ShellDomain, CoupledSolidShellDomain
from psierra.boundary import select_nodes_by_plane, DirichletConstraint, DirichletManager
from psierra.analysis import LinearStaticAnalysis

install(show_locals=False)
console = Console()


def main() -> None:
    # 0. Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]Simulation 10: Shell-Covered Beam vs Regular Beam Under Self-Weight Gravity[/bold cyan]\n"
            "[dim]Demonstrating Sag Mitigation in Soft Organ/Tissue via CoupledSolidShellDomain & LinearStaticAnalysis[/dim]",
            title="[bold blue]project-sierra — Phase 2 Structural Shells[/bold blue]",
            border_style="cyan",
        )
    )

    output_dir = Path("examples/05_simulation/sim10_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loading Base Volumetric Mesh
    console.print("\n[bold yellow][1/6] Loading Mesh Geometry[/bold yellow]")
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Beam mesh file not found: {mesh_path}")

    mesh = VolumetricMesh(mesh_path)
    surf = mesh.surface

    n_nodes = mesh.n_nodes
    n_elems = mesh.n_elements
    n_surf_nodes = surf.n_nodes
    n_surf_elems = surf.n_elements

    console.print(
        f"  • Solid Core (Volumetric): [green]{n_nodes}[/green] nodes | "
        f"[green]{n_elems}[/green] Tet4 elements | Volume: [green]{mesh.volume:.1f}[/green] mm³"
    )
    console.print(
        f"  • Boundary Shell (Skin)  : [green]{n_surf_nodes}[/green] surface nodes | "
        f"[green]{n_surf_elems}[/green] triangular shell elements"
    )
    console.print(
        f"  • Beam Span / Dimensions : X: [-60, 60] mm, Y: [-60, 60] mm, Z: [-30, 200] mm (Length = 230 mm)"
    )

    # 2. Material & Element Formulations
    console.print("\n[bold yellow][2/6] Configuring Material & Element Formulation[/bold yellow]")
    # Soft parenchymal continuum core (representative of soft brain/organ tissue)
    E_solid = 0.050       # MPa (50 kPa soft tissue)
    nu_solid = 0.40       # Poisson's ratio
    rho_solid = 1.0e-9    # tonne/mm^3 (1000 kg/m^3)

    params_solid = IsotropicElasticParameters()
    params_solid.set_from_youngs_poisson(E_solid, nu_solid, rho_solid)
    mat_solid = Hookean(params_solid)
    elem_solid = Tet4(mat_solid)

    # Stiff membrane/shell reinforcement (representative of dura mater / fibrous fascia)
    E_shell = 10.0        # MPa (200x stiffer than parenchymal core)
    nu_shell = 0.30
    rho_shell = 1.1e-9    # tonne/mm^3 (1100 kg/m^3)
    t_shell = 0.50        # mm (0.5 mm membrane wall thickness)

    params_shell = IsotropicElasticParameters()
    params_shell.set_from_youngs_poisson(E_shell, nu_shell, rho_shell)
    mat_shell = Hookean(params_shell)
    elem_shell = Shell3(mat_shell, thickness=t_shell)

    console.print(f"  • Soft Solid Core  : E = [green]{E_solid * 1e3:.1f} kPa[/green] | ν = {nu_solid:.2f} | ρ = {rho_solid * 1e12:.0f} kg/m³")
    console.print(f"  • Exterior Shell   : E = [green]{E_shell:.1f} MPa[/green] (200x stiffer) | t = [green]{t_shell:.2f} mm[/green] | ν = {nu_shell:.2f}")

    # 3. Setting Up Physical Domains
    console.print("\n[bold yellow][3/6] Setting Up Physical Continuum Domains[/bold yellow]")
    domain_regular = SolidDomain(mesh=mesh, material=mat_solid, element=elem_solid)
    domain_core = SolidDomain(mesh=mesh, material=mat_solid, element=elem_solid)
    shell_domain = ShellDomain(mesh=surf, material=mat_shell, element=elem_shell)
    coupled_domain = CoupledSolidShellDomain(solid_domain=domain_core, shell_domain=shell_domain)

    console.print(f"  • Regular Domain   : [green]{domain_regular.n_dofs}[/green] DOFs (SolidDomain)")
    console.print(f"  • Coupled Domain   : [green]{coupled_domain.n_dofs}[/green] total DOFs ({domain_core.n_dofs} solid + {shell_domain.mesh.n_nodes * 3} shell rotations)")

    # 4. Assemble Gravity Body Force Vector & Boundary Conditions
    console.print("\n[bold yellow][4/6] Applying Boundary Conditions & External Loads[/bold yellow]")
    g_acc = 9810.0  # mm/s^2 (Standard Earth gravity 9.81 m/s^2 directed along -Y)
    f_grav_solid = np.zeros(domain_regular.n_dofs, dtype=np.float64)
    elem_vols = mesh.volumes

    for e_idx, conn in enumerate(mesh.elements):
        vol_e = elem_vols[e_idx]
        f_node_y = -(rho_solid * vol_e / 4.0) * g_acc
        for nid in conn:
            f_grav_solid[nid * 3 + 1] += f_node_y

    total_weight_N = abs(np.sum(f_grav_solid[1::3]))
    console.print(f"  • Gravity Field     : g = -9.81 m/s² along Y-axis (transverse downward)")
    console.print(f"  • Total Self-Weight : [green]{total_weight_N:.3f} N[/green] (Mass = {total_weight_N / 9.81 * 1e3:.1f} g)")

    # Boundary conditions: Clamped at root face (Z = -30 mm)
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
    tip_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
    console.print(f"  • Clamped Root Face : [green]{len(root_nodes)}[/green] nodes constrained at Z = min")
    console.print(f"  • Monitored Tip Face: [green]{len(tip_nodes)}[/green] nodes at Z = max (Cantilever tip)")

    # Dirichlet boundary managers
    dm_regular = DirichletManager(domain_regular.n_dofs)
    dm_regular.add_constraint(DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0))

    dm_coupled = DirichletManager(coupled_domain.n_dofs)
    dm_coupled.add_constraint(
        DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0, dofs_per_node=3)
    )
    surf_root_mask = np.isin(surf.node_indices, root_nodes)
    surf_root_indices = np.where(surf_root_mask)[0]
    rot_dofs = [
        3 * n_nodes + 3 * s + c for s in surf_root_indices for c in (0, 1, 2)
    ]
    dm_coupled.prescribe_dofs(rot_dofs, value=0.0)

    f_grav_coupled = np.zeros(coupled_domain.n_dofs, dtype=np.float64)
    f_grav_coupled[: domain_core.n_dofs] = f_grav_solid

    # 5. Executing Solvers
    console.print("\n[bold yellow][5/6] Executing Simulation Solvers[/bold yellow]")
    console.print("  [cyan]• Model A: Regular Soft Beam (SolidDomain + LinearStaticAnalysis)...[/cyan]")
    analysis_regular = LinearStaticAnalysis(domain_regular)
    res_regular = analysis_regular.solve(dirichlet=dm_regular, f_ext=f_grav_solid, verbose=False)

    tip_def_regular = float(np.mean(res_regular.displacement[tip_nodes * 3 + 1]))
    energy_regular = res_regular.strain_energy
    console.print(f"    - Maximum Tip Sag (Uy): [bold red]{tip_def_regular:.3f} mm[/bold red]")

    console.print("  [cyan]• Model B: Shell-Covered Beam (CoupledSolidShellDomain + LinearStaticAnalysis)...[/cyan]")
    analysis_coupled = LinearStaticAnalysis(coupled_domain)
    res_coupled = analysis_coupled.solve(dirichlet=dm_coupled, f_ext=f_grav_coupled, verbose=False)

    tip_def_coupled = float(np.mean(res_coupled.displacement[tip_nodes * 3 + 1]))
    energy_coupled = res_coupled.strain_energy
    reduction_ratio = abs(tip_def_regular / tip_def_coupled)

    console.print(f"    - Maximum Tip Sag (Uy): [bold green]{tip_def_coupled:.3f} mm[/bold green]")
    console.print(
        f"    - Deflection Reduction : [bold green]{reduction_ratio:.2f}x reduction[/bold green] "
        f"({(1.0 - abs(tip_def_coupled / tip_def_regular)) * 100:.1f}% less sag)"
    )

    # 6. Save Deformed Visualization Results & Summary
    console.print("\n[bold yellow][6/6] Verifying Results & Post-Processing[/bold yellow]")
    domain_regular.save_state(output_dir / "beam_regular_deformed.vtu")
    coupled_domain.save_solid_state(output_dir / "beam_shell_covered_deformed.vtu")
    coupled_domain.save_shell_state(output_dir / "beam_outer_shell_deformed.vtu")
    coupled_domain.save_pvd(output_dir / "beam_shell_covered.pvd")

    console.print(f"  • Saved [green]beam_regular_deformed.vtu[/green]")
    console.print(f"  • Saved [green]beam_shell_covered_deformed.vtu[/green]")
    console.print(f"  • Saved [green]beam_outer_shell_deformed.vtu[/green]")
    console.print(f"  • Saved [green]beam_shell_covered.pvd[/green] (ParaView multi-domain collection)")

    # 7. Comparison Summary Table
    table = Table(title="Gravity-Induced Cantilever Bending: Regular vs Shell-Covered Beam")
    table.add_column("Property / Metric", style="cyan")
    table.add_column("Regular Soft Beam", style="magenta")
    table.add_column("Shell-Covered Beam", style="green")
    table.add_column("Impact / Reduction", style="yellow")

    table.add_row("Domain Class", "SolidDomain", "CoupledSolidShellDomain", "Monolithic coupling")
    table.add_row("Core Modulus (E_solid)", f"{E_solid * 1e3:.0f} kPa", f"{E_solid * 1e3:.0f} kPa", "Identical soft core")
    table.add_row("Shell Modulus (E_shell)", "None (0.0 MPa)", f"{E_shell:.1f} MPa", "200x stiffer membrane")
    table.add_row("Shell Wall Thickness", "0.0 mm", f"{t_shell:.2f} mm", "Thin exoskeleton")
    table.add_row("Total System DOFs", f"{domain_regular.n_dofs}", f"{coupled_domain.n_dofs}", f"+{3 * n_surf_nodes} shell DOFs")
    table.add_row(
        "Tip Sag (Deflection Uy)",
        f"{tip_def_regular:.2f} mm",
        f"{tip_def_coupled:.2f} mm",
        f"[bold green]{reduction_ratio:.2f}x sag reduction[/bold green]",
    )
    table.add_row(
        "Strain Energy",
        f"{energy_regular:.2f} mJ",
        f"{energy_coupled:.2f} mJ",
        f"[bold green]{energy_regular / energy_coupled:.2f}x stiffer[/bold green]",
    )
    table.add_row(
        "Deformation State",
        "[bold red]Severe Sagging[/bold red]",
        "[bold green]Restrained Stability[/bold green]",
        "Membrane tension barrier",
    )

    console.print("\n")
    console.print(table)
    console.print(
        f"\n[bold green]✓ Shell coating successfully restrains the soft beam, reducing gravity sag by {reduction_ratio:.2f}x ({abs(tip_def_regular):.1f} mm → {abs(tip_def_coupled):.1f} mm).[/bold green]\n"
    )


if __name__ == "__main__":
    main()
