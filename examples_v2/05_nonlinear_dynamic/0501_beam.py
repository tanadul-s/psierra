"""
Example 0501: Master Nonlinear Dynamic Cantilever Beam Vibration under Gravity.

Demonstrates:
1. Native beam coordinate frame (Z in [-30, 200] mm, clamped at Z = min).
2. Multiple kinematic formulations:
   - Total Lagrangian (TL)
   - Updated Lagrangian (UL)
   - St. Venant-Kirchhoff (SVK)
   - Corotational (CR)
3. Advanced nonlinear material models:
   - Compressible Neo-Hookean
   - Ogden hyperelasticity (Order 1 / monotonic)
   - Viscoelasticity (Prony series stress relaxation)
4. Implicit time integration with controllable algorithmic damping:
   - Generalized-alpha (spectral_radius = 0.8)
   - Newmark-beta (spectral_radius = 1.0)
   - Hilber-Hughes-Taylor (HHT-alpha, spectral_radius = 0.5)
5. Hydrostatic fluid immersion with Archimedean buoyant relief (--buoyancy).
6. Full comparative matrix across kinematics and material models (--all).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psierra.material import (
    IsotropicElasticParameters,
    Hookean,
    NeoHookean,
    Ogden,
    OgdenParameters,
    Viscoelastic,
    PronyTerm,
)
from psierra.element import (
    Tet4TotalLagrangian,
    Tet4UpdatedLagrangian,
    Tet4Corotational,
)
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import (
    select_nodes_by_plane,
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
    BuoyancyLoad,
)
from psierra.damping import RayleighDamping
from psierra.analysis import NonlinearDynamicAnalysis


INTEGRATOR_SPECTRAL_RADIUS = {
    "generalized_alpha": 0.8,
    "newmark": 1.0,
    "hht": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0501: Master Nonlinear Dynamic Cantilever Beam"
    )
    parser.add_argument(
        "--formulation",
        type=str,
        choices=["tl", "ul", "svk", "cr"],
        default="tl",
        help="Kinematic formulation: tl (Total Lagrangian), ul (Updated Lagrangian), svk (St. Venant-Kirchhoff), cr (Corotational) [default: tl]",
    )
    parser.add_argument(
        "--integrator",
        type=str,
        choices=list(INTEGRATOR_SPECTRAL_RADIUS.keys()),
        default="generalized_alpha",
        help="Time integration algorithm (default: generalized_alpha)",
    )
    parser.add_argument(
        "--material",
        type=str,
        choices=["neohookean", "ogden", "viscoelastic"],
        default="neohookean",
        help="Constitutive material law (default: neohookean)",
    )
    parser.add_argument(
        "--buoyancy",
        action="store_true",
        help="Apply hydrostatic fluid buoyancy opposing gravity (water density 1000 kg/m³)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of transient time increments (default: 200)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005,
        help="Time step size in seconds (default: 0.005 s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/05_nonlinear_dynamic/0501_beam",
        help="Base directory for output datasets (default: results/05_nonlinear_dynamic/0501_beam)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comprehensive matrix benchmark across formulations and materials",
    )
    return parser.parse_args()


def build_material(mat_type: str, params: IsotropicElasticParameters):
    if mat_type == "neohookean":
        return NeoHookean(params)
    elif mat_type == "svk":
        return Hookean(params)
    elif mat_type == "ogden":
        ogden_params = OgdenParameters.from_isotropic(params, order=1, alpha=[2.0])
        return Ogden(ogden_params, volumetric_model="log")
    elif mat_type == "viscoelastic":
        base = NeoHookean(params)
        prony = [
            PronyTerm(relative_modulus=0.30, relaxation_time=0.05),
            PronyTerm(relative_modulus=0.20, relaxation_time=0.50),
        ]
        return Viscoelastic(base, prony)
    raise ValueError(f"Unknown material type: {mat_type}")


def build_element(material, formulation: str):
    form = formulation.lower()
    if form in ("tl", "svk"):
        return Tet4TotalLagrangian(material)
    elif form == "ul":
        return Tet4UpdatedLagrangian(material)
    elif form == "cr":
        return Tet4Corotational(material)
    raise ValueError(f"Unknown formulation: {formulation}")


def run_single_simulation(
    formulation: str,
    integrator: str,
    material_name: str,
    buoyancy: bool,
    n_steps: int,
    dt: float,
    output_dir: Path,
    mesh_path: Path,
    verbose: bool = True,
) -> dict[str, str | float]:
    console = Console()
    mesh = VolumetricMesh(mesh_path)

    # Material parameters (E = 21,000 MPa, nu = 0.35, rho = 1100e-6 tonne/mm³)
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000.0,
        poissons_ratio=0.35,
        mass_density=1100e-6,
    )

    mat_eff_name = "svk" if formulation == "svk" else material_name
    material = build_material(mat_eff_name, params)
    element = build_element(material, formulation)
    domain = SolidDomain(mesh, material, element)

    # Root clamp at Z = min
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min", tolerance=1e-3)
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Gravity along -Y
    g_acc = -9810.0
    gravity = GravityLoad(acceleration=(0.0, g_acc, 0.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    # Optional Buoyancy
    if buoyancy:
        fluid_density = 0.5 * params.mass_density  # 50% buoyant relief
        buoy = BuoyancyLoad(
            fluid_density=fluid_density,
            acceleration=(0.0, g_acc, 0.0),
        )
        f_buoy = buoy.compute_force_vector(
            domain, mass_matrix=M, solid_density=params.mass_density
        )
        f_ext = f_ext + f_buoy

    damping = RayleighDamping(alpha=2.0, beta=0.001)
    rho_inf = INTEGRATOR_SPECTRAL_RADIUS[integrator]
    analysis = NonlinearDynamicAnalysis(domain, damping=damping, spectral_radius=rho_inf)

    case_tag = f"{formulation}_{mat_eff_name}_{integrator}"
    if buoyancy:
        case_tag += "_buoyant"
    run_out_dir = output_dir / case_tag
    run_out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        console.print(f"  • Running case: [bold cyan]{case_tag}[/bold cyan] ({n_steps} steps, dt={dt:.4f}s)")

    t0 = time.perf_counter()
    result = analysis.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        dt=dt,
        n_steps=n_steps,
        nr_iterations=15,
        nr_tol=1e-4,
        save_directory=run_out_dir,
        save_interval=5,
        verbose=False,
    )
    wall_time = time.perf_counter() - t0

    max_disp = float(np.linalg.norm(domain.displacement, axis=1).max())
    tip_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max", tolerance=1e-3)
    tip_sag = float(-domain.displacement[tip_nodes, 1].mean())

    return {
        "case": case_tag,
        "formulation": formulation.upper(),
        "material": mat_eff_name.capitalize(),
        "integrator": integrator,
        "buoyancy": "Yes" if buoyancy else "No",
        "converged": result.converged,
        "steps_completed": result.steps_completed,
        "max_disp_mm": max_disp,
        "tip_sag_mm": tip_sag,
        "wall_time_s": wall_time,
        "output_path": str(run_out_dir / "result.xdmf"),
    }


def main() -> None:
    args = parse_args()
    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Example 0501: Master Nonlinear Dynamic Cantilever Beam[/bold cyan]\n"
            f"[dim]Kinematics: {args.formulation.upper()} | Material: {args.material.capitalize()} | Integrator: {args.integrator}[/dim]",
            title="[bold blue]project-sierra — Nonlinear Dynamics[/bold blue]",
            border_style="cyan",
        )
    )

    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Beam mesh file not found: {mesh_path}")

    base_out = Path(args.output_dir)

    if args.all:
        console.print("\n[bold yellow]Executing Full Comparative Matrix Across Dynamics Suite...[/bold yellow]")
        cases = [
            ("tl", "generalized_alpha", "neohookean", False),
            ("ul", "generalized_alpha", "neohookean", False),
            ("svk", "generalized_alpha", "svk", False),
            ("cr", "generalized_alpha", "neohookean", False),
            ("tl", "newmark", "neohookean", False),
            ("tl", "hht", "neohookean", False),
            ("tl", "generalized_alpha", "ogden", False),
            ("tl", "generalized_alpha", "viscoelastic", False),
            ("tl", "generalized_alpha", "neohookean", True),
        ]
        results = []
        for form, integ, mat, buoy in cases:
            res = run_single_simulation(
                formulation=form,
                integrator=integ,
                material_name=mat,
                buoyancy=buoy,
                n_steps=min(args.steps, 50),  # fast matrix
                dt=args.dt,
                output_dir=base_out,
                mesh_path=mesh_path,
                verbose=True,
            )
            results.append(res)

        table = Table(title="Example 0501: Comparative Matrix Summary", border_style="cyan")
        table.add_column("Case Tag", style="bold white")
        table.add_column("Formulation", style="cyan")
        table.add_column("Material", style="magenta")
        table.add_column("Integrator", style="yellow")
        table.add_column("Buoyancy", style="green")
        table.add_column("Steps", style="white")
        table.add_column("Max Disp (mm)", style="green")
        table.add_column("Tip Sag (mm)", style="green")
        table.add_column("Runtime (s)", style="blue")

        for r in results:
            table.add_row(
                str(r["case"]),
                str(r["formulation"]),
                str(r["material"]),
                str(r["integrator"]),
                str(r["buoyancy"]),
                f"{r['steps_completed']}/{50}",
                f"{r['max_disp_mm']:.3f}",
                f"{r['tip_sag_mm']:.3f}",
                f"{r['wall_time_s']:.2f}",
            )
        console.print(table)

    else:
        # Default canonical run
        console.print(f"\n[bold yellow]Executing Canonical Run ({args.steps} steps)...[/bold yellow]")
        res = run_single_simulation(
            formulation=args.formulation,
            integrator=args.integrator,
            material_name=args.material,
            buoyancy=args.buoyancy,
            n_steps=args.steps,
            dt=args.dt,
            output_dir=base_out,
            mesh_path=mesh_path,
            verbose=True,
        )

        table = Table(title=f"Example 0501: {res['case']}", border_style="cyan")
        table.add_column("Quantity", style="bold white")
        table.add_column("Value", style="cyan")
        table.add_row("Formulation", str(res["formulation"]))
        table.add_row("Constitutive Law", str(res["material"]))
        table.add_row("Time Integrator", str(res["integrator"]))
        table.add_row("Hydrostatic Buoyancy", str(res["buoyancy"]))
        table.add_row("Steps Completed", f"{res['steps_completed']}/{args.steps}")
        table.add_row("Max Displacement", f"{res['max_disp_mm']:.4f} mm")
        table.add_row("Mean Tip Sag", f"{res['tip_sag_mm']:.4f} mm")
        table.add_row("Solver Runtime", f"{res['wall_time_s']:.2f} s")
        table.add_row("ParaView Output", str(res["output_path"]))
        console.print(table)
        console.print(f"\n[bold green]✓ Simulation complete. XDMF dataset saved to:[/bold green] [cyan]{res['output_path']}[/cyan]\n")


if __name__ == "__main__":
    main()
