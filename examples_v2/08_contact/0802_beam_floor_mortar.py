"""
Example 0802: Segment-to-Segment Mortar Contact — Dynamic Beam on 3D Floor.

Demonstrates:
1. MultiBodySystem assembly incorporating BOTH Beam and Floor domains.
2. Segment-to-Segment Mortar Contact:
   - Sutherland-Hodgman polygon clipping on contact interface.
   - Dual shape functions satisfying biorthogonality.
3. Configurable floor compliance:
   - Rigid floor: Perfectly rigid obstacle with all floor nodes clamped.
   - Deformable floor: Two-way deformable-to-deformable mortar coupling with flexible floor plate.
4. Generalized-alpha dynamic time integration with Augmented Lagrangian Uzawa enforcement.
5. ParaView XDMF / PVD export in results/08_contact/0802_beam_floor_mortar.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.multibody import MultiBodySystem
from psierra.boundary import (
    select_nodes_by_bounds,
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
)
from psierra.damping import RayleighDamping
from psierra.analysis import NonlinearDynamicAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0802: Mortar Contact Dynamic Beam on 3D Floor"
    )
    parser.add_argument(
        "--floor",
        type=str,
        choices=["rigid", "deformable"],
        default="rigid",
        help="Floor compliance mode: rigid or deformable (default: rigid)",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        choices=["penalty", "augmented_lagrangian"],
        default="penalty",
        help="Mortar enforcement formulation: 'penalty' or 'augmented_lagrangian' (default: penalty)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of dynamic time steps (default: 25)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="Time step in seconds (default: 0.002 s = 500 Hz)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="Initial vertical gap between cantilever beam and floor in mm (default: 0.0 mm)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=50000.0,
        help="Normal penalty contact stiffness in N/mm (default: 50000.0 N/mm)",
    )
    parser.add_argument(
        "--cdamp",
        type=float,
        default=10.0,
        help="Contact normal damping coefficient in N·s/mm (default: 10.0 N·s/mm)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/08_contact/0802_beam_floor_mortar",
        help="Base output directory (default: results/08_contact/0802_beam_floor_mortar)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comparison between rigid and deformable floor mortar contact",
    )
    return parser.parse_args()


def run_mortar_case(
    floor_mode: str,
    formulation: str,
    n_steps: int,
    dt: float,
    gap: float,
    kp: float,
    cdamp: float,
    base_out: Path,
    beam_mesh_path: Path,
    floor_mesh_path: Path,
    verbose: bool = True,
) -> dict[str, str | float]:
    console = Console()
    case_tag = f"mortar_{floor_mode}_floor"
    run_out_dir = base_out / case_tag
    run_out_dir.mkdir(parents=True, exist_ok=True)

    beam_mesh = VolumetricMesh(beam_mesh_path)
    beam_bottom_y = float(beam_mesh.bounds[2])
    min_z = float(beam_mesh.bounds[4])

    params_beam = IsotropicElasticParameters()
    params_beam.set_from_youngs_poisson(
        youngs_modulus=21000.0,
        poissons_ratio=0.35,
        mass_density=1100e-6,
    )
    mat_beam = NeoHookean(params_beam)
    elem_beam = Tet4TotalLagrangian(mat_beam)
    beam_domain = SolidDomain(beam_mesh, mat_beam, elem_beam)

    floor_mesh = VolumetricMesh(floor_mesh_path)
    orig_top_y = float(floor_mesh.bounds[3])
    target_top_y = beam_bottom_y - gap
    dy = target_top_y - orig_top_y
    floor_mesh.transform(translation=(0.0, dy, 150.0))

    params_floor = IsotropicElasticParameters()
    params_floor.set_from_youngs_poisson(
        youngs_modulus=5000.0 if floor_mode == "deformable" else 210000.0,
        poissons_ratio=0.35,
        mass_density=1100e-6,
    )
    mat_floor = NeoHookean(params_floor)
    elem_floor = Tet4TotalLagrangian(mat_floor)
    floor_domain = SolidDomain(floor_mesh, mat_floor, elem_floor)

    system = MultiBodySystem({"Beam": beam_domain, "Floor": floor_domain})
    contact = system.add_mortar_contact(
        slave="Beam",
        master="Floor",
        k_penalty=kp,
        c_damping=cdamp,
        formulation=formulation,
        search_radius=30.0,
    )

    # Clamped beam root at min Z
    base_nodes = select_nodes_by_bounds(
        beam_mesh, bounds=(-1000.0, 1000.0, -1000.0, 1000.0, min_z - 1.0, min_z + 5.0)
    )
    beam_dof_offset = system.get_body_dof_slice("Beam").start
    clamp_beam = DirichletConstraint(
        base_nodes, components=(0, 1, 2), value=0.0, dof_offset=beam_dof_offset
    )

    floor_dof_offset = system.get_body_dof_slice("Floor").start
    if floor_mode == "rigid":
        clamp_floor = DirichletConstraint(
            np.arange(floor_mesh.n_nodes),
            components=(0, 1, 2),
            value=0.0,
            dof_offset=floor_dof_offset,
        )
    else:
        # Clamped bottom surface of flexible floor plate
        floor_min_y = float(floor_mesh.bounds[2])
        bot_nodes = np.where(np.abs(floor_mesh.nodes[:, 1] - floor_min_y) < 1.0)[0]
        clamp_floor = DirichletConstraint(
            bot_nodes,
            components=(0, 1, 2),
            value=0.0,
            dof_offset=floor_dof_offset,
        )

    dirichlet = DirichletManager(system.n_dofs)
    dirichlet.add_constraint(clamp_beam)
    dirichlet.add_constraint(clamp_floor)

    gravity = GravityLoad(acceleration=(0.0, -9810.0, 0.0))
    M = system.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)
    damping = RayleighDamping(alpha=1.0, beta=0.002)

    if verbose:
        console.print(f"  • Running Mortar Contact: [bold cyan]{case_tag}[/bold cyan] ({n_steps} steps, dt={dt:.4f}s)")

    t0 = time.perf_counter()
    solver = NonlinearDynamicAnalysis(system, damping=damping, spectral_radius=0.8)
    res = solver.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        dt=dt,
        n_steps=n_steps,
        nr_iterations=15,
        nr_tol=1e-3,
        save_directory=run_out_dir,
        output_name="result",
        save_interval=1,
        verbose=False,
    )
    wall_time = time.perf_counter() - t0

    beam_disp = beam_domain.displacement
    max_beam_disp = float(np.linalg.norm(beam_disp, axis=1).max())
    floor_disp = floor_domain.displacement
    max_floor_disp = float(np.linalg.norm(floor_disp, axis=1).max())

    return {
        "case": case_tag,
        "floor": floor_mode.capitalize(),
        "formulation": formulation.capitalize(),
        "converged": res.converged,
        "steps": f"{res.steps_completed}/{n_steps}",
        "max_beam_disp_mm": max_beam_disp,
        "max_floor_disp_mm": max_floor_disp,
        "runtime_s": wall_time,
        "output_dir": str(run_out_dir),
    }


def main() -> None:
    args = parse_args()
    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Example 0802: Segment-to-Segment Mortar Contact Dynamic Beam[/bold cyan]\n"
            f"[dim]Floor: {args.floor.upper()} | Formulation: {args.formulation.upper()} | Penalty kp = {args.kp:.1e} N/mm[/dim]",
            title="[bold blue]project-sierra — Mortar Contact Mechanics[/bold blue]",
            border_style="cyan",
        )
    )

    beam_path = Path("data/beam1/mesh/beam1.vtk")
    floor_path = Path("data/floor1/mesh/floor1.vtk")
    if not beam_path.exists() or not floor_path.exists():
        raise FileNotFoundError("Beam or floor mesh file not found.")

    base_out = Path(args.output_dir)

    if args.all:
        console.print("\n[bold yellow]Executing Mortar Contact Comparison...[/bold yellow]")
        cases = ["rigid", "deformable"]
        results = []
        for fl in cases:
            res = run_mortar_case(
                floor_mode=fl,
                formulation=args.formulation,
                n_steps=min(args.steps, 10),
                dt=args.dt,
                gap=args.gap,
                kp=args.kp,
                cdamp=args.cdamp,
                base_out=base_out,
                beam_mesh_path=beam_path,
                floor_mesh_path=floor_path,
                verbose=True,
            )
            results.append(res)

        table = Table(title="Example 0802: Mortar Contact Comparison", border_style="cyan")
        table.add_column("Case Tag", style="bold white")
        table.add_column("Floor Mode", style="cyan")
        table.add_column("Steps Completed", style="yellow")
        table.add_column("Max Beam Disp (mm)", style="green")
        table.add_column("Max Floor Disp (mm)", style="green")
        table.add_column("Runtime (s)", style="blue")

        for r in results:
            table.add_row(
                str(r["case"]),
                str(r["floor"]),
                str(r["steps"]),
                f"{r['max_beam_disp_mm']:.3f}",
                f"{r['max_floor_disp_mm']:.3f}",
                f"{r['runtime_s']:.2f}",
            )
        console.print(table)

    else:
        console.print(f"\n[bold yellow]Executing Canonical Mortar Contact ({args.floor.upper()} FLOOR)...[/bold yellow]")
        res = run_mortar_case(
            floor_mode=args.floor,
            formulation=args.formulation,
            n_steps=args.steps,
            dt=args.dt,
            gap=args.gap,
            kp=args.kp,
            cdamp=args.cdamp,
            base_out=base_out,
            beam_mesh_path=beam_path,
            floor_mesh_path=floor_path,
            verbose=True,
        )

        table = Table(title=f"Example 0802: {res['case']}", border_style="cyan")
        table.add_column("Parameter", style="bold white")
        table.add_column("Value", style="cyan")
        table.add_row("Floor Mode", str(res["floor"]))
        table.add_row("Mortar Formulation", str(res["formulation"]))
        table.add_row("Steps Completed", str(res["steps"]))
        table.add_row("Convergence Status", "[bold green]Converged[/bold green]" if res["converged"] else "[bold red]Diverged[/bold red]")
        table.add_row("Max Beam Contact Displacement", f"{res['max_beam_disp_mm']:.4f} mm")
        table.add_row("Max Floor Foundation Deflection", f"{res['max_floor_disp_mm']:.4f} mm")
        table.add_row("Execution Runtime", f"{res['runtime_s']:.2f} s")
        table.add_row("Output Directory", str(res["output_dir"]))
        console.print(table)
        console.print(f"\n[bold green]✓ Mortar contact simulation complete. Dataset saved to:[/bold green] [cyan]{res['output_dir']}[/cyan]\n")


if __name__ == "__main__":
    main()
