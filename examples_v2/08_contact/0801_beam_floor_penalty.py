"""
Example 0801: Node-to-Surface Penalty Contact — Flexible Beam on 3D Floor.

Demonstrates:
1. MultiBodySystem assembly incorporating BOTH Beam and Floor domains.
2. Node-to-surface penalty contact mechanics (RigidContact or DeformableContact).
3. Configurable floor compliance:
   - Rigid floor: Perfectly rigid boundary where all floor nodes are clamped.
   - Deformable floor: Compliant foundation clamped only at floor bottom.
4. Flexible analysis options:
   - Incremental Nonlinear Static equilibrium (NonlinearStaticAnalysis).
   - Generalized-alpha finite-strain dynamics (NonlinearDynamicAnalysis).
5. Output written to ParaView datasets in results/08_contact/0801_beam_floor_penalty.
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
from psierra.element import Tet4TotalLagrangian, Tet4Linear
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
from psierra.analysis import NonlinearStaticAnalysis, NonlinearDynamicAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0801: Penalty Contact Beam on 3D Floor"
    )
    parser.add_argument(
        "--floor",
        type=str,
        choices=["rigid", "deformable"],
        default="rigid",
        help="Floor boundary mode (default: rigid)",
    )
    parser.add_argument(
        "--analysis",
        type=str,
        choices=["static", "dynamic"],
        default="static",
        help="Analysis mode: static or dynamic (default: static)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of load steps / dynamic time steps (default: 10)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="Initial vertical gap between beam bottom and floor top in mm (default: 0.0 mm)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=500000.0,
        help="Normal penalty contact stiffness in N/mm (default: 500000.0 N/mm)",
    )
    parser.add_argument(
        "--load",
        type=float,
        default=2.0e5,
        help="Total downward tip load in N for static analysis (default: 2.0e5 N)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/08_contact/0801_beam_floor_penalty",
        help="Output directory path (default: results/08_contact/0801_beam_floor_penalty)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comparative study across rigid/deformable floors and static/dynamic modes",
    )
    return parser.parse_args()


def run_contact_case(
    floor_mode: str,
    analysis_mode: str,
    n_steps: int,
    gap: float,
    kp: float,
    tip_load: float,
    base_out: Path,
    beam_mesh_path: Path,
    floor_mesh_path: Path,
    verbose: bool = True,
) -> dict[str, str | float]:
    console = Console()
    case_tag = f"{analysis_mode}_{floor_mode}_floor"
    run_out_dir = base_out / case_tag
    run_out_dir.mkdir(parents=True, exist_ok=True)

    beam_mesh = VolumetricMesh(beam_mesh_path)
    beam_bottom_y = float(beam_mesh.bounds[2])
    min_z = float(beam_mesh.bounds[4])
    max_z = float(beam_mesh.bounds[5])

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
        youngs_modulus=21000.0 if floor_mode == "deformable" else 210000.0,
        poissons_ratio=0.35,
        mass_density=1100e-6,
    )
    mat_floor = NeoHookean(params_floor)
    elem_floor = Tet4Linear(mat_floor)
    floor_domain = SolidDomain(floor_mesh, mat_floor, elem_floor)

    system = MultiBodySystem({"Beam": beam_domain, "Floor": floor_domain})
    contact = system.add_contact_pair(slave="Beam", master="Floor", k_penalty=kp)

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
        # Deformable: clamp only bottom boundary of floor
        floor_min_y = float(floor_mesh.bounds[2])
        floor_bottom_nodes = select_nodes_by_bounds(
            floor_mesh, bounds=(-1000.0, 1000.0, floor_min_y - 1.0, floor_min_y + 1.0, -1000.0, 1000.0)
        )
        clamp_floor = DirichletConstraint(
            floor_bottom_nodes,
            components=(0, 1, 2),
            value=0.0,
            dof_offset=floor_dof_offset,
        )

    dirichlet = DirichletManager(system.n_dofs)
    dirichlet.add_constraint(clamp_beam)
    dirichlet.add_constraint(clamp_floor)

    if verbose:
        console.print(f"  • Running case: [bold cyan]{case_tag}[/bold cyan] ({n_steps} steps, floor={floor_mode})")

    t0 = time.perf_counter()

    if analysis_mode == "static":
        tip_nodes = select_nodes_by_bounds(
            beam_mesh, bounds=(-1000.0, 1000.0, -1000.0, 1000.0, max_z - 20.0, max_z + 1.0)
        )
        f_ext = np.zeros(system.n_dofs, dtype=np.float64)
        tip_force_per_node = -tip_load / len(tip_nodes)
        for n in tip_nodes:
            f_ext[beam_dof_offset + 3 * n + 1] = tip_force_per_node

        solver = NonlinearStaticAnalysis(system)
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            n_steps=n_steps,
            nr_tol=1e-3,
            factor=0.0,
            save_directory=run_out_dir,
            output_name="result",
            save_interval=1,
            verbose=False,
        )
        converged = res.converged
        steps_done = res.steps_completed
    else:
        # Dynamic gravity drop
        gravity = GravityLoad(acceleration=(0.0, -9810.0, 0.0))
        M = system.assemble_mass_matrix()
        f_ext = gravity.compute_force_vector(M)
        damping = RayleighDamping(alpha=1.0, beta=0.002)

        solver = NonlinearDynamicAnalysis(system, damping=damping, spectral_radius=0.8)
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=0.005,
            n_steps=n_steps,
            nr_iterations=15,
            nr_tol=1e-3,
            save_directory=run_out_dir,
            output_name="result",
            save_interval=1,
            verbose=False,
        )
        converged = res.converged
        steps_done = res.steps_completed

    wall_time = time.perf_counter() - t0

    beam_disp = beam_domain.displacement
    max_beam_disp = float(np.linalg.norm(beam_disp, axis=1).max())
    floor_disp = floor_domain.displacement
    max_floor_disp = float(np.linalg.norm(floor_disp, axis=1).max())

    return {
        "case": case_tag,
        "floor": floor_mode.capitalize(),
        "analysis": analysis_mode.capitalize(),
        "converged": converged,
        "steps": f"{steps_done}/{n_steps}",
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
            "[bold cyan]Example 0801: Penalty Contact — Flexible Beam on 3D Floor[/bold cyan]\n"
            f"[dim]Floor: {args.floor.upper()} | Analysis: {args.analysis.upper()} | Penalty kp = {args.kp:.1e} N/mm[/dim]",
            title="[bold blue]project-sierra — Contact Mechanics[/bold blue]",
            border_style="cyan",
        )
    )

    beam_path = Path("data/beam1/mesh/beam1.vtk")
    floor_path = Path("data/floor1/mesh/floor1.vtk")
    if not beam_path.exists() or not floor_path.exists():
        raise FileNotFoundError("Beam or floor mesh file not found.")

    base_out = Path(args.output_dir)

    if args.all:
        console.print("\n[bold yellow]Executing Contact Parametric Matrix...[/bold yellow]")
        cases = [
            ("rigid", "static"),
            ("deformable", "static"),
            ("rigid", "dynamic"),
        ]
        results = []
        for fl, an in cases:
            res = run_contact_case(
                floor_mode=fl,
                analysis_mode=an,
                n_steps=min(args.steps, 5),
                gap=args.gap,
                kp=args.kp,
                tip_load=args.load,
                base_out=base_out,
                beam_mesh_path=beam_path,
                floor_mesh_path=floor_path,
                verbose=True,
            )
            results.append(res)

        table = Table(title="Example 0801: Contact Benchmark Summary", border_style="cyan")
        table.add_column("Case Tag", style="bold white")
        table.add_column("Floor", style="cyan")
        table.add_column("Analysis", style="magenta")
        table.add_column("Steps", style="yellow")
        table.add_column("Beam Disp (mm)", style="green")
        table.add_column("Floor Disp (mm)", style="green")
        table.add_column("Runtime (s)", style="blue")

        for r in results:
            table.add_row(
                str(r["case"]),
                str(r["floor"]),
                str(r["analysis"]),
                str(r["steps"]),
                f"{r['max_beam_disp_mm']:.3f}",
                f"{r['max_floor_disp_mm']:.3f}",
                f"{r['runtime_s']:.2f}",
            )
        console.print(table)

    else:
        console.print(f"\n[bold yellow]Executing Canonical Contact Run ({args.analysis.upper()} / {args.floor.upper()} FLOOR)...[/bold yellow]")
        res = run_contact_case(
            floor_mode=args.floor,
            analysis_mode=args.analysis,
            n_steps=args.steps,
            gap=args.gap,
            kp=args.kp,
            tip_load=args.load,
            base_out=base_out,
            beam_mesh_path=beam_path,
            floor_mesh_path=floor_path,
            verbose=True,
        )

        table = Table(title=f"Example 0801: {res['case']}", border_style="cyan")
        table.add_column("Parameter", style="bold white")
        table.add_column("Value", style="cyan")
        table.add_row("Floor Boundary Mode", str(res["floor"]))
        table.add_row("Analysis Regime", str(res["analysis"]))
        table.add_row("Contact Steps Completed", str(res["steps"]))
        table.add_row("Convergence Status", "[bold green]Converged[/bold green]" if res["converged"] else "[bold red]Failed[/bold red]")
        table.add_row("Max Beam Contact Deflection", f"{res['max_beam_disp_mm']:.4f} mm")
        table.add_row("Max Floor Foundation Deflection", f"{res['max_floor_disp_mm']:.4f} mm")
        table.add_row("Solver Execution Time", f"{res['runtime_s']:.2f} s")
        table.add_row("ParaView Output Directory", str(res["output_dir"]))
        console.print(table)
        console.print(f"\n[bold green]✓ Contact solve complete. Results written to:[/bold green] [cyan]{res['output_dir']}[/cyan]\n")


if __name__ == "__main__":
    main()
