"""
Example 0701: Full-Order Viscoelastic Quasi-Static Cantilever Beam Analysis.

Demonstrates:
1. Native beam coordinate frame (Z in [-30, 200] mm, clamped at Z = min).
2. Rate-dependent viscoelasticity without inertia via QuasiStaticAnalysis:
   - Generalized Maxwell / Prony series stress relaxation:
     S(t) = S_inf + sum_i Q_i(t)
   - Resolving physical viscoelastic creep and stress relaxation under sustained loading.
3. Configurable loading schedules:
   - Smooth ramp-and-hold: load linearly increases over t_ramp, then held constant.
4. ParaView XDMF / PVD export of temporal creep trajectory.
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
    NeoHookean,
    Viscoelastic,
    PronyTerm,
)
from psierra.element import Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import (
    select_nodes_by_plane,
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
)
from psierra.analysis import QuasiStaticAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0701: Viscoelastic Quasi-Static Beam Bending"
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=1.0,
        help="Total physical observation duration in seconds (default: 1.0 s)",
    )
    parser.add_argument(
        "--ramp-time",
        type=float,
        default=0.2,
        help="Duration of initial linear load ramp in seconds (default: 0.2 s)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Quasi-static time step increment in seconds (default: 0.05 s)",
    )
    parser.add_argument(
        "--relaxation-fraction",
        type=float,
        default=0.5,
        help="Prony series relative relaxation modulus g1 (default: 0.5)",
    )
    parser.add_argument(
        "--relaxation-time",
        type=float,
        default=0.2,
        help="Prony series relaxation time tau1 in seconds (default: 0.2 s)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["xdmf", "pvd"],
        default="xdmf",
        help="ParaView output format (default: xdmf)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/07_quasi_static/0701_beam",
        help="Base output directory (default: results/07_quasi_static/0701_beam)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comparative study with different relaxation rates",
    )
    return parser.parse_args()


def run_quasistatic_beam(
    g1: float,
    tau1: float,
    total_time: float,
    ramp_time: float,
    dt: float,
    fmt: str,
    output_dir: Path,
    mesh_path: Path,
    verbose: bool = True,
) -> dict[str, str | float]:
    console = Console()
    mesh = VolumetricMesh(mesh_path)

    # Elastic parameters: E = 21,000 MPa, nu = 0.35, rho = 1100e-6 tonne/mm³
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000.0,
        poissons_ratio=0.35,
        mass_density=1100e-6,
    )
    base_mat = NeoHookean(params)
    prony_terms = [PronyTerm(relative_modulus=g1, relaxation_time=tau1)]
    mat = Viscoelastic(base_mat, prony_terms)
    elem = Tet4TotalLagrangian(mat)
    domain = SolidDomain(mesh, mat, elem)

    # Clamped root face at Z = min
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min", tolerance=1e-3)
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Gravity body force along -Y (0.1g)
    g_acc = -981.0
    gravity = GravityLoad(acceleration=(0.0, g_acc, 0.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    case_tag = f"visco_g{int(g1*100)}_tau{int(tau1*1000)}"
    run_out_dir = output_dir / case_tag
    run_out_dir.mkdir(parents=True, exist_ok=True)

    def load_schedule(t_cur: float) -> float:
        return min(1.0, t_cur / max(ramp_time, 1e-6))

    history_disp: list[float] = []

    def callback_step(step: int, t_cur: float, u_vec: np.ndarray) -> None:
        u_mag = float(np.linalg.norm(u_vec.reshape(-1, 3), axis=1).max())
        history_disp.append(u_mag)

    if verbose:
        console.print(f"  • Running QuasiStatic: [bold cyan]{case_tag}[/bold cyan] (g1={g1:.2f}, tau1={tau1:.2f}s, T={total_time:.2f}s)")

    t0 = time.perf_counter()
    solver = QuasiStaticAnalysis(domain)
    res = solver.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        total_time=total_time,
        dt=dt,
        load_schedule=load_schedule,
        nr_iterations=30,
        save_directory=run_out_dir,
        format=fmt,
        verbose=False,
        callback=callback_step,
    )
    wall_time = time.perf_counter() - t0

    disp_final = float(np.linalg.norm(domain.displacement, axis=1).max())
    tip_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max", tolerance=1e-3)
    tip_sag_final = float(-domain.displacement[tip_nodes, 1].mean())

    ramp_idx = int(round(ramp_time / dt))
    disp_ramp = history_disp[ramp_idx] if ramp_idx < len(history_disp) else disp_final
    creep_factor = disp_final / max(disp_ramp, 1e-9)

    out_file = run_out_dir / f"result.{fmt}"

    return {
        "case": case_tag,
        "g1": g1,
        "tau1_s": tau1,
        "steps": res.steps_completed,
        "disp_ramp_mm": disp_ramp,
        "disp_final_mm": disp_final,
        "tip_sag_mm": tip_sag_final,
        "creep_factor": creep_factor,
        "runtime_s": wall_time,
        "output_path": str(out_file),
    }


def main() -> None:
    args = parse_args()
    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Example 0701: Viscoelastic Quasi-Static Cantilever Beam Analysis[/bold cyan]\n"
            f"[dim]Ramp: {args.ramp_time:.2f}s | Hold to {args.total_time:.2f}s | dt={args.dt:.3f}s | g1={args.relaxation_fraction:.2f}[/dim]",
            title="[bold blue]project-sierra — Quasi-Static Mechanics[/bold blue]",
            border_style="cyan",
        )
    )

    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Beam mesh not found: {mesh_path}")

    base_out = Path(args.output_dir)

    if args.all:
        console.print("\n[bold yellow]Executing Parametric Relaxation Rate Study...[/bold yellow]")
        cases = [
            (0.30, 0.20),
            (0.50, 0.20),
            (0.70, 0.20),
        ]
        results = []
        for g1, tau1 in cases:
            res = run_quasistatic_beam(
                g1=g1,
                tau1=tau1,
                total_time=args.total_time,
                ramp_time=args.ramp_time,
                dt=args.dt,
                fmt=args.format,
                output_dir=base_out,
                mesh_path=mesh_path,
                verbose=True,
            )
            results.append(res)

        table = Table(title="Example 0701: Viscoelastic Relaxation & Creep Study", border_style="cyan")
        table.add_column("Case Tag", style="bold white")
        table.add_column("Relaxation g1", style="cyan")
        table.add_column("Ramp Defl (mm)", style="yellow")
        table.add_column("Final Defl (mm)", style="green")
        table.add_column("Tip Sag (mm)", style="green")
        table.add_column("Creep Factor", style="magenta")
        table.add_column("Runtime (s)", style="blue")

        for r in results:
            table.add_row(
                str(r["case"]),
                f"{r['g1']:.2f}",
                f"{r['disp_ramp_mm']:.3f}",
                f"{r['disp_final_mm']:.3f}",
                f"{r['tip_sag_mm']:.3f}",
                f"{r['creep_factor']:.2f}x",
                f"{r['runtime_s']:.2f}",
            )
        console.print(table)

    else:
        console.print(f"\n[bold yellow]Executing Canonical Quasi-Static Run ({args.total_time:.2f} s)...[/bold yellow]")
        res = run_quasistatic_beam(
            g1=args.relaxation_fraction,
            tau1=args.relaxation_time,
            total_time=args.total_time,
            ramp_time=args.ramp_time,
            dt=args.dt,
            fmt=args.format,
            output_dir=base_out,
            mesh_path=mesh_path,
            verbose=True,
        )

        table = Table(title=f"Example 0701: {res['case']}", border_style="cyan")
        table.add_column("Quantity", style="bold white")
        table.add_column("Value", style="cyan")
        table.add_row("Prony Relative Modulus (g1)", f"{res['g1']:.2f}")
        table.add_row("Prony Relaxation Time (tau1)", f"{res['tau1_s']:.3f} s")
        table.add_row("Time Increments Completed", f"{res['steps']} steps")
        table.add_row("Instantaneous Deflection (t_ramp)", f"{res['disp_ramp_mm']:.4f} mm")
        table.add_row("Long-Term Equilibrium Deflection", f"{res['disp_final_mm']:.4f} mm")
        table.add_row("Final Tip Sag", f"{res['tip_sag_mm']:.4f} mm")
        table.add_row("Viscoelastic Creep Amplification", f"{res['creep_factor']:.3f}x")
        table.add_row("Solver Runtime", f"{res['runtime_s']:.2f} s")
        table.add_row("ParaView Trajectory", str(res["output_path"]))
        console.print(table)
        console.print(f"\n[bold green]✓ Quasi-static simulation complete. Dataset saved to:[/bold green] [cyan]{res['output_path']}[/cyan]\n")


if __name__ == "__main__":
    main()
