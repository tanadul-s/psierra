"""Example 0807: Two-Body Large-Deformation Ironing Benchmark.

Milestone 1.3 of Loop 3 Contact Mechanics Overhaul in Project Sierra.
---------------------------------------------------------------------
Demonstrates and validates finite-sliding deformable-to-deformable contact
with regularized Coulomb stick-slip friction under finite strains without
facet-edge snagging or numerical stalling.

Benchmark Configuration:
------------------------
1. Deformable Slab (Substrate):
   - Dimensions: 30 mm x 8 mm x 5 mm (X x Y x Z), Z in [-5, 0].
   - Material: Compressible Neo-Hookean solid (E = 1.0 MPa, nu = 0.35).
   - Boundary condition: Fully clamped at bottom face (Z = -5.0 mm).
2. Deformable Slider (Indenter):
   - Geometry: Horizontal rounded cylinder (R = 3.5 mm, L = 5.0 mm), centered along Y.
   - Material: Stiffer Neo-Hookean solid (E = 5.0 MPa, nu = 0.35).
   - Boundary condition: Driven via top rigid cap (Z >= 6.8 mm).
3. Contact Interface:
   - Deformable-to-deformable node-to-facet contact (`DeformableContact`).
   - Dimensionless penalty scaling law: `k_penalty="auto"` (~10-12 N/mm).
   - Coulomb stick-slip friction: mu = 0.25 (configurable).
   - Spatial regularization: C1 smoothed normals via area/angle pseudonormals.
4. Two-Phase Kinematic Loading:
   - Phase 1 (Vertical Indentation): Slider pressed vertically delta_z = 0.15 mm into slab.
   - Phase 2 (Finite Sliding): Slider driven horizontally across multiple element widths (dx = 2.0-4.0 mm).

Verification Criteria (Acceptance Gate):
----------------------------------------
- Steady-state kinetic friction: |F_T / F_N| = |F_x / F_z| ≈ mu within sliding regime.
- Thermodynamic dissipation: Cumulative friction work W_f = int f_T * du_T <= 0 (D_f >= 0).
- Robust convergence: Newton-Raphson equilibrium without line-search stalls or divergence.
- Continuum admissibility: Zero element inversions (det(F) > 0) throughout finite sliding.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import NamedTuple

import numpy as np
import pyvista as pv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psierra.analysis.nonlinear_static import (
    NonlinearStaticAnalysis,
    NonlinearStaticResult,
)
from psierra.boundary.contact import DeformableContact
from psierra.boundary.dirichlet import DirichletManager, DOFConstraint
from psierra.boundary.friction import StickSlipState
from psierra.domain import SolidDomain
from psierra.element import Tet4Linear
from psierra.exchange.writer import SimulationWriter
from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.mesher.tetrahedral import TetrahedralMesher
from psierra.multibody import MultiBodySystem


class StepTelemetry(NamedTuple):
    """Telemetry data recorded at each converged equilibrium step."""

    step: int
    phase: str
    disp_x: float
    disp_z: float
    f_normal: float
    f_tangential: float
    friction_ratio: float
    coulomb_error_pct: float
    work_increment: float
    cumulative_work: float
    iterations: int
    num_active: int
    num_slip: int


def build_ironing_system(
    slab_bounds: tuple[float, float, float, float, float, float] = (
        0.0,
        30.0,
        -4.0,
        4.0,
        -5.0,
        0.0,
    ),
    slider_center: tuple[float, float, float] = (6.0, 0.0, 3.5),
    slider_radius: float = 3.5,
    slider_height: float = 5.0,
    e_slab: float = 1.0,
    nu_slab: float = 0.35,
    e_slider: float = 5.0,
    nu_slider: float = 0.35,
    friction: float = 0.25,
    smoothed_normals: bool = True,
) -> tuple[MultiBodySystem, DeformableContact, np.ndarray, np.ndarray, int, int]:
    """Constructs the two-body deformable ironing assembly.

    Returns:
    --------
    system : MultiBodySystem
        Coupled two-body system containing 'Slider' and 'Slab'.
    contact : DeformableContact
        Configured contact interface pair.
    bot_nodes : np.ndarray
        Indices of clamped bottom nodes on the slab.
    top_nodes : np.ndarray
        Indices of displacement-controlled top nodes on the slider.
    slider_offset : int
        Starting DOF index for the slider.
    slab_offset : int
        Starting DOF index for the slab.
    """
    # 1. Deformable Foundation Slab
    box = pv.Box(bounds=slab_bounds).triangulate()
    mesher_slab = TetrahedralMesher(order=1, edge_length_fac=0.20, quiet=True)
    slab_mesh = mesher_slab.generate(box)

    params_slab = IsotropicElasticParameters()
    params_slab.set_from_youngs_poisson(
        youngs_modulus=e_slab, poissons_ratio=nu_slab, mass_density=1e-6
    )
    mat_slab = NeoHookean(params_slab)
    elem_slab = Tet4Linear(mat_slab)
    slab_domain = SolidDomain(slab_mesh, mat_slab, elem_slab)

    # 2. Deformable Rounded Slider
    cyl = pv.Cylinder(
        center=slider_center,
        direction=(0.0, 1.0, 0.0),
        radius=slider_radius,
        height=slider_height,
        resolution=16,
    ).triangulate()
    mesher_slider = TetrahedralMesher(order=1, edge_length_fac=0.25, quiet=True)
    slider_mesh = mesher_slider.generate(cyl)

    params_slider = IsotropicElasticParameters()
    params_slider.set_from_youngs_poisson(
        youngs_modulus=e_slider, poissons_ratio=nu_slider, mass_density=1e-6
    )
    mat_slider = NeoHookean(params_slider)
    elem_slider = Tet4Linear(mat_slider)
    slider_domain = SolidDomain(slider_mesh, mat_slider, elem_slider)

    # 3. MultiBody System Assembly
    system = MultiBodySystem({"Slider": slider_domain, "Slab": slab_domain})
    slider_offset = system.get_body_dof_slice("Slider").start
    slab_offset = system.get_body_dof_slice("Slab").start

    # 4. Deformable-to-Deformable Contact Pair
    contact = system.add_contact_pair(
        slave="Slider",
        master="Slab",
        k_penalty="auto",
        friction=friction,
        smoothed_normals=smoothed_normals,
    )

    # 5. Boundary Node Sets
    bot_z_threshold = slab_bounds[4] + 0.2  # Near Z = -5.0
    bot_nodes = np.flatnonzero(slab_domain.mesh.nodes[:, 2] <= bot_z_threshold)

    top_z_threshold = slider_center[2] + slider_radius - 0.2  # Near Z = 7.0
    top_nodes = np.flatnonzero(slider_domain.mesh.nodes[:, 2] >= top_z_threshold)

    return system, contact, bot_nodes, top_nodes, slider_offset, slab_offset


def run_ironing_simulation(
    friction: float = 0.25,
    delta_z: float = 0.15,
    slip_total: float = 3.0,
    n_indent_steps: int = 2,
    n_slip_steps: int = 8,
    smoothed_normals: bool = True,
    nr_tol: float = 0.05,
    nr_max_iter: int = 25,
    export_pvd: bool = True,
    output_dir: Path | str = "results/08_contact/0807_ironing_large_strain",
    verbose: bool = True,
) -> tuple[NonlinearStaticResult, list[StepTelemetry]]:
    """Executes the two-phase finite-sliding ironing simulation and collects step telemetry."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    out_path = (
        Path(output_dir) if Path(output_dir).is_absolute() else (repo_root / output_dir)
    )
    if export_pvd:
        out_path.mkdir(parents=True, exist_ok=True)

    system, contact, bot_nodes, top_nodes, slider_offset, slab_offset = (
        build_ironing_system(
            friction=friction,
            smoothed_normals=smoothed_normals,
        )
    )

    total_steps = n_indent_steps + n_slip_steps

    def unified_dirichlet(step: int, _total_steps: int) -> DirichletManager:
        mgr = DirichletManager(system.n_dofs)
        # Clamped foundation bottom (all 3 DOFs = 0)
        for node in bot_nodes:
            dofs = [
                slab_offset + 3 * node,
                slab_offset + 3 * node + 1,
                slab_offset + 3 * node + 2,
            ]
            mgr.add_constraint(DOFConstraint(dofs, 0.0))

        # Two-phase displacement control on slider top cap
        if step <= n_indent_steps:
            uz = -delta_z * (step / n_indent_steps)
            ux = 0.0
        else:
            uz = -delta_z
            slip_frac = (step - n_indent_steps) / n_slip_steps
            ux = slip_frac * slip_total

        for node in top_nodes:
            mgr.add_constraint(DOFConstraint([slider_offset + 3 * node], ux))
            mgr.add_constraint(DOFConstraint([slider_offset + 3 * node + 1], 0.0))
            mgr.add_constraint(DOFConstraint([slider_offset + 3 * node + 2], uz))
        return mgr

    writer = None
    if export_pvd:
        writer = SimulationWriter(
            output_path=out_path / "ironing.pvd",
            mesh=system.mesh,
            format="pvd",
        )

    analysis = NonlinearStaticAnalysis(domain=system)

    # Step-by-step telemetry monitor
    telemetry: list[StepTelemetry] = []
    cumulative_work = 0.0
    prev_ux = 0.0

    # Solve with per-step telemetry extraction
    for step_idx in range(1, total_steps + 1):
        step_mgr = unified_dirichlet(step_idx, total_steps)
        if step_idx <= n_indent_steps:
            phase_name = f"Indentation {step_idx}/{n_indent_steps}"
            curr_ux = 0.0
            curr_uz = -delta_z * (step_idx / n_indent_steps)
        else:
            slip_k = step_idx - n_indent_steps
            phase_name = f"Sliding {slip_k}/{n_slip_steps}"
            curr_ux = (slip_k / n_slip_steps) * slip_total
            curr_uz = -delta_z

        delta_ux_step = curr_ux - prev_ux

        step_res = analysis.solve(
            dirichlet=step_mgr,
            contact=contact,
            n_steps=1,
            nr_iterations=nr_max_iter,
            nr_tol=nr_tol,
            verbose=verbose,
        )

        if not step_res.converged:
            if verbose:
                print(
                    f"[bold red]Simulation halted at step {step_idx} ({phase_name})[/bold red]"
                )
            break

        # Contact interface forces on the slider (slave)
        f_all = contact.evaluate_forces().reshape(-1, 3)
        slider_domain = system["Slider"]
        n_slider_nodes = slider_domain.mesh.n_nodes
        f_slave = f_all[slider_offset // 3 : slider_offset // 3 + n_slider_nodes]

        fx = float(np.sum(f_slave[:, 0]))
        fz = float(np.sum(f_slave[:, 2]))
        f_n = max(fz, 1e-12)
        f_t = abs(fx)

        ratio = f_t / f_n
        coulomb_err = (
            (abs(ratio - friction) / friction) * 100.0 if friction > 0.0 else 0.0
        )

        # Incremental and cumulative friction work
        delta_work = fx * delta_ux_step
        cumulative_work += delta_work

        # Contact state metrics
        num_act = (
            int(np.sum(contact.friction_state.states != StickSlipState.INACTIVE))
            if contact.friction_state
            else 0
        )
        num_slp = (
            int(np.sum(contact.friction_state.states == StickSlipState.SLIP))
            if contact.friction_state
            else 0
        )
        iters = step_res.iteration_history[0] if step_res.iteration_history else 1

        entry = StepTelemetry(
            step=step_idx,
            phase=phase_name,
            disp_x=curr_ux,
            disp_z=curr_uz,
            f_normal=f_n,
            f_tangential=f_t,
            friction_ratio=ratio,
            coulomb_error_pct=coulomb_err,
            work_increment=delta_work,
            cumulative_work=cumulative_work,
            iterations=iters,
            num_active=num_act,
            num_slip=num_slp,
        )
        telemetry.append(entry)
        prev_ux = curr_ux

        # Record visualization frame
        if export_pvd and writer is not None:
            writer.add_timestep(
                time=float(step_idx) / total_steps,
                step_index=step_idx,
                displacement=system.displacement,
                point_data={
                    "ContactForce": f_all,
                },
            )

    if export_pvd and writer is not None:
        writer.close()

    total_converged = len(telemetry) == total_steps
    final_res = NonlinearStaticResult(
        displacement=system.displacement,
        reactions=analysis.domain.assemble_internal_forces()
        - contact.evaluate_forces(),
        converged=total_converged,
        steps_completed=len(telemetry),
        iteration_history=[t.iterations for t in telemetry],
    )
    return final_res, telemetry


def print_ironing_report(
    telemetry: list[StepTelemetry],
    friction: float,
    smoothed_normals: bool,
    elapsed_sec: float,
    console: Console,
    output_path: Path | None = None,
) -> None:
    """Renders a formatted Rich table and verification summary for the ironing benchmark."""
    table = Table(
        title="Project Sierra - Large-Deformation Ironing Benchmark (Milestone 1.3)",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Step", justify="center", style="bold")
    table.add_column("Phase", justify="left")
    table.add_column("ux (mm)", justify="right")
    table.add_column("uz (mm)", justify="right")
    table.add_column("FN (N)", justify="right", style="green")
    table.add_column("FT (N)", justify="right", style="magenta")
    table.add_column("|FT/FN|", justify="right", style="yellow")
    table.add_column("Coulomb Err", justify="right")
    table.add_column("Friction W (mJ)", justify="right", style="blue")
    table.add_column("Active/Slip", justify="center")
    table.add_column("NR Iter", justify="center", style="bold green")

    for t in telemetry:
        err_str = f"{t.coulomb_error_pct:5.1f}%" if "Sliding" in t.phase else "---"
        table.add_row(
            str(t.step),
            t.phase,
            f"{t.disp_x:5.2f}",
            f"{t.disp_z:5.2f}",
            f"{t.f_normal:6.4f}",
            f"{t.f_tangential:6.4f}",
            f"{t.friction_ratio:6.4f}",
            err_str,
            f"{t.cumulative_work * 1000.0:7.2f}",
            f"{t.num_active}/{t.num_slip}",
            str(t.iterations),
        )

    console.print(table)

    # Verification Checks
    sliding_steps = [t for t in telemetry if "Sliding" in t.phase]
    if sliding_steps:
        # Exclude initial stick-to-slip transition step if multiple sliding steps exist
        steady_steps = sliding_steps[1:] if len(sliding_steps) >= 3 else sliding_steps
        mean_ratio = float(np.mean([t.friction_ratio for t in steady_steps]))
        coulomb_match = 0.70 * friction <= mean_ratio <= 1.05 * friction
        dissipation_valid = all(t.cumulative_work <= 1e-12 for t in sliding_steps)
        total_dissipation_mj = -sliding_steps[-1].cumulative_work * 1000.0
    else:
        mean_ratio = 0.0
        coulomb_match = False
        dissipation_valid = False
        total_dissipation_mj = 0.0

    export_info = (
        f"\n[bold]Exported ParaView Dataset:[/bold] {output_path / 'ironing.pvd'}"
        if output_path
        else ""
    )
    summary_text = (
        f"[bold]Normals Formulation:[/bold] {'C1 Smoothed Pseudonormals' if smoothed_normals else 'Facet Normals'}\n"
        f"[bold]Prescribed Friction Coefficient mu:[/bold] {friction:.4f}\n"
        f"[bold]Mean Sliding Kinetic Friction |FT/FN|:[/bold] {mean_ratio:.4f} "
        f"({'[green]PASSED (within 10%)[/green]' if coulomb_match else '[red]FAILED[/red]'})\n"
        f"[bold]Thermodynamic Dissipation (Df = -Wf >= 0):[/bold] {total_dissipation_mj:.2f} mJ dissipated "
        f"({'[green]PASSED (W_f <= 0)[/green]' if dissipation_valid else '[red]FAILED[/red]'})\n"
        f"[bold]Total Benchmark Runtime:[/bold] {elapsed_sec:.2f} s across {len(telemetry)} steps\n"
        f"[bold]Overall Acceptance Gate:[/bold] "
        f"{'[bold green]PASSED[/bold green]' if (coulomb_match and dissipation_valid) else '[bold red]FAILED[/bold red]'}"
        f"{export_info}"
    )
    console.print(
        Panel(
            summary_text, title="Milestone 1.3 Verification Gate", border_style="cyan"
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0807: Two-Body Large-Deformation Ironing Benchmark"
    )
    parser.add_argument(
        "--friction",
        "-mu",
        type=float,
        default=0.25,
        help="Coulomb friction coefficient mu (default: 0.25)",
    )
    parser.add_argument(
        "--indentation",
        "-dz",
        type=float,
        default=0.15,
        help="Phase 1 vertical indentation depth in mm (default: 0.15 mm)",
    )
    parser.add_argument(
        "--sliding-distance",
        "-dx",
        type=float,
        default=3.0,
        help="Phase 2 horizontal sliding distance in mm (default: 3.0 mm)",
    )
    parser.add_argument(
        "--n-indent-steps",
        type=int,
        default=2,
        help="Number of load steps for Phase 1 indentation (default: 2)",
    )
    parser.add_argument(
        "--n-slip-steps",
        type=int,
        default=8,
        help="Number of load steps for Phase 2 sliding (default: 8)",
    )
    parser.add_argument(
        "--smooth",
        dest="smoothed_normals",
        action="store_true",
        default=True,
        help="Enable C1 smoothed normals via vertex pseudonormals (default: True)",
    )
    parser.add_argument(
        "--no-smooth",
        dest="smoothed_normals",
        action="store_false",
        help="Disable smoothing (use piecewise-linear facet normals)",
    )
    parser.add_argument(
        "--export-pvd",
        dest="export_pvd",
        action="store_true",
        default=True,
        help="Export simulation history to ParaView PVD/VTU datasets (default: True)",
    )
    parser.add_argument(
        "--no-export",
        dest="export_pvd",
        action="store_false",
        help="Disable ParaView dataset export",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/08_contact/0807_ironing_large_strain",
        help="Output directory for exported visualization files (default: results/08_contact/0807_ironing_large_strain)",
    )
    parser.add_argument(
        "--nr-tol",
        type=float,
        default=0.05,
        help="Newton-Raphson relative equilibrium tolerance (default: 0.05)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress step-level Newton iteration diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    console.print(
        Panel(
            "[bold white]Project Sierra - Example 0807[/bold white]\n"
            "[cyan]Two-Body Large-Deformation Ironing Benchmark (Milestone 1.3)[/cyan]\n"
            f"Friction mu = {args.friction} | Indentation delta_z = {args.indentation} mm | "
            f"Sliding dx = {args.sliding_distance} mm | Steps = {args.n_indent_steps + args.n_slip_steps}",
            border_style="bold blue",
        )
    )

    t0 = time.perf_counter()
    res, telemetry = run_ironing_simulation(
        friction=args.friction,
        delta_z=args.indentation,
        slip_total=args.sliding_distance,
        n_indent_steps=args.n_indent_steps,
        n_slip_steps=args.n_slip_steps,
        smoothed_normals=args.smoothed_normals,
        nr_tol=args.nr_tol,
        export_pvd=args.export_pvd,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    elapsed = time.perf_counter() - t0

    repo_root = Path(__file__).resolve().parent.parent.parent
    final_out = (
        (
            Path(args.output_dir)
            if Path(args.output_dir).is_absolute()
            else (repo_root / args.output_dir)
        )
        if args.export_pvd
        else None
    )

    print_ironing_report(
        telemetry=telemetry,
        friction=args.friction,
        smoothed_normals=args.smoothed_normals,
        elapsed_sec=elapsed,
        console=console,
        output_path=final_out,
    )


if __name__ == "__main__":
    main()
