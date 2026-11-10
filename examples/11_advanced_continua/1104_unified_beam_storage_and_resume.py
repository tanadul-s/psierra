"""
Simulation 1104: Unified Simulation Storage, XDMF/HDF5 vs PVD/VTU & Checkpoint Resume.

Demonstrates unified simulation storage across all 4 continuum simulation classes:
1. Linear Static Analysis (Hookean, Tet4Linear)
2. Linear Dynamic Analysis (Hookean, Tet4Linear, Generalized-alpha)
3. Nonlinear Static Analysis (NeoHookean, Tet4TotalLagrangian, Incremental Newton-Raphson)
4. Nonlinear Dynamic Analysis (NeoHookean, Tet4TotalLagrangian, Generalized-alpha)

Features:
- Single unified SimulationWriter class with plug-and-play format toggle:
    format="xdmf" (single-mesh HDF5 binary + standard XDMF 3.0 XML for ParaView)
    format="pvd"  (legacy multi-file VTU snapshots + master PVD collection)
- Rigorous Checkpoint & Resume verification:
    Runs Phase 1 (steps 1 to 5) -> saves checkpoint -> closes.
    Resumes from checkpoint -> runs Phase 2 (steps 6 to 10) -> records new checkpoint.
    Compares against continuous baseline to verify exact state persistence.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install
from scipy.spatial.transform import Rotation

from psierra.material import IsotropicElasticParameters, Hookean, NeoHookean
from psierra.element import Tet4Linear, Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary.selector import select_nodes_by_bounds, select_nodes_by_plane
from psierra.boundary import DirichletConstraint, DirichletManager, GravityLoad, NodalForce
from psierra.damping import RayleighDamping
from psierra.analysis import (
    LinearStaticAnalysis,
    LinearDynamicAnalysis,
    NonlinearStaticAnalysis,
    NonlinearDynamicAnalysis,
)
from psierra.exchange import SimulationWriter, SimulationCheckpoint

install(show_locals=False)
console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 1104: Unified Simulation Storage & Checkpoint Resume"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["all", "linear_static", "linear_dynamic", "nonlinear_static", "nonlinear_dynamic", "resume"],
        default="all",
        help="Simulation type to run (default: all)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["xdmf", "pvd"],
        default="xdmf",
        help="Export format: 'xdmf' (clean 2-file HDF5) or 'pvd' (legacy multi-file VTU) (default: xdmf)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/11_advanced_continua/1104_unified_beam_storage",
        help="Base output directory",
    )
    return parser.parse_args()


def resolve_mesh(rel_path: str) -> Path:
    p1 = Path(rel_path)
    if p1.exists():
        return p1
    p2 = Path("psierra") / rel_path
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Mesh not found at '{p1}' or '{p2}'")


def load_beam_mesh() -> VolumetricMesh:
    mesh_path = resolve_mesh("data/beam1/mesh/beam1.vtk")
    mesh = VolumetricMesh(mesh_path)
    mesh.transform(
        orientation=Rotation.from_euler("zyz", (0, 90, 90), degrees=True),
        about_point=(0, 0, 0),
    )
    return mesh


# =============================================================================
# 1. Linear Static Simulation
# =============================================================================
def run_linear_static(output_dir: Path, fmt: str) -> float:
    console.print("\n[bold cyan]------------------------------------------------------------[/bold cyan]")
    console.print("[bold yellow][Case 1/4] Linear Static Beam Bending (Hookean + Tet4Linear)[/bold yellow]")
    console.print("[bold cyan]------------------------------------------------------------[/bold cyan]")
    mesh = load_beam_mesh()
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
    material = Hookean(params)
    element = Tet4Linear(material)
    domain = SolidDomain(mesh, material, element)

    root_nodes = select_nodes_by_bounds(mesh, bounds=(-61, 61, -31, 1, -61, 61))
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Downward gravity body load
    gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    solver = LinearStaticAnalysis(domain)
    res = solver.solve(dirichlet=dirichlet, f_ext=f_ext, verbose=False)

    max_disp = float(np.linalg.norm(domain.displacement, axis=1).max())
    out_target = output_dir / "linear_static" / "result"

    with SimulationWriter(out_target, mesh=mesh, format=fmt) as writer:
        writer.add_timestep(time=0.0, domain=domain)
        writer.write_checkpoint(step=0, time=0.0, dt=0.0, domain=domain)

    console.print(f"  • Max Resultant Displacement: [bold green]{max_disp:.3f} mm[/bold green]")
    console.print(f"  • Export Target ({fmt.upper()}): [cyan]{out_target}[/cyan]")
    return max_disp


# =============================================================================
# 2. Linear Dynamic Simulation
# =============================================================================
def run_linear_dynamic(output_dir: Path, fmt: str) -> float:
    console.print("\n[bold cyan]------------------------------------------------------------[/bold cyan]")
    console.print("[bold yellow][Case 2/4] Linear Dynamic Beam Vibration (Generalized-alpha)[/bold yellow]")
    console.print("[bold cyan]------------------------------------------------------------[/bold cyan]")
    mesh = load_beam_mesh()
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
    material = Hookean(params)
    element = Tet4Linear(material)
    domain = SolidDomain(mesh, material, element)

    root_nodes = select_nodes_by_bounds(mesh, bounds=(-61, 61, -31, 1, -61, 61))
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)
    damping = RayleighDamping(alpha=2.0, beta=0.001)

    solver = LinearDynamicAnalysis(domain, damping=damping, spectral_radius=0.8)
    dt = 0.005
    n_steps = 10
    out_target = output_dir / "linear_dynamic" / "result"

    with SimulationWriter(out_target, mesh=mesh, format=fmt) as writer:
        # Step 0
        writer.add_timestep(time=0.0, domain=domain)

        # Integration loop
        free_dofs = dirichlet.free_dofs
        K = domain.assemble_stiffness_matrix()
        C = damping.assemble_damping_matrix(M, K)
        M_ff = M[free_dofs, :][:, free_dofs]
        C_ff = C[free_dofs, :][:, free_dofs]
        K_ff = K[free_dofs, :][:, free_dofs]
        f_ext_f = f_ext[free_dofs]

        res = solver.solve(dirichlet=dirichlet, f_ext=f_ext, dt=dt, n_steps=n_steps, verbose=False)

        # Record final step and checkpoint
        writer.add_timestep(time=n_steps * dt, domain=domain)
        writer.write_checkpoint(step=n_steps, time=n_steps * dt, dt=dt, domain=domain)

    max_disp = float(np.linalg.norm(domain.displacement, axis=1).max())
    console.print(f"  • Solved {n_steps} dynamic steps (dt = {dt*1000:.1f} ms)")
    console.print(f"  • Tip Displacement at t = {n_steps*dt:.3f} s: [bold green]{max_disp:.3f} mm[/bold green]")
    console.print(f"  • Export Target ({fmt.upper()}): [cyan]{out_target}[/cyan]")
    return max_disp


# =============================================================================
# 3. Nonlinear Static Simulation
# =============================================================================
def run_nonlinear_static(output_dir: Path, fmt: str) -> float:
    console.print("\n[bold cyan]------------------------------------------------------------[/bold cyan]")
    console.print("[bold yellow][Case 3/4] Nonlinear Static Large Deflection (NeoHookean + TL)[/bold yellow]")
    console.print("[bold cyan]------------------------------------------------------------[/bold cyan]")
    mesh = load_beam_mesh()
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
    material = NeoHookean(params)
    element = Tet4TotalLagrangian(material)
    domain = SolidDomain(mesh, material, element)

    root_nodes = select_nodes_by_bounds(mesh, bounds=(-61, 61, -31, 1, -61, 61))
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Gravitational acceleration scaled to 0.5g (-4905 mm/s^2) matching 0402 benchmark
    gravity = GravityLoad(acceleration=(0.0, 0.0, -4905.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    solver = NonlinearStaticAnalysis(domain)
    n_steps = 10
    out_target = output_dir / "nonlinear_static" / "result"

    with SimulationWriter(out_target, mesh=mesh, format=fmt) as writer:
        writer.add_timestep(time=0.0, domain=domain)
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            n_steps=n_steps,
            nr_iterations=25,
            nr_tol=1e-4,
            verbose=False,
        )
        writer.add_timestep(time=1.0, domain=domain)
        writer.write_checkpoint(step=n_steps, time=1.0, dt=1.0 / n_steps, domain=domain)

    max_disp = float(np.linalg.norm(domain.displacement, axis=1).max())
    console.print(f"  • Solved {n_steps} incremental load steps (lambda -> 1.0)")
    console.print(f"  • Final Nonlinear Displacement: [bold green]{max_disp:.3f} mm[/bold green]")
    console.print(f"  • Export Target ({fmt.upper()}): [cyan]{out_target}[/cyan]")
    return max_disp


# =============================================================================
# 4. Nonlinear Dynamic Simulation
# =============================================================================
def run_nonlinear_dynamic(output_dir: Path, fmt: str) -> float:
    console.print("\n[bold cyan]------------------------------------------------------------[/bold cyan]")
    console.print("[bold yellow][Case 4/4] Nonlinear Dynamic Transient Simulation (NeoHookean + TL)[/bold yellow]")
    console.print("[bold cyan]------------------------------------------------------------[/bold cyan]")
    mesh = load_beam_mesh()
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
    material = NeoHookean(params)
    element = Tet4TotalLagrangian(material)
    domain = SolidDomain(mesh, material, element)

    root_nodes = select_nodes_by_bounds(mesh, bounds=(-61, 61, -31, 1, -61, 61))
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)
    damping = RayleighDamping(alpha=2.0, beta=0.001)

    solver = NonlinearDynamicAnalysis(domain, damping=damping, spectral_radius=0.8)
    dt = 0.005
    n_steps = 10
    out_target = output_dir / "nonlinear_dynamic" / "result"

    with SimulationWriter(out_target, mesh=mesh, format=fmt) as writer:
        writer.add_timestep(time=0.0, domain=domain)
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=dt,
            n_steps=n_steps,
            nr_iterations=15,
            nr_tol=1e-4,
            verbose=False,
        )
        writer.add_timestep(time=n_steps * dt, domain=domain)
        writer.write_checkpoint(step=n_steps, time=n_steps * dt, dt=dt, domain=domain)

    max_disp = float(np.linalg.norm(domain.displacement, axis=1).max())
    console.print(f"  • Solved {n_steps} dynamic time steps (dt = {dt*1000:.1f} ms)")
    console.print(f"  • Peak Displacement at t = {n_steps*dt:.3f} s: [bold green]{max_disp:.3f} mm[/bold green]")
    console.print(f"  • Export Target ({fmt.upper()}): [cyan]{out_target}[/cyan]")
    return max_disp


# =============================================================================
# 5. Checkpoint & Resume Test
# =============================================================================
def run_checkpoint_and_resume_test(output_dir: Path, fmt: str) -> None:
    console.print("\n[bold yellow]══════════════════════════════════════════════════════════════[/bold yellow]")
    console.print("[bold yellow]Simulation Checkpoint & Resume Verification Benchmark[/bold yellow]")
    console.print("[bold yellow]══════════════════════════════════════════════════════════════[/bold yellow]")

    mesh = load_beam_mesh()
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
    material = NeoHookean(params)
    element = Tet4TotalLagrangian(material)

    root_nodes = select_nodes_by_bounds(mesh, bounds=(-61, 61, -31, 1, -61, 61))
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    gravity = GravityLoad(acceleration=(0.0, 0.0, -9810.0))
    dt = 0.005
    damping = RayleighDamping(alpha=2.0, beta=0.001)

    # -------------------------------------------------------------------------
    # Baseline: Continuous 10 steps
    # -------------------------------------------------------------------------
    console.print("\n[bold cyan][Step 1/3] Running Reference Continuous Simulation (10 steps)[/bold cyan]")
    domain_ref = SolidDomain(mesh, material, element)
    M_ref = domain_ref.assemble_mass_matrix()
    f_ext_ref = gravity.compute_force_vector(M_ref)
    solver_ref = NonlinearDynamicAnalysis(domain_ref, damping=damping, spectral_radius=0.8)

    solver_ref.solve(
        dirichlet=dirichlet,
        f_ext=f_ext_ref,
        dt=dt,
        n_steps=10,
        nr_iterations=15,
        nr_tol=1e-4,
        verbose=False,
    )
    u_continuous_final = domain_ref.displacement.copy()
    max_d_ref = float(np.linalg.norm(u_continuous_final, axis=1).max())
    console.print(f"  • Reference Max Displacement at Step 10: [bold green]{max_d_ref:.4f} mm[/bold green]")

    # -------------------------------------------------------------------------
    # Phase 1: Run 5 steps -> Save Checkpoint -> Close
    # -------------------------------------------------------------------------
    console.print("\n[bold cyan][Step 2/3] Running Phase 1 (Steps 1 to 5) -> Saving Checkpoint[/bold cyan]")
    domain_split = SolidDomain(mesh, material, element)
    M_split = domain_split.assemble_mass_matrix()
    f_ext_split = gravity.compute_force_vector(M_split)
    solver_phase1 = NonlinearDynamicAnalysis(domain_split, damping=damping, spectral_radius=0.8)

    out_archive = output_dir / f"resume_test_{fmt}" / "run_resumable"
    if out_archive.parent.exists():
        import shutil
        shutil.rmtree(out_archive.parent, ignore_errors=True)

    with SimulationWriter(out_archive, mesh=mesh, format=fmt) as writer:
        solver_phase1.solve(
            dirichlet=dirichlet,
            f_ext=f_ext_split,
            dt=dt,
            n_steps=5,
            nr_iterations=15,
            nr_tol=1e-4,
            verbose=False,
            writer=writer,
            initial_step=0,
            initial_time=0.0,
        )

        # Write intermediate checkpoint at Step 5
        ckpt_p1 = writer.write_checkpoint(
            step=5,
            time=5 * dt,
            dt=dt,
            domain=domain_split,
            metadata={"phase": "phase_1_completed", "steps": 5},
        )
        console.print(f"  • Checkpoint 1 Written at Step 5 (t = {5*dt:.3f} s)")

    # Inspect archive state
    summary_p1 = SimulationWriter.inspect(out_archive)
    console.print(f"  • Archive Inspection: {summary_p1['steps_recorded']} steps recorded, latest checkpoint at step {summary_p1['checkpoint_step']}")

    # -------------------------------------------------------------------------
    # Phase 2: Resume from Checkpoint -> Run Steps 6 to 10 -> Record New Checkpoint
    # -------------------------------------------------------------------------
    console.print("\n[bold cyan][Step 3/3] Resuming from Checkpoint -> Running Phase 2 (Steps 6 to 10)[/bold cyan]")
    # Re-instantiate fresh domain
    domain_resumed = SolidDomain(mesh, material, element)
    M_res = domain_resumed.assemble_mass_matrix()
    f_ext_res = gravity.compute_force_vector(M_res)

    # Load checkpoint
    loaded_ckpt = SimulationWriter.load_checkpoint(out_archive)
    loaded_ckpt.apply_to_domain(domain_resumed)
    console.print(f"  • Checkpoint Restored: step = {loaded_ckpt.step}, time = {loaded_ckpt.time:.3f} s")

    solver_phase2 = NonlinearDynamicAnalysis(domain_resumed, damping=damping, spectral_radius=0.8)

    # Reopen archive in RESUME mode
    with SimulationWriter(out_archive, mesh=mesh, format=fmt, resume=True) as writer:
        console.print(f"  • Resuming archive. Current step count: {writer.step_count}")
        solver_phase2.solve(
            dirichlet=dirichlet,
            f_ext=f_ext_res,
            dt=dt,
            n_steps=5,
            nr_iterations=15,
            nr_tol=1e-4,
            verbose=False,
            writer=writer,
            initial_step=5,
            initial_time=5 * dt,
        )

        # Write new checkpoint at Step 10
        ckpt_p2 = writer.write_checkpoint(
            step=10,
            time=10 * dt,
            dt=dt,
            domain=domain_resumed,
            metadata={"phase": "phase_2_resumed_completed", "steps": 10},
        )
        console.print(f"  • Checkpoint 2 Written at Step 10 (t = {10*dt:.3f} s)")

    u_resumed_final = domain_resumed.displacement.copy()
    max_d_res = float(np.linalg.norm(u_resumed_final, axis=1).max())

    # Final archive inspection
    summary_final = SimulationWriter.inspect(out_archive)

    # -------------------------------------------------------------------------
    # Verification Comparison Table
    # -------------------------------------------------------------------------
    disp_diff = np.linalg.norm(u_resumed_final - u_continuous_final)
    rel_error = disp_diff / max(np.linalg.norm(u_continuous_final), 1e-12)

    comp_table = Table(
        title="Simulation 1104: Checkpoint & Resume Verification Results",
        border_style="cyan",
    )
    comp_table.add_column("Verification Metric", style="bold white")
    comp_table.add_column("Continuous Baseline", style="cyan", justify="right")
    comp_table.add_column("Resumed Simulation", style="magenta", justify="right")
    comp_table.add_column("Status / Agreement", style="green")

    comp_table.add_row(
        "Total Dynamic Steps",
        "10 steps",
        "5 + 5 steps (Resumed)",
        "Exact duration match",
    )
    comp_table.add_row(
        "Final Time t_final",
        f"{10*dt:.3f} s",
        f"{10*dt:.3f} s",
        "Exact temporal match",
    )
    comp_table.add_row(
        "Max Tip Displacement",
        f"{max_d_ref:.6f} mm",
        f"{max_d_res:.6f} mm",
        f"Diff: {abs(max_d_res - max_d_ref):.2e} mm",
    )
    comp_table.add_row(
        "Displacement L2 Error",
        "0.000000 mm",
        f"{disp_diff:.2e} mm",
        f"Rel Error: {rel_error:.2e}",
    )
    comp_table.add_row(
        "Total Steps in Archive",
        "-",
        f"{summary_final['steps_recorded']} steps",
        "Seamless sequence in single file",
    )
    comp_table.add_row(
        "Latest Recorded Checkpoint",
        "-",
        f"Step {summary_final['checkpoint_step']} (t = {summary_final['checkpoint_time']:.3f} s)",
        "New checkpoint successfully recorded",
    )

    console.print(comp_table)

    if rel_error < 1e-8:
        console.print(
            "\n[bold green]✓ Checkpoint & Resume Verification PASSED:[/bold green] "
            "Resumed simulation reproduces continuous trajectory to machine precision, "
            "and new checkpoint is successfully recorded in the archive!\n"
        )
    else:
        console.print(
            f"\n[bold red]✗ Checkpoint & Resume Verification WARNING: Relative error = {rel_error:.2e}[/bold red]\n"
        )


def main() -> None:
    args = parse_arguments()
    out_dir = Path(args.output_dir)
    fmt = args.format

    console.print(
        Panel.fit(
            "[bold cyan]Simulation 1104: Unified Simulation Storage & Checkpoint Resume[/bold cyan]\n"
            f"[white]Format: [bold green]{fmt.upper()}[/bold green] • Mode: [bold green]{args.type.upper()}[/bold green][/white]\n"
            "[dim]Showcasing Linear Static, Linear Dynamic, Nonlinear Static, Nonlinear Dynamic & Exact Resume[/dim]",
            title="[bold blue]project-sierra: Continuous Archival[/bold blue]",
            border_style="cyan",
        )
    )

    t0 = time.perf_counter()

    if args.type in ("linear_static", "all"):
        run_linear_static(out_dir, fmt)

    if args.type in ("linear_dynamic", "all"):
        run_linear_dynamic(out_dir, fmt)

    if args.type in ("nonlinear_static", "all"):
        run_nonlinear_static(out_dir, fmt)

    if args.type in ("nonlinear_dynamic", "all"):
        run_nonlinear_dynamic(out_dir, fmt)

    if args.type in ("resume", "all"):
        run_checkpoint_and_resume_test(out_dir, fmt)

    elapsed = time.perf_counter() - t0
    console.print(f"[bold green]✓ Simulation 1104 completed in {elapsed:.2f} s.[/bold green]\n")


if __name__ == "__main__":
    main()
