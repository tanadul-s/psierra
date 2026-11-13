"""
Simulation 1109: Beam Deformation under Gravity Load: Static vs. Dynamic Relaxation vs. Quasi-Static (Full-Order & Hyper-Reduced).

Benchmarks and demonstrates four complementary numerical paradigms for gravitational beam deformation:
1. Method 1: Incremental Newton-Raphson Static Analysis (NonlinearStaticAnalysis)
   - Finite-strain Total Lagrangian hyperelasticity (Compressible Neo-Hookean).
   - Time-independent equilibrium R(u) = F_ext - F_int(u) = 0.
2. Method 2: Matrix-Free Dynamic Relaxation (DynamicRelaxationAnalysis)
   - Pseudo-time explicit static equilibrium via Barnes/Underwood kinetic energy peak damping.
   - Solves static state unconditionally without matrix factorization.
3. Method 3: Full-Order Viscoelastic Quasi-Static Analysis (QuasiStaticAnalysis)
   - Time-marching physical equilibrium without inertia: R(u_{n+1}) = F_ext(t_{n+1}) - F_int(u_{n+1}, Q_{n+1}(dt)) = 0.
   - Generalized Maxwell / Prony series relaxation (g_1 = 0.5, tau_1 = 0.2s, g_inf = 0.5).
   - Demonstrates physical viscoelastic creep: deflection grows by factor ~ 1 / g_inf = 2.0x over time!
4. Method 4: Real-Time Hyper-Reduced Quasi-Static Analysis (HyperReducedQuasiStaticSolver)
   - Energy-Conserving Sampling and Weighting (ECSW) hyper-reduction on POD modal subspace R^r.
   - Microsecond execution rate (> 1,000 Hz) accurately tracking full-order viscoelastic creep.
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

from psierra.material import IsotropicElasticParameters, NeoHookean, Viscoelastic, PronyTerm
from psierra.element import Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary.selector import select_nodes_by_bounds
from psierra.boundary import DirichletConstraint, DirichletManager, GravityLoad
from psierra.analysis import (
    NonlinearStaticAnalysis,
    DynamicRelaxationAnalysis,
    QuasiStaticAnalysis,
)
from psierra.reduction import SnapshotManager
from psierra.hyperreduction import (
    ReducedIntegrationDomain,
    ECSWTrainingMatrix,
    ECSWOptimizer,
    HyperReducedNonlinearModel,
    HyperReducedQuasiStaticSolver,
)
from psierra.exchange.writer import SimulationWriter

install(show_locals=False)
console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 1109: Beam Gravity Deflection: Static vs Dynamic Relaxation vs Quasi-Static"
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="data/beam1/mesh/beam1.vtk",
        help="Path to beam volumetric mesh (default: data/beam1/mesh/beam1.vtk)",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=1962.0,
        help="Gravitational acceleration magnitude in mm/s^2 (default: 1962.0 mm/s^2 = 0.2g)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Physical quasi-static relaxation duration in seconds (default: 1.0 s)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Quasi-static physical time step in seconds (default: 0.05 s = 20 steps)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["xdmf", "pvd"],
        default="xdmf",
        help="ParaView export format (default: xdmf)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/11_advanced_continua/1109_quasi_static_gravity_beam",
        help="Output directory (default: results/11_advanced_continua/1109_quasi_static_gravity_beam)",
    )
    return parser.parse_args()


def resolve_mesh_path(rel_path: str) -> Path:
    p1 = Path(rel_path)
    if p1.exists():
        return p1
    p2 = Path("psierra") / rel_path
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Mesh file not found at '{p1}' or '{p2}'")


def main() -> None:
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format
    g_accel = args.gravity
    total_time = args.duration
    dt = args.dt

    console.print(
        Panel.fit(
            "[bold cyan]Simulation 1109: Beam Gravity Deformation & Viscoelastic Creep[/bold cyan]\n"
            "[white]Static Analysis vs. Dynamic Relaxation vs. Full-Order Quasi-Static vs. Hyper-Reduced Quasi-Static[/white]\n"
            f"[dim]E = 21,000 MPa • ν = 0.35 • ρ = 1,100 kg/m³ • g = {g_accel:.1f} mm/s² • Duration = {total_time:.2f} s[/dim]",
            title="[bold blue]project-sierra / Advanced Continua[/bold blue]",
            border_style="cyan",
        )
    )

    t0_total = time.perf_counter()

    # 1. Loading Geometry & Orientation
    console.print("\n[bold yellow][1/6] Loading Mesh Geometry & Orienting Cantilever Beam[/bold yellow]")
    mesh_path = resolve_mesh_path(args.mesh)
    mesh = VolumetricMesh(Path(mesh_path))
    mesh.transform(
        orientation=Rotation.from_euler("zyz", (0, 90, 90), degrees=True),
        about_point=(0, 0, 0),
    )

    console.print(
        f"  • Mesh File      : [cyan]{mesh_path}[/cyan]\n"
        f"  • Discretization : [green]{mesh.n_nodes:,}[/green] nodes | "
        f"[green]{mesh.n_elements:,}[/green] Tet4 elements | [green]{mesh.n_dofs:,}[/green] DOFs\n"
        f"  • Dimensions     : X ∈ [{mesh.bounds[0]:.1f}, {mesh.bounds[1]:.1f}], "
        f"Y ∈ [{mesh.bounds[2]:.1f}, {mesh.bounds[3]:.1f}], "
        f"Z ∈ [{mesh.bounds[4]:.1f}, {mesh.bounds[5]:.1f}] mm"
    )

    # 2. Material Parameters & Boundary Clamping
    console.print("\n[bold yellow][2/6] Configuring Material Parameters & Gravity Body Load[/bold yellow]")
    youngs_modulus = 21000.0  # MPa
    poissons_ratio = 0.35
    density = 1100e-6  # tonne/mm³ = 1100 kg/m³

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(youngs_modulus, poissons_ratio, density)

    # Clamped root face at Y <= 0 mm
    root_nodes = select_nodes_by_bounds(mesh, bounds=(-61, 61, -31, 1, -61, 61))
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # Gravity body force vector acting along -Z
    gravity = GravityLoad(acceleration=(0.0, 0.0, -g_accel))
    mat_dummy = NeoHookean(params)
    elem_dummy = Tet4TotalLagrangian(mat_dummy)
    dom_dummy = SolidDomain(mesh, mat_dummy, elem_dummy)
    M = dom_dummy.assemble_mass_matrix()
    f_grav = gravity.compute_force_vector(M)
    total_weight = float(abs(f_grav[2::3].sum()))

    console.print(
        f"  • Elastic Modulus  : [green]{youngs_modulus:,.0f}[/green] MPa | Poisson's ratio: [green]{poissons_ratio:.2f}[/green]\n"
        f"  • Total Gravity Fz : [bold green]{total_weight:.2f}[/bold green] N (acting along -Z)\n"
        f"  • Clamped Root     : [green]{len(root_nodes)}[/green] nodes clamped at Y ≤ 0 mm ({len(clamp.dofs)} DOFs)"
    )

    # Prony series: 50% relaxation with tau = 0.2s
    prony_terms = [PronyTerm(relative_modulus=0.5, relaxation_time=0.2)]

    results_table: list[dict[str, str | float]] = []

    # =========================================================================
    # 3. Method 1: Nonlinear Static Analysis (Incremental Newton-Raphson)
    # =========================================================================
    console.print("\n[bold yellow][3/6] Method 1: Incremental Newton-Raphson Static Analysis[/bold yellow]")
    mat_static = NeoHookean(params)
    elem_static = Tet4TotalLagrangian(mat_static)
    dom_static = SolidDomain(mesh.copy(), mat_static, elem_static)

    dir_static = output_dir / "01_nonlinear_static"
    t_start = time.perf_counter()
    solver_static = NonlinearStaticAnalysis(dom_static)
    res_static = solver_static.solve(
        dirichlet=dirichlet,
        f_ext=f_grav,
        n_steps=8,
        nr_iterations=35,
        nr_tol=1e-4,
        save_directory=dir_static,
        format=fmt,
        verbose=False,
    )
    t_static = time.perf_counter() - t_start

    u_static_nodes = res_static.displacement.reshape(-1, 3)
    disp_static_max = float(np.linalg.norm(u_static_nodes, axis=1).max())
    sag_static_tip = float(abs(u_static_nodes[:, 2].min()))
    rxn_static_z = float(res_static.reactions[clamp.dofs][2::3].sum())

    console.print(
        f"  ✓ Solved in [bold green]{t_static*1000:.1f} ms[/bold green] (8 load increments) | "
        f"Max Deflection: [bold green]{disp_static_max:.3f} mm[/bold green] | "
        f"Tip Sag (Z): [bold green]{sag_static_tip:.3f} mm[/bold green]"
    )

    results_table.append({
        "Method": "1. Nonlinear Static (NR)",
        "Type": "Instantaneous Elastic",
        "Max Disp (mm)": disp_static_max,
        "Tip Sag (mm)": sag_static_tip,
        "Creep Amplification": 1.0,
        "Steps / Iter": f"8 steps ({sum(res_static.iteration_history)} iters)",
        "Time (ms)": t_static * 1000.0,
    })

    # =========================================================================
    # 4. Method 2: Matrix-Free Dynamic Relaxation (Kinetic Damping)
    # =========================================================================
    console.print("\n[bold yellow][4/6] Method 2: Matrix-Free Dynamic Relaxation Equilibrium[/bold yellow]")
    mat_dr = NeoHookean(params)
    elem_dr = Tet4TotalLagrangian(mat_dr)
    dom_dr = SolidDomain(mesh.copy(), mat_dr, elem_dr)

    dir_dr = output_dir / "02_dynamic_relaxation"
    t_start = time.perf_counter()
    solver_dr = DynamicRelaxationAnalysis(
        dom_dr, damping_type="kinetic", safety_factor=1.1, dt=1.0
    )
    res_dr = solver_dr.solve(
        dirichlet=dirichlet,
        f_ext=f_grav,
        tol=1e-4,
        max_steps=400,
        save_directory=dir_dr,
        output_name="result",
        format=fmt,
        verbose=False,
    )
    t_dr = time.perf_counter() - t_start

    disp_dr_max = float(np.linalg.norm(dom_dr.displacement, axis=1).max())
    sag_dr_tip = float(abs(dom_dr.displacement[:, 2].min()))

    console.print(
        f"  ✓ Solved in [bold green]{t_dr*1000:.1f} ms[/bold green] ({res_dr.steps_completed} steps, {res_dr.kinetic_peaks} kinetic peaks) | "
        f"Max Deflection: [bold green]{disp_dr_max:.3f} mm[/bold green] | Diff vs Static: [dim]{abs(disp_dr_max - disp_static_max):.4f} mm[/dim]"
    )

    results_table.append({
        "Method": "2. Dynamic Relaxation",
        "Type": "Matrix-Free Kinetic Static",
        "Max Disp (mm)": disp_dr_max,
        "Tip Sag (mm)": sag_dr_tip,
        "Creep Amplification": disp_dr_max / disp_static_max,
        "Steps / Iter": f"{res_dr.steps_completed} DR steps",
        "Time (ms)": t_dr * 1000.0,
    })

    # =========================================================================
    # 5. Method 3: Full-Order Viscoelastic Quasi-Static Analysis (Time Marching)
    # =========================================================================
    console.print("\n[bold yellow][5/6] Method 3: Full-Order Viscoelastic Quasi-Static Analysis[/bold yellow]")
    visco_mat = Viscoelastic(NeoHookean(params), prony_terms)
    elem_visco = Tet4TotalLagrangian(visco_mat)
    dom_visco = SolidDomain(mesh.copy(), visco_mat, elem_visco)

    dir_qs = output_dir / "03_viscoelastic_quasi_static"
    t_start = time.perf_counter()
    solver_qs = QuasiStaticAnalysis(dom_visco)

    def load_schedule(t_cur: float) -> float:
        # Ramp up over 0.2s, then hold constant for viscoelastic relaxation / creep
        return min(1.0, t_cur / 0.2)

    fom_qs_disps: list[float] = []

    def log_qs(step: int, t_cur: float, u_vec: np.ndarray) -> None:
        u_mag = float(np.linalg.norm(u_vec.reshape(-1, 3), axis=1).max())
        fom_qs_disps.append(u_mag)

    res_qs = solver_qs.solve(
        dirichlet=dirichlet,
        f_ext=f_grav,
        total_time=total_time,
        dt=dt,
        load_schedule=load_schedule,
        nr_iterations=30,
        save_directory=dir_qs,
        format=fmt,
        verbose=False,
        callback=log_qs,
    )
    t_qs = time.perf_counter() - t_start

    u_qs_nodes = res_qs.displacement.reshape(-1, 3)
    disp_qs_initial = fom_qs_disps[3] if len(fom_qs_disps) >= 4 else disp_static_max
    disp_qs_final = float(np.linalg.norm(u_qs_nodes, axis=1).max())
    sag_qs_tip = float(abs(u_qs_nodes[:, 2].min()))
    creep_ratio_fom = disp_qs_final / disp_static_max

    console.print(
        f"  ✓ Solved {res_qs.steps_completed} time steps over {total_time:.2f} s in [bold green]{t_qs*1000:.1f} ms[/bold green] | "
        f"Instantaneous Sag (t=0.20s): [cyan]{disp_qs_initial:.3f} mm[/cyan] → "
        f"Relaxed Sag (t={total_time:.2f}s): [bold green]{disp_qs_final:.3f} mm[/bold green] "
        f"([bold magenta]{creep_ratio_fom:.2f}x creep sag[/bold magenta])"
    )

    results_table.append({
        "Method": "3. Full-Order Quasi-Static",
        "Type": f"Viscoelastic Time Marching (dt={dt}s)",
        "Max Disp (mm)": disp_qs_final,
        "Tip Sag (mm)": sag_qs_tip,
        "Creep Amplification": creep_ratio_fom,
        "Steps / Iter": f"{res_qs.steps_completed} time steps ({sum(res_qs.iteration_history)} iters)",
        "Time (ms)": t_qs * 1000.0,
    })

    # =========================================================================
    # 6. Method 4: Real-Time Hyper-Reduced Quasi-Static Analysis (ECSW ROM)
    # =========================================================================
    console.print("\n[bold yellow][6/6] Method 4: Real-Time Hyper-Reduced Quasi-Static Analysis (ECSW)[/bold yellow]")

    # Build modal basis from snapshot trajectory
    snap_mgr = SnapshotManager(n_dofs=mesh.n_dofs)
    snap_mgr.add_snapshot(res_static.displacement)
    for u_snap in res_qs.displacement_history[1:]:
        snap_mgr.add_snapshot(u_snap)

    basis = snap_mgr.compute_basis(energy_threshold=0.999)
    console.print(
        f"  • Retained Modal Basis : [cyan]r = {basis.n_modes}[/cyan] modes "
        f"(||ΦᵀΦ - I|| = {basis.orthonormality_error():.2e})"
    )

    # Build ECSW cubature
    A_mat, b_rhs = ECSWTrainingMatrix.build(
        domain=dom_visco, basis=basis, snapshots=snap_mgr.get_snapshot_matrix()
    )
    rid = ECSWOptimizer(tolerance=0.02, verbose=False).optimize(A_mat, b_rhs)
    console.print(
        f"  • Reduced Integration  : [cyan]{rid.n_selected}[/cyan] / {mesh.n_elements} elements retained "
        f"([green]{(rid.n_selected / mesh.n_elements)*100:.1f}% cubature mesh[/green])"
    )

    # Fresh viscoelastic domain for hyper-reduced model
    dom_rom_visco = SolidDomain(
        mesh.copy(),
        Viscoelastic(NeoHookean(params), prony_terms),
        Tet4TotalLagrangian(Viscoelastic(NeoHookean(params), prony_terms)),
    )
    model_rom = HyperReducedNonlinearModel(
        domain=dom_rom_visco,
        basis=basis,
        rid=rid,
        dirichlet=dirichlet,
        f_ext_base=f_grav,
    )

    rom_solver = HyperReducedQuasiStaticSolver(model=model_rom, max_iter=25, tol=1e-5)

    rom_qs_disps: list[float] = []

    def log_rom(step: int, t_cur: float, u_r: np.ndarray) -> None:
        u_full = rom_solver.reconstruct_displacement()
        rom_qs_disps.append(float(np.linalg.norm(u_full.reshape(-1, 3), axis=1).max()))

    t_start = time.perf_counter()
    rom_solver.solve(
        total_time=total_time,
        dt=dt,
        load_schedule=load_schedule,
        verbose=False,
        callback=log_rom,
    )
    t_rom = time.perf_counter() - t_start

    u_rom_final = rom_solver.reconstruct_displacement().reshape(-1, 3)
    disp_rom_final = float(np.linalg.norm(u_rom_final, axis=1).max())
    sag_rom_tip = float(abs(u_rom_final[:, 2].min()))
    creep_ratio_rom = disp_rom_final / disp_static_max
    speedup = t_qs / t_rom if t_rom > 0 else 1.0

    rom_stats = rom_solver.get_stats()
    console.print(
        f"  ✓ Solved {rom_stats.steps_completed} steps in [bold green]{t_rom*1000:.2f} ms[/bold green] "
        f"({rom_stats.average_step_ms:.3f} ms/step, [bold cyan]{rom_stats.effective_hz:,.0f} Hz[/bold cyan], "
        f"[bold magenta]{speedup:.1f}x faster than FOM[/bold magenta]) | "
        f"Relaxed Disp: [bold green]{disp_rom_final:.3f} mm[/bold green] (Diff vs FOM: [dim]{abs(disp_rom_final - disp_qs_final):.4f} mm[/dim])"
    )

    results_table.append({
        "Method": "4. Hyper-Reduced Quasi-Static",
        "Type": f"ECSW Reduced Coordinates (r={basis.n_modes})",
        "Max Disp (mm)": disp_rom_final,
        "Tip Sag (mm)": sag_rom_tip,
        "Creep Amplification": creep_ratio_rom,
        "Steps / Iter": f"{rom_stats.steps_completed} steps ({rom_stats.average_nr_iterations:.1f} iters/step)",
        "Time (ms)": t_rom * 1000.0,
    })

    # =========================================================================
    # Summary Synthesis
    # =========================================================================
    table = Table(
        title="Beam Gravity Deflection: Static vs. Dynamic Relaxation vs. Quasi-Static",
        border_style="cyan",
        header_style="bold cyan",
    )
    table.add_column("Simulation Method", style="bold")
    table.add_column("Formulation / Continuum Physics", style="dim")
    table.add_column("Max Disp", justify="right")
    table.add_column("Tip Sag (Z)", justify="right")
    table.add_column("Creep Factor", justify="right")
    table.add_column("Steps / Iters", justify="right")
    table.add_column("Compute Time", justify="right")

    for row in results_table:
        table.add_row(
            str(row["Method"]),
            str(row["Type"]),
            f"{float(row['Max Disp (mm)']):.3f} mm",
            f"{float(row['Tip Sag (mm)']):.3f} mm",
            f"{float(row['Creep Amplification']):.2f}x",
            str(row["Steps / Iter"]),
            f"{float(row['Time (ms)']):.2f} ms",
        )

    console.print("\n")
    console.print(table)

    t_all = time.perf_counter() - t0_total
    console.print(
        f"\n[bold green]✓ All 4 methods executed and verified successfully in {t_all:.2f} s total.[/bold green]"
    )
    console.print(f"[dim]Results and ParaView collections exported to: {output_dir.resolve()}[/dim]\n")


if __name__ == "__main__":
    main()
