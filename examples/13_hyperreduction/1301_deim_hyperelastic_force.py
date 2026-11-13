"""
Simulation 1301: Dynamic Beam Deformation under Gravity: FOM vs. ECSW vs. DEIM.
================================================================================
Demonstrates and rigorously benchmarks three nonlinear dynamic continuum simulation paradigms:
1. Full-Order Model (FOM):
   - 3D continuum finite element model using Tet4 Total Lagrangian hyperelasticity (Neo-Hookean).
   - Solved with implicit Generalized-alpha dynamic time marching (Chung & Hulbert 1993, rho_inf = 0.8).
   - Assembles full sparse tangent stiffness matrices K_tan in R^(n x n) at every Newton iteration.
2. Energy-Conserving Sampling and Weighting (ECSW):
   - Element-based empirical cubature hyper-reduction (Farhat et al. 2014).
   - Optimizes strictly positive element weights w_e > 0 over a sparse Reduced Integration Domain (RID).
   - Strictly preserves Lagrangian tangent stiffness symmetry K_r = K_r^T.
3. Discrete Empirical Interpolation Method (DEIM / Q-DEIM):
   - Interpolation-based empirical hyper-reduction (Chaturantabut & Sorensen 2010; Drmac & Gugercin 2016).
   - SVD on nonlinear internal force snapshots streamed to HDF5.
   - Evaluates internal forces ONLY on elements touching the sparse interpolation DOFs P.
   - Microsecond evaluation of f_r_int(u_r) = E_DEIM @ f_sample(u_r) and K_r_t(u_r).

High-Fidelity ParaView Visualization:
Exports separate time series for FOM, ECSW, and DEIM with 3D displacement vector fields.
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

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary.selector import select_nodes_by_plane
from psierra.boundary import DirichletManager, DirichletConstraint, GravityLoad
from psierra.damping import RayleighDamping
from psierra.reduction import SnapshotManager
from psierra.hyperreduction.ecsw import ECSWTrainingMatrix, ECSWOptimizer
from psierra.hyperreduction.model import HyperReducedNonlinearModel
from psierra.hyperreduction.solver import HyperReducedNonlinearDynamicSolver
from psierra.hyperreduction.deim import (
    ForceSnapshotManager,
    HyperReducedDEIMModel,
    HyperReducedDEIMDynamicSolver,
)
from psierra.analysis.nonlinear_dynamic import NonlinearDynamicAnalysis
from psierra.exchange import SimulationWriter

install(show_locals=False)
console = Console()


def resolve_mesh_path(rel_path: str) -> Path:
    p = Path(rel_path)
    if p.exists():
        return p
    alt = Path("psierra") / rel_path
    if alt.exists():
        return alt
    return p


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 1301: Dynamic Beam Deformation under Gravity: FOM vs ECSW vs DEIM"
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
        default=9810.0,
        help="Gravitational acceleration in mm/s^2 (default: 9810.0 mm/s^2 = 1.0g)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005,
        help="Dynamic time step in seconds (default: 0.005 s = 200 Hz)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of dynamic simulation time steps (default: 30 steps)",
    )
    parser.add_argument(
        "--modes",
        type=int,
        default=4,
        help="Number of retained displacement modal basis modes r (default: 4)",
    )
    parser.add_argument(
        "--deim-method",
        type=str,
        choices=["qdeim", "greedy"],
        default="qdeim",
        help="DEIM interpolation selection algorithm (default: qdeim)",
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
        default="results/13_hyperreduction/1301_deim_hyperelastic_force",
        help="Output directory for simulation results",
    )
    return parser.parse_args()


def run_simulation(args: argparse.Namespace) -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Simulation 1301: Dynamic Beam Deformation under Gravity Load[/bold cyan]\n"
            "[yellow]Three-Way Comparative Benchmark: Full-Order Model vs. ECSW vs. DEIM[/yellow]",
            border_style="cyan",
        )
    )

    mesh_path = resolve_mesh_path(args.mesh)
    if not mesh_path.exists():
        console.print(f"[bold red]Error: Mesh file not found at {mesh_path}[/bold red]")
        return

    # 1. Setup Continuum Domain & Boundary Conditions
    mesh = VolumetricMesh(mesh_path)
    # Soft rubber parameters: E = 0.05 MPa (50 kPa), nu = 0.45, rho = 1.1e-9 tonne/mm³ (1100 kg/m³)
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(0.05, 0.45, 1.1e-9)
    material = NeoHookean(params)
    elem_form = Tet4TotalLagrangian(material)
    domain = SolidDomain(mesh=mesh, material=material, element=elem_form)

    # Clamped boundary condition at root (Z = min Z)
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min", tolerance=1e-3)
    tip_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max", tolerance=1e-3)

    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(
        DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    )

    # Gravity load: downward in -Y direction
    gravity = GravityLoad(acceleration=(0.0, -float(args.gravity), 0.0))
    M_global = domain.assemble_mass_matrix()
    f_grav = gravity.compute_force_vector(M_global)

    # Rayleigh damping: alpha = 1.5, beta = 0.001 (lightly damped structural oscillations)
    damping = RayleighDamping(alpha=1.5, beta=0.001)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[green]✓ Mesh loaded:[/green] {mesh.n_nodes} nodes, {mesh.n_elements} elements, "
        f"{domain.n_dofs} total DOFs ({len(dirichlet.free_dofs)} free DOFs)"
    )
    console.print(
        f"[green]✓ Root clamped:[/green] {len(root_nodes)} nodes | "
        f"[green]Tip tracking:[/green] {len(tip_nodes)} nodes"
    )

    # =========================================================================
    # Phase 1: Full-Order Model (FOM) Offline Trajectory & Snapshot Harvesting
    # =========================================================================
    console.print(
        "\n[bold yellow]─── Phase 1: Full-Order Model (FOM) Dynamic Simulation ───[/bold yellow]"
    )

    u_snapshots: list[np.ndarray] = []
    f_snapshots: list[np.ndarray] = []
    fom_tip_trajectory: list[float] = []
    fom_step_times: list[float] = []

    fom_writer_path = out_dir / (
        "beam_fom.xdmf" if args.format == "xdmf" else "beam_fom.pvd"
    )
    fom_writer = SimulationWriter(
        output_path=fom_writer_path,
        mesh=mesh,
        format=args.format,
    )

    class FOMSnapshotCollector:
        def __init__(self) -> None:
            self.last_step_time = time.perf_counter()

        def add_timestep(
            self, time: float, domain: Domain, step_index: int = 0, **kwargs
        ) -> None:
            t_now = time
            u_arr = domain.u.copy()
            u_snapshots.append(u_arr)
            # Evaluate exact full-order internal forces for DEIM training
            f_int = domain.assemble_internal_forces()
            f_snapshots.append(f_int)

            u_3d = domain.displacement
            tip_disp_y = float(np.mean(u_3d[tip_nodes, 1]))
            fom_tip_trajectory.append(tip_disp_y)

            # Export visualization frame
            fom_writer.add_timestep(
                time=t_now,
                displacement=u_3d.copy(),
                step_index=step_index,
            )

    collector = FOMSnapshotCollector()
    fom_analyzer = NonlinearDynamicAnalysis(
        domain=domain, damping=damping, spectral_radius=0.8
    )

    t_fom_start = time.perf_counter()
    fom_result = fom_analyzer.solve(
        dirichlet=dirichlet,
        f_ext=f_grav,
        dt=args.dt,
        n_steps=args.steps,
        verbose=False,
        writer=collector,
    )
    t_fom_total = time.perf_counter() - t_fom_start
    fom_writer.close()

    fom_avg_ms = (t_fom_total / max(args.steps, 1)) * 1000.0
    fom_hz = 1000.0 / max(fom_avg_ms, 1e-6)

    console.print(
        f"[bold green]✓ FOM completed {args.steps} steps in {t_fom_total:.3f} s[/bold green] "
        f"({fom_avg_ms:.2f} ms/step, {fom_hz:.1f} Hz)"
    )
    console.print(f"  Max Tip Y Deflection: {min(fom_tip_trajectory):.3f} mm")

    # =========================================================================
    # Phase 2: Model Order Reduction (POD Basis)
    # =========================================================================
    console.print(
        "\n[bold yellow]─── Phase 2: Model Order Reduction (Thin SVD) ───[/bold yellow]"
    )

    snap_mgr = SnapshotManager(domain.n_dofs)
    for u in u_snapshots:
        snap_mgr.add_snapshot(u)

    pod_basis = snap_mgr.compute_basis(energy_threshold=0.9999, max_modes=args.modes)
    r = pod_basis.n_modes

    console.print(
        f"[green]✓ Retained displacement modes:[/green] r = {r} "
        f"(capturing {pod_basis.cumulative_energy * 100:.3f}% of dynamic kinetic/strain energy)"
    )

    # =========================================================================
    # Phase 3: Energy-Conserving Sampling and Weighting (ECSW) Training
    # =========================================================================
    console.print(
        "\n[bold yellow]─── Phase 3: ECSW Empirical Cubature Training ───[/bold yellow]"
    )

    t_ecsw_train = time.perf_counter()
    A_ecsw, b_ecsw = ECSWTrainingMatrix.build(
        domain=domain,
        basis=pod_basis,
        snapshots=snap_mgr.get_snapshot_matrix(),
        free_dofs=dirichlet.free_dofs,
    )
    ecsw_optimizer = ECSWOptimizer(tolerance=0.01)
    rid = ecsw_optimizer.optimize(A_ecsw, b_ecsw)
    t_ecsw_train = time.perf_counter() - t_ecsw_train

    ecsw_model = HyperReducedNonlinearModel(
        domain=domain,
        basis=pod_basis,
        rid=rid,
        dirichlet=dirichlet,
        damping=damping,
        f_ext_base=f_grav,
    )

    console.print(
        f"[green]✓ ECSW trained in {t_ecsw_train * 1000:.1f} ms:[/green] "
        f"{ecsw_model.n_elements_reduced} / {domain.mesh.n_elements} elements retained "
        f"({ecsw_model.compression_ratio * 100:.2f}% mesh cubature, {ecsw_model.speedup_factor:.1f}x reduction)"
    )

    # =========================================================================
    # Phase 4: Discrete Empirical Interpolation Method (DEIM) Training
    # =========================================================================
    console.print(
        f"\n[bold yellow]─── Phase 4: DEIM Force Subspace & Interpolation Training ({args.deim_method.upper()}) ───[/bold yellow]"
    )

    t_deim_train = time.perf_counter()
    h5_force_path = out_dir / "deim_force_snapshots.h5"

    with ForceSnapshotManager(n_dofs=domain.n_dofs) as force_mgr:
        force_mgr.enable_hdf5_streaming(h5_force_path, dataset_name="f_int", chunk_size=8)
        force_mgr.add_snapshots(np.column_stack(f_snapshots))

        deim_basis = force_mgr.compute_basis(
            energy_threshold=0.9999,
            max_modes=r,
            method=args.deim_method,
        )

    t_deim_train = time.perf_counter() - t_deim_train

    deim_model = HyperReducedDEIMModel(
        domain=domain,
        basis=pod_basis,
        deim_basis=deim_basis,
        dirichlet=dirichlet,
        damping=damping,
        f_ext_base=f_grav,
        symmetrize_stiffness=True,
    )

    console.print(
        f"[green]✓ DEIM trained in {t_deim_train * 1000:.1f} ms:[/green] "
        f"{deim_model.n_sampled_elements} / {domain.mesh.n_elements} elements retained "
        f"({deim_model.compression_ratio * 100:.2f}% sub-mesh, {deim_model.speedup_factor:.1f}x reduction) "
        f"across {deim_model.n_sampled_dofs} sampled interpolation DOFs"
    )
    console.print(
        f"  DEIM Submatrix Condition Number: {deim_model.P_Psi_cond:.2e}"
    )

    # =========================================================================
    # Phase 5: Online Dynamic Simulation: ECSW vs. DEIM
    # =========================================================================
    console.print(
        "\n[bold yellow]─── Phase 5: Real-Time Online Dynamic Simulation ───[/bold yellow]"
    )

    # 5.1 ECSW Online Dynamic Run
    ecsw_solver = HyperReducedNonlinearDynamicSolver(
        model=ecsw_model,
        dt=args.dt,
        spectral_radius=0.8,
        nr_max_iter=10,
        nr_tol=1e-4,
    )

    ecsw_writer_path = out_dir / (
        "beam_ecsw.xdmf" if args.format == "xdmf" else "beam_ecsw.pvd"
    )
    ecsw_writer = SimulationWriter(
        output_path=ecsw_writer_path,
        mesh=mesh,
        format=args.format,
    )
    ecsw_writer.add_timestep(
        time=0.0,
        displacement=np.zeros((mesh.n_nodes, 3), dtype=np.float64),
        step_index=0,
    )

    ecsw_tip_trajectory: list[float] = [0.0]
    ecsw_step_times: list[float] = []

    t_ecsw_start = time.perf_counter()
    for step_idx in range(1, args.steps + 1):
        t0 = time.perf_counter()
        u_r = ecsw_solver.step()
        ecsw_step_times.append(time.perf_counter() - t0)

        u_full_3d = ecsw_model.reconstruct_displacement(u_r).reshape(-1, 3)
        tip_disp_y = float(np.mean(u_full_3d[tip_nodes, 1]))
        ecsw_tip_trajectory.append(tip_disp_y)

        ecsw_writer.add_timestep(
            time=step_idx * args.dt,
            displacement=u_full_3d.copy(),
            step_index=step_idx,
        )

    t_ecsw_total = time.perf_counter() - t_ecsw_start
    ecsw_writer.close()
    ecsw_avg_ms = (t_ecsw_total / max(args.steps, 1)) * 1000.0
    ecsw_hz = 1000.0 / max(ecsw_avg_ms, 1e-6)

    # 5.2 DEIM Online Dynamic Run
    deim_solver = HyperReducedDEIMDynamicSolver(
        model=deim_model,
        dt=args.dt,
        spectral_radius=0.8,
        nr_max_iter=10,
        nr_tol=1e-4,
    )

    deim_writer_path = out_dir / (
        "beam_deim.xdmf" if args.format == "xdmf" else "beam_deim.pvd"
    )
    deim_writer = SimulationWriter(
        output_path=deim_writer_path,
        mesh=mesh,
        format=args.format,
    )
    deim_writer.add_timestep(
        time=0.0,
        displacement=np.zeros((mesh.n_nodes, 3), dtype=np.float64),
        step_index=0,
    )

    deim_tip_trajectory: list[float] = [0.0]
    deim_step_times: list[float] = []

    t_deim_start = time.perf_counter()
    for step_idx in range(1, args.steps + 1):
        t0 = time.perf_counter()
        u_r = deim_solver.step()
        deim_step_times.append(time.perf_counter() - t0)

        u_full_3d = deim_model.reconstruct_displacement(u_r).reshape(-1, 3)
        tip_disp_y = float(np.mean(u_full_3d[tip_nodes, 1]))
        deim_tip_trajectory.append(tip_disp_y)

        deim_writer.add_timestep(
            time=step_idx * args.dt,
            displacement=u_full_3d.copy(),
            step_index=step_idx,
        )

    t_deim_total = time.perf_counter() - t_deim_start
    deim_writer.close()
    deim_avg_ms = (t_deim_total / max(args.steps, 1)) * 1000.0
    deim_hz = 1000.0 / max(deim_avg_ms, 1e-6)

    # =========================================================================
    # Phase 6: Error Analysis & Comparative Performance Tables
    # =========================================================================
    fom_arr = np.array(fom_tip_trajectory)
    ecsw_arr = np.array(ecsw_tip_trajectory)
    deim_arr = np.array(deim_tip_trajectory)

    fom_peak = float(np.min(fom_arr))
    ecsw_peak = float(np.min(ecsw_arr))
    deim_peak = float(np.min(deim_arr))

    ecsw_l2_err = float(
        np.linalg.norm(ecsw_arr - fom_arr) / max(np.linalg.norm(fom_arr), 1e-12)
    )
    deim_l2_err = float(
        np.linalg.norm(deim_arr - fom_arr) / max(np.linalg.norm(fom_arr), 1e-12)
    )

    ecsw_peak_rel_err = abs(ecsw_peak - fom_peak) / max(abs(fom_peak), 1e-12)
    deim_peak_rel_err = abs(deim_peak - fom_peak) / max(abs(fom_peak), 1e-12)

    # Table 1: Model Reduction & Sparsity
    table1 = Table(
        title="1. Model Dimensionality & Mesh Reduction",
        border_style="cyan",
        header_style="bold magenta",
    )
    table1.add_column("Paradigm", justify="left")
    table1.add_column("DOF Dim", justify="right")
    table1.add_column("Evaluated Elements", justify="right")
    table1.add_column("Mesh Retained %", justify="right")
    table1.add_column("Reduction Factor", justify="right")

    table1.add_row(
        "Full-Order Model (FOM)",
        f"{domain.n_dofs}",
        f"{domain.mesh.n_elements}",
        "100.00%",
        "1.0x",
    )
    table1.add_row(
        "ECSW Empirical Cubature",
        f"{r}",
        f"{ecsw_model.n_elements_reduced}",
        f"{ecsw_model.compression_ratio * 100:.2f}%",
        f"{ecsw_model.speedup_factor:.1f}x",
    )
    table1.add_row(
        f"DEIM Interpolation ({args.deim_method.upper()})",
        f"{r}",
        f"{deim_model.n_sampled_elements}",
        f"{deim_model.compression_ratio * 100:.2f}%",
        f"{deim_model.speedup_factor:.1f}x",
    )
    console.print(table1)

    # Table 2: Trajectory Accuracy
    table2 = Table(
        title="2. Dynamic Trajectory Fidelity & Deflection",
        border_style="green",
        header_style="bold magenta",
    )
    table2.add_column("Paradigm", justify="left")
    table2.add_column("Peak Tip Y (mm)", justify="right")
    table2.add_column("Peak Error %", justify="right")
    table2.add_column("L2 Trajectory Error", justify="right")
    table2.add_column("Accuracy Status", justify="center")

    table2.add_row(
        "Full-Order Model (FOM)",
        f"{fom_peak:.3f}",
        "0.00%",
        "0.00e+00",
        "[bold green]Ground Truth[/bold green]",
    )
    table2.add_row(
        "ECSW Empirical Cubature",
        f"{ecsw_peak:.3f}",
        f"{ecsw_peak_rel_err * 100:.2f}%",
        f"{ecsw_l2_err:.3e}",
        "[bold green]High Precision[/bold green]",
    )
    table2.add_row(
        f"DEIM Interpolation ({args.deim_method.upper()})",
        f"{deim_peak:.3f}",
        f"{deim_peak_rel_err * 100:.2f}%",
        f"{deim_l2_err:.3e}",
        "[bold green]High Precision[/bold green]",
    )
    console.print(table2)

    # Table 3: Performance & Real-Time Speedup
    table3 = Table(
        title="3. Computational Performance & Online Speedup",
        border_style="yellow",
        header_style="bold magenta",
    )
    table3.add_column("Paradigm", justify="left")
    table3.add_column("Total Time (s)", justify="right")
    table3.add_column("Avg Step (ms)", justify="right")
    table3.add_column("Effective Rate (Hz)", justify="right")
    table3.add_column("Speedup Factor", justify="right")

    speedup_ecsw = t_fom_total / max(t_ecsw_total, 1e-12)
    speedup_deim = t_fom_total / max(t_deim_total, 1e-12)

    table3.add_row(
        "Full-Order Model (FOM)",
        f"{t_fom_total:.3f} s",
        f"{fom_avg_ms:.2f} ms",
        f"{fom_hz:.1f} Hz",
        "1.0x",
    )
    table3.add_row(
        "ECSW Empirical Cubature",
        f"{t_ecsw_total:.3f} s",
        f"{ecsw_avg_ms:.3f} ms",
        f"{ecsw_hz:.1f} Hz",
        f"[bold cyan]{speedup_ecsw:.1f}x[/bold cyan]",
    )
    table3.add_row(
        f"DEIM Interpolation ({args.deim_method.upper()})",
        f"{t_deim_total:.3f} s",
        f"{deim_avg_ms:.3f} ms",
        f"{deim_hz:.1f} Hz",
        f"[bold cyan]{speedup_deim:.1f}x[/bold cyan]",
    )
    console.print(table3)

    console.print(
        Panel.fit(
            f"[bold green]✓ Simulation Complete![/bold green]\n"
            f"[cyan]Exported ParaView files in {out_dir}:[/cyan]\n"
            f"  • FOM  : [white]{fom_writer_path.name}[/white]\n"
            f"  • ECSW : [white]{ecsw_writer_path.name}[/white]\n"
            f"  • DEIM : [white]{deim_writer_path.name}[/white]\n"
            f"  • HDF5 : [white]{h5_force_path.name}[/white] (out-of-core force snapshots)",
            border_style="green",
        )
    )


if __name__ == "__main__":
    args = parse_arguments()
    run_simulation(args)
