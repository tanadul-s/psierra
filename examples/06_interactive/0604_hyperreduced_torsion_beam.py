"""
Simulation 0604 (Interactive MOR): Real-Time Hyper-Reduced Nonlinear Axial Torsion Pipeline.

Demonstrates high-performance interactive 3D continuum torsional FEA simulation
using Energy-Conserving Sampling and Weighting (ECSW) and Discrete Empirical Interpolation
Method (DEIM) hyper-reduction with Tet4 Total Lagrangian (Tet4TL) finite-strain Neo-Hookean
hyperelastic kinematics.

Features demonstrated:
1. Real-time dynamic loop operating with fixed time step dt decoupled from rendering frame rate.
2. Selectable hyper-reduction engine:
   - ECSW: Energy-Conserving Sampling and Weighting cubature with symmetric stiffness K_r = K_r^T.
   - DEIM / Q-DEIM: Discrete Empirical Interpolation Method evaluating forces on sampled elements.
3. Offline snapshot harvesting and automatic .npz caching for instantaneous subsequent startup.
4. Exact pure torsional force couple applied at the free tip face producing dynamic twisting moment M_z.
5. Natural torsional wave propagation, elastic spring-back vibration, and numerical damping.
6. Perceptually uniform diverging colormap ('coolwarm') visualizing local twist angle in degrees:
   - 0° (neutral resting state / clamped root): Soft neutral grey
   - Positive twist (+θ, clockwise): Warm red
   - Negative twist (-θ, counter-clockwise): Cool blue
7. Interactive user torque disturbances: clockwise/counter-clockwise step pulses and harmonic resonance.
8. Execution telemetry tracking tip rotation angle, torque, simulation speed ratio, and physics rate.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
from typing import Sequence
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
from psierra.boundary import DirichletManager, DirichletConstraint
from psierra.damping import RayleighDamping
from psierra.analysis import InteractiveDynamicSimulation
from psierra.reduction import SnapshotManager
from psierra.hyperreduction.ecsw import ECSWTrainingMatrix, ECSWOptimizer
from psierra.hyperreduction.model import HyperReducedECSWModel
from psierra.hyperreduction.deim import ForceSnapshotManager, HyperReducedDEIMModel
from psierra.hyperreduction.solver import HyperReducedDynamicSolver
from psierra.exchange import SimulationWriter

install(show_locals=False)
console = Console()


def compute_twist_angles(
    nodes_ref: np.ndarray, disp: np.ndarray, center_xy: Sequence[float] = (0.0, 0.0)
) -> np.ndarray:
    """
    Computes instantaneous nodal torsional rotation angles theta_z in degrees
    about the beam longitudinal centroidal Z axis.

    Parameters:
    -----------
    nodes_ref : np.ndarray of shape (N, 3)
        Reference undeformed nodal coordinates [X, Y, Z].
    disp : np.ndarray of shape (N, 3)
        Current displacement vectors [u_x, u_y, u_z].
    center_xy : tuple of (float, float), default=(0.0, 0.0)
        Transverse centroid coordinates (X_bar, Y_bar).

    Returns:
    --------
    theta_deg : np.ndarray of shape (N,)
        Instantaneous rotation angle in degrees relative to reference state.
    """
    Xc = nodes_ref[:, 0] - center_xy[0]
    Yc = nodes_ref[:, 1] - center_xy[1]
    r = np.hypot(Xc, Yc)
    valid = r > 1e-2

    x_def = Xc + disp[:, 0]
    y_def = Yc + disp[:, 1]

    theta_rad = np.zeros(len(nodes_ref), dtype=np.float64)
    theta_rad[valid] = np.arctan2(y_def[valid], x_def[valid]) - np.arctan2(
        Yc[valid], Xc[valid]
    )
    # Wrap to [-pi, pi]
    theta_rad = (theta_rad + np.pi) % (2.0 * np.pi) - np.pi

    # For points directly on the axis (r ~ 0), extrapolate from surrounding nodes
    if not np.all(valid):
        mean_valid = np.mean(theta_rad[valid]) if np.any(valid) else 0.0
        theta_rad[~valid] = mean_valid

    return np.rad2deg(theta_rad)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 0604 (Interactive MOR): Real-Time Hyper-Reduced Nonlinear Axial Torsion Pipeline"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ecsw", "deim"],
        default="ecsw",
        help="Hyper-reduction method: 'ecsw' (default, empirical cubature) or 'deim' (discrete empirical interpolation)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless batch demonstration mode without opening a GUI window",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Physical simulation duration in seconds for demonstration sequence (default: 2.0 s)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="Physics simulation time step dt in seconds (default: 0.002 s = 500 Hz)",
    )
    parser.add_argument(
        "--spectral-radius",
        type=float,
        default=0.8,
        help="Generalized-alpha spectral radius rho_inf in [0.0, 1.0] (default: 0.8)",
    )
    parser.add_argument(
        "--modes",
        type=int,
        default=6,
        help="Number of retained modal basis modes r (default: 6)",
    )
    parser.add_argument(
        "--torque",
        type=float,
        default=2.5e8,
        help="Base applied twisting moment M_z in N*mm (default: 2.5e8 N*mm = 250 kN*m)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        default=False,
        help="Force recomputation of offline snapshots and ROM training data bypassing cache",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/beam1/mor/0604_hyperreduced_torsion",
        help="Directory path for cached snapshots and hyper-reduction cubature / basis",
    )
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        default=False,
        help="Export time-series VTU and PVD snapshots for ParaView post-processing inspection",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/06_interactive/0604_hyperreduced_torsion_beam",
        help="Output directory path for saved snapshots (default: results/06_interactive/0604_hyperreduced_torsion_beam)",
    )
    return parser.parse_args()


def resolve_path(rel_path: str | Path) -> Path:
    p = Path(rel_path)
    if p.exists():
        return p
    alt = Path("psierra") / rel_path
    if alt.exists():
        return alt
    return p


def main() -> None:
    args = parse_arguments()

    # Automatically enforce headless mode if no DISPLAY is present
    has_display = "DISPLAY" in os.environ and bool(os.environ["DISPLAY"])
    is_headless = args.headless or not has_display

    method_name = args.method.upper()

    # 0. Header Banner
    console.print(
        Panel.fit(
            f"[bold cyan]Simulation 0604 (Interactive MOR): Hyper-Reduced Dynamic Axial Torsion Pipeline[/bold cyan]\n"
            f"[dim]Decoupled Generalized-alpha Dynamic Engine with {method_name} Hyper-Reduction & Tet4TL Neo-Hookean Kinematics[/dim]\n"
            f"[dim]Mode: {'Headless Verification Mode' if is_headless else 'Interactive PyVista 3D GUI Mode'} | Paradigm: {method_name}[/dim]",
            title="[bold blue]project-sierra / hyperreduction[/bold blue]",
            border_style="cyan",
        )
    )

    # 1. Loading Mesh Geometry
    console.print("\n[bold yellow][1/6] Loading Mesh Geometry[/bold yellow]")
    mesh_path = resolve_path("data/beam1/mesh/beam1.vtk")
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

    # 2. Material & Physical Continuum Configuration (Tet4 Total Lagrangian)
    console.print(
        "\n[bold yellow][2/6] Configuring Nonlinear Material & Kinematics (Tet4TL)[/bold yellow]"
    )
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000.0,  # MPa (N/mm^2)
        poissons_ratio=0.35,
        mass_density=1100e-6,  # tonne/mm^3 (1100 kg/m^3)
    )

    material = NeoHookean(params)
    element = Tet4TotalLagrangian(material)
    domain = SolidDomain(mesh, material, element)

    console.print(f"  • Constitutive Law : [cyan]{material.__class__.__name__}[/cyan] (Finite-Strain Neo-Hookean)")
    console.print(f"  • Element Kinematics: [cyan]{element.__class__.__name__}[/cyan] (Green-Lagrange Strain, 2nd PK Stress)")
    console.print(
        f"  • Elastic Modulus  : [green]{params.youngs_modulus:.1f}[/green] MPa | "
        f"Poisson's Ratio: [green]{params.poissons_ratio:.2f}[/green] | "
        f"Shear Modulus: [green]{params.shear_modulus:.1f}[/green] MPa"
    )

    # 3. Boundary Conditions & Pure Torsional Couple Setup
    console.print(
        "\n[bold yellow][3/6] Setting Up Boundary Conditions & Torsional Force Couple[/bold yellow]"
    )
    root_nodes = select_nodes_by_plane(mesh, plane="xy", bound="min")
    tip_nodes = select_nodes_by_plane(mesh, plane="xy", bound="max")
    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0))

    # Tip face geometric center and discrete polar moment of inertia J_tip
    tip_coords = mesh.nodes[tip_nodes]
    tip_center = tip_coords.mean(axis=0)
    tip_Xc = tip_coords[:, 0] - tip_center[0]
    tip_Yc = tip_coords[:, 1] - tip_center[1]
    J_tip = float(np.sum(tip_Xc**2 + tip_Yc**2))

    # Unit torsional force vectors per tip node: Fx = -Y/J, Fy = X/J
    unit_torque_forces = np.zeros((len(tip_nodes), 3), dtype=np.float64)
    unit_torque_forces[:, 0] = -tip_Yc / J_tip
    unit_torque_forces[:, 1] = tip_Xc / J_tip

    # Full-order unit torque force vector in R^n
    f_unit_torque_full = np.zeros(domain.n_dofs, dtype=np.float64)
    for idx, node in enumerate(tip_nodes):
        f_unit_torque_full[3 * node : 3 * node + 3] = unit_torque_forces[idx]

    damping = RayleighDamping(alpha=1.0, beta=0.002)

    console.print(
        f"  • Clamped Root Face (Z = -30 mm) : [green]{len(root_nodes)}[/green] nodes ({len(dirichlet.free_dofs)} free DOFs)"
    )
    console.print(
        f"  • Twisted Tip Face  (Z = 200 mm) : [green]{len(tip_nodes)}[/green] nodes (J_tip = {J_tip:.2e} mm⁴)"
    )
    console.print(
        f"  • Baseline Twisting Torque (M_z) : [green]{args.torque / 1e6:.1f}[/green] kN·m"
    )

    # 4. Offline Snapshot Management & Hyper-Reduction Training
    console.print(
        "\n[bold yellow][4/6] Offline Snapshot Management & Hyper-Reduction Training[/bold yellow]"
    )
    cache_dir = resolve_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshots_cache_path = cache_dir / "torsion_snapshots.npz"
    ecsw_cache_path = cache_dir / "torsion_ecsw_rid.npz"
    deim_cache_path = cache_dir / "torsion_deim.npz"

    # 4.1 Collect or load displacement & force snapshots
    dt = args.dt

    def compute_fom_torsion_training_snapshots(mgr: SnapshotManager) -> None:
        console.print(
            "    [dim]Executing FOM training sequence (CW step, CCW step, harmonic pulse) to collect snapshots...[/dim]"
        )
        t_fom_start = time.perf_counter()
        fom_sim = InteractiveDynamicSimulation(
            domain=domain,
            damping=damping,
            dirichlet=dirichlet,
            dt=dt,
            spectral_radius=args.spectral_radius,
            nonlinear=True,
            nr_max_iter=5,
            modified_newton=True,
        )

        n_train_steps = 80
        force_snaps_list: list[np.ndarray] = []

        for step in range(n_train_steps):
            t_curr = step * dt
            # Multi-regime training excitation
            if t_curr < 0.04:
                # CW torque
                fom_sim.clear_perturbation_forces()
                fom_sim.apply_perturbation_forces(tip_nodes, unit_torque_forces * args.torque)
            elif t_curr < 0.08:
                # CCW torque
                fom_sim.clear_perturbation_forces()
                fom_sim.apply_perturbation_forces(tip_nodes, -unit_torque_forces * args.torque)
            elif t_curr < 0.12:
                # Harmonic excitation (20 Hz)
                m_harm = args.torque * np.sin(2.0 * np.pi * 20.0 * (t_curr - 0.08))
                fom_sim.clear_perturbation_forces()
                fom_sim.apply_perturbation_forces(tip_nodes, unit_torque_forces * m_harm)
            else:
                # Free springback decay
                fom_sim.clear_perturbation_forces()

            fom_sim.step()
            mgr.add_snapshot(fom_sim.u_full)
            force_snaps_list.append(domain.assemble_internal_forces().copy())

        t_fom_elapsed = time.perf_counter() - t_fom_start
        console.print(
            f"    [green]✓ FOM training complete in {t_fom_elapsed:.2f} s ({mgr.n_snapshots} snapshots collected)[/green]"
        )
        # Store force snapshots in extra cache
        np.savez_compressed(
            cache_dir / "torsion_force_snapshots.npz",
            forces=np.column_stack(force_snaps_list),
        )

    t0_snap = time.perf_counter()
    snapshot_mgr = SnapshotManager.get_or_compute(
        cache_path=snapshots_cache_path,
        compute_fn=compute_fom_torsion_training_snapshots,
        force_recompute=args.force_recompute,
        verbose=True,
        dt=dt,
        torque=args.torque,
    )
    t_snap_load = time.perf_counter() - t0_snap
    console.print(
        f"  • Displacement Snapshots : [green]{snapshot_mgr.n_snapshots:,}[/green] vectors in {t_snap_load*1000:.1f} ms"
    )

    # 4.2 Compute POD Basis
    t_svd_0 = time.perf_counter()
    pod_basis = snapshot_mgr.compute_basis(
        energy_threshold=0.9999,
        max_modes=args.modes,
    )
    t_svd = time.perf_counter() - t_svd_0
    r = pod_basis.n_modes

    console.print(
        f"  • Retained Modal Basis   : [green]r = {r} modes[/green] in {t_svd*1000:.2f} ms "
        f"({pod_basis.cumulative_energy * 100:.3f}% energy)"
    )

    # 4.3 Build Hyper-Reduced Model according to selected method
    hyper_model: HyperReducedECSWModel | HyperReducedDEIMModel

    if args.method == "ecsw":
        def compute_ecsw_torsion_training() -> tuple[np.ndarray, np.ndarray]:
            console.print("    [dim]Building ECSW virtual work training matrix...[/dim]")
            snap_mat = snapshot_mgr.get_snapshot_matrix()
            stride = max(1, snap_mat.shape[1] // 40)
            return ECSWTrainingMatrix.build(
                domain=domain,
                basis=pod_basis,
                snapshots=snap_mat[:, ::stride],
                free_dofs=dirichlet.free_dofs,
            )

        rid = ECSWOptimizer.get_or_compute(
            cache_path=ecsw_cache_path,
            compute_fn=compute_ecsw_torsion_training,
            tolerance=0.01,
            force_recompute=args.force_recompute,
            verbose=True,
        )

        hyper_model = HyperReducedECSWModel(
            domain=domain,
            basis=pod_basis,
            rid=rid,
            dirichlet=dirichlet,
            damping=damping,
            surface_nodes=mesh.surface.node_indices,
        )

        console.print(
            f"  • ECSW Cubature Domain   : [green]{rid.n_selected}[/green] / {mesh.n_elements} elements retained "
            f"({rid.compression_ratio * 100:.2f}% mesh, {rid.speedup_factor:.1f}x theoretical speedup)"
        )
        console.print(
            f"  • Virtual Work Accuracy  : relative error = [green]{rid.relative_error * 100:.3f}%[/green]"
        )

    else:  # DEIM
        force_cache_file = cache_dir / "torsion_force_snapshots.npz"
        if not force_cache_file.exists() or args.force_recompute:
            # Generate force snapshots if absent
            f_snaps_list: list[np.ndarray] = []
            fom_sim = InteractiveDynamicSimulation(
                domain=domain,
                damping=damping,
                dirichlet=dirichlet,
                dt=dt,
                spectral_radius=args.spectral_radius,
                nonlinear=True,
                nr_max_iter=5,
                modified_newton=True,
            )
            for step in range(80):
                t_curr = step * dt
                if t_curr < 0.04:
                    fom_sim.clear_perturbation_forces()
                    fom_sim.apply_perturbation_forces(tip_nodes, unit_torque_forces * args.torque)
                elif t_curr < 0.08:
                    fom_sim.clear_perturbation_forces()
                    fom_sim.apply_perturbation_forces(tip_nodes, -unit_torque_forces * args.torque)
                else:
                    fom_sim.clear_perturbation_forces()
                fom_sim.step()
                f_snaps_list.append(domain.assemble_internal_forces().copy())
            force_snaps_mat = np.column_stack(f_snaps_list)
            np.savez_compressed(force_cache_file, forces=force_snaps_mat)
        else:
            force_snaps_mat = np.load(force_cache_file)["forces"]

        with ForceSnapshotManager(n_dofs=domain.n_dofs) as force_mgr:
            force_mgr.add_snapshots(force_snaps_mat)
            deim_basis = force_mgr.compute_basis(
                energy_threshold=0.9999,
                max_modes=r,
                method="qdeim",
            )

        hyper_model = HyperReducedDEIMModel(
            domain=domain,
            basis=pod_basis,
            deim_basis=deim_basis,
            dirichlet=dirichlet,
            damping=damping,
            symmetrize_stiffness=True,
        )

        console.print(
            f"  • DEIM Sampled Sub-Mesh  : [green]{hyper_model.n_sampled_elements}[/green] / {mesh.n_elements} elements "
            f"({hyper_model.compression_ratio * 100:.2f}% mesh, {hyper_model.speedup_factor:.1f}x theoretical speedup) "
            f"across {hyper_model.n_sampled_dofs} sampled DOFs"
        )

    # Reduced unit torque vector: f_r_torque = Phi^T * f_unit_torque_full
    f_unit_torque_r = hyper_model.project_load(f_unit_torque_full)

    # 5. Online Hyper-Reduced Interactive Engine Initialization
    console.print(
        f"\n[bold yellow][5/6] Initializing Online {method_name} Dynamic Engine[/bold yellow]"
    )
    solver = HyperReducedDynamicSolver(
        model=hyper_model,
        dt=dt,
        spectral_radius=args.spectral_radius,
        nr_max_iter=5,
        nr_tol=1e-4,
    )

    console.print(
        f"  • Dynamic Time Step (dt)       : [green]{solver.dt * 1000.0:.2f}[/green] ms ({1.0 / solver.dt:.0f} Hz)"
    )
    console.print(
        f"  • Reduced System Dimensions    : [green]r = {r} DOFs[/green] (Dense {r}x{r} Newton-Raphson)"
    )
    console.print(
        f"  • High-Frequency Spectral Radius: [cyan]{solver.spectral_radius:.2f}[/cyan] (rho_inf)"
    )
    console.print(
        f"  • Rayleigh Physical Damping    : alpha = 1.0 (mass), beta = 0.002 (stiffness)"
    )

    def set_active_torque(moment_z: float) -> None:
        if abs(moment_z) > 1e-6:
            solver.set_perturbation(f_unit_torque_r * moment_z)
        else:
            solver.clear_perturbation()

    # Snapshot management
    save_snapshots = args.save_snapshots
    output_dir = Path(args.output_dir)
    writer = None
    if save_snapshots:
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SimulationWriter(output_dir / "result.xdmf", mesh=domain.mesh, format="xdmf")
        console.print(f"  • Exporting ParaView XDMF archive to: [dim]{output_dir}[/dim]")

    # 6. Execution Phase
    console.print(f"\n[bold yellow][6/6] Running Hyper-Reduced ({method_name}) Torsion Pipeline[/bold yellow]")

    if is_headless:
        # ---------------------------------------------------------------------
        # Headless Demonstration Sequence
        # ---------------------------------------------------------------------
        console.print(
            "  • Executing automated interactive torsional perturbation sequence:"
        )
        console.print(
            "    - [cyan]Phase 1 (0.00s - 0.30s)[/cyan]: Resting equilibrium under Generalized-alpha dissipation"
        )
        console.print(
            f"    - [cyan]Phase 2 (0.30s - 0.50s)[/cyan]: Clockwise torque step (+{args.torque/1e6:.1f} kN·m)"
        )
        console.print(
            "    - [cyan]Phase 3 (0.50s - 1.00s)[/cyan]: Instant release & free torsional spring-back vibration"
        )
        console.print(
            "    - [cyan]Phase 4 (1.00s - 1.60s)[/cyan]: Dynamic harmonic torsional excitation (M(t) = M0*sin(2*pi*20*t))"
        )
        console.print(
            "    - [cyan]Phase 5 (1.60s - 2.00s)[/cyan]: Final decay back to equilibrium"
        )

        total_duration = args.duration
        n_steps = int(round(total_duration / dt))
        snapshot_interval = max(1, int(round(0.02 / dt)))  # Snapshot every 20 ms

        tip_theta_history: list[float] = []
        time_history: list[float] = []
        snapshot_count = 0

        t_sim_start = time.perf_counter()
        for step_idx in range(n_steps):
            current_t = solver.sim_time

            # Phase 2: Clockwise torque pulse
            if 0.30 <= current_t < 0.50:
                set_active_torque(args.torque)
            elif current_t >= 0.50 and current_t < 0.52:
                set_active_torque(0.0)

            # Phase 4: Dynamic harmonic torsional excitation (20 Hz)
            if 1.00 <= current_t < 1.60:
                m_harm = args.torque * np.sin(2.0 * np.pi * 20.0 * (current_t - 1.00))
                set_active_torque(m_harm)
            elif current_t >= 1.60 and current_t < 1.62:
                set_active_torque(0.0)

            # Execute single physics step in reduced coordinates
            u_r = solver.step()

            # Reconstruct 3D full-field displacement
            u_full_3d = hyper_model.reconstruct_displacement(u_r).reshape(-1, 3)

            # Record tip rotation angle
            angles_all = compute_twist_angles(
                mesh.nodes, u_full_3d, (tip_center[0], tip_center[1])
            )
            tip_angle = float(np.mean(angles_all[tip_nodes]))
            tip_theta_history.append(tip_angle)
            time_history.append(current_t)

            # Export snapshot
            if save_snapshots and (
                step_idx % snapshot_interval == 0 or step_idx == n_steps - 1
            ):
                if writer is not None:
                    writer.add_timestep(
                        time=current_t,
                        domain=domain,
                        point_data={"Twist_Angle_deg": angles_all},
                        step_index=snapshot_count,
                    )
                snapshot_count += 1

        total_wall_loop = time.perf_counter() - t_sim_start
        if save_snapshots and writer is not None:
            writer.close()
            console.print(
                f"  • Exported [green]{snapshot_count}[/green] frames to [cyan]{output_dir / 'result.xdmf'}[/cyan] (+ .h5)"
            )

    else:
        # ---------------------------------------------------------------------
        # Interactive PyVista GUI Mode
        # ---------------------------------------------------------------------
        import pyvista as pv

        console.print(f"  • Launching interactive 3D window ({method_name} Hyper-ROM)...")
        plotter = pv.Plotter(
            window_size=[1280, 800],
            title=f"Sierra FEA - Real-Time Hyper-Reduced ({method_name}) Axial Torsion Simulation",
        )
        plotter.set_background("#1e1e24", top="#141418")

        # PyVista mesh setup
        grid = domain.to_pyvista(factor=1.0)
        grid.point_data["Twist_Angle_deg"] = np.zeros(mesh.n_nodes, dtype=np.float64)

        actor = plotter.add_mesh(
            grid,
            scalars="Twist_Angle_deg",
            clim=[-25.0, 25.0],
            cmap="coolwarm",
            show_edges=True,
            edge_color="#444444",
            scalar_bar_args={"title": "Twist Angle θz (deg)", "color": "white"},
        )

        # Camera position looking down the beam towards the tip
        plotter.camera_position = [
            (250.0, 180.0, 380.0),
            (0.0, 0.0, 85.0),
            (0.0, 1.0, 0.0),
        ]

        # Interactive state
        state = {
            "harmonic_active": False,
            "torque_steps": 0,
            "current_torque": 0.0,
            "exit_requested": False,
        }

        def is_window_active() -> bool:
            if state["exit_requested"]:
                return False
            if getattr(plotter, "_closed", False):
                return False
            if getattr(plotter, "render_window", None) is None:
                return False
            return True

        def key_twist_cw() -> None:
            state["torque_steps"] = int(round(0.15 / solver.dt))  # 150 ms pulse
            state["current_torque"] = args.torque
            console.print(
                f"  [bold red]▶ [KEY] Clockwise twisting torque (+{args.torque/1e6:.1f} kN·m) applied[/bold red]"
            )

        def key_twist_ccw() -> None:
            state["torque_steps"] = int(round(0.15 / solver.dt))
            state["current_torque"] = -args.torque
            console.print(
                f"  [bold blue]▶ [KEY] Counter-clockwise twisting torque (-{args.torque/1e6:.1f} kN·m) applied[/bold blue]"
            )

        def key_toggle_harmonic() -> None:
            state["harmonic_active"] = not state["harmonic_active"]
            if state["harmonic_active"]:
                console.print(
                    f"  [bold magenta]▶ [KEY] Harmonic torsional excitation ENABLED (20 Hz, {args.torque/1e6:.1f} kN·m)[/bold magenta]"
                )
            else:
                set_active_torque(0.0)
                console.print(
                    "  [bold dim]▶ [KEY] Harmonic torsional excitation DISABLED[/bold dim]"
                )

        def key_clear() -> None:
            state["harmonic_active"] = False
            state["torque_steps"] = 0
            state["current_torque"] = 0.0
            set_active_torque(0.0)
            console.print(
                "  [bold yellow]▶ [KEY] All dynamic twisting torques cleared[/bold yellow]"
            )

        def key_reset() -> None:
            key_clear()
            solver.reset_state()
            grid.points = mesh.nodes.copy()
            grid.point_data["Twist_Angle_deg"] = np.zeros(mesh.n_nodes, dtype=np.float64)
            grid.point_data["Displacement"] = np.zeros((mesh.n_nodes, 3), dtype=np.float64)
            try:
                plotter.update_scalar_bar_range([-25.0, 25.0])
                plotter.update()
            except Exception:
                pass
            console.print(
                "  [bold cyan]▶ [KEY] Simulation RESET to undeformed resting state[/bold cyan]"
            )

        def key_quit() -> None:
            state["exit_requested"] = True
            if not getattr(plotter, "_closed", False):
                plotter.close()
            console.print(
                "  [bold red]▶ [KEY] Exiting interactive window...[/bold red]"
            )

        plotter.add_key_event("t", key_twist_cw)
        plotter.add_key_event("T", key_twist_cw)
        plotter.add_key_event("y", key_twist_ccw)
        plotter.add_key_event("Y", key_twist_ccw)
        plotter.add_key_event("h", key_toggle_harmonic)
        plotter.add_key_event("c", key_clear)
        plotter.add_key_event("r", key_reset)
        plotter.add_key_event("q", key_quit)
        plotter.add_key_event("Q", key_quit)

        # On-screen instructions
        info_text = (
            f"Interactive Hyper-Reduced ({method_name}) Axial Torsion Controls:\n"
            f"  [T] Clockwise Twisting Torque (+{args.torque/1e6:.1f} kN·m)\n"
            f"  [Y] Counter-Clockwise Torque (-{args.torque/1e6:.1f} kN·m)\n"
            "  [H] Toggle Harmonic Resonance (20 Hz)\n"
            "  [C] Clear Dynamic Torque (Spring-Back)\n"
            "  [R] Reset Simulation\n"
            "  [Q] Exit Window"
        )
        plotter.add_text(info_text, position="upper_left", font_size=10, color="white")

        telemetry_actor = plotter.add_text(
            "Telemetry: Initializing...",
            position="lower_left",
            font_size=10,
            color="#00ffcc",
        )

        plotter.show(interactive_update=True)

        last_time = time.perf_counter()
        fps_timer = time.perf_counter()
        frame_counter = 0

        while is_window_active():
            t_now = time.perf_counter()
            dt_wall = t_now - last_time
            last_time = t_now

            # Synchronize physics steps to real wall-clock time
            n_substeps = max(1, min(int(round(dt_wall / solver.dt)), 30))

            for _ in range(n_substeps):
                if state["harmonic_active"]:
                    m_harm = args.torque * np.sin(2.0 * np.pi * 20.0 * solver.sim_time)
                    set_active_torque(m_harm)
                elif state["torque_steps"] > 0:
                    set_active_torque(state["current_torque"])
                    state["torque_steps"] -= 1
                    if state["torque_steps"] == 0:
                        set_active_torque(0.0)

                u_r = solver.step()

            # Reconstruct 3D displacement and update PyVista grid
            u_full_3d = hyper_model.reconstruct_displacement(u_r).reshape(-1, 3)
            grid.points = mesh.nodes + u_full_3d
            grid.point_data["Displacement"] = u_full_3d

            angles_all = compute_twist_angles(
                mesh.nodes, u_full_3d, (tip_center[0], tip_center[1])
            )
            grid.point_data["Twist_Angle_deg"] = angles_all

            # Synchronize symmetric colormap bounds
            max_angle = float(np.max(np.abs(angles_all)))
            clim_extent = max(25.0, max_angle)
            try:
                plotter.update_scalar_bar_range([-clim_extent, clim_extent])
                plotter.update()
            except Exception:
                break

            if not is_window_active():
                break

            frame_counter += 1
            if t_now - fps_timer >= 0.25:
                render_fps = frame_counter / (t_now - fps_timer)
                fps_timer = t_now
                frame_counter = 0
                stats = solver.get_stats()
                tip_angle_curr = float(np.mean(angles_all[tip_nodes]))
                speed_ratio = (stats.sim_time / max(stats.wall_time, 1e-6))

                telemetry_str = (
                    f"Model: {method_name} Hyper-ROM (Tet4TL Neo-Hookean, r={r})\n"
                    f"Sim Time: {stats.sim_time:.3f} s | Wall Time: {stats.wall_time:.3f} s | "
                    f"Speed Ratio: {speed_ratio:.1f}x Real-Time\n"
                    f"Physics Rate: {stats.effective_hz:.0f} Hz ({stats.average_step_ms*1000.0:.1f} µs/step) | "
                    f"Render FPS: {render_fps:.1f} FPS\n"
                    f"Tip Twist Angle: {tip_angle_curr:+.2f}° | Total Energy: {stats.total_energy:.2e} J"
                )
                try:
                    telemetry_actor.SetText(0, telemetry_str)
                except Exception:
                    pass

        if not getattr(plotter, "_closed", False):
            plotter.close()

    # Reporting Telemetry
    console.print("\n[bold yellow]Performance Telemetry Summary[/bold yellow]")
    final_stats = solver.get_stats()
    u_final_3d = hyper_model.reconstruct_displacement(solver.u_r).reshape(-1, 3)
    final_angles = compute_twist_angles(
        mesh.nodes, u_final_3d, (tip_center[0], tip_center[1])
    )
    tip_angle_final = float(np.mean(final_angles[tip_nodes]))
    max_twist = float(np.max(np.abs(final_angles)))
    speed_ratio_final = final_stats.sim_time / max(final_stats.wall_time, 1e-6)

    table = Table(
        title=f"Simulation 0604 (Interactive {method_name}): Telemetry Summary",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Metric Description", style="dim", width=30)
    table.add_column("Measured Value", justify="right", style="bold green", width=22)
    table.add_column("Engineering Notes", style="cyan", width=30)

    table.add_row(
        "Hyper-Reduction Paradigm",
        f"{method_name} (r={r} modes)",
        f"{'Empirical Cubature' if args.method == 'ecsw' else 'Force Interpolation'}",
    )
    table.add_row(
        "Continuum Formulation",
        "Tet4 Total Lagrangian",
        "Finite-Strain Neo-Hookean",
    )
    table.add_row(
        "Simulated Physical Time",
        f"{final_stats.sim_time:.3f} s",
        f"Duration: {args.duration:.2f} s",
    )
    table.add_row(
        "Total Wall-Clock Time",
        f"{final_stats.wall_time:.4f} s",
        "Elapsed computation time",
    )
    table.add_row(
        "Physics Time Steps Executed",
        f"{final_stats.steps_completed:,}",
        f"Fixed dt = {solver.dt*1000.0:.1f} ms",
    )
    table.add_row(
        "Average Physics Step Duration",
        f"{final_stats.average_step_ms*1000.0:.1f} µs ({final_stats.average_step_ms:.4f} ms)",
        "Step solve latency",
    )
    table.add_row(
        "Effective Physics Rate",
        f"{final_stats.effective_hz:,.0f} Hz",
        "Physics update frequency (>1,000 Hz)",
    )
    table.add_row(
        "Real-Time Speed Ratio",
        f"{speed_ratio_final:.1f}x Real-Time",
        "Online speedup factor",
    )
    table.add_row(
        "Final Tip Twist Angle",
        f"{tip_angle_final:+.2f}°",
        "Mean tip cross-section twist",
    )
    table.add_row(
        "Peak Beam Twist Angle",
        f"{max_twist:.2f}°",
        "Max dynamic twist amplitude",
    )
    table.add_row(
        "Instantaneous Total Energy",
        f"{final_stats.total_energy:.4e} J",
        "E_total = E_k + U",
    )

    console.print(table)
    console.print(
        f"\n[bold green]✔ Interactive Hyper-Reduced ({method_name}) Axial Torsion Pipeline completed successfully![/bold green]\n"
    )


if __name__ == "__main__":
    main()
