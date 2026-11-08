"""
Simulation 1005: Clinical Flagship Neurosurgical Brain Shift iFEM Shape Sensing.
Real-Time 3D Brain Deformation and Subcortical Tumor Drift Reconstruction from Sparse Cortical Surface Markers.

Clinical Background:
--------------------
During craniotomy and dural opening, loss of cerebrospinal fluid (CSF) and gravity cause
the brain parenchyma to sag and deform by 10-25 mm ("brain shift"). Pre-operative MRI/CT scans
become spatially inaccurate, endangering critical neurovascular structures and subcortical targets.
This simulation demonstrates intraoperative real-time shape sensing: tracking 10-20 optical markers
on the exposed cortical craniotomy window to continuously reconstruct the full 3D internal deformation
field and locate the shifted deep tumor centroid at > 10,000 Hz.

Key Architecture:
-----------------
1. Anatomical Datasets: Volumetric Brain (Tet4 solid continuum, 55,191 DOFs), Skull Base Anchor,
   and Segmented Subcortical Lesion Target (270 matched nodes).
2. Physical Forward State: Surgical posture gravity body load inducing realistic gravitational sag.
3. Intraoperative Optical Sensing: Discrete surface markers on the fronto-parietal craniotomy window
   mapped via barycentric observation operator H in R^(3 * N_sensor x n_dofs).
4. Dual Inversion Architecture:
   - Full-Order iFEM (FOM): High-fidelity continuous reconstruction benchmark.
   - Reduced-Order iFEM (ROM): Subspace-projected dense inversion in < 100 microseconds (> 10,000 Hz)
     using POD basis from multi-axial surgical trajectory snapshots.
5. Deep Tumor Centroid Drift Tracking: Direct subcortical landmark recovery via reconstruct_target_displacement.
6. Rich Telemetry & 3D Visualization: Comparative error quantification, HUD reporting, and .vtu export.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pyvista as pv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install
from scipy.spatial import cKDTree
import scipy.sparse.linalg as spspla

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import DirichletManager, DirichletConstraint, GravityLoad
from psierra.reduction import SnapshotManager
from psierra.reduction.basis import PODBasis
from psierra.inverse import (
    SensorObservationOperator,
    ReducedSensorObservationOperator,
    FullOrderShapeSensingSolver,
    ReducedOrderShapeSensingSolver,
)

install(show_locals=False)
console = Console()

SURGICAL_POSTURES: dict[str, tuple[float, float, float, str]] = {
    "lateral": (4905.0, -8495.7, 0.0, "30° Lateral Head Rotation (g_x = 0.5g, g_y = -0.866g)"),
    "supine": (0.0, -9810.0, 0.0, "Standard Supine Posture (g_y = -1.0g)"),
    "prone": (0.0, 9810.0, 0.0, "Prone Surgical Posture (g_y = +1.0g)"),
    "pterional": (3355.0, -8495.7, -3570.0, "Pterional 3-Axis Tilt (g_x=0.34g, g_y=-0.87g, g_z=-0.36g)"),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 1005: Clinical Flagship Neurosurgical Brain Shift iFEM Shape Sensing"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in non-GUI headless validation mode (auto-enabled if DISPLAY is absent)",
    )
    parser.add_argument(
        "--posture",
        type=str,
        default="lateral",
        choices=list(SURGICAL_POSTURES.keys()),
        help="Surgical posture preset determining gravity orientation (default: lateral)",
    )
    parser.add_argument(
        "--n-sensors",
        type=int,
        default=16,
        help="Number of optical surface tracking markers on craniotomy window (default: 16)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-5,
        help="Tikhonov smoothing regularizer weight (default: 1e-5)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise standard deviation added to optical markers in mm (default: 0.0 mm)",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.9999,
        help="SVD cumulative energy threshold for POD basis (default: 0.9999)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/1005_brain_shape_sensing",
        help="Output directory for reconstructed fields (.vtu) and telemetry",
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
    is_headless = args.headless or (
        not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            "[bold cyan]Project Sierra - Simulation 1005 (iFEM Clinical Flagship)[/bold cyan]\n"
            "[white]Real-Time 3D Neurosurgical Brain Shift & Subcortical Tumor Drift Shape Sensing[/white]\n"
            "[dim]Sparse Cortical Optical Tracking • Instant ROM Subspace Inversion • Sub-0.1 mm Lesion Precision[/dim]",
            border_style="cyan",
        )
    )

    # --------------------------------------------------------------------------
    # [1/6] Loading Anatomical Meshes & Subcortical Lesion Target
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][1/6] Loading Patient Anatomical Meshes & Subcortical Lesion Target[/bold yellow]"
    )
    brain_path = resolve_mesh_path("data/mrbraintumor1/mesh/brain.vtk")
    lesion_path = resolve_mesh_path("data/mrbraintumor1/mesh/lesion.vtk")
    skull_path = resolve_mesh_path("data/mrbraintumor1/mesh/skull.vtk")

    mesh = VolumetricMesh(brain_path)
    console.print(
        f"  • Volumetric Brain  : [green]{mesh.n_nodes:,}[/green] nodes | "
        f"[green]{mesh.n_elements:,}[/green] Tet4 elements | "
        f"[green]{mesh.n_dofs:,}[/green] DOFs"
    )

    # Identify lesion nodes via spatial KD-Tree
    lesion_pv = pv.read(lesion_path)
    brain_tree = cKDTree(mesh.nodes)
    _, lesion_point_to_brain = brain_tree.query(lesion_pv.points)
    lesion_node_indices = np.unique(lesion_point_to_brain)
    initial_lesion_center = mesh.nodes[lesion_node_indices].mean(axis=0)

    console.print(
        f"  • Lesion Target     : [green]{len(lesion_node_indices)}[/green] nodes mapped | "
        f"Initial Center: [dim]({initial_lesion_center[0]:.1f}, {initial_lesion_center[1]:.1f}, {initial_lesion_center[2]:.1f}) mm[/dim]"
    )

    # --------------------------------------------------------------------------
    # [2/6] Configuring Soft Tissue Physics & Cranial Base Clamping
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][2/6] Configuring Soft Tissue Physics & Cranial Base Clamping[/bold yellow]"
    )
    # Young's modulus E = 3.0 kPa = 0.003 MPa, nu = 0.45, density = 1040 kg/m³ = 1040e-12 tonne/mm³
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=0.003,
        poissons_ratio=0.45,
        mass_density=1040e-12,
    )
    material = Hookean(params)
    element = Tet4Linear(material)
    domain = SolidDomain(mesh=mesh, material=material, element=element)

    # Clamp skull base & posterior fossa: Y < -20.0 mm
    clamped_node_mask = mesh.nodes[:, 1] < -20.0
    clamped_nodes = np.where(clamped_node_mask)[0]
    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(
        DirichletConstraint(clamped_nodes, components=(0, 1, 2), value=0.0)
    )
    free_dofs = dirichlet.free_dofs

    console.print(
        f"  • Cranial Base Anchor: [green]{len(clamped_nodes):,}[/green] fixed nodes (Y < -20 mm) | "
        f"[green]{len(free_dofs):,}[/green] Free DOFs"
    )

    K_global = domain.assemble_stiffness_matrix()
    M_global = domain.assemble_mass_matrix()

    # --------------------------------------------------------------------------
    # [3/6] Forward Ground Truth Brain Shift Simulation
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][3/6] Simulating Ground Truth Surgical Brain Shift[/bold yellow]"
    )
    gx, gy, gz, posture_desc = SURGICAL_POSTURES[args.posture]
    console.print(f"  • Surgical Posture  : [cyan]{args.posture.upper()}[/cyan] ({posture_desc})")

    f_ext = GravityLoad(acceleration=(gx, gy, gz)).compute_force_vector(M_global)
    K_ff = K_global[free_dofs, :][:, free_dofs]
    f_f = f_ext[free_dofs]

    t0_fom = time.perf_counter()
    u_f = spspla.spsolve(K_ff, f_f)
    t_fom_forward = time.perf_counter() - t0_fom

    u_truth = np.zeros(domain.n_dofs, dtype=np.float64)
    u_truth[free_dofs] = u_f
    u_truth_3d = u_truth.reshape(-1, 3)

    true_lesion_shift = np.mean(u_truth_3d[lesion_node_indices], axis=0)
    true_lesion_drift_mag = float(np.linalg.norm(true_lesion_shift))
    true_max_surface_shift = float(
        np.max(np.linalg.norm(u_truth_3d[mesh.surface.node_indices], axis=1))
    )

    console.print(
        f"  • Max Cortical Shift: [green]{true_max_surface_shift:.2f} mm[/green]\n"
        f"  • Deep Lesion Drift : [bold green]{true_lesion_drift_mag:.2f} mm[/bold green] "
        f"vector: [dim]({true_lesion_shift[0]:.2f}, {true_lesion_shift[1]:.2f}, {true_lesion_shift[2]:.2f}) mm[/dim]"
    )

    # --------------------------------------------------------------------------
    # [4/6] Sparse Cortical Craniotomy Optical Marker Tracking
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][4/6] Deploying Intraoperative Optical Surface Tracking Markers[/bold yellow]"
    )
    # Fronto-parietal craniotomy window above lesion:
    # Z > 45 mm, Y in [5, 55] mm, |X| < 30 mm on cortical boundary surface
    surf_nodes = mesh.surface.node_indices
    surf_pts = mesh.nodes[surf_nodes]
    cran_mask = (
        (surf_pts[:, 2] > 45.0)
        & (surf_pts[:, 1] > 5.0)
        & (surf_pts[:, 1] < 55.0)
        & (np.abs(surf_pts[:, 0]) < 30.0)
    )
    cran_nodes = surf_nodes[cran_mask]

    if len(cran_nodes) < args.n_sensors:
        raise RuntimeError(
            f"Insufficient craniotomy candidate nodes ({len(cran_nodes)}) for {args.n_sensors} sensors."
        )

    # Evenly distribute optical marker locations across craniotomy exposure
    idx_samples = np.linspace(0, len(cran_nodes) - 1, args.n_sensors, dtype=int)
    marker_node_indices = cran_nodes[idx_samples]
    marker_coords = mesh.nodes[marker_node_indices]

    obs_operator = SensorObservationOperator(mesh, marker_coords)
    console.print(
        f"  • Craniotomy Exposure: [green]{len(cran_nodes)}[/green] candidate cortical surface nodes\n"
        f"  • Deployed Markers   : [bold cyan]{obs_operator.n_sensors}[/bold cyan] optical tracking markers\n"
        f"  • Observation Matrix : [dim]H in R^({obs_operator.matrix.shape[0]} x {obs_operator.matrix.shape[1]})[/dim]"
    )

    # Synthesize intraoperative optical measurement with optional camera noise
    d_hat_clean = obs_operator.observe(u_truth)
    if args.noise_std > 0.0:
        np.random.seed(42)
        noise = np.random.normal(0.0, args.noise_std, size=d_hat_clean.shape)
        d_hat = d_hat_clean + noise
        console.print(f"  • Measurement Noise : Gaussian [cyan]sigma = {args.noise_std:.2f} mm[/cyan] applied")
    else:
        d_hat = d_hat_clean
        console.print("  • Measurement Noise : [green]Ideal optical tracking (clean signal)[/green]")

    # --------------------------------------------------------------------------
    # [5/6] Executing Variational Shape Sensing (FOM vs ROM)
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][5/6] Executing Variational Shape Sensing Inverse FEM[/bold yellow]"
    )

    # A. Full-Order iFEM Shape Sensing (FOM)
    console.print("  [dim]Solving Full-Order iFEM System (A_full = H^T W H + alpha * K)...[/dim]")
    fom_solver = FullOrderShapeSensingSolver(
        observation_operator=obs_operator,
        stiffness_matrix=K_global,
        alpha=args.alpha,
    )
    res_fom = fom_solver.solve(d_hat)
    fom_lesion_shift = np.mean(res_fom.displacement_3d[lesion_node_indices], axis=0)
    fom_lesion_err = float(np.linalg.norm(fom_lesion_shift - true_lesion_shift))

    # B. Reduced-Order iFEM Shape Sensing (ROM)
    console.print("  [dim]Loading & Projecting POD Subspace Basis...[/dim]")
    snap_path = resolve_mesh_path("data/mrbraintumor1/mor/0902_reduced_brain_shift/snapshots.npz")
    snap_mgr = SnapshotManager.load(snap_path)
    basis = snap_mgr.compute_basis(energy_threshold=args.energy_threshold)
    r = basis.n_modes

    console.print(
        f"  • Reduced POD Basis  : [bold green]{r}[/bold green] modes "
        f"(capturing [green]{basis.cumulative_energy * 100:.3f}%[/green] energy)"
    )

    # Project stiffness K_r and observation operator H_r
    K_r = basis.project_matrix(K_global)
    H_r = obs_operator.matrix @ basis.basis
    red_obs_op = ReducedSensorObservationOperator(
        matrix=H_r,
        sensors=obs_operator.sensors,
        n_dofs=domain.n_dofs,
        basis=basis,
    )

    rom_solver = ReducedOrderShapeSensingSolver(
        reduced_observation_operator=red_obs_op,
        reduced_stiffness=K_r,
        basis=basis,
        alpha=args.alpha,
    )

    # Warm-up solve
    _ = rom_solver.solve(d_hat)

    # Benchmark microsecond solve rate
    n_bench = 100
    t0_bench = time.perf_counter()
    for _ in range(n_bench):
        res_rom = rom_solver.solve(d_hat, reconstruct_full_field=True)
    t_rom_avg = (time.perf_counter() - t0_bench) / n_bench

    # Reconstruct subcortical lesion target shift directly via reduced coordinates
    assert res_rom.reduced_coordinates is not None
    recon_lesion_u = rom_solver.reconstruct_target_displacement(
        res_rom.reduced_coordinates, lesion_node_indices
    )
    rom_lesion_shift = np.mean(recon_lesion_u, axis=0)
    rom_lesion_err = float(np.linalg.norm(rom_lesion_shift - true_lesion_shift))
    rom_rel_lesion_err = (rom_lesion_err / true_lesion_drift_mag) * 100.0

    # --------------------------------------------------------------------------
    # [6/6] Verifying Results, Performance Benchmarks, & State Export
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][6/6] Clinical Telemetry & Inversion Accuracy Verification[/bold yellow]"
    )

    results_table = Table(title="Neurosurgical Shape Sensing Inversion Benchmark", border_style="cyan")
    results_table.add_column("Metric / Indicator", style="bold white")
    results_table.add_column("Ground Truth", justify="right", style="dim")
    results_table.add_column("FOM Shape Sensing", justify="right", style="yellow")
    results_table.add_column("ROM Shape Sensing", justify="right", style="bold green")

    results_table.add_row(
        "Active System Dimension",
        f"{domain.n_dofs:,} DOFs",
        f"{domain.n_dofs:,} DOFs",
        f"{r} Modal Coordinates",
    )
    results_table.add_row(
        "Lesion Centroid Drift",
        f"{true_lesion_drift_mag:.2f} mm",
        f"{np.linalg.norm(fom_lesion_shift):.2f} mm",
        f"{np.linalg.norm(rom_lesion_shift):.2f} mm",
    )
    results_table.add_row(
        "Lesion Tracking Error",
        "---",
        f"{fom_lesion_err:.3f} mm",
        f"{rom_lesion_err:.3f} mm ({rom_rel_lesion_err:.2f}%)",
    )
    results_table.add_row(
        "Sensor RMS Residual",
        "---",
        f"{res_fom.measurement_rms_error:.4e} mm",
        f"{res_rom.measurement_rms_error:.4e} mm",
    )
    results_table.add_row(
        "Wall-Clock Solve Time",
        f"{t_fom_forward * 1e3:.2f} ms (forward)",
        f"{res_fom.solve_time_seconds * 1e3:.2f} ms",
        f"{t_rom_avg * 1e6:.1f} µs ({1.0 / t_rom_avg:,.0f} Hz)",
    )
    console.print(results_table)

    # Export VTU for inspection in ParaView / 3D Slicer
    export_path = out_dir / f"brain_shape_sensing_{args.posture}.vtu"
    grid = mesh.ugrid.copy()
    grid.point_data["Displacement_Truth"] = u_truth_3d
    grid.point_data["Displacement_ROM"] = res_rom.displacement_3d
    grid.point_data["Tracking_Error_Vector"] = res_rom.displacement_3d - u_truth_3d
    grid.point_data["Tracking_Error_Magnitude"] = np.linalg.norm(
        res_rom.displacement_3d - u_truth_3d, axis=1
    )
    grid.save(str(export_path))
    console.print(f"\n[green]✓ Exported clinical 3D volumetric fields to:[/green] [cyan]{export_path}[/cyan]")

    # Check clinical tolerance assertion
    if rom_lesion_err < 0.5:
        console.print(
            f"[bold green]✓ CLINICAL ACCURACY PASSED: Deep tumor error {rom_lesion_err:.3f} mm < 0.5 mm threshold![/bold green]"
        )
    else:
        console.print(
            f"[bold yellow]! High tumor tracking error ({rom_lesion_err:.3f} mm)[/bold yellow]"
        )

    # --------------------------------------------------------------------------
    # Interactive PyVista 3D Visualization (GUI Mode)
    # --------------------------------------------------------------------------
    if not is_headless:
        console.print("\n[bold cyan]Launching Interactive 3D Neurosurgical Visualization...[/bold cyan]")
        plotter = pv.Plotter(title="pSierra 1005: Clinical Brain Shift Shape Sensing")
        plotter.set_background("#1e1e1e")

        # 1. Undeformed ghost brain
        plotter.add_mesh(
            mesh.surface.polydata.copy(),
            color="lightgray",
            opacity=0.15,
            show_edges=False,
            label="Undeformed Scan Surface",
        )

        # 2. Deformed reconstructed brain surface with magnitude colormap
        surf_poly = mesh.surface.polydata.copy()
        surf_poly.points = (
            mesh.surface.nodes + res_rom.displacement_3d[mesh.surface.node_indices]
        )
        surf_disp_mag = np.linalg.norm(
            res_rom.displacement_3d[mesh.surface.node_indices], axis=1
        )
        surf_poly["Displacement_mm"] = surf_disp_mag
        plotter.add_mesh(
            surf_poly,
            scalars="Displacement_mm",
            cmap="turbo",
            opacity=0.85,
            show_scalar_bar=True,
            scalar_bar_args={"title": "Cortical Shift (mm)"},
        )

        # 3. Subcortical Lesion Target (True vs Reconstructed)
        lesion_true_poly = lesion_pv.copy()
        lesion_true_poly.points += true_lesion_shift
        plotter.add_mesh(
            lesion_true_poly,
            color="red",
            opacity=0.7,
            label=f"True Tumor (Shift: {true_lesion_drift_mag:.1f} mm)",
        )

        lesion_recon_poly = lesion_pv.copy()
        lesion_recon_poly.points += rom_lesion_shift
        plotter.add_mesh(
            lesion_recon_poly,
            color="cyan",
            style="wireframe",
            line_width=2.0,
            label=f"Reconstructed Tumor (Error: {rom_lesion_err:.2f} mm)",
        )

        # 4. Optical Tracking Markers
        marker_poly = pv.PolyData(marker_coords + d_hat.reshape(-1, 3))
        plotter.add_mesh(
            marker_poly,
            color="yellow",
            point_size=12,
            render_points_as_spheres=True,
            label=f"Optical Markers ({args.n_sensors}x)",
        )

        # 5. Cranial Vault context
        if skull_path.exists():
            skull_pv = pv.read(skull_path)
            plotter.add_mesh(
                skull_pv,
                color="white",
                opacity=0.08,
                style="wireframe",
                label="Cranial Vault",
            )

        plotter.add_legend(bcolor="#2d2d2d", border=True)
        plotter.camera_position = "iso"
        plotter.show()


if __name__ == "__main__":
    main()
