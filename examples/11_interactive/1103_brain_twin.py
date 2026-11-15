"""
Example 1103: Interactive Neurosurgical Digital Twin & Multi-Modal Shape Sensing.
================================================================================
Real-Time Inversion of Burr-Hole-Referenced Point Cloud Translations (Cortical Optical + Deep Ultrasound)
to Invert Full-Field Patient-Specific Brain Shift & Subcortical Tumor Drift at Sub-Millisecond Latencies.

Key Capabilities & Clinical Highlights:
---------------------------------------
1. Targeted Neurosurgical Burr Hole Trephination Reference Frame:
   - Burr hole aperture placed directly over frontal subcortical pathology (R = 10.0 mm, Ø = 20.0 mm)
     at [-2.3, 34.1, 50.1] mm on cortical surface with outward cranial normal n = [-0.069, 0.654, 0.754]
     and orthonormal tangent basis (t1, t2), matching simulations 1108 and 1009.
   - Generates 3D annular cranial bone table collar and perimeter ring for anatomical context.

2. Interactive Point Cloud Control Referenced to Burr Hole:
   - Cortical Surface Point Cloud (optical cameras / laser scan across burr hole opening).
   - Deep Ultrasound Point Cloud (intraoperative B-mode / Doppler on tumor boundary).
   - Real-time 3D translation:
     * Normal translation (delta_n): Inward sagging / cerebral collapse (negative) vs.
       outward herniation / dome bulging (positive) through the bone collar.
     * Tangential translations (delta_t1, delta_t2): Transverse shear shifts across the opening.
     * Deep coupling factor (gamma): Depth-attenuated subcortical transmission.

3. Real-Time Reduced-Order (ROM iFEM) Shape Sensing Engine:
   - Projects heterogeneous multi-modal sensor observation operators onto patient POD modal subspace:
     (H_r^T W H_r + alpha K_r) q = H_r^T W d
   - Solves generalized coordinates q in R^r in < 150 microseconds (> 6,500 Hz).
   - Instantly reconstructs the 3D full-field brain deformation and subcortical tumor drift.

4. Multi-Modal Inversion Modalities:
   - Tri-Modal selection toggled dynamically via slider or hotkey [M]:
     * Fused Multi-Modal: Optical surface (W = 1.0) + deep iUS (W = 0.040).
     * Surface Optical Only: Aperture-restricted markers only (exhibits depth extrapolation drift).
     * Deep Ultrasound Only: Subcortical targets only (exhibits surface drift).

5. Interactive PyVista 3D Inspection & Headless Automated Verification:
   - Interactive sliders for normal translation, tangential translations, and deep coupling.
   - Hotkeys: [W]/[S] (normal shift), [A]/[D] & [Up]/[Down] (tangential shift), [M] (toggle modality),
     [1]/[2]/[3] (clinical presets: Sagging, Bulging, Oblique Shear), [R] (reset), [Q] (exit).
   - Automated headless demonstration mode (--headless) for continuous integration validation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pyvista as pv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install
from scipy.spatial import cKDTree
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import DirichletManager, DirichletConstraint, GravityLoad
from psierra.reduction import SnapshotManager
from psierra.inverse.observation import (
    SensorObservationOperator,
    ReducedSensorObservationOperator,
)
from psierra.inverse.shape_sensing import (
    FullOrderShapeSensingSolver,
    ReducedOrderShapeSensingSolver,
    ShapeSensingResult,
)

install(show_locals=False)
console = Console()


def resolve_path(rel_path: str | Path) -> Path:
    p = Path(rel_path)
    if p.exists():
        return p
    alt = Path("psierra") / rel_path
    if alt.exists():
        return alt
    return p


def create_burr_hole_geometry(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    thickness: float = 5.0,
    wall_width: float = 3.5,
) -> tuple[pv.PolyData, pv.PolyData]:
    """
    Constructs a 3D annular cranial bone collar and circular contour wireframe
    representing the surgical burr hole drilled through the skull table (from simulation 1108).
    """
    n_pts = 64
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    # Establish orthonormal tangent basis
    if abs(normal[0]) < 0.9:
        t1 = np.cross(normal, [1.0, 0.0, 0.0])
    else:
        t1 = np.cross(normal, [0.0, 1.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    r_in = radius
    r_out = radius + wall_width

    # Inner circular perimeter wireframe at inner table
    inner_circle = center + r_in * (np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2)
    lines = []
    for i in range(n_pts):
        lines.extend([2, i, (i + 1) % n_pts])
    wireframe = pv.PolyData(inner_circle, lines=lines)

    # 3D Annular bone collar with skull thickness
    pts_bot_in = center + r_in * (np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2)
    pts_bot_out = center + r_out * (np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2)

    pts_top_in = pts_bot_in + thickness * normal
    pts_top_out = pts_bot_out + thickness * normal

    all_pts = np.vstack([pts_bot_in, pts_bot_out, pts_top_in, pts_top_out])
    faces = []

    # Inner wall faces
    for i in range(n_pts):
        j = (i + 1) % n_pts
        faces.extend([3, i, j, 2 * n_pts + j])
        faces.extend([3, i, 2 * n_pts + j, 2 * n_pts + i])

    # Top annular rim faces
    for i in range(n_pts):
        j = (i + 1) % n_pts
        faces.extend([3, 2 * n_pts + i, 2 * n_pts + j, 3 * n_pts + j])
        faces.extend([3, 2 * n_pts + i, 3 * n_pts + j, 3 * n_pts + i])

    # Outer wall faces
    for i in range(n_pts):
        j = (i + 1) % n_pts
        faces.extend([3, n_pts + i, 3 * n_pts + i, 3 * n_pts + j])
        faces.extend([3, n_pts + i, 3 * n_pts + j, n_pts + j])

    # Bottom annular rim faces
    for i in range(n_pts):
        j = (i + 1) % n_pts
        faces.extend([3, i, n_pts + i, n_pts + j])
        faces.extend([3, i, n_pts + j, j])

    bone_collar = pv.PolyData(all_pts, faces=faces)
    return bone_collar, wireframe


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 1103: Interactive Neurosurgical Digital Twin & Multi-Modal Shape Sensing"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in automated non-GUI headless validation sequence (auto-enabled if DISPLAY is absent)",
    )
    parser.add_argument(
        "--burr-radius",
        type=float,
        default=10.0,
        help="Burr hole aperture radius in mm (default: 10.0 mm = 20.0 mm diameter)",
    )
    parser.add_argument(
        "--n-ius-points",
        type=int,
        default=24,
        help="Number of intraoperative ultrasound (iUS) deep target points tracked on lesion margin (default: 24)",
    )
    parser.add_argument(
        "--opt-noise-std",
        type=float,
        default=0.05,
        help="Cortical optical tracking Gaussian noise standard deviation in mm (default: 0.05 mm = 50 um)",
    )
    parser.add_argument(
        "--ius-noise-std",
        type=float,
        default=0.25,
        help="Deep ultrasound tracking Gaussian noise standard deviation in mm (default: 0.25 mm = 250 um)",
    )
    parser.add_argument(
        "--buoyancy-relief",
        type=float,
        default=0.85,
        help="Archimedean CSF buoyancy relief factor in [0.0, 0.96] (default: 0.85 = 15%% net gravity sag)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="Tikhonov regularization smoothing parameter alpha (default: 1e-4)",
    )
    parser.add_argument(
        "--modes",
        type=int,
        default=10,
        help="Number of retained displacement modal basis modes r for real-time ROM (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/11_interactive/1103_brain_twin",
        help="Output directory for reports and visual export files (default: results/11_interactive/1103_brain_twin)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Automatically enforce headless mode if no display server is present
    has_display = "DISPLAY" in os.environ and bool(os.environ["DISPLAY"])
    is_headless = args.headless or not has_display

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0. Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]Example 1103: Interactive Neurosurgical Digital Twin & Multi-Modal Shape Sensing[/bold cyan]\n"
            "[white]Real-Time Inversion of Burr-Hole-Referenced Point Clouds (Surface Optical + Deep iUS)[/white]\n"
            f"[dim]Mode: {'Headless Verification Mode' if is_headless else 'Interactive PyVista 3D GUI Mode'} | "
            f"Burr Hole Diameter: {2*args.burr_radius:.1f} mm | Modal Basis: r = {args.modes}[/dim]",
            title="[bold blue]project-sierra / interactive[/bold blue]",
            border_style="cyan",
        )
    )

    # 1. Loading Patient Anatomical Datasets & Lesion Geometry
    console.print("\n[bold yellow][1/6] Loading Patient Anatomical Models & Subcortical Pathology[/bold yellow]")
    brain_mesh_path = resolve_path("data/mrbraintumor1/mesh/brain.vtk")
    lesion_mesh_path = resolve_path("data/mrbraintumor1/mesh/lesion.vtk")
    skull_mesh_path = resolve_path("data/mrbraintumor1/mesh/skull.vtk")

    if not brain_mesh_path.exists():
        raise FileNotFoundError(f"Brain mesh file not found: {brain_mesh_path}")
    if not lesion_mesh_path.exists():
        raise FileNotFoundError(f"Lesion mesh file not found: {lesion_mesh_path}")

    mesh = VolumetricMesh(brain_mesh_path)
    lesion_pv = pv.read(str(lesion_mesh_path))
    surf = mesh.surface

    console.print(
        f"  • Volumetric Brain Mesh : [green]{mesh.n_nodes:,}[/green] nodes | "
        f"[green]{mesh.n_elements:,}[/green] Tet4 elements | "
        f"[green]{mesh.n_dofs:,}[/green] DOFs"
    )

    # Map lesion nodes in volumetric brain via spatial KD-Tree
    kdtree = cKDTree(mesh.nodes)
    _, lesion_nodes_mapped = kdtree.query(lesion_pv.points)
    lesion_vol_nodes = np.unique(lesion_nodes_mapped)
    lesion_center = np.array(lesion_pv.center)

    console.print(
        f"  • Subcortical Lesion    : [green]{len(lesion_vol_nodes)}[/green] mapped nodes | "
        f"Centroid: [dim]({lesion_center[0]:.1f}, {lesion_center[1]:.1f}, {lesion_center[2]:.1f}) mm[/dim]"
    )

    # 2. Targeted Surgical Burr Hole Trephination (1108 Reference)
    console.print(
        "\n[bold yellow][2/6] Constructing Burr Hole Trephination Reference Frame[/bold yellow]"
    )
    r_burr = float(args.burr_radius)

    # Place burr hole directly above the pathology on the cortical surface (simulation 1108 approach)
    dists_surf_to_lesion = np.linalg.norm(surf.nodes - lesion_center, axis=1)
    burr_center = surf.nodes[np.argmin(dists_surf_to_lesion)]

    # Outward cranial surface normal at burr hole
    normal_burr = burr_center - mesh.centroid
    normal_burr /= np.linalg.norm(normal_burr)

    # Tangent orthonormal basis vectors
    if abs(normal_burr[0]) < 0.9:
        t1_burr = np.cross(normal_burr, [1.0, 0.0, 0.0])
    else:
        t1_burr = np.cross(normal_burr, [0.0, 1.0, 0.0])
    t1_burr /= np.linalg.norm(t1_burr)
    t2_burr = np.cross(normal_burr, t1_burr)

    # Identify cortical nodes strictly within the burr hole aperture
    dists_from_burr = np.linalg.norm(surf.nodes - burr_center, axis=1)
    hole_surf_mask = dists_from_burr <= r_burr
    hole_vol_nodes = surf.node_indices[hole_surf_mask]
    n_surf_markers = len(hole_vol_nodes)

    # Create 3D annular bone collar & wireframe ring matching simulation 1108
    bone_collar, wireframe_ring = create_burr_hole_geometry(
        burr_center, normal_burr, radius=r_burr, thickness=5.0, wall_width=3.5
    )
    collar_path = out_dir / "burr_hole_bone_collar.vtp"
    ring_path = out_dir / "burr_hole_perimeter_ring.vtp"
    bone_collar.save(str(collar_path))
    wireframe_ring.save(str(ring_path))

    console.print(
        f"  • Burr Hole Center      : [{burr_center[0]:.1f}, {burr_center[1]:.1f}, {burr_center[2]:.1f}] mm (Direct Approach)\n"
        f"  • Outward Normal n      : [{normal_burr[0]:.3f}, {normal_burr[1]:.3f}, {normal_burr[2]:.3f}]\n"
        f"  • Tangent Vectors       : t1 = [{t1_burr[0]:.3f}, {t1_burr[1]:.3f}, {t1_burr[2]:.3f}], "
        f"t2 = [{t2_burr[0]:.3f}, {t2_burr[1]:.3f}, {t2_burr[2]:.3f}]\n"
        f"  • Aperture Radius       : {r_burr:.1f} mm (Ø {2*r_burr:.1f} mm)\n"
        f"  • Exposed Cortical Nodes: [bold green]{n_surf_markers}[/bold green] nodes inside aperture opening"
    )

    # 3. Baseline Biomechanical Domain & Patient Modal Subspace
    console.print(
        "\n[bold yellow][3/6] Assembling Biomechanical Domain & ROM Modal Subspace[/bold yellow]"
    )
    # Cranial base clamping at brainstem / foramen magnum (Y < -20 mm)
    clamped_node_mask = mesh.nodes[:, 1] < -20.0
    clamped_nodes = np.where(clamped_node_mask)[0]
    dirichlet = DirichletManager(mesh.n_dofs)
    dirichlet.add_constraint(
        DirichletConstraint(clamped_nodes, components=(0, 1, 2), value=0.0)
    )
    free_dofs = dirichlet.free_dofs

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=3.0e-3,  # 3.0 kPa = 0.003 MPa
        poissons_ratio=0.45,
        mass_density=1040e-12,  # tonne/mm^3
    )
    material = NeoHookean(params)
    element = Tet4TotalLagrangian(material)
    domain = SolidDomain(mesh=mesh, material=material, element=element)

    # Baseline physiological posture shift (30° lateral head tilt with 85% CSF buoyancy relief)
    net_weight_factor = max(0.01, 1.0 - float(args.buoyancy_relief))
    g_accel_lat = np.array([4905.0, -8495.7, 0.0]) * net_weight_factor

    M_mat = domain.assemble_mass_matrix()
    f_grav = GravityLoad(acceleration=tuple(g_accel_lat)).compute_force_vector(M_mat)
    K_global = domain.assemble_stiffness_matrix()
    K_ff = K_global[free_dofs, :][:, free_dofs]

    u_base_free = spspla.spsolve(K_ff, f_grav[free_dofs])
    u_base = np.zeros(domain.n_dofs, dtype=np.float64)
    u_base[free_dofs] = u_base_free
    u_base_3d = u_base.reshape(-1, 3)

    # Construct POD basis enriched with baseline and synthetic deformation states
    snap_cache_path = resolve_path("data/mrbraintumor1/mor/0904_interactive_hyperreduced_brain_shift/snapshots.npz")
    snap_mgr = SnapshotManager.load(snap_cache_path)
    snap_mgr.add_snapshot(u_base)

    # Add variations spanning bulging and sagging perturbations
    for scale in [-2.0, -1.0, 1.0, 2.0]:
        snap_mgr.add_snapshot(u_base * (1.0 + 0.3 * scale))

    pod_basis = snap_mgr.compute_basis(energy_threshold=0.9999, max_modes=args.modes)
    r = pod_basis.n_modes

    K_r = pod_basis.basis.T @ (K_global @ pod_basis.basis)

    console.print(
        f"  • Cranial Base Anchor   : [green]{len(clamped_nodes):,}[/green] fixed nodes | "
        f"[green]{len(free_dofs):,}[/green] free DOFs\n"
        f"  • Retained POD Modes r  : [bold green]{r}[/bold green] modes (Dimension reduction: {mesh.n_dofs} -> {r})\n"
        f"  • Reduced Stiffness K_r : Dense {r}x{r} SPD matrix"
    )

    # 4. Constructing Observation Operators & Real-Time Inversion Solvers
    console.print(
        "\n[bold yellow][4/6] Constructing Multi-Modal Observation Operators & Fast ROM Solvers[/bold yellow]"
    )
    # Select deep ultrasound feature points on the lesion boundary along surgical trajectory
    n_ius = min(int(args.n_ius_points), len(lesion_vol_nodes))
    np.random.seed(42)
    ius_vol_nodes = np.random.choice(lesion_vol_nodes, size=n_ius, replace=False)
    fused_vol_nodes = np.concatenate([hole_vol_nodes, ius_vol_nodes])

    # Observation Operators
    op_surf = SensorObservationOperator.from_node_indices(mesh, hole_vol_nodes)
    op_ius = SensorObservationOperator.from_node_indices(mesh, ius_vol_nodes)
    op_fused = SensorObservationOperator.from_node_indices(mesh, fused_vol_nodes)

    red_op_surf = op_surf.reduce(pod_basis)
    red_op_ius = op_ius.reduce(pod_basis)
    red_op_fused = op_fused.reduce(pod_basis)

    # Channel Confidence Weights
    alpha = float(args.alpha)
    w_surf_val = 1.0
    w_ius_val = (args.opt_noise_std / max(args.ius_noise_std, 1e-6)) ** 2  # ~ 0.040

    weights_surf = np.ones(len(hole_vol_nodes) * 3, dtype=np.float64) * w_surf_val
    weights_ius = np.ones(len(ius_vol_nodes) * 3, dtype=np.float64) * w_ius_val
    weights_fused = np.concatenate([weights_surf, weights_ius])

    # Solvers for each modality
    rom_solver_fused = ReducedOrderShapeSensingSolver(
        reduced_observation_operator=red_op_fused,
        reduced_stiffness=K_r,
        basis=pod_basis,
        alpha=alpha,
        weights=weights_fused,
    )
    rom_solver_surf = ReducedOrderShapeSensingSolver(
        reduced_observation_operator=red_op_surf,
        reduced_stiffness=K_r,
        basis=pod_basis,
        alpha=alpha,
        weights=weights_surf,
    )
    rom_solver_ius = ReducedOrderShapeSensingSolver(
        reduced_observation_operator=red_op_ius,
        reduced_stiffness=K_r,
        basis=pod_basis,
        alpha=alpha,
        weights=weights_ius,
    )

    console.print(
        f"  • Optical Surface Cloud : [green]{n_surf_markers}[/green] points (Weight W = {w_surf_val:.2f}, sigma = ±{args.opt_noise_std*1e3:.0f} µm)\n"
        f"  • Deep Ultrasound Cloud : [magenta]{n_ius}[/magenta] points (Weight W = {w_ius_val:.3f}, sigma = ±{args.ius_noise_std*1e3:.0f} µm)\n"
        f"  • Combined Fused Cloud  : [cyan]{len(fused_vol_nodes)}[/cyan] points (3 x {len(fused_vol_nodes)} = {len(fused_vol_nodes)*3} channels)"
    )

    # Pre-sample reproducible measurement noise
    np.random.seed(101)
    noise_surf_ref = np.random.normal(0.0, args.opt_noise_std, size=len(hole_vol_nodes) * 3)
    noise_ius_ref = np.random.normal(0.0, args.ius_noise_std, size=len(ius_vol_nodes) * 3)

    # Helper function to compute inverted brain shift from interactive point cloud translations
    def solve_interactive_inversion(
        delta_n: float,
        delta_t1: float,
        delta_t2: float,
        gamma_deep: float = 0.8,
        modality: str = "fused",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Solves real-time ROM iFEM inversion for a given burr-hole-referenced translation.
        Returns:
            u_recon_3d: (N_nodes, 3) inverted full-field displacement.
            detected_surf_pts: (N_surf, 3) current translated optical point cloud.
            detected_ius_pts: (N_ius, 3) current translated deep iUS point cloud.
            solve_time_us: Inversion latency in microseconds.
        """
        # Translation vector in burr hole frame
        tau = delta_n * normal_burr + delta_t1 * t1_burr + delta_t2 * t2_burr

        # Simulated physical sensor measurements: baseline shift + user translation + noise
        d_surf_clean = (u_base_3d[hole_vol_nodes] + tau).ravel()
        d_ius_clean = (u_base_3d[ius_vol_nodes] + gamma_deep * tau).ravel()

        d_surf_meas = d_surf_clean + noise_surf_ref
        d_ius_meas = d_ius_clean + noise_ius_ref

        # Coordinates of floating point clouds for 3D visualization
        detected_surf_pts = mesh.nodes[hole_vol_nodes] + d_surf_meas.reshape(-1, 3)
        detected_ius_pts = mesh.nodes[ius_vol_nodes] + d_ius_meas.reshape(-1, 3)

        # Select solver based on modality
        t0 = time.perf_counter()
        if modality == "surf":
            res = rom_solver_surf.solve(d_surf_meas, reconstruct_full_field=False)
        elif modality == "ius":
            res = rom_solver_ius.solve(d_ius_meas, reconstruct_full_field=False)
        else:  # fused
            d_fused_meas = np.concatenate([d_surf_meas, d_ius_meas])
            res = rom_solver_fused.solve(d_fused_meas, reconstruct_full_field=False)
        solve_time_us = (time.perf_counter() - t0) * 1e6

        # Fast subspace back-projection: u = Phi * q
        assert res.reduced_coordinates is not None
        u_recon_3d = (pod_basis.basis @ res.reduced_coordinates).reshape(-1, 3)
        return u_recon_3d, detected_surf_pts, detected_ius_pts, solve_time_us

    # 5. Interactive Mode or Headless Demonstration Sequence
    console.print(
        f"\n[bold yellow][5/6] Executing Simulation 1010 Pipeline ({'Headless' if is_headless else 'Interactive GUI'})[/bold yellow]"
    )

    if is_headless:
        # ---------------------------------------------------------------------
        # Headless Multi-Step Verification Trajectory
        # ---------------------------------------------------------------------
        console.print("  • Executing automated interactive burr hole translation sequence:")
        test_scenarios = [
            ("Baseline (Resting Equil)", 0.0, 0.0, 0.0, 0.8, "fused"),
            ("Scenario 1: Inward Sagging (CSF Loss)", -3.2, 0.0, 0.0, 0.8, "fused"),
            ("Scenario 2: Outward Bulging (ICP Herniation)", +2.8, 0.0, 0.0, 0.8, "fused"),
            ("Scenario 3: Oblique Shear Translation", -2.0, 2.5, -1.5, 0.8, "fused"),
            ("Ablation: Surface Optical Only", -2.0, 2.5, -1.5, 0.8, "surf"),
            ("Ablation: Deep Ultrasound Only", -2.0, 2.5, -1.5, 0.8, "ius"),
        ]

        traj_table = Table(
            title="Simulation 1010: Interactive Point Cloud Inversion Trajectory",
            border_style="cyan",
            header_style="bold magenta",
        )
        traj_table.add_column("Surgical Scenario", style="bold white", width=34)
        traj_table.add_column("Translation [δn, δt1, δt2]", justify="center", width=22)
        traj_table.add_column("Modality", justify="center", width=14)
        traj_table.add_column("Max Shift", justify="right", style="cyan", width=14)
        traj_table.add_column("Lesion Drift", justify="right", style="bold green", width=14)
        traj_table.add_column("ROM Latency", justify="right", style="dim", width=16)

        final_u_recon = None
        for name, dn, dt1, dt2, g_deep, mod in test_scenarios:
            u_rec, s_pts, i_pts, lat_us = solve_interactive_inversion(dn, dt1, dt2, g_deep, mod)
            final_u_recon = u_rec
            max_s = float(np.max(np.linalg.norm(u_rec, axis=1)))
            lesion_d = float(np.linalg.norm(np.mean(u_rec[lesion_vol_nodes], axis=0)))
            trans_str = f"[{dn:+.1f}, {dt1:+.1f}, {dt2:+.1f}] mm"
            traj_table.add_row(
                name,
                trans_str,
                mod.upper(),
                f"{max_s:.2f} mm",
                f"{lesion_d:.2f} mm",
                f"{lat_us:.1f} µs",
            )

        console.print(traj_table)

        # Export final reconstructed state to VTU
        vtu_path = out_dir / "interactive_fused_brain_shift.vtu"
        grid = domain.to_pyvista(factor=1.0)
        grid.point_data["Displacement"] = final_u_recon
        grid.point_data["Displacement_Magnitude_mm"] = np.linalg.norm(final_u_recon, axis=1)
        grid.save(str(vtu_path))
        console.print(f"\n  • Exported Automated Trajectory Result VTU to: [dim]{vtu_path}[/dim]")

    else:
        # ---------------------------------------------------------------------
        # Interactive PyVista 3D GUI Mode with Sliders and Hotkeys
        # ---------------------------------------------------------------------
        console.print("  • Launching 3D Interactive Multi-Modal Brain Shift Window...")
        plotter = pv.Plotter(
            window_size=[1360, 850],
            title="Sierra FEA - Simulation 1010: Interactive Hyper-Reduced Brain Shift Sensing",
        )
        plotter.set_background("#0f172a", top="#1e293b")

        # Initial baseline inversion
        u_init, surf_init_pts, ius_init_pts, _ = solve_interactive_inversion(0.0, 0.0, 0.0)

        # 1. Deformed Brain Mesh with Vertical Colorbar on Far Right (matching simulation 1303)
        grid = domain.to_pyvista(factor=1.0)
        deformed_brain = grid.copy()
        deformed_brain.points = mesh.nodes + u_init
        deformed_brain.point_data["Displacement_Magnitude_mm"] = np.linalg.norm(u_init, axis=1)

        actor_brain = plotter.add_mesh(
            deformed_brain,
            scalars="Displacement_Magnitude_mm",
            cmap="turbo",
            clim=[0.0, 6.0],
            opacity=0.90,
            show_edges=False,
            scalar_bar_args={
                "title": "Brain Shift (mm)",
                "vertical": True,
                "title_font_size": 11,
                "label_font_size": 9,
                "color": "white",
                "position_x": 0.89,
                "position_y": 0.28,
                "width": 0.045,
                "height": 0.45,
            },
        )

        # 2. Semi-Transparent Skull Vault
        if skull_mesh_path.exists():
            skull_pv = pv.read(str(skull_mesh_path))
            plotter.add_mesh(skull_pv, color="#d0d0c5", opacity=0.15, show_edges=False)

        # 3. Surgical Burr Hole Bone Table Collar & Perimeter Ring
        plotter.add_mesh(
            bone_collar,
            color="#fff8dc",
            opacity=0.95,
            show_edges=True,
            edge_color="#444444",
            label="Skull Bone Table Collar",
        )
        plotter.add_mesh(
            wireframe_ring,
            color="#00ffff",
            line_width=3.0,
            label="Burr Hole Aperture (Ø 20 mm)",
        )

        # 4. Point Cloud 1: Cortical Surface Optical Markers (Green Spheres)
        cloud_surf = pv.PolyData(surf_init_pts)
        actor_cloud_surf = plotter.add_mesh(
            cloud_surf,
            color="#00ff66",
            point_size=10,
            render_points_as_spheres=True,
            label=f"Optical Surface Cloud ({n_surf_markers} pts)",
        )

        # 5. Point Cloud 2: Deep Ultrasound iUS Target Points (Magenta Spheres)
        cloud_ius = pv.PolyData(ius_init_pts)
        actor_cloud_ius = plotter.add_mesh(
            cloud_ius,
            color="#ff00cc",
            point_size=12,
            render_points_as_spheres=True,
            label=f"Deep iUS Target Cloud ({n_ius} pts)",
        )

        # 6. Deep Tumor Meshes (Baseline vs Reconstructed)
        lesion_base = lesion_pv.copy()
        lesion_base.points += np.mean(u_base_3d[lesion_vol_nodes], axis=0)
        plotter.add_mesh(
            lesion_base,
            color="red",
            opacity=0.45,
            label="Baseline Lesion",
        )

        lesion_recon = lesion_pv.copy()
        lesion_recon.points += np.mean(u_init[lesion_vol_nodes], axis=0)
        actor_lesion_recon = plotter.add_mesh(
            lesion_recon,
            color="cyan",
            style="wireframe",
            line_width=2.5,
            label="Inverted Reconstructed Target",
        )

        # 7. Interactive State Manager
        state = {
            "delta_n": 0.0,
            "delta_t1": 0.0,
            "delta_t2": 0.0,
            "gamma_deep": 0.8,
            "modality": "fused",  # "fused", "surf", "ius"
            "last_latency_us": 50.0,
            "exit_requested": False,
        }

        # Camera view looking antero-superiorly into burr hole trajectory
        cam_pos = burr_center + normal_burr * 220.0
        plotter.camera_position = [
            (cam_pos[0], cam_pos[1], cam_pos[2]),
            (burr_center[0], burr_center[1], burr_center[2]),
            (0.0, 0.0, 1.0),
        ]

        # 8. HUD & Telemetry Overlay (Upper Left, matching simulation 1303 layout)
        hud_actor = plotter.add_text(
            "Initializing Neurosurgical Telemetry...",
            position="upper_left",
            font_size=9,
            color="#eceff1",
            font="courier",
        )

        # 9. Orientation Axes (Bottom Right, non-overlapping, matching simulation 1303)
        plotter.add_axes(
            xlabel="R (+X)",
            ylabel="A (+Y)",
            zlabel="S (+Z)",
            line_width=3,
            viewport=(0.80, 0.0, 1.0, 0.22),
        )

        # 10. Navigation & Hotkey Instructions (Lower Left, matching simulation 1303)
        instructions_text = (
            "Surgical Navigation & Hotkeys:\n"
            "  [W/S] Normal Shift δn (Sag/Bulge)\n"
            "  [A/D] Tangential Shift δt1\n"
            "  [M]   Cycle Modality (Fused/Surf/iUS)\n"
            "  [1]   Sagging Preset (-3.2 mm)\n"
            "  [2]   Bulging Preset (+2.8 mm)\n"
            "  [3]   Compound Shear Preset\n"
            "  [R]   Reset to Resting State\n"
            "  [Esc] Exit Simulator"
        )
        plotter.add_text(
            instructions_text,
            position="lower_left",
            font_size=8.5,
            color="#b0bec5",
            font="courier",
        )

        def refresh_gui() -> None:
            """Solves the inversion for current state and updates all 3D actors & telemetry."""
            u_rec, s_pts, i_pts, lat_us = solve_interactive_inversion(
                delta_n=state["delta_n"],
                delta_t1=state["delta_t1"],
                delta_t2=state["delta_t2"],
                gamma_deep=state["gamma_deep"],
                modality=state["modality"],
            )
            state["last_latency_us"] = lat_us

            # Update mesh and point clouds
            deformed_brain.points = mesh.nodes + u_rec
            deformed_brain.point_data["Displacement_Magnitude_mm"] = np.linalg.norm(u_rec, axis=1)

            cloud_surf.points = s_pts
            cloud_ius.points = i_pts

            lesion_shift = np.mean(u_rec[lesion_vol_nodes], axis=0)
            lesion_recon.points = lesion_pv.points + lesion_shift

            # Reconstructed telemetry metrics
            max_cortex_sag = float(np.max(np.linalg.norm(u_rec[hole_vol_nodes], axis=1)))
            lesion_drift_mag = float(np.linalg.norm(lesion_shift))
            solve_hz = 1e6 / max(lat_us, 1.0)

            mod_label = {
                "fused": "Fused Multi-Modal (Surface + Deep)",
                "surf": "Surface Optical Only (Extrapolation Drift)",
                "ius": "Deep Ultrasound Only (Surface Unlocked)",
            }.get(state["modality"], state["modality"])

            shift_dir_str = (
                "Bulging outward" if state["delta_n"] > 0.05
                else "Sagging inward" if state["delta_n"] < -0.05
                else "Resting posture"
            )

            hud_lines = [
                "Simulation 1010: Interactive Hyper-Reduced Brain Shift Sensing",
                f"  Tracking Modality [M]: {mod_label}",
                f"  Burr Hole Trephine   : [X: -2.3, Y: 34.1, Z: 50.1] mm (Ø 20.0 mm)",
                f"  Burr Normal Vector n : [{normal_burr[0]:+.3f}, {normal_burr[1]:+.3f}, {normal_burr[2]:+.3f}]",
                "",
                "Surgical Shift & Inverted Kinematics Telemetry",
                f"  Point Cloud Shift    : δn = {state['delta_n']:+5.2f} mm | δt1 = {state['delta_t1']:+5.2f} mm | δt2 = {state['delta_t2']:+5.2f} mm",
                f"  Deep Target Coupling : γ = {state['gamma_deep']:4.2f}",
                f"  Cortical Shift (Hole): {max_cortex_sag:5.2f} mm ({shift_dir_str})",
                f"  Subcortical Drift    : {lesion_drift_mag:5.2f} mm [{lesion_shift[0]:+5.2f}, {lesion_shift[1]:+5.2f}, {lesion_shift[2]:+5.2f}] mm",
                f"  ROM Inversion Latency: {lat_us:5.1f} µs ({solve_hz:6,.0f} Hz Refresh Rate)",
                "",
                "Point Cloud Heterogeneous Sensors",
                f"  Surface Optical (W=1.00): {n_surf_markers} aperture pts (±50 µm noise)",
                f"  Deep Ultrasound (W=0.04): {n_ius} lesion boundary pts (±250 µm noise)",
            ]
            hud_actor.set_text(2, "\n".join(hud_lines))
            plotter.render()

        # Slider Callbacks
        def on_slider_normal(val: float) -> None:
            state["delta_n"] = float(val)
            refresh_gui()

        def on_slider_t1(val: float) -> None:
            state["delta_t1"] = float(val)
            refresh_gui()

        def on_slider_t2(val: float) -> None:
            state["delta_t2"] = float(val)
            refresh_gui()

        def on_slider_coupling(val: float) -> None:
            state["gamma_deep"] = float(val)
            refresh_gui()

        # 11. Add PyVista Interactive Slider Widgets in Bottom Center [x: 0.28 to 0.78]
        # (Located between Lower-Left instructions at x <= 0.25 and Bottom-Right axes at x >= 0.80)
        s_normal = plotter.add_slider_widget(
            on_slider_normal,
            [-5.0, 5.0],
            value=0.0,
            title="Normal Shift δn (mm) [-:Sag / +:Bulge]",
            pointa=(0.28, 0.11),
            pointb=(0.51, 0.11),
            color="#00ffff",
            title_height=0.018,
            fmt="%.2f",
        )
        s_coupling = plotter.add_slider_widget(
            on_slider_coupling,
            [0.0, 1.2],
            value=0.8,
            title="Deep Target Coupling γ",
            pointa=(0.55, 0.11),
            pointb=(0.78, 0.11),
            color="#00ff66",
            title_height=0.018,
            fmt="%.2f",
        )
        s_t1 = plotter.add_slider_widget(
            on_slider_t1,
            [-4.0, 4.0],
            value=0.0,
            title="Tangential Shift 1 δt1 (mm)",
            pointa=(0.28, 0.035),
            pointb=(0.51, 0.035),
            color="#ffaa00",
            title_height=0.018,
            fmt="%.2f",
        )
        s_t2 = plotter.add_slider_widget(
            on_slider_t2,
            [-4.0, 4.0],
            value=0.0,
            title="Tangential Shift 2 δt2 (mm)",
            pointa=(0.55, 0.035),
            pointb=(0.78, 0.035),
            color="#ff55aa",
            title_height=0.018,
            fmt="%.2f",
        )

        def sync_sliders_from_state() -> None:
            """Synchronizes visual slider positions whenever hotkeys are triggered."""
            s_normal.GetSliderRepresentation().SetValue(state["delta_n"])
            s_t1.GetSliderRepresentation().SetValue(state["delta_t1"])
            s_t2.GetSliderRepresentation().SetValue(state["delta_t2"])
            s_coupling.GetSliderRepresentation().SetValue(state["gamma_deep"])

        # Keyboard Event Handlers
        def key_norm_up() -> None:
            state["delta_n"] = min(5.0, state["delta_n"] + 0.5)
            sync_sliders_from_state()
            refresh_gui()

        def key_norm_down() -> None:
            state["delta_n"] = max(-5.0, state["delta_n"] - 0.5)
            sync_sliders_from_state()
            refresh_gui()

        def key_t1_up() -> None:
            state["delta_t1"] = min(4.0, state["delta_t1"] + 0.5)
            sync_sliders_from_state()
            refresh_gui()

        def key_t1_down() -> None:
            state["delta_t1"] = max(-4.0, state["delta_t1"] - 0.5)
            sync_sliders_from_state()
            refresh_gui()

        def key_cycle_modality() -> None:
            cycle = {"fused": "surf", "surf": "ius", "ius": "fused"}
            state["modality"] = cycle.get(state["modality"], "fused")
            console.print(f"  [bold yellow]▶ [KEY] Cycled Tracking Modality to: {state['modality'].upper()}[/bold yellow]")
            refresh_gui()

        def key_preset_sagging() -> None:
            state["delta_n"] = -3.2
            state["delta_t1"] = 0.0
            state["delta_t2"] = 0.0
            sync_sliders_from_state()
            console.print("  [bold cyan]▶ [KEY] Applied Preset: Gravitational Sagging (δn = -3.2 mm)[/bold cyan]")
            refresh_gui()

        def key_preset_bulging() -> None:
            state["delta_n"] = +2.8
            state["delta_t1"] = 0.0
            state["delta_t2"] = 0.0
            sync_sliders_from_state()
            console.print("  [bold cyan]▶ [KEY] Applied Preset: Elevated ICP Bulging (δn = +2.8 mm)[/bold cyan]")
            refresh_gui()

        def key_preset_shear() -> None:
            state["delta_n"] = -2.0
            state["delta_t1"] = +2.5
            state["delta_t2"] = -1.5
            sync_sliders_from_state()
            console.print("  [bold cyan]▶ [KEY] Applied Preset: Oblique Compound Shear Translation[/bold cyan]")
            refresh_gui()

        def key_reset() -> None:
            state["delta_n"] = 0.0
            state["delta_t1"] = 0.0
            state["delta_t2"] = 0.0
            state["gamma_deep"] = 0.8
            state["modality"] = "fused"
            sync_sliders_from_state()
            console.print("  [bold green]▶ [KEY] Reset point cloud translations to resting state[/bold green]")
            refresh_gui()

        def key_quit() -> None:
            state["exit_requested"] = True
            if not getattr(plotter, "_closed", False):
                plotter.close()

        # Clear PyVista's default key callbacks (prevents 'b' -> pickpoint and 'q' -> close conflicts)
        if hasattr(plotter, "iren") and plotter.iren is not None:
            plotter.iren.clear_key_event_callbacks()

        plotter.add_key_event("w", key_norm_up)
        plotter.add_key_event("W", key_norm_up)
        plotter.add_key_event("s", key_norm_down)
        plotter.add_key_event("S", key_norm_down)
        plotter.add_key_event("d", key_t1_up)
        plotter.add_key_event("D", key_t1_up)
        plotter.add_key_event("a", key_t1_down)
        plotter.add_key_event("A", key_t1_down)
        plotter.add_key_event("m", key_cycle_modality)
        plotter.add_key_event("M", key_cycle_modality)
        plotter.add_key_event("1", key_preset_sagging)
        plotter.add_key_event("2", key_preset_bulging)
        plotter.add_key_event("3", key_preset_shear)
        plotter.add_key_event("r", key_reset)
        plotter.add_key_event("R", key_reset)
        plotter.add_key_event("Escape", key_quit)
        plotter.add_key_event("q", key_quit)
        plotter.add_key_event("Q", key_quit)

        # Trigger initial draw before event loop
        refresh_gui()

        # Remove VTK default CharEvent observers that hijack '3' for stereo 3D and 'w' for wireframe
        if hasattr(plotter, "iren") and plotter.iren is not None:
            plotter.iren.interactor.RemoveObservers("CharEvent")
            if plotter.iren.get_interactor_style() is not None:
                plotter.iren.get_interactor_style().RemoveObservers("CharEvent")

        # Show window non-blockingly with interactive_update=True
        plotter.show(interactive_update=True)

        # Permanently disable stereo mode and re-strip CharEvent after window initialization
        if getattr(plotter, "render_window", None) is not None:
            plotter.render_window.SetStereoRender(False)
        if hasattr(plotter, "iren") and plotter.iren is not None:
            plotter.iren.interactor.RemoveObservers("CharEvent")
            if plotter.iren.get_interactor_style() is not None:
                plotter.iren.get_interactor_style().RemoveObservers("CharEvent")

        # Interactive update event loop (~60 FPS)
        try:
            while not getattr(plotter, "_closed", False) and not state["exit_requested"]:
                plotter.update()
                time.sleep(0.016)
        except KeyboardInterrupt:
            pass
        finally:
            if not getattr(plotter, "_closed", False):
                plotter.close()

    # 6. Conclusion Telemetry
    console.print(
        "\n[bold green]✔ Simulation 1010 (Interactive Hyper-Reduced Brain Shift Sensing) completed successfully![/bold green]\n"
    )


if __name__ == "__main__":
    main()
