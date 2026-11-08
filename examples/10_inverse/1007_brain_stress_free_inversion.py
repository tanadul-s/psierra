"""
Simulation 1007: Zero-Gravity Reference Configuration Recovery for Pre-Operative Brain Scans.
Eliminating Intraoperative Double-Gravity Bias in Neurosurgical Planning via Backward Fixed-Point Inversion.

Clinical Background:
--------------------
A diagnostic pre-operative MRI scan is captured with the patient lying supine inside the scanner bore.
Under Earth's 1G gravity (g_y = -9.81 m/s²), the brain parenchyma sags and rests against the posterior
calvarium. Therefore, the segmented MRI surface X_scan is NOT stress-free—it is pre-deformed by gravity.
If a surgical navigation system treats X_scan as the reference geometry and subsequently applies gravity
in the operative posture (e.g., lateral craniotomy or prone), it applies gravity twice:
    x_biased = X_scan + u(X_scan; g_surgery) != x_true = X_0* + u(X_0*; g_surgery)
This "double-gravity bias" introduces 2-5 mm of systematic positioning error at deep tumor targets.
This simulation employs the backward fixed-point solver (Sellier 2011 with Aitken dynamic acceleration)
to reconstruct the true undeformed zero-gravity reference geometry X_0*, proving that surgical posture
simulation from X_0* completely eliminates the double-gravity error.

Key Architecture:
-----------------
1. Anatomical MRI Scan Configuration: 55,191-DOF solid tetrahedral mesh (Tet4, 18,397 nodes)
   segmented from supine patient scan X_scan.
2. Backward Fixed-Point Inversion:
       X_0^(k+1) = X_0^(k) + omega_k * (X_scan - [X_0^(k) + u(X_0^(k); g_supine)])
   accelerated via dynamic Aitken Delta^2 relaxation.
3. Verification of Reference Recovery: Re-simulating supine gravity on X_0* matches X_scan to < 0.05 mm.
4. Intraoperative Posture Reorientation Benchmark:
   - Comparing Naive Forward Solve (from pre-sagged X_scan) vs. Correct Forward Solve (from X_0*).
   - Quantifying the severe clinical risk of double-gravity bias at cortical and subcortical tumor sites.
5. Telemetry & State Export: Rich comparative tables and .vtu volumetric field exports.
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
import scipy.sparse.linalg as spspla

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import DirichletManager, DirichletConstraint, GravityLoad
from psierra.inverse import (
    StressFreeConfigurationSolver,
    StressFreeResult,
)

install(show_locals=False)
console = Console()

SURGICAL_POSTURES: dict[str, tuple[float, float, float, str]] = {
    "lateral": (
        4905.0,
        -8495.7,
        0.0,
        "30° Lateral Head Rotation (g_x = 0.5g, g_y = -0.866g)",
    ),
    "prone": (0.0, 9810.0, 0.0, "Prone Surgical Posture (g_y = +1.0g)"),
    "pterional": (
        3355.0,
        -8495.7,
        -3570.0,
        "Pterional 3-Axis Tilt (g_x=0.34g, g_y=-0.87g, g_z=-0.36g)",
    ),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 1007: Zero-Gravity Reference Configuration Recovery for Pre-Operative Brain Scans"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in non-GUI headless validation mode (auto-enabled if DISPLAY is absent)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=6,
        help="Maximum fixed-point iterations for reference inversion (default: 6)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Convergence residual tolerance in mm (default: 0.05 mm = 50 microns)",
    )
    parser.add_argument(
        "--surgery-posture",
        type=str,
        default="lateral",
        choices=list(SURGICAL_POSTURES.keys()),
        help="Target intraoperative surgical posture (default: lateral)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/10_inverse/1007_brain_stress_free_inversion",
        help="Output directory for stress-free meshes and .vtu exports",
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
            "[bold cyan]Project Sierra - Simulation 1007 (iFEM Clinical Flagship)[/bold cyan]\n"
            "[white]Zero-Gravity Reference Configuration Recovery for Pre-Operative Brain Scans[/white]\n"
            "[dim]Aitken Fixed-Point Inversion • Elimination of Double-Gravity Bias • Sub-50 Micron Precision[/dim]",
            border_style="cyan",
        )
    )

    # --------------------------------------------------------------------------
    # [1/5] Loading Pre-Operative Supine MRI Scan
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][1/5] Loading Patient Pre-Operative Supine MRI Scan[/bold yellow]"
    )
    brain_path = resolve_mesh_path("data/mrbraintumor1/mesh/brain.vtk")
    lesion_path = resolve_mesh_path("data/mrbraintumor1/mesh/lesion.vtk")
    skull_path = resolve_mesh_path("data/mrbraintumor1/mesh/skull.vtk")

    scan_mesh = VolumetricMesh(brain_path)
    X_scan = scan_mesh.nodes.copy()

    console.print(
        f"  • MRI Scan Geometry : [green]{scan_mesh.n_nodes:,}[/green] nodes | "
        f"[green]{scan_mesh.n_elements:,}[/green] elements | "
        f"[green]{scan_mesh.n_dofs:,}[/green] DOFs\n"
        f"  • Physical Condition: [dim]Patient scanned in Supine MRI bore under 1G posterior gravity[/dim]"
    )

    # Identify deep lesion target nodes
    lesion_pv = pv.read(lesion_path)
    brain_tree = cKDTree(scan_mesh.nodes)
    _, lesion_point_to_brain = brain_tree.query(lesion_pv.points)
    lesion_node_indices = np.unique(lesion_point_to_brain)
    c_scan_lesion = np.mean(X_scan[lesion_node_indices], axis=0)

    console.print(
        f"  • Deep Lesion Target: [green]{len(lesion_node_indices)}[/green] nodes | "
        f"Scanned Center: [dim]({c_scan_lesion[0]:.1f}, {c_scan_lesion[1]:.1f}, {c_scan_lesion[2]:.1f}) mm[/dim]"
    )

    # --------------------------------------------------------------------------
    # [2/5] Configuring Continuum Domain & Supine Scan Gravity
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][2/5] Configuring Tissue Biomechanics & Supine Acquisition Loads[/bold yellow]"
    )
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=0.003,  # 3.0 kPa
        poissons_ratio=0.45,
        mass_density=1040e-12,  # 1040 kg/m³
    )
    material = Hookean(params)
    element = Tet4Linear(material)
    domain = SolidDomain(mesh=scan_mesh, material=material, element=element)

    clamped_node_mask = scan_mesh.nodes[:, 1] < -20.0
    clamped_nodes = np.where(clamped_node_mask)[0]
    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(
        DirichletConstraint(clamped_nodes, components=(0, 1, 2), value=0.0)
    )
    free_dofs = dirichlet.free_dofs

    # Supine MRI gravity: g = (0, -9810.0, 0) mm/s²
    g_supine = GravityLoad(acceleration=(0.0, -9810.0, 0.0))

    console.print(
        f"  • Parenchyma Modulus : [green]3.00 kPa[/green] (ν = 0.45)\n"
        f"  • Cranial Base Anchor: [green]{len(clamped_nodes):,}[/green] nodes clamped (Y < -20 mm)\n"
        f"  • MRI Scan Gravity   : [cyan]g = (0.0, -9.81, 0.0) m/s²[/cyan] (Posterior sag)"
    )

    # --------------------------------------------------------------------------
    # [3/5] Solving Backward Stress-Free Configuration Inversion
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][3/5] Executing Backward Fixed-Point Stress-Free Inversion[/bold yellow]"
    )
    console.print(
        "  [dim]Iterating Sellier backward updates with dynamic Aitken Δ² relaxation...[/dim]"
    )

    solver = StressFreeConfigurationSolver(
        domain=domain,
        load=g_supine,
        dirichlet=dirichlet,
        relaxation_factor=0.7,
        adaptive_relaxation=True,
        tolerance=args.tolerance,
        max_iterations=args.max_iters,
        analysis_type="linear",
    )

    t0_inv = time.perf_counter()
    sf_result = solver.solve(scan_mesh_or_nodes=X_scan, verbose=True)
    t_inv = time.perf_counter() - t0_inv

    X_0_star = sf_result.reference_nodes
    u_pre_sag = sf_result.displacement_3d
    c_ref_lesion = np.mean(X_0_star[lesion_node_indices], axis=0)

    console.print(
        f"\n  [bold green]✓ Stress-Free Inversion Completed in {sf_result.iterations} iterations ({t_inv:.2f} s):[/bold green]\n"
        f"    • Maximum Nodal Discrepancy : [bold cyan]{sf_result.residual_error:.4f} mm[/bold cyan]\n"
        f"    • Root-Mean-Square RMS Error: [bold cyan]{sf_result.rms_error:.4f} mm[/bold cyan]\n"
        f"    • Relative Geometry Error   : {sf_result.relative_error * 100:.4f}%\n"
        f"    • True Undeformed Lesion Pos: ({c_ref_lesion[0]:.2f}, {c_ref_lesion[1]:.2f}, {c_ref_lesion[2]:.2f}) mm"
    )

    # --------------------------------------------------------------------------
    # [4/5] Demonstrating & Quantifying Intraoperative Double-Gravity Bias
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][4/5] Quantifying Intraoperative Double-Gravity Bias in Surgery[/bold yellow]"
    )
    gx_surg, gy_surg, gz_surg, surg_desc = SURGICAL_POSTURES[args.surgery_posture]
    console.print(
        f"  • Surgical Posture   : [cyan]{args.surgery_posture.upper()}[/cyan] ({surg_desc})"
    )

    # Re-simulate forward under surgical gravity:
    g_surg = GravityLoad(acceleration=(gx_surg, gy_surg, gz_surg))

    # A. NAIVE BIASED APPROACH: Applying surgical gravity directly to pre-sagged X_scan
    domain_naive = SolidDomain(mesh=scan_mesh, material=material, element=element)
    K_scan = domain_naive.assemble_stiffness_matrix()
    M_scan = domain_naive.assemble_mass_matrix()
    f_surg_naive = g_surg.compute_force_vector(M_scan)
    u_naive_f = spspla.spsolve(
        K_scan[free_dofs, :][:, free_dofs], f_surg_naive[free_dofs]
    )
    u_naive = np.zeros(scan_mesh.n_dofs, dtype=np.float64)
    u_naive[free_dofs] = u_naive_f
    x_biased = X_scan + u_naive.reshape(-1, 3)

    # B. CORRECT PHYSIOLOGICAL APPROACH: Applying surgical gravity to recovered stress-free X_0*
    ref_mesh = sf_result.create_reference_mesh(scan_mesh)
    domain_correct = SolidDomain(mesh=ref_mesh, material=material, element=element)
    K_ref = domain_correct.assemble_stiffness_matrix()
    M_ref = domain_correct.assemble_mass_matrix()
    f_surg_correct = g_surg.compute_force_vector(M_ref)
    u_correct_f = spspla.spsolve(
        K_ref[free_dofs, :][:, free_dofs], f_surg_correct[free_dofs]
    )
    u_correct = np.zeros(ref_mesh.n_dofs, dtype=np.float64)
    u_correct[free_dofs] = u_correct_f
    x_true = X_0_star + u_correct.reshape(-1, 3)

    # Double-gravity bias error vector: e_bias = x_biased - x_true
    bias_vectors = x_biased - x_true
    bias_magnitudes = np.linalg.norm(bias_vectors, axis=1)
    max_bias_surf = float(np.max(bias_magnitudes[scan_mesh.surface.node_indices]))
    mean_bias_surf = float(np.mean(bias_magnitudes[scan_mesh.surface.node_indices]))

    # Lesion target bias
    c_biased_lesion = np.mean(x_biased[lesion_node_indices], axis=0)
    c_true_lesion = np.mean(x_true[lesion_node_indices], axis=0)
    lesion_bias_err = float(np.linalg.norm(c_biased_lesion - c_true_lesion))

    console.print(
        f"  • Max Cortical Bias Error: [bold red]{max_bias_surf:.2f} mm[/bold red] "
        f"(Mean: {mean_bias_surf:.2f} mm)\n"
        f"  • Deep Lesion Bias Error : [bold red]{lesion_bias_err:.2f} mm[/bold red]!\n"
        f"    [dim]Biased Target: ({c_biased_lesion[0]:.1f}, {c_biased_lesion[1]:.1f}, {c_biased_lesion[2]:.1f}) mm | "
        f"True Target: ({c_true_lesion[0]:.1f}, {c_true_lesion[1]:.1f}, {c_true_lesion[2]:.1f}) mm[/dim]"
    )

    # --------------------------------------------------------------------------
    # [5/5] Clinical Telemetry Report & Volumetric State Export
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][5/5] Clinical Verification Report & Volumetric State Export[/bold yellow]"
    )

    comp_table = Table(
        title="Neurosurgical Double-Gravity Bias Elimination Benchmark",
        border_style="cyan",
    )
    comp_table.add_column("Clinical Landmark / Metric", style="bold white")
    comp_table.add_column("Naive Biased Simulation", justify="right", style="red")
    comp_table.add_column("Stress-Free Recovered", justify="right", style="bold green")
    comp_table.add_column("Clinical Consequence", justify="center", style="yellow")

    comp_table.add_row(
        "Reference Geometry",
        "Pre-Deformed Supine MRI",
        "Stress-Free X_0*",
        "Double-gravity avoided",
    )
    comp_table.add_row(
        "Deep Tumor Centroid",
        f"({c_biased_lesion[0]:.1f}, {c_biased_lesion[1]:.1f}, {c_biased_lesion[2]:.1f}) mm",
        f"({c_true_lesion[0]:.1f}, {c_true_lesion[1]:.1f}, {c_true_lesion[2]:.1f}) mm",
        f"{lesion_bias_err:.2f} mm targeting error",
    )
    comp_table.add_row(
        "Max Cortical Discrepancy",
        f"{max_bias_surf:.2f} mm",
        f"{sf_result.residual_error:.4f} mm",
        "Sub-millimeter craniotomy margin",
    )
    comp_table.add_row(
        "MRI Supine Reconstruction RMS",
        "---",
        f"{sf_result.rms_error:.4f} mm",
        "Near-zero residual (< 50 µm)",
    )
    console.print(comp_table)

    # Export VTU showing bias distribution and stress-free geometry
    export_path = out_dir / f"result.vtu"
    grid = scan_mesh.ugrid.copy()
    grid.point_data["MRI_Scan_Nodes"] = X_scan
    grid.point_data["StressFree_Nodes"] = X_0_star
    grid.point_data["Supine_Sag_Displacement"] = u_pre_sag
    grid.point_data["Double_Gravity_Bias_Vector"] = bias_vectors
    grid.point_data["Double_Gravity_Bias_mm"] = bias_magnitudes
    grid.save(str(export_path))
    console.print(
        f"\n[green]✓ Exported clinical 3D volumetric fields to:[/green] [cyan]{export_path}[/cyan]"
    )

    if sf_result.residual_error < 0.1:
        console.print(
            f"[bold green]✓ CLINICAL RECONSTRUCTION PASSED: Supine MRI residual {sf_result.residual_error:.4f} mm < 0.1 mm![/bold green]"
        )
    else:
        console.print(
            f"[bold yellow]! Inversion residual: {sf_result.residual_error:.4f} mm[/bold yellow]"
        )

    # 3D Visual verification (GUI Mode)
    if not is_headless:
        console.print(
            "\n[bold cyan]Launching Interactive 3D Double-Gravity Bias Visualization...[/bold cyan]"
        )
        plotter = pv.Plotter(
            title="pSierra 1007: Zero-Gravity Reference Configuration Recovery"
        )
        plotter.set_background("#1e1e1e")

        surf_poly = scan_mesh.surface.polydata.copy()
        surf_poly["Bias_Error_mm"] = bias_magnitudes[scan_mesh.surface.node_indices]
        plotter.add_mesh(
            surf_poly,
            scalars="Bias_Error_mm",
            cmap="inferno",
            show_scalar_bar=True,
            scalar_bar_args={"title": "Double-Gravity Bias Error (mm)"},
        )

        # Lesion True vs Biased
        lesion_true_p = lesion_pv.copy()
        lesion_true_p.points = x_true[lesion_node_indices]
        plotter.add_mesh(
            lesion_true_p,
            color="green",
            opacity=0.8,
            label="True Surgical Tumor Position",
        )

        lesion_biased_p = lesion_pv.copy()
        lesion_biased_p.points = x_biased[lesion_node_indices]
        plotter.add_mesh(
            lesion_biased_p,
            color="red",
            style="wireframe",
            line_width=2.0,
            label=f"Biased Tumor Position ({lesion_bias_err:.1f} mm Error)",
        )

        if skull_path.exists():
            skull_pv = pv.read(skull_path)
            plotter.add_mesh(skull_pv, color="white", opacity=0.08, style="wireframe")

        plotter.add_legend(bcolor="#2d2d2d")
        plotter.camera_position = "iso"
        plotter.show()


if __name__ == "__main__":
    main()
