"""
Simulation 1006: Patient-Specific Brain Parenchyma Stiffness Calibration via Intraoperative Indentation.
Inverse Parameter Identification of Young's Modulus E from Surgical Indenter Force-Deflection Curves.

Clinical Background:
--------------------
Brain tissue stiffness exhibits marked inter-patient variability (ranging from 1.0 to 10.0 kPa)
depending on pathology (high-grade glioma, peritumoral vasogenic edema, hydrocephalus, or age).
Prior population-average material properties lead to erroneous finite element brain-shift predictions.
During craniotomy, a surgical robotic tool or handheld tactile probe performs gentle indentation
(depth delta in [0.5, 2.5] mm, force F in [0.05, 0.25] N) on the exposed cortical surface.
This simulation calibrates the patient-specific Young's modulus E in real time from force-deflection
observations, recovering the true stiffness from a heavily miscalibrated initial estimate (E_0 = 10.0 kPa vs 3.0 kPa).

Key Architecture:
-----------------
1. Patient-Specific Brain Geometry: 55,191-DOF solid tetrahedral model (Tet4, 18,397 nodes)
   with clamped skull base and posterior fossa anchors.
2. Clinical Indentation Protocol: Vertical compression probe applied to an exposed cortical
   fronto-parietal gyral crown across 5 incremental load stages.
3. Dual Inversion Solvers:
   - Closed-Form Linear Least-Squares: Instantaneous (< 1 microsecond) exact analytical recovery.
   - Non-Linear Trust-Region Reflective (TRF): Bound-constrained numerical optimization
     with parameter convergence history tracking and residual reduction.
4. Robustness & Noise Sensitivity: Evaluating parameter recovery under synthetic force measurement noise.
5. Telemetry Reporting & State Export: Rich comparative tables and .vtu visualization export.
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
import scipy.sparse.linalg as spspla

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import DirichletManager, DirichletConstraint
from psierra.inverse import (
    MaterialParameterIdentifier,
    ParameterBounds,
    IdentificationResult,
)

install(show_locals=False)
console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 1006: Patient-Specific Brain Parenchyma Stiffness Calibration"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in non-GUI headless validation mode (auto-enabled if DISPLAY is absent)",
    )
    parser.add_argument(
        "--true-modulus",
        type=float,
        default=3.0,
        help="True patient brain parenchyma Young's modulus in kPa (default: 3.0 kPa)",
    )
    parser.add_argument(
        "--initial-guess",
        type=float,
        default=10.0,
        help="Initial uncalibrated modulus guess E_0 in kPa (default: 10.0 kPa)",
    )
    parser.add_argument(
        "--poissons-ratio",
        type=float,
        default=0.45,
        help="Poisson's ratio for soft brain tissue (default: 0.45)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Force measurement Gaussian noise standard deviation in N (default: 0.0 N)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/1006_brain_parameter_identification",
        help="Output directory for calibration reports and .vtu exports",
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
            "[bold cyan]Project Sierra - Simulation 1006 (iFEM Clinical Flagship)[/bold cyan]\n"
            "[white]Patient-Specific Brain Stiffness Calibration via Intraoperative Indentation[/white]\n"
            "[dim]Surgical Tactile Probe Inversion • Instant Closed-Form & TRF Optimization • Sub-0.01% Precision[/dim]",
            border_style="cyan",
        )
    )

    # --------------------------------------------------------------------------
    # [1/5] Loading Anatomical Model & Setting Up Indentation Boundary
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][1/5] Loading Anatomical Datasets & Indentation Site[/bold yellow]"
    )
    brain_path = resolve_mesh_path("data/mrbraintumor1/mesh/brain.vtk")
    skull_path = resolve_mesh_path("data/mrbraintumor1/mesh/skull.vtk")
    mesh = VolumetricMesh(brain_path)

    console.print(
        f"  • Volumetric Brain  : [green]{mesh.n_nodes:,}[/green] nodes | "
        f"[green]{mesh.n_elements:,}[/green] Tet4 elements | "
        f"[green]{mesh.n_dofs:,}[/green] DOFs"
    )

    # Clamp skull base & posterior fossa: Y < -20.0 mm
    clamped_node_mask = mesh.nodes[:, 1] < -20.0
    clamped_nodes = np.where(clamped_node_mask)[0]
    dirichlet = DirichletManager(mesh.n_dofs)
    dirichlet.add_constraint(
        DirichletConstraint(clamped_nodes, components=(0, 1, 2), value=0.0)
    )
    free_dofs = dirichlet.free_dofs

    console.print(
        f"  • Cranial Base Anchor: [green]{len(clamped_nodes):,}[/green] fixed nodes | "
        f"[green]{len(free_dofs):,}[/green] Free DOFs"
    )

    # Locate exposed craniotomy cortical indentation site:
    # Fronto-parietal gyral summit near (X ~ 0, Y ~ 30, Z ~ 58 mm)
    surf_nodes = mesh.surface.node_indices
    surf_pts = mesh.nodes[surf_nodes]
    target_site = np.array([0.0, 30.0, 58.0], dtype=np.float64)
    dists = np.linalg.norm(surf_pts - target_site, axis=1)
    indenter_node = surf_nodes[np.argmin(dists)]
    indenter_coord = mesh.nodes[indenter_node]

    console.print(
        f"  • Indentation Site  : Node [cyan]#{indenter_node}[/cyan] at "
        f"({indenter_coord[0]:.1f}, {indenter_coord[1]:.1f}, {indenter_coord[2]:.1f}) mm on cortical crown"
    )

    # --------------------------------------------------------------------------
    # [2/5] Synthesizing Experimental In-Vivo Indentation Data
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][2/5] Simulating Ground Truth Surgical Probe Indentation[/bold yellow]"
    )
    E_true_kpa = args.true_modulus
    E_true_mpa = E_true_kpa * 1e-3  # kPa -> MPa (N/mm²)
    nu = args.poissons_ratio

    params_true = IsotropicElasticParameters()
    params_true.set_from_youngs_poisson(
        youngs_modulus=E_true_mpa,
        poissons_ratio=nu,
        mass_density=1040e-12,
    )
    material_true = Hookean(params_true)
    element_true = Tet4Linear(material_true)
    domain_true = SolidDomain(mesh=mesh, material=material_true, element=element_true)

    # Assemble stiffness matrix at true modulus
    K_true = domain_true.assemble_stiffness_matrix()
    K_true_ff = K_true[free_dofs, :][:, free_dofs]

    # Prescribed indentation depths: delta = [0.5, 1.0, 1.5, 2.0, 2.5] mm in -Z direction
    prescribed_depths = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    # Unit vertical load in -Z direction at indenter node
    indenter_dof_z = 3 * indenter_node + 2
    f_unit = np.zeros(mesh.n_dofs, dtype=np.float64)
    f_unit[indenter_dof_z] = -1.0  # 1.0 N downward force

    f_unit_f = f_unit[free_dofs]
    u_unit_f = spspla.spsolve(K_true_ff, f_unit_f)
    u_unit = np.zeros(mesh.n_dofs, dtype=np.float64)
    u_unit[free_dofs] = u_unit_f

    # Compliance at indenter site (deflection per unit load)
    compliance_true = -u_unit[indenter_dof_z]  # mm / N
    stiffness_true = 1.0 / compliance_true      # N / mm

    # Ground truth reaction forces for the prescribed indentation depths
    F_true = prescribed_depths * stiffness_true

    # Add optional tactile sensor noise
    if args.noise_std > 0.0:
        np.random.seed(42)
        noise = np.random.normal(0.0, args.noise_std, size=F_true.shape)
        F_exp = np.maximum(F_true + noise, 0.001)
        console.print(
            f"  • Sensor Noise      : Gaussian [cyan]sigma = {args.noise_std * 1e3:.1f} mN[/cyan] added"
        )
    else:
        F_exp = F_true
        console.print("  • Sensor Noise      : [green]Ideal tactile sensor (zero noise)[/green]")

    table_data = Table(title="Tactile Indentation Observation Data", border_style="cyan")
    table_data.add_column("Indentation Stage", justify="center")
    table_data.add_column("Prescribed Depth δ (mm)", justify="right")
    table_data.add_column("True Reaction Force F (N)", justify="right", style="green")
    table_data.add_column("Measured Load F_exp (N)", justify="right", style="cyan")

    for idx, (d, ft, fe) in enumerate(zip(prescribed_depths, F_true, F_exp)):
        table_data.add_row(f"Stage {idx + 1}", f"{d:.2f}", f"{ft:.4f}", f"{fe:.4f}")
    console.print(table_data)

    console.print(
        f"  • True Parenchyma E : [bold green]{E_true_kpa:.3f} kPa[/bold green] "
        f"(Tissue Compliance: {compliance_true:.3f} mm/N)"
    )

    # --------------------------------------------------------------------------
    # [3/5] Forward Physics Model with Initial Miscalibrated Prior
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][3/5] Configuring Initial Guess & Forward Simulation Model[/bold yellow]"
    )
    E_0_kpa = args.initial_guess
    E_0_mpa = E_0_kpa * 1e-3

    console.print(
        f"  • Prior Initial E_0 : [bold red]{E_0_kpa:.2f} kPa[/bold red] "
        f"(Miscalibrated by {(E_0_kpa - E_true_kpa) / E_true_kpa * 100:+.1f}%)"
    )

    # Unit compliance kernel: In linear elasticity, deflection delta(E) = delta_1 / E,
    # so reaction force at depth d is F(E) = d * (E / compliance_1)
    # We compute baseline forces with E_0:
    params_0 = IsotropicElasticParameters()
    params_0.set_from_youngs_poisson(
        youngs_modulus=E_0_mpa,
        poissons_ratio=nu,
        mass_density=1040e-12,
    )
    material_0 = Hookean(params_0)
    domain_0 = SolidDomain(mesh=mesh, material=material_0, element=element_true)
    K_0 = domain_0.assemble_stiffness_matrix()
    K_0_ff = K_0[free_dofs, :][:, free_dofs]

    u_0_f = spspla.spsolve(K_0_ff, f_unit_f)
    u_0 = np.zeros(mesh.n_dofs, dtype=np.float64)
    u_0[free_dofs] = u_0_f
    compliance_0 = -u_0[indenter_dof_z]
    stiffness_0 = 1.0 / compliance_0
    F_baseline_0 = prescribed_depths * stiffness_0

    # --------------------------------------------------------------------------
    # [4/5] Executing Inverse Parameter Identification
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][4/5] Executing Inverse Constitutive Calibration[/bold yellow]"
    )

    # Method 1: Instant Closed-Form Linear Least-Squares (< 1 µs)
    t0_cf = time.perf_counter()
    cf_res = MaterialParameterIdentifier.calibrate_linear_elasticity(
        experimental_forces=F_exp,
        baseline_forces=F_baseline_0,
        baseline_modulus=E_0_kpa,
        beta=0.0,
    )
    t_cf = time.perf_counter() - t0_cf
    E_cf_kpa = cf_res.parameters["youngs_modulus"]
    err_cf = abs(E_cf_kpa - E_true_kpa)
    rel_err_cf = (err_cf / E_true_kpa) * 100.0

    console.print(
        f"  [bold green]✓ Method 1 (Closed-Form Analytical LLS):[/bold green]\n"
        f"    • Calibrated E*    : [bold cyan]{E_cf_kpa:.4f} kPa[/bold cyan]\n"
        f"    • Absolute Error   : {err_cf:.4e} kPa ({rel_err_cf:.4f}%)\n"
        f"    • Wall-Clock Time  : {t_cf * 1e6:.1f} µs"
    )

    # Method 2: Trust-Region Reflective (TRF) Iterative Least-Squares
    bounds = [
        ParameterBounds(
            name="youngs_modulus",
            initial_value=E_0_kpa,
            lower_bound=0.1,
            upper_bound=50.0,
            description="Brain Young's modulus (kPa)",
        )
    ]

    # Forward model function mapping E (kPa) -> Simulated indenter forces (N)
    def forward_model(params: dict[str, float]) -> np.ndarray:
        E_eval = params["youngs_modulus"]
        # Scale baseline forces linearly by (E / E_0)
        return (E_eval / E_0_kpa) * F_baseline_0

    identifier = MaterialParameterIdentifier(
        parameters=bounds,
        method="trf",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )

    t0_opt = time.perf_counter()
    trf_res = identifier.identify(
        forward_model=forward_model,
        experimental_forces=F_exp,
    )
    t_opt = time.perf_counter() - t0_opt
    E_trf_kpa = trf_res.parameters["youngs_modulus"]
    err_trf = abs(E_trf_kpa - E_true_kpa)
    rel_err_trf = (err_trf / E_true_kpa) * 100.0

    console.print(
        f"\n  [bold green]✓ Method 2 (Trust-Region Reflective Optimization):[/bold green]\n"
        f"    • Calibrated E*    : [bold cyan]{E_trf_kpa:.4f} kPa[/bold cyan]\n"
        f"    • Absolute Error   : {err_trf:.4e} kPa ({rel_err_trf:.4f}%)\n"
        f"    • Iterations / F-Ev: {trf_res.iterations} iters / {trf_res.n_evaluations} evals\n"
        f"    • Force RMS Error  : {trf_res.rms_error:.4e} N\n"
        f"    • Wall-Clock Time  : {t_opt * 1e3:.2f} ms"
    )

    # --------------------------------------------------------------------------
    # [5/5] Clinical Reporting & Visual Verification
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][5/5] Clinical Telemetry Report & Calibration Summary[/bold yellow]"
    )

    report_table = Table(title="Brain Parenchyma Stiffness Calibration Summary", border_style="cyan")
    report_table.add_column("Parameter / Indicator", style="bold white")
    report_table.add_column("Prior (Uncalibrated)", justify="right", style="red")
    report_table.add_column("Closed-Form LLS", justify="right", style="yellow")
    report_table.add_column("TRF Optimization", justify="right", style="bold green")
    report_table.add_column("Ground Truth", justify="right", style="dim")

    report_table.add_row(
        "Young's Modulus E",
        f"{E_0_kpa:.2f} kPa",
        f"{E_cf_kpa:.4f} kPa",
        f"{E_trf_kpa:.4f} kPa",
        f"{E_true_kpa:.4f} kPa",
    )
    report_table.add_row(
        "Modulus Discrepancy",
        f"{(E_0_kpa - E_true_kpa) / E_true_kpa * 100:+.2f}%",
        f"{rel_err_cf:.4f}%",
        f"{rel_err_trf:.4f}%",
        "0.00%",
    )
    report_table.add_row(
        "Force Residual RMS",
        f"{np.sqrt(np.mean((F_baseline_0 - F_exp)**2)):.4f} N",
        f"{cf_res.rms_error:.4e} N",
        f"{trf_res.rms_error:.4e} N",
        "0.00 N",
    )
    report_table.add_row(
        "Computational Time",
        "---",
        f"{t_cf * 1e6:.1f} µs",
        f"{t_opt * 1e3:.2f} ms",
        "---",
    )
    console.print(report_table)

    # Export VTU showing displacement under calibrated indentation
    export_path = out_dir / "brain_calibrated_indentation.vtu"
    grid = mesh.ugrid.copy()
    # Displacement at calibrated indentation (max depth 2.5 mm)
    grid.point_data["Indentation_Displacement"] = (2.5 / compliance_true) * u_unit.reshape(-1, 3)
    grid.point_data["Displacement_Magnitude"] = np.linalg.norm(
        grid.point_data["Indentation_Displacement"], axis=1
    )
    grid.save(str(export_path))
    console.print(f"\n[green]✓ Exported calibrated 3D indentation field to:[/green] [cyan]{export_path}[/cyan]")

    if rel_err_cf < 0.1:
        console.print(
            f"[bold green]✓ CLINICAL CALIBRATION PASSED: Modulus error {rel_err_cf:.4f}% < 0.1% threshold![/bold green]"
        )
    else:
        console.print(f"[bold yellow]! High calibration error ({rel_err_cf:.4f}%)[/bold yellow]")

    # 3D Visual verification (GUI Mode)
    if not is_headless:
        console.print("\n[bold cyan]Launching Interactive 3D Indentation Visualization...[/bold cyan]")
        plotter = pv.Plotter(title="pSierra 1006: Patient-Specific Brain Stiffness Calibration")
        plotter.set_background("#1e1e1e")

        surf_poly = mesh.surface.polydata.copy()
        surf_disp = (2.5 / compliance_true) * u_unit.reshape(-1, 3)[mesh.surface.node_indices]
        surf_poly.points = mesh.surface.nodes + surf_disp
        surf_poly["Displacement_mm"] = np.linalg.norm(surf_disp, axis=1)

        plotter.add_mesh(
            surf_poly,
            scalars="Displacement_mm",
            cmap="plasma",
            show_scalar_bar=True,
            scalar_bar_args={"title": "Indentation Deflection (mm)"},
        )

        # Highlight indentation probe contact point
        probe_tip = pv.Sphere(radius=2.5, center=indenter_coord + np.array([0, 0, -2.5]))
        plotter.add_mesh(probe_tip, color="cyan", label="Surgical Indenter Probe (2.5 mm)")

        if skull_path.exists():
            skull_pv = pv.read(skull_path)
            plotter.add_mesh(skull_pv, color="white", opacity=0.08, style="wireframe")

        plotter.add_legend(bcolor="#2d2d2d")
        plotter.camera_position = "iso"
        plotter.show()


if __name__ == "__main__":
    main()
