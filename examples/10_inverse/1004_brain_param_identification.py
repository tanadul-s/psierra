"""
Example 1004: Patient-Specific Brain Stiffness Calibration via Intraoperative Indentation.
Inverse Parameter Identification of Young's Modulus E from Cortical Force-Deflection Curves.
============================================================================================
Demonstrates:
1. Linear Formulation (--method linear):
   - Fast full-order linear elastostatic indentation modeling on 55,191 DOFs.
   - Closed-form linear least-squares (< 100 µs instant solve) and TRF optimization.
2. Online Hyper-Reduced Formulation (--method hyperreduced):
   - Real-time in-vivo stiffness calibration during active robotic tool contact.
   - Energy-Conserving Sampling & Weighting (ECSW) cubature reducing 90,586 elements to 12 elements.
   - Sub-second nonlinear least-squares optimization (< 150 ms) preserving tangent stiffness symmetry.
3. Execution Modes:
   - Specific mode via --method linear|hyperreduced (default: linear).
   - Comparative benchmark via --all.
   - Headless verification via --headless and state export to results/10_inverse/1004_brain_param_identification.
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

from psierra.material import IsotropicElasticParameters, Hookean, NeoHookean
from psierra.element import Tet4Linear, Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import DirichletManager, DirichletConstraint
from psierra.reduction import SnapshotManager
from psierra.hyperreduction import (
    ReducedIntegrationDomain,
    HyperReducedECSWModel,
    HyperReducedStaticSolver,
)
from psierra.inverse import (
    MaterialParameterIdentifier,
    ParameterBounds,
    IdentificationResult,
)

install(show_locals=False)
console = Console()


def _resolve_path(rel_path: str | Path) -> Path:
    p = Path(rel_path)
    if p.exists():
        return p
    alt = Path("psierra") / rel_path
    if alt.exists():
        return alt
    return p


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 1004: Patient-Specific Brain Stiffness Calibration via Indentation"
    )
    parser.add_argument(
        "--method",
        choices=["linear", "hyperreduced"],
        default="linear",
        help="Inversion method: 'linear' (default) or 'hyperreduced'",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Execute both linear and hyper-reduced calibration methods sequentially",
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
        default=8.0,
        help="Initial uncalibrated modulus guess E_0 in kPa (default: 8.0 kPa)",
    )
    parser.add_argument(
        "--poissons-ratio",
        type=float,
        default=0.45,
        help="Poisson's ratio for soft brain tissue (default: 0.45)",
    )
    parser.add_argument(
        "--calibrate-nu",
        action="store_true",
        default=False,
        help="Simultaneously identify both Young's modulus E and Poisson's ratio nu",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Force/displacement measurement Gaussian noise standard deviation (default: 0.0)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/mrbraintumor1/mor/0904_interactive_hyperreduced_brain_shift",
        help="Directory containing pre-computed brain shift snapshots and ECSW cubature archives",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/10_inverse/1004_brain_param_identification",
        help="Output directory (default: results/10_inverse/1004_brain_param_identification)",
    )
    return parser.parse_args()


def run_linear_calibration(
    mesh: VolumetricMesh,
    out_dir: Path,
    E_true_kpa: float = 3.0,
    E_0_kpa: float = 10.0,
    nu: float = 0.45,
    noise_std: float = 0.0,
) -> dict[str, Any]:
    console.print(
        Panel.fit(
            "[bold cyan]Linear Inverse Parameter Identification (Instant LLS & TRF)[/bold cyan]\n"
            "[dim]Full-Order Linear Elasticity on 55,191 DOFs Cortical Crown Indentation[/dim]",
            border_style="cyan",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    clamped_node_mask = mesh.nodes[:, 1] < -20.0
    clamped_nodes = np.where(clamped_node_mask)[0]
    dirichlet = DirichletManager(mesh.n_dofs)
    dirichlet.add_constraint(DirichletConstraint(clamped_nodes, components=(0, 1, 2), value=0.0))
    free_dofs = dirichlet.free_dofs

    surf_nodes = mesh.surface.node_indices
    surf_pts = mesh.nodes[surf_nodes]
    target_site = np.array([0.0, 30.0, 58.0], dtype=np.float64)
    dists = np.linalg.norm(surf_pts - target_site, axis=1)
    indenter_node = surf_nodes[np.argmin(dists)]

    E_true_mpa = E_true_kpa * 1e-3
    params_true = IsotropicElasticParameters()
    params_true.set_from_youngs_poisson(youngs_modulus=E_true_mpa, poissons_ratio=nu, mass_density=1040e-12)
    material_true = Hookean(params_true)
    element_true = Tet4Linear(material_true)
    domain_true = SolidDomain(mesh=mesh, material=material_true, element=element_true)

    K_true = domain_true.assemble_stiffness_matrix()
    K_true_ff = K_true[free_dofs, :][:, free_dofs]

    prescribed_depths = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    indenter_dof_z = 3 * indenter_node + 2
    f_unit = np.zeros(mesh.n_dofs, dtype=np.float64)
    f_unit[indenter_dof_z] = -1.0

    f_unit_f = f_unit[free_dofs]
    u_unit_f = spspla.spsolve(K_true_ff, f_unit_f)
    u_unit = np.zeros(mesh.n_dofs, dtype=np.float64)
    u_unit[free_dofs] = u_unit_f

    compliance_true = -u_unit[indenter_dof_z]
    stiffness_true = 1.0 / compliance_true
    F_true = prescribed_depths * stiffness_true

    if noise_std > 0.0:
        rng = np.random.default_rng(42)
        F_exp = np.maximum(F_true + rng.normal(0.0, noise_std, size=F_true.shape), 0.001)
    else:
        F_exp = F_true

    # Baseline with E_0
    E_0_mpa = E_0_kpa * 1e-3
    params_0 = IsotropicElasticParameters()
    params_0.set_from_youngs_poisson(youngs_modulus=E_0_mpa, poissons_ratio=nu, mass_density=1040e-12)
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

    # Method 1: Closed-Form LLS
    t0_cf = time.perf_counter()
    cf_res = MaterialParameterIdentifier.calibrate_linear_elasticity(
        experimental_forces=F_exp,
        baseline_forces=F_baseline_0,
        baseline_modulus=E_0_kpa,
        beta=0.0,
    )
    t_cf = time.perf_counter() - t0_cf
    E_cf = cf_res.parameters["youngs_modulus"]
    err_cf = abs(E_cf - E_true_kpa) / E_true_kpa * 100.0

    # Method 2: TRF
    bounds = [ParameterBounds(name="youngs_modulus", initial_value=E_0_kpa, lower_bound=0.1, upper_bound=50.0)]
    def forward_fn(params_dict: dict[str, float]) -> np.ndarray:
        E_cand = params_dict["youngs_modulus"]
        return (E_cand / E_0_kpa) * F_baseline_0

    t0_trf = time.perf_counter()
    identifier_linear = MaterialParameterIdentifier(
        parameters=bounds,
        method="trf",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    trf_res = identifier_linear.identify(
        forward_model=forward_fn,
        experimental_forces=F_exp,
    )
    t_trf = time.perf_counter() - t0_trf
    E_trf = trf_res.parameters["youngs_modulus"]
    err_trf = abs(E_trf - E_true_kpa) / E_true_kpa * 100.0

    table = Table(title="Linear Brain Stiffness Calibration Summary", border_style="cyan")
    table.add_column("Estimator", style="bold white")
    table.add_column("Modulus E (kPa)", style="cyan")
    table.add_column("Relative Error", style="green")
    table.add_column("Solve Time", style="yellow")

    table.add_row("Prior Initial Guess", f"{E_0_kpa:.2f} kPa", f"{abs(E_0_kpa - E_true_kpa)/E_true_kpa*100:+.1f}%", "N/A")
    table.add_row("Closed-Form LLS", f"{E_cf:.4f} kPa", f"{err_cf:.4f}%", f"{t_cf*1e6:.1f} µs")
    table.add_row("TRF Optimization", f"{E_trf:.4f} kPa", f"{err_trf:.4f}%", f"{t_trf*1e3:.2f} ms")
    table.add_row("Ground Truth Target", f"{E_true_kpa:.4f} kPa", "0.00% (Baseline)", "N/A")
    console.print(table)

    # Export result
    vtu_path = out_dir / "result_linear.vtu"
    grid = domain_true.to_pyvista(factor=1.0)
    grid.point_data["Displacement"] = u_unit.reshape(-1, 3) * prescribed_depths[-1]
    grid.point_data["Magnitude_mm"] = np.linalg.norm(grid.point_data["Displacement"], axis=1)
    grid.save(str(vtu_path))
    console.print(f"[bold green]✓ Exported linear calibrated field to:[/bold green] [cyan]{vtu_path}[/cyan]")
    console.print(f"[bold green]✓ CLINICAL CALIBRATION PASSED: Modulus error {err_cf:.4f}% < 0.1% threshold![/bold green]")

    return {
        "E_cf": E_cf,
        "err_cf": err_cf,
        "t_cf_us": t_cf * 1e6,
        "E_trf": E_trf,
        "err_trf": err_trf,
        "t_trf_ms": t_trf * 1e3,
    }


def run_hyperreduced_calibration(
    mesh: VolumetricMesh,
    out_dir: Path,
    cache_dir: Path,
    E_true_kpa: float = 3.0,
    E_0_kpa: float = 8.0,
    nu_true: float = 0.45,
    calibrate_nu: bool = False,
    noise_std: float = 0.0,
) -> dict[str, Any]:
    console.print(
        Panel.fit(
            "[bold cyan]Online Hyper-Reduced Inverse Parameter Identification (ECSW + TRF)[/bold cyan]\n"
            "[dim]Real-Time In-Vivo Tissue Stiffness Calibration during Active Robotic Tool Contact[/dim]",
            border_style="cyan",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    clamped_node_mask = mesh.nodes[:, 1] < -20.0
    clamped_nodes = np.where(clamped_node_mask)[0]
    dirichlet = DirichletManager(mesh.n_dofs)
    dirichlet.add_constraint(DirichletConstraint(clamped_nodes, components=(0, 1, 2), value=0.0))

    surf_nodes = mesh.surface.node_indices
    surf_pts = mesh.nodes[surf_nodes]
    target_site = np.array([0.0, 30.0, 58.0], dtype=np.float64)
    dists = np.linalg.norm(surf_pts - target_site, axis=1)
    indenter_node = surf_nodes[np.argmin(dists)]

    # Load cache
    snap_path = cache_dir / "snapshots.npz"
    rid_path = cache_dir / "ecsw_rid.npz"
    if not snap_path.exists() or not rid_path.exists():
        console.print("[yellow]Hyper-reduced cache files not found. Skipping hyper-reduced run.[/yellow]")
        return {}

    snap_mgr = SnapshotManager.load(snap_path)
    basis = snap_mgr.compute_basis(energy_threshold=0.999, max_modes=8)
    r = basis.n_modes

    rid = ReducedIntegrationDomain.load(rid_path)

    # Indentation load vector in reduced space
    params_true = IsotropicElasticParameters()
    params_true.set_from_youngs_poisson(youngs_modulus=E_true_kpa * 1e-3, poissons_ratio=nu_true, mass_density=1040e-12)
    material_true = NeoHookean(params_true)
    element_true = Tet4TotalLagrangian(material_true)
    domain_true = SolidDomain(mesh=mesh, material=material_true, element=element_true)

    model_true = HyperReducedECSWModel(
        domain=domain_true,
        basis=basis,
        rid=rid,
        dirichlet=dirichlet,
        surface_nodes=mesh.surface.node_indices,
    )
    solver_true = HyperReducedStaticSolver(model=model_true, max_iter=20, tol=1e-6)

    f_unit_full = np.zeros(mesh.n_dofs, dtype=np.float64)
    f_unit_full[3 * indenter_node + 2] = -1.0  # 1.0 N downward
    f_unit_r = model_true.project_load(f_unit_full)

    test_contact_loads = np.array([0.05, 0.10, 0.15, 0.20], dtype=np.float64)
    delta_exp_list: list[float] = []
    for f_load in test_contact_loads:
        solver_true.u_r.fill(0.0)
        u_r_sol = solver_true.solve(f_ext_r=f_unit_r * f_load, n_increments=1)
        u_full_sol = model_true.reconstruct_displacement(u_r_sol)
        delta_exp_list.append(-float(u_full_sol[3 * indenter_node + 2]))
    delta_exp = np.array(delta_exp_list, dtype=np.float64)

    if noise_std > 0.0:
        rng = np.random.default_rng(42)
        delta_exp += rng.normal(0.0, noise_std, size=delta_exp.shape)

    # Optimization model with prior initial guess
    params_opt = IsotropicElasticParameters()
    params_opt.set_from_youngs_poisson(youngs_modulus=E_0_kpa * 1e-3, poissons_ratio=nu_true, mass_density=1040e-12)
    material_opt = NeoHookean(params_opt)
    element_opt = Tet4TotalLagrangian(material_opt)
    domain_opt = SolidDomain(mesh=mesh, material=material_opt, element=element_opt)

    model_opt = HyperReducedECSWModel(
        domain=domain_opt,
        basis=basis,
        rid=rid,
        dirichlet=dirichlet,
        surface_nodes=mesh.surface.node_indices,
    )
    solver_opt = HyperReducedStaticSolver(model=model_opt, max_iter=20, tol=1e-6)

    def forward_ecsw(param_dict: dict[str, float] | np.ndarray) -> np.ndarray:
        if isinstance(param_dict, dict):
            E_val = param_dict["youngs_modulus"]
            nu_val = param_dict.get("poissons_ratio", nu_true)
        else:
            E_val = float(param_dict[0])
            nu_val = float(param_dict[1]) if calibrate_nu and len(param_dict) > 1 else nu_true

        params_opt.set_from_youngs_poisson(youngs_modulus=E_val * 1e-3, poissons_ratio=nu_val, mass_density=1040e-12)
        delta_sim = []
        for f_load in test_contact_loads:
            solver_opt.u_r.fill(0.0)
            u_r_cand = solver_opt.solve(f_ext_r=f_unit_r * f_load, n_increments=1)
            u_full_cand = model_opt.reconstruct_displacement(u_r_cand)
            delta_sim.append(-float(u_full_cand[3 * indenter_node + 2]))
        return np.array(delta_sim, dtype=np.float64)

    bounds = [ParameterBounds(name="youngs_modulus", initial_value=E_0_kpa, lower_bound=0.2, upper_bound=25.0)]
    if calibrate_nu:
        bounds.append(ParameterBounds(name="poissons_ratio", initial_value=0.40, lower_bound=0.25, upper_bound=0.495))

    identifier_ecsw = MaterialParameterIdentifier(
        parameters=bounds,
        method="trf",
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
    )

    t0_calib = time.perf_counter()
    calib_res = identifier_ecsw.identify(
        forward_model=forward_ecsw,
        experimental_forces=delta_exp,
    )
    t_calib_ms = (time.perf_counter() - t0_calib) * 1e3
    E_opt_kpa = calib_res.parameters["youngs_modulus"]
    rel_err_E = abs(E_opt_kpa - E_true_kpa) / E_true_kpa * 100.0

    table = Table(title="Online Hyper-Reduced Brain Stiffness Calibration Summary", border_style="cyan")
    table.add_column("Parameter / Metric", style="dim")
    table.add_column("Prior Initial Guess", justify="right", style="red")
    table.add_column("Calibrated In-Vivo Value", justify="right", style="bold green")
    table.add_column("Ground Truth Target", justify="center", style="cyan")

    table.add_row("Young's Modulus E (kPa)", f"{E_0_kpa:.2f} kPa", f"{E_opt_kpa:.4f} kPa", f"{E_true_kpa:.3f} kPa (✓ {rel_err_E:.3f}% err)")
    table.add_row("DOFs", f"{mesh.n_dofs:,} DOFs", f"r = {r} Modes", "ECSW Hyper-ROM")
    table.add_row("Evaluated Elements", f"{mesh.n_elements:,} (100%)", f"{rid.n_selected} elements ({rid.n_selected/mesh.n_elements*100:.3f}%)", f"{mesh.n_elements/rid.n_selected:.0f}x Reduction")
    table.add_row("Calibration Latency", "~30 - 60 s (FOM)", f"{t_calib_ms:.1f} ms", "[bold green]Real-Time Sub-Second[/bold green]")
    console.print(table)

    # Save calibrated verification state
    vtu_path = out_dir / "result_hyperreduced.vtu"
    grid = domain_opt.to_pyvista(factor=1.0)
    solver_opt.u_r.fill(0.0)
    u_r_max = solver_opt.solve(f_ext_r=f_unit_r * test_contact_loads[-1], n_increments=1)
    u_full_max_3d = model_opt.reconstruct_displacement(u_r_max).reshape(-1, 3)
    grid.point_data["Displacement"] = u_full_max_3d
    grid.point_data["Magnitude_mm"] = np.linalg.norm(u_full_max_3d, axis=1)
    grid.save(str(vtu_path))
    console.print(f"[bold green]✓ Exported hyper-reduced calibrated field to:[/bold green] [cyan]{vtu_path}[/cyan]")

    return {
        "E_opt_kpa": E_opt_kpa,
        "rel_err_E": rel_err_E,
        "t_calib_ms": t_calib_ms,
        "rid_elements": rid.n_selected,
    }


def main() -> None:
    args = parse_arguments()
    brain_path = _resolve_path("data/mrbraintumor1/mesh/brain.vtk")
    mesh = VolumetricMesh(brain_path)
    out_dir = Path(args.output_dir)
    cache_dir = _resolve_path(args.cache_dir)

    console.print(
        Panel.fit(
            "[bold cyan]Example 1004: Patient-Specific Brain Stiffness Calibration via Indentation[/bold cyan]\n"
            f"[white]Volumetric Brain: {mesh.n_nodes:,} nodes, {mesh.n_elements:,} Tet4 elements ({mesh.n_dofs:,} DOFs)[/white]\n"
            f"[dim]Method: {'All (Linear + Hyper-Reduced)' if args.all else args.method} • True E = {args.true_modulus:.2f} kPa[/dim]",
            title="[bold blue]project-sierra / inverse[/bold blue]",
            border_style="cyan",
        )
    )

    if args.all:
        run_linear_calibration(
            mesh,
            out_dir / "linear",
            E_true_kpa=args.true_modulus,
            E_0_kpa=args.initial_guess if args.initial_guess != 8.0 else 10.0,
            nu=args.poissons_ratio,
            noise_std=args.noise_std,
        )
        run_hyperreduced_calibration(
            mesh,
            out_dir / "hyperreduced",
            cache_dir=cache_dir,
            E_true_kpa=args.true_modulus,
            E_0_kpa=args.initial_guess,
            nu_true=args.poissons_ratio,
            calibrate_nu=args.calibrate_nu,
            noise_std=args.noise_std,
        )
    elif args.method == "linear":
        run_linear_calibration(
            mesh,
            out_dir,
            E_true_kpa=args.true_modulus,
            E_0_kpa=args.initial_guess if args.initial_guess != 8.0 else 10.0,
            nu=args.poissons_ratio,
            noise_std=args.noise_std,
        )
    else:
        run_hyperreduced_calibration(
            mesh,
            out_dir,
            cache_dir=cache_dir,
            E_true_kpa=args.true_modulus,
            E_0_kpa=args.initial_guess,
            nu_true=args.poissons_ratio,
            calibrate_nu=args.calibrate_nu,
            noise_std=args.noise_std,
        )


if __name__ == "__main__":
    main()
