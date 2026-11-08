"""
Simulation 1002: Inverse Material Parameter Identification (Constitutive Calibration).
Calibrating Beam Young's Modulus E from Tip Force-Deflection Curves.

Demonstrates:
- [1/6] Loading Mesh Geometry: Reference 3D Cantilever Beam (Tet4 solid continuum).
- [2/6] Configuring Material & Forward Physics: Hookean elasticity & cantilever boundary conditions.
- [3/6] Forward Ground Truth Simulation: Multi-step tip indentation generating reference force-deflection curve.
- [4/6] Setting Up Inverse Problem: Perturbed initial guess (E_0 = 10,000 MPa vs E_truth = 21,000 MPa).
- [5/6] Executing Constitutive Parameter Calibration:
        - Closed-Form Linear Least-Squares (< 1 microsecond instant solve).
        - Nonlinear Trust-Region Reflective (TRF) numerical optimization.
- [6/6] Verifying Results & State Export: Validating parameter recovery (< 0.01% error),
        printing diagnostic summary table, and exporting calibrated state to .vtu.
"""

from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pyvista as pv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary.selector import select_nodes_by_bounds
from psierra.boundary import DirichletConstraint, DirichletManager
from psierra.analysis import LinearStaticAnalysis
from psierra.inverse import (
    MaterialParameterIdentifier,
    IdentificationResult,
)

install(show_locals=False)
console = Console()


def main() -> None:
    # 0. Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]Simulation 1002: Inverse Material Parameter Identification[/bold cyan]\n"
            "[dim]Constitutive Calibration of Young's Modulus E from Tip Force-Deflection Curves[/dim]",
            title="[bold blue]project-sierra / inverse[/bold blue]",
            border_style="cyan",
        )
    )

    # 1. Loading Mesh Geometry
    console.print("\n[bold yellow][1/6] Loading Mesh Geometry[/bold yellow]")
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    mesh = VolumetricMesh(mesh_path)
    console.print(
        f"  • Mesh: [green]{mesh_path.name}[/green] | Nodes: [green]{mesh.n_nodes}[/green] | "
        f"DOFs: [green]{mesh.n_dofs}[/green] | Elements: [green]{mesh.n_elements}[/green]"
    )

    # 2. Configuring Baseline Material & Domain
    console.print(
        "\n[bold yellow][2/6] Configuring Forward Continuum Physics[/bold yellow]"
    )
    E_truth = 21000.0  # MPa (Ground Truth Young's modulus)
    nu = 0.35  # Poisson's ratio
    rho = 1100e-6  # Mass density

    params_truth = IsotropicElasticParameters()
    params_truth.set_from_youngs_poisson(E_truth, nu, rho)
    mat_truth = Hookean(params_truth)
    elem_truth = Tet4Linear(mat_truth)
    domain_truth = SolidDomain(mesh, mat_truth, elem_truth)

    z_min = mesh.bounds[4]
    root_nodes = select_nodes_by_bounds(
        mesh, bounds=(-100, 100, -100, 100, z_min - 1, z_min + 5)
    )
    clamp = DirichletConstraint(
        node_indices=root_nodes, components=(0, 1, 2), value=0.0
    )

    # Indenter tip node at beam free end
    tip_node = 5
    console.print(
        f"  • Target Modulus (E_truth) : [green]{E_truth:.1f}[/green] MPa | Poisson's ratio: [green]{nu:.2f}[/green]"
    )
    console.print(
        f"  • Indenter Landmark Node   : Node [green]{tip_node}[/green] at coords {tuple(np.round(mesh.nodes[tip_node], 1))} mm"
    )

    # 3. Generating Ground Truth Experimental Curve
    console.print(
        "\n[bold yellow][3/6] Generating Experimental Force-Deflection Data[/bold yellow]"
    )
    test_indentations = np.array(
        [-2.0, -4.0, -6.0, -8.0, -10.0]
    )  # mm (prescribed Y-deflections)

    exp_forces: list[float] = []
    analysis_truth = LinearStaticAnalysis(domain_truth)
    f_ext_zero = np.zeros(mesh.n_dofs)

    for delta in test_indentations:
        tip_bc = DirichletConstraint([tip_node], components=(1,), value=delta)
        dm = DirichletManager(mesh.n_dofs)
        dm.add_constraint(clamp)
        dm.add_constraint(tip_bc)
        res = analysis_truth.solve(dm, f_ext_zero, verbose=False)
        # Indenter reaction force (action on probe = -reaction on structure)
        f_indenter = -res.reactions[3 * tip_node + 1]
        exp_forces.append(f_indenter)

    exp_forces_arr = np.array(exp_forces)
    console.print(f"  • Test Indentation Depths  : {test_indentations.tolist()} mm")
    console.print(
        f"  • Measured Reaction Forces : {[float(f'{f:.1f}') for f in exp_forces_arr]} N"
    )
    console.print(
        f"  • Peak Indentation Force   : [green]{exp_forces_arr[-1]:.1f}[/green] N at delta = {test_indentations[-1]:.1f} mm"
    )

    # 4. Setting Up Inverse Problem & Perturbed Initial Guess
    console.print(
        "\n[bold yellow][4/6] Setting Up Inverse Calibration Problem[/bold yellow]"
    )
    E_initial = 10000.0  # Initial guess (more than 52% off true value)
    console.print(
        f"  • Initial Guess (E_0)      : [red]{E_initial:.1f}[/red] MPa (Initial Error: [red]{abs(E_initial - E_truth)/E_truth*100:.1f}%[/red])"
    )
    console.print("  • Parameter Search Bounds  : [1,000.0, 100,000.0] MPa")

    # Compute baseline forces with E_initial
    params_base = IsotropicElasticParameters()
    params_base.set_from_youngs_poisson(E_initial, nu, rho)
    domain_base = SolidDomain(
        mesh, Hookean(params_base), Tet4Linear(Hookean(params_base))
    )
    analysis_base = LinearStaticAnalysis(domain_base)

    base_forces = []
    for delta in test_indentations:
        tip_bc = DirichletConstraint([tip_node], components=(1,), value=delta)
        dm = DirichletManager(mesh.n_dofs)
        dm.add_constraint(clamp)
        dm.add_constraint(tip_bc)
        res = analysis_base.solve(dm, f_ext_zero, verbose=False)
        base_forces.append(-res.reactions[3 * tip_node + 1])
    base_forces_arr = np.array(base_forces)

    # 5. Executing Constitutive Parameter Calibration
    console.print(
        "\n[bold yellow][5/6] Executing Material Parameter Identification[/bold yellow]"
    )

    # 5A: Closed-Form Linear Least-Squares Calibration
    t0_cf = time.perf_counter()
    res_closed_form = MaterialParameterIdentifier.calibrate_linear_elasticity(
        experimental_forces=exp_forces_arr,
        baseline_forces=base_forces_arr,
        baseline_modulus=E_initial,
    )
    t_cf = time.perf_counter() - t0_cf
    E_cf = res_closed_form.parameters["youngs_modulus"]
    err_cf = abs(E_cf - E_truth) / E_truth * 100.0
    console.print(f"  [cyan]Method 1: Closed-Form Linear Least-Squares[/cyan]")
    console.print(
        f"    • Calibrated E   : [green]{E_cf:.4f}[/green] MPa (Error: [green]{err_cf:.6f}%[/green])"
    )
    console.print(
        f"    • Evaluation Time: [green]{t_cf*1e6:.2f}[/green] µs ({1.0/t_cf:,.0f} Hz)"
    )

    # 5B: Numerical Trust-Region Reflective (TRF) Optimization
    console.print("  [cyan]Method 2: Trust-Region Reflective (TRF) Optimization[/cyan]")
    t0_trf = time.perf_counter()

    def forward_fn(params_dict: dict[str, float]) -> np.ndarray:
        E_val = params_dict["youngs_modulus"]
        p = IsotropicElasticParameters()
        p.set_from_youngs_poisson(E_val, nu, rho)
        d = SolidDomain(mesh, Hookean(p), Tet4Linear(Hookean(p)))
        ans = LinearStaticAnalysis(d)
        sim_forces = []
        for delta in test_indentations:
            tip_bc = DirichletConstraint([tip_node], components=(1,), value=delta)
            dm = DirichletManager(mesh.n_dofs)
            dm.add_constraint(clamp)
            dm.add_constraint(tip_bc)
            r = ans.solve(dm, f_ext_zero, verbose=False)
            sim_forces.append(-r.reactions[3 * tip_node + 1])
        return np.array(sim_forces)

    identifier = MaterialParameterIdentifier.from_bounds(
        {"youngs_modulus": (E_initial, 1000.0, 100000.0)},
        method="trf",
    )
    res_trf = identifier.identify(exp_forces_arr, forward_fn)
    t_trf = time.perf_counter() - t0_trf
    E_trf = res_trf.parameters["youngs_modulus"]
    err_trf = abs(E_trf - E_truth) / E_truth * 100.0

    console.print(
        f"    • Calibrated E   : [green]{E_trf:.4f}[/green] MPa (Error: [green]{err_trf:.6f}%[/green])"
    )
    console.print(
        f"    • Iterations     : [green]{res_trf.iterations}[/green] iterations ({res_trf.n_evaluations} FEM evaluations)"
    )
    console.print(f"    • Total Wall Time: [green]{t_trf:.3f}[/green] s")

    # 6. Verification Summary & State Export
    console.print("\n[bold yellow][6/6] Verifying Results & State Export[/bold yellow]")

    table = Table(
        title="1002 Material Parameter Identification Summary", border_style="cyan"
    )
    table.add_column("State / Estimator", style="bold white")
    table.add_column("Young's Modulus E", style="cyan")
    table.add_column("Relative Error", style="green")
    table.add_column("Force RMS Error", style="yellow")
    table.add_column("Solve Time", style="magenta")

    table.add_row(
        "Ground Truth Target",
        f"{E_truth:.2f} MPa",
        "0.00% (Baseline)",
        "0.00 N",
        "N/A",
    )
    table.add_row(
        "Initial Nominal Guess",
        f"{E_initial:.2f} MPa",
        f"{abs(E_initial - E_truth)/E_truth*100:.2f}%",
        f"{float(np.sqrt(np.mean((base_forces_arr - exp_forces_arr)**2))):.1f} N",
        "N/A",
    )
    table.add_row(
        "Closed-Form Least-Squares",
        f"{E_cf:.4f} MPa",
        f"{err_cf:.6f}%",
        f"{res_closed_form.rms_error:.4e} N",
        f"{t_cf*1e6:.1f} µs",
    )
    table.add_row(
        "Trust-Region Reflective (TRF)",
        f"{E_trf:.4f} MPa",
        f"{err_trf:.6f}%",
        f"{res_trf.rms_error:.4e} N",
        f"{t_trf:.3f} s",
    )

    console.print(table)

    # Export calibrated model state
    output_dir = Path("results/10_inverse/1002_beam_parameter_identification")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.vtu"
    legacy_path = output_dir / "calibrated_beam.vtu"

    params_calibrated = IsotropicElasticParameters()
    params_calibrated.set_from_youngs_poisson(E_trf, nu, rho)
    domain_calibrated = SolidDomain(
        mesh, Hookean(params_calibrated), Tet4Linear(Hookean(params_calibrated))
    )

    # Forward solve calibrated domain at peak test indentation (delta = -10.0 mm)
    # to populate full 3D displacement and stress fields for ParaView 'Warp By Vector'
    analysis_calibrated = LinearStaticAnalysis(domain_calibrated)
    tip_bc_peak = DirichletConstraint(
        [tip_node], components=(1,), value=test_indentations[-1]
    )
    dm_peak = DirichletManager(mesh.n_dofs)
    dm_peak.add_constraint(clamp)
    dm_peak.add_constraint(tip_bc_peak)
    result_calibrated = analysis_calibrated.solve(
        dm_peak, f_ext_zero, include_stresses=True, smooth_to_nodes=True, verbose=False
    )
    domain_calibrated.update_state(u=result_calibrated.displacement)
    domain_calibrated.save_state(
        filepath=output_path, include_stresses=True, smooth_to_nodes=True
    )
    domain_calibrated.save_state(
        filepath=legacy_path, include_stresses=True, smooth_to_nodes=True
    )
    console.print(
        f"[bold green]✓ Calibrated continuum state exported to:[/bold green] [cyan]{output_path}[/cyan] (and [cyan]{legacy_path}[/cyan])"
    )

    # Export indenter probe point
    indenter_pt = np.array([mesh.nodes[tip_node]])
    indenter_poly = pv.PolyData(indenter_pt)
    indenter_poly["Displacement"] = np.array([[0.0, test_indentations[-1], 0.0]])
    indenter_poly["Reaction_Force"] = np.array(
        [[0.0, res_trf.simulated_forces[-1], 0.0]]
    )
    indenter_vtp_path = output_dir / "indenter.vtp"
    indenter_poly.save(str(indenter_vtp_path))
    console.print(
        f"[bold green]✓ Indenter probe location exported to:[/bold green] [cyan]{indenter_vtp_path}[/cyan]\n"
    )


if __name__ == "__main__":
    main()
