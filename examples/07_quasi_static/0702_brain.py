"""
Example 0702: Patient-Specific Neurosurgical Brain Relaxation via Quasi-Static Analysis.

Demonstrates:
1. Patient-specific neurosurgical anatomy in native RAS coordinates (brain.vtk and lesion.vtk).
2. Rate-dependent viscoelastic tissue relaxation under sustained surgical gravity without inertia:
   - Franceschini et al. physiological relaxation spectrum for parenchymal tissue.
   - QuasiStaticAnalysis marching physical equilibrium R(u_{n+1}) = 0.
3. Intraoperative time progression:
   - Initial surgical posture positioning ramp (t <= t_ramp).
   - Sustained gravitational holding phase (t > t_ramp) capturing viscoelastic creep.
4. Continuous tracking of deep subcortical tumor centroid migration and cortical sag.
5. ParaView XDMF / PVD export of temporal relaxation sequence.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import pyvista as pv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.spatial import cKDTree

from psierra.material import (
    IsotropicElasticParameters,
    NeoHookean,
    Viscoelastic,
    PronyTerm,
)
from psierra.element import Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import (
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
)
from psierra.analysis import QuasiStaticAnalysis


SURGICAL_POSTURES: dict[str, tuple[float, float, float, str]] = {
    "supine": (0.0, -9810.0, 0.0, "Supine Posture (Posterior sag along -Y)"),
    "prone": (0.0, 9810.0, 0.0, "Prone Posture (Anterior sag along +Y)"),
    "lateral": (4905.0, -8495.7, 0.0, "30° Lateral Head Rotation"),
    "vertical": (0.0, 0.0, -9810.0, "Sitting / Upright Posture (Caudal sag along -Z)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0702: Neurosurgical Brain Viscoelastic Relaxation"
    )
    parser.add_argument(
        "--posture",
        type=str,
        choices=list(SURGICAL_POSTURES.keys()),
        default="supine",
        help="Surgical patient posture (default: supine)",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=0.5,
        help="Total observation duration in seconds (default: 0.5 s)",
    )
    parser.add_argument(
        "--ramp-time",
        type=float,
        default=0.1,
        help="Duration of posture settling ramp in seconds (default: 0.1 s)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Quasi-static time step increment in seconds (default: 0.05 s)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["xdmf", "pvd"],
        default="xdmf",
        help="ParaView output format (default: xdmf)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/07_quasi_static/0702_brain",
        help="Base output directory (default: results/07_quasi_static/0702_brain)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comparative analysis across postures",
    )
    return parser.parse_args()


def run_brain_quasistatic(
    posture_key: str,
    total_time: float,
    ramp_time: float,
    dt: float,
    fmt: str,
    base_out: Path,
    brain_mesh_path: Path,
    lesion_mesh_path: Path | None,
    verbose: bool = True,
) -> dict[str, str | float]:
    console = Console()
    gx, gy, gz, posture_desc = SURGICAL_POSTURES[posture_key]

    case_tag = f"qs_{posture_key}"
    run_out_dir = base_out / case_tag
    run_out_dir.mkdir(parents=True, exist_ok=True)

    mesh = VolumetricMesh(brain_mesh_path)

    # Subcortical lesion tracking
    has_lesion = lesion_mesh_path is not None and lesion_mesh_path.exists()
    if has_lesion:
        lesion_pv = pv.read(lesion_mesh_path)
        tree = cKDTree(mesh.nodes)
        _, lesion_point_to_brain = tree.query(lesion_pv.points)
        lesion_node_indices = np.unique(lesion_point_to_brain)
        initial_lesion_center = np.mean(mesh.nodes[lesion_node_indices], axis=0)
    else:
        lesion_node_indices = np.array([], dtype=int)
        initial_lesion_center = np.zeros(3)

    # Cortical surface nodes
    surf_poly = mesh.ugrid.extract_surface(algorithm="dataset_surface")
    cortical_surface_all = np.unique(surf_poly.point_data["vtkOriginalPointIds"])

    # Physiological soft tissue properties (Franceschini 2-term viscoelastic spectrum)
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=3.0e-3,       # 3.0 kPa in MPa
        poissons_ratio=0.45,
        mass_density=1040.0 * 1e-12, # tonne/mm³
    )
    base_mat = NeoHookean(params)
    prony_terms = [
        PronyTerm(relative_modulus=0.35, relaxation_time=0.05),
        PronyTerm(relative_modulus=0.25, relaxation_time=0.25),
    ]
    mat = Viscoelastic(base_mat, prony_terms)
    elem = Tet4TotalLagrangian(mat)
    domain = SolidDomain(mesh, mat, elem)

    # Skull base / posterior cranial cradle clamp (Y < -20 mm in RAS coordinates)
    clamped_node_mask = mesh.nodes[:, 1] < -20.0
    clamped_nodes = np.where(clamped_node_mask)[0]
    cortical_free_nodes = np.array([idx for idx in cortical_surface_all if idx not in clamped_nodes])

    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(DirichletConstraint(clamped_nodes, components=(0, 1, 2), value=0.0))

    # Gravity force vector
    gravity = GravityLoad(acceleration=(gx, gy, gz))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    def load_schedule(t_cur: float) -> float:
        return min(1.0, t_cur / max(ramp_time, 1e-6))

    history_disp: list[float] = []

    def callback_step(step: int, t_cur: float, u_vec: np.ndarray) -> None:
        u_mag = float(np.linalg.norm(u_vec.reshape(-1, 3), axis=1).max())
        history_disp.append(u_mag)

    if verbose:
        console.print(f"  • Simulating quasi-static brain shift: [bold cyan]{case_tag}[/bold cyan] ({posture_desc})")

    t0 = time.perf_counter()
    solver = QuasiStaticAnalysis(domain)
    res = solver.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        total_time=total_time,
        dt=dt,
        load_schedule=load_schedule,
        nr_iterations=20,
        save_directory=run_out_dir,
        format=fmt,
        verbose=False,
        callback=callback_step,
    )
    wall_time = time.perf_counter() - t0

    disp = domain.displacement
    max_disp = float(np.linalg.norm(disp, axis=1).max())
    max_cort_sag = float(np.linalg.norm(disp[cortical_free_nodes], axis=1).max()) if len(cortical_free_nodes) > 0 else 0.0

    if has_lesion and len(lesion_node_indices) > 0:
        final_lesion_center = np.mean((mesh.nodes + disp)[lesion_node_indices], axis=0)
        lesion_drift = float(np.linalg.norm(final_lesion_center - initial_lesion_center))
    else:
        lesion_drift = 0.0

    ramp_idx = int(round(ramp_time / dt))
    disp_ramp = history_disp[ramp_idx] if ramp_idx < len(history_disp) else max_disp
    creep_factor = max_disp / max(disp_ramp, 1e-9)

    out_file = run_out_dir / f"result.{fmt}"

    return {
        "case": case_tag,
        "posture": posture_key.upper(),
        "steps": res.steps_completed,
        "disp_ramp_mm": disp_ramp,
        "disp_final_mm": max_disp,
        "cortical_sag_mm": max_cort_sag,
        "tumor_drift_mm": lesion_drift,
        "creep_factor": creep_factor,
        "runtime_s": wall_time,
        "output_path": str(out_file),
    }


def main() -> None:
    args = parse_args()
    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Example 0702: Patient-Specific Neurosurgical Brain Relaxation[/bold cyan]\n"
            f"[dim]Posture: {args.posture.upper()} | Total Time: {args.total_time:.2f} s | dt = {args.dt:.3f} s[/dim]",
            title="[bold blue]project-sierra — Neurosurgical Biomechanics[/bold blue]",
            border_style="cyan",
        )
    )

    brain_path = Path("data/mrbraintumor1/mesh/brain.vtk")
    lesion_path = Path("data/mrbraintumor1/mesh/lesion.vtk")
    if not brain_path.exists():
        raise FileNotFoundError(f"Brain mesh not found: {brain_path}")

    base_out = Path(args.output_dir)

    if args.all:
        console.print("\n[bold yellow]Executing Multi-Posture Brain Relaxation Study...[/bold yellow]")
        cases = ["supine", "prone", "lateral", "vertical"]
        results = []
        for post in cases:
            res = run_brain_quasistatic(
                posture_key=post,
                total_time=args.total_time,
                ramp_time=args.ramp_time,
                dt=args.dt,
                fmt=args.format,
                base_out=base_out,
                brain_mesh_path=brain_path,
                lesion_mesh_path=lesion_path,
                verbose=True,
            )
            results.append(res)

        table = Table(title="Example 0702: Multi-Posture Brain Relaxation Summary", border_style="cyan")
        table.add_column("Posture", style="bold white")
        table.add_column("Ramp Sag (mm)", style="cyan")
        table.add_column("Relaxed Sag (mm)", style="green")
        table.add_column("Cortical Sag (mm)", style="green")
        table.add_column("Tumor Drift (mm)", style="yellow")
        table.add_column("Creep Factor", style="magenta")
        table.add_column("Runtime (s)", style="blue")

        for r in results:
            table.add_row(
                str(r["posture"]),
                f"{r['disp_ramp_mm']:.3f}",
                f"{r['disp_final_mm']:.3f}",
                f"{r['cortical_sag_mm']:.3f}",
                f"{r['tumor_drift_mm']:.3f}",
                f"{r['creep_factor']:.2f}x",
                f"{r['runtime_s']:.2f}",
            )
        console.print(table)

    else:
        console.print(f"\n[bold yellow]Executing Canonical Brain Relaxation ({args.posture.upper()})...[/bold yellow]")
        res = run_brain_quasistatic(
            posture_key=args.posture,
            total_time=args.total_time,
            ramp_time=args.ramp_time,
            dt=args.dt,
            fmt=args.format,
            base_out=base_out,
            brain_mesh_path=brain_path,
            lesion_mesh_path=lesion_path,
            verbose=True,
        )

        table = Table(title=f"Example 0702: {res['case']}", border_style="cyan")
        table.add_column("Clinical Parameter", style="bold white")
        table.add_column("Observed Value", style="cyan")
        table.add_row("Surgical Posture", str(res["posture"]))
        table.add_row("Quasi-Static Increments", f"{res['steps']} steps (dt={args.dt:.3f} s)")
        table.add_row("Post-Ramp Peak Sag", f"{res['disp_ramp_mm']:.4f} mm")
        table.add_row("Relaxed Equilibrium Sag", f"{res['disp_final_mm']:.4f} mm")
        table.add_row("Cortical Surface Sag", f"{res['cortical_sag_mm']:.4f} mm")
        table.add_row("Deep Subcortical Tumor Drift", f"{res['tumor_drift_mm']:.4f} mm")
        table.add_row("Viscoelastic Creep Ratio", f"{res['creep_factor']:.3f}x")
        table.add_row("Solver Runtime", f"{res['runtime_s']:.2f} s")
        table.add_row("ParaView Trajectory", str(res["output_path"]))
        console.print(table)
        console.print(f"\n[bold green]✓ Brain relaxation simulation complete. Output saved to:[/bold green] [cyan]{res['output_path']}[/cyan]\n")


if __name__ == "__main__":
    main()
