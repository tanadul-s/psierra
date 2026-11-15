"""
Example 0502: Master Nonlinear Dynamic Neurosurgical Brain Shift under Gravity.

Demonstrates:
1. Patient-specific neurosurgical anatomy in native RAS coordinates:
   - Brain parenchyma mesh (18,397 nodes, 90,586 Tet4 elements)
   - Deep subcortical tumor lesion tracking via KD-Tree
2. Physiological soft tissue constitutive models:
   - Pure Hyperelasticity (Compressible Neo-Hookean, E = 3.0 kPa, nu = 0.45)
   - Viscoelasticity (Franceschini et al. multi-term Prony relaxation spectrum)
3. Intraoperative surgical patient postures:
   - Supine: Posterior sag (g along -Y)
   - Prone: Anterior sag (g along +Y)
   - Lateral: 30° head rotation (compound gx, gy)
   - Vertical: Upright sitting craniotomy (caudal sag along -Z)
4. Cerebrospinal Fluid (CSF) hydrostatic buoyancy (--buoyancy, 96.2% buoyant relief).
5. High-performance implicit dynamics via InteractiveDynamicSimulation / Generalized-alpha.
6. Deep subcortical lesion migration and cortical sag telemetry.
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
    BuoyancyLoad,
)
from psierra.damping import RayleighDamping
from psierra.analysis import InteractiveDynamicSimulation
from psierra.exchange import SimulationWriter


SURGICAL_POSTURES: dict[str, tuple[float, float, float, str]] = {
    "supine": (0.0, -9810.0, 0.0, "Supine Posture (Posterior Sag along -Y)"),
    "prone": (0.0, 9810.0, 0.0, "Prone Posture (Anterior Sag along +Y)"),
    "lateral": (4905.0, -8495.7, 0.0, "30° Lateral Head Rotation (gx = 0.5g, gy = -0.866g)"),
    "vertical": (0.0, 0.0, -9810.0, "Sitting / Upright Posture (Caudal Sag along -Z)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0502: Master Nonlinear Dynamic Brain Shift"
    )
    parser.add_argument(
        "--posture",
        type=str,
        choices=list(SURGICAL_POSTURES.keys()),
        default="supine",
        help="Surgical patient posture (default: supine)",
    )
    parser.add_argument(
        "--material",
        type=str,
        choices=["hyperelastic", "viscoelastic"],
        default="hyperelastic",
        help="Tissue constitutive model (default: hyperelastic)",
    )
    parser.add_argument(
        "--buoyancy",
        action="store_true",
        help="Include Cerebrospinal Fluid (CSF) hydrostatic buoyancy (rho_csf = 1000 kg/m³)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of dynamic time increments (default: 200)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005,
        help="Time step size in seconds (default: 0.005 s)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Output frame interval (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/05_nonlinear_dynamic/0502_brain",
        help="Base directory for output datasets (default: results/05_nonlinear_dynamic/0502_brain)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comparative benchmark across postures, materials, and CSF buoyancy",
    )
    return parser.parse_args()


def run_brain_simulation(
    posture_key: str,
    material_type: str,
    buoyancy: bool,
    n_steps: int,
    dt: float,
    save_interval: int,
    base_out: Path,
    brain_mesh_path: Path,
    lesion_mesh_path: Path | None,
    verbose: bool = True,
) -> dict[str, str | float]:
    console = Console()
    gx, gy, gz, posture_desc = SURGICAL_POSTURES[posture_key]

    case_tag = f"{posture_key}_{material_type}"
    if buoyancy:
        case_tag += "_buoyant"
    out_dir = base_out / case_tag
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Physiological material parameters (E = 3.0 kPa, nu = 0.45, rho = 1040 kg/m^3)
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=3000.0,       # Pa (g/(mm·s²))
        poissons_ratio=0.45,
        mass_density=1040e-6,        # g/mm³
    )
    base_mat = NeoHookean(params)

    if material_type == "viscoelastic":
        prony_terms = [
            PronyTerm(relative_modulus=0.35, relaxation_time=0.05),
            PronyTerm(relative_modulus=0.25, relaxation_time=0.25),
        ]
        material = Viscoelastic(base_mat, prony_terms)
    else:
        material = base_mat

    element = Tet4TotalLagrangian(material)
    domain = SolidDomain(mesh, material, element)

    # Cranial base support clamping (Y < -20.0 mm in RAS coordinates)
    clamped_node_mask = mesh.nodes[:, 1] < -20.0
    clamped_nodes = np.where(clamped_node_mask)[0]
    cortical_free_nodes = np.array([idx for idx in cortical_surface_all if idx not in clamped_nodes])

    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(DirichletConstraint(clamped_nodes, components=(0, 1, 2), value=0.0))

    # Gravity force
    gravity = GravityLoad(acceleration=(gx, gy, gz))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    # Optional CSF fluid buoyancy
    if buoyancy:
        fluid_density = 1000e-6  # CSF density 1000 kg/m³ = 1000e-6 g/mm³
        buoy = BuoyancyLoad(
            fluid_density=fluid_density,
            acceleration=(gx, gy, gz),
        )
        f_buoy = buoy.compute_force_vector(
            domain, mass_matrix=M, solid_density=params.mass_density
        )
        f_ext = f_ext + f_buoy

    damping = RayleighDamping(alpha=1.5, beta=0.005)

    if verbose:
        console.print(f"  • Simulating: [bold cyan]{case_tag}[/bold cyan] ({n_steps} steps, dt={dt:.4f}s)")

    out_file = out_dir / "result.xdmf"
    writer = SimulationWriter(out_file, mesh=mesh, format="xdmf")
    writer.add_timestep(time=0.0, domain=domain, step_index=0)

    sim = InteractiveDynamicSimulation(
        domain=domain,
        dirichlet=dirichlet,
        damping=damping,
        dt=dt,
        spectral_radius=0.8,
        f_ext_base=f_ext,
        nonlinear=True,
        modified_newton=True,
        nr_max_iter=2,
        nr_tol=1e-3,
    )

    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        sim.step()
        if step % save_interval == 0 or step == n_steps:
            writer.add_timestep(time=step * dt, domain=domain, step_index=step)
    wall_time = time.perf_counter() - t0
    writer.close()

    # Telemetry
    disp = domain.displacement
    max_disp = float(np.linalg.norm(disp, axis=1).max())
    max_cort_sag = float(np.linalg.norm(disp[cortical_free_nodes], axis=1).max()) if len(cortical_free_nodes) > 0 else 0.0

    if has_lesion and len(lesion_node_indices) > 0:
        final_lesion_center = np.mean((mesh.nodes + disp)[lesion_node_indices], axis=0)
        lesion_drift = float(np.linalg.norm(final_lesion_center - initial_lesion_center))
    else:
        lesion_drift = 0.0

    return {
        "case": case_tag,
        "posture": posture_key.upper(),
        "material": material_type.capitalize(),
        "buoyancy": "Yes" if buoyancy else "No",
        "steps": n_steps,
        "max_disp_mm": max_disp,
        "cortical_sag_mm": max_cort_sag,
        "tumor_drift_mm": lesion_drift,
        "runtime_s": wall_time,
        "output_path": str(out_file),
    }


def main() -> None:
    args = parse_args()
    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Example 0502: Master Nonlinear Dynamic Neurosurgical Brain Shift[/bold cyan]\n"
            f"[dim]Posture: {args.posture.upper()} | Material: {args.material.capitalize()} | Buoyancy: {args.buoyancy}[/dim]",
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
        console.print("\n[bold yellow]Executing Comparative Neurosurgical Matrix...[/bold yellow]")
        cases = [
            ("supine", "hyperelastic", False),
            ("prone", "hyperelastic", False),
            ("lateral", "hyperelastic", False),
            ("vertical", "hyperelastic", False),
            ("supine", "viscoelastic", False),
            ("supine", "hyperelastic", True),
        ]
        results = []
        for post, mat, buoy in cases:
            res = run_brain_simulation(
                posture_key=post,
                material_type=mat,
                buoyancy=buoy,
                n_steps=min(args.steps, 30),  # fast matrix
                dt=args.dt,
                save_interval=args.save_interval,
                base_out=base_out,
                brain_mesh_path=brain_path,
                lesion_mesh_path=lesion_path,
                verbose=True,
            )
            results.append(res)

        table = Table(title="Example 0502: Neurosurgical Posture & Tissue Comparison", border_style="cyan")
        table.add_column("Case Tag", style="bold white")
        table.add_column("Posture", style="cyan")
        table.add_column("Material", style="magenta")
        table.add_column("Buoyancy", style="green")
        table.add_column("Peak Sag (mm)", style="green")
        table.add_column("Cortical Sag (mm)", style="green")
        table.add_column("Tumor Drift (mm)", style="yellow")
        table.add_column("Runtime (s)", style="blue")

        for r in results:
            table.add_row(
                str(r["case"]),
                str(r["posture"]),
                str(r["material"]),
                str(r["buoyancy"]),
                f"{r['max_disp_mm']:.3f}",
                f"{r['cortical_sag_mm']:.3f}",
                f"{r['tumor_drift_mm']:.3f}",
                f"{r['runtime_s']:.2f}",
            )
        console.print(table)

    else:
        console.print(f"\n[bold yellow]Executing Canonical Brain Shift ({args.steps} steps)...[/bold yellow]")
        res = run_brain_simulation(
            posture_key=args.posture,
            material_type=args.material,
            buoyancy=args.buoyancy,
            n_steps=args.steps,
            dt=args.dt,
            save_interval=args.save_interval,
            base_out=base_out,
            brain_mesh_path=brain_path,
            lesion_mesh_path=lesion_path,
            verbose=True,
        )

        table = Table(title=f"Example 0502: {res['case']}", border_style="cyan")
        table.add_column("Clinical Parameter", style="bold white")
        table.add_column("Observed Value", style="cyan")
        table.add_row("Surgical Posture", str(res["posture"]))
        table.add_row("Soft Tissue Model", str(res["material"]))
        table.add_row("CSF Hydrostatic Buoyancy", str(res["buoyancy"]))
        table.add_row("Dynamic Increments", f"{res['steps']} steps (dt={args.dt:.4f} s)")
        table.add_row("Peak Parenchymal Sag", f"{res['max_disp_mm']:.4f} mm")
        table.add_row("Cortical Surface Sag", f"{res['cortical_sag_mm']:.4f} mm")
        table.add_row("Deep Subcortical Tumor Drift", f"{res['tumor_drift_mm']:.4f} mm")
        table.add_row("Solver Runtime", f"{res['runtime_s']:.2f} s")
        table.add_row("ParaView Output", str(res["output_path"]))
        console.print(table)
        console.print(f"\n[bold green]✓ Simulation complete. XDMF dataset saved to:[/bold green] [cyan]{res['output_path']}[/cyan]\n")


if __name__ == "__main__":
    main()
