"""
Example 0602: Neurosurgical Burr Hole Brain Shift: Cortical Bulging vs. Gravitational Sagging.

Demonstrates:
1. Patient-specific neurosurgical anatomy (MRI brain mesh + subcortical tumor lesion).
2. Frontal burr hole trephination (radius R = 10.0 mm) placed over deep pathology.
3. Matrix-free Dynamic Relaxation (DynamicRelaxationAnalysis) resolving equilibrium:
   - Scenario 1 (Bulging / Prolapse): Elevated Intracranial Pressure (ICP) driving parenchymal herniation outward through the aperture.
   - Scenario 2 (Sagging / Collapse): Intraoperative CSF drainage causing loss of buoyant lift and inward cortical collapse (pneumocephalus).
4. Direct Linear Static solver comparison.
5. Export of 3D annular cranial bone collar (.vtp) and ParaView dataset.
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

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear
from psierra.mesh import VolumetricMesh
from psierra.mesh.surface import SurfaceMesh
from psierra.domain import SolidDomain
from psierra.boundary import (
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
    BuoyancyLoad,
    HydrostaticPressureLoad,
)
from psierra.analysis import (
    LinearStaticAnalysis,
    DynamicRelaxationAnalysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0602: Neurosurgical Burr Hole Brain Shift"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["bulging", "sagging", "both"],
        default="bulging",
        help="Surgical scenario to simulate (default: bulging)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["dr", "static", "both"],
        default="dr",
        help="Solver methodology: 'dr' (Dynamic Relaxation), 'static' (Direct Linear Static), or 'both' (default: dr)",
    )
    parser.add_argument(
        "--burr-radius",
        type=float,
        default=10.0,
        help="Burr hole aperture radius in mm (default: 10.0 mm)",
    )
    parser.add_argument(
        "--icp-pressure",
        type=float,
        default=12.0,
        help="Internal cystic/ventricular ICP pressure in kPa for bulging scenario (default: 12.0 kPa)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Max pseudo-time iterations for Dynamic Relaxation (default: 60)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/06_dynamic_relaxation/0602_brain",
        help="Base output directory (default: results/06_dynamic_relaxation/0602_brain)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comprehensive study across bulging and sagging scenarios with DR and static solvers",
    )
    return parser.parse_args()


def create_burr_hole_geometry(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    thickness: float = 4.0,
    wall_width: float = 3.5,
) -> tuple[pv.PolyData, pv.PolyData]:
    n_pts = 64
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    if abs(normal[0]) < 0.9:
        t1 = np.cross(normal, [1.0, 0.0, 0.0])
    else:
        t1 = np.cross(normal, [0.0, 1.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    r_in = radius
    r_out = radius + wall_width

    inner_circle = center + r_in * (np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2)
    lines = []
    for i in range(n_pts):
        lines.extend([2, i, (i + 1) % n_pts])
    wireframe = pv.PolyData(inner_circle, lines=lines)

    pts_bot_in = center + r_in * (np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2)
    pts_bot_out = center + r_out * (np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2)
    pts_top_in = pts_bot_in + thickness * normal
    pts_top_out = pts_bot_out + thickness * normal

    all_pts = np.vstack([pts_bot_in, pts_bot_out, pts_top_in, pts_top_out])
    faces = []
    for i in range(n_pts):
        j = (i + 1) % n_pts
        faces.extend([3, i, j, 2 * n_pts + j])
        faces.extend([3, i, 2 * n_pts + j, 2 * n_pts + i])
        faces.extend([3, 2 * n_pts + i, 2 * n_pts + j, 3 * n_pts + j])
        faces.extend([3, 2 * n_pts + i, 3 * n_pts + j, 3 * n_pts + i])
        faces.extend([3, n_pts + j, n_pts + i, 3 * n_pts + i])
        faces.extend([3, n_pts + j, 3 * n_pts + i, 3 * n_pts + j])
        faces.extend([3, n_pts + i, i, 2 * n_pts + i])
        faces.extend([3, n_pts + i, 2 * n_pts + i, 3 * n_pts + i])

    bone_collar = pv.PolyData(all_pts, faces=faces)
    return bone_collar, wireframe


def main() -> None:
    args = parse_args()
    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Example 0602: Neurosurgical Burr Hole Brain Shift[/bold cyan]\n"
            f"[dim]Scenario: {args.scenario.upper()} | Solver: {args.solver.upper()} | Aperture Radius: {args.burr_radius:.1f} mm[/dim]",
            title="[bold blue]project-sierra — Dynamic Relaxation & Neurosurgery[/bold blue]",
            border_style="cyan",
        )
    )

    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    # 1. Loading Anatomical Datasets
    console.print("\n[bold yellow][1/5] Loading Patient Anatomical Datasets & Burr Hole Drilling[/bold yellow]")
    brain_mesh_path = Path("data/mrbraintumor1/mesh/brain.vtk")
    lesion_mesh_path = Path("data/mrbraintumor1/mesh/lesion.vtk")
    if not brain_mesh_path.exists() or not lesion_mesh_path.exists():
        raise FileNotFoundError("Brain or lesion mesh dataset not found.")

    mesh = VolumetricMesh(brain_mesh_path)
    lesion_pv = pv.read(lesion_mesh_path)

    # Extract brain cortical surface
    surf = mesh.surface

    # Map lesion boundary vertices to volumetric mesh
    lesion_surf_pv = lesion_pv.extract_surface(algorithm=None)
    faces_lesion = lesion_surf_pv.faces.reshape(-1, 4)[:, 1:4]
    faces_inv = np.column_stack([faces_lesion[:, 0], faces_lesion[:, 2], faces_lesion[:, 1]])
    lesion_surf_inv = pv.PolyData(
        lesion_surf_pv.points,
        faces=np.column_stack([np.full(len(faces_lesion), 3), faces_inv]).ravel(),
    )
    kdtree = cKDTree(mesh.nodes)
    _, global_indices = kdtree.query(lesion_surf_pv.points)
    lesion_smesh = SurfaceMesh(source=lesion_surf_inv, node_indices=global_indices)

    # Burr hole placement directly over lesion
    lesion_center = np.array(lesion_pv.center)
    dists_surf_to_lesion = np.linalg.norm(surf.nodes - lesion_center, axis=1)
    burr_center = surf.nodes[np.argmin(dists_surf_to_lesion)]
    normal_burr = burr_center - mesh.centroid
    normal_burr /= np.linalg.norm(normal_burr)

    r_burr = args.burr_radius
    dists_surf = np.linalg.norm(surf.nodes - burr_center, axis=1)
    hole_surf_mask = dists_surf <= r_burr
    hole_vol_nodes = surf.node_indices[hole_surf_mask]
    cranial_surf_nodes = surf.node_indices[~hole_surf_mask]
    skull_base_nodes = np.where(mesh.nodes[:, 1] < -20.0)[0]

    console.print(f"  • Brain Nodes / Elements : [green]{mesh.n_nodes:,}[/green] nodes | [green]{mesh.n_elements:,}[/green] Tet4 elements")
    console.print(f"  • Burr Hole Center       : ({burr_center[0]:.1f}, {burr_center[1]:.1f}, {burr_center[2]:.1f}) mm (Diameter: [bold green]{2*r_burr:.1f} mm[/bold green])")
    console.print(f"  • Burr Hole Aperture Nodes: [bold green]{len(hole_vol_nodes)}[/bold green] unconstrained boundary nodes")

    # Save bone collar and perimeter ring
    bone_collar, wireframe = create_burr_hole_geometry(burr_center, normal_burr, radius=r_burr)
    collar_path = base_out / "burr_hole_bone_collar.vtp"
    wireframe_path = base_out / "burr_hole_perimeter_ring.vtp"
    bone_collar.save(str(collar_path))
    wireframe.save(str(wireframe_path))

    # 2. Tissue & Fluid Mechanics
    console.print("\n[bold yellow][2/5] Configuring Physiological Parenchyma & Fluid Mechanics[/bold yellow]")
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=0.003,      # 3.0 kPa
        poissons_ratio=0.45,
        mass_density=1040e-12,     # tonne/mm³
    )
    mat = Hookean(params)
    elem = Tet4Linear(mat)

    # 3. Solver Execution Setup
    scenarios_to_run = []
    if args.all or args.scenario == "both":
        scenarios_to_run = ["bulging", "sagging"]
    else:
        scenarios_to_run = [args.scenario]

    solvers_to_run = []
    if args.all or args.solver == "both":
        solvers_to_run = ["dr", "static"]
    else:
        solvers_to_run = [args.solver]

    results_table: list[dict[str, str | float]] = []

    g_accel = 9810.0
    rho_csf = 1000e-12

    for scn in scenarios_to_run:
        console.print(f"\n[bold yellow]=== Processing Scenario: {scn.upper()} ===[/bold yellow]")
        dom = SolidDomain(mesh.copy(), mat, elem)

        # Boundary conditions
        # Intact cranial vault + skull base are rigid, burr hole aperture is released!
        constrained_nodes = np.unique(np.concatenate([cranial_surf_nodes, skull_base_nodes]))
        clamp = DirichletConstraint(constrained_nodes, components=(0, 1, 2), value=0.0)
        dm = DirichletManager(dom.mesh.n_dofs)
        dm.add_constraint(clamp)

        # Downward gravity (supine posture along -Y)
        gravity = GravityLoad(acceleration=(0.0, -g_accel, 0.0))
        M = dom.assemble_mass_matrix()
        f_gravity = gravity.compute_force_vector(M)

        if scn == "bulging":
            # Scenario 1: Elevated ICP on lesion cavity + full CSF buoyancy
            p_icp = args.icp_pressure * 1e-3  # kPa to MPa
            h_head_icp = p_icp / (rho_csf * g_accel)
            h_fluid_cyst = lesion_pv.bounds[3] + h_head_icp
            icp_load = HydrostaticPressureLoad(
                fluid_density=rho_csf,
                acceleration=(0.0, -g_accel, 0.0),
                fluid_level=h_fluid_cyst,
                surface_mesh=lesion_smesh,
            )
            f_icp = icp_load.compute_force_vector(total_dofs=dom.n_dofs)

            h_csf_intact = float(mesh.bounds[3])
            buoyancy = BuoyancyLoad(
                fluid_density=rho_csf,
                acceleration=(0.0, -g_accel, 0.0),
                fluid_level=h_csf_intact,
            )
            f_buoy = buoyancy.compute_force_vector(dom)

            f_total = f_gravity + f_buoy + f_icp
        else:
            # Scenario 2: Partial CSF drainage (fluid level drained to 0.0 mm)
            buoyancy_drain = BuoyancyLoad(
                fluid_density=1000e-12,
                acceleration=(0.0, -9810.0, 0.0),
                fluid_level=0.0,
            )
            f_buoy_drain = buoyancy_drain.compute_force_vector(dom)
            f_total = f_gravity + f_buoy_drain

        for slv in solvers_to_run:
            case_out = base_out / f"{scn}_{slv}"
            case_out.mkdir(parents=True, exist_ok=True)
            dom_run = SolidDomain(mesh.copy(), mat, elem)

            if slv == "dr":
                console.print(f"  • Running Matrix-Free Dynamic Relaxation ({scn})...")
                t0 = time.perf_counter()
                dr = DynamicRelaxationAnalysis(dom_run, damping_type="kinetic", dt=1.0)
                res = dr.solve(
                    dirichlet=dm,
                    f_ext=f_total,
                    tol=1e-2,
                    max_steps=args.max_steps,
                    save_directory=case_out,
                    output_name="result",
                    format="xdmf",
                    verbose=False,
                )
                t_sol = time.perf_counter() - t0
                disp = dom_run.displacement
                steps_txt = f"{res.steps_completed} DR steps"
            else:
                console.print(f"  • Running Direct Linear Static Solver ({scn})...")
                t0 = time.perf_counter()
                st = LinearStaticAnalysis(dom_run)
                res = st.solve(dirichlet=dm, f_ext=f_total)
                t_sol = time.perf_counter() - t0
                disp = dom_run.displacement
                steps_txt = "Direct Solve"
                dom_run.save_state(filepath=case_out / "result.vtu")

            # Outward normal projection at burr hole aperture
            hole_proj = np.dot(disp[hole_vol_nodes], normal_burr)
            if scn == "bulging":
                shift_val = float(hole_proj.max())
                shift_txt = f"+{shift_val:.3f} mm (Outward Prolapse)"
            else:
                shift_val = float(hole_proj.min())
                shift_txt = f"{shift_val:.3f} mm (Inward Collapse)"

            peak_disp = float(np.linalg.norm(disp, axis=1).max())

            results_table.append({
                "scenario": scn.capitalize(),
                "solver": slv.upper(),
                "steps": steps_txt,
                "burr_hole_shift": shift_txt,
                "peak_disp_mm": f"{peak_disp:.3f}",
                "runtime_s": f"{t_sol:.2f}",
                "output": str(case_out),
            })

    # Summary table
    table = Table(title="Example 0602: Burr Hole Brain Shift Summary", border_style="cyan")
    table.add_column("Scenario", style="bold white")
    table.add_column("Solver", style="cyan")
    table.add_column("Convergence", style="yellow")
    table.add_column("Burr Hole Aperture Shift", style="bold green")
    table.add_column("Peak Brain Sag (mm)", style="green")
    table.add_column("Runtime (s)", style="blue")

    for r in results_table:
        table.add_row(
            str(r["scenario"]),
            str(r["solver"]),
            str(r["steps"]),
            str(r["burr_hole_shift"]),
            str(r["peak_disp_mm"]),
            str(r["runtime_s"]),
        )
    console.print(table)
    console.print(f"\n[bold green]✓ Simulation completed. Results written to:[/bold green] [cyan]{base_out}[/cyan]\n")


if __name__ == "__main__":
    main()
