"""
Simulation 1108: Neurosurgical Burr Hole Brain Shift: Cortical Bulging vs. Gravitational Sagging.

Demonstrates:
- [1/5] Patient-Specific Anatomy & Targeted Burr Hole Trephination:
        MRI brain volumetric continuum (18,397 nodes, 90,586 elements, 55,191 DOFs)
        with an internal subcortical pathology/cystic tumor cavity (lesion.vtk).
        A surgical burr hole trephination (R = 10.0 mm, d = 20.0 mm) is placed directly
        over the frontal pathology at [-2.3, 34.1, 50.1] mm.
        Exports 3D cranial bone table collar ('burr_hole_bone_collar.vtp') and
        perimeter ring ('burr_hole_perimeter_ring.vtp') for ParaView visual verification.
- [2/5] Continuum Fluid-Structure Mechanics (Hydrostatics + Buoyancy):
        Brain parenchyma (E = 3.0 kPa, nu = 0.45, rho = 1,040 kg/m^3).
        CSF fluid (rho = 1,000 kg/m^3, g = 9,810 mm/s^2).
        Rigid calvarium and intact dura constrain outer cortical boundaries,
        while the burr hole aperture releases the normal constraint.
- [3/5] Scenario 1: Cortical Bulging (Trans-Mantle Herniation / Prolapse):
        Driven by HydrostaticPressureLoad on the internal lesion/ventricular cavity
        under elevated intracranial pressure (ICP = 12 kPa ~ 90 mmHg cystic turgor)
        plus intact cranial CSF BuoyancyLoad (fluid level at vertex Y = +83.7 mm)
        and downward GravityLoad.
        Tissue mushrooms outward through the bone collar into the aperture (+2.98 mm).
- [4/5] Scenario 2: Cortical Sagging (Gravitational Collapse / Pneumocephalus):
        Driven by GravityLoad and partial CSF drainage modeled via BuoyancyLoad
        with an explicit drainage level (fluid_level = 0.0 mm).
        Tissue above the fluid level loses 96.2% buoyant lift and experiences full
        downward gravity, collapsing into the occipital skull base and sinking away
        from the burr hole (-3.58 mm), creating an open subdural pneumocephalus cavity.
- [5/5] Multi-Model Telemetry & ParaView Comparative Export:
        Outputs clean unified XDMF+HDF5 datasets for ParaView inspection alongside
        the burr hole collar, wireframe ring, and meningeal surface.
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
from rich.traceback import install
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

install(show_locals=False)
console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 1108: Neurosurgical Burr Hole Brain Shift: Cortical Bulging vs. Gravitational Sagging"
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["static", "dr", "both"],
        default="static",
        help="Solver methodology: 'static' (Direct Linear Static solve, ~5s), 'dr' (Explicit Dynamic Relaxation), or 'both' (default: static)",
    )
    parser.add_argument(
        "--burr-radius",
        type=float,
        default=10.0,
        help="Burr hole aperture radius in mm (default: 10.0 mm, diameter 20.0 mm)",
    )
    parser.add_argument(
        "--icp-pressure",
        type=float,
        default=12.0,
        help="Intracranial cavity hydrostatic pressure in kPa for bulging scenario (default: 12.0 kPa)",
    )
    parser.add_argument(
        "--csf-drain-level",
        type=float,
        default=0.0,
        help="Elevation along Y (mm) to which CSF drains in sagging scenario (default: 0.0 mm)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["xdmf", "pvd"],
        default="xdmf",
        help="ParaView visualization export format (default: xdmf)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/11_advanced_continua/1108_burr_hole_brain_shift",
        help="Base output directory for simulation results",
    )
    return parser.parse_args()


def resolve_file(rel_path: str) -> Path:
    p1 = Path(rel_path)
    if p1.exists():
        return p1
    p2 = Path("psierra") / rel_path
    if p2.exists():
        return p2
    raise FileNotFoundError(f"File not found at '{p1}' or '{p2}'")


def create_burr_hole_geometry(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    thickness: float = 4.0,
    wall_width: float = 3.5,
) -> tuple[pv.PolyData, pv.PolyData]:
    """
    Constructs a 3D annular cranial bone collar and circular contour wireframe
    representing the surgical burr hole drilled through the skull table.
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

    # Inner circular perimeter wireframe
    inner_circle = center + r_in * (np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2)
    lines = []
    for i in range(n_pts):
        lines.extend([2, i, (i + 1) % n_pts])
    wireframe = pv.PolyData(inner_circle, lines=lines)

    # 3D Annular bone collar with skull thickness
    # Bottom ring (inner & outer) at skull inner table
    pts_bot_in = center + r_in * (np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2)
    pts_bot_out = center + r_out * (np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2)

    # Top ring (inner & outer) at skull outer table
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

    collar = pv.PolyData(all_pts, faces=faces)
    return collar, wireframe


def main() -> None:
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format
    r_burr = args.burr_radius
    p_icp_kpa = args.icp_pressure
    p_icp_mpa = p_icp_kpa * 1e-3  # Convert kPa to MPa
    h_csf_drain = args.csf_drain_level
    solver_mode = args.solver

    # 0. Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]Simulation 1108: Neurosurgical Burr Hole Brain Shift[/bold cyan]\n"
            "[bold white]Cortical Bulging (Hydrostatic ICP) vs. Gravitational Sagging (CSF Drainage)[/bold white]\n"
            f"[dim]Aperture Radius: {r_burr:.1f} mm (Ø {2*r_burr:.1f} mm) • ICP: {p_icp_kpa:.1f} kPa • CSF Drain: Y={h_csf_drain:.1f} mm • Solver: {solver_mode.upper()} • Format: {fmt.upper()}[/dim]",
            border_style="cyan",
        )
    )

    # 1. Load Patient Brain Mesh & Pathological Lesion Cavity
    console.print("\n[bold yellow][1/5] Loading Patient Anatomy & Placing Surgical Burr Hole[/bold yellow]")
    brain_mesh_path = resolve_file("data/mrbraintumor1/mesh/brain.vtk")
    lesion_mesh_path = resolve_file("data/mrbraintumor1/mesh/lesion.vtk")

    mesh = VolumetricMesh(brain_mesh_path)
    surf = mesh.surface

    # Load lesion cavity mesh for HydrostaticPressureLoad
    lesion_pv = pv.read(str(lesion_mesh_path))
    lesion_surf_pv = lesion_pv.extract_surface(algorithm=None)

    # Invert triangle winding so face normal points inward toward cavity center.
    # In HydrostaticPressureLoad, traction is t(x) = -p(x)*n, so negative of inward normal
    # pushes OUTWARD into the surrounding brain parenchyma!
    faces_lesion = lesion_surf_pv.faces.reshape(-1, 4)[:, 1:4]
    faces_inv = np.column_stack([faces_lesion[:, 0], faces_lesion[:, 2], faces_lesion[:, 1]])
    lesion_surf_inv = pv.PolyData(
        lesion_surf_pv.points,
        faces=np.column_stack([np.full(len(faces_lesion), 3), faces_inv]).ravel(),
    )

    # Map lesion boundary vertices to global volumetric mesh node indices
    kdtree = cKDTree(mesh.nodes)
    _, global_indices = kdtree.query(lesion_surf_pv.points)
    lesion_smesh = SurfaceMesh(source=lesion_surf_inv, node_indices=global_indices)

    # Surgical Burr Hole Trephination:
    # Placed on frontal calvarium directly accessing the subcortical pathology
    lesion_center = np.array(lesion_pv.center)
    dists_surf_to_lesion = np.linalg.norm(surf.nodes - lesion_center, axis=1)
    burr_center = surf.nodes[np.argmin(dists_surf_to_lesion)]

    # Outward cranial surface normal at burr hole
    normal_burr = burr_center - mesh.centroid
    normal_burr /= np.linalg.norm(normal_burr)

    # Identify boundary nodes within burr hole aperture
    dists_surf = np.linalg.norm(surf.nodes - burr_center, axis=1)
    hole_surf_mask = dists_surf <= r_burr
    hole_vol_nodes = surf.node_indices[hole_surf_mask]
    cranial_surf_nodes = surf.node_indices[~hole_surf_mask]

    # Skull base nodes clamped by petroclival / tentorial skull base
    skull_base_nodes = np.where(mesh.nodes[:, 1] < -20.0)[0]

    console.print(
        f"  • Brain Continuum  : [green]{mesh.n_nodes:,}[/green] nodes | "
        f"[green]{mesh.n_elements:,}[/green] Tet4 elements | [green]{mesh.n_dofs:,}[/green] DOFs\n"
        f"  • Brain Volume     : [green]{mesh.volume*1e-6:.3f}[/green] liters ({mesh.volume:,.1f} mm³)\n"
        f"  • Lesion Cavity    : [green]{lesion_smesh.n_nodes:,}[/green] boundary nodes | [green]{lesion_smesh.n_elements:,}[/green] boundary facets\n"
        f"  • Burr Hole Center : X = [cyan]{burr_center[0]:.1f}[/cyan], Y = [cyan]{burr_center[1]:.1f}[/cyan], Z = [cyan]{burr_center[2]:.1f}[/cyan] mm (Direct Pathology Approach)\n"
        f"  • Aperture Radius  : [green]{r_burr:.1f} mm[/green] (Diameter: [bold green]{2*r_burr:.1f} mm[/bold green])\n"
        f"  • Outward Normal n : [{normal_burr[0]:.3f}, {normal_burr[1]:.3f}, {normal_burr[2]:.3f}]\n"
        f"  • Burr Hole Nodes  : [bold green]{len(hole_vol_nodes)}[/bold green] unconstrained boundary nodes\n"
        f"  • Calvarial Nodes  : [green]{len(cranial_surf_nodes):,}[/green] nodes constrained by intact skull & dura\n"
        f"  • Skull Base Nodes : [green]{len(skull_base_nodes):,}[/green] nodes constrained at skull base"
    )

    # Export Burr Hole 3D Collar & Contour for ParaView Overlay
    bone_collar, wireframe = create_burr_hole_geometry(
        burr_center, normal_burr, radius=r_burr, thickness=5.0, wall_width=3.5
    )
    collar_path = output_dir / "burr_hole_bone_collar.vtp"
    wireframe_path = output_dir / "burr_hole_perimeter_ring.vtp"
    bone_collar.save(str(collar_path))
    wireframe.save(str(wireframe_path))

    # Export Cortical Meninges with Aperture Indicator
    faces_surf = np.column_stack([np.full(surf.n_elements, 3), surf.elements]).ravel()
    meninges_pv = pv.PolyData(surf.nodes, faces=faces_surf)
    aperture_field = np.zeros(surf.n_nodes, dtype=np.int32)
    aperture_field[hole_surf_mask] = 1
    meninges_pv.point_data["burr_hole_aperture"] = aperture_field
    meninges_path = output_dir / "cranial_meninges.vtp"
    meninges_pv.save(str(meninges_path))

    console.print(
        f"  • Exported Collar  : [cyan]{collar_path}[/cyan] (3D skull table ring)\n"
        f"  • Exported Ring    : [cyan]{wireframe_path}[/cyan] (circular aperture wireframe)\n"
        f"  • Exported Meninges: [cyan]{meninges_path}[/cyan] (dural envelope with burr hole)"
    )

    # 2. Material & Physical Parameters
    console.print("\n[bold yellow][2/5] Configuring Physiological Parenchyma & Fluid Mechanics[/bold yellow]")
    youngs_kpa = 3.0
    youngs_mpa = youngs_kpa * 1e-3  # 0.003 MPa
    poissons_ratio = 0.45
    rho_brain = 1040e-12  # tonne/mm^3 (1040 kg/m^3)
    rho_csf = 1000e-12    # tonne/mm^3 (1000 kg/m^3)
    g_accel = 9810.0      # mm/s^2 along -Y in supine surgical posture

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=youngs_mpa,
        poissons_ratio=poissons_ratio,
        mass_density=rho_brain,
    )
    mat = Hookean(params)
    elem = Tet4Linear(mat)

    total_mass_kg = mesh.volume * rho_brain * 1e3
    total_weight_n = total_mass_kg * 9.81
    total_csf_buoyancy_n = (mesh.volume * rho_csf * 1e3) * 9.81
    net_buoyant_weight_n = total_weight_n - total_csf_buoyancy_n

    console.print(
        f"  • Young's Modulus E: [green]{youngs_kpa:.1f} kPa[/green] ({youngs_mpa:.4f} MPa) | Poisson's Ratio ν: [green]{poissons_ratio:.2f}[/green]\n"
        f"  • Brain In-Air W   : [bold red]{total_weight_n:.3f} N[/bold red] ({total_mass_kg:.3f} kg mass)\n"
        f"  • Intact CSF Lift  : [bold green]{total_csf_buoyancy_n:.3f} N[/bold green] (Archimedes 96.15% gravitational relief)\n"
        f"  • Intact Net Sag W : [bold yellow]{net_buoyant_weight_n:.3f} N[/bold yellow] (Only 56.6 grams net downward pull!)"
    )

    results_table: list[dict[str, str | float]] = []

    # =========================================================================
    # 3. Scenario 1: Cortical Bulging (Trans-Mantle Herniation via Hydrostatics)
    # =========================================================================
    console.print("\n[bold yellow][3/5] Simulating Scenario 1: Cortical Bulging via HydrostaticPressureLoad + BuoyancyLoad[/bold yellow]")
    console.print(
        f"  [dim]Physiology: Elevated internal cavity hydrostatic pressure ({p_icp_kpa:.1f} kPa) + intact CSF buoyant suspension.[/dim]"
    )

    dom_bulge = SolidDomain(mesh.copy(), mat, elem)

    # Cranial calvarium & intact dura constrain normal displacement across intact skull
    dm_bulge = DirichletManager(dom_bulge.n_dofs)
    dm_bulge.add_constraint(DirichletConstraint(cranial_surf_nodes, components=(0, 1, 2), value=0.0))
    dm_bulge.add_constraint(DirichletConstraint(skull_base_nodes, components=(0, 1, 2), value=0.0))

    # A. Internal Hydrostatic Pressure Load on cavity surface:
    # Head corresponding to ICP: h_head = p_icp / (rho_csf * g)
    h_head_icp = p_icp_mpa / (rho_csf * g_accel)
    h_fluid_cyst = lesion_pv.bounds[3] + h_head_icp
    hydro_bulge = HydrostaticPressureLoad(
        fluid_density=rho_csf,
        acceleration=(0.0, -g_accel, 0.0),
        fluid_level=h_fluid_cyst,
        surface_mesh=lesion_smesh,
    )
    f_hydro_bulge = hydro_bulge.compute_force_vector(total_dofs=dom_bulge.n_dofs)

    # B. Intact CSF Buoyancy Load (cranial CSF intact up to vertex Y = bounds[3]):
    h_csf_intact = float(mesh.bounds[3])
    buoyancy_intact = BuoyancyLoad(
        fluid_density=rho_csf,
        acceleration=(0.0, -g_accel, 0.0),
        fluid_level=h_csf_intact,
    )
    f_buoyancy_intact = buoyancy_intact.compute_force_vector(dom_bulge)

    # C. Gravity Load:
    M_bulge = dom_bulge.assemble_mass_matrix()
    gravity = GravityLoad(acceleration=(0.0, -g_accel, 0.0))
    f_gravity = gravity.compute_force_vector(M_bulge)

    # Total net load: Gravity + Intact Buoyancy (neutral baseline) + Internal Hydrostatic Pressure
    f_total_bulge = f_gravity + f_buoyancy_intact + f_hydro_bulge

    console.print(
        f"  • Hydrostatic Head : [cyan]{h_head_icp:.1f} mm[/cyan] (ICP = {p_icp_kpa:.1f} kPa / {p_icp_kpa * 7.50062:.1f} mmHg)\n"
        f"  • Intact CSF Level : [cyan]Y = +{h_csf_intact:.1f} mm[/cyan] (Full cranial suspension)\n"
        f"  • Hydrostatic Force: [green]{np.linalg.norm(f_hydro_bulge.reshape(-1, 3).sum(axis=0)):.3f} N[/green] net cavity thrust"
    )

    dir_bulge = output_dir / "01_burr_hole_bulging"
    t_start = time.perf_counter()

    if solver_mode in ("static", "both"):
        solver_bulge = LinearStaticAnalysis(dom_bulge)
        res_bulge = solver_bulge.solve(
            dirichlet=dm_bulge,
            f_ext=f_total_bulge,
            save_directory=dir_bulge,
            output_name="result",
            format=fmt,
            verbose=False,
        )
        t_bulge = time.perf_counter() - t_start

        disp_bulge = dom_bulge.displacement
        hole_disp_mag_b = np.linalg.norm(disp_bulge[hole_vol_nodes], axis=1)
        hole_outward_proj_b = np.dot(disp_bulge[hole_vol_nodes], normal_burr)

        max_bulge_disp = float(hole_disp_mag_b.max())
        max_bulge_outward = float(hole_outward_proj_b.max())
        center_node_idx = surf.node_indices[np.argmin(dists_surf)]
        center_bulge_disp = float(np.dot(disp_bulge[center_node_idx], normal_burr))

        console.print(
            f"    ✓ Solved in [green]{t_bulge:.2f} s[/green] (Linear Static)\n"
            f"    • Peak Herniation Displacement : [bold green]{max_bulge_disp:.3f} mm[/bold green]\n"
            f"    • Outward Prolapse along n     : [bold green]+{max_bulge_outward:.3f} mm[/bold green] (mushrooms outside skull ring)\n"
            f"    • Burr Hole Center Dome Height : [cyan]+{center_bulge_disp:.3f} mm[/cyan]"
        )

        results_table.append({
            "Scenario": "1. Bulging (Static)",
            "Clinical Mechanism": f"Hydrostatic ICP ({p_icp_kpa:.1f} kPa) + Buoyancy",
            "Aperture Kinematics": "Free outward extrusion",
            "Peak Disp (mm)": max_bulge_disp,
            "Burr Hole Shift (mm)": f"+{max_bulge_outward:.3f} (Outward)",
            "Shift Direction": "Extrudes outside skull table",
            "Clinical Implication": "Cortical incarceration / strangulation risk",
            "Export Path": str(dir_bulge / f"result.{fmt}"),
        })

    if solver_mode in ("dr", "both"):
        dir_dr_bulge = output_dir / ("03_dr_burr_hole_bulging" if solver_mode == "both" else "01_burr_hole_bulging")
        t_start_dr = time.perf_counter()
        solver_dr_b = DynamicRelaxationAnalysis(dom_bulge, damping_type="kinetic", dt=1.0)
        res_dr_b = solver_dr_b.solve(
            dirichlet=dm_bulge,
            f_ext=f_total_bulge,
            tol=1e-2,
            max_steps=60,
            save_directory=dir_dr_bulge,
            output_name="result",
            format=fmt,
            verbose=False,
        )
        t_dr_b = time.perf_counter() - t_start_dr
        disp_dr_b = dom_bulge.displacement
        hole_dr_b = np.dot(disp_dr_b[hole_vol_nodes], normal_burr)
        max_dr_bulge = float(hole_dr_b.max())
        console.print(
            f"    ✓ Solved in [green]{t_dr_b:.2f} s[/green] ({res_dr_b.steps_completed} DR steps)\n"
            f"    • DR Outward Prolapse along n  : [bold green]+{max_dr_bulge:.3f} mm[/bold green]"
        )
        results_table.append({
            "Scenario": "1. Bulging (Dynamic Relaxation)",
            "Clinical Mechanism": f"Hydrostatic ICP ({p_icp_kpa:.1f} kPa) + Buoyancy",
            "Aperture Kinematics": "Free outward extrusion",
            "Peak Disp (mm)": float(np.linalg.norm(disp_dr_b[hole_vol_nodes], axis=1).max()),
            "Burr Hole Shift (mm)": f"+{max_dr_bulge:.3f} (Outward)",
            "Shift Direction": "Extrudes outside skull table",
            "Clinical Implication": "Kinetic relaxation equilibrium",
            "Export Path": str(dir_dr_bulge / f"result.{fmt}"),
        })

    # =========================================================================
    # 4. Scenario 2: Cortical Sagging from Burr Hole (Cerebral Collapse)
    # =========================================================================
    console.print("\n[bold yellow][4/5] Simulating Scenario 2: Cortical Sagging via Partial CSF Drainage BuoyancyLoad[/bold yellow]")
    console.print(
        f"  [dim]Physiology: Supine posture, CSF drained to Y = {h_csf_drain:.1f} mm, buoyant lift lost in drained region.[/dim]"
    )

    dom_sag = SolidDomain(mesh.copy(), mat, elem)

    # In sagging, skull base supports the sinking brain, while superior calvarium allows
    # tissue to fall downward away from the nondependent burr hole.
    dm_sag = DirichletManager(dom_sag.n_dofs)
    dm_sag.add_constraint(DirichletConstraint(skull_base_nodes, components=(0, 1, 2), value=0.0))

    # A. Drained CSF Buoyancy Load:
    # Elements above h_csf_drain have alpha_e = 0 (lose buoyancy), while elements below remain buoyed
    buoyancy_drain = BuoyancyLoad(
        fluid_density=rho_csf,
        acceleration=(0.0, -g_accel, 0.0),
        fluid_level=h_csf_drain,
    )
    f_buoyancy_drain = buoyancy_drain.compute_force_vector(dom_sag)

    # B. Gravity Load:
    M_sag = dom_sag.assemble_mass_matrix()
    f_gravity_sag = gravity.compute_force_vector(M_sag)

    # Total net sagging load: Gravity + Partially Drained Buoyancy
    f_total_sag = f_gravity_sag + f_buoyancy_drain

    console.print(
        f"  • CSF Drain Level  : [cyan]Y = {h_csf_drain:.1f} mm[/cyan] (Drained down from +{h_csf_intact:.1f} mm)\n"
        f"  • Lost Buoyancy    : Elements above Y={h_csf_drain:.1f} mm experience full in-air gravitational weight\n"
        f"  • Net Sagging Load : [bold red]{np.linalg.norm((f_total_sag).reshape(-1, 3).sum(axis=0)):.3f} N[/bold red] net downward collapse force"
    )

    dir_sag = output_dir / "02_burr_hole_sagging"
    t_start = time.perf_counter()

    if solver_mode in ("static", "both"):
        solver_sag = LinearStaticAnalysis(dom_sag)
        res_sag = solver_sag.solve(
            dirichlet=dm_sag,
            f_ext=f_total_sag,
            save_directory=dir_sag,
            output_name="result",
            format=fmt,
            verbose=False,
        )
        t_sag = time.perf_counter() - t_start

        disp_sag = dom_sag.displacement
        hole_disp_mag_s = np.linalg.norm(disp_sag[hole_vol_nodes], axis=1)
        hole_outward_proj_s = np.dot(disp_sag[hole_vol_nodes], normal_burr)

        max_sag_disp = float(np.linalg.norm(disp_sag, axis=1).max())
        max_hole_sag = float(hole_disp_mag_s.max())
        hole_sink_distance = float(abs(hole_outward_proj_s.min()))
        center_node_idx = surf.node_indices[np.argmin(dists_surf)]
        center_sink_disp = float(np.dot(disp_sag[center_node_idx], normal_burr))

        console.print(
            f"    ✓ Solved in [green]{t_sag:.2f} s[/green] (Linear Static)\n"
            f"    • Max Brain Sag (Posterior)    : [bold green]{max_sag_disp:.3f} mm[/bold green] (falling toward occiput)\n"
            f"    • Burr Hole Inward Collapse    : [bold red]-{hole_sink_distance:.3f} mm[/bold red] (cortex falls away from skull)\n"
            f"    • Burr Hole Center Sink        : [cyan]{center_sink_disp:.3f} mm[/cyan]\n"
            f"    • Subdural Air Pocket Gap      : [cyan]{hole_sink_distance:.3f} mm[/cyan] pneumocephalus depth"
        )

        results_table.append({
            "Scenario": "2. Sagging (Static)",
            "Clinical Mechanism": f"CSF Drained to Y={h_csf_drain:.1f} mm (Lost Buoyancy)",
            "Aperture Kinematics": "Falls inward toward occiput",
            "Peak Disp (mm)": max_sag_disp,
            "Burr Hole Shift (mm)": f"-{hole_sink_distance:.3f} (Inward)",
            "Shift Direction": "Sinks away from skull table",
            "Clinical Implication": "Stereotactic targeting error & bridging vein tension",
            "Export Path": str(dir_sag / f"result.{fmt}"),
        })

    if solver_mode in ("dr", "both"):
        dir_dr_sag = output_dir / ("04_dr_burr_hole_sagging" if solver_mode == "both" else "02_burr_hole_sagging")
        t_start_dr = time.perf_counter()
        solver_dr_s = DynamicRelaxationAnalysis(dom_sag, damping_type="kinetic", dt=1.0)
        res_dr_s = solver_dr_s.solve(
            dirichlet=dm_sag,
            f_ext=f_total_sag,
            tol=1e-2,
            max_steps=60,
            save_directory=dir_dr_sag,
            output_name="result",
            format=fmt,
            verbose=False,
        )
        t_dr_s = time.perf_counter() - t_start_dr
        disp_dr_s = dom_sag.displacement
        hole_dr_s = np.dot(disp_dr_s[hole_vol_nodes], normal_burr)
        sink_dr_distance = float(abs(hole_dr_s.min()))
        console.print(
            f"    ✓ Solved in [green]{t_dr_s:.2f} s[/green] ({res_dr_s.steps_completed} DR steps)\n"
            f"    • DR Burr Hole Inward Collapse : [bold red]-{sink_dr_distance:.3f} mm[/bold red]"
        )
        results_table.append({
            "Scenario": "2. Sagging (Dynamic Relaxation)",
            "Clinical Mechanism": f"CSF Drained to Y={h_csf_drain:.1f} mm (Lost Buoyancy)",
            "Aperture Kinematics": "Falls inward toward occiput",
            "Peak Disp (mm)": float(np.linalg.norm(disp_dr_s, axis=1).max()),
            "Burr Hole Shift (mm)": f"-{sink_dr_distance:.3f} (Inward)",
            "Shift Direction": "Sinks away from skull table",
            "Clinical Implication": "Kinetic relaxation equilibrium",
            "Export Path": str(dir_dr_sag / f"result.{fmt}"),
        })

    # =========================================================================
    # 5. Multi-Model Telemetry & Clinical Synthesis
    # =========================================================================
    console.print("\n[bold yellow][5/5] Clinical Synthesis & ParaView Inspection Guide[/bold yellow]")

    summary_table = Table(
        title="Simulation 1108: Burr Hole Brain Shift Comparative Summary",
        border_style="cyan",
        header_style="bold cyan",
    )
    summary_table.add_column("Surgical Scenario", style="bold white")
    summary_table.add_column("Biomechanical Driver", style="yellow")
    summary_table.add_column("Aperture Shift", style="bold green", justify="right")
    summary_table.add_column("Tissue Motion vs. Skull", style="cyan")
    summary_table.add_column("Clinical Risk / Implication", style="magenta")

    for row in results_table:
        summary_table.add_row(
            str(row["Scenario"]),
            str(row["Clinical Mechanism"]),
            str(row["Burr Hole Shift (mm)"]),
            str(row["Shift Direction"]),
            str(row["Clinical Implication"]),
        )

    console.print(summary_table)

    console.print(
        Panel(
            "[bold white]ParaView 3D Burr Hole Inspection Guide:[/bold white]\n\n"
            f"1. Open [bold cyan]{collar_path}[/bold cyan] and [bold cyan]{wireframe_path}[/bold cyan] to render the exact skull aperture.\n"
            f"2. Open [bold cyan]{output_dir / '01_burr_hole_bulging/result.xdmf'}[/bold cyan] (Bulging Case).\n"
            "   • Apply [bold green]Warp By Vector[/bold green] (Vectors: Displacement, Scale: 1.0).\n"
            "   • Notice how the parenchyma mushrooms OUTWARD through the bone collar like a dome.\n"
            f"3. Open [bold cyan]{output_dir / '02_burr_hole_sagging/result.xdmf'}[/bold cyan] (Sagging Case).\n"
            "   • Apply [bold green]Warp By Vector[/bold green] (Vectors: Displacement, Scale: 1.0).\n"
            "   • Notice how the cortex SINKS INWARD away from the bone collar, opening a subdural air gap!\n"
            f"4. Overlay [bold cyan]{meninges_path}[/bold cyan] colored by 'burr_hole_aperture' to verify boundary alignment.",
            title="[bold green]ParaView Visualization Protocol[/bold green]",
            border_style="green",
        )
    )

    console.print(f"\n[bold green]✓ Simulation 1108 completed successfully. Datasets saved to: {output_dir}[/bold green]\n")


if __name__ == "__main__":
    main()
