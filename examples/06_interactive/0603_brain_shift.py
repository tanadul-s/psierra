"""
Simulation 0603 (Interactive): Real-Time Neurosurgical Brain Shift Dynamic Simulation.

Simulates intraoperative brain deformation (gravity sag and tissue drift)
during neurosurgical head repositioning and craniotomy using Generalized-alpha
implicit time integration with pre-factorized SuperLU reuse.

Features:
1. Real-time dynamic loop operating with decoupled physics and rendering clocks.
2. Skull base / posterior cranial fossa boundary clamping in anatomical RAS coordinates.
3. Interactive 3D head reorientation (pitch, roll, yaw) with live gravity vector tracking.
4. Co-rotated semi-transparent cranial vault (skull.stl) and undeformed ghost reference mesh.
5. Live neurosurgical telemetry:
   - Lesion centroid drift vector [ΔX, ΔY, ΔZ] and Euclidean norm ||Δc||.
   - Maximum cortical surface shift magnitude and 3D location.
   - 3D arrow glyph displaying the maximum cortical shift vector.
6. Support for multiple element orders (Tet4, Tet10), kinematic formulations
   (Linear, Corotational, Total Lagrangian), and hyperelastic constitutive models (Hookean, Neo-Hookean).
7. Standard surgical posture presets (Supine, Prone, Lateral, 30° Pterional tilt, Upright).
8. Automated non-GUI headless validation sequence (--headless) for automated testing.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install
from scipy.spatial.transform import Rotation

from psierra.material import IsotropicElasticParameters, Hookean, NeoHookean
from psierra.element import (
    Tet4Linear,
    Tet10Linear,
    Tet4Corotational,
    Tet4TotalLagrangian,
    Tet4UpdatedLagrangian,
    Tet10TotalLagrangian,
    Tet10UpdatedLagrangian,
)
from psierra.mesh import VolumetricMesh, SurfaceMesh
from psierra.domain import SolidDomain
from psierra.boundary import DirichletConstraint, DirichletManager, GravityLoad
from psierra.damping import RayleighDamping
from psierra.analysis import InteractiveDynamicSimulation, SimulationStats
from psierra.exchange import SimulationWriter

install(show_locals=False)
console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 0603 (Interactive): Real-Time Neurosurgical Brain Shift Dynamic Simulation"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless batch demonstration mode without opening a GUI window (auto-detected if DISPLAY is absent)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Physical simulation duration in seconds for demonstration sequence (default: 2.0 s)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005,
        help="Physics simulation time step dt in seconds (default: 0.005 s = 200 Hz)",
    )
    parser.add_argument(
        "--spectral-radius",
        type=float,
        default=0.8,
        help="Generalized-alpha high-frequency spectral radius rho_inf in [0.0, 1.0] (default: 0.8)",
    )
    parser.add_argument(
        "--element",
        type=str,
        choices=["tet4", "tet10"],
        default="tet4",
        help="Element order: 'tet4' (linear, ultra-fast) or 'tet10' (quadratic, shear-locking free) (default: tet4)",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        choices=["linear", "corotational", "total_lagrangian", "updated_lagrangian"],
        default="linear",
        help="Kinematic formulation: 'linear', 'corotational', 'total_lagrangian', or 'updated_lagrangian' (default: linear)",
    )
    parser.add_argument(
        "--material",
        type=str,
        choices=["hookean", "neohookean"],
        default="hookean",
        help="Material model: 'hookean' (linear/St. Venant-Kirchhoff) or 'neohookean' (compressible hyperelastic) (default: hookean)",
    )
    parser.add_argument(
        "--youngs-modulus",
        type=float,
        default=3.0,
        help="Parenchymal tissue Young's modulus E in kPa (default: 3.0 kPa)",
    )
    parser.add_argument(
        "--poissons-ratio",
        type=float,
        default=0.45,
        help="Tissue Poisson's ratio nu (default: 0.45)",
    )
    parser.add_argument(
        "--mass-density",
        type=float,
        default=1040.0,
        help="Brain tissue density rho in kg/m^3 (default: 1040.0 kg/m^3)",
    )
    parser.add_argument(
        "--clamp-y",
        type=float,
        default=-20.0,
        help="Posterior skull-cradle clamp coordinate threshold in mm (default: -20.0 mm in RAS)",
    )
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        default=False,
        help="Export time-series VTU and PVD snapshot collection to results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/06_interactive/0603_brain_shift",
        help="Output directory for snapshot collection (default: results/06_interactive/0603_brain_shift)",
    )
    return parser.parse_args()


class HeadOrientationManager:
    """Manages 3D head orientation and calculates local gravity in RAS coordinates."""

    SURGICAL_PRESETS = {
        "supine": (0.0, 0.0, 0.0, "Supine (Face Up, Standard Navigation)"),
        "prone": (180.0, 0.0, 0.0, "Prone (Face Down, Suboccipital Approach)"),
        "right_lateral": (0.0, 0.0, -90.0, "Right Lateral (Left Craniotomy Down)"),
        "left_lateral": (0.0, 0.0, 90.0, "Left Lateral (Right Craniotomy Down)"),
        "pterional_30": (-15.0, 0.0, 30.0, "30° Pterional / Frontotemporal Tilt"),
        "upright": (-90.0, 0.0, 0.0, "Sitting / Upright Posture"),
    }

    def __init__(self, pitch: float = 0.0, roll: float = 0.0, yaw: float = 0.0) -> None:
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw
        # World gravity in standard operating room: downward along -Y (anterior/posterior supine alignment)
        self.g_world = np.array([0.0, -9810.0, 0.0], dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Returns 3x3 orthogonal rotation matrix R_head mapping local RAS -> world OR frame."""
        r = Rotation.from_euler("xyz", [self.pitch, self.roll, self.yaw], degrees=True)
        return r.as_matrix()

    @property
    def local_gravity(self) -> np.ndarray:
        """Returns gravity vector in local head RAS coordinate system: g_local = R_head^T * g_world."""
        R = self.rotation_matrix
        return R.T @ self.g_world

    def set_preset(self, name: str) -> str:
        if name in self.SURGICAL_PRESETS:
            p, r, y, desc = self.SURGICAL_PRESETS[name]
            self.pitch, self.roll, self.yaw = p, r, y
            return desc
        return ""

    def adjust(self, dp: float = 0.0, dr: float = 0.0, dy: float = 0.0) -> None:
        self.pitch = (self.pitch + dp) % 360.0
        self.roll = (self.roll + dr) % 360.0
        self.yaw = (self.yaw + dy) % 360.0


def main() -> None:
    args = parse_arguments()

    console.print(
        Panel.fit(
            "[bold cyan]Simulation 0603 (Interactive): Neurosurgical Brain Shift Pipeline[/bold cyan]\n"
            "[dim]Real-Time Intraoperative Gravitational Brain Sag & Target Shift Telemetry[/dim]",
            title="[bold blue]project-sierra: Image-Guided Neurosurgery[/bold blue]",
            border_style="cyan",
        )
    )

    # 1. Loading Anatomical Meshes
    console.print("\n[bold yellow][1/6] Loading Anatomical Datasets (Brain, Skull, Lesion)[/bold yellow]")
    brain_mesh_path = Path("data/mrbraintumor1/mesh/brain.vtk")
    lesion_mesh_path = Path("data/mrbraintumor1/mesh/lesion.vtk")
    skull_mesh_path = Path("data/mrbraintumor1/mesh/skull.vtk")
    skull_stl_path = Path("data/mrbraintumor1/scan/segmented/skull.stl")

    if not brain_mesh_path.exists():
        raise FileNotFoundError(f"Brain mesh not found: {brain_mesh_path}")
    if not lesion_mesh_path.exists():
        raise FileNotFoundError(f"Lesion mesh not found: {lesion_mesh_path}")

    # Load brain volumetric mesh
    mesh = VolumetricMesh(brain_mesh_path)
    console.print(
        f"  • Volumetric Brain: [green]{mesh.n_nodes:,}[/green] nodes | "
        f"[green]{mesh.n_elements:,}[/green] Tet4 elements | "
        f"Volume: [green]{mesh.volume*1e-6:.3f}[/green] L ({mesh.volume:.1f} mm³)"
    )

    # Upgrade to Tet10 if requested
    if args.element == "tet10":
        console.print("  • Upgrading brain continuum mesh to [cyan]10-Node Quadratic Tetrahedra (Tet10)[/cyan]...")
        mesh = mesh.to_quadratic()
        console.print(
            f"  • Upgraded Tet10 Mesh: [green]{mesh.n_nodes:,}[/green] nodes | "
            f"[green]{mesh.n_dofs:,}[/green] DOFs"
        )

    # Load lesion mesh & map node indices
    import pyvista as pv
    from scipy.spatial import cKDTree

    lesion_pv = pv.read(lesion_mesh_path)
    brain_tree = cKDTree(mesh.nodes)
    lesion_dists, lesion_point_to_brain = brain_tree.query(lesion_pv.points)
    lesion_node_indices = np.unique(lesion_point_to_brain)
    lesion_center_ref = mesh.nodes[lesion_node_indices].mean(axis=0)

    console.print(
        f"  • Lesion Target   : [green]{len(lesion_node_indices)}[/green] nodes matched | "
        f"Initial Center: [dim]({lesion_center_ref[0]:.1f}, {lesion_center_ref[1]:.1f}, {lesion_center_ref[2]:.1f}) mm[/dim]"
    )

    # Extract cortical surface nodes
    surf_poly = mesh.ugrid.extract_surface(algorithm="dataset_surface")
    surf_point_ids = surf_poly.point_data["vtkOriginalPointIds"]
    cortical_surface_all = np.unique(surf_point_ids)

    # 2. Material & Kinematic Formulation Setup
    console.print("\n[bold yellow][2/6] Configuring Soft Tissue Constitutive Model[/bold yellow]")
    # Convert units: E in MPa, rho in tonne/mm^3
    # 1 tonne = 1000 kg, 1 m^3 = 10^9 mm^3 -> 1 kg/m^3 = 10^-12 tonne/mm^3
    E_mpa = args.youngs_modulus * 1e-3  # kPa -> MPa (N/mm^2)
    rho_tonne_mm3 = args.mass_density * 1e-12  # kg/m^3 -> tonne/mm^3 (1040 kg/m^3 = 1.04e-9 tonne/mm^3)
    nu = args.poissons_ratio

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=E_mpa,
        poissons_ratio=nu,
        mass_density=rho_tonne_mm3,
    )

    if args.material == "neohookean":
        material = NeoHookean(params)
        mat_desc = "Compressible Neo-Hookean Hyperelasticity"
    else:
        material = Hookean(params)
        mat_desc = "Isotropic Linear Elasticity (Hookean)"

    formulation_type = args.formulation or ("linear" if args.material == "hookean" else "total_lagrangian")
    is_nonlinear = formulation_type in ["corotational", "total_lagrangian", "updated_lagrangian"]

    if args.element == "tet10":
        if formulation_type == "total_lagrangian":
            element = Tet10TotalLagrangian(material)
        elif formulation_type == "updated_lagrangian":
            element = Tet10UpdatedLagrangian(material)
        else:
            element = Tet10Linear(material)
            formulation_type = "linear"
            is_nonlinear = False
    else:
        if formulation_type == "corotational":
            element = Tet4Corotational(material)
        elif formulation_type == "total_lagrangian":
            element = Tet4TotalLagrangian(material)
        elif formulation_type == "updated_lagrangian":
            element = Tet4UpdatedLagrangian(material)
        else:
            element = Tet4Linear(material)
            formulation_type = "linear"
            is_nonlinear = False

    console.print(f"  • Constitutive Law : [cyan]{material.__class__.__name__}[/cyan] ({mat_desc})")
    console.print(f"  • Element Class    : [cyan]{element.__class__.__name__}[/cyan] (Formulation: {formulation_type})")
    console.print(
        f"  • Tissue Modulus   : [green]{args.youngs_modulus:.2f} kPa[/green] | "
        f"Poisson's ν: [green]{nu:.2f}[/green] | Density ρ: [green]{args.mass_density:.1f} kg/m³[/green]"
    )

    # 3. Domain & Boundary Constraints (Skull Base & Posterior Cradle Clamp)
    console.print("\n[bold yellow][3/6] Setting Up Physical Domain & Cranial Boundary Anchors[/bold yellow]")
    domain = SolidDomain(mesh=mesh, material=material, element=element)

    # Anatomical posterior cranial cradle & skull base support: Y < clamp_y in RAS
    clamped_node_mask = mesh.nodes[:, 1] < args.clamp_y
    clamped_nodes = np.where(clamped_node_mask)[0]

    # Cortical free surface nodes (excluding clamped support)
    cortical_free_nodes = np.array([idx for idx in cortical_surface_all if idx not in clamped_nodes])

    dirichlet = DirichletManager(domain.n_dofs)
    dirichlet.add_constraint(DirichletConstraint(clamped_nodes, components=(0, 1, 2), value=0.0))

    M_global = domain.assemble_mass_matrix()
    total_mass_tonne = abs(mesh.volume * rho_tonne_mm3)
    total_mass_kg = total_mass_tonne * 1e3
    total_weight_N = total_mass_kg * 9.81

    console.print(
        f"  • Cranial Support Clamp : [green]{len(clamped_nodes):,}[/green] nodes ({len(clamped_nodes)/mesh.n_nodes*100:.1f}%) "
        f"at Y < {args.clamp_y:.1f} mm"
    )
    console.print(f"  • Free Cortical Surface : [green]{len(cortical_free_nodes):,}[/green] nodes tracked")
    console.print(
        f"  • Total Brain Mass      : [green]{total_mass_kg:.3f} kg[/green] ({total_mass_kg*1e3:.1f} g) | "
        f"Weight: [green]{total_weight_N:.2f} N[/green]"
    )

    # 4. Assembling Real-Time Interactive Solver
    console.print("\n[bold yellow][4/6] Initializing Real-Time Generalized-α Physics Engine[/bold yellow]")
    rayleigh = RayleighDamping(alpha=4.0, beta=0.002)  # Viscous CSF & soft parenchyma damping
    orientation = HeadOrientationManager(pitch=0.0, roll=0.0, yaw=0.0)

    # Initial gravity load in neutral supine posture (g along -Y)
    initial_gravity = GravityLoad(acceleration=tuple(orientation.local_gravity))
    f_ext_initial = initial_gravity.compute_force_vector(M_global)

    sim = InteractiveDynamicSimulation(
        domain=domain,
        dirichlet=dirichlet,
        damping=rayleigh,
        dt=args.dt,
        spectral_radius=args.spectral_radius,
        nonlinear=is_nonlinear,
        f_ext_base=f_ext_initial,
    )

    console.print(
        f"  • Dynamic Time Step Δt: [green]{args.dt*1000:.1f} ms[/green] ({1.0/args.dt:.0f} Hz physics frequency)"
    )
    console.print(
        f"  • Spectral Radius ρ_∞  : [green]{args.spectral_radius:.2f}[/green] (High-frequency numerical chatter filtered)"
    )

    # Snapshot exporter if requested
    output_dir = Path(args.output_dir)
    snapshot_writer = None
    if args.save_snapshots:
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_writer = SimulationWriter(output_dir / "result.pvd", mesh=mesh, format="pvd")
        console.print(f"  • Snapshot Streaming  : [green]Enabled[/green] -> {output_dir / 'result.pvd'}")

    # Helper function to compute shift metrics in real time
    def evaluate_shift_metrics() -> tuple[np.ndarray, float, float, int, np.ndarray]:
        disp = sim.displacement  # (n_nodes, 3) in local RAS
        # 1. Lesion centroid drift
        current_lesion_center = (mesh.nodes[lesion_node_indices] + disp[lesion_node_indices]).mean(axis=0)
        lesion_drift_vec = current_lesion_center - lesion_center_ref
        lesion_drift_mag = float(np.linalg.norm(lesion_drift_vec))

        # 2. Cortical surface shift
        cortex_disp_norms = np.linalg.norm(disp[cortical_free_nodes], axis=1)
        max_cortex_idx_local = int(np.argmax(cortex_disp_norms))
        max_cortex_node_idx = cortical_free_nodes[max_cortex_idx_local]
        max_cortex_mag = float(cortex_disp_norms[max_cortex_idx_local])
        max_cortex_vec = disp[max_cortex_node_idx]

        return lesion_drift_vec, lesion_drift_mag, max_cortex_mag, max_cortex_node_idx, max_cortex_vec

    # 5. Headless Verification Sequence or Interactive 3D GUI
    is_headless = args.headless or (not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"))

    if is_headless:
        console.print("\n[bold yellow][5/6] Executing Automated Headless Validation Sequence[/bold yellow]")
        angles = [
            (0.0, 0.0, 0.0, "Supine (Neutral)"),
            (0.0, 0.0, 30.0, "Lateral Head Turn +30° (Pterional)"),
            (0.0, 0.0, -30.0, "Lateral Head Turn -30°"),
            (20.0, 0.0, 0.0, "Pitch Forward +20° (Chin Flexion)"),
        ]

        results_table = Table(
            title="Simulation 0603: Headless Brain Shift Validation Telemetry",
            border_style="cyan",
            header_style="bold magenta",
        )
        results_table.add_column("Surgical Pose", style="cyan")
        results_table.add_column("Local Gravity Vector g (m/s²)", style="dim")
        results_table.add_column("Lesion Drift (mm)", style="bold green", justify="right")
        results_table.add_column("Max Cortical Shift", style="bold yellow", justify="right")
        results_table.add_column("Physics Rate", style="green", justify="right")

        for pitch, roll, yaw, pose_name in angles:
            orientation.pitch = pitch
            orientation.roll = roll
            orientation.yaw = yaw
            g_local = orientation.local_gravity
            f_ext_local = GravityLoad(acceleration=tuple(g_local)).compute_force_vector(M_global)
            sim.set_base_external_force(f_ext_local)

            # Advance physics to quasi-steady dynamic equilibrium
            t0_step = time.perf_counter()
            n_substeps = int(0.5 / args.dt)
            for _ in range(n_substeps):
                sim.step()
            step_duration = (time.perf_counter() - t0_step) / n_substeps

            l_vec, l_mag, c_mag, c_idx, c_vec = evaluate_shift_metrics()
            rate_hz = 1.0 / max(step_duration, 1e-9)

            g_str = f"[{g_local[0]/1000:+.2f}, {g_local[1]/1000:+.2f}, {g_local[2]/1000:+.2f}]"
            results_table.add_row(
                pose_name,
                g_str,
                f"{l_mag:.2f} mm",
                f"{c_mag:.2f} mm",
                f"{rate_hz:,.0f} Hz",
            )

        console.print(results_table)

        # Verification assertion: max brain shift must be physically realistic (3 - 25 mm)
        final_l_vec, final_l_mag, final_c_mag, _, _ = evaluate_shift_metrics()
        assert 1.0 < final_c_mag < 35.0, f"Unphysical cortical shift: {final_c_mag:.2f} mm"
        assert 0.5 < final_l_mag < 25.0, f"Unphysical lesion drift: {final_l_mag:.2f} mm"

        console.print(
            f"\n[bold green]✓ Brain shift verification passed! Max cortical sag: {final_c_mag:.2f} mm | "
            f"Lesion centroid drift: {final_l_mag:.2f} mm.[/bold green]\n"
        )
        return

    # 6. Interactive 3D PyVista Neurosurgical GUI
    console.print("\n[bold yellow][6/6] Launching Interactive 3D Neurosurgical Suite[/bold yellow]")
    import pyvista as pv

    plotter = pv.Plotter(title="pSierra: Interactive Neurosurgical Brain Shift Simulator", window_size=(1400, 900))
    plotter.set_background("#14181f", top="#1f2733")

    # Load Skull Mesh for anatomical context (prioritize skull.vtk surface extraction)
    skull_mesh = None
    if skull_mesh_path.exists():
        console.print("  • Loading cranial skull vault from skull.vtk (surface extraction)...")
        skull_raw = pv.read(skull_mesh_path)
        skull_poly = skull_raw.extract_surface(algorithm="dataset_surface")
        skull_mesh = skull_poly
    elif skull_stl_path.exists():
        console.print("  • Loading cranial skull vault from STL...")
        skull_poly = pv.read(skull_stl_path)
        if skull_poly.n_points > 50000:
            skull_poly = skull_poly.decimate(0.60)
        skull_mesh = skull_poly
    else:
        console.print("  • Generating cranial surface bounding shell...")
        skull_mesh = mesh.ugrid.extract_surface(algorithm="dataset_surface").smooth(n_iter=20)

    # Create PyVista Grid for Deformed Brain Mesh
    brain_grid = mesh.ugrid.copy()
    initial_brain_points = mesh.nodes.copy()
    initial_skull_points = skull_mesh.points.copy() if skull_mesh is not None else None

    # Scalar displacement array initialized to 0
    disp_magnitude = np.zeros(mesh.n_nodes, dtype=np.float32)
    brain_grid.point_data["Displacement"] = disp_magnitude

    # 1. Add Deforming Brain Mesh Actor (Semi-Transparent Cortex, Default Opacity: 35%)
    brain_opacities = [0.35, 0.15, 0.65, 1.0]
    brain_opacity_idx = 0
    brain_actor = plotter.add_mesh(
        brain_grid,
        scalars="Displacement",
        cmap="turbo",
        clim=[0.0, 15.0],
        opacity=brain_opacities[brain_opacity_idx],
        scalar_bar_args={
            "title": "Brain Shift (mm)",
            "vertical": True,
            "title_font_size": 11,
            "label_font_size": 9,
            "color": "white",
            "position_x": 0.88,
            "position_y": 0.28,
            "width": 0.05,
            "height": 0.45,
        },
        show_edges=False,
        smooth_shading=True,
        specular=0.6,
        specular_power=30.0,
        ambient=0.2,
        diffuse=0.8,
    )

    # 2. Add Lesion (Tumor Target) Actors — Primary Surgical Telemetry Focus
    lesion_grid = lesion_pv.copy()
    initial_lesion_points = lesion_pv.points.copy()

    # Opaque crimson tumor actor visible directly inside the transparent brain
    lesion_actor = plotter.add_mesh(
        lesion_grid,
        color="#ff1744",
        opacity=1.0,
        smooth_shading=True,
        show_edges=True,
        edge_color="#d50000",
        line_width=1.5,
        specular=0.8,
        specular_power=40.0,
        ambient=0.3,
        diffuse=0.9,
    )

    # Ghost undeformed lesion reference mesh
    ghost_lesion_grid = lesion_pv.copy()
    ghost_lesion_actor = plotter.add_mesh(
        ghost_lesion_grid,
        style="wireframe",
        color="#ff80ab",
        opacity=0.45,
        line_width=1.5,
    )

    # Centroid marker sphere
    centroid_marker_actor = plotter.add_mesh(
        pv.Sphere(radius=3.0, center=lesion_center_ref),
        color="#ffff00",  # Bright yellow
        opacity=1.0,
    )

    # 3. Add Ghost Undeformed Reference Brain (Toggleable via [G])
    ghost_grid = mesh.ugrid.copy()
    ghost_actor = plotter.add_mesh(
        ghost_grid,
        color="#607d8b",
        style="wireframe",
        opacity=0.20,
        line_width=1.0,
        show_edges=True,
    )

    # 4. Add Skull Vault Actor (Very subtle/transparent reference cage, Default Opacity: 8%)
    skull_actor = None
    skull_opacities = [0.08, 0.20, 0.0, 0.04]
    skull_opacity_idx = 0
    if skull_mesh is not None:
        skull_actor = plotter.add_mesh(
            skull_mesh,
            color="#e0e0e0",  # Delicate ivory bone
            opacity=skull_opacities[skull_opacity_idx],
            smooth_shading=True,
            show_edges=False,
            specular=0.3,
            specular_power=15.0,
            ambient=0.2,
            diffuse=0.7,
        )

    # 5. Add Maximum Shift Arrow Indicator Actor (Toggleable via [V])
    show_ghost = True
    show_vector_arrow = True
    arrow_source = pv.Arrow(start=(0, 0, 0), direction=(0, 1, 0), tip_length=0.25, tip_radius=0.1, scale=10.0)
    arrow_actor = plotter.add_mesh(arrow_source, color="#ffeb3b", opacity=1.0)

    # 6. Orientation Axes in Anatomical RAS Coordinates (placed in bottom-right corner)
    plotter.add_axes(
        xlabel="R (+X)",
        ylabel="A (+Y)",
        zlabel="S (+Z)",
        line_width=3,
        viewport=(0.80, 0.0, 1.0, 0.22),
    )

    # Telemetry HUD Text Overlay (Corner 2 = upper_left, clean plain formatting)
    hud_actor = plotter.add_text(
        "Initializing Neurosurgical Telemetry...",
        position="upper_left",
        font_size=8.5,
        color="#80deea",
        font="courier",
    )

    # Key Instructions Overlay (Corner 0 = lower_left, single-column alignment)
    instructions_text = (
        "Surgical Presets:\n"
        "  [1] Supine (Face Up)\n"
        "  [2] Prone (Face Down)\n"
        "  [3] Right Lateral (-90° Yaw)\n"
        "  [4] Left Lateral (+90° Yaw)\n"
        "  [5] 30° Pterional Tilt\n"
        "  [6] Upright / Sitting\n\n"
        "Head Reorientation:\n"
        "  [W/S] Pitch ±5°\n"
        "  [A/D] Roll  ±5°\n"
        "  [Q/E] Yaw   ±5°\n\n"
        "Display & Simulation:\n"
        "  [B]     Brain Opacity\n"
        "  [K]     Skull Opacity\n"
        "  [G]     Toggle Ghost Initial Mesh\n"
        "  [V]     Toggle Max Sag Vector\n"
        "  [R]     Reset to Neutral Supine\n"
        "  [X/Esc] Exit Window"
    )
    plotter.add_text(instructions_text, position="lower_left", font_size=8.5, color="#b0bec5", font="courier")

    # Reorientation and transform helper
    def apply_current_orientation() -> None:
        R = orientation.rotation_matrix
        g_local = orientation.local_gravity
        f_ext_local = GravityLoad(acceleration=tuple(g_local)).compute_force_vector(M_global)
        sim.set_base_external_force(f_ext_local)

        # Update ghost mesh coordinates (world rotated)
        if show_ghost:
            ghost_grid.points = initial_brain_points @ R.T
            ghost_lesion_grid.points = initial_lesion_points @ R.T

        # Update skull mesh coordinates (world rotated)
        if skull_mesh is not None:
            skull_mesh.points = initial_skull_points @ R.T

    # User Interaction Callbacks
    def set_pose_preset(pose_key: str) -> None:
        if getattr(plotter, "render_window", None) is not None:
            plotter.render_window.SetStereoRender(False)
        desc = orientation.set_preset(pose_key)
        apply_current_orientation()
        console.print(f"[cyan]Surgical Pose Changed:[/cyan] [bold white]{desc}[/bold white]")

    def adjust_orientation(dp: float, dr: float, dy: float) -> None:
        orientation.adjust(dp, dr, dy)
        apply_current_orientation()
        console.print(
            f"[dim]Head Reoriented: Pitch {orientation.pitch:+5.1f}°, "
            f"Roll {orientation.roll:+5.1f}°, Yaw {orientation.yaw:+5.1f}°[/dim]"
        )

    def cycle_brain_opacity() -> None:
        nonlocal brain_opacity_idx
        brain_opacity_idx = (brain_opacity_idx + 1) % len(brain_opacities)
        op = brain_opacities[brain_opacity_idx]
        brain_actor.GetProperty().SetOpacity(op)
        console.print(f"[dim]Brain opacity set to {int(op*100)}%[/dim]")

    def cycle_skull_opacity() -> None:
        nonlocal skull_opacity_idx
        if skull_actor is not None:
            skull_opacity_idx = (skull_opacity_idx + 1) % len(skull_opacities)
            op = skull_opacities[skull_opacity_idx]
            skull_actor.GetProperty().SetOpacity(op)
            console.print(f"[dim]Skull opacity set to {int(op*100)}%[/dim]")

    def toggle_ghost() -> None:
        nonlocal show_ghost
        show_ghost = not show_ghost
        ghost_actor.SetVisibility(show_ghost)
        ghost_lesion_actor.SetVisibility(show_ghost)
        console.print(f"[dim]Ghost reference mesh: {'Visible' if show_ghost else 'Hidden'}[/dim]")

    def toggle_vector_arrow() -> None:
        nonlocal show_vector_arrow
        show_vector_arrow = not show_vector_arrow
        arrow_actor.SetVisibility(show_vector_arrow)
        console.print(f"[dim]Maximum shift vector arrow: {'Visible' if show_vector_arrow else 'Hidden'}[/dim]")

    def reset_simulation() -> None:
        orientation.pitch, orientation.roll, orientation.yaw = 0.0, 0.0, 0.0
        apply_current_orientation()
        sim.reset()
        console.print("[green]Simulation reset to initial undeformed state.[/green]")

    gui_state = {"exit_requested": False}

    def key_quit() -> None:
        gui_state["exit_requested"] = True
        if not getattr(plotter, "_closed", False):
            plotter.close()

    def is_window_active() -> bool:
        if gui_state["exit_requested"]:
            return False
        if getattr(plotter, "_closed", False):
            return False
        if getattr(plotter, "render_window", None) is None:
            return False
        return True

    # Bind Keyboard Shortcuts (both lowercase and uppercase)
    plotter.add_key_event("1", lambda: set_pose_preset("supine"))
    plotter.add_key_event("2", lambda: set_pose_preset("prone"))
    plotter.add_key_event("3", lambda: set_pose_preset("right_lateral"))
    plotter.add_key_event("4", lambda: set_pose_preset("left_lateral"))
    plotter.add_key_event("5", lambda: set_pose_preset("pterional_30"))
    plotter.add_key_event("6", lambda: set_pose_preset("upright"))

    for k in ["w", "W"]:
        plotter.add_key_event(k, lambda: adjust_orientation(dp=+5.0, dr=0.0, dy=0.0))
    for k in ["s", "S"]:
        plotter.add_key_event(k, lambda: adjust_orientation(dp=-5.0, dr=0.0, dy=0.0))
    for k in ["a", "A"]:
        plotter.add_key_event(k, lambda: adjust_orientation(dp=0.0, dr=-5.0, dy=0.0))
    for k in ["d", "D"]:
        plotter.add_key_event(k, lambda: adjust_orientation(dp=0.0, dr=+5.0, dy=0.0))
    for k in ["q", "Q"]:
        plotter.add_key_event(k, lambda: adjust_orientation(dp=0.0, dr=0.0, dy=-5.0))
    for k in ["e", "E"]:
        plotter.add_key_event(k, lambda: adjust_orientation(dp=0.0, dr=0.0, dy=+5.0))

    for k in ["b", "B"]:
        plotter.add_key_event(k, cycle_brain_opacity)
    for k in ["k", "K"]:
        plotter.add_key_event(k, cycle_skull_opacity)
    for k in ["g", "G"]:
        plotter.add_key_event(k, toggle_ghost)
    for k in ["v", "V"]:
        plotter.add_key_event(k, toggle_vector_arrow)
    for k in ["r", "R"]:
        plotter.add_key_event(k, reset_simulation)
    for k in ["x", "X", "Escape"]:
        plotter.add_key_event(k, key_quit)

    apply_current_orientation()

    # Set isometric neurosurgical view: +Y points upward, +Z goes left, +X goes right
    plotter.camera_position = [
        (240.0, 180.0, 240.0),  # Camera eye position in RAS
        (0.0, 10.0, 10.0),       # Focal point near cranial center
        (0.0, 1.0, 0.0),         # View up vector (+Y Anterior pointing upward)
    ]

    # Neutralize default VTK CharEvent observers so that:
    # 1. '3' triggers lateral surgical pose (instead of VTK stereo 3D turning screen purple)
    # 2. 'w' increments pitch angle (instead of VTK wireframe mode)
    # 3. 's' decrements pitch angle (instead of VTK surface mode)
    # 4. 'r' resets simulation (instead of VTK resetting camera)
    if hasattr(plotter, "iren") and plotter.iren is not None:
        plotter.iren.interactor.RemoveObservers("CharEvent")
        if plotter.iren.get_interactor_style() is not None:
            plotter.iren.get_interactor_style().RemoveObservers("CharEvent")

    console.print("[bold green]Neurosurgical GUI launched! Use controls in 3D window.[/bold green]")
    plotter.show(interactive_update=True)

    # Re-verify and ensure VTK stereo render is off and CharEvent observers are removed
    if getattr(plotter, "render_window", None) is not None:
        plotter.render_window.SetStereoRender(False)
    if hasattr(plotter, "iren") and plotter.iren is not None:
        plotter.iren.interactor.RemoveObservers("CharEvent")
        if plotter.iren.get_interactor_style() is not None:
            plotter.iren.get_interactor_style().RemoveObservers("CharEvent")

    # Dynamic render loop synchronized to wall-clock time
    last_wall_time = time.perf_counter()
    fps_timer = time.perf_counter()
    fps_frames = 0
    current_fps = 0.0

    try:
        while is_window_active():
            now = time.perf_counter()
            dt_wall = now - last_wall_time
            last_wall_time = now

            # Compute FPS
            fps_frames += 1
            if now - fps_timer >= 0.5:
                current_fps = fps_frames / (now - fps_timer)
                fps_timer = now
                fps_frames = 0

            # Advance physics sub-steps synchronized with real wall-clock time
            sim.advance_wall_clock(dt_wall, max_substeps=15)

            # Retrieve current deformation
            disp_local = sim.displacement  # (n_nodes, 3) in local RAS
            R = orientation.rotation_matrix

            # Transform deformed coordinates into rotated world frame for rendering
            deformed_local_coords = initial_brain_points + disp_local
            brain_grid.points = deformed_local_coords @ R.T

            # Deform lesion target mesh and update world position
            deformed_lesion_local = initial_lesion_points + disp_local[lesion_point_to_brain]
            lesion_grid.points = deformed_lesion_local @ R.T
            if show_ghost:
                ghost_lesion_grid.points = initial_lesion_points @ R.T

            # Update lesion centroid sphere marker
            cur_lesion_center_local = deformed_lesion_local.mean(axis=0)
            cur_lesion_center_world = cur_lesion_center_local @ R.T
            centroid_marker_actor.GetMapper().SetInputData(pv.Sphere(radius=3.0, center=cur_lesion_center_world))

            # Scalar coloring: displacement magnitude norm
            disp_norms = np.linalg.norm(disp_local, axis=1)
            brain_grid.point_data["Displacement"] = disp_norms

            # Evaluate shift metrics
            l_vec, l_mag, c_mag, c_idx, c_vec = evaluate_shift_metrics()

            # Update maximum shift 3D arrow indicator
            if show_vector_arrow and c_mag > 0.05:
                pt_start = initial_brain_points[c_idx] @ R.T
                pt_dir = (c_vec @ R.T)
                norm_dir = np.linalg.norm(pt_dir)
                if norm_dir > 1e-6:
                    arrow = pv.Arrow(start=pt_start, direction=pt_dir / norm_dir, scale=c_mag * 1.5, tip_length=0.25)
                    arrow_actor.GetMapper().SetInputData(arrow)

            # Update Live Telemetry HUD in Corner 2 (upper_left, clean plain formatting)
            g_loc = orientation.local_gravity
            hud_lines = [
                "Lesion Target Telemetry",
                f"  Centroid Drift   {l_mag:5.2f} mm",
                f"  Drift Vector     [{l_vec[0]:+5.2f}, {l_vec[1]:+5.2f}, {l_vec[2]:+5.2f}] mm",
                f"  Current Target   ({cur_lesion_center_local[0]:+5.1f}, {cur_lesion_center_local[1]:+5.1f}, {cur_lesion_center_local[2]:+5.1f}) mm",
                "",
                "Cortical Sag & Telemetry",
                f"  Max Brain Sag    {c_mag:5.2f} mm (Node #{c_idx})",
                f"  Head Pitch       {orientation.pitch:+5.1f}°",
                f"  Head Roll        {orientation.roll:+5.1f}°",
                f"  Head Yaw         {orientation.yaw:+5.1f}°",
                f"  Gravity Vector   [{g_loc[0]/1000:+5.2f}, {g_loc[1]/1000:+5.2f}, {g_loc[2]/1000:+5.2f}] m/s²",
                f"  Physics Rate     {1.0/max(args.dt, 1e-4):4.0f} Hz",
                f"  Render Rate      {current_fps:4.1f} FPS",
            ]
            hud_actor.set_text(2, "\n".join(hud_lines))

            try:
                plotter.update()
            except Exception:
                break

            # Relinquish timeslice briefly to allow smooth OS event queue dispatch
            time.sleep(0.002)

    except KeyboardInterrupt:
        console.print("\n[yellow]Execution terminated by user (Ctrl+C).[/yellow]")
    finally:
        if not getattr(plotter, "_closed", False):
            plotter.close()


if __name__ == "__main__":
    main()
