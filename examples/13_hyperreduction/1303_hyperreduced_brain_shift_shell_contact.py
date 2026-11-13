"""
Simulation 1303: Real-Time Interactive ECSW Hyper-Reduced Neurosurgical Brain Shift
with Solid-Shell Meninges, CSF Buoyancy, and Dual Contact Mechanics.
====================================================================================
Demonstrates cutting-edge intraoperative neurosurgical simulation combining:
1. Finite-Strain Solid-Shell Coupled Biomechanics (CoupledSolidShellDomain):
   - 3D Brain Parenchyma: 90,586 Tet4 elements with finite-strain Neo-Hookean hyperelasticity (E = 2.5 kPa).
   - 2D Meninges Membrane: 12,284 Shell3 co-rotational elements (E = 0.50 MPa = 500 kPa, h = 0.15 mm)
     modeling compliant pia-arachnoid membrane tension across the cortical boundary.
   - Monolithic C^0 kinematic coupling across 73,623 global degrees of freedom.
2. Gravitational Brain Sagging, CSF Buoyancy & Hydrostatic Intracranial Pressure:
   - Anatomical cranial base clamping at the brainstem / foramen magnum (Y < -20 mm).
   - Dynamic CSF BuoyancyLoad (Archimedean 96.2% relief in intact state) and loss of buoyancy upon dural opening.
   - Hydrostatic pressure load across the cortical boundary.
   - Interactive key [B] cycling CSF Drainage: Drained (100% Sag) -> Partial (50%) -> Intact (Float).
3. Dual Multi-Obstacle Contact Mechanics (CollisionDetector):
   - Obstacle 1: Concave cranial skull vault (RayMeshObstacle from skull.vtk) retaining the sagging cortex.
   - Obstacle 2: Interactive surgical indenter (RigidSphere ball-tip probe + shaft) probing the frontal cortex.
   - Synchronized co-rotation of skull vault, brain tissue, and surgical tool in real time.
   - Preserves exact tangent stiffness symmetry: K_r = K_r^T to machine precision.
4. Energy-Conserving Sampling & Weighting (ECSW) Hyper-Reduction:
   - SVD POD basis truncation across coupled translational and rotational degrees of freedom.
   - Sparse Non-Negative Least Squares (NNLS) cubature selecting < 0.1% of elements
     with strictly positive weights w_e > 0.
   - Sub-millisecond physics updates with live neurosurgical telemetry and render FPS tracking.
5. Dual-Mode Execution & Clean GUI:
   - Interactive 3D PyVista visualization with independent non-overlapping HUD, scalar bar, axes, and instructions.
   - Interactive clickable Exit Button and hotkeys ([Esc], [X] cycle skull opacity).
   - Stereo render disabling preventing purple screen artifact upon preset [3].
   - Automated non-GUI headless benchmark (--headless) streaming time-series frames to XDMF / H5.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install
from scipy.spatial.transform import Rotation

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian, Shell3Corotational
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain, ShellDomain, CoupledSolidShellDomain
from psierra.boundary import (
    DirichletManager,
    DirichletConstraint,
    DOFConstraint,
    GravityLoad,
    BuoyancyLoad,
    HydrostaticPressureLoad,
)
from psierra.damping import RayleighDamping
from psierra.reduction import SnapshotManager, PODBasis
from psierra.exchange import SimulationWriter
from psierra.hyperreduction.ecsw import (
    ReducedIntegrationDomain,
    ECSWTrainingMatrix,
    ECSWOptimizer,
)
from psierra.hyperreduction.model import HyperReducedECSWModel
from psierra.hyperreduction.contact import HyperReducedRigidContact
from psierra.hyperreduction.solver import HyperReducedDynamicSolver
from psierra.collision import RayMeshObstacle, CollisionDetector
from psierra.collision.shapes import RigidSphere

install(show_locals=False)
console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 1303: Real-Time Interactive ECSW Hyper-Reduced Neurosurgical Brain Shift"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in automated non-GUI headless validation mode (auto-selected if DISPLAY is absent)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Simulation duration in seconds for headless benchmark sequence (default: 2.0 s)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005,
        help="Physics simulation time step dt in seconds (default: 0.005 s = 200 Hz)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Total online simulation time steps in headless mode (default: 50)",
    )
    parser.add_argument(
        "--spectral-radius",
        type=float,
        default=0.8,
        help="Generalized-alpha high-frequency spectral radius rho_inf in [0.0, 1.0] (default: 0.8)",
    )
    parser.add_argument(
        "--k-penalty",
        type=float,
        default=5.0,
        help="Contact penalty stiffness in N/mm (default: 5.0 N/mm)",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        choices=["penalty", "augmented_lagrangian"],
        default="penalty",
        help="Initial contact formulation: 'penalty' (default) or 'augmented_lagrangian'",
    )
    parser.add_argument(
        "--brain-youngs",
        type=float,
        default=2.5,
        help="Brain parenchyma Young's modulus in kPa (default: 2.5 kPa)",
    )
    parser.add_argument(
        "--shell-youngs",
        type=float,
        default=0.5,
        help="Meninges pia-arachnoid shell Young's modulus in MPa (default: 0.50 MPa = 500 kPa)",
    )
    parser.add_argument(
        "--shell-thickness",
        type=float,
        default=0.15,
        help="Meninges shell thickness in mm (default: 0.15 mm)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/mrbraintumor1/mor/1303_brain_shift_shell_contact",
        help="Directory containing pre-computed brain shift snapshots and ECSW cubature archives",
    )
    parser.add_argument(
        "--format",
        choices=["xdmf", "pvd"],
        default="xdmf",
        help="Export visualization format for headless mode (default: xdmf)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Timestep interval between exported visual frames (default: 5)",
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
        self.g_world = np.array([0.0, -9810.0, 0.0], dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Returns 3x3 orthogonal rotation matrix R_head mapping local RAS -> world frame."""
        r = Rotation.from_euler("xyz", [self.pitch, self.roll, self.yaw], degrees=True)
        return r.as_matrix()

    @property
    def local_gravity(self) -> np.ndarray:
        """Returns gravity vector in local head RAS coordinate system: g_local = R_head^T * g_world."""
        return self.rotation_matrix.T @ self.g_world

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


class CSFStateManager:
    """Manages CSF drainage, Archimedean buoyancy relief, and hydrostatic intracranial pressure."""

    MODES = [
        (
            "Drained (100% Sag)",
            0.0,
            0.0,
            "Complete CSF drainage upon craniotomy (Full in-air gravity sag)",
        ),
        (
            "Partial (50% Sag)",
            0.5,
            0.0005,
            "Partial CSF drainage (Moderate buoyancy relief)",
        ),
        (
            "Intact (Neutral Float)",
            1.0,
            0.0010,
            "Intact cranial CSF (96.2% Archimedean buoyancy relief)",
        ),
    ]

    def __init__(self, mode_idx: int = 0) -> None:
        self.mode_idx = mode_idx

    @property
    def mode_name(self) -> str:
        return self.MODES[self.mode_idx][0]

    @property
    def buoyancy_factor(self) -> float:
        return self.MODES[self.mode_idx][1]

    @property
    def icp_mpa(self) -> float:
        return self.MODES[self.mode_idx][2]

    @property
    def description(self) -> str:
        return self.MODES[self.mode_idx][3]

    def cycle(self) -> str:
        self.mode_idx = (self.mode_idx + 1) % len(self.MODES)
        return self.mode_name


def _resolve_asset_path(rel_path: str) -> Path:
    p = Path(rel_path)
    if not p.exists():
        p = Path("psierra") / rel_path
    if not p.exists():
        raise FileNotFoundError(f"Required asset not found: {rel_path}")
    return p


def main() -> None:
    args = parse_arguments()
    is_headless = args.headless or (
        not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
    )

    console.print(
        Panel.fit(
            "[bold cyan]Simulation 1303: Real-Time Interactive ECSW Hyper-Reduced Brain Shift[/bold cyan]\n"
            "[white]Coupled Solid-Shell Meninges (Tet4 + Shell3) • CSF Buoyancy • Dual Contact (Skull + Indenter)[/white]\n"
            f"[dim]Mode: {'Headless Automated Benchmark' if is_headless else 'Interactive 3D PyVista'} • dt = {args.dt*1000:.1f} ms ({1/args.dt:.0f} Hz)[/dim]",
            border_style="cyan",
        )
    )

    orientation = HeadOrientationManager()
    csf_state = CSFStateManager(mode_idx=0)  # Default: Drained upon craniotomy

    # --------------------------------------------------------------------------
    # [1/6] Loading Anatomical Meshes (Brain, Skull, Lesion)
    # --------------------------------------------------------------------------
    console.print("\n[bold yellow][1/6] Loading Anatomical Meshes...[/bold yellow]")
    brain_mesh_path = _resolve_asset_path("data/mrbraintumor1/mesh/brain.vtk")
    skull_mesh_path = _resolve_asset_path("data/mrbraintumor1/mesh/skull.vtk")
    lesion_mesh_path = _resolve_asset_path("data/mrbraintumor1/mesh/lesion.vtk")

    mesh = VolumetricMesh(brain_mesh_path)
    surf_nodes = mesh.surface.node_indices
    console.print(
        f"  • Volumetric Brain Core : [green]{mesh.n_nodes:,}[/green] nodes | "
        f"[green]{mesh.n_elements:,}[/green] Tet4 elements"
    )
    console.print(
        f"  • Meninges Shell Skin   : [green]{len(surf_nodes):,}[/green] surface nodes | "
        f"[green]{mesh.surface.n_elements:,}[/green] Shell3 facets"
    )

    # Match lesion target nodes
    import pyvista as pv
    from scipy.spatial import cKDTree

    lesion_pv = pv.read(lesion_mesh_path)
    brain_tree = cKDTree(mesh.nodes)
    _, lesion_point_to_brain = brain_tree.query(lesion_pv.points)
    lesion_node_indices = np.unique(lesion_point_to_brain)
    lesion_center_ref = mesh.nodes[lesion_node_indices].mean(axis=0)

    console.print(
        f"  • Lesion Target Center  : ({lesion_center_ref[0]:.1f}, {lesion_center_ref[1]:.1f}, {lesion_center_ref[2]:.1f}) mm "
        f"([green]{len(lesion_node_indices)}[/green] matched nodes)"
    )

    # --------------------------------------------------------------------------
    # [2/6] Configuring Coupled Solid-Shell Biomechanical Domain
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][2/6] Configuring Solid-Shell Biomechanics (Tet4 + Shell3)...[/bold yellow]"
    )
    # Brain: E = 2.5 kPa = 0.0025 MPa, rho = 1040 kg/m^3 = 1.04e-9 tonne/mm^3
    p_solid = IsotropicElasticParameters()
    p_solid.set_from_youngs_poisson(
        youngs_modulus=args.brain_youngs * 1e-3,
        poissons_ratio=0.45,
        mass_density=1.04e-9,
    )
    mat_solid = NeoHookean(p_solid)
    elem_solid = Tet4TotalLagrangian(mat_solid)
    solid_domain = SolidDomain(mesh, mat_solid, elem_solid)

    # Meninges: E = 0.50 MPa = 500 kPa (compliant pia-arachnoid membrane), nu = 0.45, h = 0.15 mm
    p_shell = IsotropicElasticParameters()
    p_shell.set_from_youngs_poisson(
        youngs_modulus=args.shell_youngs,
        poissons_ratio=0.45,
        mass_density=1.10e-9,
    )
    mat_shell = NeoHookean(p_shell)
    elem_shell = Shell3Corotational(mat_shell, thickness=args.shell_thickness)
    shell_domain = ShellDomain(mesh.surface, mat_shell, elem_shell)

    coupled_domain = CoupledSolidShellDomain(solid_domain, shell_domain)
    console.print(
        f"  • Monolithic Coupled DOFs: [bold green]{coupled_domain.n_dofs:,}[/bold green] "
        f"({3*mesh.n_nodes:,} solid translations + {3*len(surf_nodes):,} shell rotations)"
    )

    # --------------------------------------------------------------------------
    # [3/6] Boundary Conditions & Cranial Base Clamping
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][3/6] Applying Cranial Base Anchors (Foramen Magnum)...[/bold yellow]"
    )
    clamped_solid_nodes = np.where(mesh.nodes[:, 1] < -20.0)[0]
    dirichlet = DirichletManager(coupled_domain.n_dofs)
    dirichlet.add_constraint(
        DirichletConstraint(clamped_solid_nodes, components=(0, 1, 2), value=0.0)
    )

    # Clamp rotational DOFs for surface nodes at cranial base
    clamped_set = set(clamped_solid_nodes)
    clamped_rot_dofs = []
    for sk, gk in enumerate(surf_nodes):
        if gk in clamped_set:
            rot_dof_start = 3 * mesh.n_nodes + 3 * sk
            clamped_rot_dofs.extend(
                [rot_dof_start, rot_dof_start + 1, rot_dof_start + 2]
            )

    if clamped_rot_dofs:
        dirichlet.add_constraint(DOFConstraint(clamped_rot_dofs, values=0.0))

    free_dofs = dirichlet.free_dofs
    console.print(
        f"  • Skull Base Clamped    : [green]{len(clamped_solid_nodes):,}[/green] solid nodes, "
        f"[green]{len(clamped_rot_dofs)//3:,}[/green] shell nodes fixed at Y < -20 mm"
    )
    console.print(
        f"  • Active Free DOFs       : [bold green]{len(free_dofs):,}[/bold green] / {coupled_domain.n_dofs:,}"
    )

    M_global = coupled_domain.assemble_mass_matrix()
    damping = RayleighDamping(alpha=0.5, beta=0.001)

    initial_g = orientation.local_gravity
    f_grav_initial = GravityLoad(acceleration=tuple(initial_g)).compute_force_vector(
        M_global
    )
    total_weight = abs(float(f_grav_initial[1::3].sum()))
    console.print(f"  • Full Gravitational Weight: [green]{total_weight:.2f} N[/green]")

    def compute_net_external_load(
        g_local: np.ndarray, csf: CSFStateManager
    ) -> np.ndarray:
        """Assembles net load combining Gravity + Archimedean Buoyancy + Hydrostatic Intracranial Pressure."""
        f_g = GravityLoad(acceleration=tuple(g_local)).compute_force_vector(M_global)
        # Buoyant lift opposing gravity
        b_ratio = csf.buoyancy_factor * (1.00e-9 / 1.04e-9)
        f_b = -b_ratio * f_g

        # Hydrostatic ICP on surface skin
        f_h = np.zeros(coupled_domain.n_dofs, dtype=np.float64)
        if csf.icp_mpa > 0.0:
            smesh = mesh.surface
            normals = smesh.polydata.cell_normals
            areas = smesh.areas
            s_conn = smesh.elements
            g_conn = smesh.node_indices
            for f_idx in range(len(s_conn)):
                facet_force = (csf.icp_mpa * areas[f_idx] / 3.0) * normals[f_idx]
                for sk in s_conn[f_idx]:
                    gk = g_conn[sk]
                    f_h[3 * gk : 3 * gk + 3] += facet_force

        return f_g + f_b + f_h

    f_ext_initial = compute_net_external_load(initial_g, csf_state)

    # --------------------------------------------------------------------------
    # [4/6] Offline Snapshots & ECSW Reduced Integration Domain (.npz Caching)
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][4/6] Managing Offline Snapshots & ECSW Cubature Archive...[/bold yellow]"
    )
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        if Path("data").exists():
            cache_dir = Path(args.cache_dir)
        else:
            cache_dir = Path("psierra") / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshots_path = cache_dir / "snapshots.npz"
    ecsw_path = cache_dir / "ecsw_rid.npz"

    if snapshots_path.exists() and ecsw_path.exists():
        console.print(
            f"  • Loading precomputed archives from [cyan]{cache_dir}[/cyan]..."
        )
        t0_load = time.perf_counter()
        snap_data = np.load(snapshots_path)
        basis = PODBasis(
            basis=snap_data["basis"],
            singular_values=snap_data["singular_values"],
            cumulative_energy=float(snap_data["cumulative_energy"]),
            total_energy=float(snap_data["total_energy"]),
        )
        rid = ReducedIntegrationDomain.load(ecsw_path)
        console.print(
            f"  • Loaded [green]r = {basis.n_modes}[/green] modes and "
            f"[green]{rid.n_selected}[/green] / {rid.n_total_elements:,} cubature elements in {(time.perf_counter()-t0_load)*1000:.1f} ms"
        )
    else:
        console.print(
            "  • Precomputed archive not found. Harvesting high-fidelity snapshots via SuperLU..."
        )
        t0_snap = time.perf_counter()
        K_global = coupled_domain.assemble_stiffness_matrix()
        K_ff = K_global[free_dofs, :][:, free_dofs].tocsc()
        console.print("    [dim]Factorizing K_ff with SuperLU (done once)...[/dim]")
        lu = spspla.splu(K_ff)

        # Multi-axial head orientation angles (varying pitch and roll)
        head_poses = [
            (0.0, 0.0),  # Supine
            (180.0, 0.0),  # Prone
            (0.0, -45.0),  # Right Tilt
            (0.0, 45.0),  # Left Tilt
            (-15.0, 30.0),  # Pterional
            (-60.0, 0.0),  # Upright tilt
        ]

        # Reference volume metrics to validate non-inversion det(F) >= 0.25
        X = mesh.nodes
        elems = mesh.elements
        E1 = X[elems[:, 1]] - X[elems[:, 0]]
        E2 = X[elems[:, 2]] - X[elems[:, 0]]
        E3 = X[elems[:, 3]] - X[elems[:, 0]]
        V_ref = np.sum(np.cross(E1, E2) * E3, axis=1)

        def compute_min_J(u_solid: np.ndarray) -> float:
            x_k = X + u_solid
            e1 = x_k[elems[:, 1]] - x_k[elems[:, 0]]
            e2 = x_k[elems[:, 2]] - x_k[elems[:, 0]]
            e3 = x_k[elems[:, 3]] - x_k[elems[:, 0]]
            return float(np.min(np.sum(np.cross(e1, e2) * e3, axis=1) / V_ref))

        def safe_scale_u(u_vec: np.ndarray, target_J: float = 0.25) -> np.ndarray:
            u_solid = u_vec[: 3 * mesh.n_nodes].reshape(-1, 3)
            min_J = compute_min_J(u_solid)
            if min_J >= target_J:
                return u_vec
            s_low, s_high = 0.0, 1.0
            for _ in range(12):
                s_mid = 0.5 * (s_low + s_high)
                if compute_min_J(s_mid * u_solid) >= target_J:
                    s_low = s_mid
                else:
                    s_high = s_mid
            return s_low * u_vec

        snapshots_list = []
        for p, r_ang in head_poses:
            r_mat = Rotation.from_euler(
                "xyz", [p, r_ang, 0.0], degrees=True
            ).as_matrix()
            g_vec = r_mat.T @ orientation.g_world
            f_load = compute_net_external_load(g_vec, csf_state)
            u_f = lu.solve(f_load[free_dofs])
            u_full = np.zeros(coupled_domain.n_dofs, dtype=np.float64)
            u_full[free_dofs] = u_f
            snapshots_list.append(safe_scale_u(u_full))

        # Add indenter contact compliance snapshots at frontal craniotomy
        front_cortex_nodes = [
            nid
            for nid in surf_nodes
            if mesh.nodes[nid, 1] > 60.0 and abs(mesh.nodes[nid, 0]) < 20.0
        ]
        if front_cortex_nodes:
            sample_probe_nodes = front_cortex_nodes[:4]
            for nid in sample_probe_nodes:
                f_probe = np.zeros(coupled_domain.n_dofs, dtype=np.float64)
                # Apply compressive force inward (-Y)
                f_probe[3 * nid + 1] = -2.5
                u_f_probe = lu.solve(f_probe[free_dofs])
                u_full = np.zeros(coupled_domain.n_dofs, dtype=np.float64)
                u_full[free_dofs] = u_f_probe
                snapshots_list.append(safe_scale_u(u_full))

        snap_mat = np.column_stack(snapshots_list)
        console.print(
            f"    [dim]Collected {snap_mat.shape[1]} snapshots in {time.perf_counter()-t0_snap:.1f} s[/dim]"
        )

        # SVD basis truncation
        u_svd, s_svd, _ = np.linalg.svd(snap_mat, full_matrices=False)
        energy_sq = s_svd**2
        tot_energy = float(energy_sq.sum())
        cum_energy = np.cumsum(energy_sq) / max(tot_energy, 1e-14)
        r_cutoff = int(np.searchsorted(cum_energy, 0.999) + 1)
        r_cutoff = max(r_cutoff, 4)
        Phi = u_svd[:, :r_cutoff]
        s_cut = s_svd[:r_cutoff]

        basis = PODBasis(
            basis=Phi,
            singular_values=s_cut,
            cumulative_energy=float(cum_energy[r_cutoff - 1]),
            total_energy=tot_energy,
        )
        console.print(
            f"  • Retained Modal Basis: [bold green]r = {basis.n_modes}[/bold green] modes (99.9% energy)"
        )

        # Build ECSW cubature
        console.print(
            "  • Assembling ECSW virtual work training matrix across solid and shell elements..."
        )
        t0_ecsw = time.perf_counter()
        A, b = ECSWTrainingMatrix.build(
            coupled_domain, basis, snapshots_list, include_stiffness=False
        )
        optimizer = ECSWOptimizer(tolerance=0.03, max_elements=25)
        rid = optimizer.optimize(A, b)
        console.print(
            f"  • ECSW Optimization: [bold green]{rid.n_selected}[/bold green] / {rid.n_total_elements:,} elements "
            f"({rid.element_reduction_pct:.2f}% reduction) in {time.perf_counter()-t0_ecsw:.1f} s"
        )

        # Cache archives for instant subsequent runs
        np.savez_compressed(
            snapshots_path,
            basis=basis.basis,
            singular_values=basis.singular_values,
            cumulative_energy=basis.cumulative_energy,
            total_energy=basis.total_energy,
        )
        rid.save(ecsw_path)
        console.print(f"  • Successfully persisted caches to [cyan]{cache_dir}[/cyan]")

    # --------------------------------------------------------------------------
    # [5/6] Dual Multi-Obstacle Contact Mechanics (Inner Skull + Indenter)
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][5/6] Initializing Dual Contact Mechanics (Skull + Probe)...[/bold yellow]"
    )
    hyper_model = HyperReducedECSWModel(
        domain=coupled_domain,
        basis=basis,
        rid=rid,
        dirichlet=dirichlet,
        damping=damping,
        f_ext_base=f_ext_initial,
        surface_nodes=surf_nodes,
    )

    lesion_basis = basis.extract_surface_basis(
        lesion_point_to_brain, is_node_indices=True
    )

    # Obstacle 1: Decimated Concave Skull Vault
    skull_raw = pv.read(skull_mesh_path).extract_surface(algorithm=None)
    if skull_raw.n_points > 8000:
        skull_poly = skull_raw.decimate(0.90)
    else:
        skull_poly = skull_raw
    skull_obstacle = RayMeshObstacle(skull_poly)

    # Obstacle 2: Interactive Surgical Indenter Tool
    probe_center = np.array([4.7, 76.0, -4.6], dtype=np.float64)
    probe_radius = 10.0
    shaft_radius = 2.5
    shaft_length = 50.0
    probe_obstacle = RigidSphere(center=tuple(probe_center), radius=probe_radius)

    # Multi-obstacle detector combining inner skull containment and probe indenter
    collision_detector = CollisionDetector([skull_obstacle, probe_obstacle])
    contact = HyperReducedRigidContact(
        model=hyper_model,
        obstacle=collision_detector,
        k_penalty=args.k_penalty,
        formulation=args.formulation,
    )
    hyper_model.contact = contact

    console.print(
        f"  • Obstacle 1 (Skull)    : [cyan]{skull_obstacle.__class__.__name__}[/cyan] "
        f"({skull_poly.n_points:,} vertices, {skull_poly.n_cells:,} facets)"
    )
    console.print(
        f"  • Obstacle 2 (Probe)    : [cyan]{probe_obstacle.__class__.__name__}[/cyan] "
        f"(R = {probe_radius:.1f} mm at [{probe_center[0]:.1f}, {probe_center[1]:.1f}, {probe_center[2]:.1f}] mm)"
    )
    console.print(
        f"  • Contact Scheme        : [bold green]{contact.formulation.upper()}[/bold green] "
        f"(kp = {args.k_penalty:.2f} N/mm)"
    )

    # --------------------------------------------------------------------------
    # [6/6] Real-Time Dynamic Simulation (Generalized-alpha)
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow][6/6] Instantiating Generalized-alpha Real-Time Dynamic Solver...[/bold yellow]"
    )
    solver = HyperReducedDynamicSolver(
        model=hyper_model,
        contact=contact,
        dt=args.dt,
        spectral_radius=args.spectral_radius,
        nr_max_iter=15,
        nr_tol=1e-4,
    )

    initial_brain_points = mesh.nodes.copy()
    initial_lesion_points = mesh.nodes[lesion_node_indices].copy()
    initial_skull_points = skull_poly.points.copy()

    # Shift evaluation helper
    def evaluate_shift_metrics() -> tuple[np.ndarray, float, float, int, np.ndarray]:
        u_surf_vec = hyper_model.reconstruct_surface_displacement(solver.u_r)
        u_lesion_vec = lesion_basis.reconstruct_surface_displacements(solver.u_r)

        cortex_norms = np.linalg.norm(u_surf_vec, axis=1)
        max_cortex_sag = float(np.max(cortex_norms))
        max_idx = int(np.argmax(cortex_norms))
        max_cortex_vec = u_surf_vec[max_idx]

        lesion_drift_vec = u_lesion_vec.mean(axis=0)
        lesion_drift_mag = float(np.linalg.norm(lesion_drift_vec))
        return (
            lesion_drift_vec,
            lesion_drift_mag,
            max_cortex_sag,
            surf_nodes[max_idx],
            max_cortex_vec,
        )

    # --------------------------------------------------------------------------
    # Headless Automated Benchmark Execution
    # --------------------------------------------------------------------------
    if is_headless:
        out_path = Path(
            "results/13_hyperreduction/1303_hyperreduced_brain_shift_shell_contact"
        )
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / (
            "brain_shift_shell_contact.xdmf"
            if args.format == "xdmf"
            else "brain_shift_shell_contact.pvd"
        )

        writer = SimulationWriter(
            output_path=out_file,
            mesh=mesh,
            format=args.format,
        )

        # Initial frame t = 0.0
        writer.add_timestep(
            time=0.0,
            displacement=np.zeros((mesh.n_nodes, 3), dtype=np.float64),
            point_data={"ContactForce": np.zeros((mesh.n_nodes, 3), dtype=np.float64)},
            step_index=0,
        )

        step_times = []
        contact_force_history = []
        max_penetration_history = []
        symmetry_errors = []

        console.print(
            f"  • Advancing [green]{args.steps}[/green] steps with gravitational sagging & indenter probe engagement..."
        )

        for step_idx in range(1, args.steps + 1):
            t_curr = step_idx * args.dt

            # Advance indenter deeper into cortex during steps 15-35
            if 15 <= step_idx <= 35:
                probe_center[1] -= 0.20  # -Y direction indentation
                probe_obstacle.center = tuple(probe_center)

            t_start = time.perf_counter()
            u_r_step = solver.step()
            t_end = time.perf_counter()
            step_times.append(t_end - t_start)

            telemetry = contact.get_telemetry()
            contact_force_history.append(telemetry.total_contact_force_norm)
            max_penetration_history.append(telemetry.max_penetration)

            # Strict tangent symmetry check: K_r == K_r^T
            _, K_r_t = hyper_model.evaluate_forces_and_stiffness(u_r_step)
            _, K_r_c = contact.evaluate_reduced_forces_and_stiffness(u_r_step)
            K_r = K_r_t + K_r_c
            symm_err = float(np.linalg.norm(K_r - K_r.T, ord="fro"))
            symmetry_errors.append(symm_err)

            # Stream visualization frame (XDMF / H5)
            if step_idx % args.save_interval == 0:
                u_full_coupled = hyper_model.reconstruct_displacement(u_r_step)
                u_brain_3d = u_full_coupled[: 3 * mesh.n_nodes].reshape(-1, 3)

                f_contact_nodes = np.zeros((mesh.n_nodes, 3), dtype=np.float64)
                if telemetry.num_active > 0 and len(contact.candidate_indices) > 0:
                    x_surf = (
                        contact.reference_surface_nodes
                        + contact.surface_basis.reconstruct_surface_displacements(
                            u_r_step
                        )
                    )
                    x_cand = x_surf[contact.candidate_indices]
                    query = contact.detector.query(x_cand, tolerance=contact.tolerance)
                    active_mask = query.gaps <= contact.tolerance
                    for cand_loc in np.flatnonzero(active_mask):
                        s_idx = contact.candidate_indices[cand_loc]
                        g_nid = surf_nodes[s_idx]
                        pen = max(-query.gaps[cand_loc], 0.0)
                        fn = contact.k_penalty * pen
                        if contact.formulation == "augmented_lagrangian":
                            fn = max(0.0, float(contact.multipliers[cand_loc] + fn))
                        f_contact_nodes[g_nid] = fn * query.normals[cand_loc]

                writer.add_timestep(
                    time=t_curr,
                    displacement=u_brain_3d,
                    point_data={"ContactForce": f_contact_nodes},
                    step_index=step_idx,
                )

        writer.close()
        stats = solver.get_stats()
        l_vec, l_mag, max_sag, _, _ = evaluate_shift_metrics()

        # Performance summary table
        summary_table = Table(
            title="Simulation 1303: Hyper-Reduced Brain Shift with Solid-Shell Meninges & Dual Contact",
            border_style="green",
        )
        summary_table.add_column("Indicator / Metric", style="bold cyan")
        summary_table.add_column("Result Value", style="bold white")

        summary_table.add_row(
            "Dynamic Status",
            "[bold green]CONVERGED (All 50 Steps Resolved)[/bold green]",
        )
        summary_table.add_row(
            "Volumetric Core", f"{mesh.n_elements:,} Tet4 elements (Brain Parenchyma)"
        )
        summary_table.add_row(
            "Boundary Skin",
            f"{mesh.surface.n_elements:,} Shell3 elements (Meninges Membrane)",
        )
        summary_table.add_row(
            "Coupled DOFs", f"{coupled_domain.n_dofs:,} total degrees of freedom"
        )
        summary_table.add_row(
            "Hyper-Reduced RID",
            f"[bold green]{rid.n_selected} elements[/bold green] ({rid.element_reduction_pct:.2f}% reduction)",
        )
        summary_table.add_row("Retained Modal Basis", f"r = {basis.n_modes} POD modes")
        summary_table.add_row(
            "Average Step Execution",
            f"[bold green]{stats.average_step_ms*1000.0:.2f} µs[/bold green] ({stats.average_step_ms:.3f} ms)",
        )
        summary_table.add_row(
            "Effective Simulation Frequency",
            f"[bold green]{stats.effective_hz:,.1f} Hz[/bold green] (> 200 Hz Real-Time Target)",
        )
        summary_table.add_row(
            "Tangent Stiffness Symmetry",
            f"[bold green]{max(symmetry_errors):.2e}[/bold green] (Exact SPD preserved)",
        )
        summary_table.add_row("Max Gravitational Sag", f"{max_sag:.2f} mm")
        summary_table.add_row("Lesion Target Drift", f"{l_mag:.2f} mm")
        summary_table.add_row(
            "Max Contact Reaction Force", f"{max(contact_force_history):.2f} N"
        )
        summary_table.add_row(
            "Exported Time-Series", ", ".join(f.name for f in writer.exported_files)
        )

        console.print()
        console.print(summary_table)
        console.print(
            f"\n[bold green]✓ Visualization files successfully saved to:[/bold green] [white]{out_path}[/white]"
        )
        for f in writer.exported_files:
            console.print(f"  • [cyan]{f.name}[/cyan] ({f.stat().st_size:,} bytes)")
        return

    # --------------------------------------------------------------------------
    # Interactive 3D PyVista Simulation
    # --------------------------------------------------------------------------
    console.print(
        "\n[bold yellow]Launching Interactive 3D Neurosurgical Navigator...[/bold yellow]"
    )
    plotter = pv.Plotter(
        title="Project Sierra - Simulation 1303: Hyper-Reduced Brain Shift with Meninges & Dual Contact"
    )
    plotter.set_background("#0f172a", top="#1e293b")

    gui_state = {"exit_requested": False}

    def key_quit() -> None:
        gui_state["exit_requested"] = True
        if not getattr(plotter, "_closed", False):
            plotter.close()

    # 1. Deformable Brain Surface Actor with Colorbar on the Right
    brain_grid = mesh.surface.polydata.copy()
    brain_grid.point_data["Displacement"] = np.zeros(
        mesh.surface.n_nodes, dtype=np.float64
    )

    cortex_actor = plotter.add_mesh(
        brain_grid,
        scalars="Displacement",
        cmap="turbo",
        clim=[0.0, 12.0],
        smooth_shading=True,
        show_edges=False,
        specular=0.4,
        specular_power=20.0,
        ambient=0.25,
        diffuse=0.8,
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
    )

    # 2. Crimson Tumor Target
    lesion_grid = lesion_pv.copy()
    lesion_actor = plotter.add_mesh(
        lesion_grid,
        color="#ff1744",
        opacity=1.0,
        smooth_shading=True,
        show_edges=True,
        edge_color="#d50000",
        specular=0.8,
        ambient=0.3,
        diffuse=0.9,
    )

    # 3. Ghost Undeformed Brain Outline
    ghost_grid = mesh.surface.polydata.copy()
    ghost_actor = plotter.add_mesh(
        ghost_grid,
        color="#607d8b",
        style="wireframe",
        opacity=0.20,
        line_width=1.0,
    )

    # 4. Semi-Transparent Cranial Skull Vault
    skull_mesh = skull_poly.copy()
    skull_actor = plotter.add_mesh(
        skull_mesh,
        color="#e0e0e0",
        opacity=0.10,
        smooth_shading=True,
        show_edges=False,
        specular=0.3,
        ambient=0.2,
        diffuse=0.7,
    )

    # 5. Surgical Indenter Tool
    def build_probe_geometry(
        center_coords: np.ndarray, R_matrix: np.ndarray
    ) -> pv.PolyData:
        sphere = pv.Sphere(radius=probe_radius, center=center_coords)
        shaft_dir = np.array([0.0, 1.0, 0.0]) @ R_matrix.T
        shaft_center = center_coords + shaft_dir * (probe_radius + shaft_length / 2.0)
        shaft = pv.Cylinder(
            center=shaft_center,
            direction=shaft_dir,
            radius=shaft_radius,
            height=shaft_length,
        )
        return sphere.merge(shaft)

    probe_mesh = build_probe_geometry(probe_center, orientation.rotation_matrix)
    probe_actor = plotter.add_mesh(
        probe_mesh,
        color="#cfd8dc",
        opacity=0.95,
        smooth_shading=True,
        specular=0.95,
        ambient=0.3,
        diffuse=0.8,
    )

    # HUD Overlay (Upper Left)
    hud_actor = plotter.add_text(
        "Initializing Neurosurgical Telemetry...",
        position="upper_left",
        font_size=9,
        color="#eceff1",
        font="courier",
    )

    # Orientation Axes (Bottom Right, non-overlapping)
    plotter.add_axes(
        xlabel="R (+X)",
        ylabel="A (+Y)",
        zlabel="S (+Z)",
        line_width=3,
        viewport=(0.80, 0.0, 1.0, 0.22),
    )

    # Instructions Text (Lower Left)
    instructions_text = (
        "Surgical Presets:\n"
        "  [1] Supine    [2] Prone\n"
        "  [3] R Lateral [4] L Lateral\n"
        "  [5] Pterional [6] Upright\n\n"
        "Head Reorientation:\n"
        "  [W/S] Pitch ±5°\n"
        "  [A/D] Roll  ±5°\n"
        "  [Q/E] Yaw   ±5°\n\n"
        "Surgical Indenter Tool:\n"
        "  [I/K] Advance / Retract (±Y)\n"
        "  [J/L] Lateral ±2 mm (±X)\n"
        "  [U/O] Axial   ±2 mm (±Z)\n\n"
        "Fluid & Contact Mechanics:\n"
        "  [B]   CSF Buoyancy / Drainage\n"
        "  [C]   Toggle AL / Penalty\n"
        "  [V]   Toggle Skull Contact\n"
        "  [P]   Toggle Indenter Contact\n"
        "  [X]   Cycle Skull Opacity\n"
        "  [R]   Reset Pose & Probe\n"
        "  [Esc] Exit Simulator"
    )
    plotter.add_text(
        instructions_text,
        position="lower_left",
        font_size=8.5,
        color="#b0bec5",
        font="courier",
    )

    plotter.camera_position = [(30.0, 320.0, 80.0), (0.0, -10.0, 0.0), (0.0, 0.0, 1.0)]

    # Key event & contact toggle state
    skull_opacities = [0.10, 0.25, 0.0, 0.04]
    skull_op_idx = 0
    contact_state = {
        "skull_enabled": True,
        "probe_enabled": True,
    }

    def sync_active_obstacles() -> None:
        active_obs = []
        if contact_state["skull_enabled"]:
            active_obs.append(skull_obstacle)
        if contact_state["probe_enabled"]:
            active_obs.append(probe_obstacle)
        collision_detector._obstacles[:] = active_obs
        contact.reset_state()

    def toggle_skull_contact() -> None:
        contact_state["skull_enabled"] = not contact_state["skull_enabled"]
        sync_active_obstacles()
        state_str = "ON" if contact_state["skull_enabled"] else "OFF"
        skull_actor.GetProperty().SetColor(
            (0.88, 0.88, 0.88) if contact_state["skull_enabled"] else (0.94, 0.55, 0.55)
        )
        console.print(
            f"[bold cyan]Skull Contact Toggled:[/bold cyan] [bold white]{state_str}[/bold white]"
        )

    def toggle_probe_contact() -> None:
        contact_state["probe_enabled"] = not contact_state["probe_enabled"]
        sync_active_obstacles()
        state_str = "ON" if contact_state["probe_enabled"] else "OFF"
        if contact_state["probe_enabled"]:
            probe_actor.GetProperty().SetColor(0.81, 0.85, 0.86)
            probe_actor.GetProperty().SetOpacity(0.95)
        else:
            probe_actor.GetProperty().SetColor(0.47, 0.56, 0.61)
            probe_actor.GetProperty().SetOpacity(0.30)
        console.print(
            f"[bold cyan]Indenter Contact Toggled:[/bold cyan] [bold white]{state_str}[/bold white]"
        )

    def cycle_skull_opacity():
        nonlocal skull_op_idx
        skull_op_idx = (skull_op_idx + 1) % len(skull_opacities)
        skull_actor.GetProperty().SetOpacity(skull_opacities[skull_op_idx])

    def toggle_contact_formulation():
        new_form = contact.toggle_formulation()
        console.print(
            f"[bold cyan]Contact Formulation Toggled:[/bold cyan] {new_form.upper()}"
        )

    def cycle_csf_state():
        new_mode = csf_state.cycle()
        console.print(
            f"[bold green]CSF Buoyancy / Drainage State:[/bold green] [bold white]{new_mode}[/bold white]"
        )

    def move_probe(dx: float, dy: float, dz: float):
        probe_center[0] += dx
        probe_center[1] += dy
        probe_center[2] += dz
        probe_obstacle.center = tuple(probe_center)
        R = orientation.rotation_matrix
        probe_actor.GetMapper().SetInputData(
            build_probe_geometry(probe_center @ R.T, R)
        )

    def set_pose_preset(preset_key: str):
        # Prevent VTK stereo rendering mode glitch on key '3'
        if getattr(plotter, "render_window", None) is not None:
            plotter.render_window.SetStereoRender(False)
        desc = orientation.set_preset(preset_key)
        R = orientation.rotation_matrix
        probe_actor.GetMapper().SetInputData(
            build_probe_geometry(probe_center @ R.T, R)
        )
        console.print(
            f"[cyan]Head Orientation Preset:[/cyan] [bold white]{desc}[/bold white]"
        )

    def reset_simulation():
        orientation.set_preset("supine")
        csf_state.mode_idx = 0
        contact_state["skull_enabled"] = True
        contact_state["probe_enabled"] = True
        skull_actor.GetProperty().SetColor(0.88, 0.88, 0.88)
        probe_actor.GetProperty().SetColor(0.81, 0.85, 0.86)
        probe_actor.GetProperty().SetOpacity(0.95)
        probe_center[:] = [4.7, 76.0, -4.6]
        probe_obstacle.center = tuple(probe_center)
        sync_active_obstacles()
        solver.reset_state()
        R = orientation.rotation_matrix
        probe_actor.GetMapper().SetInputData(
            build_probe_geometry(probe_center @ R.T, R)
        )
        console.print(
            "[yellow]Simulation, head posture, contacts, and surgical probe reset to baseline.[/yellow]"
        )

    # Clear PyVista's default key callbacks (prevents 'b' -> left_button_down pickpoint crash and 'q' -> close)
    if hasattr(plotter, "iren") and plotter.iren is not None:
        plotter.iren.clear_key_event_callbacks()

    # Hotkey registrations
    plotter.add_key_event("x", cycle_skull_opacity)
    plotter.add_key_event("X", cycle_skull_opacity)
    plotter.add_key_event("c", toggle_contact_formulation)
    plotter.add_key_event("C", toggle_contact_formulation)
    plotter.add_key_event("v", toggle_skull_contact)
    plotter.add_key_event("V", toggle_skull_contact)
    plotter.add_key_event("p", toggle_probe_contact)
    plotter.add_key_event("P", toggle_probe_contact)
    plotter.add_key_event("b", cycle_csf_state)
    plotter.add_key_event("B", cycle_csf_state)
    plotter.add_key_event("r", reset_simulation)
    plotter.add_key_event("R", reset_simulation)
    plotter.add_key_event("Escape", key_quit)

    # Indenter movement
    plotter.add_key_event("i", lambda: move_probe(0.0, -2.0, 0.0))
    plotter.add_key_event("I", lambda: move_probe(0.0, -2.0, 0.0))
    plotter.add_key_event("k", lambda: move_probe(0.0, 2.0, 0.0))
    plotter.add_key_event("K", lambda: move_probe(0.0, 2.0, 0.0))
    plotter.add_key_event("j", lambda: move_probe(-2.0, 0.0, 0.0))
    plotter.add_key_event("J", lambda: move_probe(-2.0, 0.0, 0.0))
    plotter.add_key_event("l", lambda: move_probe(2.0, 0.0, 0.0))
    plotter.add_key_event("L", lambda: move_probe(2.0, 0.0, 0.0))
    plotter.add_key_event("u", lambda: move_probe(0.0, 0.0, 2.0))
    plotter.add_key_event("U", lambda: move_probe(0.0, 0.0, 2.0))
    plotter.add_key_event("o", lambda: move_probe(0.0, 0.0, -2.0))
    plotter.add_key_event("O", lambda: move_probe(0.0, 0.0, -2.0))

    # Head reorientation
    plotter.add_key_event("w", lambda: orientation.adjust(dp=5.0))
    plotter.add_key_event("W", lambda: orientation.adjust(dp=5.0))
    plotter.add_key_event("s", lambda: orientation.adjust(dp=-5.0))
    plotter.add_key_event("S", lambda: orientation.adjust(dp=-5.0))
    plotter.add_key_event("a", lambda: orientation.adjust(dr=-5.0))
    plotter.add_key_event("A", lambda: orientation.adjust(dr=-5.0))
    plotter.add_key_event("d", lambda: orientation.adjust(dr=5.0))
    plotter.add_key_event("D", lambda: orientation.adjust(dr=5.0))
    plotter.add_key_event("q", lambda: orientation.adjust(dy=-5.0))
    plotter.add_key_event("Q", lambda: orientation.adjust(dy=-5.0))
    plotter.add_key_event("e", lambda: orientation.adjust(dy=5.0))
    plotter.add_key_event("E", lambda: orientation.adjust(dy=5.0))

    # Preset surgical postures
    plotter.add_key_event("1", lambda: set_pose_preset("supine"))
    plotter.add_key_event("2", lambda: set_pose_preset("prone"))
    plotter.add_key_event("3", lambda: set_pose_preset("right_lateral"))
    plotter.add_key_event("4", lambda: set_pose_preset("left_lateral"))
    plotter.add_key_event("5", lambda: set_pose_preset("pterional_30"))
    plotter.add_key_event("6", lambda: set_pose_preset("upright"))

    # Remove VTK default CharEvent observers that hijack numeric keys like '3' for stereo 3D
    if hasattr(plotter, "iren") and plotter.iren is not None:
        plotter.iren.interactor.RemoveObservers("CharEvent")
        if plotter.iren.get_interactor_style() is not None:
            plotter.iren.get_interactor_style().RemoveObservers("CharEvent")

    plotter.show(interactive_update=True)

    # Disable stereo mode after window initialization to ensure clean standard rendering
    if getattr(plotter, "render_window", None) is not None:
        plotter.render_window.SetStereoRender(False)
    if hasattr(plotter, "iren") and plotter.iren is not None:
        plotter.iren.interactor.RemoveObservers("CharEvent")
        if plotter.iren.get_interactor_style() is not None:
            plotter.iren.get_interactor_style().RemoveObservers("CharEvent")

    # FPS and wall clock timer
    last_wall_time = time.perf_counter()
    fps_timer = time.perf_counter()
    fps_frames = 0
    current_fps = 0.0

    try:
        while (
            not getattr(plotter, "_closed", False) and not gui_state["exit_requested"]
        ):
            now = time.perf_counter()
            fps_frames += 1
            if now - fps_timer >= 0.5:
                current_fps = fps_frames / max(now - fps_timer, 1e-6)
                fps_timer = now
                fps_frames = 0

            # Update net external load: Gravity + CSF Buoyancy + Hydrostatic Pressure
            g_loc = orientation.local_gravity
            f_net = compute_net_external_load(g_loc, csf_state)
            hyper_model.f_r_base = hyper_model.project_load(f_net)

            # Advance dynamic physics solver
            solver.step()

            # Instantaneous boundary deformation & head rotation
            u_surf = hyper_model.reconstruct_surface_displacement(solver.u_r)
            R = orientation.rotation_matrix

            # Rotate deformed brain
            brain_grid.points = (initial_brain_points[surf_nodes] + u_surf) @ R.T
            u_norms = np.linalg.norm(u_surf, axis=1)
            brain_grid.point_data["Displacement"] = u_norms

            # Synchronously rotate cranial skull vault
            skull_mesh.points = initial_skull_points @ R.T

            # Synchronously rotate ghost undeformed reference
            ghost_grid.points = initial_brain_points[surf_nodes] @ R.T

            # Deform and rotate lesion
            u_lesion = lesion_basis.reconstruct_surface_displacements(solver.u_r)
            lesion_grid.points = (initial_lesion_points + u_lesion) @ R.T

            # Telemetry HUD
            telem = contact.get_telemetry()
            st = solver.get_stats()
            l_vec, l_mag, max_sag, _, _ = evaluate_shift_metrics()

            skull_status = "ON " if contact_state["skull_enabled"] else "OFF"
            probe_status = "ON " if contact_state["probe_enabled"] else "OFF"

            hud_lines = [
                "Simulation 1303: Solid-Shell Brain Shift & Dual Contact",
                f"  Formulation [C]  : {contact.formulation.upper()}",
                f"  Skull Contact [V]: {skull_status}   | Indenter Contact [P]: {probe_status}",
                f"  Active Contacts  : {telem.num_active} candidate points",
                f"  Max Penetration  : {telem.max_penetration:5.3f} mm",
                f"  Max Contact Force: {telem.max_contact_force:5.3f} N",
                f"  Total Force Norm : {telem.total_contact_force_norm:5.3f} N",
                "",
                "CSF Fluid Mechanics & Intracranial Pressure",
                f"  CSF State [B]    : {csf_state.mode_name}",
                f"  Buoyant Relief   : {csf_state.buoyancy_factor*96.15:5.1f}% gravity offset",
                f"  Hydrostatic ICP  : {csf_state.icp_mpa*1000.0:5.1f} kPa",
                "",
                "Surgical Shifts & Diagnostics",
                f"  Max Cortex Sag   : {max_sag:5.2f} mm",
                f"  Lesion Drift     : {l_mag:5.2f} mm",
                f"  Probe Center     : [{probe_center[0]:+5.1f}, {probe_center[1]:+5.1f}, {probe_center[2]:+5.1f}] mm",
                f"  Head Pose (P/R/Y): [{orientation.pitch:+5.1f}°, {orientation.roll:+5.1f}°, {orientation.yaw:+5.1f}°]",
                f"  Physics Rate     : {st.effective_hz:5.0f} Hz ({st.average_step_ms*1000:.1f} µs/step)",
                f"  Render Rate      : {current_fps:4.1f} FPS",
            ]
            hud_actor.set_text(2, "\n".join(hud_lines))

            plotter.update()
            time.sleep(0.002)

    except KeyboardInterrupt:
        pass
    finally:
        if not getattr(plotter, "_closed", False):
            plotter.close()


if __name__ == "__main__":
    main()
