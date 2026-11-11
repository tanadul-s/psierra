"""
Simulation 0808: Exact Coulomb Stick-Slip Friction — Dynamic Sliding Block on Rigid Floor.
========================================================================================
Demonstrates:
1. 3D Active-Set Coulomb Friction Formulation via Elastoplastic Return Mapping:
   - Yield criterion: Phi_trial = ||f_T^trial|| - mu * p_N <= 0
   - Stick regime: zero relative slip rate, elastic tangential spring force
   - Slip regime: collinear sliding opposing relative motion with ||f_T|| = mu * p_N
2. Consistent Algorithmic Elastoplastic Tangent Operator:
   - Zero stiffness along sliding direction (plastic flow direction: K_TT * s = 0)
   - Consistent normal-tangential coupling K_TN = mu * k_N * (s (x) n)
3. Dynamic Energy Dissipation and Deceleration:
   - Initial horizontal velocity v_x braking under frictional interface traction
   - Generalized-alpha time integration (rho_inf = 0.8)
4. High-Fidelity Time-Series Export:
   - Unified XDMF/HDF5 or PVD output in results/08_contact/0808_frictional_stick_slip
   - Live telemetry tables with rich console formatting
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

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import (
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
    RigidContact,
    StickSlipState,
)
from psierra.collision.shapes import RigidPlane
from psierra.damping import RayleighDamping
from psierra.analysis import NonlinearDynamicAnalysis

install(show_locals=False)
console = Console()


def create_sliding_block_mesh(
    origin: tuple[float, float, float] = (0.0, 0.5, 0.0),
    size: tuple[float, float, float] = (20.0, 10.0, 20.0),
) -> VolumetricMesh:
    """Procedurally generates a 5-tetrahedron subdivision of a bounding block."""
    x0, y0, z0 = origin
    dx, dy, dz = size
    corners = np.array([
        [x0, y0, z0],
        [x0 + dx, y0, z0],
        [x0 + dx, y0 + dy, z0],
        [x0, y0 + dy, z0],
        [x0, y0, z0 + dz],
        [x0 + dx, y0, z0 + dz],
        [x0 + dx, y0 + dy, z0 + dz],
        [x0, y0 + dy, z0 + dz],
    ], dtype=np.float64)

    tets = np.array([
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
    ], dtype=np.int64)

    cells = np.hstack([np.full((len(tets), 1), 4, dtype=np.int64), tets]).ravel()
    celltypes = np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, corners)
    return VolumetricMesh(grid)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation 0808: Exact Coulomb Stick-Slip Friction on Rigid Floor"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Dynamic time increment in seconds (default: 0.001 s = 1000 Hz)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of dynamic time steps (default: 30 steps = 0.03 s)",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.35,
        help="Coulomb friction coefficient mu (default: 0.35)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=5000.0,
        help="Normal penalty contact stiffness in N/mm (default: 5000.0 N/mm)",
    )
    parser.add_argument(
        "--kt",
        type=float,
        default=None,
        help="Tangential stick penalty stiffness in N/mm (default: same as kp)",
    )
    parser.add_argument(
        "--cdamp",
        type=float,
        default=0.005,
        help="Contact normal velocity damping in N*s/mm (default: 0.005 N*s/mm)",
    )
    parser.add_argument(
        "--v-init",
        type=float,
        default=250.0,
        help="Initial sliding velocity v_x in mm/s (default: 250.0 mm/s)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="xdmf",
        choices=["xdmf", "pvd"],
        help="Export format for ParaView visualization (default: xdmf)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Output streaming stride (default: 1 step)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/08_contact/0808_frictional_stick_slip",
        help="Output directory (default: results/08_contact/0808_frictional_stick_slip)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    console.print(
        Panel.fit(
            f"[bold cyan]Simulation 0808: Exact Coulomb Stick-Slip Friction[/bold cyan]\n"
            f"[dim]Dynamic sliding block on rigid plane with return mapping & algorithmic tangent[/dim]\n\n"
            f"Friction Coefficient mu  : [yellow]{args.friction}[/yellow]\n"
            f"Normal Penalty kp        : [green]{args.kp:.1e} N/mm[/green]\n"
            f"Tangential Penalty kt    : [green]{args.kt or args.kp:.1e} N/mm[/green]\n"
            f"Time Increment dt        : [magenta]{args.dt * 1000:.2f} ms[/magenta]\n"
            f"Total Steps              : [cyan]{args.steps}[/cyan] (Total time: {args.steps * args.dt:.3f} s)\n"
            f"Initial Sliding Speed    : [yellow]{args.v_init:.1f} mm/s[/yellow]\n"
            f"Export Output            : [white]{args.output_dir}[/white] ([bold]{args.format}[/bold])",
            border_style="cyan",
        )
    )

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Geometry and Material Setup
    # Create block elevated slightly (y = 0.5 mm) above rigid floor (y = 0.0 mm)
    mesh = create_sliding_block_mesh(origin=(0.0, 0.5, 0.0), size=(20.0, 10.0, 20.0))
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(youngs_modulus=5e4, poissons_ratio=0.25, mass_density=2.7e-6)
    material = NeoHookean(params)
    element = Tet4TotalLagrangian(material)
    domain = SolidDomain(mesh, material, element)

    # 2. Initial Velocities: Forward horizontal sliding + small downward velocity
    v_init = np.zeros((mesh.n_nodes, 3), dtype=np.float64)
    v_init[:, 0] = args.v_init  # Sliding along X
    v_init[:, 1] = -50.0        # Downward impact towards floor along Y
    domain.update_state(v=v_init.ravel())

    # 3. Rigid Floor Obstacle (Normal pointing upward +Y)
    floor = RigidPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))

    # 4. Coulomb Frictional Contact Condition
    contact = RigidContact(
        slave=domain,
        master=floor,
        k_penalty=args.kp,
        k_tangential=args.kt or args.kp,
        c_damping=args.cdamp,
        friction=args.friction,
        formulation="penalty",
    )

    # 5. External Gravity Load (Downward -Y)
    gravity = GravityLoad(acceleration=(0.0, -9810.0, 0.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    # Essential boundary conditions: none (free sliding body)
    dirichlet = DirichletManager(domain.n_dofs)

    # 6. Generalized-alpha Dynamic Solver
    damping = RayleighDamping(alpha=0.1, beta=0.0005)
    solver = NonlinearDynamicAnalysis(
        domain=domain,
        damping=damping,
        spectral_radius=0.8,
    )

    start_time = time.perf_counter()
    sol = solver.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        dt=args.dt,
        n_steps=args.steps,
        contact=contact,
        nr_iterations=20,
        nr_tol=1e-4,
        save_directory=out_path,
        output_name="sliding_block",
        format=args.format,
        save_interval=args.save_interval,
        include_stresses=True,
        smooth_to_nodes=True,
        verbose=True,
    )
    elapsed = time.perf_counter() - start_time

    # 7. Summary Diagnostics
    v_final = domain.v.reshape(-1, 3)
    mean_vx_final = float(np.mean(v_final[:, 0]))
    decel_pct = ((args.v_init - mean_vx_final) / args.v_init) * 100.0

    contact_info = contact.get_contact_info()
    max_f_T = float(contact_info.get("max_friction_force", 0.0))
    max_slip = float(contact_info.get("max_slip_distance", 0.0))

    summary_table = Table(title="Simulation 0808 Telemetry Summary", border_style="green")
    summary_table.add_column("Metric", style="bold cyan")
    summary_table.add_column("Value", style="bold white")

    summary_table.add_row("Status", "[bold green]CONVERGED[/bold green]" if sol.converged else "[bold red]FAILED[/bold red]")
    summary_table.add_row("Wall Time", f"{elapsed:.2f} s")
    summary_table.add_row("Steps Completed", f"{sol.steps_completed} / {args.steps}")
    summary_table.add_row("Initial Velocity (X)", f"{args.v_init:.2f} mm/s")
    summary_table.add_row("Final Mean Velocity (X)", f"{mean_vx_final:.2f} mm/s")
    summary_table.add_row("Frictional Deceleration", f"[yellow]{decel_pct:.1f}%[/yellow]")
    summary_table.add_row("Peak Tangential Force", f"{max_f_T:.2f} N")
    summary_table.add_row("Max Cumulative Slip Distance", f"{max_slip:.3f} mm")
    summary_table.add_row("Active Slip Candidates", f"{contact_info.get('num_slip', 0)}")
    summary_table.add_row("Active Stick Candidates", f"{contact_info.get('num_stick', 0)}")
    summary_table.add_row("Saved Export Files", f"{len(sol.time_series_files)} frame(s)")

    console.print()
    console.print(summary_table)

    if sol.time_series_files:
        console.print(
            f"\n[bold green]✓ Visualization files successfully saved to:[/bold green] [white]{out_path}[/white]"
        )


if __name__ == "__main__":
    main()
