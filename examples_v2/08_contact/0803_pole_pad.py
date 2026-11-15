"""
Example 0803: Curved Contact Mechanics — Deformable Annular Pad Impact and Sliding on Pole.

Demonstrates:
1. Contact mechanics on curved cylindrical interfaces:
   - Floating annular pad: data/pad1/mesh/pad1.vtk
   - Center pole: data/pole1/mesh/pole1.vtk
2. Configurable pole compliance:
   - Rigid pole: Stationary obstacle with RayMeshObstacle.
   - Deformable pole: Full two-way MultiBodySystem continuum coupling.
3. 3D Active-Set Coulomb Stick-Slip Friction:
   - Lateral gravity (gx = -9810 mm/s²) establishes normal contact.
   - Axial gravity (gy = -2000 mm/s²) drives sliding along the cylindrical surface.
4. Generalized-alpha dynamic time stepping with ParaView export.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4TotalLagrangian
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.multibody import MultiBodySystem
from psierra.boundary import (
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
    RigidContact,
)
from psierra.collision import RayMeshObstacle
from psierra.damping import RayleighDamping
from psierra.analysis import NonlinearDynamicAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0803: Curved Pad on Pole Contact Mechanics"
    )
    parser.add_argument(
        "--pole",
        type=str,
        choices=["rigid", "deformable"],
        default="rigid",
        help="Pole representation: rigid obstacle or deformable continuum (default: rigid)",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.25,
        help="Coulomb friction coefficient mu (default: 0.25)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of dynamic time steps (default: 20)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="Time step size in seconds (default: 0.002 s = 500 Hz)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=5000.0,
        help="Normal penalty contact stiffness in N/mm (default: 5000.0 N/mm)",
    )
    parser.add_argument(
        "--cdamp",
        type=float,
        default=0.005,
        help="Contact normal velocity damping in N·s/mm (default: 0.005 N·s/mm)",
    )
    parser.add_argument(
        "--gx",
        type=float,
        default=-9810.0,
        help="Lateral gravity acceleration in mm/s² (default: -9810.0)",
    )
    parser.add_argument(
        "--gy",
        type=float,
        default=-2000.0,
        help="Axial gravity acceleration in mm/s² (default: -2000.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/08_contact/0803_pole_pad",
        help="Base output directory (default: results/08_contact/0803_pole_pad)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comparative study between rigid and deformable pole contact",
    )
    return parser.parse_args()


def run_pole_pad_case(
    pole_mode: str,
    mu: float,
    n_steps: int,
    dt: float,
    kp: float,
    cdamp: float,
    gx: float,
    gy: float,
    base_out: Path,
    pad_file: Path,
    pole_file: Path,
    verbose: bool = True,
) -> dict[str, str | float]:
    console = Console()
    case_tag = f"pad_{pole_mode}_pole"
    run_out_dir = base_out / case_tag
    run_out_dir.mkdir(parents=True, exist_ok=True)

    pad_mesh = VolumetricMesh(pad_file)
    pole_mesh = VolumetricMesh(pole_file)

    # Pad material (compliant elastomer E = 1000 MPa, nu = 0.30, rho = 1.2e-6 tonne/mm³)
    params_pad = IsotropicElasticParameters()
    params_pad.set_from_youngs_poisson(1000.0, 0.30, 1.2e-6)
    mat_pad = NeoHookean(params_pad)
    elem_pad = Tet4TotalLagrangian(mat_pad)
    pad_domain = SolidDomain(pad_mesh, mat_pad, elem_pad)

    # Candidate contact nodes on Pad inner hole (r < 65 mm)
    surf_nodes = pad_mesh.surface.node_indices
    coords = pad_mesh.nodes[surf_nodes]
    r_surf = np.sqrt(coords[:, 0] ** 2 + coords[:, 2] ** 2)
    inner_nodes = surf_nodes[r_surf < 65.0]

    damping = RayleighDamping(alpha=1.0, beta=0.002)

    if pole_mode == "rigid":
        # Rigid pole obstacle
        obstacle = RayMeshObstacle(pole_file)
        contact = RigidContact(
            slave=pad_domain,
            master=obstacle,
            k_penalty=kp,
            k_tangential=kp,
            c_damping=cdamp,
            friction=mu,
            slave_nodes=inner_nodes,
            formulation="penalty",
        )
        dirichlet = DirichletManager(pad_domain.n_dofs)
        gravity = GravityLoad(acceleration=(gx, gy, 0.0))
        M = pad_domain.assemble_mass_matrix()
        f_ext = gravity.compute_force_vector(M)

        solver = NonlinearDynamicAnalysis(pad_domain, damping=damping, spectral_radius=0.8)
        sim_target = pad_domain
        run_contact_arg = contact
    else:
        # Deformable pole
        params_pole = IsotropicElasticParameters()
        params_pole.set_from_youngs_poisson(20000.0, 0.30, 7.8e-6)
        mat_pole = NeoHookean(params_pole)
        elem_pole = Tet4TotalLagrangian(mat_pole)
        pole_domain = SolidDomain(pole_mesh, mat_pole, elem_pole)

        system = MultiBodySystem({"Pole": pole_domain, "Pad": pad_domain})
        contact = system.add_contact_pair(
            slave="Pad",
            master="Pole",
            slave_nodes=inner_nodes,
            k_penalty=kp,
            k_tangential=kp,
            c_damping=cdamp,
            friction=mu,
            formulation="penalty",
            smoothed_normals=True,
            smoothing_weight="angle",
        )

        pole_inner_nodes = np.flatnonzero(
            np.sqrt(pole_mesh.nodes[:, 0] ** 2 + pole_mesh.nodes[:, 2] ** 2) < 20.0
        )
        pole_offset = int(system._dof_offsets[0])
        dirichlet = DirichletManager(system.n_dofs)
        dirichlet.add_constraint(
            DirichletConstraint(pole_inner_nodes, components=(0, 1, 2), dof_offset=pole_offset)
        )

        gravity = GravityLoad(acceleration=(gx, gy, 0.0))
        M = system.assemble_mass_matrix()
        f_ext = gravity.compute_force_vector(M)

        solver = NonlinearDynamicAnalysis(system, damping=damping, spectral_radius=0.8)
        sim_target = system
        run_contact_arg = None  # encapsulated in MultiBodySystem

    if verbose:
        console.print(f"  • Simulating: [bold cyan]{case_tag}[/bold cyan] ({n_steps} steps, mu={mu})")

    t0 = time.perf_counter()
    if run_contact_arg is not None:
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            contact=run_contact_arg,
            dt=dt,
            n_steps=n_steps,
            nr_iterations=15,
            nr_tol=1e-3,
            save_directory=run_out_dir,
            output_name="result",
            save_interval=1,
            verbose=False,
        )
    else:
        res = solver.solve(
            dirichlet=dirichlet,
            f_ext=f_ext,
            dt=dt,
            n_steps=n_steps,
            nr_iterations=15,
            nr_tol=1e-3,
            save_directory=run_out_dir,
            output_name="result",
            save_interval=1,
            verbose=False,
        )
    wall_time = time.perf_counter() - t0

    pad_disp = pad_domain.displacement
    max_disp = float(np.linalg.norm(pad_disp, axis=1).max())
    axial_slide = float(abs(pad_disp[:, 1].min()))

    return {
        "case": case_tag,
        "pole": pole_mode.capitalize(),
        "friction": mu,
        "converged": res.converged,
        "steps": f"{res.steps_completed}/{n_steps}",
        "max_pad_disp_mm": max_disp,
        "axial_sliding_mm": axial_slide,
        "runtime_s": wall_time,
        "output_dir": str(run_out_dir),
    }


def main() -> None:
    args = parse_args()
    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Example 0803: Curved Pad on Pole Contact Mechanics[/bold cyan]\n"
            f"[dim]Pole: {args.pole.upper()} | Friction mu: {args.friction:.2f} | Penalty kp: {args.kp:.1e} N/mm[/dim]",
            title="[bold blue]project-sierra — Curved Contact[/bold blue]",
            border_style="cyan",
        )
    )

    pad_path = Path("data/pad1/mesh/pad1.vtk")
    pole_path = Path("data/pole1/mesh/pole1.vtk")
    if not pad_path.exists() or not pole_path.exists():
        raise FileNotFoundError("Pad or pole mesh file not found.")

    base_out = Path(args.output_dir)

    if args.all:
        console.print("\n[bold yellow]Executing Comparative Pole Compliance Study...[/bold yellow]")
        cases = ["rigid", "deformable"]
        results = []
        for p_mode in cases:
            res = run_pole_pad_case(
                pole_mode=p_mode,
                mu=args.friction,
                n_steps=min(args.steps, 10),
                dt=args.dt,
                kp=args.kp,
                cdamp=args.cdamp,
                gx=args.gx,
                gy=args.gy,
                base_out=base_out,
                pad_file=pad_path,
                pole_file=pole_path,
                verbose=True,
            )
            results.append(res)

        table = Table(title="Example 0803: Pole-Pad Contact Benchmark", border_style="cyan")
        table.add_column("Case Tag", style="bold white")
        table.add_column("Pole Mode", style="cyan")
        table.add_column("Steps", style="yellow")
        table.add_column("Max Disp (mm)", style="green")
        table.add_column("Axial Slide (mm)", style="green")
        table.add_column("Runtime (s)", style="blue")

        for r in results:
            table.add_row(
                str(r["case"]),
                str(r["pole"]),
                str(r["steps"]),
                f"{r['max_pad_disp_mm']:.3f}",
                f"{r['axial_sliding_mm']:.3f}",
                f"{r['runtime_s']:.2f}",
            )
        console.print(table)

    else:
        console.print(f"\n[bold yellow]Executing Canonical Contact Run ({args.pole.upper()} POLE)...[/bold yellow]")
        res = run_pole_pad_case(
            pole_mode=args.pole,
            mu=args.friction,
            n_steps=args.steps,
            dt=args.dt,
            kp=args.kp,
            cdamp=args.cdamp,
            gx=args.gx,
            gy=args.gy,
            base_out=base_out,
            pad_file=pad_path,
            pole_file=pole_path,
            verbose=True,
        )

        table = Table(title=f"Example 0803: {res['case']}", border_style="cyan")
        table.add_column("Parameter", style="bold white")
        table.add_column("Value", style="cyan")
        table.add_row("Pole Representation", str(res["pole"]))
        table.add_row("Coulomb Friction Coefficient", f"{res['friction']:.2f}")
        table.add_row("Time Steps Completed", str(res["steps"]))
        table.add_row("Convergence Status", "[bold green]Converged[/bold green]" if res["converged"] else "[bold red]Diverged[/bold red]")
        table.add_row("Max Pad Displacement", f"{res['max_pad_disp_mm']:.4f} mm")
        table.add_row("Axial Sliding Distance", f"{res['axial_sliding_mm']:.4f} mm")
        table.add_row("Solver Runtime", f"{res['runtime_s']:.2f} s")
        table.add_row("ParaView Output Directory", str(res["output_dir"]))
        console.print(table)
        console.print(f"\n[bold green]✓ Pole-pad contact simulation complete. Dataset saved to:[/bold green] [cyan]{res['output_dir']}[/cyan]\n")


if __name__ == "__main__":
    main()
