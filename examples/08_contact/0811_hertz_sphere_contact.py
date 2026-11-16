"""Example 0811: Classical 3D Hertzian Spherical Contact Benchmark.

Milestone 1.1 of Loop 3 Contact Mechanics Overhaul in Project Sierra.
---------------------------------------------------------------------
Validates node-to-surface penalty and augmented-Lagrangian contact against
the exact analytical closed-form solution of Heinrich Hertz (1882) for an
elastic sphere in frictionless contact with a rigid flat plane.

Governing Analytical Relations (Hertz Ground Truth):
----------------------------------------------------
1. Effective Modulus:
       E* = E_1 / (1 - nu_1^2)
2. Normal Indentation Depth delta [length]:
       delta = a^2 / R
3. Contact Circle Radius a [length]:
       a = (3 * F_N * R / (4 * E*))^(1/3) = sqrt(R * delta)
4. Peak Normal Contact Pressure p_0 [stress]:
       p_0 = 3 * F_N / (2 * pi * a^2)
5. Hemispherical Pressure Profile p(r) [stress]:
       p(r) = p_0 * sqrt(max(0, 1 - (r/a)^2))
6. Total Normal Contact Force F_N [force]:
       F_N = (4/3) * E* * sqrt(R) * delta^(3/2)

Outputs:
--------
- Terminal: Rich summary table with FE vs. Analytical comparison.
- Results: ParaView VTU/PVD dataset with Displacements, ContactForce, and ContactPressure.
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

from psierra.material import IsotropicElasticParameters, NeoHookean
from psierra.element import Tet4Linear
from psierra.mesh import VolumetricMesh
from psierra.mesher import TetrahedralMesher
from psierra.domain import SolidDomain
from psierra.boundary.contact import RigidContact
from psierra.collision.shapes import RigidPlane
from psierra.boundary.dirichlet import DirichletManager, DOFConstraint
from psierra.analysis.nonlinear_static import NonlinearStaticAnalysis
from psierra.contact.hertz import (
    HertzSphereContact,
    compute_nodal_tributary_areas,
    reconstruct_nodal_pressures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0811: Hertzian Spherical Contact Benchmark"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=10.0,
        help="Sphere radius R in mm (default: 10.0 mm)",
    )
    parser.add_argument(
        "--modulus",
        type=float,
        default=2.0,
        help="Young's modulus E in MPa / N/mm^2 (default: 2.0 MPa)",
    )
    parser.add_argument(
        "--poisson",
        type=float,
        default=0.3,
        help="Poisson's ratio nu (default: 0.3)",
    )
    parser.add_argument(
        "--indentation",
        type=float,
        default=0.15,
        help="Prescribed downward vertical indentation delta in mm (default: 0.15 mm)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=20000.0,
        help="Normal penalty contact stiffness in N/mm (default: 20000.0 N/mm)",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        choices=["penalty", "augmented_lagrangian"],
        default="penalty",
        help="Contact constraint formulation (default: penalty)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of incremental load steps (default: 5)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.08,
        help="Relative tetrahedral edge length factor (default: 0.08)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/08_contact/0811_hertz_sphere_contact",
        help="Directory to save simulation results (default: results/08_contact/0811_hertz_sphere_contact)",
    )
    return parser.parse_args()


def run_hertz_sphere_benchmark(
    radius: float = 10.0,
    modulus: float = 2.0,
    poisson: float = 0.3,
    indentation: float = 0.15,
    kp: float = 20000.0,
    formulation: str = "penalty",
    n_steps: int = 5,
    resolution: float = 0.08,
    output_dir: str | Path = "results/08_contact/0811_hertz_sphere_contact",
    verbose: bool = True,
) -> dict[str, float | str | bool]:
    """Execute the spherical indentation benchmark and validate against analytical Hertz theory."""
    console = Console()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # 1. Initialize exact analytical ground truth
    hertz = HertzSphereContact(radius=radius, E_1=modulus, nu_1=poisson)

    # 2. Generate hemispherical volumetric mesh
    # Sphere center at (0, 0, R), pole at (0, 0, 0), flat top at z = R
    sphere_geom = pv.Sphere(
        radius=radius,
        center=(0.0, 0.0, radius),
        theta_resolution=36,
        phi_resolution=36,
    )
    # Clip to lower hemisphere
    hemisphere = sphere_geom.clip(normal=(0, 0, 1), origin=(0, 0, radius))

    mesher = TetrahedralMesher(order=1, edge_length_fac=resolution, quiet=True)
    vol_mesh = mesher.generate(hemisphere)
    surf_mesh = vol_mesh.surface

    if verbose:
        console.print(
            f"  [cyan]Generated mesh:[/cyan] {vol_mesh.n_nodes} nodes, {vol_mesh.n_elements} Tet4 elements"
        )

    # 3. Constitutive model and SolidDomain
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=modulus,
        poissons_ratio=poisson,
        mass_density=1.0e-6,
    )
    mat = NeoHookean(params)
    elem = Tet4Linear(mat)
    domain = SolidDomain(vol_mesh, mat, elem)

    # 4. Rigid plane obstacle at z = 0
    plane = RigidPlane(origin=(0, 0, 0), normal=(0, 0, 1))
    contact = RigidContact(
        slave=domain,
        master=plane,
        k_penalty=kp,
        formulation=formulation,
    )

    # 5. Boundary conditions: indent flat top downwards by delta
    top_nodes = np.flatnonzero(vol_mesh.nodes[:, 2] >= radius - 0.15)
    manager = DirichletManager(domain.n_dofs)
    for node in top_nodes:
        # Prevent lateral rigid translation
        manager.add_constraint(DOFConstraint([3 * node, 3 * node + 1], 0.0))
        # Prescribe vertical indentation
        manager.add_constraint(DOFConstraint([3 * node + 2], -indentation))

    # 6. Nonlinear static equilibrium solve
    analysis = NonlinearStaticAnalysis(domain=domain)
    res = analysis.solve(
        dirichlet=manager,
        contact=contact,
        n_steps=n_steps,
        nr_iterations=20,
        nr_tol=1e-4,
        save_directory=out_path,
        output_name="hertz_sphere_contact",
        save_history=False,
        verbose=False,
    )

    solve_time = time.perf_counter() - t0

    # 7. Contact post-processing & pressure reconstruction
    f_contact = contact.evaluate_forces().reshape(-1, 3)
    contact_node_mask = f_contact[:, 2] > 1e-5
    contact_indices = np.flatnonzero(contact_node_mask)

    total_Fn_fe = float(np.sum(f_contact[contact_indices, 2]))

    # Surface nodal tributary areas and pressure
    trib_areas = compute_nodal_tributary_areas(surf_mesh, vol_mesh.n_nodes)
    pressures = reconstruct_nodal_pressures(f_contact, trib_areas, normal_component=2, direction=1.0)

    # Contact radius from FE current deformed coordinates
    current_coords = domain.current_nodes
    r_coords = np.sqrt(current_coords[:, 0] ** 2 + current_coords[:, 1] ** 2)
    contact_radii = r_coords[contact_indices]
    a_fe = float(np.max(contact_radii)) if len(contact_radii) > 0 else 0.0
    p0_fe = float(np.max(pressures)) if len(contact_indices) > 0 else 0.0

    # Actual bottom pole penetration into plane
    pole_idx = int(np.argmin(vol_mesh.nodes[:, 2]))
    delta_pole_fe = float(-current_coords[pole_idx, 2])
    # The actual indentation relative to the rigid plane is the prescribed top displacement
    # minus the bulk compression of the hemisphere
    delta_eff_fe = indentation - (current_coords[pole_idx, 2] + indentation)

    # 8. Analytical Hertz calculations for the FE normal force
    a_ana = hertz.contact_radius(total_Fn_fe)
    p0_ana = hertz.peak_pressure(total_Fn_fe)
    delta_ana = hertz.penetration(total_Fn_fe)
    area_fe = float(np.pi * a_fe**2)
    area_ana = hertz.contact_area(total_Fn_fe)

    # Goodness-of-fit of pressure profile
    fit_metrics = hertz.evaluate_fit(contact_radii, pressures[contact_indices], total_Fn_fe)

    # Errors
    err_a = abs(a_fe - a_ana) / a_ana if a_ana > 0 else 0.0
    err_p0 = abs(p0_fe - p0_ana) / p0_ana if p0_ana > 0 else 0.0
    err_area = abs(area_fe - area_ana) / area_ana if area_ana > 0 else 0.0

    # Export VTU with contact fields
    ugrid = vol_mesh.ugrid.copy()
    ugrid.point_data["Displacement"] = domain.displacement
    ugrid.point_data["ContactForce"] = f_contact
    ugrid.point_data["ContactPressure"] = pressures
    ugrid.point_data["TributaryArea"] = trib_areas
    vtu_file = out_path / "hertz_sphere_fields.vtu"
    ugrid.save(str(vtu_file))

    results = {
        "converged": res.converged,
        "n_contact_nodes": len(contact_indices),
        "total_Fn_fe": total_Fn_fe,
        "contact_radius_fe": a_fe,
        "contact_radius_ana": a_ana,
        "error_contact_radius": err_a,
        "peak_pressure_fe": p0_fe,
        "peak_pressure_ana": p0_ana,
        "error_peak_pressure": err_p0,
        "contact_area_fe": area_fe,
        "contact_area_ana": area_ana,
        "error_contact_area": err_area,
        "pressure_fit_r2": fit_metrics["r2"],
        "pressure_fit_rmse": fit_metrics["rmse"],
        "runtime_s": solve_time,
        "vtu_path": str(vtu_file),
    }

    if verbose:
        table = Table(
            title="[bold green]Hertz Spherical Contact: FE Simulation vs. Analytical Ground Truth[/bold green]",
            header_style="bold magenta",
        )
        table.add_column("Physical Metric", style="cyan")
        table.add_column("FE Numerical", justify="right")
        table.add_column("Hertz Analytical", justify="right")
        table.add_column("Relative Error", justify="right", style="bold yellow")

        table.add_row(
            "Total Normal Force F_N",
            f"{total_Fn_fe:.4f} N",
            f"{total_Fn_fe:.4f} N",
            "Reference",
        )
        table.add_row(
            "Contact Radius a",
            f"{a_fe:.4f} mm",
            f"{a_ana:.4f} mm",
            f"{err_a * 100:.2f}%",
        )
        table.add_row(
            "Peak Pressure p_0",
            f"{p0_fe:.4f} MPa",
            f"{p0_ana:.4f} MPa",
            f"{err_p0 * 100:.2f}%",
        )
        table.add_row(
            "Contact Area A",
            f"{area_fe:.4f} mm²",
            f"{area_ana:.4f} mm²",
            f"{err_area * 100:.2f}%",
        )
        table.add_row(
            "Pressure Profile R²",
            f"{fit_metrics['r2']:.4f}",
            "1.0000",
            f"RMSE: {fit_metrics['rmse']:.4f} MPa",
        )

        console.print()
        console.print(table)
        console.print(f"[dim]Output VTU saved to: {vtu_file}[/dim]\n")

    return results


def main() -> None:
    args = parse_args()
    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Example 0811: Classical 3D Hertzian Contact Benchmark[/bold cyan]\n"
            f"[dim]Radius: {args.radius} mm | E = {args.modulus} MPa | nu = {args.poisson} | Indentation = {args.indentation} mm[/dim]",
            title="[bold blue]project-sierra — Analytical Contact Verification[/bold blue]",
            border_style="cyan",
        )
    )

    run_hertz_sphere_benchmark(
        radius=args.radius,
        modulus=args.modulus,
        poisson=args.poisson,
        indentation=args.indentation,
        kp=args.kp,
        formulation=args.formulation,
        n_steps=args.steps,
        resolution=args.resolution,
        output_dir=args.output_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
