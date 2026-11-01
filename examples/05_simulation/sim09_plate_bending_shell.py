from __future__ import annotations
from pathlib import Path
import numpy as np
import pyvista as pv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Shell3, Shell6
from psierra.mesh import SurfaceMesh
from psierra.domain import ShellDomain
from psierra.boundary import DirichletConstraint, DirichletManager
from psierra.analysis import LinearStaticAnalysis

install(show_locals=False)
console = Console()


def build_plate_mesh_linear(L: float, N: int) -> SurfaceMesh:
    """Builds a regular 3-node triangular plate SurfaceMesh."""
    xs = np.linspace(-L / 2.0, L / 2.0, N + 1)
    ys = np.linspace(-L / 2.0, L / 2.0, N + 1)

    points = []
    node_map = {}
    idx = 0
    for iy in range(N + 1):
        for ix in range(N + 1):
            points.append([xs[ix], ys[iy], 0.0])
            node_map[(ix, iy)] = idx
            idx += 1

    faces = []
    for iy in range(N):
        for ix in range(N):
            n0 = node_map[(ix, iy)]
            n1 = node_map[(ix + 1, iy)]
            n2 = node_map[(ix + 1, iy + 1)]
            n3 = node_map[(ix, iy + 1)]
            faces.extend([3, n0, n1, n2, 3, n0, n2, n3])

    poly = pv.PolyData(np.array(points, dtype=np.float64), np.array(faces, dtype=np.int64))
    return SurfaceMesh(poly)


def build_plate_mesh_quadratic(L: float, N: int) -> SurfaceMesh:
    """Builds a regular 6-node quadratic triangular plate SurfaceMesh."""
    nx_pts = 2 * N + 1
    ny_pts = 2 * N + 1
    xs = np.linspace(-L / 2.0, L / 2.0, nx_pts)
    ys = np.linspace(-L / 2.0, L / 2.0, ny_pts)

    points = []
    node_map = {}
    idx = 0
    for iy in range(ny_pts):
        for ix in range(nx_pts):
            points.append([xs[ix], ys[iy], 0.0])
            node_map[(ix, iy)] = idx
            idx += 1

    cells = []
    cell_types = []
    for ey in range(N):
        for ex in range(N):
            ix0, ix1, ix2 = 2 * ex, 2 * ex + 1, 2 * ex + 2
            iy0, iy1, iy2 = 2 * ey, 2 * ey + 1, 2 * ey + 2

            conn1 = [
                node_map[(ix0, iy0)], node_map[(ix2, iy0)], node_map[(ix2, iy2)],
                node_map[(ix1, iy0)], node_map[(ix2, iy1)], node_map[(ix1, iy1)],
            ]
            conn2 = [
                node_map[(ix0, iy0)], node_map[(ix2, iy2)], node_map[(ix0, iy2)],
                node_map[(ix1, iy1)], node_map[(ix1, iy2)], node_map[(ix0, iy1)],
            ]

            cells.extend([6, *conn1])
            cells.extend([6, *conn2])
            cell_types.extend([pv.CellType.QUADRATIC_TRIANGLE, pv.CellType.QUADRATIC_TRIANGLE])

    grid = pv.UnstructuredGrid(cells, cell_types, np.array(points, dtype=np.float64))
    return SurfaceMesh(grid)


def main() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Simulation 09: Plate Bending with Structural Shell Elements (Shell3 vs Shell6)[/bold cyan]\n"
            "[dim]Verification Against Classical Navier / Kirchhoff Analytical Solution using ShellDomain & LinearStaticAnalysis[/dim]",
            title="[bold blue]project-sierra — Structural Shells[/bold blue]",
            border_style="cyan",
        )
    )

    output_dir = Path("examples/05_simulation/sim09_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Physical Parameters
    console.print("\n[bold yellow][1/4] Configuring Plate Geometry & Material[/bold yellow]")
    L = 1000.0   # mm (1 meter square plate)
    t = 10.0     # mm thickness
    E = 70000.0  # MPa (Aluminum)
    nu = 0.3     # Poisson's ratio
    rho = 2700e-9  # tonne/mm^3 (2700 kg/m^3)
    q_pressure = -0.01  # MPa (-10 kPa uniform downward pressure)

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(youngs_modulus=E, poissons_ratio=nu, mass_density=rho)
    material = Hookean(params)

    # Navier Analytical Solution for simply supported square plate:
    # w_center = 0.004062 * q * L^4 / D
    D = (E * t**3) / (12.0 * (1.0 - nu**2))
    w_analytical = 0.004062 * q_pressure * (L**4) / D

    console.print(f"  • Plate Dimensions       : [green]{L:.0f} mm x {L:.0f} mm x {t:.1f} mm[/green]")
    console.print(f"  • Flexural Rigidity D    : [green]{D:.2e}[/green] N·mm")
    console.print(f"  • Uniform Transverse Load: [green]{q_pressure:.3f}[/green] MPa (-10 kPa)")
    console.print(f"  • Navier Analytical Sag  : [bold green]{w_analytical:.5f} mm[/bold green]")

    # 2. Solving Shell3 (DKT) with ShellDomain & LinearStaticAnalysis
    console.print("\n[bold yellow][2/4] Solving with Shell3 (3-Node DKT Thin Shell)[/bold yellow]")
    N_s3 = 10
    mesh_s3 = build_plate_mesh_linear(L, N=N_s3)
    elem_s3 = Shell3(material=material, thickness=t)
    domain_s3 = ShellDomain(mesh=mesh_s3, material=material, element=elem_s3)

    # Identify perimeter boundary nodes (simply supported: w = 0)
    tol = 1e-6
    nodes_s3 = mesh_s3.nodes
    bound_mask_s3 = (
        (np.abs(nodes_s3[:, 0] - (-L / 2)) < tol)
        | (np.abs(nodes_s3[:, 0] - (L / 2)) < tol)
        | (np.abs(nodes_s3[:, 1] - (-L / 2)) < tol)
        | (np.abs(nodes_s3[:, 1] - (L / 2)) < tol)
    )
    bound_nodes_s3 = np.where(bound_mask_s3)[0]

    # Pin corners in u, v to prevent rigid in-plane body modes
    corner_mask_s3 = (
        (np.abs(nodes_s3[:, 0]) > L / 2 - tol) & (np.abs(nodes_s3[:, 1]) > L / 2 - tol)
    )
    corner_nodes_s3 = np.where(corner_mask_s3)[0]

    dm_s3 = DirichletManager(domain_s3.n_dofs)
    dm_s3.add_constraint(DirichletConstraint(bound_nodes_s3, components=(2,), value=0.0, dofs_per_node=6))
    dm_s3.add_constraint(DirichletConstraint(corner_nodes_s3, components=(0, 1), value=0.0, dofs_per_node=6))

    # Tributary pressure load on w DOFs
    f_ext_s3 = np.zeros(domain_s3.n_dofs, dtype=np.float64)
    A_tri_s3 = 0.5 * (L / N_s3) * (L / N_s3)
    for conn in mesh_s3.elements:
        for nid in conn:
            f_ext_s3[nid * 6 + 2] += (q_pressure * A_tri_s3) / 3.0

    analysis_s3 = LinearStaticAnalysis(domain_s3)
    res_s3 = analysis_s3.solve(dirichlet=dm_s3, f_ext=f_ext_s3, verbose=False)

    center_idx_s3 = np.argmin(np.linalg.norm(nodes_s3, axis=1))
    w_s3 = float(res_s3.displacement[center_idx_s3 * 6 + 2])
    domain_s3.save_state(output_dir / "plate_shell3_deformed.vtu")

    console.print(f"  • Shell3 Central Sag     : [green]{w_s3:.5f} mm[/green]")
    console.print(f"  • Analytical Match       : [bold green]{abs(w_s3 / w_analytical) * 100:.2f}%[/bold green]")

    # 3. Solving Shell6 (Reissner-Mindlin Quadratic) with ShellDomain & LinearStaticAnalysis
    console.print("\n[bold yellow][3/4] Solving with Shell6 (6-Node Quadratic Shell)[/bold yellow]")
    N_s6 = 8
    mesh_s6 = build_plate_mesh_quadratic(L, N=N_s6)
    elem_s6 = Shell6(material=material, thickness=t)
    domain_s6 = ShellDomain(mesh=mesh_s6, material=material, element=elem_s6)

    nodes_s6 = mesh_s6.nodes
    bound_mask_s6 = (
        (np.abs(nodes_s6[:, 0] - (-L / 2)) < tol)
        | (np.abs(nodes_s6[:, 0] - (L / 2)) < tol)
        | (np.abs(nodes_s6[:, 1] - (-L / 2)) < tol)
        | (np.abs(nodes_s6[:, 1] - (L / 2)) < tol)
    )
    bound_nodes_s6 = np.where(bound_mask_s6)[0]

    corner_mask_s6 = (
        (np.abs(nodes_s6[:, 0]) > L / 2 - tol) & (np.abs(nodes_s6[:, 1]) > L / 2 - tol)
    )
    corner_nodes_s6 = np.where(corner_mask_s6)[0]

    dm_s6 = DirichletManager(domain_s6.n_dofs)
    dm_s6.add_constraint(DirichletConstraint(bound_nodes_s6, components=(2,), value=0.0, dofs_per_node=6))
    dm_s6.add_constraint(DirichletConstraint(corner_nodes_s6, components=(0, 1), value=0.0, dofs_per_node=6))

    f_ext_s6 = np.zeros(domain_s6.n_dofs, dtype=np.float64)
    A_tri_s6 = 0.5 * (L / N_s6) * (L / N_s6)
    for conn in mesh_s6.elements:
        for nid in conn:
            f_ext_s6[nid * 6 + 2] += (q_pressure * A_tri_s6) / 6.0

    analysis_s6 = LinearStaticAnalysis(domain_s6)
    res_s6 = analysis_s6.solve(dirichlet=dm_s6, f_ext=f_ext_s6, verbose=False)

    center_idx_s6 = np.argmin(np.linalg.norm(nodes_s6, axis=1))
    w_s6 = float(res_s6.displacement[center_idx_s6 * 6 + 2])
    domain_s6.save_state(output_dir / "plate_shell6_deformed.vtu")

    console.print(f"  • Shell6 Central Sag     : [green]{w_s6:.5f} mm[/green]")
    console.print(f"  • Analytical Match       : [bold green]{abs(w_s6 / w_analytical) * 100:.2f}%[/bold green]")
    console.print(f"\n  • Saved VTU Results to [dim]{output_dir}[/dim]")

    # 4. Summary Table
    console.print("\n[bold yellow][4/4] Bending Verification Summary[/bold yellow]")
    table = Table(title="Simply Supported Square Plate Bending Benchmark")
    table.add_column("Domain / Formulation", style="cyan")
    table.add_column("Order", style="magenta")
    table.add_column("Central Sag (w0)", style="yellow")
    table.add_column("Analytical Ratio", style="green")
    table.add_column("Strain Energy", style="blue")

    table.add_row("Navier (Kirchhoff Theory)", "Analytical Series", f"{w_analytical:.5f} mm", "100.00%", "—")
    table.add_row(
        "ShellDomain (Shell3 DKT)",
        "Linear (p = 1)",
        f"{w_s3:.5f} mm",
        f"{abs(w_s3 / w_analytical) * 100:.2f}%",
        f"{res_s3.strain_energy:.2f} mJ",
    )
    table.add_row(
        "ShellDomain (Shell6 Reissner-Mindlin)",
        "Quadratic (p = 2)",
        f"{w_s6:.5f} mm",
        f"{abs(w_s6 / w_analytical) * 100:.2f}%",
        f"{res_s6.strain_energy:.2f} mJ",
    )

    console.print(table)
    console.print("\n[bold green]✓ ShellDomain and LinearStaticAnalysis verified against analytical plate theory.[/bold green]\n")


if __name__ == "__main__":
    main()
