from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pyvista as pv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Shell3Linear, Shell6Linear
from psierra.mesh import SurfaceMesh
from psierra.domain import ShellDomain
from psierra.boundary import DirichletConstraint, DirichletManager
from psierra.analysis import LinearStaticAnalysis

install(show_locals=False)
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0202: Plate Bending with Structural Shell Elements (Shell3 vs Shell6)"
    )
    parser.add_argument(
        "--element",
        type=str,
        choices=["shell3", "shell6"],
        default="shell3",
        help="Shell element formulation: 'shell3' (DKT linear) or 'shell6' (Reissner-Mindlin quadratic) (default: shell3)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute both Shell3 and Shell6 elements and display full comparison against Navier analytical solution",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/02_linear_static/0202_plate",
        help="Base output directory (default: results/02_linear_static/0202_plate)",
    )
    return parser.parse_args()


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

    poly = pv.PolyData(
        np.array(points, dtype=np.float64), np.array(faces, dtype=np.int64)
    )
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
                node_map[(ix0, iy0)],
                node_map[(ix2, iy0)],
                node_map[(ix2, iy2)],
                node_map[(ix1, iy0)],
                node_map[(ix2, iy1)],
                node_map[(ix1, iy1)],
            ]
            conn2 = [
                node_map[(ix0, iy0)],
                node_map[(ix2, iy2)],
                node_map[(ix0, iy2)],
                node_map[(ix1, iy1)],
                node_map[(ix1, iy2)],
                node_map[(ix0, iy1)],
            ]

            cells.extend([6, *conn1])
            cells.extend([6, *conn2])
            cell_types.extend(
                [pv.CellType.QUADRATIC_TRIANGLE, pv.CellType.QUADRATIC_TRIANGLE]
            )

    grid = pv.UnstructuredGrid(cells, cell_types, np.array(points, dtype=np.float64))
    return SurfaceMesh(grid)


def run_plate_case(
    elem_type: str,
    L: float,
    t: float,
    E: float,
    nu: float,
    rho: float,
    q_pressure: float,
    w_analytical: float,
    out_dir: Path,
) -> dict:
    tol = 1e-6
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=E, poissons_ratio=nu, mass_density=rho
    )
    material = Hookean(params)

    if elem_type == "shell3":
        N = 10
        mesh = build_plate_mesh_linear(L, N=N)
        element = Shell3Linear(material=material, thickness=t)
        order_desc = "Linear (p = 1)"
        form_desc = "ShellDomain (Shell3 DKT)"
        nodes_per_face = 3
    else:
        N = 8
        mesh = build_plate_mesh_quadratic(L, N=N)
        element = Shell6Linear(material=material, thickness=t)
        order_desc = "Quadratic (p = 2)"
        form_desc = "ShellDomain (Shell6 Reissner-Mindlin)"
        nodes_per_face = 6

    domain = ShellDomain(mesh=mesh, material=material, element=element)

    # Boundary conditions
    nodes = mesh.nodes
    bound_mask = (
        (np.abs(nodes[:, 0] - (-L / 2)) < tol)
        | (np.abs(nodes[:, 0] - (L / 2)) < tol)
        | (np.abs(nodes[:, 1] - (-L / 2)) < tol)
        | (np.abs(nodes[:, 1] - (L / 2)) < tol)
    )
    bound_nodes = np.where(bound_mask)[0]
    corner_mask = (np.abs(nodes[:, 0]) > L / 2 - tol) & (
        np.abs(nodes[:, 1]) > L / 2 - tol
    )
    corner_nodes = np.where(corner_mask)[0]

    dm = DirichletManager(domain.n_dofs)
    dm.add_constraint(
        DirichletConstraint(bound_nodes, components=(2,), value=0.0, dofs_per_node=6)
    )
    dm.add_constraint(
        DirichletConstraint(corner_nodes, components=(0, 1), value=0.0, dofs_per_node=6)
    )

    # Equivalent consistent pressure load
    f_ext = np.zeros(domain.n_dofs, dtype=np.float64)
    A_tri = 0.5 * (L / N) * (L / N)
    for conn in mesh.elements:
        for nid in conn:
            f_ext[nid * 6 + 2] += (q_pressure * A_tri) / float(nodes_per_face)

    # Solve
    res = LinearStaticAnalysis(domain).solve(dirichlet=dm, f_ext=f_ext, verbose=False)

    center_idx = np.argmin(np.linalg.norm(nodes, axis=1))
    w_center = float(res.displacement[center_idx * 6 + 2])

    vtu_file = out_dir / f"result_{elem_type}.vtu"
    domain.save_state(vtu_file)

    return {
        "elem_type": elem_type,
        "form_desc": form_desc,
        "order": order_desc,
        "nodes": mesh.n_nodes,
        "elements": mesh.n_elements,
        "dofs": domain.n_dofs,
        "w_center": w_center,
        "analytical_ratio": abs(w_center / w_analytical) * 100.0,
        "strain_energy": res.strain_energy,
        "vtu_file": vtu_file,
    }


def main() -> None:
    args = parse_args()
    console.print(
        Panel.fit(
            f"[bold cyan]Example 0202: Plate Bending with Structural Shell Elements ({'ALL' if args.all else args.element.upper()})[/bold cyan]\n"
            "[dim]Verification Against Classical Kirchhoff Analytical Solution using ShellDomain & LinearStaticAnalysis[/dim]",
            title="[bold blue]project-sierra — Linear Static[/bold blue]",
            border_style="cyan",
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plate geometry & material constants
    L = 1000.0  # mm (1 meter square plate)
    t = 10.0  # mm thickness
    E = 70.0e9  # Pa (Aluminum, 70 GPa = 70.0e9 g/(mm·s²))
    nu = 0.3  # Poisson ratio
    rho = 2700e-6  # g/mm³ (2700 kg/m³)
    q_pressure = -10000.0  # Pa (-10 kPa uniform downward pressure)

    # Navier Analytical Solution for simply supported square plate:
    # w_center = 0.004062 * q * L^4 / D
    D = (E * t**3) / (12.0 * (1.0 - nu**2))
    w_analytical = 0.004062 * q_pressure * (L**4) / D

    console.print("\n[bold yellow][1/3] Benchmark Configuration[/bold yellow]")
    console.print(
        f"  • Plate Dimensions       : [green]{L:.0f} mm x {L:.0f} mm x {t:.1f} mm[/green]"
    )
    console.print(
        f"  • Material Properties    : Aluminum ([green]E = {E*1e-9:.0f} GPa[/green], [green]ν = {nu:.2f}[/green], [green]ρ = {rho*1e6:.0f} kg/m³[/green])"
    )
    console.print(
        f"  • Flexural Rigidity D    : [green]{D:.2e}[/green] g·mm²/s² (N·mm)"
    )
    console.print(
        f"  • Transverse Pressure q  : [green]{q_pressure:.0f} Pa[/green] (uniform downward)"
    )
    console.print(
        f"  • Navier Analytical Sag  : [bold green]{w_analytical:.5f} mm[/bold green]"
    )

    elements_to_run = ["shell3", "shell6"] if args.all else [args.element]
    results = []

    console.print(
        f"\n[bold yellow][2/3] Solving Linear Shell Formulations[/bold yellow]"
    )
    for el in elements_to_run:
        console.print(f"  ▶ Executing formulation: [cyan]{el}[/cyan]...")
        res = run_plate_case(
            elem_type=el,
            L=L,
            t=t,
            E=E,
            nu=nu,
            rho=rho,
            q_pressure=q_pressure,
            w_analytical=w_analytical,
            out_dir=output_dir,
        )
        results.append(res)
        console.print(
            f"    - Nodes: [green]{res['nodes']}[/green] | DOFs: [green]{res['dofs']}[/green] | "
            f"Sag: [green]{res['w_center']:.5f} mm[/green] ([bold green]{res['analytical_ratio']:.2f}%[/bold green] analytical)"
        )

    # If single element, also save a standard result.vtu symlink or copy
    if not args.all:
        single_res = results[0]
        result_vtu = output_dir / "result.vtu"
        result_vtu.write_bytes(single_res["vtu_file"].read_bytes())

    # Results Table
    console.print(
        "\n[bold yellow][3/3] Verification Results & Comparison[/bold yellow]"
    )
    table = Table(title="Example 0202: Simply Supported Square Plate Bending Benchmark")
    table.add_column("Domain / Formulation", style="cyan")
    table.add_column("Order", style="magenta")
    table.add_column("Central Sag (w0)", style="yellow")
    table.add_column("Analytical Ratio", style="green")
    table.add_column("Strain Energy", style="blue")

    table.add_row(
        "Navier (Kirchhoff Theory)",
        "Analytical Series",
        f"{w_analytical:.5f} mm",
        "100.00%",
        "—",
    )
    for r in results:
        table.add_row(
            r["form_desc"],
            r["order"],
            f"{r['w_center']:.5f} mm",
            f"{r['analytical_ratio']:.2f}%",
            f"{r['strain_energy']:.2f} mJ",
        )

    console.print(table)
    console.print(
        f"\n[bold green]✓ Results written to:[/bold green] [cyan]{output_dir}[/cyan]\n"
    )


if __name__ == "__main__":
    main()
