"""
Example 0103: Unstructured Tetrahedral Mesh Generation (Tet4 & Tet10).

Demonstrates:
1. Ingesting CAD/surface STL geometries (Cantilever Beam, Parametric Toroid, Segmented Brain).
2. Generating 1st-order linear tetrahedra (Tet4) via envelope-constrained meshing.
3. Generating 2nd-order quadratic tetrahedra (Tet10) with mid-edge nodes.
4. Evaluating volumetric Jacobian quality and edge lengths.
5. Exporting mesh assets for simulation.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pyvista as pv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psierra.mesher import TetrahedralMesher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0103: Tetrahedral Mesh Generation (Tet4 & Tet10)"
    )
    parser.add_argument(
        "--geom",
        type=str,
        choices=["beam", "toroid", "brain"],
        default="beam",
        help="Geometry to mesh: 'beam' (CAD), 'toroid' (parametric), or 'brain' (segmented STL) (default: beam)",
    )
    parser.add_argument(
        "--order",
        type=int,
        choices=[1, 2],
        default=1,
        help="Element order: 1 for Tet4, 2 for Tet10 (default: 1)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate both Tet4 and Tet10 meshes and compare quality metrics",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/01_basics/0103_mesh_generation",
        help="Target output directory (default: results/01_basics/0103_mesh_generation)",
    )
    return parser.parse_args()


def get_surface_geometry(geom: str, out_dir: Path) -> Path:
    if geom == "beam":
        p = Path("data/beam1/cad/beam1.stl")
        if not p.exists():
            raise FileNotFoundError(f"Beam CAD file not found: {p}")
        return p
    elif geom == "brain":
        p = Path("data/mrbraintumor1/scan/segmented/brain.stl")
        if not p.exists():
            raise FileNotFoundError(f"Brain segmented STL file not found: {p}")
        return p
    elif geom == "toroid":
        p = out_dir / "toroid_surface.stl"
        out_dir.mkdir(parents=True, exist_ok=True)
        surface_stl = pv.ParametricSuperToroid(ringradius=30, crosssectionradius=10)
        surface_stl.save(str(p))
        return p
    else:
        raise ValueError(f"Unknown geometry: {geom}")


def main() -> None:
    args = parse_args()
    console = Console()
    console.print(
        Panel.fit(
            f"[bold cyan]Example 0103: Tetrahedral Mesh Generation ({args.geom.upper()})[/bold cyan]\n"
            "[dim]Envelope-Constrained Tetrahedralization: Linear Tet4 & Quadratic Tet10[/dim]",
            title="[bold blue]project-sierra — Basics[/bold blue]",
            border_style="cyan",
        )
    )

    out_dir = Path(args.output_dir) / args.geom
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest surface
    console.print("\n[bold yellow][1/3] Preparing Input Surface STL[/bold yellow]")
    stl_path = get_surface_geometry(args.geom, out_dir)
    surf = pv.read(str(stl_path))
    b = surf.bounds
    console.print(f"  • Source STL File   : [cyan]{stl_path}[/cyan]")
    console.print(f"  • Surface Vertices  : [green]{surf.n_points}[/green]")
    console.print(f"  • Surface Triangles : [green]{surf.n_cells}[/green]")
    console.print(f"  • Bounding Box      : X: [{b[0]:.1f}, {b[1]:.1f}], Y: [{b[2]:.1f}, {b[3]:.1f}], Z: [{b[4]:.1f}, {b[5]:.1f}] mm")

    orders = [1, 2] if args.all else [args.order]
    results = {}

    console.print("\n[bold yellow][2/3] Executing Tetrahedral Mesher[/bold yellow]")
    for ord_val in orders:
        elem_name = "Tet4" if ord_val == 1 else "Tet10"
        console.print(f"  ▶ Generating [bold green]{elem_name}[/bold green] (order={ord_val})...")
        mesher = TetrahedralMesher(order=ord_val, edge_length_fac=0.15, quiet=True)
        vmesh = mesher.generate(stl_path)
        
        out_vtk = out_dir / f"mesh_{elem_name.lower()}.vtk"
        vmesh.save(out_vtk)
        qualities = vmesh.compute_qualities("scaled_jacobian")
        results[elem_name] = (vmesh, qualities, out_vtk)
        console.print(f"    • Nodes / DOFs   : [green]{vmesh.n_nodes}[/green] ({vmesh.n_dofs} DOFs)")
        console.print(f"    • Elements       : [green]{vmesh.n_elements}[/green] {elem_name}")
        console.print(f"    • Saved Mesh to  : [cyan]{out_vtk}[/cyan]")

    # 3. Quality comparison
    console.print("\n[bold yellow][3/3] Mesh Quality & Metric Verification[/bold yellow]")
    table = Table(title=f"Mesh Quality Metrics ({args.geom.upper()})", border_style="cyan")
    table.add_column("Element Type", style="bold white")
    table.add_column("Order", justify="center")
    table.add_column("Nodes", justify="right", style="cyan")
    table.add_column("Elements", justify="right", style="green")
    table.add_column("Volume (mm³)", justify="right")
    table.add_column("Min Quality (SJ)", justify="right", style="yellow")
    table.add_column("Mean Quality (SJ)", justify="right", style="bold green")

    for name, (vm, q, _) in results.items():
        table.add_row(
            name,
            "Linear (1)" if name == "Tet4" else "Quadratic (2)",
            f"{vm.n_nodes}",
            f"{vm.n_elements}",
            f"{vm.volume:.1f}",
            f"{q.min():.3f}",
            f"{q.mean():.3f}",
        )
    console.print(table)
    console.print(f"\n[bold green]✓ Mesh generation complete. Output saved in:[/bold green] [cyan]{out_dir}[/cyan]\n")


if __name__ == "__main__":
    main()
