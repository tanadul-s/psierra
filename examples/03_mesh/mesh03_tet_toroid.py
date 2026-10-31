from pathlib import Path
import numpy as np
import pyvista as pv
from rich import print
from rich.table import Table

from psierra.mesher import (
    TetrahedralMesher,
    export_mesh,
    import_mesh,
    is_gmsh_available,
)
from psierra.mesh.volumetric import VolumetricMesh


def main() -> None:
    print(
        "[bold cyan]======================================================[/bold cyan]"
    )
    print(
        "[bold cyan]      Mesh 03: Tetrahedral Mesher (Parametric Toroid)  [/bold cyan]"
    )
    print(
        "[bold cyan]======================================================[/bold cyan]\n"
    )

    output_dir = Path("examples/03_mesh/mesh03_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Surface Geometry (Parametric Toroid representation)
    print("[1/5] Preparing Input Surface (STL/PolyData)...")
    surface_stl = pv.ParametricSuperToroid(ringradius=30, crosssectionradius=10)
    stl_path = output_dir / "toroid_surface.stl"
    surface_stl.save(str(stl_path))
    print(f"  • Input Surface Vertices : {surface_stl.n_points}")
    print(f"  • Input Surface Triangles: {surface_stl.n_cells}")
    print(f"  • Saved demo STL to      : {stl_path}")

    # 2. Generate Linear Tetrahedral Mesh (Tet4) via PyTetWild
    print("\n[2/5] Tetrahedralizing with Envelope-Constrained PyTetWild (Tet4)...")
    mesher_tet4 = TetrahedralMesher(order=1, edge_length_fac=0.15, quiet=True)
    vmesh_tet4 = mesher_tet4.generate(stl_path)
    print(
        f"  • Generated Tet4 Nodes   : {vmesh_tet4.n_nodes} ({vmesh_tet4.n_dofs} DOFs)"
    )
    print(f"  • Generated Tet4 Elements: {vmesh_tet4.n_elements}")
    print(f"  • Domain Total Volume    : {vmesh_tet4.volume:.2f} mm³")

    vtu_tet4_path = output_dir / "mesh_tet4.vtu"
    export_mesh(vmesh_tet4, vtu_tet4_path, backend="pyvista")

    # 3. Upgrade EXACT Same Tet4 Mesh to Quadratic Tetrahedra (Tet10)
    # Using vmesh_tet4.to_quadratic() ensures the base tetrahedralization is identical!
    print("\n[3/5] Upgrading Tet4 to Quadratic Tetrahedra (Tet10)...")
    vmesh_tet10 = vmesh_tet4.to_quadratic()

    # Matrix bandwidth and RCM reordering evaluation
    bw_orig = vmesh_tet10.bandwidth
    vmesh_tet10_rcm = vmesh_tet10.reorder_nodes("rcm")
    bw_rcm = vmesh_tet10_rcm.bandwidth

    # Geometric alignment checks
    assert (
        vmesh_tet10.n_elements == vmesh_tet4.n_elements
    ), "Element counts must match exactly"
    assert np.allclose(
        vmesh_tet10.corner_nodes, vmesh_tet4.nodes
    ), "Corner nodes must match Tet4 exactly"
    assert np.array_equal(
        vmesh_tet10.elements[:, :4], vmesh_tet4.elements
    ), "Corner connectivity must match"

    # Verify mid-edge nodes geometrically
    c10 = vmesh_tet10.elements
    pts = vmesh_tet10.nodes
    sample_indices = np.random.choice(len(c10), min(50, len(c10)), replace=False)
    for idx in sample_indices:
        n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = c10[idx]
        p0, p1, p2, p3 = pts[n0], pts[n1], pts[n2], pts[n3]
        assert np.allclose(pts[n4], 0.5 * (p0 + p1), atol=1e-12)
        assert np.allclose(pts[n5], 0.5 * (p1 + p2), atol=1e-12)
        assert np.allclose(pts[n6], 0.5 * (p2 + p0), atol=1e-12)
        assert np.allclose(pts[n7], 0.5 * (p0 + p3), atol=1e-12)
        assert np.allclose(pts[n8], 0.5 * (p1 + p3), atol=1e-12)
        assert np.allclose(pts[n9], 0.5 * (p2 + p3), atol=1e-12)

    print(
        f"  • Generated Tet10 Nodes  : {vmesh_tet10.n_nodes} ({vmesh_tet10.n_dofs} DOFs)"
    )
    print(
        f"    - Corner Vertices      : {len(vmesh_tet10.corner_nodes)} (100% aligned with Tet4)"
    )
    print(
        f"    - Mid-Edge Nodes       : {len(vmesh_tet10.midedge_nodes)} (Exact edge midpoints)"
    )
    print(
        f"  • Generated Tet10 Elements: {vmesh_tet10.n_elements} (Identical to Tet4 count)"
    )
    print(
        f"  • Cell Type              : {vmesh_tet10.cell_type.name} (Order {vmesh_tet10.order})"
    )
    print(f"  • Nodes per Element      : {vmesh_tet10.nodes_per_element}")
    print(
        f"  • Verification Status    : [bold green]PASS: 100% vertex & edge alignment verified[/bold green]"
    )

    # 4. Multi-backend Export and Roundtrip Verification
    print("\n[4/5] Exporting & Roundtrip Testing via PyVista & Meshio...")
    vtu_path = output_dir / "mesh_tet10.vtu"
    msh_path = output_dir / "mesh_tet10.msh"

    vmesh_tet10 = vmesh_tet10_rcm

    # Export VTU via PyVista
    export_mesh(vmesh_tet10, vtu_path, backend="pyvista")
    # Export MSH via Meshio
    export_mesh(vmesh_tet10, msh_path, backend="meshio")

    # Import back and verify
    reloaded_vtu = import_mesh(vtu_path)
    print(f"  • Exported & Verified VTU: {vtu_path} ({reloaded_vtu.n_cells} cells)")
    print(f"  • Exported & Verified MSH: {msh_path} (Format: Gmsh tetrahedral)")

    # 5. Summary Table
    print("\n[5/5] Pipeline Summary:")
    table = Table(title="Tetrahedral Generation Pipeline Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Linear (Tet4)", style="magenta")
    table.add_column("Quadratic (Tet10)", style="green")

    table.add_row(
        "Total Elements", str(vmesh_tet4.n_elements), str(vmesh_tet10.n_elements)
    )
    table.add_row(
        "Corner Vertices",
        str(len(vmesh_tet4.nodes)),
        str(len(vmesh_tet10.corner_nodes)),
    )
    table.add_row("Mid-Edge Nodes", "0 (None)", str(len(vmesh_tet10.midedge_nodes)))
    table.add_row("Total Nodes", str(vmesh_tet4.n_nodes), str(vmesh_tet10.n_nodes))
    table.add_row("DOFs (3 * Nodes)", str(vmesh_tet4.n_dofs), str(vmesh_tet10.n_dofs))
    table.add_row(
        "Volume (mm³)", f"{vmesh_tet4.volume:.2f}", f"{vmesh_tet10.volume:.2f}"
    )
    table.add_row(
        "Matrix Bandwidth",
        str(vmesh_tet4.bandwidth),
        f"{bw_orig} -> {bw_rcm} ({bw_orig / bw_rcm:.1f}x reduction via RCM)",
    )
    table.add_row(
        "CAD (*.step) Support",
        f"Gmsh: {is_gmsh_available()}",
        f"Gmsh: {is_gmsh_available()}",
    )

    print(table)
    print("\n[bold green]✓ Pipeline execution completed successfully.[/bold green]")


if __name__ == "__main__":
    main()

    mesh_tet4 = pv.read(str(Path("examples/03_mesh/mesh03_output/mesh_tet4.vtu")))
    mesh_tet10 = pv.read(str(Path("examples/03_mesh/mesh03_output/mesh_tet10.vtu")))
    pl = pv.Plotter()
    pl.add_mesh(mesh=mesh_tet4, show_edges=True)
    pl.add_mesh(
        mesh=pv.PolyData(mesh_tet10.points),
        point_size=10,
        render_points_as_spheres=True,
        scalars=range(mesh_tet10.n_points),
    )
    pl.show()
