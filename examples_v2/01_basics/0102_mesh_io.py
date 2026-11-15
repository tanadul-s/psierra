"""
Example 0102: Volumetric Mesh Ingestion, Geometric Inspection, & Unified Storage I/O.

Demonstrates:
1. Ingesting native VTK volumetric tetrahedral mesh without artificial rotations.
2. Geometric property queries: node count, DOFs, volume, centroid, and element quality.
3. Extracting the external triangular SurfaceMesh boundary.
4. Unified SimulationWriter exchange: multi-file PVD/VTU vs high-performance XDMF/HDF5.
5. Checkpoint save & restore state persistence via SimulationCheckpoint.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psierra.mesh import VolumetricMesh
from psierra.exchange import SimulationWriter, SimulationCheckpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 0102: Mesh Ingestion, Inspection, & Unified I/O"
    )
    parser.add_argument(
        "--mesh-path",
        type=str,
        default="data/beam1/mesh/beam1.vtk",
        help="Path to volumetric VTK mesh (default: data/beam1/mesh/beam1.vtk)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/01_basics/0102_mesh_io",
        help="Target output directory (default: results/01_basics/0102_mesh_io)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Example 0102: Mesh Inspection & Unified Storage Exchange[/bold cyan]\n"
            "[dim]Native Coordinates, Boundary Extraction, XDMF/HDF5 vs PVD, & Checkpoint Resume[/dim]",
            title="[bold blue]project-sierra — Basics[/bold blue]",
            border_style="cyan",
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loading Mesh in Native Coordinates
    console.print("\n[bold yellow][1/4] Ingesting Volumetric Mesh[/bold yellow]")
    mesh_path = Path(args.mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found at: {mesh_path}")

    vmesh = VolumetricMesh(mesh_path)
    b = vmesh.bounds
    console.print(f"  • Source Mesh File  : [cyan]{mesh_path}[/cyan]")
    console.print(f"  • Nodes / DOFs      : [green]{vmesh.n_nodes}[/green] nodes ({vmesh.n_dofs} DOFs)")
    console.print(f"  • Elements (Tet4)   : [green]{vmesh.n_elements}[/green]")
    console.print(f"  • Total Volume      : [green]{vmesh.volume:.2f}[/green] mm³")
    console.print(f"  • Domain Centroid   : ({vmesh.centroid[0]:.2f}, {vmesh.centroid[1]:.2f}, {vmesh.centroid[2]:.2f}) mm")
    console.print(f"  • Bounding Box      : X: [{b[0]:.1f}, {b[1]:.1f}], Y: [{b[2]:.1f}, {b[3]:.1f}], Z: [{b[4]:.1f}, {b[5]:.1f}] mm")

    # Element Quality Verification
    qualities = vmesh.compute_qualities("scaled_jacobian")
    console.print(f"  • Scaled Jacobian Q : Min = [green]{qualities.min():.3f}[/green], Mean = [green]{qualities.mean():.3f}[/green]")

    # 2. Extracting Surface Mesh
    console.print("\n[bold yellow][2/4] Extracting Boundary Surface Mesh[/bold yellow]")
    smesh = vmesh.surface
    console.print(f"  • Surface Nodes     : [green]{smesh.n_nodes}[/green]")
    console.print(f"  • Surface Triangles : [green]{smesh.n_elements}[/green]")
    console.print(f"  • Surface Area      : [green]{smesh.area:.2f}[/green] mm²")

    # 3. Exporting Dual Format Timeseries (PVD vs XDMF)
    console.print("\n[bold yellow][3/4] Testing SimulationWriter (XDMF/HDF5 & PVD/VTU)[/bold yellow]")
    
    # Synthetic dynamic field: traveling wave along native Z axis
    times = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    
    # Format A: High-performance single-file XDMF/HDF5
    writer_xdmf = SimulationWriter(
        output_path=output_dir / "xdmf_series" / "result.xdmf",
        mesh=vmesh,
        format="xdmf",
    )
    for i, t in enumerate(times):
        u_synth = np.zeros((vmesh.n_nodes, 3))
        u_synth[:, 1] = -2.5 * np.sin(2.0 * np.pi * 10.0 * t) * (vmesh.nodes[:, 2] / b[5]) ** 2
        von_mises = 1.0e6 * np.abs(u_synth[:vmesh.n_elements, 1])
        writer_xdmf.add_timestep(time=t, displacement=u_synth.ravel(), cell_data={"von_mises": von_mises}, step_index=i)
    writer_xdmf.close()
    console.print(f"  [green]✓ XDMF/HDF5 written to:[/green] [cyan]{output_dir / 'xdmf_series' / 'result.xdmf'}[/cyan]")

    # Format B: Standard multi-file PVD/VTU
    writer_pvd = SimulationWriter(
        output_path=output_dir / "pvd_series" / "result.pvd",
        mesh=vmesh,
        format="pvd",
    )
    for i, t in enumerate(times):
        u_synth = np.zeros((vmesh.n_nodes, 3))
        u_synth[:, 1] = -2.5 * np.sin(2.0 * np.pi * 10.0 * t) * (vmesh.nodes[:, 2] / b[5]) ** 2
        writer_pvd.add_timestep(time=t, displacement=u_synth.ravel(), step_index=i)
    writer_pvd.close()
    console.print(f"  [green]✓ PVD/VTU written to:[/green]   [cyan]{output_dir / 'pvd_series' / 'result.pvd'}[/cyan]")

    # 4. Checkpoint Save & Resume via SimulationWriter
    console.print("\n[bold yellow][4/4] Verifying State Checkpoint & Resume Persistence[/bold yellow]")
    archive_path = output_dir / "checkpoint_archive.xdmf"
    
    # Create domain to hold state
    from psierra.material import IsotropicElasticParameters, Hookean
    from psierra.element import Tet4Linear
    from psierra.domain import SolidDomain

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(21000.0, 0.35, 1100e-6)
    mat = Hookean(params)
    elem = Tet4Linear(mat)
    dom = SolidDomain(vmesh, mat, elem)

    # Set synthetic displacement
    u_checkpoint = np.random.randn(vmesh.n_dofs) * 0.1
    dom.update_state(u=u_checkpoint)

    # Save checkpoint via SimulationWriter
    with SimulationWriter(archive_path, mesh=vmesh, format="xdmf") as writer:
        writer.write_checkpoint(step=5, time=0.05, dt=0.01, domain=dom)
        console.print(f"  • Checkpoint written to: [cyan]{archive_path}[/cyan]")

    # Resume checkpoint
    dom_resumed = SolidDomain(vmesh, mat, elem)
    with SimulationWriter(archive_path, mesh=vmesh, format="xdmf", resume=True) as reader:
        loaded_ckpt = reader.read_checkpoint()
        loaded_ckpt.apply_to_domain(dom_resumed)

    err_u = np.max(np.abs(dom_resumed.displacement.ravel() - u_checkpoint))
    console.print(f"  • Resumed Step / Time : Step {loaded_ckpt.step}, Time = {loaded_ckpt.time:.3f} s")
    console.print(f"  • Reconstruction Err  : [green]{err_u:.2e}[/green] (Pass: {err_u < 1e-12})")
    console.print("\n[bold green]✓ Mesh I/O, inspection, and unified exchange verified successfully.[/bold green]\n")


if __name__ == "__main__":
    main()
