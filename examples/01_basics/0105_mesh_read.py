from pathlib import Path
import numpy as np
import pyvista as pv
from rich import print
from rich.traceback import install
from scipy.spatial.transform import Rotation

from psierra.mesh import VolumetricMesh


def create_sample_ugrid() -> pv.UnstructuredGrid:
    polydata = pv.Sphere(radius=100)
    ugrid = polydata.smooth().triangulate().delaunay_3d()
    return ugrid


def main():

    # Instantiate
    # vmesh = VolumetricMesh(create_sample_ugrid())
    vmesh = VolumetricMesh(Path("data/beam1/mesh/beam1_fine.vtk"))
    vmesh.transform(orientation=Rotation.from_euler("zyz", (0, 90, 90), degrees=True), about_point=vmesh.centroid)
    smesh = vmesh.surface

    pl = pv.Plotter()
    pl.add_mesh(vmesh.ugrid, opacity=0.8, color="teal")
    pl.add_mesh(vmesh.ugrid.extract_all_edges(), color="orange")
    pl.add_mesh(pv.PolyData(vmesh.nodes), color="black")
    pl.add_axes()
    pl.show_grid()
    pl.show()
    
    # 1. VolumetricMesh
    print(f"[bold green]1. Volumetric Mesh[/bold green]")
    print(f"  N Nodes            : {vmesh.n_nodes} ({vmesh.n_dofs} DOFs)")
    print(f"  N Elements (Tet4)  : {vmesh.n_elements}")
    print(f"  Total Volume       : {vmesh.volume:.2f} mm^3")
    print(f"  Domain Centroid    : {vmesh.centroid.round(4)}")
    print(
        f"  Scaled Jacobian Q  : [{vmesh.compute_qualities("scaled_jacobian").min():.2f}, {vmesh.compute_qualities("scaled_jacobian").max():.2f}]"
    )

    # 2. Surface Mesh
    print(f"[bold green]2. Boundary Surface Mesh[/bold green]")
    print(f"  N Surface Nodes    : {smesh.n_nodes}")
    print(f"  N Elements (Tri3)  : {smesh.n_elements}")
    print(f"  Total Surface Area : {smesh.area:.2f} mm^2")
    print(f"  Domain Centroid    : {smesh.centroid.round(4)}")
    print(f"  Normals shape      : {smesh.normals.shape}")
    print(
        f"  Triangle Quality Q : [{vmesh.compute_qualities("aspect_ratio").min():.2f}, {vmesh.compute_qualities("aspect_ratio").max():.2f}]"
    )

    # 3. Test Point Data & Cell Data
    print(f"\n[bold green]3. Point Data & Cell Data[/bold green]")

    # Assign synthetic displacement field
    synthetic_disp = np.zeros((vmesh.n_nodes, 3))
    synthetic_disp[:, 2] = -0.004 * vmesh.nodes[:, 1] ** 2  # Parabolic bending in Z
    vmesh.point_data["Displacement"] = synthetic_disp
    vmesh.point_data["Displacement Magnitude"] = np.linalg.norm(synthetic_disp, axis=1)

    # Assign synthetic element Von Mises stress
    synthetic_stress = 1.0e6 * np.abs(vmesh.centroids[:, 0])
    vmesh.cell_data["Von Mises Stress"] = synthetic_stress
    print(f"  Nodal fields       : {list(vmesh.point_data.keys())}")
    print(f"  Element fields     : {list(vmesh.cell_data.keys())}")

    pl = pv.Plotter()
    pl.add_mesh(vmesh.ugrid, color="light_gray", opacity=0.2, show_edges=True)
    # pl.add_mesh(vmesh.ugrid.warp_by_vector("Displacement"), scalars="Displacement Magnitude")
    pl.add_mesh(vmesh.ugrid.warp_by_vector("Displacement"), scalars="Von Mises Stress", show_edges=True)
    pl.add_axes()
    pl.show_grid()
    pl.show()


if __name__ == "__main__":
    install()
    main()
