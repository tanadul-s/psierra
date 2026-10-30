from pathlib import Path
import numpy as np
from numpy.random import vonmises
import pyvista as pv
from rich import print
from rich.traceback import install
from scipy.spatial.transform import Rotation

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4
from psierra.domain import SolidDomain
from psierra.mesh import VolumetricMesh


def main():

    # 1. Create mesh
    vmesh = VolumetricMesh(Path("data/beam1/mesh/beam1_fine.vtk"))
    vmesh.transform(
        orientation=Rotation.from_euler("zyz", (0, 90, 90), degrees=True),
        about_point=vmesh.centroid,
    )

    # 2. Setup material and element
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=68.0e9, poissons_ratio=0.33, mass_density=2700
    )
    material = Hookean(params)
    element = Tet4(material)

    # 3. Setup domain
    domain = SolidDomain(vmesh, material, element)

    # 4. Assemble matrices in 1 line
    K = domain.assemble_stiffness_matrix()
    M = domain.assemble_mass_matrix()

    # 5. Assign synthetic displacement field and synthetic element Von Mises stress
    synthetic_disp = np.zeros((vmesh.n_nodes, 3))
    synthetic_disp[:, 2] = -0.004 * vmesh.nodes[:, 1] ** 2  # Parabolic bending in Z
    synthetic_vonmises = 1.0e6 * np.abs(vmesh.centroids[:, 0])

    # 6. Solver updates state in 1 line
    domain.update_state(u=synthetic_disp.ravel())
    print(domain)

    # 7. Create ugrid from domain
    ugrid = domain.to_pyvista(
        include_cell_data={"Von Mises Stress": synthetic_vonmises}
    )
    print(f"  Nodal fields      : {list(ugrid.point_data.keys())}")
    print(f"  Element fields    : {list(ugrid.cell_data.keys())}")

    # 8. Save directly to ParaView in 1 line
    domain.save_state(
        filepath=Path("example/04_domain/domain01_output/output.vtu"),
        cell_data={"Von Mises Stress": synthetic_vonmises},
    )


if __name__ == "__main__":
    main()
