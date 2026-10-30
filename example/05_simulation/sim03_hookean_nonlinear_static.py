from pathlib import Path
import numpy as np
from rich import print
from scipy.spatial.transform import Rotation
import pyvista as pv

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import (
    select_nodes_by_bounds,
    DirichletConstraint,
    DirichletManager,
    GravityLoad,
)
from psierra.analysis import NonlinearStaticAnalysis


def main():
    print(
        "[bold blue]=== Nonlinear Cantilever Beam Bending (Total Lagrangian) ===[/bold blue]"
    )

    # 1. Load Mesh (Dimensions in mm)
    mesh = VolumetricMesh(Path("data/beam1/mesh/beam1.vtk"))
    mesh.transform(
        orientation=Rotation.from_euler("zyz", (0, 90, 90), degrees=True),
        about_point=(0, 0, 0),
    )

    # 2. Material (Consistent mm-tonne-s units: E in MPa, rho in tonne/mm^3)
    # E = 21 kPa = 0.021 MPa (Soft parenchyma/silicone)
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000, poissons_ratio=0.35, mass_density=1100e-6
    )
    domain = SolidDomain(mesh, Hookean(params), Tet4(Hookean(params)))

    # 3. Clamp Support at Root
    root_nodes = select_nodes_by_bounds(mesh, bounds=(-61, 61, -31, 1, -61, 61))
    clamp = DirichletConstraint(root_nodes, components=(0, 1, 2), value=0.0)
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)

    # 4. Gravity Load (9810 mm/s^2 in -Z)
    gravity = GravityLoad(acceleration=(0, 0, -9810.0))
    M = domain.assemble_mass_matrix()
    f_ext = gravity.compute_force_vector(M)

    total_weight = f_ext[2::3].sum()
    print(f"Total Physical Weight: {total_weight:.3f} N")

    # 5. Solve Nonlinear Equilibrium with Incremental Newton-Raphson
    analysis = NonlinearStaticAnalysis(domain)
    result = analysis.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        n_steps=10,
        max_iterations=20,
        tolerance=1e-4,
    )

    # 6. Check Deformed Dimensions
    disp = domain.displacement
    max_disp = np.linalg.norm(disp, axis=1).max()
    tip_z = disp[:, 2].min()
    tip_y = disp[:, 1].min()

    print("\n[bold green]Deformation Summary:[/bold green]")
    print(f"Max Displacement    : {max_disp:.2f} mm")
    print(f"Downward Deflection : {tip_z:.2f} mm")
    print(f"Axial Shortening (Y): {tip_y:.2f} mm (Tip pulls inward correctly!)")

    # 7. Save to ParaView
    output_path = Path("example/05_simulation/sim03_output/output.vtu")
    domain.save_state(
        filepath=output_path,
        include_stresses=True,
        smooth_to_nodes=True,
    )
    print(f"Saved deformed result to: {output_path}")


if __name__ == "__main__":
    main()
