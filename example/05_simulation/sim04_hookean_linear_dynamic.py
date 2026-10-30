from pathlib import Path
import numpy as np
from rich import print
from scipy.spatial.transform import Rotation

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
from psierra.damping import RayleighDamping
from psierra.analysis import LinearDynamicAnalysis


def main():
    print(
        "[bold blue]=== Dynamic Cantilever Beam Vibration (Newmark-beta) ===[/bold blue]"
    )

    # 1. Load Mesh (mm units)
    mesh = VolumetricMesh(Path("data/beam1/mesh/beam1_fine.vtk"))
    mesh.transform(
        orientation=Rotation.from_euler("zyz", (0, 90, 90), degrees=True),
        about_point=(0, 0, 0),
    )

    # 2. Material (Structural Aluminum: E = 70,000 MPa, nu = 0.33, rho = 2.7e-9 tonne/mm^3)
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=70000, poissons_ratio=0.35, mass_density=1100e-6
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

    # 5. Rayleigh Damping (alpha = 2.0 damps low frequencies, beta = 0.001 damps high frequencies)
    damping = RayleighDamping(alpha=2.0, beta=0.001)

    # 6. Run Dynamic Simulation (100 steps of dt = 0.005 s -> Total Time = 0.5 s)
    analysis = LinearDynamicAnalysis(domain, damping=damping)
    output_dir = Path("example/05_simulation/sim04_output")

    result = analysis.solve(
        dirichlet=dirichlet,
        f_ext=f_ext,
        dt=0.005,
        n_steps=100,
        include_stresses=True,
        smooth_to_nodes=True,
        save_directory=output_dir,
        save_interval=5,
    )

    print(
        f"\n[bold green]Simulation complete! Open {output_dir / 'output.pvd'} in ParaView and press Play (▶)![/bold green]"
    )


if __name__ == "__main__":
    main()
