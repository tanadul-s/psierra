from pathlib import Path
import numpy as np
from rich import print
from scipy.spatial.transform import Rotation
import pyvista as pv

from psierra.boundary.load import GravityLoad
from psierra.boundary.selector import select_nodes_by_bounds
from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4
from psierra.mesh import VolumetricMesh
from psierra.domain import SolidDomain
from psierra.boundary import (
    DirichletConstraint,
    DirichletManager,
)
from psierra.solver import LinearStaticSolver


def main():
    print("[bold blue]==================================================[/bold blue]")
    print("[bold blue]     3D Cantilever Beam Static Bending Test       [/bold blue]")
    print("[bold blue]==================================================[/bold blue]")

    # 1. Load Mesh
    mesh_file = Path("data/beam1/mesh/beam1_fine.vtk")
    mesh = VolumetricMesh(mesh_file)
    mesh.transform(
        orientation=Rotation.from_euler("zyz", (0, 90, 90), degrees=True),
        about_point=(0, 0, 0),
    )
    print(mesh)
    

    # 2. Material (Structural Aluminum: E = 70 GPa, nu = 0.33)
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=21000, poissons_ratio=0.35, mass_density=1100e-6
    )
    material = Hookean(params)
    element = Tet4(material)

    # 3. Setup Physical Solid Domain
    domain = SolidDomain(mesh, material, element)

    # 4. Boundary Conditions
    # Fixed support (Clamp root at X = X_min in all 3 directions: ux=uy=uz=0)
    root_nodes = select_nodes_by_bounds(mesh, bounds=(-61, 61, -31, 1, -61, 61))
    clamp = DirichletConstraint(
        node_indices=root_nodes, components=(0, 1, 2), value=0.0
    )
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)
    print(f"Clamped {len(root_nodes)} nodes ({len(clamp.dofs)} DOFs) at root (X_min).")

    gravity_load = GravityLoad(acceleration=(0, 0, -9.81e+3))
    M = domain.assemble_mass_matrix()
    f_ext = gravity_load.assemble(mass_matrix=M)
    
    # 6. Solve Linear Statics
    solver = LinearStaticSolver(domain)
    result = solver.solve(dirichlet=dirichlet, f_ext=f_ext)

    # 7. Physical Verification Checks
    total_mass = params.mass_density * mesh.volume
    theoretical_weight_z = -total_mass * 9.81e+3
    assembled_weight_z = f_ext[2::3].sum()  # Sum of all Z components
    print(f"Total Body Mass       : {total_mass:.2f} kg")
    print(f"Theoretical Weight    : {theoretical_weight_z:.2f} N")
    print(f"Assembled F_gravity   : {assembled_weight_z:.2f} N (Error: {abs(assembled_weight_z - theoretical_weight_z):.2e} N)")


    # Reaction Force Equilibrium Check: Sum(R_z) + Sum(F_z) == 0
    total_reaction_z = result.reactions[clamp.dofs][2::3].sum()
    print(f"Total Support Reaction: {total_reaction_z:.2f} N")
    print(f"Equilibrium Error     : {abs(total_reaction_z + assembled_weight_z):.2e} N")

    # 8. Export to ParaView
    output_path = Path("example/05_simulation/sim02_output/output.vtu")
    domain.save_state(
        filepath=output_path,
        include_stresses=True,
        smooth_to_nodes=True,
    )

    print(f"\n[bold green]Saved deformed solution to:[/bold green] {output_path}")
    
    pl = pv.Plotter()
    pl.add_mesh(mesh.ugrid, color="light_gray", opacity=0.2, show_edges=True)
    pl.add_points(mesh.nodes[root_nodes])
    pl.add_mesh(domain.to_pyvista().warp_by_vector("Displacement"), scalars="Displacement", show_edges=True)
    pl.show_grid()
    # pl.show()


if __name__ == "__main__":
    main()
