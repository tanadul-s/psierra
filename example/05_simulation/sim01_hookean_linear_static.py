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
    select_nodes_by_plane,
    DirichletConstraint,
    DirichletManager,
    NodalForce,
)
from psierra.analysis import LinearStaticAnalysis


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
    root_nodes = select_nodes_by_plane(mesh, plane="xz", bound="min")
    clamp = DirichletConstraint(
        node_indices=root_nodes, components=(0, 1, 2), value=0.0
    )
    dirichlet = DirichletManager(domain.mesh.n_dofs)
    dirichlet.add_constraint(clamp)
    print(f"Clamped {len(root_nodes)} nodes ({len(clamp.dofs)} DOFs) at root (X_min).")

    # 5. External Load: Apply total downward force F_z = -1000 N distributed on tip nodes (X_max)
    tip_nodes = select_nodes_by_plane(mesh, plane="xz", bound="max")
    total_force_z = -150e+3  # -1 kN in Z
    force_per_node = total_force_z / len(tip_nodes)
    tip_load = NodalForce(
        node_indices=tip_nodes,
        force_vector=[0.0, 0.0, force_per_node],
    )
    f_ext = tip_load.compute_force_vector(total_dofs=domain.mesh.n_dofs)
    print(f"Applied total F_z = {total_force_z} N across {len(tip_nodes)} tip nodes.")

    # 6. Solve Linear Statics
    analysis = LinearStaticAnalysis(domain)
    result = analysis.solve(dirichlet=dirichlet, f_ext=f_ext)

    # 7. Physical Verification Checks
    print("\n[bold green]Results & Verification Checks:[/bold green]")
    max_disp = np.linalg.norm(domain.displacement, axis=1).max()
    tip_disp_z = domain.displacement[tip_nodes, 2].mean()
    print(f"Max Resultant Displacement : {max_disp * 1000:.4f} mm")
    print(f"Mean Tip Z-Deflection      : {tip_disp_z * 1000:.4f} mm")
    print(f"Total Elastic Strain Energy: {result.strain_energy:.6e} J")

    # Reaction Force Equilibrium Check: Sum(R_z) + Sum(F_z) == 0
    total_reaction_z = result.reactions[clamp.dofs][2::3].sum()
    equilibrium_error = abs(total_reaction_z + total_force_z)
    print(f"Total Reaction Force R_z   : {total_reaction_z:.4f} N")
    print(f"Total Applied Force F_z    : {total_force_z:.4f} N")
    print(
        f"Equilibrium Error          : {equilibrium_error:.2e} N (Pass: {equilibrium_error < 1e-6})"
    )

    # 8. Export to ParaView
    output_path = Path("example/05_simulation/sim01_output/output.vtu")
    domain.save_state(output_path)
    print(f"\n[bold green]Saved deformed solution to:[/bold green] {output_path}")


if __name__ == "__main__":
    main()
