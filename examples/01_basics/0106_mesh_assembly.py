from pathlib import Path
import numpy as np
import pyvista as pv
from rich import print
from rich.traceback import install
from scipy.spatial.transform import Rotation

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear
from psierra.mesh import VolumetricMesh
from psierra.utility.indexing import compute_element_dofs
from psierra.utility.assembly import create_coo_sparsity_indices, assemble_sparse_matrix


def main():
    vmesh = VolumetricMesh(Path("data/beam1/mesh/beam1_fine.vtk"))
    vmesh.transform(
        orientation=Rotation.from_euler("zyz", (0, 90, 90), degrees=True),
        about_point=vmesh.centroid,
    )

    # 4. Construct Global FEM Equations
    print(f"\n[bold green]4. Global Equation Assembly (M a + K u = F):[/bold green]")
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=68.0e9, poissons_ratio=0.33, mass_density=2700
    )
    material = Hookean(params)
    element = Tet4Linear(material)

    element_dofs = compute_element_dofs(vmesh.elements)
    rows, cols = create_coo_sparsity_indices(element_dofs)

    k_all = np.zeros((vmesh.n_elements, 12, 12), dtype=np.float64)
    m_all = np.zeros((vmesh.n_elements, 12, 12), dtype=np.float64)

    V = vmesh.volumes
    for e in range(vmesh.n_elements):
        X = vmesh.nodes[vmesh.elements[e]]
        k_all[e] = element.compute_linear_stiffness_matrix(X)
        m_all[e] = element.compute_mass_matrix(V[e])

    K = assemble_sparse_matrix(k_all, rows=rows, cols=cols, total_dofs=vmesh.n_dofs)
    M = assemble_sparse_matrix(m_all, rows=rows, cols=cols, total_dofs=vmesh.n_dofs)

    print(f"   Global K Matrix    : {K.shape}, non-zeros = {K.nnz}")
    print(f"   Global M Matrix    : {M.shape}, non-zeros = {M.nnz}")

    # 5. Physics & Symmetry Verification

    # Relative symmetry error (Scale-independent)
    k_rel_asym = np.abs(K - K.T).max() / np.abs(K).max()
    m_rel_asym = np.abs(M - M.T).max() / np.abs(M).max()
    mass_diag = np.sum(M.diagonal()) / 3.0
    mass_diag_compensated = (
        mass_diag / 0.4
    )  # mass lives in on-diagonal 40% and off-diagonal 60%
    mass_exact = params.mass_density * vmesh.volume

    print(f"\n[bold green]5. Verification Summary:[/bold green]")
    print(
        f"   K Relative Symmetry Error : {k_rel_asym:.2e} (Pass: {k_rel_asym < 1e-12})"
    )
    print(
        f"   M Relative Symmetry Error : {m_rel_asym:.2e} (Pass: {m_rel_asym < 1e-12})"
    )
    print(
        f"   Total Mass Check          : {mass_diag_compensated:.4e} kg vs {mass_exact:.4e} kg (Diff: {abs(mass_diag_compensated - mass_exact):.4e} kg)"
    )


if __name__ == "__main__":
    install()
    main()
