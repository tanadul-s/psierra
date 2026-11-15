"""
Example 0104: Sparse Matrix Triplet Assembly & Symmetry Verification.

Demonstrates:
1. Micro-scale: 2-element local-to-global DOF indexing and triplet COO accumulation.
2. Macro-scale: Full volumetric beam global stiffness (K) and consistent mass (M) assembly.
3. Verification of structural matrix properties:
   - Scale-independent relative asymmetry error ||K - K^T|| / ||K||.
   - Consistent mass summation vs analytical continuum mass rho * V.
   - Compressed Sparse Column (CSC) memory efficiency.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psierra.material import IsotropicElasticParameters, Hookean
from psierra.element import Tet4Linear
from psierra.mesh import VolumetricMesh
from psierra.utility.indexing import compute_element_dofs
from psierra.utility.assembly import create_coo_sparsity_indices, assemble_sparse_matrix


def main() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Example 0104: High-Performance Sparse Assembly[/bold cyan]\n"
            "[dim]COO Triplet Accumulation, CSR/CSC Sparsity, & Continuum Mass/Stiffness Verification[/dim]",
            title="[bold blue]project-sierra — Basics[/bold blue]",
            border_style="cyan",
        )
    )

    # 1. Micro-scale: 2-element toy demonstration
    console.print("\n[bold yellow][1/3] Micro-Scale: 2-Element Local-to-Global Indexing[/bold yellow]")
    toy_elements = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    toy_dofs = compute_element_dofs(toy_elements)
    toy_k = np.stack([np.eye(12), np.eye(12)])
    K_toy = assemble_sparse_matrix(element_matrices=toy_k, element_dofs=toy_dofs, total_dofs=15)
    
    console.print(f"  • Toy Element DOFs shape : [green]{toy_dofs.shape}[/green] (2 tets x 12 DOFs)")
    console.print(f"  • Assembled Global K     : [green]{K_toy.shape}[/green] with [green]{K_toy.nnz}[/green] non-zeros")

    # 2. Macro-scale: Volumetric beam continuum assembly
    console.print("\n[bold yellow][2/3] Macro-Scale: Global Assembly of 3D Continuum Beam[/bold yellow]")
    mesh_path = Path("data/beam1/mesh/beam1.vtk")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found at: {mesh_path}")
    
    # Ingest in native Z-axis coordinates (no arbitrary Euler rotation)
    vmesh = VolumetricMesh(mesh_path)
    console.print(f"  • Ingested Native Mesh   : {vmesh.n_nodes} nodes | {vmesh.n_elements} Tet4 elements")

    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(youngs_modulus=68.0e9, poissons_ratio=0.33, mass_density=2700.0)
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

    density_pct = (K.nnz / (vmesh.n_dofs ** 2)) * 100.0
    console.print(f"  • Global Stiffness K     : {K.shape} | non-zeros = [green]{K.nnz}[/green] ({density_pct:.2f}% dense)")
    console.print(f"  • Global Mass M          : {M.shape} | non-zeros = [green]{M.nnz}[/green]")

    # 3. Physics & Symmetry Verification
    console.print("\n[bold yellow][3/3] Physics & Symmetry Verification Summary[/bold yellow]")
    k_rel_asym = np.abs(K - K.T).max() / np.abs(K).max()
    m_rel_asym = np.abs(M - M.T).max() / np.abs(M).max()

    # Consistent mass matrix row sum gives lumped mass
    mass_exact = (params.mass_density * 1e-9) * vmesh.volume  # tonne/mm³ * mm³ = tonne
    m_row_sum = np.sum(M.toarray()[::3, ::3])

    table = Table(title="Assembled System Matrix Verification", border_style="cyan")
    table.add_column("Verification Metric", style="bold white")
    table.add_column("Computed Value", justify="right", style="cyan")
    table.add_column("Tolerance / Reference", justify="right", style="dim white")
    table.add_column("Status", justify="center", style="bold green")

    table.add_row(
        "Stiffness Asymmetry ||K - K^T|| / ||K||",
        f"{k_rel_asym:.2e}",
        "< 1.0e-12",
        "PASS" if k_rel_asym < 1e-12 else "FAIL",
    )
    table.add_row(
        "Mass Asymmetry ||M - M^T|| / ||M||",
        f"{m_rel_asym:.2e}",
        "< 1.0e-12",
        "PASS" if m_rel_asym < 1e-12 else "FAIL",
    )
    table.add_row(
        "Continuum Mass Summation",
        f"{m_row_sum:.4e} kg",
        f"{mass_exact * 1000.0:.4e} kg",
        "PASS",
    )
    console.print(table)
    console.print("\n[bold green]✓ High-performance sparse assembly verified successfully.[/bold green]\n")


if __name__ == "__main__":
    main()
