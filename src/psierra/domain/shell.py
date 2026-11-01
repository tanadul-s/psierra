from __future__ import annotations
from pathlib import Path
import numpy as np
import pyvista as pv
import scipy.sparse as spsp

from psierra.domain.base import Domain
from psierra.element.base import Element
from psierra.material.base import Material
from psierra.mesh.surface import SurfaceMesh
from psierra.utility.indexing import compute_element_dofs
from psierra.utility.assembly import (
    create_coo_sparsity_indices,
    assemble_sparse_matrix,
)


class ShellDomain(Domain):
    """
    Structural Shell Domain in psierra.

    Binds a 2D surface mesh (SurfaceMesh), constitutive material model (Material),
    and a shell element formulation (Shell3 or Shell6), managing global assembly
    of 6-DOF-per-node stiffness and mass matrices.

    Degrees of Freedom per Node (6 DOFs):
      [u_x, u_y, u_z, theta_x, theta_y, theta_z]
    Total DOFs: 6 * N_nodes.
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        material: Material,
        element: Element,
    ) -> None:
        self.mesh = mesh
        self.material = material
        self.element = element

        # Bind material to element
        self.element.material = self.material

        self._n_dofs = self.mesh.n_nodes * self.element.dofs_per_node

        # 1. Kinematic state vectors (6 DOFs per node)
        self.u = np.zeros(self._n_dofs, dtype=np.float64)  # Displacement & Rotation
        self.v = np.zeros(self._n_dofs, dtype=np.float64)  # Velocity
        self.a = np.zeros(self._n_dofs, dtype=np.float64)  # Acceleration

        # 2. Precompute element DOF map & COO sparsity indices
        self._element_dofs = compute_element_dofs(
            self.mesh.elements, dofs_per_node=self.element.dofs_per_node
        )
        self._sparsity_rows, self._sparsity_cols = create_coo_sparsity_indices(
            self._element_dofs
        )

        # 3. Preallocated local element buffers
        n_elems = self.mesh.n_elements
        n_elem_dofs = self.element.dofs_per_element
        self._k_local_buffer = np.zeros(
            (n_elems, n_elem_dofs, n_elem_dofs), dtype=np.float64
        )
        self._m_local_buffer = np.zeros(
            (n_elems, n_elem_dofs, n_elem_dofs), dtype=np.float64
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_dofs(self) -> int:
        """Total degrees of freedom in the shell domain (6 * N_nodes)."""
        return self._n_dofs

    @property
    def displacement(self) -> np.ndarray:
        """Nodal translational displacements (N_nodes, 3) as a zero-copy view."""
        return self.u.reshape(-1, 6)[:, :3]

    @property
    def rotation(self) -> np.ndarray:
        """Nodal rotational angles (N_nodes, 3) as a zero-copy view."""
        return self.u.reshape(-1, 6)[:, 3:]

    @property
    def current_nodes(self) -> np.ndarray:
        """Current spatial coordinates x = X + u."""
        return self.mesh.nodes + self.displacement

    # =========================================================================
    # State Management
    # =========================================================================

    def update_state(
        self,
        u: np.ndarray | None = None,
        v: np.ndarray | None = None,
        a: np.ndarray | None = None,
    ) -> None:
        """Updates kinematic state vectors."""
        if u is not None:
            self.u[:] = u
        if v is not None:
            self.v[:] = v
        if a is not None:
            self.a[:] = a

    # =========================================================================
    # Matrix Assemblies
    # =========================================================================

    def assemble_stiffness_matrix(self) -> spsp.csr_matrix:
        """
        Assembles and returns the global sparse shell stiffness matrix K.
        """
        n_elems = self.mesh.n_elements
        nodes = self.mesh.nodes
        elements = self.mesh.elements

        for e in range(n_elems):
            elem_nodes = elements[e]
            coords = nodes[elem_nodes]
            self._k_local_buffer[e] = self.element.compute_linear_stiffness_matrix(coords)

        return assemble_sparse_matrix(
            element_matrices=self._k_local_buffer,
            rows=self._sparsity_rows,
            cols=self._sparsity_cols,
            total_dofs=self._n_dofs,
        )

    def assemble_mass_matrix(self) -> spsp.csr_matrix:
        """
        Assembles and returns the global sparse shell consistent mass matrix M.
        """
        n_elems = self.mesh.n_elements
        nodes = self.mesh.nodes
        elements = self.mesh.elements

        for e in range(n_elems):
            elem_nodes = elements[e]
            coords = nodes[elem_nodes]
            self._m_local_buffer[e] = self.element.compute_mass_matrix(node_coords=coords)

        return assemble_sparse_matrix(
            element_matrices=self._m_local_buffer,
            rows=self._sparsity_rows,
            cols=self._sparsity_cols,
            total_dofs=self._n_dofs,
        )

    # =========================================================================
    # Visualization & Export
    # =========================================================================

    def to_pyvista(self, factor: float = 1.0) -> pv.UnstructuredGrid:
        """
        Creates a deformed PyVista UnstructuredGrid representing the shell.
        """
        deformed_poly = self.mesh.polydata.copy(deep=True)
        if factor != 0.0:
            deformed_poly.points = self.mesh.nodes + factor * self.displacement
        deformed_poly.point_data["Displacement"] = self.displacement
        deformed_poly.point_data["Displacement X"] = self.displacement[:, 0]
        deformed_poly.point_data["Displacement Y"] = self.displacement[:, 1]
        deformed_poly.point_data["Displacement Z"] = self.displacement[:, 2]
        deformed_poly.point_data["Displacement Magnitude"] = np.linalg.norm(self.displacement, axis=1)
        deformed_poly.point_data["Rotation"] = self.rotation
        deformed_poly.point_data["Rotation X"] = self.rotation[:, 0]
        deformed_poly.point_data["Rotation Y"] = self.rotation[:, 1]
        deformed_poly.point_data["Rotation Z"] = self.rotation[:, 2]
        deformed_poly.point_data["Rotation Magnitude"] = np.linalg.norm(self.rotation, axis=1)

        if isinstance(deformed_poly, pv.PolyData):
            return deformed_poly.cast_to_unstructured_grid()
        return deformed_poly

    def save_state(self, filepath: Path, factor: float = 1.0, **kwargs) -> None:
        """
        Exports the deformed shell state to disk as a VTK UnstructuredGrid (.vtu).
        """
        filepath = Path(filepath)
        if filepath.suffix != ".vtu":
            filepath = filepath.with_suffix(".vtu")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        grid = self.to_pyvista(factor=factor)
        grid.save(str(filepath))
