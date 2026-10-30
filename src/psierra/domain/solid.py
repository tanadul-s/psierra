from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.sparse as spsp
import pyvista as pv

from psierra.mesh.volumetric import VolumetricMesh
from psierra.mesh.surface import SurfaceMesh
from psierra.material.base import Material
from psierra.element.base import Element
from psierra.utility.indexing import compute_element_dofs
from psierra.utility.assembly import (
    create_coo_sparsity_indices,
    assemble_sparse_matrix,
    assemble_vector,
)


class SolidDomain:
    """
    Continuum Solid Body in psierra.

    Binds reference geometry (VolumetricMesh), constitutive law (MaterialModel),
    element formulation (Element), and kinematic state (Displacement, Velocity, Acceleration).
    Precomputes sparsity indices and manages assembly of global K, M, and F_int.
    """

    def __init__(
        self,
        mesh: VolumetricMesh,
        material: Material,
        element: Element,
    ) -> None:
        self.mesh = mesh
        self.material = material
        self.element = element

        # Ensure element points to the correct material
        self.element.material = self.material

        # 1. State vectors stored as pure 1D NumPy arrays (Solver-ready)
        self.u = np.zeros(self.mesh.n_dofs, dtype=np.float64)  # Displacement
        self.v = np.zeros(self.mesh.n_dofs, dtype=np.float64)  # Velocity
        self.a = np.zeros(self.mesh.n_dofs, dtype=np.float64)  # Acceleration

        # 2. Precompute Global DOF indices and COO Sparsity Pattern
        self._element_dofs = compute_element_dofs(
            self.mesh.elements, dofs_per_node=self.element.dofs_per_node
        )
        self._sparsity_rows, self._sparsity_cols = create_coo_sparsity_indices(
            self._element_dofs
        )

        # 3. Preallocated local element buffers for zero-allocation assembly
        n_elems = self.mesh.n_elements
        n_elem_dofs = self.element.dofs_per_element
        self._k_local_buffer = np.zeros(
            (n_elems, n_elem_dofs, n_elem_dofs), dtype=np.float64
        )
        self._m_local_buffer = np.zeros(
            (n_elems, n_elem_dofs, n_elem_dofs), dtype=np.float64
        )
        self._f_local_buffer = np.zeros((n_elems, n_elem_dofs), dtype=np.float64)

    # =========================================================================
    # Kinematic Properties (Zero-Copy Views)
    # =========================================================================

    @property
    def displacement(self) -> np.ndarray:
        """Nodal displacement reshaped to (N_nodes, 3) as a zero-copy view."""
        return self.u.reshape(-1, 3)

    @property
    def velocity(self) -> np.ndarray:
        """Nodal velocity reshaped to (N_nodes, 3) as a zero-copy view."""
        return self.v.reshape(-1, 3)

    @property
    def acceleration(self) -> np.ndarray:
        """Nodal acceleration reshaped to (N_nodes, 3) as a zero-copy view."""
        return self.a.reshape(-1, 3)

    @property
    def current_nodes(self) -> np.ndarray:
        """Current spatial coordinates x = X + u."""
        return self.mesh.nodes + self.displacement

    @property
    def current_surface_nodes(self) -> np.ndarray:
        """Current spatial coordinates of boundary surface nodes (for contact)."""
        return self.current_nodes[self.mesh.surface.node_indices]

    @property
    def current_surface(self) -> SurfaceMesh:
        """
        Returns a SurfaceMesh deformed to current spatial coordinates
        with updated surface normals for contact detection.
        """
        deformed_polydata = self.mesh.surface.polydata.copy(deep=True)
        deformed_polydata.points = self.current_surface_nodes
        return SurfaceMesh(
            deformed_polydata, node_indices=self.mesh.surface.node_indices
        )

    # =========================================================================
    # State Updates
    # =========================================================================

    def update_state(
        self,
        u: np.ndarray | None = None,
        v: np.ndarray | None = None,
        a: np.ndarray | None = None,
    ) -> None:
        """Updates kinematic state vectors (u, v, a) in-place."""
        if u is not None:
            self.u[:] = u
        if v is not None:
            self.v[:] = v
        if a is not None:
            self.a[:] = a

    def reset_state(self) -> None:
        """Resets displacement, velocity, and acceleration to zero."""
        self.u.fill(0.0)
        self.v.fill(0.0)
        self.a.fill(0.0)

    # =========================================================================
    # Global Matrix & Vector Assembly
    # =========================================================================

    def assemble_stiffness_matrix(self) -> spsp.csr_matrix:
        """
        Assembles the linear global stiffness matrix K from reference geometry.

        Returns:
        --------
        K : spsp.csr_matrix of shape (n_dofs, n_dofs)
        """
        n_elems = self.mesh.n_elements
        nodes = self.mesh.nodes
        elements = self.mesh.elements

        for e in range(n_elems):
            coords = nodes[elements[e]]
            self._k_local_buffer[e] = self.element.compute_linear_stiffness_matrix(
                coords
            )

        return assemble_sparse_matrix(
            element_matrices=self._k_local_buffer,
            rows=self._sparsity_rows,
            cols=self._sparsity_cols,
            total_dofs=self.mesh.n_dofs,
        )

    def assemble_mass_matrix(self) -> spsp.csr_matrix:
        """
        Assembles the global consistent mass matrix M from reference geometry.

        Returns:
        --------
        M : spsp.csr_matrix of shape (n_dofs, n_dofs)
        """
        n_elems = self.mesh.n_elements
        elem_volumes = self.mesh.volumes

        for e in range(n_elems):
            self._m_local_buffer[e] = self.element.compute_mass_matrix(elem_volumes[e])

        return assemble_sparse_matrix(
            element_matrices=self._m_local_buffer,
            rows=self._sparsity_rows,
            cols=self._sparsity_cols,
            total_dofs=self.mesh.n_dofs,
        )

    def assemble_internal_forces(self) -> np.ndarray:
        """
        Assembles the nonlinear internal force vector f_int(u) using Total Lagrangian formulation.

        Returns:
        --------
        f_int : np.ndarray of shape (n_dofs,)
        """
        n_elems = self.mesh.n_elements
        nodes = self.mesh.nodes
        elements = self.mesh.elements
        disp = self.displacement

        for e in range(n_elems):
            elem_nodes = elements[e]
            coords = nodes[elem_nodes]
            u_elem = disp[elem_nodes]
            k_tan_elem, f_int_elem = (
                self.element.compute_nonlinear_force_and_tangent_stiffness(
                    coords, u_elem
                )
            )
            self._f_local_buffer[e] = f_int_elem

        return assemble_vector(
            element_vectors=self._f_local_buffer,
            element_dofs=self._element_dofs,
            total_dofs=self.mesh.n_dofs,
        )

    def assemble_tangent_stiffness_matrix(self) -> spsp.csr_matrix:
        """
        Assembles the Total Lagrangian tangent stiffness matrix K_tan(u) = K_mat + K_geo.

        Returns:
        --------
        K_tan : spsp.csr_matrix of shape (n_dofs, n_dofs)
        """
        n_elems = self.mesh.n_elements
        nodes = self.mesh.nodes
        elements = self.mesh.elements
        disp = self.displacement

        for e in range(n_elems):
            elem_nodes = elements[e]
            coords = nodes[elem_nodes]
            u_elem = disp[elem_nodes]
            k_tan_elem, _ = (
                self.element.compute_nonlinear_force_and_tangent_stiffness(
                    coords, u_elem
                )
            )
            self._k_local_buffer[e] = k_tan_elem

        return assemble_sparse_matrix(
            element_matrices=self._k_local_buffer,
            rows=self._sparsity_rows,
            cols=self._sparsity_cols,
            total_dofs=self.mesh.n_dofs,
        )

    # =========================================================================
    # Energy & Stresses
    # =========================================================================

    def compute_kinetic_energy(self) -> float:
        """Computes kinetic energy: E_k = 0.5 * v^T M v."""
        M = self.assemble_mass_matrix()
        return 0.5 * float(self.v @ (M @ self.v))

    def compute_strain_energy(self) -> float:
        """Computes linear strain energy: E_s = 0.5 * u^T K u."""
        K = self.assemble_stiffness_matrix()
        return 0.5 * float(self.u @ (K @ self.u))

    # =========================================================================
    # Stress & Strain Post-Processing
    # =========================================================================

    def compute_element_stresses(self) -> dict[str, np.ndarray]:
        """
        Computes element-wise Cauchy stresses, Von Mises stress, hydrostatic pressure,
        principal stresses, and Green-Lagrange strains for the current deformed state.

        Returns:
        --------
        stresses : dict containing:
            - 'Von Mises Stress': (n_elements,) array in MPa/Pa
            - 'Hydrostatic Pressure': (n_elements,) array
            - 'Max Principal Stress': (n_elements,) array
            - 'Min Principal Stress': (n_elements,) array
            - 'Cauchy Stress': (n_elements, 6) array in Voigt order (xx, yy, zz, xy, yz, zx)
            - 'Green-Lagrange Strain': (n_elements, 6) array in Voigt order
        """
        n_elems = self.mesh.n_elements
        nodes = self.mesh.nodes
        elements = self.mesh.elements
        disp = self.displacement

        von_mises = np.zeros(n_elems, dtype=np.float64)
        pressure = np.zeros(n_elems, dtype=np.float64)
        max_principal = np.zeros(n_elems, dtype=np.float64)
        min_principal = np.zeros(n_elems, dtype=np.float64)
        cauchy_voigt = np.zeros((n_elems, 6), dtype=np.float64)
        gl_strain_voigt = np.zeros((n_elems, 6), dtype=np.float64)

        for e in range(n_elems):
            elem_nodes = elements[e]
            X_elem = nodes[elem_nodes]
            u_elem = disp[elem_nodes]
            x_elem = X_elem + u_elem

            # 1. Deformation gradient F
            _, H = self.element.compute_volume_and_gradients(X_elem)
            F = x_elem.T @ H
            J = float(np.linalg.det(F))
            if J <= 0.0:
                continue

            # 2. Green-Lagrange Strain E = 0.5 * (F^T F - I)
            C_tensor = F.T @ F
            E_tensor = 0.5 * (C_tensor - np.identity(3, dtype=np.float64))
            gl_strain_voigt[e] = [
                E_tensor[0, 0],
                E_tensor[1, 1],
                E_tensor[2, 2],
                2.0 * E_tensor[0, 1],
                2.0 * E_tensor[1, 2],
                2.0 * E_tensor[2, 0],
            ]

            # 3. 2nd Piola-Kirchhoff Stress S and Cauchy Stress sigma = (1/J) * F @ S @ F.T
            _, S = self.material.compute_stress_and_tangent(F)
            sigma = (1.0 / J) * (F @ S @ F.T)
            cauchy_voigt[e] = [
                sigma[0, 0],
                sigma[1, 1],
                sigma[2, 2],
                sigma[0, 1],
                sigma[1, 2],
                sigma[2, 0],
            ]

            # 4. Hydrostatic Pressure p = -1/3 * tr(sigma)
            p = -(1.0 / 3.0) * float(np.trace(sigma))
            pressure[e] = p

            # 5. Von Mises Stress = sqrt(1.5 * dev(sigma) : dev(sigma))
            dev_s = sigma + p * np.identity(3, dtype=np.float64)
            von_mises[e] = np.sqrt(1.5 * np.tensordot(dev_s, dev_s))

            # 6. Principal Stresses (Eigenvalues of Cauchy tensor)
            eigvals = np.linalg.eigvalsh(sigma)
            min_principal[e] = eigvals[0]
            max_principal[e] = eigvals[2]

        return {
            "Von Mises Stress": von_mises,
            "Hydrostatic Pressure": pressure,
            "Max Principal Stress": max_principal,
            "Min Principal Stress": min_principal,
            "Cauchy Stress": cauchy_voigt,
            "Green-Lagrange Strain": gl_strain_voigt,
        }

    # =========================================================================
    # Visualization & ParaView Export (Updated with include_stresses)
    # =========================================================================

    def to_pyvista(
        self,
        factor: float = 1.0,
        include_stresses: bool = False,
        smooth_to_nodes: bool = False,
        include_cell_data: dict[str, np.ndarray] | None = None,
    ) -> pv.UnstructuredGrid:
        """
        Creates a deformed PyVista UnstructuredGrid for rendering or ParaView export.

        Parameters:
        -----------
        factor : float, default=1.0
            Displacement magnification factor.
        include_stresses : bool, default=False
            If True, automatically evaluates and attaches VonMises, Pressure, and Principal stresses.
        smooth_to_nodes : bool, default=False
            If True, interpolates cell stresses to point data for smooth gradient contours in ParaView.
        include_cell_data : dict[str, np.ndarray], optional
            Extra custom element-wise arrays.
        """
        grid = self.mesh.ugrid.copy(deep=True)

        grid.point_data["Displacement"] = self.displacement
        
        # Attach computed stresses
        if include_stresses:
            stress_dict = self.compute_element_stresses()
            for name, array in stress_dict.items():
                grid.cell_data[name] = array

        # Attach custom cell data
        if include_cell_data:
            for name, array in include_cell_data.items():
                grid.cell_data[name] = array

        # Smooth cell data to nodal point data for beautiful rainbow contours
        if smooth_to_nodes:
            grid = grid.cell_data_to_point_data()

        return grid

    def save_state(
        self,
        filepath: Path,
        factor: float = 1.0,
        include_stresses: bool = False,
        smooth_to_nodes: bool = False,
        cell_data: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Exports the current deformed state with optional stresses to a .vtu file."""
        if not isinstance(filepath, Path):
            filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        grid = self.to_pyvista(
            factor=factor,
            include_stresses=include_stresses,
            smooth_to_nodes=smooth_to_nodes,
            include_cell_data=cell_data,
        )
        grid.save(str(filepath))

    def __repr__(self) -> str:
        max_disp = (
            np.linalg.norm(self.displacement, axis=1).max() if self.u.any() else 0.0
        )
        return (
            f"SolidDomain"
            f"\n  Mesh             : {self.mesh.n_nodes} nodes, {self.mesh.n_elements} elements"
            f"\n  Material         : {self.material.__class__.__name__}"
            f"\n  Element          : {self.element.__class__.__name__}"
            f"\n  Max Displacement : {max_disp:.4e} unit"
        )
