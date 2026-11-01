from __future__ import annotations
from pathlib import Path
import numpy as np
import pyvista as pv
import scipy.sparse as spsp

from psierra.domain.base import Domain
from psierra.domain.solid import SolidDomain
from psierra.domain.shell import ShellDomain
from psierra.mesh.volumetric import VolumetricMesh
from psierra.exchange.pvd import PvdCollectionWriter
from psierra.utility.assembly import (
    create_coo_sparsity_indices,
    assemble_sparse_matrix,
)


class CoupledSolidShellDomain(Domain):
    """
    Coupled Solid-Shell Continuum Domain in psierra.

    Monolithically couples a 3D solid continuum body (SolidDomain) with a 2D
    structural shell layer (ShellDomain) adhering to its outer boundary skin.

    Kinematic Coupling:
      - 3D Solid Nodes: 3 translational DOFs [u_x, u_y, u_z] per node (0 ... 3*N_solid - 1).
      - Boundary Shell Nodes: Share the 3 translational DOFs of the solid core, and add
        3 rotational DOFs [theta_x, theta_y, theta_z] per surface node
        (3*N_solid ... 3*N_solid + 3*N_surf - 1).

    Total Coupled DOFs = 3 * N_solid_nodes + 3 * N_shell_nodes.

    This formulation guarantees exact C^0 displacement continuity across the
    solid-shell interface without penalty multipliers, master-slave constraints,
    or interfacial slip.
    """

    def __init__(
        self,
        solid_domain: SolidDomain,
        shell_domain: ShellDomain,
    ) -> None:
        self.solid_domain = solid_domain
        self.shell_domain = shell_domain

        if self.shell_domain.mesh.node_indices is None:
            raise ValueError(
                "ShellDomain surface mesh must have 'node_indices' mapping surface nodes "
                "to parent VolumetricMesh nodes (e.g. from vmesh.surface)."
            )

        self._n_solid_nodes = self.solid_domain.mesh.n_nodes
        self._n_surf_nodes = self.shell_domain.mesh.n_nodes
        self._n_solid_dofs = 3 * self._n_solid_nodes
        self._n_shell_rot_dofs = 3 * self._n_surf_nodes
        self._n_dofs = self._n_solid_dofs + self._n_shell_rot_dofs

        # 1. Unified Kinematic State Vectors
        self.u = np.zeros(self._n_dofs, dtype=np.float64)
        self.v = np.zeros(self._n_dofs, dtype=np.float64)
        self.a = np.zeros(self._n_dofs, dtype=np.float64)

        # 2. Precompute Coupled DOF Map for Shell Elements
        # Each shell node has 6 DOFs in the unified system:
        # [3*gk, 3*gk+1, 3*gk+2] (translational, shared with solid)
        # [3*N_solid + 3*sk, 3*N_solid + 3*sk+1, 3*N_solid + 3*sk+2] (rotational)
        n_shell_elems = self.shell_domain.mesh.n_elements
        nodes_per_shell = self.shell_domain.element.nodes_per_element
        dofs_per_shell_elem = self.shell_domain.element.dofs_per_element

        self._shell_coupled_dofs = np.zeros(
            (n_shell_elems, dofs_per_shell_elem), dtype=np.int64
        )
        node_indices = self.shell_domain.mesh.node_indices

        for e in range(n_shell_elems):
            s_conn = self.shell_domain.mesh.elements[e]
            g_conn = node_indices[s_conn]
            for k in range(nodes_per_shell):
                sk = s_conn[k]
                gk = g_conn[k]
                # Translational DOFs
                self._shell_coupled_dofs[e, k * 6 : k * 6 + 3] = [
                    3 * gk,
                    3 * gk + 1,
                    3 * gk + 2,
                ]
                # Rotational DOFs
                self._shell_coupled_dofs[e, k * 6 + 3 : k * 6 + 6] = [
                    self._n_solid_dofs + 3 * sk,
                    self._n_solid_dofs + 3 * sk + 1,
                    self._n_solid_dofs + 3 * sk + 2,
                ]

        self._shell_sparsity_rows, self._shell_sparsity_cols = (
            create_coo_sparsity_indices(self._shell_coupled_dofs)
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def mesh(self) -> VolumetricMesh:
        """Access underlying VolumetricMesh of the solid core."""
        return self.solid_domain.mesh

    @property
    def n_dofs(self) -> int:
        """Total degrees of freedom in the coupled solid-shell system."""
        return self._n_dofs

    @property
    def displacement(self) -> np.ndarray:
        """Nodal translational displacements (N_solid_nodes, 3)."""
        return self.solid_domain.displacement

    @property
    def current_nodes(self) -> np.ndarray:
        """Current spatial coordinates of solid continuum nodes."""
        return self.solid_domain.current_nodes

    @property
    def current_surface_nodes(self) -> np.ndarray:
        """Current spatial coordinates of boundary surface nodes."""
        return self.solid_domain.current_surface_nodes

    # =========================================================================
    # State Management
    # =========================================================================

    def update_state(
        self,
        u: np.ndarray | None = None,
        v: np.ndarray | None = None,
        a: np.ndarray | None = None,
    ) -> None:
        """Updates kinematic state across both solid and shell domains."""
        if u is not None:
            self.u[:] = u
            u_solid = self.u[: self._n_solid_dofs]
            self.solid_domain.update_state(u=u_solid)

            # Map to shell domain state
            u_shell = np.zeros(self.shell_domain.n_dofs, dtype=np.float64)
            node_indices = self.shell_domain.mesh.node_indices
            for sk, gk in enumerate(node_indices):
                u_shell[sk * 6 : sk * 6 + 3] = u_solid[gk * 3 : gk * 3 + 3]
                u_shell[sk * 6 + 3 : sk * 6 + 6] = self.u[
                    self._n_solid_dofs + 3 * sk : self._n_solid_dofs + 3 * sk + 3
                ]
            self.shell_domain.update_state(u=u_shell)

        if v is not None:
            self.v[:] = v
            self.solid_domain.update_state(v=self.v[: self._n_solid_dofs])

        if a is not None:
            self.a[:] = a
            self.solid_domain.update_state(a=self.a[: self._n_solid_dofs])

    # =========================================================================
    # Matrix Assemblies
    # =========================================================================

    def assemble_stiffness_matrix(self) -> spsp.csr_matrix:
        """
        Assembles and returns the unified monolithic sparse stiffness matrix K.
        Combines 3D solid continuum elements and 2D surface shell elements.
        """
        # 1. Solid stiffness
        n_solid_elems = self.solid_domain.mesh.n_elements
        solid_nodes = self.solid_domain.mesh.nodes
        solid_elems = self.solid_domain.mesh.elements

        for e in range(n_solid_elems):
            coords = solid_nodes[solid_elems[e]]
            self.solid_domain._k_local_buffer[e] = (
                self.solid_domain.element.compute_linear_stiffness_matrix(coords)
            )

        K_solid = assemble_sparse_matrix(
            self.solid_domain._k_local_buffer,
            rows=self.solid_domain._sparsity_rows,
            cols=self.solid_domain._sparsity_cols,
            total_dofs=self._n_dofs,
        )

        # 2. Shell stiffness
        n_shell_elems = self.shell_domain.mesh.n_elements
        node_indices = self.shell_domain.mesh.node_indices
        shell_elems = self.shell_domain.mesh.elements

        for e in range(n_shell_elems):
            coords = solid_nodes[node_indices[shell_elems[e]]]
            self.shell_domain._k_local_buffer[e] = (
                self.shell_domain.element.compute_linear_stiffness_matrix(coords)
            )

        K_shell = assemble_sparse_matrix(
            self.shell_domain._k_local_buffer,
            rows=self._shell_sparsity_rows,
            cols=self._shell_sparsity_cols,
            total_dofs=self._n_dofs,
        )

        return K_solid + K_shell

    def assemble_mass_matrix(self) -> spsp.csr_matrix:
        """
        Assembles and returns the unified monolithic sparse consistent mass matrix M.
        """
        # 1. Solid mass
        n_solid_elems = self.solid_domain.mesh.n_elements
        elem_volumes = self.solid_domain.mesh.volumes

        for e in range(n_solid_elems):
            self.solid_domain._m_local_buffer[e] = (
                self.solid_domain.element.compute_mass_matrix(elem_volumes[e])
            )

        M_solid = assemble_sparse_matrix(
            self.solid_domain._m_local_buffer,
            rows=self.solid_domain._sparsity_rows,
            cols=self.solid_domain._sparsity_cols,
            total_dofs=self._n_dofs,
        )

        # 2. Shell mass
        n_shell_elems = self.shell_domain.mesh.n_elements
        node_indices = self.shell_domain.mesh.node_indices
        shell_elems = self.shell_domain.mesh.elements
        solid_nodes = self.solid_domain.mesh.nodes

        for e in range(n_shell_elems):
            coords = solid_nodes[node_indices[shell_elems[e]]]
            self.shell_domain._m_local_buffer[e] = (
                self.shell_domain.element.compute_mass_matrix(node_coords=coords)
            )

        M_shell = assemble_sparse_matrix(
            self.shell_domain._m_local_buffer,
            rows=self._shell_sparsity_rows,
            cols=self._shell_sparsity_cols,
            total_dofs=self._n_dofs,
        )

        return M_solid + M_shell

    # =========================================================================
    # Visualization & Export
    # =========================================================================

    def to_pyvista_solid(
        self,
        factor: float = 1.0,
        include_stresses: bool = False,
        smooth_to_nodes: bool = False,
        include_cell_data: dict[str, np.ndarray] | None = None,
    ) -> pv.UnstructuredGrid:
        """
        Exports the deformed solid continuum mesh of the coupled domain.
        """
        return self.solid_domain.to_pyvista(
            factor=factor,
            include_stresses=include_stresses,
            smooth_to_nodes=smooth_to_nodes,
            include_cell_data=include_cell_data,
        )

    def to_pyvista_shell(self, factor: float = 1.0) -> pv.UnstructuredGrid:
        """
        Exports the deformed shell skin mesh of the coupled domain.
        """
        return self.shell_domain.to_pyvista(factor=factor)

    def to_pyvista(
        self,
        factor: float = 1.0,
        include_stresses: bool = False,
        smooth_to_nodes: bool = False,
        include_cell_data: dict[str, np.ndarray] | None = None,
    ) -> pv.UnstructuredGrid:
        """
        Default visualization method: exports the deformed solid continuum mesh.
        """
        return self.to_pyvista_solid(
            factor=factor,
            include_stresses=include_stresses,
            smooth_to_nodes=smooth_to_nodes,
            include_cell_data=include_cell_data,
        )

    def save_solid_state(self, filepath: Path, **kwargs) -> None:
        """
        Exports the deformed solid continuum state to disk as a VTU unstructured mesh (.vtu).
        """
        filepath = Path(filepath)
        if filepath.suffix != ".vtu":
            filepath = filepath.with_suffix(".vtu")
        self.solid_domain.save_state(filepath, **kwargs)

    def save_shell_state(self, filepath: Path, **kwargs) -> None:
        """
        Exports the deformed shell skin state to disk as a VTU unstructured mesh (.vtu).
        """
        filepath = Path(filepath)
        if filepath.suffix != ".vtu":
            filepath = filepath.with_suffix(".vtu")
        self.shell_domain.save_state(filepath, **kwargs)

    def save_pvd(
        self,
        filepath: Path,
        timestep: float | None = None,
        factor: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Exports both solid continuum and shell skin as .vtu files and creates
        a master ParaView Data (.pvd) collection linking them together.
        """
        filepath = Path(filepath)
        if filepath.suffix != ".pvd":
            filepath = filepath.with_suffix(".pvd")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        stem = filepath.stem
        solid_file = filepath.parent / f"{stem}_solid.vtu"
        shell_file = filepath.parent / f"{stem}_shell.vtu"

        self.save_solid_state(solid_file, factor=factor, **kwargs)
        self.save_shell_state(shell_file, factor=factor, **kwargs)

        writer = PvdCollectionWriter(filepath)
        writer.add_dataset(filename=solid_file.name, timestep=timestep, part=0, group="solid")
        writer.add_dataset(filename=shell_file.name, timestep=timestep, part=1, group="shell")
        writer.write()

    def save_state(self, filepath: Path, **kwargs) -> None:
        """
        Exports coupled domain state.

        - If filepath ends in '.pvd', exports both solid and shell components and a master .pvd.
        - Otherwise exports the solid continuum state (.vtu).
        For explicit field export, prefer save_solid_state(), save_shell_state(), or save_pvd().
        """
        filepath = Path(filepath)
        if filepath.suffix == ".pvd":
            self.save_pvd(filepath, **kwargs)
        else:
            self.save_solid_state(filepath, **kwargs)
