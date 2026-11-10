from __future__ import annotations
import numpy as np
import scipy.sparse as spsp

from psierra.element.tet4_total_lagrangian import Tet4TotalLagrangian
from psierra.material.base import Material
from psierra.mesh.volumetric import VolumetricMesh
from psierra.utility.tensor import tensor_to_voigt_stress
from psierra.utility.indexing import compute_element_dofs
from psierra.utility.assembly import (
    create_coo_sparsity_indices,
    assemble_sparse_matrix,
    assemble_vector,
)


class Tet4Fbar(Tet4TotalLagrangian):
    """
    Total Lagrangian 4-Node Tetrahedral Solid Element with F-bar Volumetric Locking Alleviation.

    Implements the Flory volumetric-isochoric split:
        F_bar = (J_bar / J)^(1/3) * F
    where J_bar is the smoothed volume dilatation averaged over element vertices from
    nodal volume patches:
        v_node_a = sum_{e in E_a} (1/4) * V_e * J_e
        J_node_a = v_node_a / V_0_node_a
        J_bar_e  = (1/4) * sum_{i=1}^4 J_node_{a_i}

    Eliminates volumetric locking in nearly incompressible continua (nu -> 0.5) while
    preserving quadratic Newton-Raphson convergence through consistent deviatoric and
    nodal-coupled volumetric tangent stiffness matrices.
    """

    supports_element_indexing = True

    def __init__(self, material: Material) -> None:
        super().__init__(material)
        self._mesh: VolumetricMesh | None = None
        self._H_list: np.ndarray | None = None
        self._V_list: np.ndarray | None = None
        self._nodal_v0: np.ndarray | None = None
        self._rows_dev: np.ndarray | None = None
        self._cols_dev: np.ndarray | None = None
        self._node_elements: list[np.ndarray] | None = None
        self._rows_vol: np.ndarray | None = None
        self._cols_vol: np.ndarray | None = None
        self._vol_block_slices: list[tuple[int, np.ndarray, int, int]] | None = None
        self._domain_J_bar: np.ndarray | None = None

    # =========================================================================
    # Geometry & Topology Caching for Domain Assembly
    # =========================================================================

    def _setup_cache(self, mesh: VolumetricMesh) -> None:
        """Precomputes and caches reference geometry and sparsity structures."""
        if self._mesh is mesh:
            return

        self._mesh = mesh
        n_elems = mesh.n_elements
        n_nodes = mesh.n_nodes
        elements = mesh.elements
        nodes = mesh.nodes

        H_list = np.zeros((n_elems, 4, 3), dtype=np.float64)
        V_list = np.zeros(n_elems, dtype=np.float64)
        for e in range(n_elems):
            V, H = self.compute_volume_and_gradients(nodes[elements[e]])
            V_list[e] = V
            H_list[e] = H
        self._H_list = H_list
        self._V_list = V_list

        nodal_v0 = np.bincount(
            elements.ravel(),
            weights=np.repeat(0.25 * V_list, 4),
            minlength=n_nodes,
        )
        self._nodal_v0 = nodal_v0

        el_dofs = compute_element_dofs(elements, 3)
        self._rows_dev, self._cols_dev = create_coo_sparsity_indices(el_dofs)

        node_elements: list[list[int]] = [[] for _ in range(n_nodes)]
        for e in range(n_elems):
            for a in elements[e]:
                node_elements[a].append(e)
        self._node_elements = [
            np.array(el_list, dtype=np.int32) for el_list in node_elements
        ]

        vol_rows: list[np.ndarray] = []
        vol_cols: list[np.ndarray] = []
        vol_slices: list[tuple[int, np.ndarray, int, int]] = []
        curr_offset = 0

        for a in range(n_nodes):
            elems_a = self._node_elements[a]
            nodes_a = np.unique(elements[elems_a].ravel())
            dofs_a = np.repeat(nodes_a * 3, 3) + np.tile(np.arange(3), len(nodes_a))
            r_grid, c_grid = np.meshgrid(dofs_a, dofs_a, indexing="ij")
            r_flat = r_grid.ravel()
            c_flat = c_grid.ravel()
            vol_rows.append(r_flat)
            vol_cols.append(c_flat)
            size = len(r_flat)
            vol_slices.append((a, dofs_a, curr_offset, curr_offset + size))
            curr_offset += size

        self._rows_vol = np.concatenate(vol_rows)
        self._cols_vol = np.concatenate(vol_cols)
        self._vol_block_slices = vol_slices

    def _get_bulk_modulus(self) -> float:
        """Retrieves bulk modulus kappa from material parameters or constitutive matrix."""
        params = getattr(self._material, "_parameters", None)
        if params is not None and hasattr(params, "bulk_modulus"):
            return float(params.bulk_modulus)
        C_small = self._material.compute_constitutive_matrix()
        return float(np.sum(C_small[:3, :3])) / 9.0

    # =========================================================================
    # Single-Element Interface (Consistent with Tet4Base & Tet4TotalLagrangian)
    # =========================================================================

    def compute_linear_stiffness_matrix(
        self, node_coords: np.ndarray, elem_index: int | None = None
    ) -> np.ndarray:
        """
        Computes the element linear stiffness matrix (12x12).

        Parameters:
        -----------
        node_coords : np.ndarray of shape (4, 3)
            Reference coordinates X.
        elem_index : int | None, optional
            Index of the element in the mesh (if part of domain assembly).

        Returns:
        --------
        k_e : np.ndarray of shape (12, 12)
        """
        V, H = self.compute_volume_and_gradients(node_coords)
        BL = self.compute_linear_B_matrix(H)
        C = self._material.compute_constitutive_matrix()
        return (BL.T @ C @ BL) * V

    def compute_nonlinear_force_and_tangent_stiffness(
        self,
        node_coords: np.ndarray,
        u_nodes: np.ndarray,
        elem_index: int | None = None,
        J_bar: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the Total Lagrangian F-bar element tangent stiffness matrix k_tan (12x12)
        and internal load vector f_int (12,) for the current deformed state.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (4, 3)
            Reference coordinates X.
        u_nodes : np.ndarray of shape (4, 3)
            Nodal displacements u.
        elem_index : int | None, optional
            Element index for querying cached smoothed Jacobian J_bar.
        J_bar : float | None, optional
            Explicit smoothed volume ratio J_bar. If None and elem_index is provided with
            precomputed domain volumes, uses the precomputed value; otherwise defaults to J.

        Returns:
        --------
        k_tan : np.ndarray of shape (12, 12)
            Element tangent stiffness matrix.
        f_int : np.ndarray of shape (12,)
            Element internal force vector.
        """
        V, H = self.compute_volume_and_gradients(node_coords)
        l = u_nodes.T @ H
        F = np.eye(3, dtype=np.float64) + l
        J = float(np.linalg.det(F))
        if J <= 0.0:
            raise ValueError(f"Element inversion detected: det(F) = {J:.4e} <= 0.")

        if J_bar is None:
            if elem_index is not None and self._domain_J_bar is not None:
                J_bar = float(self._domain_J_bar[elem_index])
            else:
                J_bar = J

        ratio = (J_bar / J) ** (1.0 / 3.0)
        F_bar = ratio * F
        C_tan, S_bar = self._material.compute_stress_and_tangent(F_bar)

        B = self.compute_nonlinear_B_matrix(H, u_nodes)
        G = self.compute_geometric_matrix(H)

        b_vol = (B[0, :] + B[1, :] + B[2, :]) / 3.0
        m_voigt = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        B_dev = B - np.outer(m_voigt, b_vol)
        B_bar = (ratio**2.0) * B_dev + np.outer(m_voigt, b_vol)

        k_mat = (B_bar.T @ C_tan @ B_bar) * V
        S_hat = np.kron(np.identity(3, dtype=np.float64), S_bar)
        k_geo = (G.T @ S_hat @ G) * V
        k_tan = k_mat + k_geo

        S_voigt = tensor_to_voigt_stress(S_bar)
        f_int = (B_bar.T @ S_voigt) * V

        return k_tan, f_int

    # =========================================================================
    # Domain-Level Assembly Interface
    # =========================================================================

    def prepare_for_linear_assembly(self, mesh: VolumetricMesh) -> None:
        """Initializes geometry and topology cache for linear assembly."""
        self._setup_cache(mesh)

    def prepare_for_assembly(
        self, mesh: VolumetricMesh, displacement: np.ndarray
    ) -> None:
        """Computes and updates nodal-averaged volume ratio J_bar across the domain."""
        self._setup_cache(mesh)
        assert self._H_list is not None
        assert self._V_list is not None
        assert self._nodal_v0 is not None

        n_elems = mesh.n_elements
        n_nodes = mesh.n_nodes
        elements = mesh.elements
        nodes = mesh.nodes
        disp = displacement.reshape(-1, 3)

        J_arr = np.zeros(n_elems, dtype=np.float64)
        eye3 = np.eye(3, dtype=np.float64)
        for e in range(n_elems):
            elem_nodes = elements[e]
            u_elem = disp[elem_nodes]
            l = u_elem.T @ self._H_list[e]
            F = eye3 + l
            J = float(np.linalg.det(F))
            if J <= 0.0:
                raise ValueError(
                    f"Element inversion detected at element {e}: det(F) = {J:.4e} <= 0."
                )
            J_arr[e] = J

        v_node = np.bincount(
            elements.ravel(),
            weights=np.repeat(0.25 * self._V_list * J_arr, 4),
            minlength=n_nodes,
        )
        J_node = v_node / self._nodal_v0
        self._domain_J_bar = np.mean(J_node[elements], axis=1)

    def assemble_stiffness_matrix(self, mesh: VolumetricMesh) -> spsp.csr_matrix:
        """Assembles linear global stiffness matrix with nodal B-bar volumetric split."""
        self._setup_cache(mesh)
        disp0 = np.zeros((mesh.n_nodes, 3), dtype=np.float64)
        return self.assemble_tangent_stiffness_matrix(mesh, disp0)

    def assemble_internal_forces(
        self, mesh: VolumetricMesh, displacement: np.ndarray
    ) -> np.ndarray:
        """
        Assembles the global internal force vector using nodal F-bar formulation.

        Parameters:
        -----------
        mesh : VolumetricMesh
            Reference tetrahedral mesh.
        displacement : np.ndarray of shape (n_nodes, 3) or (n_dofs,)
            Current nodal displacements.

        Returns:
        --------
        f_int : np.ndarray of shape (n_dofs,)
        """
        self._setup_cache(mesh)
        assert self._H_list is not None
        assert self._V_list is not None
        assert self._nodal_v0 is not None

        n_elems = mesh.n_elements
        n_nodes = mesh.n_nodes
        elements = mesh.elements
        nodes = mesh.nodes
        disp = displacement.reshape(-1, 3)

        F_list = np.zeros((n_elems, 3, 3), dtype=np.float64)
        J_arr = np.zeros(n_elems, dtype=np.float64)
        eye3 = np.eye(3, dtype=np.float64)
        for e in range(n_elems):
            elem_nodes = elements[e]
            u_elem = disp[elem_nodes]
            l = u_elem.T @ self._H_list[e]
            F = eye3 + l
            J = float(np.linalg.det(F))
            if J <= 0.0:
                raise ValueError(
                    f"Element inversion detected at element {e}: det(F) = {J:.4e} <= 0."
                )
            F_list[e] = F
            J_arr[e] = J

        v_node = np.bincount(
            elements.ravel(),
            weights=np.repeat(0.25 * self._V_list * J_arr, 4),
            minlength=n_nodes,
        )
        J_node = v_node / self._nodal_v0
        J_bar_elems = np.mean(J_node[elements], axis=1)
        self._domain_J_bar = J_bar_elems

        f_local = np.zeros((n_elems, 12), dtype=np.float64)
        m_voigt = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        for e in range(n_elems):
            ratio = (J_bar_elems[e] / J_arr[e]) ** (1.0 / 3.0)
            F_bar = ratio * F_list[e]
            if getattr(self._material, "supports_element_indexing", False):
                _, S_bar = self._material.compute_stress_and_tangent(F_bar, elem_index=e)
            else:
                _, S_bar = self._material.compute_stress_and_tangent(F_bar)

            u_elem = disp[elements[e]]
            B_tot = self.compute_nonlinear_B_matrix(self._H_list[e], u_elem)
            b_vol = (B_tot[0] + B_tot[1] + B_tot[2]) / 3.0
            B_dev = B_tot - np.outer(m_voigt, b_vol)
            B_bar = (ratio**2.0) * B_dev + np.outer(m_voigt, b_vol)

            S_voigt = tensor_to_voigt_stress(S_bar)
            f_local[e] = (B_bar.T @ S_voigt) * self._V_list[e]

        el_dofs = compute_element_dofs(elements, 3)
        return assemble_vector(f_local, el_dofs, mesh.n_dofs)

    def assemble_tangent_stiffness_matrix(
        self, mesh: VolumetricMesh, displacement: np.ndarray
    ) -> spsp.csr_matrix:
        """
        Assembles the consistent global tangent stiffness matrix K_tan(u) = K_dev + K_vol.

        Parameters:
        -----------
        mesh : VolumetricMesh
            Reference tetrahedral mesh.
        displacement : np.ndarray of shape (n_nodes, 3) or (n_dofs,)
            Current nodal displacements.

        Returns:
        --------
        K_tan : spsp.csr_matrix of shape (n_dofs, n_dofs)
        """
        self._setup_cache(mesh)
        assert self._H_list is not None
        assert self._V_list is not None
        assert self._nodal_v0 is not None
        assert self._rows_dev is not None
        assert self._cols_dev is not None
        assert self._node_elements is not None
        assert self._rows_vol is not None
        assert self._cols_vol is not None
        assert self._vol_block_slices is not None

        n_elems = mesh.n_elements
        n_nodes = mesh.n_nodes
        elements = mesh.elements
        nodes = mesh.nodes
        disp = displacement.reshape(-1, 3)
        K_bulk = self._get_bulk_modulus()

        F_list = np.zeros((n_elems, 3, 3), dtype=np.float64)
        J_arr = np.zeros(n_elems, dtype=np.float64)
        eye3 = np.eye(3, dtype=np.float64)
        for e in range(n_elems):
            elem_nodes = elements[e]
            u_elem = disp[elem_nodes]
            l = u_elem.T @ self._H_list[e]
            F = eye3 + l
            J = float(np.linalg.det(F))
            if J <= 0.0:
                raise ValueError(
                    f"Element inversion detected at element {e}: det(F) = {J:.4e} <= 0."
                )
            F_list[e] = F
            J_arr[e] = J

        v_node = np.bincount(
            elements.ravel(),
            weights=np.repeat(0.25 * self._V_list * J_arr, 4),
            minlength=n_nodes,
        )
        J_node = v_node / self._nodal_v0
        J_bar_elems = np.mean(J_node[elements], axis=1)
        self._domain_J_bar = J_bar_elems

        m_voigt = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        C_vol = K_bulk * np.outer(m_voigt, m_voigt)

        k_dev_local = np.zeros((n_elems, 12, 12), dtype=np.float64)
        div_e_list = np.zeros((n_elems, 12), dtype=np.float64)

        for e in range(n_elems):
            ratio = (J_bar_elems[e] / J_arr[e]) ** (1.0 / 3.0)
            F_bar = ratio * F_list[e]
            if getattr(self._material, "supports_element_indexing", False):
                C_tan, S_bar = self._material.compute_stress_and_tangent(F_bar, elem_index=e)
            else:
                C_tan, S_bar = self._material.compute_stress_and_tangent(F_bar)
            C_dev = C_tan - C_vol

            u_elem = disp[elements[e]]
            B_tot = self.compute_nonlinear_B_matrix(self._H_list[e], u_elem)
            div_e_list[e] = B_tot[0] + B_tot[1] + B_tot[2]

            b_vol = div_e_list[e] / 3.0
            B_dev = B_tot - np.outer(m_voigt, b_vol)
            B_bar_dev = (ratio**2.0) * B_dev

            k_mat_dev = (B_bar_dev.T @ C_dev @ B_bar_dev) * self._V_list[e]
            G = self.compute_geometric_matrix(self._H_list[e])
            S_hat = np.kron(np.identity(3, dtype=np.float64), S_bar)
            k_geo = (G.T @ S_hat @ G) * self._V_list[e]

            k_dev_local[e] = k_mat_dev + k_geo

        K_dev = assemble_sparse_matrix(
            element_matrices=k_dev_local,
            rows=self._rows_dev,
            cols=self._cols_dev,
            total_dofs=mesh.n_dofs,
        )

        vol_data = np.zeros(len(self._rows_vol), dtype=np.float64)
        el_dofs = compute_element_dofs(elements, 3)

        for a, dofs_a, s_start, s_end in self._vol_block_slices:
            elems_a = self._node_elements[a]
            g_a = np.zeros(len(dofs_a), dtype=np.float64)
            dof_to_local = {dof: idx for idx, dof in enumerate(dofs_a)}
            for e in elems_a:
                w_e = 0.25 * self._V_list[e] * J_arr[e] / self._nodal_v0[a]
                div_e = div_e_list[e]
                for idx_dof, global_dof in enumerate(el_dofs[e]):
                    loc_idx = dof_to_local[global_dof]
                    g_a[loc_idx] += w_e * div_e[idx_dof]

            block_a = (K_bulk * self._nodal_v0[a]) * np.outer(g_a, g_a)
            vol_data[s_start:s_end] = block_a.ravel()

        K_vol = spsp.coo_matrix(
            (vol_data, (self._rows_vol, self._cols_vol)),
            shape=(mesh.n_dofs, mesh.n_dofs),
        ).tocsr()

        return K_dev + K_vol
