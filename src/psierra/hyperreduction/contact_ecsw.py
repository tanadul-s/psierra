"""
Contact-Aware ECSW & Enriched POD Basis (Method 2 Hyper-Reduced Contact).
========================================================================
Implements Method 2 hyper-reduced contact mechanics:
1. Contact-Aware Reduced Integration Domain (RID) with contact boundary clustering.
2. Joint Cubature Training Matrix incorporating virtual work from both internal continuum
   and active contact boundary interfaces:
       A_joint = [ A_vol ; alpha_c * A_contact ]
       b_joint = [ b_vol ; alpha_c * b_contact ]
3. Automatic element clustering around candidate contact interfaces.
4. ContactAwareHyperReducedModel combining EnrichedPODBasis, ContactAwareRID, and contact operators
   with guaranteed tangent stiffness symmetry (K_r = K_r^T).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import time
import numpy as np
import scipy.optimize as spopt

from psierra.domain.solid import SolidDomain
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint
from psierra.damping.rayleigh import RayleighDamping
from psierra.reduction.basis import PODBasis, EnrichedPODBasis
from psierra.hyperreduction.ecsw import ReducedIntegrationDomain
from psierra.hyperreduction.model import HyperReducedNonlinearModel
from psierra.hyperreduction.contact import HyperReducedRigidContact


@dataclass
class ContactAwareReducedIntegrationDomain(ReducedIntegrationDomain):
    """
    Contact-Aware Reduced Integration Domain (RID).

    Extends standard ECSW cubature with explicit tracking and clustering of
    elements located within the contact interface boundary zone.

    Attributes:
    -----------
    contact_element_indices : np.ndarray
        Global indices of selected cubature elements that share nodes with the
        candidate contact surface Gamma_c.
    bulk_element_indices : np.ndarray
        Global indices of selected cubature elements residing entirely in the
        interior / bulk domain.
    n_contact_elements_in_full_mesh : int
        Total number of elements in the full-order mesh that touch the contact surface.
    """

    contact_element_indices: np.ndarray = None  # type: ignore[assignment]
    bulk_element_indices: np.ndarray = None  # type: ignore[assignment]
    n_contact_elements_in_full_mesh: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.contact_element_indices is None:
            self.contact_element_indices = np.array([], dtype=np.int64)
        else:
            self.contact_element_indices = np.asarray(
                self.contact_element_indices, dtype=np.int64
            ).ravel()

        if self.bulk_element_indices is None:
            self.bulk_element_indices = np.array([], dtype=np.int64)
        else:
            self.bulk_element_indices = np.asarray(
                self.bulk_element_indices, dtype=np.int64
            ).ravel()

        self.n_contact_elements_in_full_mesh = int(self.n_contact_elements_in_full_mesh)

    @property
    def n_contact_selected(self) -> int:
        """Number of contact-zone elements retained in the cubature domain."""
        return int(self.contact_element_indices.size)

    @property
    def n_bulk_selected(self) -> int:
        """Number of interior / bulk elements retained in the cubature domain."""
        return int(self.bulk_element_indices.size)

    @property
    def contact_cluster_ratio(self) -> float:
        """
        Fraction of full-mesh contact elements retained in RID:
            n_contact_selected / max(1, n_contact_elements_in_full_mesh)
        """
        if self.n_contact_elements_in_full_mesh <= 0:
            return 0.0
        return float(self.n_contact_selected / self.n_contact_elements_in_full_mesh)

    def save(self, path: Path | str, **extra_metadata) -> Path:
        """Saves the contact-aware reduced integration domain to a .npz archive."""
        out_path = Path(path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        meta = dict(self.metadata or {})
        meta.update(extra_metadata)

        np.savez_compressed(
            out_path,
            element_indices=self.element_indices,
            weights=self.weights,
            n_total_elements=np.array([self.n_total_elements], dtype=np.int64),
            relative_error=np.array([self.relative_error], dtype=np.float64),
            contact_element_indices=self.contact_element_indices,
            bulk_element_indices=self.bulk_element_indices,
            n_contact_elements_in_full_mesh=np.array(
                [self.n_contact_elements_in_full_mesh], dtype=np.int64
            ),
            metadata=meta,
        )
        return out_path

    @classmethod
    def load(cls, path: Path | str) -> ContactAwareReducedIntegrationDomain:
        """Loads a ContactAwareReducedIntegrationDomain from a compressed .npz archive."""
        src_path = Path(path).resolve()
        if not src_path.is_file():
            raise FileNotFoundError(f"Contact-aware ECSW RID archive not found: {src_path}")

        with np.load(src_path, allow_pickle=True) as data:
            element_indices = data["element_indices"]
            weights = data["weights"]
            n_total_elements = int(data["n_total_elements"][0])
            relative_error = float(data["relative_error"][0])
            contact_elem_indices = (
                data["contact_element_indices"]
                if "contact_element_indices" in data
                else np.array([], dtype=np.int64)
            )
            bulk_elem_indices = (
                data["bulk_element_indices"]
                if "bulk_element_indices" in data
                else np.array([], dtype=np.int64)
            )
            n_contact_full = (
                int(data["n_contact_elements_in_full_mesh"][0])
                if "n_contact_elements_in_full_mesh" in data
                else 0
            )
            metadata = data["metadata"].item() if "metadata" in data else {}

        return cls(
            element_indices=element_indices,
            weights=weights,
            n_total_elements=n_total_elements,
            relative_error=relative_error,
            contact_element_indices=contact_elem_indices,
            bulk_element_indices=bulk_elem_indices,
            n_contact_elements_in_full_mesh=n_contact_full,
            metadata=metadata,
        )


class ContactAwareECSWTrainingMatrix:
    """
    Constructs the Joint Virtual Work Training Matrix for Contact-Aware ECSW.

    Governing Mathematical Formulation:
    -----------------------------------
    The virtual work residual incorporates both continuum internal force virtual work
    and boundary contact interface virtual work across all snapshot states:

        A_joint = [         A_vol         ]  in R^{(m_v*r + m_c*r) x N_e}
                  [ alpha_c * A_contact   ]

        b_joint = [         b_vol         ]  in R^{m_v*r + m_c*r}
                  [ alpha_c * b_contact   ]

    For an element e in the full mesh:
    1. Volumetric Virtual Work:
       c_{v, e}^{(k)} = Phi_e^T * f_{int, e}(u_e^{(k)}) in R^r
    2. Contact Interface Virtual Work:
       If element e has nodes on the candidate contact boundary Gamma_c:
           f_{c, e, i}^{(j)} = f_{c, i}^{(j)} / valence(i)
           c_{c, e}^{(j)} = Phi_e^T * f_{c, e}^{(j)} in R^r
       If element e does not touch Gamma_c:
           c_{c, e}^{(j)} = 0 in R^r.

    Exact Conservation Property:
        sum_{e=1}^{N_e} c_{c, e}^{(j)} = Phi^T * f_c^{(j)} = b_{c}^{(j)}
    holds identically across all contact snapshots.
    """

    @staticmethod
    def build_joint(
        domain: SolidDomain,
        basis: PODBasis | EnrichedPODBasis,
        snapshots: np.ndarray | Sequence[np.ndarray],
        contact_forces: np.ndarray | Sequence[np.ndarray] | None = None,
        contact_nodes: Sequence[int] | np.ndarray | None = None,
        alpha_c: float | None = None,
        include_stiffness: bool = False,
        stiffness_weight: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Builds the joint continuum-contact training matrix A_joint and target vector b_joint.

        Parameters:
        -----------
        domain : SolidDomain
            Physical continuum domain.
        basis : PODBasis or EnrichedPODBasis
            Reduced modal basis Phi in R^(n_dofs x r).
        snapshots : np.ndarray of shape (n_dofs, m) or sequence of (n_dofs,) vectors
            State displacement snapshot vectors.
        contact_forces : np.ndarray of shape (n_dofs, m_c) or sequence of (n_dofs,) vectors, optional
            Nodal contact force vectors corresponding to contact-active states.
        contact_nodes : Sequence[int] or np.ndarray, optional
            Node indices comprising the candidate contact interface Gamma_c.
            If None, inferred from nodes where contact_forces are non-zero.
        alpha_c : float, optional
            Weighting parameter balancing contact virtual work with volumetric virtual work.
            If None, automatically balances the row Frobenius norms of both blocks.
        include_stiffness : bool, default=False
            Whether to augment the training matrix with the initial linear stiffness projection.
        stiffness_weight : float, optional
            Multiplier for stiffness augmentation block scaling.

        Returns:
        --------
        A_joint : np.ndarray
            Joint virtual work training matrix.
        b_joint : np.ndarray
            Target virtual work vector.
        info : dict
            Diagnostic metrics (block row sizes, scaling factor alpha_c, contact element count).
        """
        # Format snapshot matrix
        if isinstance(snapshots, np.ndarray):
            snap_mat = snapshots if snapshots.ndim == 2 else snapshots.reshape(-1, 1)
        elif isinstance(snapshots, (list, tuple)):
            if len(snapshots) == 0:
                raise ValueError("snapshots sequence cannot be empty.")
            snap_mat = np.column_stack(snapshots)
        else:
            raise ValueError(f"Unsupported snapshot type: {type(snapshots)}")

        n_dofs, m_snapshots = snap_mat.shape
        if n_dofs != domain.n_dofs:
            raise ValueError(
                f"Snapshot DOF count ({n_dofs}) does not match domain DOF count ({domain.n_dofs})."
            )

        r = basis.n_modes
        n_elems = domain.mesh.n_elements
        nodes = domain.mesh.nodes
        elements = domain.mesh.elements
        elem_form = domain.element
        dofs_per_node = elem_form.dofs_per_node
        Phi = basis.basis_matrix

        # 1. Precompute elemental DOFs and elemental modal sub-bases Phi_e
        elem_dofs_list: list[np.ndarray] = []
        Phi_e_list: list[np.ndarray] = []
        for e in range(n_elems):
            elem_nodes = elements[e]
            edofs = np.concatenate(
                [
                    np.arange(dofs_per_node * nid, dofs_per_node * nid + dofs_per_node)
                    for nid in elem_nodes
                ]
            )
            elem_dofs_list.append(edofs)
            Phi_e_list.append(Phi[edofs, :])

        # 2. Build Volumetric Virtual Work Block A_vol
        M_vol_rows = m_snapshots * r
        A_vol = np.zeros((M_vol_rows, n_elems), dtype=np.float64)

        for k in range(m_snapshots):
            u_k = snap_mat[:, k]
            u_k_nodes = u_k.reshape(-1, dofs_per_node)
            row_start = k * r
            row_end = row_start + r

            for e in range(n_elems):
                elem_nodes = elements[e]
                coords = nodes[elem_nodes]
                u_elem = u_k_nodes[elem_nodes]

                _, f_int_e = elem_form.compute_nonlinear_force_and_tangent_stiffness(
                    coords, u_elem
                )
                f_int_e = f_int_e.ravel()

                Phi_e = Phi_e_list[e]
                c_e = Phi_e.T @ f_int_e
                A_vol[row_start:row_end, e] = c_e

        b_vol = A_vol.sum(axis=1)

        # 3. Build Contact Interface Virtual Work Block A_contact
        A_contact = np.zeros((0, n_elems), dtype=np.float64)
        b_contact = np.zeros(0, dtype=np.float64)
        contact_elem_indices: np.ndarray = np.array([], dtype=np.int64)

        has_contact_data = contact_forces is not None and len(contact_forces) > 0

        if has_contact_data:
            if isinstance(contact_forces, np.ndarray):
                fc_mat = (
                    contact_forces
                    if contact_forces.ndim == 2
                    else contact_forces.reshape(-1, 1)
                )
            elif isinstance(contact_forces, (list, tuple)):
                fc_mat = np.column_stack(contact_forces)
            else:
                raise ValueError(f"Unsupported contact_forces type: {type(contact_forces)}")

            m_contact = fc_mat.shape[1]

            # Resolve contact candidate nodes
            if contact_nodes is not None:
                c_nodes_set = set(np.asarray(contact_nodes, dtype=np.int64).ravel())
            else:
                # Infer from active contact force entries
                fc_norms = np.linalg.norm(fc_mat.reshape(-1, dofs_per_node, m_contact), axis=(1, 2))
                c_nodes_set = set(np.flatnonzero(fc_norms > 1e-12))

            # Compute node valences (number of elements sharing each node)
            valences = np.zeros(domain.mesh.n_nodes, dtype=np.int64)
            for e in range(n_elems):
                valences[elements[e]] += 1
            valences_safe = np.maximum(valences, 1)

            # Identify contact elements (elements touching candidate contact nodes)
            c_elem_list = [
                e for e in range(n_elems) if any(nid in c_nodes_set for nid in elements[e])
            ]
            contact_elem_indices = np.array(c_elem_list, dtype=np.int64)

            # Assemble A_contact: (m_contact * r, N_e)
            M_contact_rows = m_contact * r
            A_contact = np.zeros((M_contact_rows, n_elems), dtype=np.float64)

            for j in range(m_contact):
                fc_j = fc_mat[:, j]
                fc_j_nodes = fc_j.reshape(-1, dofs_per_node)
                row_start = j * r
                row_end = row_start + r

                for e in contact_elem_indices:
                    elem_nodes = elements[e]
                    f_c_e = np.zeros((len(elem_nodes), dofs_per_node), dtype=np.float64)

                    for loc_idx, nid in enumerate(elem_nodes):
                        if nid in c_nodes_set:
                            f_c_e[loc_idx] = fc_j_nodes[nid] / valences_safe[nid]

                    f_c_e_flat = f_c_e.ravel()
                    Phi_e = Phi_e_list[e]
                    c_c_e = Phi_e.T @ f_c_e_flat
                    A_contact[row_start:row_end, e] = c_c_e

            b_contact = A_contact.sum(axis=1)

        # 4. Resolve Interface Weighting Factor alpha_c
        resolved_alpha_c = 1.0
        if has_contact_data and A_contact.shape[0] > 0:
            norm_vol = float(np.linalg.norm(A_vol)) / np.sqrt(max(A_vol.shape[0], 1))
            norm_contact = float(np.linalg.norm(A_contact)) / np.sqrt(
                max(A_contact.shape[0], 1)
            )
            if alpha_c is not None:
                resolved_alpha_c = float(alpha_c)
            elif norm_vol > 1e-12 and norm_contact > 1e-12:
                # Auto-balance block magnitudes
                resolved_alpha_c = norm_vol / norm_contact
            else:
                resolved_alpha_c = 1.0

            A_contact_scaled = A_contact * resolved_alpha_c
            b_contact_scaled = b_contact * resolved_alpha_c
            A_joint = np.vstack([A_vol, A_contact_scaled])
            b_joint = np.concatenate([b_vol, b_contact_scaled])
        else:
            A_joint = A_vol
            b_joint = b_vol

        # 5. Optional Linear Stiffness Augmentation
        if include_stiffness:
            triu_idx = np.triu_indices(r)
            n_triu = len(triu_idx[0])
            A_stiff = np.zeros((n_triu, n_elems), dtype=np.float64)
            u_zero = np.zeros(
                (elem_form.nodes_per_element, elem_form.dofs_per_node), dtype=np.float64
            )

            for e in range(n_elems):
                coords = nodes[elements[e]]
                K_0_e, _ = elem_form.compute_nonlinear_force_and_tangent_stiffness(
                    coords, u_zero
                )
                Phi_e = Phi_e_list[e]
                K_r_e = Phi_e.T @ K_0_e @ Phi_e
                A_stiff[:, e] = K_r_e[triu_idx]

            s_force = float(np.linalg.norm(A_joint)) / np.sqrt(max(A_joint.shape[0], 1))
            s_stiff = float(np.linalg.norm(A_stiff)) / np.sqrt(max(n_triu, 1))
            if s_force > 1e-12 and s_stiff > 1e-12:
                weight = (
                    float(stiffness_weight) if stiffness_weight is not None else 1.0
                )
                A_stiff *= (s_force / s_stiff) * weight

            A_joint = np.vstack([A_joint, A_stiff])
            b_joint = np.concatenate([b_joint, A_stiff.sum(axis=1)])

        info = {
            "n_vol_rows": M_vol_rows,
            "n_contact_rows": A_contact.shape[0],
            "n_total_rows": A_joint.shape[0],
            "n_elements": n_elems,
            "alpha_c": resolved_alpha_c,
            "n_contact_elements": len(contact_elem_indices),
            "contact_elements": contact_elem_indices,
        }

        return A_joint, b_joint, info


class ContactAwareECSWOptimizer:
    """
    Contact-Aware Greedy Active-Set NNLS Optimizer with Contact Boundary Clustering.

    Solves the constrained cubature optimization problem:
        min_{w >= 0} ||A_joint * w - b_joint||_2^2   subject to  ||w||_0 is small
    while enforcing clustered element sampling in the high-gradient contact zone.

    Clustering Strategy:
    --------------------
    1. Elements in the candidate contact zone E_contact are prioritized with a
       scoring boost factor beta_c >= 1.0:
           s_e = (a_e^T * r) / (||a_e||_2 + eps) * (beta_c if e in E_contact else 1.0)
    2. An optional lower bound min_contact_elements guarantees that a minimum number
       of cubature points reside directly beneath the contact interface.
    3. Guarantees strictly positive weights (w_e > 0) via Lawson-Hanson active-set NNLS.
    """

    def __init__(
        self,
        tolerance: float = 0.01,
        max_elements: int | None = None,
        contact_priority_weight: float = 1.5,
        min_contact_elements: int = 0,
        verbose: bool = True,
    ) -> None:
        """
        Parameters:
        -----------
        tolerance : float, default=0.01
            Target relative residual error ||r||_2 / ||b||_2.
        max_elements : int, optional
            Upper limit on total retained cubature elements.
        contact_priority_weight : float, default=1.5
            Greedy scoring boost multiplier beta_c >= 1.0 for elements touching Gamma_c.
        min_contact_elements : int, default=0
            Minimum number of contact-zone elements required in the resulting RID.
        verbose : bool, default=True
            Whether to display real-time optimization logs.
        """
        if tolerance <= 0.0:
            raise ValueError(f"tolerance must be strictly positive, got {tolerance}")
        if contact_priority_weight < 1.0:
            raise ValueError(
                f"contact_priority_weight must be >= 1.0, got {contact_priority_weight}"
            )
        self.tolerance = float(tolerance)
        self.max_elements = int(max_elements) if max_elements is not None else None
        self.contact_priority_weight = float(contact_priority_weight)
        self.min_contact_elements = int(max(0, min_contact_elements))
        self.verbose = bool(verbose)

    def optimize(
        self,
        A: np.ndarray,
        b: np.ndarray | None = None,
        contact_elements: Sequence[int] | np.ndarray | None = None,
        tolerance: float | None = None,
        max_elements: int | None = None,
    ) -> ContactAwareReducedIntegrationDomain:
        """
        Executes contact-aware greedy active-set NNLS cubature selection.

        Parameters:
        -----------
        A : np.ndarray of shape (M, N_e)
            Joint training matrix (volumetric + contact).
        b : np.ndarray of shape (M,), optional
            Target vector. If None, computes b = A.sum(axis=1).
        contact_elements : Sequence[int] or np.ndarray, optional
            Global indices of elements residing on the contact boundary Gamma_c.
        tolerance : float, optional
            Overrides default residual tolerance.
        max_elements : int, optional
            Overrides default maximum elements limit.

        Returns:
        --------
        rid : ContactAwareReducedIntegrationDomain
        """
        t_start = time.perf_counter()
        A = np.asarray(A, dtype=np.float64)
        M_rows, n_elems = A.shape

        if b is not None:
            b = np.asarray(b, dtype=np.float64).ravel()
            if b.size != M_rows:
                raise ValueError(f"Target vector size ({b.size}) must match rows of A ({M_rows}).")
        else:
            b = A.sum(axis=1)

        norm_b = float(np.linalg.norm(b))
        if norm_b < 1e-14:
            norm_b = 1.0

        tol = float(tolerance) if tolerance is not None else self.tolerance
        p_max = int(max_elements) if max_elements is not None else (self.max_elements or n_elems)
        p_max = min(p_max, n_elems)

        # Contact elements set
        if contact_elements is not None:
            c_elem_set = set(np.asarray(contact_elements, dtype=np.int64).ravel())
        else:
            c_elem_set = set()
        n_contact_total = len(c_elem_set)

        # Precompute column norms
        col_norms = np.linalg.norm(A, axis=0)
        col_norms_safe = np.where(col_norms > 1e-12, col_norms, 1.0)

        # Build contact priority multiplier vector
        priority_multipliers = np.ones(n_elems, dtype=np.float64)
        if c_elem_set:
            c_indices = np.array(list(c_elem_set), dtype=np.int64)
            priority_multipliers[c_indices] = self.contact_priority_weight

        active_indices: list[int] = []
        weights: np.ndarray = np.array([], dtype=np.float64)
        residual = b.copy()
        rel_error = 1.0

        if self.verbose:
            print(
                f"[Contact-ECSW] Starting joint NNLS cubature: {n_elems:,} total elements "
                f"({n_contact_total:,} in contact zone), target error <= {tol*100:.2f}%, "
                f"max elements = {p_max}, priority boost = {self.contact_priority_weight:.2f}"
            )

        for iteration in range(1, p_max + 1):
            in_active_mask = np.zeros(n_elems, dtype=bool)
            if active_indices:
                in_active_mask[active_indices] = True

            # Candidate scoring with contact priority boost
            proj_scores = (A.T @ residual) / col_norms_safe
            proj_scores *= priority_multipliers
            proj_scores[in_active_mask] = -np.inf

            # Enforce minimum contact elements constraint if needed
            n_contact_current = sum(1 for idx in active_indices if idx in c_elem_set)
            if n_contact_current < self.min_contact_elements and c_elem_set:
                # Restrict candidate search to unselected contact elements
                c_candidates = [idx for idx in c_elem_set if not in_active_mask[idx]]
                if c_candidates:
                    c_cand_arr = np.array(c_candidates, dtype=np.int64)
                    best_elem = int(c_cand_arr[np.argmax(proj_scores[c_cand_arr])])
                else:
                    best_elem = int(np.argmax(proj_scores))
            else:
                best_elem = int(np.argmax(proj_scores))

            if np.isneginf(proj_scores[best_elem]):
                break

            active_indices.append(best_elem)

            # Solve NNLS on active sub-matrix
            A_sub = A[:, active_indices]
            w_sub, _ = spopt.nnls(A_sub, b)

            # Prune elements with zero weight
            max_w = np.max(w_sub) if w_sub.size > 0 else 0.0
            cutoff = 1e-10 * max(max_w, 1.0)
            nonzero_pos = np.where(w_sub > cutoff)[0]

            active_indices = [active_indices[i] for i in nonzero_pos]
            weights = w_sub[nonzero_pos]

            residual = b - A[:, active_indices] @ weights
            rel_error = float(np.linalg.norm(residual) / norm_b)

            if self.verbose and (
                iteration % 5 == 0 or rel_error <= tol or iteration == p_max
            ):
                n_c_act = sum(1 for idx in active_indices if idx in c_elem_set)
                print(
                    f"  • Iteration {iteration:3d}: {len(active_indices):3d} elements "
                    f"({n_c_act:2d} contact, {len(active_indices)-n_c_act:2d} bulk) | "
                    f"Relative error = {rel_error*100:.3f}%"
                )

            # Check termination
            n_contact_current = sum(1 for idx in active_indices if idx in c_elem_set)
            if rel_error <= tol and n_contact_current >= self.min_contact_elements:
                break

        t_elapsed = time.perf_counter() - t_start

        # Classify final retained elements into contact vs bulk
        act_arr = np.array(active_indices, dtype=np.int64)
        is_c = np.array([idx in c_elem_set for idx in act_arr], dtype=bool)
        contact_retained = act_arr[is_c]
        bulk_retained = act_arr[~is_c]

        if self.verbose:
            print(
                f"[Contact-ECSW] Completed in {t_elapsed:.3f} s: {len(act_arr):,}/{n_elems:,} "
                f"elements ({len(contact_retained):,} contact, {len(bulk_retained):,} bulk) | "
                f"Final error = {rel_error*100:.3f}%"
            )

        return ContactAwareReducedIntegrationDomain(
            element_indices=act_arr,
            weights=weights,
            n_total_elements=n_elems,
            relative_error=rel_error,
            contact_element_indices=contact_retained,
            bulk_element_indices=bulk_retained,
            n_contact_elements_in_full_mesh=n_contact_total,
            metadata={
                "method": "ContactAwareECSW_Method2",
                "tolerance": tol,
                "contact_priority_weight": self.contact_priority_weight,
                "min_contact_elements": self.min_contact_elements,
                "elapsed_seconds": t_elapsed,
            },
        )


class ContactAwareHyperReducedModel(HyperReducedNonlinearModel):
    """
    Method 2 Contact-Aware Hyper-Reduced Mechanical Model.

    Combines:
    - EnrichedPODBasis: Orthonormal basis capturing global bulk modes + localized contact indentation.
    - ContactAwareReducedIntegrationDomain: Clustered cubature mesh directly supporting contact interface.
    - HyperReducedRigidContact or two-way contact operators.

    Guarantees:
    - Strict Tangent Stiffness Symmetry: K_r = K_r^T (variational energy conservation).
    - Microsecond Execution: O(|RID|) elemental evaluations + O(r^2) dense solves.
    """

    def __init__(
        self,
        domain: SolidDomain,
        basis: PODBasis | EnrichedPODBasis,
        rid: ContactAwareReducedIntegrationDomain,
        dirichlet: DirichletManager | Sequence[DirichletConstraint] | None = None,
        damping: RayleighDamping | None = None,
        f_ext_base: np.ndarray | None = None,
        surface_nodes: np.ndarray | None = None,
        contact: HyperReducedRigidContact | None = None,
    ) -> None:
        super().__init__(
            domain=domain,
            basis=basis,
            rid=rid,
            dirichlet=dirichlet,
            damping=damping,
            f_ext_base=f_ext_base,
            surface_nodes=surface_nodes,
        )
        self.contact_rid: ContactAwareReducedIntegrationDomain = rid
        self.contact: HyperReducedRigidContact | None = contact

    @property
    def is_enriched(self) -> bool:
        """Whether the model utilizes an EnrichedPODBasis with localized contact modes."""
        return isinstance(self.basis, EnrichedPODBasis)

    def evaluate_contact(
        self, u_r: np.ndarray, v_r: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates projected contact forces and symmetric tangent stiffness in R^r.

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,)
            Reduced displacement coordinates.
        v_r : np.ndarray of shape (r,), optional
            Reduced velocity coordinates (for contact velocity damping).

        Returns:
        --------
        f_r_c : np.ndarray of shape (r,)
            Projected contact restoring force vector.
        K_r_c : np.ndarray of shape (r, r)
            Projected symmetric contact tangent stiffness matrix (K_r_c = K_r_c^T).
        """
        if self.contact is None:
            return np.zeros(self.r, dtype=np.float64), np.zeros(
                (self.r, self.r), dtype=np.float64
            )
        return self.contact.evaluate_reduced_forces_and_stiffness(u_r, v_r)

    def evaluate_total_residual_and_tangent(
        self,
        u_r: np.ndarray,
        v_r: np.ndarray | None = None,
        f_ext_r: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the combined out-of-balance residual and symmetric tangent matrix in R^r:
            R_r = f_ext_r + f_{r, c}(u_r) - f_{r, int}(u_r)
            K_r = K_{r, int}(u_r) + K_{r, c}(u_r)

        Returns:
        --------
        R_r : np.ndarray of shape (r,)
        K_r : np.ndarray of shape (r, r) with K_r = K_r^T.
        """
        f_int_r, K_int_r = self.evaluate_forces_and_stiffness(u_r)
        f_c_r, K_c_r = self.evaluate_contact(u_r, v_r)

        if f_ext_r is None:
            f_ext_r = self.f_r_base

        R_r = f_ext_r + f_c_r - f_int_r
        K_r = K_int_r + K_c_r

        # Enforce exact floating point symmetry
        K_r = 0.5 * (K_r + K_r.T)
        return R_r, K_r
