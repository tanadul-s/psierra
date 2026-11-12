"""
Energy-Conserving Sampling and Weighting (ECSW) Hyper-Reduction.

Implements the element-based empirical cubature method of Farhat et al. (2014)
for nonlinear finite element model order reduction:
1. Constructs the virtual work snapshot training matrix A in R^{(m*r) x N_e} and target vector b in R^{m*r}.
2. Solves a sparse Non-Negative Least Squares (NNLS) optimization problem to select
   a small subset of elements (the Reduced Integration Domain, RID) with strictly positive weights w_e > 0.
3. Preserves exact Lagrangian structure, tangent stiffness symmetry, and numerical energy conservation.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import time
import numpy as np
import scipy.optimize as spopt

from psierra.domain.base import Domain
from psierra.reduction.basis import PODBasis


@dataclass
class ReducedIntegrationDomain:
    """
    Container for the ECSW Reduced Integration Domain (RID).

    Attributes:
    -----------
    element_indices : np.ndarray
        Global element indices selected in the reduced mesh (p,).
    weights : np.ndarray
        Strictly positive cubature weights w_e > 0 (p,).
    n_total_elements : int
        Total number of elements in the full mesh (N_e).
    relative_error : float
        Relative virtual work residual error achieved during NNLS training.
    metadata : dict
        Additional training metadata (e.g. tolerance, snapshot count, date).
    """

    element_indices: np.ndarray
    weights: np.ndarray
    n_total_elements: int
    relative_error: float
    metadata: dict | None = None

    def __post_init__(self) -> None:
        self.element_indices = np.asarray(self.element_indices, dtype=np.int64).ravel()
        self.weights = np.asarray(self.weights, dtype=np.float64).ravel()
        self.n_total_elements = int(self.n_total_elements)
        self.relative_error = float(self.relative_error)
        if self.metadata is None:
            self.metadata = {}

        if self.element_indices.size != self.weights.size:
            raise ValueError(
                f"Element indices length ({self.element_indices.size}) must match weights length ({self.weights.size})."
            )
        if np.any(self.weights <= 0.0):
            raise ValueError("All ECSW element weights must be strictly positive (w_e > 0).")
        if self.n_total_elements <= 0:
            raise ValueError(f"n_total_elements must be positive, got {self.n_total_elements}.")

    @property
    def n_selected(self) -> int:
        """Number of retained elements in the reduced integration domain (p = |RID|)."""
        return int(self.element_indices.size)

    @property
    def compression_ratio(self) -> float:
        """Fraction of total mesh elements retained: p / N_e in (0, 1]."""
        return float(self.n_selected / self.n_total_elements)

    @property
    def element_reduction_pct(self) -> float:
        """Percentage of mesh elements eliminated: (1 - p / N_e) * 100%."""
        return float((1.0 - self.compression_ratio) * 100.0)

    @property
    def speedup_factor(self) -> float:
        """Theoretical elemental evaluation speedup: N_e / p."""
        return float(self.n_total_elements / max(self.n_selected, 1))

    def save(self, path: Path | str, **extra_metadata) -> Path:
        """
        Saves the reduced integration domain to a compressed NumPy (.npz) archive.

        Parameters:
        -----------
        path : Path or str
            Destination archive path.
        extra_metadata : kwargs
            Additional user metadata key-value pairs.

        Returns:
        --------
        saved_path : Path
        """
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
            metadata=meta,
        )
        return out_path

    @classmethod
    def load(cls, path: Path | str) -> ReducedIntegrationDomain:
        """
        Loads a ReducedIntegrationDomain from a compressed .npz archive.

        Parameters:
        -----------
        path : Path or str
            Source archive path.

        Returns:
        --------
        rid : ReducedIntegrationDomain
        """
        src_path = Path(path).resolve()
        if not src_path.is_file():
            raise FileNotFoundError(f"ECSW RID archive not found: {src_path}")

        with np.load(src_path, allow_pickle=True) as data:
            element_indices = data["element_indices"]
            weights = data["weights"]
            n_total_elements = int(data["n_total_elements"][0])
            relative_error = float(data["relative_error"][0])
            metadata = data["metadata"].item() if "metadata" in data else {}

        return cls(
            element_indices=element_indices,
            weights=weights,
            n_total_elements=n_total_elements,
            relative_error=relative_error,
            metadata=metadata,
        )


class ECSWTrainingMatrix:
    """
    Constructs the virtual work training matrix for ECSW hyper-reduction.

    Mathematical Formulation:
    -------------------------
    For each snapshot displacement vector u^{(k)} (k = 1, ..., m) and each element e = 0, ..., N_e - 1:
        c_e^{(k)} = Phi_e^T * f_{int, e}(u_e^{(k)}) in R^r
    where:
        Phi_e = Phi[elem_dofs_e, :] in R^{n_e x r}
        f_{int, e} is the elemental internal force vector in R^{n_e}

    Column e of the training matrix A is formed by stacking c_e across all m snapshots:
        a_e = [ c_e^{(1)T}, c_e^{(2)T}, ..., c_e^{(m)T} ]^T in R^{m * r}

    The training matrix A in R^{(m * r) x N_e} and target exact virtual work vector b in R^{m * r}:
        b = sum_{e=0}^{N_e - 1} a_e = A * 1_{N_e}
    """

    @staticmethod
    def build(
        domain: Domain,
        basis: PODBasis,
        snapshots: np.ndarray | Sequence[np.ndarray],
        free_dofs: np.ndarray | None = None,
        include_stiffness: bool = False,
        stiffness_weight: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs the ECSW virtual work training matrix A and target vector b.

        Parameters:
        -----------
        domain : Domain
            Physical continuum domain (SolidDomain).
        basis : PODBasis
            Reduced modal basis Phi in R^{n x r}.
        snapshots : np.ndarray of shape (n_dofs, m) or list of (n_dofs,) vectors
            Displacement snapshot field vectors.
        free_dofs : np.ndarray, optional
            Free unconstrained degrees of freedom. If None, assumes full domain DOFs.
        include_stiffness : bool, default=False
            If True, augments the training matrix with the upper-triangular entries of
            the linear stiffness projection Phi_e^T * K_{0, e} * Phi_e (Farhat et al. 2014,
            Section 3.2). This guarantees rank sufficiency and physical positive-definiteness
            of the reduced tangent stiffness matrix K_{r, t}(u_r).
        stiffness_weight : float, optional
            Multiplier for stiffness augmentation block scaling. If None, scales stiffness
            block automatically to match the average row norm of the virtual work force block.

        Returns:
        --------
        A : np.ndarray of shape (m * r, N_e) or (m * r + r*(r+1)/2, N_e)
            Elemental virtual work (and optional stiffness) training matrix.
        b : np.ndarray of shape (A.shape[0],)
            Target projected full-order quantity vector across all snapshots.
        """
        # Format snapshot matrix: (n_dofs, m)
        if isinstance(snapshots, np.ndarray):
            if snapshots.ndim == 2:
                snap_mat = snapshots
            elif snapshots.ndim == 1:
                snap_mat = snapshots.reshape(-1, 1)
            else:
                raise ValueError(f"Snapshot array must be 1D or 2D, got shape {snapshots.shape}")
        elif isinstance(snapshots, (list, tuple)):
            if len(snapshots) == 0:
                raise ValueError("Snapshots list cannot be empty.")
            snap_mat = np.column_stack(snapshots)
        else:
            raise ValueError(
                f"Snapshots must be a 2D array or sequence of 1D vectors, got type {type(snapshots)}"
            )

        n_dofs, m_snapshots = snap_mat.shape
        if n_dofs != domain.n_dofs:
            raise ValueError(
                f"Snapshot DOF count ({n_dofs}) does not match domain DOF count ({domain.n_dofs})."
            )

        r = basis.n_modes
        Phi = basis.basis_matrix  # (n_dofs, r)
        is_coupled = hasattr(domain, "solid_domain") and hasattr(domain, "shell_domain")

        if is_coupled:
            n_solid_elems = domain.solid_domain.mesh.n_elements
            n_shell_elems = domain.shell_domain.mesh.n_elements
            n_elems = n_solid_elems + n_shell_elems
            solid_nodes = domain.solid_domain.mesh.nodes
            solid_elems = domain.solid_domain.mesh.elements
            solid_form = domain.solid_domain.element

            shell_elems = domain.shell_domain.mesh.elements
            shell_form = domain.shell_domain.element
            shell_coupled_dofs = domain._shell_coupled_dofs
            shell_node_indices = domain.shell_domain.mesh.node_indices

            coords_list = []
            form_list = []
            dpn_list = []
            dofs_list = []
            Phi_e_list = []

            for e in range(n_solid_elems):
                enodes = solid_elems[e]
                coords_list.append(solid_nodes[enodes])
                form_list.append(solid_form)
                dpn_list.append(3)
                edofs = np.concatenate([np.arange(3 * nid, 3 * nid + 3) for nid in enodes])
                dofs_list.append(edofs)
                Phi_e_list.append(Phi[edofs, :])

            for s in range(n_shell_elems):
                s_conn = shell_elems[s]
                g_conn = shell_node_indices[s_conn]
                coords_list.append(solid_nodes[g_conn])
                form_list.append(shell_form)
                dpn_list.append(6)
                edofs = shell_coupled_dofs[s]
                dofs_list.append(edofs)
                Phi_e_list.append(Phi[edofs, :])

            M_rows = m_snapshots * r
            A = np.zeros((M_rows, n_elems), dtype=np.float64)

            for k in range(m_snapshots):
                u_k = snap_mat[:, k]
                row_start = k * r
                row_end = row_start + r

                for e in range(n_elems):
                    coords = coords_list[e]
                    form = form_list[e]
                    dpn = dpn_list[e]
                    edofs = dofs_list[e]
                    u_elem = u_k[edofs].reshape(-1, dpn)

                    _, f_int_e = form.compute_nonlinear_force_and_tangent_stiffness(coords, u_elem)
                    f_int_e = f_int_e.ravel()

                    c_e = Phi_e_list[e].T @ f_int_e
                    A[row_start:row_end, e] = c_e

            if include_stiffness:
                triu_idx = np.triu_indices(r)
                n_triu = len(triu_idx[0])
                A_stiff = np.zeros((n_triu, n_elems), dtype=np.float64)

                for e in range(n_elems):
                    coords = coords_list[e]
                    form = form_list[e]
                    dpn = dpn_list[e]
                    u_zero = np.zeros((coords.shape[0], dpn), dtype=np.float64)
                    K_0_e, _ = form.compute_nonlinear_force_and_tangent_stiffness(coords, u_zero)
                    Phi_e = Phi_e_list[e]
                    K_r_e = Phi_e.T @ K_0_e @ Phi_e
                    A_stiff[:, e] = K_r_e[triu_idx]

                s_force = float(np.linalg.norm(A)) / np.sqrt(max(A.shape[0], 1))
                s_stiff = float(np.linalg.norm(A_stiff)) / np.sqrt(max(n_triu, 1))
                if s_force > 1e-12 and s_stiff > 1e-12:
                    weight = float(stiffness_weight) if stiffness_weight is not None else 1.0
                    scale = (s_force / s_stiff) * weight
                    A_stiff *= scale

                A = np.vstack([A, A_stiff])

            b = A.sum(axis=1)
            return A, b

        n_elems = domain.mesh.n_elements
        nodes = domain.mesh.nodes
        elements = domain.mesh.elements
        elem_form = domain.element

        dofs_per_node = elem_form.dofs_per_node

        M_rows = m_snapshots * r
        A = np.zeros((M_rows, n_elems), dtype=np.float64)

        # Precompute elemental DOFs and elemental modal sub-bases Phi_e = Phi[elem_dofs, :]
        elem_dofs_list = []
        Phi_e_list = []
        for e in range(n_elems):
            elem_nodes = elements[e]
            # Concatenate node DOFs
            edofs = np.concatenate(
                [np.arange(dofs_per_node * nid, dofs_per_node * nid + dofs_per_node) for nid in elem_nodes]
            )
            elem_dofs_list.append(edofs)
            Phi_e_list.append(Phi[edofs, :])  # shape: (n_elem_dofs, r)

        # Evaluate elemental virtual work for each snapshot
        for k in range(m_snapshots):
            u_k = snap_mat[:, k]
            u_k_nodes = u_k.reshape(-1, dofs_per_node)
            row_start = k * r
            row_end = row_start + r

            for e in range(n_elems):
                elem_nodes = elements[e]
                coords = nodes[elem_nodes]
                u_elem = u_k_nodes[elem_nodes]

                # Compute elemental internal force
                _, f_int_e = elem_form.compute_nonlinear_force_and_tangent_stiffness(coords, u_elem)
                f_int_e = f_int_e.ravel()

                # Elemental contribution: c_e = Phi_e^T * f_int_e in R^r
                Phi_e = Phi_e_list[e]
                c_e = Phi_e.T @ f_int_e

                A[row_start:row_end, e] = c_e

        # Optional linear stiffness matrix augmentation (Farhat et al. 2014, Sec 3.2)
        if include_stiffness:
            triu_idx = np.triu_indices(r)
            n_triu = len(triu_idx[0])
            A_stiff = np.zeros((n_triu, n_elems), dtype=np.float64)
            u_zero = np.zeros((elem_form.nodes_per_element, elem_form.dofs_per_node), dtype=np.float64)

            for e in range(n_elems):
                coords = nodes[elements[e]]
                K_0_e, _ = elem_form.compute_nonlinear_force_and_tangent_stiffness(coords, u_zero)
                Phi_e = Phi_e_list[e]
                K_r_e = Phi_e.T @ K_0_e @ Phi_e
                A_stiff[:, e] = K_r_e[triu_idx]

            # Scale stiffness block to balance virtual work force block
            s_force = float(np.linalg.norm(A)) / np.sqrt(max(A.shape[0], 1))
            s_stiff = float(np.linalg.norm(A_stiff)) / np.sqrt(max(n_triu, 1))
            if s_force > 1e-12 and s_stiff > 1e-12:
                weight = float(stiffness_weight) if stiffness_weight is not None else 1.0
                scale = (s_force / s_stiff) * weight
                A_stiff *= scale

            A = np.vstack([A, A_stiff])

        # Target vector b = A * 1_{N_e} = sum_{e=0}^{N_e - 1} a_e
        b = A.sum(axis=1)
        return A, b


class ECSWOptimizer:
    """
    Greedy Active-Set Non-Negative Least Squares (NNLS) Optimizer for ECSW.

    Solves the sparse cubature optimization problem:
        min_{w >= 0} ||A * w - b||_2^2   subject to   ||w||_0 is small

    Algorithm (Farhat et al., 2014):
    --------------------------------
    1. Initialize active element set tilde_E = empty, residual r = b.
    2. While ||r|| / ||b|| > tolerance and |tilde_E| < max_elements:
       a. Score candidate elements: s_e = (a_e^T * r) / (||a_e||_2 + eps).
       b. Select best element e* = argmax_{e not in tilde_E} s_e.
       c. Update active set: tilde_E = tilde_E union {e*}.
       d. Solve NNLS on the active sub-matrix:
              w_{tilde_E} = argmin_{w >= 0} ||A_{:, tilde_E} * w - b||_2^2
       e. Prune elements whose weight vanished (w_e <= eps).
       f. Update residual: r = b - A_{:, tilde_E} * w_{tilde_E}.
    3. Return optimal ReducedIntegrationDomain containing tilde_E and w > 0.
    """

    def __init__(
        self,
        tolerance: float = 0.01,
        max_elements: int | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Parameters:
        -----------
        tolerance : float, default=0.01
            Relative target residual error tau = ||r||_2 / ||b||_2 (0.01 = 99% accuracy).
        max_elements : int, optional
            Maximum allowed number of elements in the reduced integration domain.
        verbose : bool, default=True
            Print progress during optimization iterations.
        """
        if tolerance <= 0.0:
            raise ValueError(f"tolerance must be strictly positive, got {tolerance}")
        self.tolerance = float(tolerance)
        self.max_elements = int(max_elements) if max_elements is not None else None
        self.verbose = bool(verbose)

    def optimize(
        self,
        A: np.ndarray,
        b: np.ndarray | None = None,
        tolerance: float | None = None,
        max_elements: int | None = None,
    ) -> ReducedIntegrationDomain:
        """
        Executes the greedy active-set NNLS optimization.

        Parameters:
        -----------
        A : np.ndarray of shape (M, N_e)
            Elemental virtual work training matrix.
        b : np.ndarray of shape (M,), optional
            Target vector. If None, computes b = A.sum(axis=1).
        tolerance : float, optional
            Overrides default residual tolerance.
        max_elements : int, optional
            Overrides default maximum elements limit.

        Returns:
        --------
        rid : ReducedIntegrationDomain
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

        # Precompute column L2 norms for cosine scoring
        col_norms = np.linalg.norm(A, axis=0)
        col_norms_safe = np.where(col_norms > 1e-12, col_norms, 1.0)

        active_indices: list[int] = []
        weights: np.ndarray = np.array([], dtype=np.float64)
        residual = b.copy()
        rel_error = 1.0

        if self.verbose:
            print(
                f"[ECSW] Starting greedy NNLS cubature: {n_elems:,} candidate elements, "
                f"target error <= {tol*100:.2f}%, max elements = {p_max}"
            )

        for iteration in range(1, p_max + 1):
            # 1. Candidate scoring via normalized residual correlation
            # For each element e not in active set: score = (a_e . r) / ||a_e||
            in_active_mask = np.zeros(n_elems, dtype=bool)
            if active_indices:
                in_active_mask[active_indices] = True

            proj_scores = (A.T @ residual) / col_norms_safe
            proj_scores[in_active_mask] = -np.inf

            best_elem = int(np.argmax(proj_scores))
            if np.isneginf(proj_scores[best_elem]):
                break

            active_indices.append(best_elem)

            # 2. Solve constrained NNLS on current active sub-matrix
            A_sub = A[:, active_indices]
            w_sub, _ = spopt.nnls(A_sub, b)

            # 3. Prune elements whose weight is zero
            max_w = np.max(w_sub) if w_sub.size > 0 else 0.0
            cutoff = 1e-10 * max(max_w, 1.0)
            nonzero_pos = np.where(w_sub > cutoff)[0]

            active_indices = [active_indices[i] for i in nonzero_pos]
            weights = w_sub[nonzero_pos]

            # 4. Update residual
            residual = b - A[:, active_indices] @ weights
            rel_error = float(np.linalg.norm(residual) / norm_b)

            if self.verbose and (iteration % 5 == 0 or rel_error <= tol or iteration == p_max):
                print(
                    f"  • Iteration {iteration:3d}: {len(active_indices):3d} elements retained | "
                    f"Relative error = {rel_error*100:.3f}%"
                )

            if rel_error <= tol:
                break

        t_elapsed = time.perf_counter() - t_start

        if self.verbose:
            print(
                f"[ECSW] ✓ Converged in {t_elapsed:.2f}s: {len(active_indices):,} / {n_elems:,} elements "
                f"({len(active_indices)/n_elems*100:.2f}%) with relative error = {rel_error*100:.3f}%"
            )

        return ReducedIntegrationDomain(
            element_indices=np.array(active_indices, dtype=np.int64),
            weights=weights,
            n_total_elements=n_elems,
            relative_error=rel_error,
            metadata={
                "tolerance": tol,
                "n_snapshots": M_rows // A.shape[0] if M_rows > 0 else 0,
                "elapsed_seconds": t_elapsed,
            },
        )

    @classmethod
    def get_or_compute(
        cls,
        cache_path: Path | str,
        compute_fn: Callable[[], tuple[np.ndarray, np.ndarray]],
        tolerance: float = 0.01,
        max_elements: int | None = None,
        force_recompute: bool = False,
        verbose: bool = True,
        **metadata,
    ) -> ReducedIntegrationDomain:
        """
        Retrieves a cached ReducedIntegrationDomain archive from disk, or computes
        it via greedy NNLS optimization and saves it to cache.

        Parameters:
        -----------
        cache_path : Path or str
            Destination .npz file path.
        compute_fn : Callable[[], tuple[np.ndarray, np.ndarray]]
            Callback returning (A, b) training matrix and target vector.
        tolerance : float, default=0.01
            Desired relative error threshold.
        max_elements : int, optional
            Maximum allowed elements in the reduced domain.
        force_recompute : bool, default=False
            If True, recomputes optimization even if cache exists.
        verbose : bool, default=True
            Print status messages.
        metadata : kwargs
            Additional metadata tags stored with the archive.

        Returns:
        --------
        rid : ReducedIntegrationDomain
        """
        cache = Path(cache_path).resolve()

        if cache.is_file() and not force_recompute:
            try:
                rid = ReducedIntegrationDomain.load(cache)
                if verbose:
                    print(
                        f"[ECSW] ✓ Loaded cached Reduced Integration Domain from {cache} "
                        f"({rid.n_selected:,} / {rid.n_total_elements:,} elements, error: {rid.relative_error*100:.2f}%)"
                    )
                return rid
            except Exception as e:
                if verbose:
                    print(f"[ECSW] ⚠️ Cache load error ({e}). Recomputing ECSW optimization...")

        if verbose:
            print(f"[ECSW] Assembling training matrix and computing optimal cubature weights...")

        A, b = compute_fn()
        optimizer = cls(tolerance=tolerance, max_elements=max_elements, verbose=verbose)
        rid = optimizer.optimize(A=A, b=b)
        rid.save(cache, **metadata)
        if verbose:
            print(f"[ECSW] ✓ Saved Reduced Integration Domain to cache: {cache}")

        return rid
