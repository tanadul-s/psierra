"""
Model Order Reduction Snapshot Matrix Management and SVD Basis Construction.

Implements offline displacement snapshot harvesting, persistent .npz disk caching,
thin Singular Value Decomposition (SVD), and cumulative energy spectrum truncation.

Governing Equations:
--------------------
Displacement Snapshot Matrix:
    S = [u_1, u_2, ..., u_m] in R^(n x m)

Thin Singular Value Decomposition (SVD):
    S = U Sigma V^T
    where U in R^(n x m), Sigma = diag(sigma_1, ..., sigma_m), V in R^(m x m)

Cumulative Energy Spectrum Criterion:
    E(r) = (sum_{i=1}^r sigma_i^2) / (sum_{i=1}^m sigma_i^2) >= eta  (typically eta >= 0.999)
    Phi = U[:, :r] in R^(n x r)
"""

from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
import scipy.linalg as spla
from rich import print

from psierra.reduction.basis import PODBasis


class SnapshotManager:
    """
    Manages offline displacement snapshot collection, persistent disk caching,
    and Proper Orthogonal Decomposition (POD) basis generation.

    Features:
    ---------
    - Validates array sizes to guarantee dimension conformance.
    - Native compressed disk caching (.npz) with embedded metadata.
    - Automated 'get_or_compute' caching workflow: loads cached snapshots in ~10 ms,
      or triggers offline training only when the file is absent.
    - Thin SVD with energy-criterion truncation (default >= 99.9% energy).
    """

    def __init__(self, n_dofs: int | None = None) -> None:
        """
        Parameters:
        -----------
        n_dofs : int, optional
            Expected degree-of-freedom size of the physical domain.
            If None, inferred automatically from the first collected snapshot.
        """
        self.n_dofs: int | None = int(n_dofs) if n_dofs is not None else None
        self._snapshots: list[np.ndarray] = []
        self._metadata: dict[str, Any] = {}

    @property
    def n_snapshots(self) -> int:
        """Number of collected snapshots m."""
        return len(self._snapshots)

    @property
    def metadata(self) -> dict[str, Any]:
        """Dictionary of stored simulation metadata."""
        return dict(self._metadata)

    def clear(self) -> None:
        """Clears all stored snapshots from memory."""
        self._snapshots.clear()

    def add_snapshot(self, u: np.ndarray) -> None:
        """
        Appends a 1D state displacement snapshot to the collection.

        Parameters:
        -----------
        u : np.ndarray of shape (n_dofs,)
            State displacement vector from a simulation time step or load step.
        """
        u_arr = np.asarray(u, dtype=np.float64).ravel()
        if self.n_dofs is None:
            self.n_dofs = u_arr.size
        elif u_arr.size != self.n_dofs:
            raise ValueError(
                f"Snapshot dimension {u_arr.size} does not match expected n_dofs {self.n_dofs}."
            )
        self._snapshots.append(u_arr.copy())

    def add_snapshots(self, U: np.ndarray) -> None:
        """
        Appends multiple snapshots from a 2D matrix.

        Parameters:
        -----------
        U : np.ndarray of shape (n_dofs, m)
            Matrix containing m snapshot column vectors.
        """
        U_arr = np.asarray(U, dtype=np.float64)
        if U_arr.ndim != 2:
            raise ValueError(
                f"Multi-snapshot array must be 2D of shape (n_dofs, m), got {U_arr.shape}"
            )

        n, m = U_arr.shape
        if self.n_dofs is None:
            self.n_dofs = n
        elif n != self.n_dofs:
            raise ValueError(
                f"Snapshot matrix rows ({n}) do not match expected n_dofs ({self.n_dofs})."
            )

        for col in range(m):
            self._snapshots.append(U_arr[:, col].copy())

    def get_snapshot_matrix(self) -> np.ndarray:
        """
        Assembles and returns the full snapshot matrix:
            S = [u_1, u_2, ..., u_m] in R^(n_dofs x n_snapshots)

        Returns:
        --------
        S : np.ndarray of shape (n_dofs, n_snapshots)
        """
        if not self._snapshots:
            raise ValueError(
                "Cannot construct snapshot matrix: no snapshots have been collected."
            )
        return np.column_stack(self._snapshots)

    def save(self, filepath: Path | str, **metadata) -> Path:
        """
        Saves the snapshot matrix S and simulation metadata to a compressed .npz archive.

        Parameters:
        -----------
        filepath : Path or str
            Destination file path (e.g. 'snapshots.npz').
        **metadata : Any
            Additional arbitrary metadata (e.g., dt=0.005, material='Hookean', mesh_file='beam.vtk').

        Returns:
        --------
        saved_path : Path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        S = self.get_snapshot_matrix()
        meta_dict = {**self._metadata, **metadata}

        np.savez_compressed(
            path,
            snapshots=S,
            n_dofs=np.int64(self.n_dofs if self.n_dofs is not None else S.shape[0]),
            **meta_dict,
        )
        return path

    @classmethod
    def load(cls, filepath: Path | str) -> SnapshotManager:
        """
        Loads a snapshot matrix and metadata from a compressed .npz archive.

        Parameters:
        -----------
        filepath : Path or str
            Path to the .npz archive.

        Returns:
        --------
        manager : SnapshotManager
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Snapshot archive not found at: {path}")

        archive = np.load(path, allow_pickle=True)
        if "snapshots" not in archive:
            raise KeyError(
                f"Invalid snapshot archive '{path}': missing required array 'snapshots'."
            )

        S = archive["snapshots"]
        n_dofs = int(archive["n_dofs"]) if "n_dofs" in archive else S.shape[0]

        manager = cls(n_dofs=n_dofs)
        manager.add_snapshots(S)

        # Populate metadata
        for key in archive.files:
            if key not in ("snapshots", "n_dofs"):
                manager._metadata[key] = archive[key]

        return manager

    @classmethod
    def get_or_compute(
        cls,
        cache_path: Path | str,
        compute_fn: Callable[[SnapshotManager], None],
        force_recompute: bool = False,
        verbose: bool = True,
        **save_kwargs,
    ) -> SnapshotManager:
        """
        Automatic disk caching helper:
        - If cache_path exists and force_recompute is False, loads instantly from disk.
        - Otherwise, instantiates a new SnapshotManager, invokes compute_fn(manager),
          saves the resulting snapshots to cache_path, and returns the populated manager.

        Parameters:
        -----------
        cache_path : Path or str
            File path for snapshot persistence.
        compute_fn : Callable[[SnapshotManager], None]
            Function taking the SnapshotManager and executing the FOM simulation to append snapshots.
        force_recompute : bool, default=False
            If True, ignores existing cache and forces re-execution.
        verbose : bool, default=True
            Whether to display status messages.
        **save_kwargs : Any
            Additional metadata passed to .save().

        Returns:
        --------
        manager : SnapshotManager
        """
        path = Path(cache_path)
        if path.is_file() and not force_recompute:
            if verbose:
                print(f"[bold green]✓ Loading cached snapshots from: {path}[/bold green]")
            return cls.load(path)

        if verbose:
            print(
                f"[bold yellow]⚡ No valid snapshot cache at {path}. Computing offline FOM snapshots...[/bold yellow]"
            )

        manager = cls()
        compute_fn(manager)

        if manager.n_snapshots == 0:
            raise RuntimeError(
                f"compute_fn failed to add any snapshots to SnapshotManager."
            )

        manager.save(path, **save_kwargs)
        if verbose:
            print(
                f"[bold green]✓ Offline snapshots saved to cache: {path} ({manager.n_snapshots} snapshots)[/bold green]"
            )
        return manager

    def compute_basis(
        self,
        energy_threshold: float = 0.999,
        max_modes: int | None = None,
    ) -> PODBasis:
        """
        Computes thin Singular Value Decomposition (SVD) and extracts the optimal
        orthonormal reduced basis matrix Phi satisfying the cumulative energy threshold:
            S = U Sigma V^T
            E(r) = (sum_{i=1}^r sigma_i^2) / (sum_{i=1}^m sigma_i^2) >= energy_threshold
            Phi = U[:, :r]

        Parameters:
        -----------
        energy_threshold : float, default=0.999
            Fraction of total singular value energy to capture (0.0 < energy_threshold <= 1.0).
            Default 0.999 guarantees >= 99.9% energy retention.
        max_modes : int, optional
            Hard upper limit on the number of retained modes r.

        Returns:
        --------
        basis : PODBasis
        """
        if not (0.0 < energy_threshold <= 1.0):
            raise ValueError(
                f"energy_threshold must be in (0.0, 1.0], got {energy_threshold}"
            )
        if max_modes is not None and max_modes < 1:
            raise ValueError(f"max_modes must be >= 1, got {max_modes}")

        S = self.get_snapshot_matrix()

        # Thin SVD: S in R^(n x m) -> U in R^(n x m), s in R^m, Vt in R^(m x m)
        U, s, _ = spla.svd(S, full_matrices=False)

        # Singular values variance / energy
        energies = s**2
        total_energy = float(np.sum(energies))

        if total_energy < 1e-16:
            # Degenerate zero deformation
            r = 1
            cum_energy = 1.0
        else:
            cum_energies = np.cumsum(energies) / total_energy
            # Find smallest index where cumulative energy meets or exceeds threshold
            r = int(np.searchsorted(cum_energies, energy_threshold) + 1)
            r = min(r, len(s))
            if max_modes is not None:
                r = min(r, max_modes)
            r = max(1, r)
            cum_energy = float(cum_energies[r - 1])

        Phi = U[:, :r].copy()
        return PODBasis(
            basis=Phi,
            singular_values=s[:r].copy(),
            cumulative_energy=cum_energy,
            total_energy=total_energy,
        )
