from __future__ import annotations
from pathlib import Path
from typing import Any, Literal
import h5py
import numpy as np

from psierra.domain.base import Domain
from psierra.exchange.checkpoint import SimulationCheckpoint
from psierra.exchange.xdmf import XdmfWriter
from psierra.exchange.pvd_writer import PvdWriter


class SimulationWriter:
    """
    Unified Simulation Recorder and State Persistence Manager for Project pSierra.

    Seamlessly toggles between modern XDMF+HDF5 (clean 2-file binary storage with single-mesh topology
    and zero-overhead ParaView loading) and legacy PVD+VTU (multi-file XML collections).

    Both backends share identical API signatures, full context-manager support,
    and automatic checkpoint/restart capabilities.

    Parameters:
    -----------
    output_path : Path or str
        Destination directory or file prefix (e.g. 'results/beam_dynamic').
    mesh : Any
        VolumetricMesh or SurfaceMesh representing the reference geometry.
    format : Literal["xdmf", "pvd", "h5", "vtu"], default="xdmf"
        Underlying file storage and visualization backend.
    store_checkpoints : bool, default=True
        Whether to store binary restart checkpoints for exact simulation resume.
    resume : bool, default=False
        If True, opens existing archive in append mode to continue recording.
    """

    def __init__(
        self,
        output_path: Path | str,
        mesh: Any,
        format: Literal["xdmf", "pvd", "h5", "vtu", "hdf5"] = "xdmf",
        store_checkpoints: bool = True,
        resume: bool = False,
    ) -> None:
        self.output_path = Path(output_path)
        self.format_str = str(format).lower()
        self.mesh = mesh
        self.store_checkpoints = store_checkpoints
        self.resume = resume

        if self.format_str in ("xdmf", "h5", "hdf5"):
            self.format_type = "xdmf"
            self._backend: XdmfWriter | PvdWriter = XdmfWriter(
                output_path=self.output_path,
                mesh=self.mesh,
                store_checkpoints=self.store_checkpoints,
                resume=self.resume,
            )
        elif self.format_str in ("pvd", "vtu"):
            self.format_type = "pvd"
            self._backend = PvdWriter(
                output_path=self.output_path,
                mesh=self.mesh,
                store_checkpoints=self.store_checkpoints,
                resume=self.resume,
            )
        else:
            raise ValueError(
                f"Unsupported format '{format}'. Supported formats: 'xdmf' (or 'h5') and 'pvd' (or 'vtu')."
            )

    @property
    def format(self) -> str:
        """Active storage format: 'xdmf' or 'pvd'."""
        return self.format_type

    @property
    def step_count(self) -> int:
        """Total number of time steps recorded so far."""
        if isinstance(self._backend, XdmfWriter):
            return len(self._backend.steps)
        return self._backend.step_count

    def add_timestep(
        self,
        time: float,
        domain: Domain | None = None,
        displacement: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        acceleration: np.ndarray | None = None,
        point_data: dict[str, np.ndarray] | None = None,
        cell_data: dict[str, np.ndarray] | None = None,
        step_index: int | None = None,
    ) -> None:
        """
        Records a simulation time step using the active backend strategy.

        Parameters:
        -----------
        time : float
            Current physical time in seconds.
        domain : Domain, optional
            Active physical domain providing coordinates and displacement.
        displacement : np.ndarray, optional
            Nodal displacement vector.
        velocity : np.ndarray, optional
            Nodal velocity vector.
        acceleration : np.ndarray, optional
            Nodal acceleration vector.
        point_data : dict[str, np.ndarray], optional
            Additional nodal scalar or vector quantities (e.g. von Mises stress).
        cell_data : dict[str, np.ndarray], optional
            Additional element-level quantities.
        step_index : int, optional
            Explicit time step counter.
        """
        self._backend.add_timestep(
            time=time,
            domain=domain,
            displacement=displacement,
            velocity=velocity,
            acceleration=acceleration,
            point_data=point_data,
            cell_data=cell_data,
            step_index=step_index,
        )

    def write_checkpoint(
        self,
        step: int,
        time: float,
        dt: float,
        domain: Domain | None = None,
        u: np.ndarray | None = None,
        v: np.ndarray | None = None,
        a: np.ndarray | None = None,
        history_variables: dict[str, np.ndarray] | None = None,
        f_ext: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationCheckpoint:
        """
        Persists a complete binary restart state.

        Parameters:
        -----------
        step : int
            Current time step index.
        time : float
            Current physical time in seconds.
        dt : float
            Active time increment.
        domain : Domain, optional
            If provided, automatically extracts u, v, a, and material history variables.
        """
        return self._backend.write_checkpoint(
            step=step,
            time=time,
            dt=dt,
            domain=domain,
            u=u,
            v=v,
            a=a,
            history_variables=history_variables,
            f_ext=f_ext,
            metadata=metadata,
        )

    def read_checkpoint(self, step: int | None = None) -> SimulationCheckpoint:
        """
        Reads a saved checkpoint from this writer's archive.
        """
        return self._backend.read_checkpoint(step=step)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path | str,
        step: int | None = None,
    ) -> SimulationCheckpoint:
        """
        Class method to load a simulation checkpoint directly from an HDF5 archive, XDMF file,
        or directory.

        Parameters:
        -----------
        path : Path or str
            Path to .h5 file, .xdmf file, or output directory.
        step : int, optional
            Specific step index to load. If None, loads the latest checkpoint.
        """
        p = Path(path)
        candidate_h5: Path | None = None

        if p.with_suffix(".h5").exists():
            candidate_h5 = p.with_suffix(".h5")
        elif p.is_file():
            if p.suffix.lower() in (".h5", ".hdf5"):
                candidate_h5 = p
            elif p.suffix.lower() == ".xdmf":
                candidate_h5 = p.with_suffix(".h5")
        elif p.is_dir():
            # Check for result.h5, checkpoints.h5, or any .h5
            if (p / "checkpoints.h5").exists():
                candidate_h5 = p / "checkpoints.h5"
            else:
                h5_files = list(p.glob("*.h5"))
                if h5_files:
                    candidate_h5 = h5_files[0]

        if candidate_h5 is None or not candidate_h5.exists():
            raise FileNotFoundError(f"Could not locate an HDF5 checkpoint archive at '{path}'.")

        with h5py.File(candidate_h5, "r") as h5f:
            if "checkpoints" not in h5f:
                raise KeyError(f"No 'checkpoints' group found in '{candidate_h5}'.")
            ckpt_grp = h5f["checkpoints"]
            if step is None:
                if "latest" not in ckpt_grp:
                    raise KeyError(f"No 'latest' checkpoint found in '{candidate_h5}'.")
                target = ckpt_grp["latest"]
            else:
                key = f"checkpoint_{step:06d}"
                if key not in ckpt_grp:
                    raise KeyError(f"Checkpoint for step {step} not found in '{candidate_h5}'.")
                target = ckpt_grp[key]
            return SimulationCheckpoint.from_hdf5(target)

    @classmethod
    def inspect(cls, path: Path | str) -> dict[str, Any]:
        """
        Inspects an existing simulation archive, returning step counts, available datasets,
        and checkpoint status.
        """
        p = Path(path)
        h5_path: Path | None = None

        if p.is_dir():
            if (p / "checkpoints.h5").exists():
                h5_path = p / "checkpoints.h5"
            else:
                matches = list(p.glob("*.h5"))
                if matches:
                    h5_path = matches[0]
        elif p.is_file():
            if p.suffix.lower() in (".h5", ".hdf5"):
                h5_path = p
            elif p.suffix.lower() == ".xdmf":
                h5_path = p.with_suffix(".h5")
        elif p.with_suffix(".h5").exists():
            h5_path = p.with_suffix(".h5")

        summary: dict[str, Any] = {"path": str(p)}

        # If it's a PVD directory with VTU files
        if p.is_dir():
            vtu_files = list(p.glob("*.vtu"))
            if vtu_files:
                summary["steps_recorded"] = len(vtu_files)

        if h5_path is None or not h5_path.exists():
            if "steps_recorded" in summary:
                return summary
            raise FileNotFoundError(f"Archive file or checkpoint not found at '{path}'.")

        summary["h5_path"] = str(h5_path)
        with h5py.File(h5_path, "r") as h5f:
            if "mesh" in h5f:
                summary["n_nodes"] = h5f["mesh/geometry"].shape[0]
                summary["n_elements"] = h5f["mesh/topology"].shape[0]
            if "time_series" in h5f:
                steps = sorted(
                    [k for k in h5f["time_series"].keys() if k.startswith("step_")],
                    key=lambda x: int(x.split("_")[1]),
                )
                summary["steps_recorded"] = len(steps)
                if steps:
                    summary["first_time"] = float(h5f[f"time_series/{steps[0]}"].attrs.get("time", 0.0))
                    summary["last_time"] = float(h5f[f"time_series/{steps[-1]}"].attrs.get("time", 0.0))
            if "checkpoints" in h5f:
                has_latest = "latest" in h5f["checkpoints"]
                summary["has_checkpoint"] = has_latest
                if has_latest:
                    summary["checkpoint_step"] = int(h5f["checkpoints/latest"].attrs.get("step", -1))
                    summary["checkpoint_time"] = float(h5f["checkpoints/latest"].attrs.get("time", 0.0))
        return summary

    def close(self) -> None:
        """Flushes buffers and closes file handles."""
        self._backend.close()

    def __enter__(self) -> SimulationWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
