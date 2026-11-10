from __future__ import annotations
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import pyvista as pv

from psierra.domain.base import Domain
from psierra.exchange.pvd import PvdCollectionWriter
from psierra.exchange.checkpoint import SimulationCheckpoint


class PvdWriter:
    """
    Legacy visualization writer exporting individual .vtu XML files and a master .pvd collection.
    Provides identical API and usage structure as XdmfWriter.
    """

    def __init__(
        self,
        output_path: Path | str,
        mesh: Any,
        store_checkpoints: bool = True,
        resume: bool = False,
    ) -> None:
        raw_path = Path(output_path)
        if raw_path.suffix.lower() == ".pvd":
            self.output_dir = raw_path.parent
            self.pvd_path = raw_path
        else:
            self.output_dir = raw_path
            self.pvd_path = self.output_dir / "result.pvd"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mesh = mesh
        self.store_checkpoints = store_checkpoints
        self.pvd_writer = PvdCollectionWriter(self.pvd_path)
        self.step_count: int = 0
        self.exported_files: list[Path] = []
        self._ckpt_file = self.output_dir / "checkpoints.h5"

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
        Exports a .vtu snapshot and registers it in the .pvd collection.
        """
        idx = step_index if step_index is not None else self.step_count
        vtu_filename = f"step_{idx:05d}.vtu"
        vtu_path = self.output_dir / vtu_filename

        if domain is not None:
            # Use domain save state
            domain.save_state(vtu_path)
        else:
            # Construct pyvista mesh from mesh geometry and displacement
            ugrid = self.mesh.ugrid.copy() if hasattr(self.mesh, "ugrid") else pv.UnstructuredGrid()
            if displacement is not None:
                disp_np = np.asarray(displacement, dtype=np.float64)
                if disp_np.ndim == 1:
                    disp_np = disp_np.reshape(-1, 3)
                ugrid.point_data["Displacement"] = disp_np

            if velocity is not None:
                v_np = np.asarray(velocity, dtype=np.float64)
                if v_np.ndim == 1:
                    v_np = v_np.reshape(-1, 3)
                ugrid.point_data["Velocity"] = v_np

            if acceleration is not None:
                a_np = np.asarray(acceleration, dtype=np.float64)
                if a_np.ndim == 1:
                    a_np = a_np.reshape(-1, 3)
                ugrid.point_data["Acceleration"] = a_np

            if point_data:
                for k, v in point_data.items():
                    ugrid.point_data[k] = np.asarray(v)

            if cell_data:
                for k, v in cell_data.items():
                    ugrid.cell_data[k] = np.asarray(v)

            ugrid.save(str(vtu_path))

        self.pvd_writer.add_timestep(timestep=float(time), filename=vtu_filename)
        self.pvd_writer.write()
        self.exported_files.append(vtu_path)
        self.step_count = idx + 1

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
        """Writes binary checkpoint file in output directory."""
        if domain is not None:
            checkpoint = SimulationCheckpoint.from_domain(
                domain=domain,
                step=step,
                time=time,
                dt=dt,
                f_ext=f_ext,
                metadata=metadata,
            )
        else:
            if u is None or v is None or a is None:
                raise ValueError("Must provide either 'domain' or 'u', 'v', 'a' explicitly.")
            checkpoint = SimulationCheckpoint(
                step=step,
                time=time,
                dt=dt,
                u=np.asarray(u, dtype=np.float64),
                v=np.asarray(v, dtype=np.float64),
                a=np.asarray(a, dtype=np.float64),
                history_variables=history_variables if history_variables is not None else {},
                f_ext=f_ext,
                metadata=metadata if metadata is not None else {},
            )

        with h5py.File(self._ckpt_file, "a") as h5f:
            ckpt_grp = h5f.require_group("checkpoints")
            latest_grp = ckpt_grp.require_group("latest")
            step_grp = ckpt_grp.require_group(f"checkpoint_{step:06d}")
            checkpoint.to_hdf5(latest_grp)
            checkpoint.to_hdf5(step_grp)
        return checkpoint

    def read_checkpoint(self, step: int | None = None) -> SimulationCheckpoint:
        """Reads checkpoint from checkpoint file."""
        if not self._ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found at '{self._ckpt_file}'.")
        with h5py.File(self._ckpt_file, "r") as h5f:
            ckpt_grp = h5f["checkpoints"]
            if step is None:
                target_grp = ckpt_grp["latest"]
            else:
                target_grp = ckpt_grp[f"checkpoint_{step:06d}"]
            return SimulationCheckpoint.from_hdf5(target_grp)

    def close(self) -> None:
        """Flushes the .pvd manifest."""
        self.pvd_writer.write()

    def __enter__(self) -> PvdWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
