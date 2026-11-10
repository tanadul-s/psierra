from __future__ import annotations
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import pyvista as pv

from psierra.domain.base import Domain
from psierra.exchange.checkpoint import SimulationCheckpoint


class PvdWriter:
    """
    Unified ParaView Data (.pvd) collection and VTU snapshot writer.
    Maintains an XML dataset collection linking individual time-step VTU frames
    and optional binary HDF5 restart checkpoints.

    Provides identical API and usage structure as XdmfWriter.
    """

    def __init__(
        self,
        output_path: Path | str,
        mesh: Any | None = None,
        store_checkpoints: bool = True,
        resume: bool = False,
    ) -> None:
        raw_path = Path(output_path)
        if raw_path.suffix.lower() == ".pvd":
            self.output_dir = raw_path.parent
            self.pvd_path = raw_path
        elif raw_path.name.lower() == "result":
            self.output_dir = raw_path.parent
            self.pvd_path = self.output_dir / "result.pvd"
        else:
            self.output_dir = raw_path
            self.pvd_path = self.output_dir / "result.pvd"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mesh = mesh
        self.store_checkpoints = store_checkpoints
        self.step_count: int = 0
        self.exported_files: list[Path] = []
        self._ckpt_file = self.output_dir / "checkpoints.h5"
        self.entries: list[dict[str, str | float | int | None]] = []

        if resume and self.pvd_path.exists():
            self._load_existing_pvd()

    def _load_existing_pvd(self) -> None:
        """Parses existing .pvd XML to populate entries and step_count when resuming."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(self.pvd_path)
            root = tree.getroot()
            for ds in root.findall(".//DataSet"):
                ts = ds.get("timestep")
                fn = ds.get("file")
                part = ds.get("part", "0")
                group = ds.get("group", "")
                if fn:
                    self.entries.append(
                        {
                            "timestep": float(ts) if ts is not None else None,
                            "filename": fn,
                            "part": int(part) if part is not None else 0,
                            "group": group,
                        }
                    )
                    vtu_path = self.output_dir / fn
                    if vtu_path.exists() and vtu_path not in self.exported_files:
                        self.exported_files.append(vtu_path)
            self.step_count = len(self.entries)
        except Exception:
            pass

    def add_dataset(
        self,
        filename: str,
        timestep: float | None = None,
        part: int = 0,
        group: str = "",
    ) -> None:
        """Appends a dataset (e.g. multi-domain component) to the collection."""
        self.entries.append(
            {
                "timestep": timestep,
                "filename": filename,
                "part": part,
                "group": group,
            }
        )
        self.write()

    def add_timestep(
        self,
        time: float | None = None,
        filename: str | None = None,
        domain: Domain | None = None,
        displacement: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        acceleration: np.ndarray | None = None,
        point_data: dict[str, np.ndarray] | None = None,
        cell_data: dict[str, np.ndarray] | None = None,
        step_index: int | None = None,
        part: int = 0,
        group: str = "",
        timestep: float | None = None,
    ) -> None:
        """
        Exports a .vtu snapshot and registers it in the .pvd collection.
        Supports both modern (time, domain=...) and legacy (timestep, filename=...) signatures.
        """
        t_val = time if time is not None else (timestep if timestep is not None else 0.0)
        idx = step_index if step_index is not None else self.step_count
        if filename is not None:
            vtu_filename = filename
            vtu_path = self.output_dir / vtu_filename
        else:
            vtu_filename = f"step_{idx:05d}.vtu"
            vtu_path = self.output_dir / vtu_filename

        # If a domain or field arrays are provided, export the .vtu snapshot file
        if domain is not None:
            if hasattr(domain, "to_pyvista"):
                ugrid = domain.to_pyvista(
                    factor=0.0,
                    include_stresses=False,
                    smooth_to_nodes=False,
                    include_cell_data=cell_data,
                )
            elif hasattr(domain, "save_state"):
                # Fallback for mock domains (e.g. SpringDomain in unit tests)
                domain.save_state(vtu_path)
                ugrid = None
            else:
                mesh_src = self.mesh or getattr(domain, "mesh", None)
                ugrid = mesh_src.ugrid.copy() if (mesh_src and hasattr(mesh_src, "ugrid")) else pv.UnstructuredGrid()

            if ugrid is not None:
                if displacement is not None:
                    ugrid.point_data["Displacement"] = np.asarray(displacement, dtype=np.float64).reshape(-1, 3)

                # Velocity: attach if passed explicitly or if dynamic fields are present on domain
                if velocity is not None:
                    ugrid.point_data["Velocity"] = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
                elif hasattr(domain, "v") and (
                    np.any(domain.v != 0.0) or (hasattr(domain, "a") and np.any(domain.a != 0.0))
                ):
                    ugrid.point_data["Velocity"] = np.asarray(domain.v, dtype=np.float64).reshape(-1, 3)

                # Acceleration: attach if passed explicitly or if dynamic fields are present on domain
                if acceleration is not None:
                    ugrid.point_data["Acceleration"] = np.asarray(acceleration, dtype=np.float64).reshape(-1, 3)
                elif hasattr(domain, "a") and (
                    np.any(domain.v != 0.0) or np.any(domain.a != 0.0)
                ):
                    ugrid.point_data["Acceleration"] = np.asarray(domain.a, dtype=np.float64).reshape(-1, 3)

                if point_data:
                    for k, v in point_data.items():
                        ugrid.point_data[k] = np.asarray(v)

                if cell_data and not hasattr(domain, "to_pyvista"):
                    for k, v in cell_data.items():
                        ugrid.cell_data[k] = np.asarray(v)

                ugrid.save(str(vtu_path))

        elif displacement is not None or velocity is not None or acceleration is not None:
            # Construct PyVista mesh from reference mesh geometry and displacement
            mesh_src = self.mesh
            ugrid = mesh_src.ugrid.copy() if (mesh_src and hasattr(mesh_src, "ugrid")) else pv.UnstructuredGrid()
            if displacement is not None:
                disp_np = np.asarray(displacement, dtype=np.float64).reshape(-1, 3)
                ugrid.point_data["Displacement"] = disp_np
            if velocity is not None:
                v_np = np.asarray(velocity, dtype=np.float64).reshape(-1, 3)
                ugrid.point_data["Velocity"] = v_np
            if acceleration is not None:
                a_np = np.asarray(acceleration, dtype=np.float64).reshape(-1, 3)
                ugrid.point_data["Acceleration"] = a_np
            if point_data:
                for k, v in point_data.items():
                    ugrid.point_data[k] = np.asarray(v)
            if cell_data:
                for k, v in cell_data.items():
                    ugrid.cell_data[k] = np.asarray(v)
            ugrid.save(str(vtu_path))

        # Register in .pvd collection
        self.entries.append(
            {
                "timestep": float(t_val),
                "filename": vtu_filename,
                "part": part,
                "group": group,
            }
        )
        self.write()
        if vtu_path not in self.exported_files:
            self.exported_files.append(vtu_path)
        self.step_count = idx + 1

    def write(self) -> None:
        """Writes the XML collection to the target .pvd file."""
        self.pvd_path.parent.mkdir(parents=True, exist_ok=True)
        xml_content = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            "  <Collection>",
        ]
        for entry in self.entries:
            ts = entry["timestep"]
            ts_str = f' timestep="{ts:.6e}"' if ts is not None else ""
            group = entry["group"]
            group_str = f' group="{group}"' if group else ""
            part_str = f' part="{entry["part"]}"'
            file_str = f' file="{entry["filename"]}"'
            xml_content.append(f"    <DataSet{ts_str}{group_str}{part_str}{file_str}/>")

        xml_content.append("  </Collection>")
        xml_content.append("</VTKFile>")
        self.pvd_path.write_text("\n".join(xml_content))

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
        self.write()

    def __enter__(self) -> PvdWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# Backward compatibility alias
PvdCollectionWriter = PvdWriter
