from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import h5py
import numpy as np

from psierra.domain.base import Domain
from psierra.exchange.checkpoint import SimulationCheckpoint


@dataclass
class XdmfTimeStepRecord:
    """Internal registry of a written time step."""

    step_index: int
    time: float
    h5_step_path: str
    attributes: list[dict[str, str]] = field(default_factory=list)


class XdmfWriter:
    """
    High-performance, single-file simulation archiver and XDMF 3.0 visualization writer.

    Stores reference geometry and element topology exactly once in HDF5,
    streams time-series kinematic and stress fields into chunked datasets,
    and maintains an XDMF temporal collection for direct, zero-overhead ParaView visualization.
    """

    def __init__(
        self,
        output_path: Path | str,
        mesh: Any,
        store_checkpoints: bool = True,
        resume: bool = False,
    ) -> None:
        raw_path = Path(output_path)
        if raw_path.suffix.lower() in (".xdmf", ".h5", ".hdf5"):
            self.base_path = raw_path.with_suffix("")
        else:
            self.base_path = raw_path

        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.xdmf_path = self.base_path.with_suffix(".xdmf")
        self.h5_path = self.base_path.with_suffix(".h5")
        self.mesh = mesh
        self.store_checkpoints = store_checkpoints

        # Determine topology type from mesh
        n_elem_nodes = int(mesh.elements.shape[1])
        if n_elem_nodes == 4:
            self.topology_type = "Tetrahedron"
        elif n_elem_nodes == 3:
            self.topology_type = "Triangle"
        elif n_elem_nodes == 8:
            self.topology_type = "Hexahedron"
        elif n_elem_nodes == 10:
            self.topology_type = "Tetrahedron_10"
        else:
            self.topology_type = "Tetrahedron"

        self.n_nodes = int(mesh.n_nodes)
        self.n_elements = int(mesh.n_elements)

        self.steps: list[XdmfTimeStepRecord] = []
        mode = "a" if resume and self.h5_path.exists() else "w"

        self.h5_file = h5py.File(self.h5_path, mode)
        self._init_hdf5_structure(resume=resume)

    def _init_hdf5_structure(self, resume: bool) -> None:
        """Initializes or resumes HDF5 groups and datasets."""
        # Reference Mesh group (stored once)
        mesh_grp = self.h5_file.require_group("mesh")
        if "geometry" not in mesh_grp:
            mesh_grp.create_dataset(
                "geometry",
                data=np.asarray(self.mesh.nodes, dtype=np.float64),
                compression="gzip",
            )
        if "topology" not in mesh_grp:
            mesh_grp.create_dataset(
                "topology",
                data=np.asarray(self.mesh.elements, dtype=np.int32),
                compression="gzip",
            )

        self.h5_file.require_group("time_series")
        self.h5_file.require_group("checkpoints")

        if resume:
            # Reconstruct step records from existing time series
            ts_grp = self.h5_file["time_series"]
            existing_steps = sorted(
                [k for k in ts_grp.keys() if k.startswith("step_")],
                key=lambda x: int(x.split("_")[1]),
            )
            for step_key in existing_steps:
                s_grp = ts_grp[step_key]
                s_idx = int(step_key.split("_")[1])
                t_val = float(s_grp.attrs["time"])
                attrs: list[dict[str, str]] = []
                for dname in s_grp.keys():
                    ds = s_grp[dname]
                    center = ds.attrs.get("center", "Node")
                    attr_type = ds.attrs.get("attr_type", "Scalar")
                    attrs.append(
                        {
                            "name": dname.capitalize(),
                            "dataset_name": dname,
                            "center": center,
                            "type": attr_type,
                            "dims": " ".join(str(d) for d in ds.shape),
                        }
                    )
                self.steps.append(
                    XdmfTimeStepRecord(
                        step_index=s_idx,
                        time=t_val,
                        h5_step_path=f"/time_series/{step_key}",
                        attributes=attrs,
                    )
                )

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
        Records a single simulation time step into the HDF5 archive and updates the XDMF manifest.

        Parameters:
        -----------
        time : float
            Simulation time in seconds.
        domain : Domain, optional
            Physical domain providing kinematic states and stresses.
        displacement : np.ndarray, optional
            Nodal displacement vector of shape (N_nodes, 3) or (N_dofs,).
        velocity : np.ndarray, optional
            Nodal velocity vector.
        acceleration : np.ndarray, optional
            Nodal acceleration vector.
        point_data : dict[str, np.ndarray], optional
            Additional nodal scalar or vector arrays.
        cell_data : dict[str, np.ndarray], optional
            Additional element-level scalar or tensor arrays.
        step_index : int, optional
            Explicit step counter (defaults to next sequential index).
        """
        if step_index is not None:
            idx = int(step_index)
        else:
            idx = len(self.steps)

        step_key = f"step_{idx:06d}"
        ts_grp = self.h5_file["time_series"].require_group(step_key)
        ts_grp.attrs["time"] = float(time)
        ts_grp.attrs["step"] = idx

        step_attributes: list[dict[str, str]] = []

        # 1. Displacement
        disp_arr: np.ndarray | None = None
        if displacement is not None:
            disp_arr = np.asarray(displacement, dtype=np.float64)
        elif domain is not None:
            disp_arr = np.asarray(domain.displacement, dtype=np.float64)

        if disp_arr is not None:
            if disp_arr.ndim == 1:
                disp_arr = disp_arr.reshape(-1, 3)
            self._write_field(ts_grp, "displacement", disp_arr, "Node", "Vector")
            step_attributes.append(
                {
                    "name": "Displacement",
                    "dataset_name": "displacement",
                    "center": "Node",
                    "type": "Vector",
                    "dims": f"{disp_arr.shape[0]} {disp_arr.shape[1]}",
                }
            )

        # 2. Velocity
        vel_arr: np.ndarray | None = None
        if velocity is not None:
            vel_arr = np.asarray(velocity, dtype=np.float64)
        elif domain is not None and hasattr(domain, "v") and (
            np.any(domain.v != 0.0) or (hasattr(domain, "a") and np.any(domain.a != 0.0))
        ):
            vel_arr = np.asarray(domain.v, dtype=np.float64).reshape(-1, 3)

        if vel_arr is not None:
            if vel_arr.ndim == 1:
                vel_arr = vel_arr.reshape(-1, 3)
            self._write_field(ts_grp, "velocity", vel_arr, "Node", "Vector")
            step_attributes.append(
                {
                    "name": "Velocity",
                    "dataset_name": "velocity",
                    "center": "Node",
                    "type": "Vector",
                    "dims": f"{vel_arr.shape[0]} {vel_arr.shape[1]}",
                }
            )

        # 3. Acceleration
        acc_arr: np.ndarray | None = None
        if acceleration is not None:
            acc_arr = np.asarray(acceleration, dtype=np.float64)
        elif domain is not None and hasattr(domain, "a") and (
            np.any(domain.v != 0.0) or np.any(domain.a != 0.0)
        ):
            acc_arr = np.asarray(domain.a, dtype=np.float64).reshape(-1, 3)

        if acc_arr is not None:
            if acc_arr.ndim == 1:
                acc_arr = acc_arr.reshape(-1, 3)
            self._write_field(ts_grp, "acceleration", acc_arr, "Node", "Vector")
            step_attributes.append(
                {
                    "name": "Acceleration",
                    "dataset_name": "acceleration",
                    "center": "Node",
                    "type": "Vector",
                    "dims": f"{acc_arr.shape[0]} {acc_arr.shape[1]}",
                }
            )

        # 4. Custom Point Data
        if point_data:
            for name, arr in point_data.items():
                arr_np = np.asarray(arr, dtype=np.float64)
                attr_type = "Vector" if arr_np.ndim > 1 and arr_np.shape[1] == 3 else "Scalar"
                self._write_field(ts_grp, name, arr_np, "Node", attr_type)
                dims_str = " ".join(str(d) for d in arr_np.shape)
                step_attributes.append(
                    {
                        "name": name,
                        "dataset_name": name,
                        "center": "Node",
                        "type": attr_type,
                        "dims": dims_str,
                    }
                )

        # 5. Custom Cell Data
        if cell_data:
            for name, arr in cell_data.items():
                arr_np = np.asarray(arr, dtype=np.float64)
                attr_type = "Vector" if arr_np.ndim > 1 and arr_np.shape[1] == 3 else "Scalar"
                self._write_field(ts_grp, name, arr_np, "Cell", attr_type)
                dims_str = " ".join(str(d) for d in arr_np.shape)
                step_attributes.append(
                    {
                        "name": name,
                        "dataset_name": name,
                        "center": "Cell",
                        "type": attr_type,
                        "dims": dims_str,
                    }
                )

        record = XdmfTimeStepRecord(
            step_index=idx,
            time=float(time),
            h5_step_path=f"/time_series/{step_key}",
            attributes=step_attributes,
        )
        self.steps.append(record)

        # Flush HDF5 and generate XDMF XML
        self.h5_file.flush()
        self._write_xdmf_xml()

    def _write_field(
        self,
        group: h5py.Group,
        name: str,
        data: np.ndarray,
        center: str,
        attr_type: str,
    ) -> None:
        """Helper to write or update a compressed dataset."""
        if name in group:
            group[name][:] = data
        else:
            ds = group.create_dataset(name, data=data, compression="gzip")
            ds.attrs["center"] = center
            ds.attrs["attr_type"] = attr_type

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
        Writes a full binary restart checkpoint into the HDF5 archive.
        """
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

        # Write to latest and historical step checkpoint
        ckpt_grp = self.h5_file.require_group("checkpoints")
        latest_grp = ckpt_grp.require_group("latest")
        step_grp = ckpt_grp.require_group(f"checkpoint_{step:06d}")

        checkpoint.to_hdf5(latest_grp)
        checkpoint.to_hdf5(step_grp)
        self.h5_file.flush()
        return checkpoint

    def read_checkpoint(self, step: int | None = None) -> SimulationCheckpoint:
        """
        Reads a simulation checkpoint from the HDF5 archive.

        Parameters:
        -----------
        step : int, optional
            Step index to read. If None, reads the latest checkpoint.
        """
        ckpt_grp = self.h5_file.require_group("checkpoints")
        if step is None:
            if "latest" not in ckpt_grp:
                raise KeyError(f"No checkpoint found in archive '{self.h5_path}'.")
            target_grp = ckpt_grp["latest"]
        else:
            key = f"checkpoint_{step:06d}"
            if key not in ckpt_grp:
                raise KeyError(f"Checkpoint for step {step} not found in archive '{self.h5_path}'.")
            target_grp = ckpt_grp[key]

        return SimulationCheckpoint.from_hdf5(target_grp)

    def _write_xdmf_xml(self) -> None:
        """Generates standard XDMF 3.0 XML pointing to the single HDF5 archive."""
        h5_filename = self.h5_path.name
        n_elem = self.n_elements
        n_nodes = self.n_nodes
        top_type = self.topology_type

        lines = [
            '<?xml version="1.0" ?>',
            '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
            '<Xdmf Version="3.0">',
            "  <Domain>",
            '    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">',
        ]

        for s in self.steps:
            lines.append(f'      <Grid Name="Step_{s.step_index:06d}" GridType="Uniform">')
            lines.append(f'        <Time Value="{s.time:.6e}"/>')
            lines.append(
                f'        <Topology TopologyType="{top_type}" NumberOfElements="{n_elem}">'
            )
            lines.append(
                f'          <DataItem Format="HDF" DataType="Int" Dimensions="{n_elem} {self.mesh.elements.shape[1]}">'
            )
            lines.append(f"            {h5_filename}:/mesh/topology")
            lines.append("          </DataItem>")
            lines.append("        </Topology>")
            lines.append('        <Geometry GeometryType="XYZ">')
            lines.append(
                f'          <DataItem Format="HDF" DataType="Float" Precision="8" Dimensions="{n_nodes} 3">'
            )
            lines.append(f"            {h5_filename}:/mesh/geometry")
            lines.append("          </DataItem>")
            lines.append("        </Geometry>")

            for attr in s.attributes:
                lines.append(
                    f'        <Attribute Name="{attr["name"]}" AttributeType="{attr["type"]}" Center="{attr["center"]}">'
                )
                lines.append(
                    f'          <DataItem Format="HDF" DataType="Float" Precision="8" Dimensions="{attr["dims"]}">'
                )
                lines.append(f'            {h5_filename}:{s.h5_step_path}/{attr["dataset_name"]}')
                lines.append("          </DataItem>")
                lines.append("        </Attribute>")

            lines.append("      </Grid>")

        lines.append("    </Grid>")
        lines.append("  </Domain>")
        lines.append("</Xdmf>")

        self.xdmf_path.write_text("\n".join(lines))

    def close(self) -> None:
        """Flushes and closes the HDF5 archive."""
        if hasattr(self, "h5_file") and self.h5_file:
            self._write_xdmf_xml()
            self.h5_file.flush()
            self.h5_file.close()

    def __enter__(self) -> XdmfWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
