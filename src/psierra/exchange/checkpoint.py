from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import h5py
import numpy as np

from psierra.domain.base import Domain


@dataclass
class SimulationCheckpoint:
    """
    Complete state representation of a continuum simulation for checkpointing and restart.

    Stores displacement, velocity, acceleration, internal history variables
    (e.g. viscoelastic non-equilibrium stress tensors), and solver telemetry.
    """

    step: int
    time: float
    dt: float
    u: np.ndarray
    v: np.ndarray
    a: np.ndarray
    history_variables: dict[str, np.ndarray] = field(default_factory=dict)
    f_ext: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_hdf5(self, group: h5py.Group) -> None:
        """
        Serializes this checkpoint into an HDF5 group.

        Parameters:
        -----------
        group : h5py.Group
            Target HDF5 group (e.g. h5file['/checkpoints/latest']).
        """
        group.attrs["step"] = int(self.step)
        group.attrs["time"] = float(self.time)
        group.attrs["dt"] = float(self.dt)
        group.attrs["metadata_json"] = json.dumps(self.metadata)

        # Datasets
        if "u" in group:
            group["u"][:] = self.u
        else:
            group.create_dataset("u", data=self.u, compression="gzip")

        if "v" in group:
            group["v"][:] = self.v
        else:
            group.create_dataset("v", data=self.v, compression="gzip")

        if "a" in group:
            group["a"][:] = self.a
        else:
            group.create_dataset("a", data=self.a, compression="gzip")

        if self.f_ext is not None:
            if "f_ext" in group:
                group["f_ext"][:] = self.f_ext
            else:
                group.create_dataset("f_ext", data=self.f_ext, compression="gzip")

        # History variables sub-group
        hist_group = group.require_group("history_variables")
        for k, v in self.history_variables.items():
            if k in hist_group:
                hist_group[k][:] = v
            else:
                hist_group.create_dataset(k, data=v, compression="gzip")

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> SimulationCheckpoint:
        """
        Deserializes a SimulationCheckpoint from an HDF5 group.

        Parameters:
        -----------
        group : h5py.Group
            HDF5 group containing serialized checkpoint datasets and attributes.

        Returns:
        --------
        checkpoint : SimulationCheckpoint
        """
        step = int(group.attrs["step"])
        time_val = float(group.attrs["time"])
        dt_val = float(group.attrs["dt"])
        metadata_str = group.attrs.get("metadata_json", "{}")
        metadata = json.loads(metadata_str)

        u = np.array(group["u"], dtype=np.float64)
        v = np.array(group["v"], dtype=np.float64)
        a = np.array(group["a"], dtype=np.float64)

        f_ext = np.array(group["f_ext"], dtype=np.float64) if "f_ext" in group else None

        hist_vars: dict[str, np.ndarray] = {}
        if "history_variables" in group:
            for k in group["history_variables"].keys():
                hist_vars[k] = np.array(group["history_variables"][k], dtype=np.float64)

        return cls(
            step=step,
            time=time_val,
            dt=dt_val,
            u=u,
            v=v,
            a=a,
            history_variables=hist_vars,
            f_ext=f_ext,
            metadata=metadata,
        )

    @classmethod
    def from_domain(
        cls,
        domain: Domain,
        step: int,
        time: float,
        dt: float,
        f_ext: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationCheckpoint:
        """
        Constructs a checkpoint by snapshotting the current kinematic and material state of a Domain.
        """
        u = domain.u.copy()
        v = domain.v.copy()
        a = domain.a.copy()
        hist_vars = (
            domain.get_history_variables()
            if hasattr(domain, "get_history_variables")
            else {}
        )
        return cls(
            step=step,
            time=time,
            dt=dt,
            u=u,
            v=v,
            a=a,
            history_variables=hist_vars,
            f_ext=f_ext.copy() if f_ext is not None else None,
            metadata=metadata if metadata is not None else {},
        )

    def apply_to_domain(self, domain: Domain) -> None:
        """
        Applies this checkpoint's state (displacements, velocities, accelerations, and internal history)
        to a physical Domain.
        """
        domain.update_state(u=self.u, v=self.v, a=self.a)

        if self.history_variables and hasattr(domain, "set_history_variables"):
            domain.set_history_variables(self.history_variables)

        if hasattr(domain, "set_time_step"):
            domain.set_time_step(self.dt)
