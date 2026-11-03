"""
Multi-Body System Container for Project Sierra.
==============================================
Acts as the top-level assembly layer (analogous to *ASSEMBLY / *INSTANCE in Abaqus),
combining multiple constituent domains (single or coupled) into a unified system
with global DOF indexing, block sparse assembly, and state propagation.
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Literal
from pathlib import Path

import numpy as np
import pyvista as pv
import scipy.sparse as spsp

from psierra.domain.base import Domain
from psierra.exchange.pvd import PvdCollectionWriter


class MultiBodySystem(Domain):
    """
    Multi-Body Assembly Container for Project Sierra.

    Aggregates multiple physical domains (e.g. deformable bodies, shell structures,
    coupled multiphysics bodies) into a single unified global equation system.

    Implements the Domain composite interface:
    - Global state vector management (concatenated DOFs across bodies).
    - Block-diagonal mass matrix: M = diag(M_1, M_2, ...).
    - Block-diagonal tangent stiffness matrix: K_tan = diag(K_1, K_2, ...).
    - Concatenated internal force vector: F_int = [F_int,1; F_int,2; ...].
    - Assembly-level contact interaction definition: system.add_contact_pair(slave, master, ...).
    - Transparent solver compatibility with NonlinearStaticAnalysis and NonlinearDynamicAnalysis.
    """

    def __init__(
        self,
        bodies: dict[str, Domain] | Sequence[Domain] | None = None,
        names: Sequence[str] | None = None,
        contacts: Sequence[Any] | None = None,
        **named_bodies: Domain,
    ) -> None:
        """
        Parameters:
        -----------
        bodies : dict[str, Domain] or Sequence[Domain], optional
            Collection of physical continuum domains, either as a dictionary mapping
            {name: domain} or as an ordered sequence of domains.
        names : Sequence[str], optional
            Descriptive names for each body (when bodies is a sequence).
        contacts : Sequence[RigidContact | DeformableContact | Any], optional
            Contact interaction constraints defined between bodies or obstacles.
        named_bodies : Domain
            Additional body domains passed as keyword arguments:
            MultiBodySystem(Beam=beam, Floor=floor).
        """
        if isinstance(bodies, dict):
            self._names = list(bodies.keys())
            self._bodies = list(bodies.values())
        elif bodies is None and named_bodies:
            self._names = list(named_bodies.keys())
            self._bodies = list(named_bodies.values())
        elif isinstance(bodies, (list, tuple)):
            self._bodies = list(bodies)
            if names is not None:
                if len(names) != len(self._bodies):
                    raise ValueError(
                        f"Number of names ({len(names)}) must match number of bodies ({len(self._bodies)})."
                    )
                self._names = list(names)
            else:
                self._names = [f"Body_{i}" for i in range(len(self._bodies))]
        else:
            raise ValueError(
                "MultiBodySystem requires a dictionary of {name: domain} or a list of domains."
            )

        if not self._bodies:
            raise ValueError("MultiBodySystem requires at least one body domain.")

        self._n_bodies = len(self._bodies)
        self._contacts: list[Any] = list(contacts) if contacts is not None else []

        # Precompute DOF offsets and slices
        self._dof_offsets = np.zeros(self._n_bodies + 1, dtype=np.int64)
        for i, body in enumerate(self._bodies):
            self._dof_offsets[i + 1] = self._dof_offsets[i] + body.n_dofs

        self._n_dofs = int(self._dof_offsets[-1])

        # Global Kinematic State Vectors
        self.u = np.zeros(self._n_dofs, dtype=np.float64)
        self.v = np.zeros(self._n_dofs, dtype=np.float64)
        self.a = np.zeros(self._n_dofs, dtype=np.float64)

        # Sync initial state from bodies if present
        for i, body in enumerate(self._bodies):
            sl = self.get_body_dof_slice(i)
            if hasattr(body, "u"):
                self.u[sl] = body.u
            if hasattr(body, "v"):
                self.v[sl] = body.v
            if hasattr(body, "a"):
                self.a[sl] = body.a

    # =========================================================================
    # Properties & Indexing
    # =========================================================================

    @property
    def n_bodies(self) -> int:
        """Number of physical bodies in the system."""
        return self._n_bodies

    @property
    def bodies(self) -> list[Domain]:
        """List of constituent domain objects."""
        return self._bodies

    @property
    def names(self) -> list[str]:
        """Names of the constituent bodies."""
        return self._names

    @property
    def n_dofs(self) -> int:
        """Total degrees of freedom in the multi-body system."""
        return self._n_dofs

    @property
    def contacts(self) -> list[Any]:
        """List of active contact constraints defined on the multi-body system."""
        return self._contacts

    def __getitem__(self, key: str | int) -> Domain:
        """Retrieves a constituent body domain by name or integer index."""
        if isinstance(key, str):
            if key not in self._names:
                raise KeyError(
                    f"Body '{key}' not found in MultiBodySystem. Available: {self._names}"
                )
            idx = self._names.index(key)
            return self._bodies[idx]
        elif isinstance(key, int):
            return self._bodies[key]
        raise TypeError(f"Key must be str or int, got {type(key)}")

    def __contains__(self, key: str | Domain) -> bool:
        """Checks if a body name or domain belongs to the system."""
        return key in self._names or key in self._bodies

    def keys(self) -> list[str]:
        """Returns the list of constituent body names."""
        return list(self._names)

    def values(self) -> list[Domain]:
        """Returns the list of constituent body domains."""
        return list(self._bodies)

    def items(self) -> list[tuple[str, Domain]]:
        """Returns (name, domain) pairs of constituent bodies."""
        return list(zip(self._names, self._bodies))

    def get_body_index(self, body: str | int | Domain) -> int:
        """Resolves body identifier (name, index, or domain instance) to integer index."""
        if isinstance(body, int):
            if body < 0 or body >= self._n_bodies:
                raise IndexError(
                    f"Body index {body} out of range [0, {self._n_bodies})."
                )
            return body
        if isinstance(body, str):
            if body not in self._names:
                raise KeyError(
                    f"Body '{body}' not found. Available bodies: {self._names}"
                )
            return self._names.index(body)
        if body in self._bodies:
            return self._bodies.index(body)
        raise ValueError(f"Body {body} is not a member of this MultiBodySystem.")

    def get_body_dof_slice(self, body: str | int | Domain) -> slice:
        """Returns the global DOF slice for specified body (by name, index, or domain)."""
        idx = self.get_body_index(body)
        start = int(self._dof_offsets[idx])
        stop = int(self._dof_offsets[idx + 1])
        return slice(start, stop)

    def get_body_displacement(self, body: str | int | Domain) -> np.ndarray:
        """Extracts displacement array for specified body."""
        idx = self.get_body_index(body)
        sl = self.get_body_dof_slice(idx)
        u_body = self.u[sl]
        b = self._bodies[idx]
        if hasattr(b, "displacement"):
            return b.displacement
        return u_body.reshape(-1, 3)

    def add_contact(self, contact: Any) -> None:
        """
        Registers a contact constraint on the multi-body system.
        Automatically binds DOF offsets if the contact's slave domain is a constituent body.
        """
        slave = getattr(contact, "slave", getattr(contact, "domain", None))
        if slave is not None and slave in self._bodies:
            contact.dof_offset = int(self.get_body_dof_slice(slave).start)
        self._contacts.append(contact)

    def add_contact_pair(
        self,
        slave: str | int | Domain,
        master: str | int | Domain | Any,
        k_penalty: float = 1e4,
        c_damping: float = 0.0,
        friction: float = 0.0,
        slave_nodes: Sequence[int] | np.ndarray | None = None,
        formulation: Literal["penalty", "augmented_lagrangian"] = "penalty",
        **kwargs: Any,
    ) -> Any:
        """
        Defines and registers a contact interaction between two assembly bodies
        or an assembly body and an external obstacle.

        Parameters:
        -----------
        slave : str, int, or Domain
            Deformable body undergoing contact (name, index, or domain).
        master : str, int, Domain, or CollisionShape
            Master obstacle or target domain (name, index, domain, or collision shape).
        k_penalty : float, default=1e4
            Normal penalty contact stiffness in N/mm.
        c_damping : float, default=0.0
            Contact velocity damping coefficient in N*s/mm.
        friction : float, default=0.0
            Friction coefficient.
        slave_nodes : Sequence[int] or np.ndarray, optional
            Specific candidate contact nodes on the slave body (defaults to surface nodes).
        formulation : {'penalty', 'augmented_lagrangian'}, default='penalty'
            Contact formulation method.

        Returns:
        --------
        contact : RigidContact or DeformableContact
            The registered contact interaction constraint.
        """
        from psierra.boundary.contact import (
            RigidContact,
            DeformableContact,
        )

        slave_body = self[slave] if isinstance(slave, (str, int)) else slave
        master_obj = (
            self[master]
            if (isinstance(master, (str, int)) and master in self)
            else master
        )

        slave_idx = self.get_body_index(slave_body)
        slave_offset = int(self._dof_offsets[slave_idx])
        dofs_per_node = getattr(slave_body, "dofs_per_node", 3)

        # Detect whether master is a flexible continuum Domain
        is_deformable_master = (
            hasattr(master_obj, "mesh")
            and hasattr(master_obj.mesh, "surface")
            and hasattr(master_obj, "n_dofs")
        )

        if is_deformable_master:
            master_idx = (
                self.get_body_index(master_obj) if master_obj in self else None
            )
            master_offset = (
                int(self._dof_offsets[master_idx]) if master_idx is not None else 0
            )

            contact = DeformableContact(
                slave=slave_body,
                master=master_obj,
                slave_nodes=slave_nodes,
                k_penalty=k_penalty,
                c_damping=c_damping,
                friction=friction,
                slave_offset=slave_offset,
                master_offset=master_offset,
                domain=self,
                formulation=formulation,
                **kwargs,
            )
        else:
            # Rigid obstacle or CollisionShape
            contact = RigidContact(
                slave=slave_body,
                master=master_obj,
                slave_nodes=slave_nodes,
                k_penalty=k_penalty,
                c_damping=c_damping,
                friction=friction,
                dof_offset=slave_offset,
                dofs_per_node=dofs_per_node,
                domain=self,
                formulation=formulation,
                **kwargs,
            )

        self._contacts.append(contact)
        return contact

    @property
    def current_nodes(self) -> np.ndarray:
        """Current spatial coordinates of all nodes across all bodies (stacked)."""
        return np.vstack([b.current_nodes for b in self._bodies])

    @property
    def displacement(self) -> np.ndarray:
        """Concatenated translational displacement vectors across all constituent bodies (N_total_nodes, 3)."""
        disps = []
        for i, body in enumerate(self._bodies):
            if hasattr(body, "displacement"):
                disps.append(body.displacement)
            else:
                sl = self.get_body_dof_slice(i)
                disps.append(self.u[sl].reshape(-1, 3))
        return np.vstack(disps) if disps else np.empty((0, 3), dtype=np.float64)

    # =========================================================================
    # Kinematic State Updates
    # =========================================================================

    def update_state(
        self,
        u: np.ndarray | None = None,
        v: np.ndarray | None = None,
        a: np.ndarray | None = None,
    ) -> None:
        """
        Updates global kinematic state vectors and dispatches respective slices
        to each constituent body domain.
        """
        if u is not None:
            self.u[:] = u
        if v is not None:
            self.v[:] = v
        if a is not None:
            self.a[:] = a

        for i, body in enumerate(self._bodies):
            sl = self.get_body_dof_slice(i)
            body_u = self.u[sl] if u is not None else None
            body_v = self.v[sl] if v is not None else None
            body_a = self.a[sl] if a is not None else None
            body.update_state(u=body_u, v=body_v, a=body_a)

    # =========================================================================
    # Global Matrix & Force Assembly
    # =========================================================================

    def assemble_mass_matrix(self) -> spsp.csr_matrix:
        """
        Assembles block-diagonal global consistent mass matrix:
        M = diag(M_0, M_1, ..., M_{N-1}).
        """
        mass_blocks = [body.assemble_mass_matrix() for body in self._bodies]
        return spsp.block_diag(mass_blocks, format="csr")

    def assemble_stiffness_matrix(self) -> spsp.csr_matrix:
        """
        Assembles block-diagonal global small-strain stiffness matrix:
        K = diag(K_0, K_1, ..., K_{N-1}).
        """
        stiff_blocks = [body.assemble_stiffness_matrix() for body in self._bodies]
        return spsp.block_diag(stiff_blocks, format="csr")

    def assemble_tangent_stiffness_matrix(self) -> spsp.csr_matrix:
        """
        Assembles block-diagonal global nonlinear tangent stiffness matrix:
        K_tan = diag(K_tan,0, K_tan,1, ..., K_tan,N-1).
        """
        k_blocks = []
        for body in self._bodies:
            if hasattr(body, "assemble_tangent_stiffness_matrix"):
                k_blocks.append(body.assemble_tangent_stiffness_matrix())
            else:
                k_blocks.append(body.assemble_stiffness_matrix())
        return spsp.block_diag(k_blocks, format="csr")

    def assemble_internal_forces(self) -> np.ndarray:
        """
        Assembles global internal force vector:
        F_int = [F_int,0; F_int,1; ...; F_int,N-1].
        """
        f_int = np.zeros(self._n_dofs, dtype=np.float64)
        for i, body in enumerate(self._bodies):
            sl = self.get_body_dof_slice(i)
            if hasattr(body, "assemble_internal_forces"):
                f_int[sl] = body.assemble_internal_forces()
            else:
                f_int[sl] = body.assemble_stiffness_matrix() @ body.u
        return f_int

    def assemble_internal_force_vector(self) -> np.ndarray:
        """Alias for assemble_internal_forces."""
        return self.assemble_internal_forces()

    # =========================================================================
    # Visualization & Export
    # =========================================================================

    def to_pyvista(
        self,
        factor: float = 0.0,
        **kwargs,
    ) -> pv.UnstructuredGrid:
        """
        Combines all constituent bodies into a unified PyVista UnstructuredGrid.
        Attaches 'Body ID' and sets active scalars to 'Displacement'.
        """
        grids: list[pv.UnstructuredGrid] = []

        for i, body in enumerate(self._bodies):
            if hasattr(body, "to_pyvista"):
                g = body.to_pyvista(factor=factor, **kwargs)
                if isinstance(g, pv.PolyData):
                    g = g.cast_to_unstructured_grid()
            elif hasattr(body, "mesh") and hasattr(body.mesh, "ugrid"):
                g = body.mesh.ugrid.copy(deep=True)
                if factor != 0.0 and hasattr(body, "displacement"):
                    g.points = body.mesh.nodes + factor * body.displacement
            else:
                continue

            # Tag cells with Body ID
            g.cell_data["Body ID"] = np.full(g.n_cells, i, dtype=np.int32)
            grids.append(g)

        if not grids:
            raise RuntimeError("None of the constituent bodies support PyVista export.")

        if len(grids) == 1:
            merged = grids[0]
        else:
            merged = grids[0]
            for g in grids[1:]:
                merged = merged.merge(g)

        if "Displacement" in merged.point_data:
            merged.set_active_scalars("Displacement")

        return merged

    def save_state(
        self,
        filepath: Path,
        factor: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Exports the multi-body system state to disk.
        If filepath ends in '.vtu', exports a single combined mesh with 'Body ID'.
        If filepath ends in '.pvd', exports individual bodies and writes a master PVD.
        """
        filepath = Path(filepath)
        if filepath.suffix != ".vtu" and filepath.suffix != ".pvd":
            filepath = filepath.with_suffix(".vtu")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.suffix == ".pvd":
            self.save_pvd(filepath, factor=factor, **kwargs)
        else:
            grid = self.to_pyvista(factor=factor, **kwargs)
            grid.save(str(filepath))

    def save_pvd(
        self,
        filepath: Path,
        timestep: float = 0.0,
        factor: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Exports each body to its own .vtu file and produces a master ParaView .pvd collection.
        """
        filepath = Path(filepath)
        if filepath.suffix != ".pvd":
            filepath = filepath.with_suffix(".pvd")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        writer = PvdCollectionWriter(filepath)
        stem = filepath.stem

        for i, body in enumerate(self._bodies):
            body_vtu = filepath.parent / f"{stem}_{self._names[i]}.vtu"
            if hasattr(body, "save_state"):
                body.save_state(body_vtu, factor=factor, **kwargs)
            else:
                grid = self.to_pyvista(factor=factor, **kwargs)
                grid.save(str(body_vtu))

            writer.add_dataset(
                filename=body_vtu.name,
                timestep=timestep,
                part=i,
                group=self._names[i],
            )
        writer.write()
