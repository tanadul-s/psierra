from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import scipy.sparse as spsp

from psierra.boundary.base import NeumannBoundaryCondition


class NodalForce(NeumannBoundaryCondition):
    """
    Concentrated point force applied to specific nodes.
    """

    def __init__(
        self,
        node_indices: Sequence[int] | np.ndarray,
        force_vector: Sequence[float] | np.ndarray,
        dofs_per_node: int = 3,
    ) -> None:
        """
        Parameters:
        -----------
        node_indices : array-like
            Nodes where the force is applied.
        force_vector : (3,) array-like
            Force components [Fx, Fy, Fz] in Newtons applied to EACH selected node.
        """
        self.node_indices = np.asarray(node_indices, dtype=np.int64)
        self.force_vector = np.asarray(force_vector, dtype=np.float64)
        self.dofs_per_node = dofs_per_node

        if self.force_vector.shape != (self.dofs_per_node,):
            raise ValueError(
                f"force_vector must have shape ({self.dofs_per_node},), got {self.force_vector.shape}"
            )

    def compute_force_vector(self, total_dofs: int) -> np.ndarray:
        """Computes and returns the assembled concentrated nodal force vector."""
        f_global = np.zeros(total_dofs, dtype=np.float64)
        for i, node in enumerate(self.node_indices):
            start_dof = node * self.dofs_per_node
            f_global[start_dof : start_dof + self.dofs_per_node] += self.force_vector
        return f_global


class GravityLoad(NeumannBoundaryCondition):
    """
    Uniform volumetric body force (Gravity) computed via global mass matrix M:
    F_gravity = M @ g_global
    """

    def __init__(
        self,
        acceleration: Sequence[float] | np.ndarray = (0.0, 0.0, -9.81),
        dofs_per_node: int = 3,
    ) -> None:
        """
        Parameters:
        -----------
        acceleration : (3,) array-like, default=(0, 0, -9.81)
            Gravitational acceleration vector [gx, gy, gz] in m/s^2.
        """
        self.acceleration = np.asarray(acceleration, dtype=np.float64)
        self.dofs_per_node = dofs_per_node

    def compute_force_vector(self, mass_matrix: spsp.csr_matrix) -> np.ndarray:
        """
        Computes F_gravity = M @ g_global.

        Parameters:
        -----------
        mass_matrix : spsp.csr_matrix
            Assembled global consistent mass matrix.
        """
        total_dofs = mass_matrix.shape[0]
        n_nodes = total_dofs // self.dofs_per_node

        # Broadcast [gx, gy, gz] to all nodes
        g_global = np.tile(self.acceleration, n_nodes)
        return mass_matrix @ g_global


class HydrostaticPressureLoad(NeumannBoundaryCondition):
    """
    Spatially-varying boundary hydrostatic pressure surface traction:
        p(x) = rho_fluid * ||g|| * max(0, h_fluid - x · u_up)
        t(x) = - p(x) * n(x)

    where:
    - rho_fluid is the fluid mass density (e.g. CSF 1000 kg/m^3 or water)
    - g is the gravitational acceleration vector [gx, gy, gz]
    - u_up = -g / ||g|| is the upward unit vector opposite to gravity
    - h_fluid is the fluid free surface elevation along u_up
    - n(x) is the outward unit normal of the boundary surface facet

    Integrates inward hydrostatic surface pressure over exterior boundary facets of
    a VolumetricMesh, SurfaceMesh, or SolidDomain:
        F_hydro = sum_facets int_facet N_i(x) t(x) dA

    Features:
    - Exact 2nd-order triangular surface quadrature for linear depth gradients:
        F_i^facet = -n * A * (p_i + sum_j p_j) / 12
    - Archimedean buoyancy: the net upward vertical resultant exactly recovers Archimedes'
      buoyant force (integral of rho_fluid * g dV) to machine precision for closed submerged meshes.
    - Captures both buoyant lift and lateral hydrostatic surface compression / squeezing.
    - Evaluates on reference mesh geometry or deformed current spatial coordinates (SolidDomain.current_surface).
    """

    def __init__(
        self,
        fluid_density: float = 1.0e-9,  # 1000 kg/m^3 in tonne/mm^3
        acceleration: Sequence[float] | np.ndarray = (0.0, 0.0, -9810.0),
        fluid_level: float | None = None,
        up_direction: Sequence[float] | np.ndarray | None = None,
        dofs_per_node: int = 3,
        surface_mesh: Any | None = None,
    ) -> None:
        """
        Parameters:
        -----------
        fluid_density : float, default=1.0e-9
            Fluid mass density in consistent units (tonne/mm^3, e.g. 1.0e-9 for water/CSF).
        acceleration : Sequence[float] or np.ndarray of shape (3,), default=(0, 0, -9810)
            Gravitational acceleration vector [gx, gy, gz] in mm/s^2.
        fluid_level : float, optional
            Free surface elevation h_fluid along the upward direction u_up.
            If None, automatically set to the maximum coordinate along u_up (fully submerged).
        up_direction : Sequence[float] or np.ndarray of shape (3,), optional
            Custom upward unit vector u_up. If None, derived as -g / ||g||.
        dofs_per_node : int, default=3
            Degrees of freedom per node.
        surface_mesh : SurfaceMesh, optional
            Pre-extracted boundary SurfaceMesh. If None, extracted on-the-fly from the domain/mesh.
        """
        self.fluid_density = float(fluid_density)
        self.acceleration = np.asarray(acceleration, dtype=np.float64)
        if self.acceleration.shape != (3,):
            raise ValueError(
                f"acceleration must have shape (3,), got {self.acceleration.shape}"
            )

        g_norm = float(np.linalg.norm(self.acceleration))
        if g_norm < 1e-12:
            raise ValueError("Gravitational acceleration norm must be strictly positive.")
        self.g_mag = g_norm

        if up_direction is not None:
            up_vec = np.asarray(up_direction, dtype=np.float64)
            norm_up = float(np.linalg.norm(up_vec))
            if norm_up < 1e-12:
                raise ValueError("up_direction norm must be strictly positive.")
            self.u_up = up_vec / norm_up
        else:
            self.u_up = -self.acceleration / g_norm

        self.fluid_level = float(fluid_level) if fluid_level is not None else None
        self.dofs_per_node = dofs_per_node
        self.surface_mesh = surface_mesh

    def _resolve_surface_and_dofs(
        self,
        domain_or_mesh: Any | None,
        total_dofs: int | None,
        current_configuration: bool,
    ) -> tuple[Any, int]:
        """Resolves target SurfaceMesh and total system DOFs from input object."""
        smesh = None
        n_dofs = total_dofs

        if domain_or_mesh is not None:
            if hasattr(domain_or_mesh, "current_surface") and current_configuration:
                smesh = domain_or_mesh.current_surface
                if n_dofs is None and hasattr(domain_or_mesh, "n_dofs"):
                    n_dofs = domain_or_mesh.n_dofs
            elif hasattr(domain_or_mesh, "mesh") and hasattr(domain_or_mesh.mesh, "surface"):
                smesh = domain_or_mesh.mesh.surface
                if n_dofs is None and hasattr(domain_or_mesh, "n_dofs"):
                    n_dofs = domain_or_mesh.n_dofs
            elif hasattr(domain_or_mesh, "surface"):
                smesh = domain_or_mesh.surface
                if n_dofs is None and hasattr(domain_or_mesh, "n_dofs"):
                    n_dofs = domain_or_mesh.n_dofs
            elif hasattr(domain_or_mesh, "elements") and hasattr(domain_or_mesh, "nodes"):
                smesh = domain_or_mesh
                if n_dofs is None and hasattr(domain_or_mesh, "n_dofs"):
                    n_dofs = domain_or_mesh.n_dofs

        if smesh is None:
            smesh = self.surface_mesh

        if smesh is None:
            raise ValueError(
                "Unable to resolve SurfaceMesh. Provide a SolidDomain, VolumetricMesh, "
                "or SurfaceMesh to compute_force_vector() or initialize with surface_mesh."
            )

        if n_dofs is None:
            if getattr(smesh, "node_indices", None) is not None:
                n_dofs = int(np.max(smesh.node_indices) + 1) * self.dofs_per_node
            else:
                n_dofs = smesh.n_nodes * self.dofs_per_node

        return smesh, n_dofs

    def compute_force_vector(
        self,
        domain_or_mesh: Any | None = None,
        total_dofs: int | None = None,
        current_configuration: bool = False,
    ) -> np.ndarray:
        """
        Computes and assembles the global hydrostatic nodal force vector.

        Parameters:
        -----------
        domain_or_mesh : SolidDomain, VolumetricMesh, or SurfaceMesh, optional
            Geometric domain or boundary surface to evaluate tractions on.
        total_dofs : int, optional
            Total number of global degrees of freedom. Inferred if domain is provided.
        current_configuration : bool, default=False
            If True and domain_or_mesh is a SolidDomain, evaluates surface normals and
            facet coordinates on the deformed current configuration.

        Returns:
        --------
        f_hydro : np.ndarray of shape (total_dofs,)
            Assembled global nodal hydrostatic force vector.
        """
        smesh, total_dofs = self._resolve_surface_and_dofs(
            domain_or_mesh, total_dofs, current_configuration
        )

        surf_coords = smesh.nodes
        elems = smesh.elements[:, :3]  # Linear 3-node triangular facets

        p1 = surf_coords[elems[:, 0]]
        p2 = surf_coords[elems[:, 1]]
        p3 = surf_coords[elems[:, 2]]

        # Outward face normals and areas
        v12 = p2 - p1
        v13 = p3 - p1
        cross = np.cross(v12, v13)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-15)

        # Elevations along upward vertical axis
        z1 = np.dot(p1, self.u_up)
        z2 = np.dot(p2, self.u_up)
        z3 = np.dot(p3, self.u_up)

        # Automatic fluid level (top of surface mesh if None)
        h_fluid = (
            self.fluid_level
            if self.fluid_level is not None
            else float(np.max([z1.max(), z2.max(), z3.max()]))
        )

        # Submerged depths below free fluid surface
        d1 = np.maximum(0.0, h_fluid - z1)
        d2 = np.maximum(0.0, h_fluid - z2)
        d3 = np.maximum(0.0, h_fluid - z3)

        # Hydrostatic pressures at 3 vertices: p = rho * g * d
        p_scale = self.fluid_density * self.g_mag
        press1 = p_scale * d1
        press2 = p_scale * d2
        press3 = p_scale * d3
        press_sum = press1 + press2 + press3

        # Consistent nodal force contributions:
        # F_i^facet = -n * A * (press_i + sum_j press_j) / 12
        f1 = -normals * (areas * (press1 + press_sum) / 12.0)[:, None]
        f2 = -normals * (areas * (press2 + press_sum) / 12.0)[:, None]
        f3 = -normals * (areas * (press3 + press_sum) / 12.0)[:, None]

        # Map to global volumetric node indices
        global_map = getattr(smesh, "node_indices", None)
        if global_map is not None:
            g1 = global_map[elems[:, 0]]
            g2 = global_map[elems[:, 1]]
            g3 = global_map[elems[:, 2]]
        else:
            g1 = elems[:, 0]
            g2 = elems[:, 1]
            g3 = elems[:, 2]

        # Assemble into global force vector
        f_global = np.zeros(total_dofs, dtype=np.float64)
        np.add.at(f_global, (3 * g1)[:, None] + np.arange(3), f1)
        np.add.at(f_global, (3 * g2)[:, None] + np.arange(3), f2)
        np.add.at(f_global, (3 * g3)[:, None] + np.arange(3), f3)

        return f_global

    def evaluate_facet_pressures(
        self,
        domain_or_mesh: Any | None = None,
        current_configuration: bool = False,
    ) -> np.ndarray:
        """
        Evaluates the average hydrostatic pressure (in MPa) at the centroid of each facet.

        Returns:
        --------
        pressures : np.ndarray of shape (N_facets,)
        """
        smesh, _ = self._resolve_surface_and_dofs(
            domain_or_mesh, None, current_configuration
        )
        centroids = smesh.centroids
        z_c = np.dot(centroids, self.u_up)
        h_fluid = (
            self.fluid_level
            if self.fluid_level is not None
            else float(np.max(np.dot(smesh.nodes, self.u_up)))
        )
        depths = np.maximum(0.0, h_fluid - z_c)
        return self.fluid_density * self.g_mag * depths

    def compute_buoyant_resultant(
        self,
        domain_or_mesh: Any | None = None,
        current_configuration: bool = False,
    ) -> tuple[np.ndarray, float]:
        """
        Computes the net 3D resultant force and vertical buoyant force magnitude.

        Returns:
        --------
        resultant_vector : np.ndarray of shape (3,)
            Total integrated 3D force vector [Fx, Fy, Fz] in Newtons.
        buoyant_magnitude : float
            Net upward vertical buoyant force along u_up in Newtons.
        """
        f_vec = self.compute_force_vector(
            domain_or_mesh, current_configuration=current_configuration
        )
        res_3d = f_vec.reshape(-1, self.dofs_per_node).sum(axis=0)
        buoyant_mag = float(np.dot(res_3d, self.u_up))
        return res_3d, buoyant_mag


class BuoyancyLoad(NeumannBoundaryCondition):
    """
    Volumetric Archimedes Buoyancy Body Force:
        F_buoyancy = -rho_fluid * V_sub * g = rho_fluid * M_unit * (-g)

    Acts as an upward volumetric body force opposing gravity.

    Features:
    - Fully submerged continua:
        F_buoyancy = (rho_fluid / rho_solid) * M * (-g_global)
        where M is the assembled consistent mass matrix.
    - Partially submerged continua:
        Integrates element volume immersion fractions alpha_e based on fluid level h_fluid.
    - Matches Archimedes' theoretical buoyancy to exact machine precision:
        ||F_buoyancy|| = rho_fluid * V_sub * ||g||
    """

    def __init__(
        self,
        fluid_density: float = 1.0e-9,  # 1000 kg/m^3 in tonne/mm^3
        acceleration: Sequence[float] | np.ndarray = (0.0, 0.0, -9810.0),
        fluid_level: float | None = None,
        up_direction: Sequence[float] | np.ndarray | None = None,
        dofs_per_node: int = 3,
    ) -> None:
        """
        Parameters:
        -----------
        fluid_density : float, default=1.0e-9
            Fluid mass density in consistent units (tonne/mm^3).
        acceleration : Sequence[float] or np.ndarray of shape (3,), default=(0, 0, -9810)
            Gravitational acceleration vector [gx, gy, gz] in mm/s^2.
        fluid_level : float, optional
            Free surface elevation h_fluid along the upward direction u_up.
            If None, the continuum is treated as fully submerged.
        up_direction : Sequence[float] or np.ndarray of shape (3,), optional
            Custom upward unit vector. If None, derived as -g / ||g||.
        dofs_per_node : int, default=3
            Degrees of freedom per node.
        """
        self.fluid_density = float(fluid_density)
        self.acceleration = np.asarray(acceleration, dtype=np.float64)
        if self.acceleration.shape != (3,):
            raise ValueError(
                f"acceleration must have shape (3,), got {self.acceleration.shape}"
            )

        g_norm = float(np.linalg.norm(self.acceleration))
        if g_norm < 1e-12:
            raise ValueError("Gravitational acceleration norm must be strictly positive.")
        self.g_mag = g_norm

        if up_direction is not None:
            up_vec = np.asarray(up_direction, dtype=np.float64)
            norm_up = float(np.linalg.norm(up_vec))
            if norm_up < 1e-12:
                raise ValueError("up_direction norm must be strictly positive.")
            self.u_up = up_vec / norm_up
        else:
            self.u_up = -self.acceleration / g_norm

        self.fluid_level = float(fluid_level) if fluid_level is not None else None
        self.dofs_per_node = dofs_per_node

    def compute_force_vector(
        self,
        domain_or_mesh: Any | None = None,
        mass_matrix: spsp.csr_matrix | None = None,
        solid_density: float | None = None,
    ) -> np.ndarray:
        """
        Computes the global upward buoyant force vector.

        Parameters:
        -----------
        domain_or_mesh : SolidDomain or VolumetricMesh, optional
            Volumetric continuum domain or mesh.
        mass_matrix : spsp.csr_matrix, optional
            Pre-assembled consistent mass matrix M. If provided along with solid_density
            and fluid_level is None, evaluates F = (rho_fluid / rho_solid) * M @ (-g).
        solid_density : float, optional
            Solid mass density matching mass_matrix.

        Returns:
        --------
        f_buoyancy : np.ndarray of shape (total_dofs,)
        """
        # Fast path for fully submerged body with pre-assembled consistent mass matrix
        if (
            mass_matrix is not None
            and solid_density is not None
            and self.fluid_level is None
        ):
            total_dofs = mass_matrix.shape[0]
            n_nodes = total_dofs // self.dofs_per_node
            buoyant_accel = -self.acceleration * (self.fluid_density / solid_density)
            g_global = np.tile(buoyant_accel, n_nodes)
            return mass_matrix @ g_global

        # Resolve VolumetricMesh
        vmesh = None
        if domain_or_mesh is not None:
            if hasattr(domain_or_mesh, "mesh"):
                vmesh = domain_or_mesh.mesh
            elif hasattr(domain_or_mesh, "elements") and hasattr(domain_or_mesh, "nodes"):
                vmesh = domain_or_mesh

        if vmesh is None:
            raise ValueError(
                "BuoyancyLoad requires a SolidDomain or VolumetricMesh to compute element volumes."
            )

        coords = vmesh.nodes
        elems = vmesh.elements[:, :4]  # Tet4 elements

        p0 = coords[elems[:, 0]]
        p1 = coords[elems[:, 1]]
        p2 = coords[elems[:, 2]]
        p3 = coords[elems[:, 3]]

        # Element volumes via 6x6 determinant
        v_elem = (
            np.abs(np.linalg.det(np.stack([p1 - p0, p2 - p0, p3 - p0], axis=-1)))
            / 6.0
        )

        # Immersion fraction per element
        if self.fluid_level is None:
            alpha = np.ones_like(v_elem)
        else:
            z0 = np.dot(p0, self.u_up)
            z1 = np.dot(p1, self.u_up)
            z2 = np.dot(p2, self.u_up)
            z3 = np.dot(p3, self.u_up)
            z_c = 0.25 * (z0 + z1 + z2 + z3)
            z_min = np.minimum.reduce([z0, z1, z2, z3])
            z_max = np.maximum.reduce([z0, z1, z2, z3])
            denom = np.maximum(z_max - z_min, 1e-12)
            alpha = np.clip((self.fluid_level - z_min) / denom, 0.0, 1.0)

        # Buoyant force per element: alpha * V_elem * rho_fluid * (-g)
        f_elem = (alpha * v_elem * self.fluid_density)[:, None] * (-self.acceleration)

        f_global = np.zeros(vmesh.n_dofs, dtype=np.float64)
        for i in range(4):
            np.add.at(
                f_global,
                (3 * elems[:, i])[:, None] + np.arange(3),
                0.25 * f_elem,
            )

        return f_global

    def compute_buoyant_resultant(
        self,
        domain_or_mesh: Any,
    ) -> tuple[np.ndarray, float]:
        """Computes net 3D buoyant resultant vector and upward magnitude."""
        f_vec = self.compute_force_vector(domain_or_mesh)
        res_3d = f_vec.reshape(-1, self.dofs_per_node).sum(axis=0)
        buoyant_mag = float(np.dot(res_3d, self.u_up))
        return res_3d, buoyant_mag

