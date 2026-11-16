"""Classical Hertzian Contact Mechanics: Exact Closed-Form Analytical Solutions.

Milestone 1.1 of Loop 3 Contact Mechanics Overhaul in Project Sierra.
---------------------------------------------------------------------
Provides exact analytical ground truth for non-conforming frictionless elastic
contact formulations, disentangled from soft-tissue large deformations:
1. Spherical Contact:
   - Sphere on flat rigid/elastic plane or two contacting spheres.
   - Contact radius: a = (3 * F_N * R* / (4 * E*))^(1/3)
   - Indentation depth: delta = a^2 / R*
   - Peak normal pressure: p0 = 3 * F_N / (2 * pi * a^2)
   - Radial pressure profile: p(r) = p0 * sqrt(1 - (r/a)^2)
   - Internal subsurface stress field (sigma_z, sigma_r, tau_max)
2. Cylindrical Contact:
   - Parallel cylinders or cylinder on flat plane.
   - Half-width: b = sqrt(4 * F_N * R* / (pi * L * E*))
   - Peak normal pressure: p0 = 2 * F_N / (pi * b * L)
   - Transverse pressure profile: p(x) = p0 * sqrt(1 - (x/b)^2)
3. Discretization & Verification Helpers:
   - Vectorized nodal tributary area integration on surface triangular meshes.
   - Reconstructed contact pressure fields from discrete nodal force vectors.
   - Goodness-of-fit and relative error evaluation against analytical profiles.
"""

from __future__ import annotations

import numpy as np

from psierra.mesh.surface import SurfaceMesh


class HertzSphereContact:
    """Exact closed-form Hertzian contact theory for 3D spherical interfaces.

    Parameters:
    -----------
    radius : float
        Radius of the primary elastic sphere R_1 [length].
    E_1 : float
        Young's modulus of body 1 [force/length^2].
    nu_1 : float
        Poisson's ratio of body 1 [dimensionless].
    E_2 : float, default=np.inf
        Young's modulus of body 2 (default np.inf for rigid obstacle).
    nu_2 : float, default=0.0
        Poisson's ratio of body 2.
    radius_2 : float, default=np.inf
        Radius of contacting body 2 (default np.inf for flat plane).
    """

    def __init__(
        self,
        radius: float,
        E_1: float,
        nu_1: float,
        E_2: float = np.inf,
        nu_2: float = 0.0,
        radius_2: float = np.inf,
    ) -> None:
        if radius <= 0.0 or not np.isfinite(radius):
            raise ValueError("Radius must be positive and finite.")
        if E_1 <= 0.0 or not np.isfinite(E_1):
            raise ValueError("Young's modulus E_1 must be positive and finite.")
        if not (-1.0 < nu_1 < 0.5):
            raise ValueError("Poisson's ratio nu_1 must be in (-1, 0.5).")

        self.radius_1 = float(radius)
        self.E_1 = float(E_1)
        self.nu_1 = float(nu_1)
        self.E_2 = float(E_2)
        self.nu_2 = float(nu_2)
        self.radius_2 = float(radius_2)

        # Effective equivalent contact modulus E*
        term1 = (1.0 - self.nu_1**2) / self.E_1
        term2 = (1.0 - self.nu_2**2) / self.E_2 if np.isfinite(self.E_2) else 0.0
        self.E_star = 1.0 / (term1 + term2)

        # Effective equivalent contact curvature radius R*
        inv_R1 = 1.0 / self.radius_1
        inv_R2 = 1.0 / self.radius_2 if np.isfinite(self.radius_2) else 0.0
        self.R_star = 1.0 / (inv_R1 + inv_R2)

    def contact_radius(self, F_N: float) -> float:
        """Contact circle radius a [length] as a function of total normal force F_N."""
        if F_N <= 0.0:
            return 0.0
        return float((3.0 * F_N * self.R_star / (4.0 * self.E_star)) ** (1.0 / 3.0))

    def contact_radius_from_penetration(self, delta: float) -> float:
        """Contact circle radius a [length] from normal approach/penetration delta."""
        if delta <= 0.0:
            return 0.0
        return float(np.sqrt(self.R_star * delta))

    def penetration(self, F_N: float) -> float:
        """Normal approach/penetration depth delta [length] from normal force F_N."""
        if F_N <= 0.0:
            return 0.0
        a = self.contact_radius(F_N)
        return float(a**2 / self.R_star)

    def force_from_penetration(self, delta: float) -> float:
        """Total normal contact force F_N [force] as a function of penetration delta."""
        if delta <= 0.0:
            return 0.0
        return float((4.0 / 3.0) * self.E_star * np.sqrt(self.R_star) * (delta**1.5))

    def peak_pressure(self, F_N: float) -> float:
        """Maximum normal contact pressure p0 [force/length^2] at r = 0."""
        if F_N <= 0.0:
            return 0.0
        a = self.contact_radius(F_N)
        return float(3.0 * F_N / (2.0 * np.pi * a**2))

    def contact_area(self, F_N: float) -> float:
        """Geometric contact area A = pi * a^2 [length^2]."""
        a = self.contact_radius(F_N)
        return float(np.pi * a**2)

    def pressure_profile(
        self, r: float | np.ndarray, F_N: float
    ) -> float | np.ndarray:
        """Hemispherical radial pressure profile p(r) = p0 * sqrt(1 - (r/a)^2)."""
        if F_N <= 0.0:
            return np.zeros_like(r) if isinstance(r, np.ndarray) else 0.0
        a = self.contact_radius(F_N)
        p0 = self.peak_pressure(F_N)
        r_arr = np.asarray(r, dtype=np.float64)
        radicand = np.maximum(0.0, 1.0 - (r_arr / a) ** 2)
        p = p0 * np.sqrt(radicand)
        return p if isinstance(r, np.ndarray) else float(p)

    def subsurface_stresses(
        self, z: float | np.ndarray, F_N: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Principal stresses along the contact centerline r = 0 as a function of depth z.

        Returns:
        --------
        sigma_z : np.ndarray
            Normal stress component sigma_z(z) <= 0.
        sigma_r : np.ndarray
            Transverse in-plane stress components sigma_r(z) = sigma_theta(z) <= 0.
        tau_max : np.ndarray
            Maximum principal shear stress tau_max = 0.5 * |sigma_z - sigma_r|.
            Peak occurs at depth z ~ 0.48 a with value ~ 0.31 p0.
        """
        a = self.contact_radius(F_N)
        p0 = self.peak_pressure(F_N)
        z_arr = np.asarray(z, dtype=np.float64)

        # Avoid zero division when computing arctan(a/z)
        z_safe = np.where(z_arr == 0.0, 1e-15, z_arr)
        zeta = z_arr / a
        factor = 1.0 / (1.0 + zeta**2)

        sigma_z = -p0 * factor
        sigma_r = -p0 * (
            (1.0 + self.nu_1) * (1.0 - (z_safe / a) * np.arctan(a / z_safe))
            - 0.5 * factor
        )
        tau_max = 0.5 * np.abs(sigma_z - sigma_r)
        return sigma_z, sigma_r, tau_max

    def evaluate_fit(
        self,
        radial_coords: np.ndarray,
        pressures: np.ndarray,
        F_N: float,
    ) -> dict[str, float]:
        """Compute statistical goodness-of-fit metrics comparing numerical vs analytical profile."""
        r_clean = np.asarray(radial_coords, dtype=np.float64).ravel()
        p_num = np.asarray(pressures, dtype=np.float64).ravel()
        p_ana = np.asarray(self.pressure_profile(r_clean, F_N)).ravel()

        p0 = self.peak_pressure(F_N)
        if p0 <= 0.0:
            return {"rmse": 0.0, "r2": 1.0, "max_relative_error": 0.0}

        rmse = float(np.sqrt(np.mean((p_num - p_ana) ** 2)))
        ss_tot = float(np.sum((p_ana - np.mean(p_ana)) ** 2))
        ss_res = float(np.sum((p_num - p_ana) ** 2))
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-14 else 1.0
        max_err = float(np.max(np.abs(p_num - p_ana)) / p0)

        return {
            "rmse": rmse,
            "r2": r2,
            "max_relative_error": max_err,
        }


class HertzCylinderContact:
    """Exact closed-form Hertzian contact theory for 2D/3D parallel cylindrical interfaces.

    Parameters:
    -----------
    radius : float
        Radius of primary cylinder R_1 [length].
    length : float
        Axial contact length L [length].
    E_1 : float
        Young's modulus of cylinder [force/length^2].
    nu_1 : float
        Poisson's ratio of cylinder [dimensionless].
    E_2 : float, default=np.inf
        Young's modulus of opposing body (default np.inf for rigid plane).
    nu_2 : float, default=0.0
        Poisson's ratio of opposing body.
    radius_2 : float, default=np.inf
        Radius of opposing cylinder (default np.inf for flat plane).
    """

    def __init__(
        self,
        radius: float,
        length: float,
        E_1: float,
        nu_1: float,
        E_2: float = np.inf,
        nu_2: float = 0.0,
        radius_2: float = np.inf,
    ) -> None:
        if radius <= 0.0 or not np.isfinite(radius):
            raise ValueError("Radius must be positive and finite.")
        if length <= 0.0 or not np.isfinite(length):
            raise ValueError("Length must be positive and finite.")
        if E_1 <= 0.0 or not np.isfinite(E_1):
            raise ValueError("Young's modulus E_1 must be positive and finite.")
        if not (-1.0 < nu_1 < 0.5):
            raise ValueError("Poisson's ratio nu_1 must be in (-1, 0.5).")

        self.radius_1 = float(radius)
        self.length = float(length)
        self.E_1 = float(E_1)
        self.nu_1 = float(nu_1)
        self.E_2 = float(E_2)
        self.nu_2 = float(nu_2)
        self.radius_2 = float(radius_2)

        term1 = (1.0 - self.nu_1**2) / self.E_1
        term2 = (1.0 - self.nu_2**2) / self.E_2 if np.isfinite(self.E_2) else 0.0
        self.E_star = 1.0 / (term1 + term2)

        inv_R1 = 1.0 / self.radius_1
        inv_R2 = 1.0 / self.radius_2 if np.isfinite(self.radius_2) else 0.0
        self.R_star = 1.0 / (inv_R1 + inv_R2)

    def half_width(self, F_N: float) -> float:
        """Contact strip half-width b [length] as a function of total normal force F_N."""
        if F_N <= 0.0:
            return 0.0
        return float(np.sqrt(4.0 * F_N * self.R_star / (np.pi * self.length * self.E_star)))

    def peak_pressure(self, F_N: float) -> float:
        """Maximum normal contact pressure p0 [force/length^2] along line x = 0."""
        if F_N <= 0.0:
            return 0.0
        b = self.half_width(F_N)
        return float(2.0 * F_N / (np.pi * b * self.length))

    def contact_area(self, F_N: float) -> float:
        """Total rectangular contact area A = 2 * b * L [length^2]."""
        b = self.half_width(F_N)
        return float(2.0 * b * self.length)

    def force_from_half_width(self, b: float) -> float:
        """Total normal force F_N from contact half-width b."""
        if b <= 0.0:
            return 0.0
        return float((np.pi * self.length * self.E_star * (b**2)) / (4.0 * self.R_star))

    def pressure_profile(
        self, x: float | np.ndarray, F_N: float
    ) -> float | np.ndarray:
        """Semi-elliptical transverse pressure profile p(x) = p0 * sqrt(1 - (x/b)^2)."""
        if F_N <= 0.0:
            return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        b = self.half_width(F_N)
        p0 = self.peak_pressure(F_N)
        x_arr = np.asarray(x, dtype=np.float64)
        radicand = np.maximum(0.0, 1.0 - (x_arr / b) ** 2)
        p = p0 * np.sqrt(radicand)
        return p if isinstance(x, np.ndarray) else float(p)


def compute_nodal_tributary_areas(
    surface: SurfaceMesh,
    n_total_nodes: int,
) -> np.ndarray:
    """Vectorized calculation of nodal tributary areas on a surface triangular mesh.

    For each triangle f = (v0, v1, v2), one third of the facet area is assigned
    to each incident vertex, mapped to global volumetric node indices.

    Parameters:
    -----------
    surface : SurfaceMesh
        Boundary surface representation providing elements, areas, and node_indices.
    n_total_nodes : int
        Total number of global nodes in the volumetric domain.

    Returns:
    --------
    tributary_areas : np.ndarray of shape (n_total_nodes,)
        Nodal surface tributary area A_i [length^2].
    """
    areas = np.zeros(int(n_total_nodes), dtype=np.float64)
    face_areas_third = surface.areas / 3.0
    has_global_map = surface.node_indices is not None

    for node_col in range(3):
        local_nodes = surface.elements[:, node_col]
        global_nodes = surface.node_indices[local_nodes] if has_global_map else local_nodes
        np.add.at(areas, global_nodes, face_areas_third)
    return areas


def reconstruct_nodal_pressures(
    f_contact: np.ndarray,
    tributary_areas: np.ndarray,
    normal_component: int = 2,
    direction: float = 1.0,
) -> np.ndarray:
    """Reconstruct normal contact pressure field p_i = f_c,i / A_i from discrete forces.

    Parameters:
    -----------
    f_contact : np.ndarray of shape (3 * N,) or (N, 3)
        Global contact force vector.
    tributary_areas : np.ndarray of shape (N,)
        Surface nodal tributary areas.
    normal_component : int, default=2
        Coordinate index for normal contact direction (0: x, 1: y, 2: z).
    direction : float, default=1.0
        Sign multiplier: +1.0 for upward repulsive normal force (along +z),
        -1.0 if force points along -z.

    Returns:
    --------
    pressures : np.ndarray of shape (N,)
        Continuous reconstructed normal contact pressure field p_i >= 0.
    """
    f_reshaped = np.asarray(f_contact, dtype=np.float64).reshape(-1, 3)
    f_normal = direction * f_reshaped[:, normal_component]
    areas = np.asarray(tributary_areas, dtype=np.float64).reshape(-1)

    # Compute pressure strictly on surface nodes with non-zero area
    pressures = np.zeros(len(areas), dtype=np.float64)
    valid_mask = (areas > 1e-14) & (f_normal > 0.0)
    pressures[valid_mask] = f_normal[valid_mask] / areas[valid_mask]
    return pressures
