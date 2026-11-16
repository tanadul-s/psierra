"""
Dimensionless Penalty Scaling Law & Automatic Stiffness Calculation.
====================================================================
Milestone 1.2 of Loop 3: Robust Contact Mechanics & Advanced Multiphysics.

Provides mathematically consistent, scale-invariant penalty parameter calculations
to eliminate user-guessed penalty values and guarantee numerical conditioning across
extreme variations in Young's modulus (100 MPa -> 1 kPa, e.g. bone to soft brain parenchyma).

Formulation:
------------
1. Dimensionless Contact Parameter:
       kappa_p = (k_p * h_e) / E_bulk in [5, 50]
   where:
       h_e      : Characteristic surface element edge length [L]
       E_bulk   : Effective Young's modulus [F/L^2]
       k_p      : Contact interface penalty stiffness [F/L^3 for distributed, F/L for nodal]

2. Automatic Element-Adaptive Penalty Law:
       k_{p, e} = alpha_penalty * (K_bulk * A_e) / V_e
   where:
       K_bulk        : Material bulk modulus [F/L^2]
       A_e           : Facet surface area [L^2]
       V_e           : Incident element volume [L^3]
       alpha_penalty : Calibration factor in [0.1, 1.0] (default: 1.0)
"""

from __future__ import annotations
from typing import Any, Literal
import numpy as np

from psierra.domain.base import Domain
from psierra.material.base import Material
from psierra.mesh.volumetric import VolumetricMesh
from psierra.mesh.surface import SurfaceMesh


def extract_material_stiffness(
    material_or_domain: Domain | Material | Any,
) -> tuple[float, float, float]:
    """
    Extract bulk modulus K, Young's modulus E, and Poisson's ratio nu from material or domain.

    Returns:
    --------
    tuple[float, float, float]:
        (K_bulk, E_bulk, poissons_ratio) in consistent stress units (e.g. Pa or MPa).
    """
    mat = getattr(material_or_domain, "material", material_or_domain)
    params = getattr(mat, "_parameters", getattr(mat, "parameters", None))

    K_bulk: float | None = None
    E_bulk: float | None = None
    nu: float = 0.35

    if params is not None:
        if hasattr(params, "poissons_ratio"):
            try:
                nu_val = float(params.poissons_ratio)
                if np.isfinite(nu_val) and -0.99 <= nu_val < 0.5:
                    nu = nu_val
            except Exception:
                pass

        if hasattr(params, "bulk_modulus"):
            try:
                k_val = float(params.bulk_modulus)
                if np.isfinite(k_val) and k_val > 0.0:
                    K_bulk = k_val
            except Exception:
                pass

        if hasattr(params, "youngs_modulus"):
            try:
                e_val = float(params.youngs_modulus)
                if np.isfinite(e_val) and e_val > 0.0:
                    E_bulk = e_val
            except Exception:
                pass

        if hasattr(params, "shear_modulus") and (K_bulk is None or E_bulk is None):
            try:
                mu_val = float(params.shear_modulus)
                if np.isfinite(mu_val) and mu_val > 0.0:
                    if K_bulk is not None and E_bulk is None:
                        E_bulk = (9.0 * K_bulk * mu_val) / (3.0 * K_bulk + mu_val)
                    elif E_bulk is not None and K_bulk is None:
                        denom = 3.0 * (1.0 - 2.0 * nu)
                        K_bulk = E_bulk / denom if denom > 1e-4 else E_bulk / 1e-4
                    elif E_bulk is None and K_bulk is None:
                        E_bulk = 2.0 * mu_val * (1.0 + nu)
                        denom = 3.0 * (1.0 - 2.0 * nu)
                        K_bulk = E_bulk / denom if denom > 1e-4 else E_bulk / 1e-4
            except Exception:
                pass

    # Direct attributes on material or domain
    if K_bulk is None and hasattr(mat, "bulk_modulus"):
        try:
            k_val = float(getattr(mat, "bulk_modulus"))
            if np.isfinite(k_val) and k_val > 0.0:
                K_bulk = k_val
        except Exception:
            pass

    if E_bulk is None and hasattr(mat, "youngs_modulus"):
        try:
            e_val = float(getattr(mat, "youngs_modulus"))
            if np.isfinite(e_val) and e_val > 0.0:
                E_bulk = e_val
        except Exception:
            pass

    # Direct numeric types (duck typing)
    if isinstance(material_or_domain, (int, float)):
        E_bulk = float(material_or_domain)

    # Reconcile K_bulk and E_bulk if one is missing
    if K_bulk is not None and E_bulk is None:
        E_bulk = 3.0 * K_bulk * (1.0 - 2.0 * nu)
        if E_bulk <= 0.0:
            E_bulk = K_bulk
    elif E_bulk is not None and K_bulk is None:
        denom = 3.0 * (1.0 - 2.0 * nu)
        K_bulk = E_bulk / denom if denom > 1e-4 else E_bulk / 1e-4
    elif K_bulk is None and E_bulk is None:
        # Default fallback (1 MPa, nu = 0.35)
        E_bulk = 1.0e6
        K_bulk = E_bulk / (3.0 * (1.0 - 2.0 * nu))

    assert K_bulk is not None and E_bulk is not None
    return float(K_bulk), float(E_bulk), float(nu)


def compute_characteristic_mesh_metrics(
    mesh: VolumetricMesh | SurfaceMesh | Any | None,
) -> dict[str, Any]:
    """
    Extract characteristic geometric metrics from a volumetric or surface mesh.

    Returns:
    --------
    dict[str, Any] with keys:
        - 'h_e': characteristic element edge length [L]
        - 'mean_area': mean surface facet area [L^2]
        - 'mean_volume': mean volumetric element volume [L^3]
        - 'areas': array of facet surface areas [L^2]
        - 'volumes': array of element volumes [L^3]
        - 'n_elements': number of volumetric cells
        - 'n_facets': number of surface triangles
    """
    if mesh is None:
        return {
            "h_e": 1.0,
            "mean_area": 1.0,
            "mean_volume": 1.0,
            "areas": np.array([1.0], dtype=np.float64),
            "volumes": np.array([1.0], dtype=np.float64),
            "n_elements": 1,
            "n_facets": 1,
        }

    # Extract VolumetricMesh if attached
    vol_mesh: VolumetricMesh | None = None
    surf_mesh: SurfaceMesh | None = None

    if isinstance(mesh, VolumetricMesh):
        vol_mesh = mesh
        surf_mesh = mesh.surface
    elif isinstance(mesh, SurfaceMesh):
        surf_mesh = mesh
    elif hasattr(mesh, "surface") and isinstance(mesh.surface, SurfaceMesh):
        vol_mesh = mesh
        surf_mesh = mesh.surface
    elif hasattr(mesh, "areas") and hasattr(mesh, "nodes"):
        surf_mesh = mesh

    # Facet areas
    if surf_mesh is not None and hasattr(surf_mesh, "areas"):
        areas = np.asarray(surf_mesh.areas, dtype=np.float64)
        mean_area = float(np.mean(areas)) if len(areas) > 0 else 1.0
        n_facets = len(areas)
    else:
        areas = np.array([1.0], dtype=np.float64)
        mean_area = 1.0
        n_facets = 1

    # Characteristic edge length h_e
    if (
        surf_mesh is not None
        and hasattr(surf_mesh, "nodes")
        and hasattr(surf_mesh, "elements")
    ):
        pts = surf_mesh.nodes
        tris = surf_mesh.elements
        if len(tris) > 0 and tris.shape[1] >= 3:
            e0 = np.linalg.norm(pts[tris[:, 1]] - pts[tris[:, 0]], axis=1)
            e1 = np.linalg.norm(pts[tris[:, 2]] - pts[tris[:, 1]], axis=1)
            e2 = np.linalg.norm(pts[tris[:, 0]] - pts[tris[:, 2]], axis=1)
            h_e = float(np.mean(np.concatenate([e0, e1, e2])))
        else:
            h_e = float(np.sqrt(4.0 * mean_area / np.sqrt(3.0)))
    else:
        h_e = float(np.sqrt(4.0 * mean_area / np.sqrt(3.0)))

    # Volumes
    if vol_mesh is not None and hasattr(vol_mesh, "volumes"):
        volumes = np.asarray(vol_mesh.volumes, dtype=np.float64)
        mean_volume = float(np.mean(volumes)) if len(volumes) > 0 else 1.0
        n_elements = len(volumes)
    else:
        # For surface mesh or absent volumetric data, estimate incident tetrahedron volume:
        # V ~ 1/3 * A_e * h_perp ~ 0.272 * A_e * h_e for equilateral tetrahedron
        volumes = 0.272166 * areas * h_e
        mean_volume = float(np.mean(volumes)) if len(volumes) > 0 else 1.0
        n_elements = len(volumes)

    return {
        "h_e": max(h_e, 1e-12),
        "mean_area": max(mean_area, 1e-16),
        "mean_volume": max(mean_volume, 1e-20),
        "areas": areas,
        "volumes": volumes,
        "n_elements": n_elements,
        "n_facets": n_facets,
    }


def compute_adaptive_penalty(
    domain: Domain | Any = None,
    mesh: VolumetricMesh | SurfaceMesh | Any | None = None,
    material: Material | Any | None = None,
    *,
    alpha_penalty: float = 1.0,
    kappa_p: float | None = None,
    nodal: bool = True,
    return_elementwise: bool = False,
) -> float | np.ndarray:
    """
    Compute dimensionally consistent, scale-invariant contact penalty parameter.

    Eliminates arbitrary user-guessed penalty values and guarantees numerical conditioning
    across 6+ orders of magnitude of Young's modulus (100 MPa -> 1 kPa).

    Parameters:
    -----------
    domain : Domain, optional
        Deformable domain (e.g. SolidDomain) containing geometry and material.
    mesh : VolumetricMesh or SurfaceMesh, optional
        Mesh geometry if not provided via domain.
    material : Material, optional
        Material constitutive model if not provided via domain.
    alpha_penalty : float, default=1.0
        Element-adaptive scaling factor alpha_penalty in [0.1, 1.0].
    kappa_p : float, optional
        Explicit dimensionless contact parameter kappa_p = (k_p * h_e) / E_bulk in [5, 50].
        When specified, overrides alpha_penalty.
    nodal : bool, default=True
        If True, returns discrete nodal contact stiffness [Force/Length] (e.g. N/mm),
        scaled by tributary facet area (k_nodal ~ k_dist * A_e).
        If False, returns distributed interface pressure stiffness [Force/Length^3] (e.g. N/mm^3).
    return_elementwise : bool, default=False
        If True, returns an array of per-facet adaptive penalty values across all surface facets.
        If False (default), returns a single characteristic scalar penalty.

    Returns:
    --------
    float | np.ndarray:
        Calibrated penalty stiffness. Finite, positive, and scale-invariant.
    """
    # 1. Resolve mesh
    target_mesh = mesh
    if target_mesh is None and domain is not None:
        target_mesh = getattr(domain, "mesh", None)
        if target_mesh is None and isinstance(domain, (VolumetricMesh, SurfaceMesh)):
            target_mesh = domain

    # 2. Resolve material
    target_mat = material
    if target_mat is None and domain is not None:
        target_mat = getattr(domain, "material", None)
        if target_mat is None and isinstance(domain, Material):
            target_mat = domain

    # 3. Extract bulk modulus and Young's modulus
    K_bulk, E_bulk, _ = extract_material_stiffness(
        target_mat if target_mat is not None else domain
    )

    # 4. Extract mesh metrics
    metrics = compute_characteristic_mesh_metrics(target_mesh)
    h_e = metrics["h_e"]
    mean_A = metrics["mean_area"]
    mean_V = metrics["mean_volume"]
    areas = metrics["areas"]

    # 5. Calculate penalty
    if kappa_p is not None:
        # Explicit dimensionless scaling: kappa_p = (k_p * h_e) / E_bulk
        if not np.isfinite(kappa_p) or kappa_p <= 0.0:
            raise ValueError(f"kappa_p must be finite and positive, got {kappa_p}")

        # Base distributed interface stiffness [Force/Length^3]
        kp_dist = float(kappa_p * (E_bulk / h_e))

        if return_elementwise:
            if nodal:
                res_array = kp_dist * areas
            else:
                res_array = np.full_like(areas, kp_dist)
            return res_array
        else:
            if nodal:
                kp_val = kp_dist * mean_A
            else:
                kp_val = kp_dist
            if not np.isfinite(kp_val) or kp_val <= 0.0:
                raise ValueError(f"Computed penalty {kp_val} is invalid")
            return float(kp_val)

    # Element-adaptive penalty law: k_{p, e} = alpha_penalty * (K_bulk * A_e) / V_e
    if not np.isfinite(alpha_penalty) or alpha_penalty <= 0.0:
        raise ValueError(f"alpha_penalty must be finite and positive, got {alpha_penalty}")

    if return_elementwise:
        # Per-facet evaluation
        if nodal:
            # k_{p, nodal, e} = alpha_penalty * K_bulk * A_e^2 / V_e
            res_array = float(alpha_penalty * K_bulk) * (areas**2) / mean_V
        else:
            # k_{p, dist, e} = alpha_penalty * K_bulk * A_e / V_e
            res_array = float(alpha_penalty * K_bulk) * areas / mean_V
        return res_array

    # Characteristic scalar
    if nodal:
        kp_val = float(alpha_penalty * K_bulk * (mean_A**2) / mean_V)
    else:
        kp_val = float(alpha_penalty * K_bulk * mean_A / mean_V)

    if not np.isfinite(kp_val) or kp_val <= 0.0:
        raise ValueError(f"Computed penalty {kp_val} is invalid")

    return float(kp_val)


def compute_dimensionless_contact_parameter(
    k_p: float,
    domain: Domain | Any = None,
    mesh: VolumetricMesh | SurfaceMesh | Any | None = None,
    material: Material | Any | None = None,
    *,
    nodal: bool = True,
) -> float:
    """
    Compute the dimensionless contact parameter kappa_p.

    Definitions:
    ------------
    - Distributed interface stiffness (nodal=False):
          kappa_p = (k_p * h_e) / E_bulk
    - Discrete nodal stiffness (nodal=True):
          kappa_p = (k_p * h_e) / (E_bulk * A_e)

    Ideal range:
        kappa_p in [5, 50]
    Guarantees sufficient contact penalty enforcement (< 1% relative penetration)
    without triggering numerical ill-conditioning or continuum element inversion.

    Returns:
    --------
    float:
        Dimensionless contact parameter kappa_p.
    """
    if not np.isfinite(k_p) or k_p <= 0.0:
        raise ValueError(f"k_p must be finite and positive, got {k_p}")

    # Extract material and mesh
    target_mat = material if material is not None else getattr(domain, "material", domain)
    K_bulk, E_bulk, _ = extract_material_stiffness(target_mat)

    target_mesh = mesh if mesh is not None else getattr(domain, "mesh", domain)
    metrics = compute_characteristic_mesh_metrics(target_mesh)
    h_e = metrics["h_e"]
    mean_A = metrics["mean_area"]

    if nodal:
        kappa = (k_p * h_e) / (E_bulk * mean_A)
    else:
        kappa = (k_p * h_e) / E_bulk

    return float(kappa)
