from __future__ import annotations
from typing import Literal
import numpy as np

from psierra.mesh.volumetric import VolumetricMesh


def select_nodes_by_plane(
    mesh: VolumetricMesh,
    plane: Literal["yz", "xz", "xy"],
    bound: Literal["min", "max"],
    tolerance: float = 1e-4,
) -> np.ndarray:
    """
    Selects nodes lying on an extremal bounding plane of the mesh.

    Parameters:
    -----------
    mesh : VolumetricMesh
        Volumetric mesh to query.
    plane : {'yz', 'xz', 'xy'}
        Plane normal direction ('yz' -> X axis, 'xz' -> Y axis, 'xy' -> Z axis).
    bound : {'min', 'max'}
        Whether to select the minimum or maximum face.
    tolerance : float, default=1e-4
        Distance tolerance around the plane coordinate.

    Returns:
    --------
    node_indices : np.ndarray of shape (N_selected,)
        Sorted array of matching node indices.
    """
    axis_map = {"yz": 0, "xz": 1, "xy": 2}
    axis = axis_map[plane]

    coords = mesh.nodes[:, axis]
    target_value = coords.min() if bound == "min" else coords.max()

    matching_nodes = np.where(np.abs(coords - target_value) <= tolerance)[0]
    return np.asarray(matching_nodes, dtype=np.int64)


def select_nodes_by_bounds(
    mesh: VolumetricMesh,
    bounds: tuple[float, float, float, float, float, float],
    surface_only: bool = False,
) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    nodes = mesh.nodes

    mask = (
        (nodes[:, 0] >= xmin)
        & (nodes[:, 0] <= xmax)
        & (nodes[:, 1] >= ymin)
        & (nodes[:, 1] <= ymax)
        & (nodes[:, 2] >= zmin)
        & (nodes[:, 2] <= zmax)
    )
    if surface_only:
        mask &= mesh.surface_node_mask  # Only keep nodes on the outer surface

    return np.where(mask)[0]
