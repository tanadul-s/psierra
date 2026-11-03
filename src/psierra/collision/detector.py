"""
Collision Detector & Proximity Queries for Project Sierra.
=========================================================
Coordinates broad-phase proximity filtering and narrow-phase surface projection
across analytical and mesh-based obstacles.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from psierra.collision.shapes import CollisionShape


@dataclass
class ContactQueryResult:
    """Container for contact and penetration query results."""

    gaps: np.ndarray  # (N,) signed gaps (negative = penetration)
    normals: np.ndarray  # (N, 3) outward unit normals
    closest_points: np.ndarray  # (N, 3) projection on obstacle surface
    obstacle_indices: np.ndarray  # (N,) index of closest obstacle
    active_mask: np.ndarray  # (N,) bool mask of points in contact (gap <= tolerance)
    active_indices: np.ndarray  # (M,) indices of contacting nodes

    @property
    def num_active(self) -> int:
        """Number of points currently in active contact."""
        return int(len(self.active_indices))

    @property
    def max_penetration(self) -> float:
        """Maximum penetration depth (strictly positive, 0.0 if no contact)."""
        if self.num_active == 0:
            return 0.0
        return float(np.max(-self.gaps[self.active_mask]))


class CollisionDetector:
    """
    Manages collision detection between candidate slave nodes and one or more obstacles.
    """

    def __init__(
        self,
        obstacles: CollisionShape | Sequence[CollisionShape],
    ) -> None:
        if isinstance(obstacles, CollisionShape):
            self._obstacles = [obstacles]
        else:
            self._obstacles = list(obstacles)

        if not self._obstacles:
            raise ValueError("CollisionDetector requires at least one obstacle.")

    @property
    def obstacles(self) -> list[CollisionShape]:
        """List of active collision shapes / obstacles."""
        return self._obstacles

    def query(
        self,
        points: np.ndarray,
        tolerance: float = 0.0,
    ) -> ContactQueryResult:
        """
        Queries all registered obstacles to identify active contacts and penetrations.

        Parameters:
        -----------
        points : np.ndarray of shape (N, 3)
            Current spatial coordinates of candidate contact nodes.
        tolerance : float, default=0.0
            Contact activation threshold (points with gap <= tolerance are active).

        Returns:
        --------
        result : ContactQueryResult
        """
        pts = np.asarray(points, dtype=np.float64)
        n_pts = pts.shape[0]

        best_gaps = np.full(n_pts, np.inf, dtype=np.float64)
        best_normals = np.zeros((n_pts, 3), dtype=np.float64)
        best_closest = np.zeros((n_pts, 3), dtype=np.float64)
        best_obs_idx = np.zeros(n_pts, dtype=np.int64)

        for obs_idx, obstacle in enumerate(self._obstacles):
            gaps, normals, closest = obstacle.query_points(pts)
            # Find the obstacle with the smallest signed gap (deepest penetration or closest)
            closer = gaps < best_gaps
            best_gaps[closer] = gaps[closer]
            best_normals[closer] = normals[closer]
            best_closest[closer] = closest[closer]
            best_obs_idx[closer] = obs_idx

        active_mask = best_gaps <= tolerance
        active_indices = np.flatnonzero(active_mask)

        return ContactQueryResult(
            gaps=best_gaps,
            normals=best_normals,
            closest_points=best_closest,
            obstacle_indices=best_obs_idx,
            active_mask=active_mask,
            active_indices=active_indices,
        )

