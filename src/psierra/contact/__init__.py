"""Canonical internal contact models and geometry providers for Project Sierra.

Milestone 1.0 Stage B of Loop 3 Contact Mechanics Overhaul.
"""

from psierra.contact.data import (
    ContactCandidate,
    DifferentiableContactGeometry,
    NormalLawState,
    FrictionalConstraintState,
    ContactAssemblyMap,
)
from psierra.contact.geometry import (
    ContactGeometryProvider,
    PlanarGeometryProvider,
    SphereGeometryProvider,
    CylinderGeometryProvider,
    TriangleGeometryProvider,
)

__all__ = [
    "ContactCandidate",
    "DifferentiableContactGeometry",
    "NormalLawState",
    "FrictionalConstraintState",
    "ContactAssemblyMap",
    "ContactGeometryProvider",
    "PlanarGeometryProvider",
    "SphereGeometryProvider",
    "CylinderGeometryProvider",
    "TriangleGeometryProvider",
]
