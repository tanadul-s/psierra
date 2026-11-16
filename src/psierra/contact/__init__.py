"""Canonical internal contact models, geometry providers, constraint laws, and assemblers.

Milestone 1.0 Stages B & C of Loop 3 Contact Mechanics Overhaul.
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
from psierra.contact.law import NormalConstraintLaw
from psierra.contact.assembly import ContactAssembler

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
    "NormalConstraintLaw",
    "ContactAssembler",
]
