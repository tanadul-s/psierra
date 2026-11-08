"""
Project Sierra - High-Performance Finite Element Structural Analysis Package.
"""

from psierra.multibody.system import MultiBodySystem
from psierra.multibody import MultiDomain
from psierra.boundary.contact import (
    RigidContact,
    DeformableContact,
    SelfContact,
)
from psierra.collision import (
    CollisionShape,
    RigidPlane,
    RigidSphere,
    RigidCylinder,
    RayMeshObstacle,
    CollisionDetector,
)
from psierra.reduction import (
    PODBasis,
    SurfacePODBasis,
    SnapshotManager,
    ReducedLinearModel,
    ReducedLinearDynamicSolver,
    ReducedLinearStaticSolver,
    ReducedSolverStats,
)
from psierra.hyperreduction import (
    ReducedIntegrationDomain,
    ECSWTrainingMatrix,
    ECSWOptimizer,
    HyperReducedNonlinearModel,
    HyperReducedNonlinearStaticSolver,
    HyperReducedNonlinearDynamicSolver,
    HyperReducedSolverStats,
    HyperReducedRigidContact,
    ContactTelemetry,
)
from psierra.inverse import (
    DiscreteSensor,
    SensorObservationOperator,
    ReducedSensorObservationOperator,
    ShapeSensingResult,
    FullOrderShapeSensingSolver,
    ReducedOrderShapeSensingSolver,
    ParameterBounds,
    IdentificationResult,
    MaterialParameterIdentifier,
    StressFreeResult,
    StressFreeConfigurationSolver,
)

__version__ = "26.10"
__all__ = [
    "MultiBodySystem",
    "MultiDomain",
    "RigidContact",
    "DeformableContact",
    "SelfContact",
    "CollisionShape",
    "RigidPlane",
    "RigidSphere",
    "RigidCylinder",
    "RayMeshObstacle",
    "CollisionDetector",
    "PODBasis",
    "SurfacePODBasis",
    "SnapshotManager",
    "ReducedLinearModel",
    "ReducedLinearDynamicSolver",
    "ReducedLinearStaticSolver",
    "ReducedSolverStats",
    "ReducedIntegrationDomain",
    "ECSWTrainingMatrix",
    "ECSWOptimizer",
    "HyperReducedNonlinearModel",
    "HyperReducedNonlinearStaticSolver",
    "HyperReducedNonlinearDynamicSolver",
    "HyperReducedSolverStats",
    "HyperReducedRigidContact",
    "ContactTelemetry",
    "DiscreteSensor",
    "SensorObservationOperator",
    "ReducedSensorObservationOperator",
    "ShapeSensingResult",
    "FullOrderShapeSensingSolver",
    "ReducedOrderShapeSensingSolver",
    "ParameterBounds",
    "IdentificationResult",
    "MaterialParameterIdentifier",
    "StressFreeResult",
    "StressFreeConfigurationSolver",
]
