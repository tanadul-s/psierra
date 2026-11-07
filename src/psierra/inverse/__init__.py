"""
Inverse Finite Element Methods (iFEM) and Parameter Identification Framework.

Provides:
- DiscreteSensor: Discrete spatial sensor definition and barycentric weights.
- SensorObservationOperator: Full-order sparse observation operator H in R^(3*N_sensor x n_dof).
- ReducedSensorObservationOperator: Microsecond-rate reduced observation operator H_r in R^(3*N_sensor x r).
- ShapeSensingResult: Full-field displacement reconstruction telemetry and metrics.
- FullOrderShapeSensingSolver: High-fidelity variational iFEM solver (sparse SPD system).
- ReducedOrderShapeSensingSolver: Microsecond-rate ROM variational iFEM solver (dense SPD system).
- ParameterBounds: Boundary constraints and initial values for material parameter calibration.
- IdentificationResult: Convergence diagnostics, calibrated parameters, and error metrics.
- MaterialParameterIdentifier: Nonlinear least-squares optimizer for constitutive parameter calibration.
"""

from psierra.inverse.observation import (
    DiscreteSensor,
    SensorObservationOperator,
    ReducedSensorObservationOperator,
)
from psierra.inverse.shape_sensing import (
    ShapeSensingResult,
    FullOrderShapeSensingSolver,
    ReducedOrderShapeSensingSolver,
)
from psierra.inverse.parameter_identification import (
    ParameterBounds,
    IdentificationResult,
    MaterialParameterIdentifier,
)

__all__ = [
    "DiscreteSensor",
    "SensorObservationOperator",
    "ReducedSensorObservationOperator",
    "ShapeSensingResult",
    "FullOrderShapeSensingSolver",
    "ReducedOrderShapeSensingSolver",
    "ParameterBounds",
    "IdentificationResult",
    "MaterialParameterIdentifier",
]
