from __future__ import annotations

from psierra.analysis.base import Analysis
from psierra.analysis.linear_static import (
    LinearStaticAnalysis,
    LinearStaticResult,
)
from psierra.analysis.linear_dynamic import (
    LinearDynamicAnalysis,
    LinearDynamicResult,
)
from psierra.analysis.nonlinear_static import (
    NonlinearStaticAnalysis,
    NonlinearStaticResult,
)
from psierra.analysis.nonlinear_dynamic import (
    NonlinearDynamicAnalysis,
    NonlinearDynamicResult,
)
from psierra.analysis.adaptive_time_step import AdaptiveTimeStepController
from psierra.analysis.line_search import (
    ContactAwareLineSearch,
    LineSearchResult,
    is_element_inversion_error,
)
from psierra.analysis.interactive_dynamic import (
    InteractiveDynamicSimulation,
    SimulationStats,
)
from psierra.analysis.dynamic_relaxation import (
    DynamicRelaxationAnalysis,
    DynamicRelaxationResult,
)
from psierra.analysis.quasi_static import (
    QuasiStaticAnalysis,
    QuasiStaticResult,
)

__all__ = [
    # Base
    "Analysis",
    # Linear Static
    "LinearStaticAnalysis",
    "LinearStaticResult",
    # Linear Dynamic
    "LinearDynamicAnalysis",
    "LinearDynamicResult",
    # Nonlinear Static
    "NonlinearStaticAnalysis",
    "NonlinearStaticResult",
    # Quasi-Static
    "QuasiStaticAnalysis",
    "QuasiStaticResult",
    # Nonlinear Dynamic
    "NonlinearDynamicAnalysis",
    "NonlinearDynamicResult",
    "AdaptiveTimeStepController",
    # Algorithmic Regularization
    "ContactAwareLineSearch",
    "LineSearchResult",
    "is_element_inversion_error",
    # Interactive / Online Dynamic
    "InteractiveDynamicSimulation",
    "SimulationStats",
    # Dynamic Relaxation
    "DynamicRelaxationAnalysis",
    "DynamicRelaxationResult",
]
