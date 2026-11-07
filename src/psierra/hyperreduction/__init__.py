"""
Energy-Conserving Sampling and Weighting (ECSW) Hyper-Reduction Framework.

Provides:
- ReducedIntegrationDomain: Sparse cubature mesh sub-domain and positive weights.
- ECSWTrainingMatrix: Elemental virtual work training matrix builder.
- ECSWOptimizer: Greedy active-set Non-Negative Least Squares (NNLS) cubature selector.
- HyperReducedNonlinearModel: ECSW hyper-reduced mechanical continuum model.
- HyperReducedNonlinearStaticSolver: Dense r x r Newton-Raphson static equilibrium solver.
- HyperReducedNonlinearDynamicSolver: Real-time Generalized-alpha nonlinear dynamic solver.
- HyperReducedSolverStats: Execution performance and energy telemetry container.
"""

from psierra.hyperreduction.ecsw import (
    ReducedIntegrationDomain,
    ECSWTrainingMatrix,
    ECSWOptimizer,
)
from psierra.hyperreduction.model import HyperReducedNonlinearModel
from psierra.hyperreduction.solver import (
    HyperReducedNonlinearStaticSolver,
    HyperReducedNonlinearDynamicSolver,
    HyperReducedSolverStats,
)

__all__ = [
    "ReducedIntegrationDomain",
    "ECSWTrainingMatrix",
    "ECSWOptimizer",
    "HyperReducedNonlinearModel",
    "HyperReducedNonlinearStaticSolver",
    "HyperReducedNonlinearDynamicSolver",
    "HyperReducedSolverStats",
]
