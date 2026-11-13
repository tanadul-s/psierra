"""
Hyper-Reduction Framework (ECSW & DEIM).

Provides:
- ReducedIntegrationDomain: Sparse cubature mesh sub-domain and positive weights.
- ECSWTrainingMatrix: Elemental virtual work training matrix builder.
- ECSWOptimizer: Greedy active-set Non-Negative Least Squares (NNLS) cubature selector.
- HyperReducedECSWModel: ECSW empirical cubature mechanical continuum model.
- HyperReducedDEIMModel: Discrete Empirical Interpolation Method (DEIM) mechanical continuum model.
- ContactAwareHyperReducedModel: Contact-aware cubature mechanical continuum model.
- HyperReducedStaticSolver: Dense r x r Newton-Raphson static equilibrium solver.
- HyperReducedDynamicSolver: Real-time Generalized-alpha nonlinear dynamic solver.
- HyperReducedQuasiStaticSolver: Rate-dependent viscoelastic quasi-static solver.
- HyperReducedDynamicRelaxationSolver: Explicit kinetic energy peak-damping solver.
- HyperReducedSolverStats: Execution performance and energy telemetry container.
"""

from psierra.hyperreduction.contact import (
    HyperReducedRigidContact,
    ContactTelemetry,
)
from psierra.hyperreduction.ecsw import (
    ReducedIntegrationDomain,
    ECSWTrainingMatrix,
    ECSWOptimizer,
)
from psierra.hyperreduction.contact_ecsw import (
    ContactAwareReducedIntegrationDomain,
    ContactAwareECSWTrainingMatrix,
    ContactAwareECSWOptimizer,
    ContactAwareHyperReducedModel,
)
from psierra.hyperreduction.deim import (
    greedy_deim,
    q_deim,
    DEIMForceBasis,
    ForceSnapshotManager,
    HyperReducedDEIMModel,
)
from psierra.hyperreduction.model import HyperReducedECSWModel
from psierra.hyperreduction.solver import (
    HyperReducedStaticSolver,
    HyperReducedDynamicSolver,
    HyperReducedDynamicRelaxationSolver,
    HyperReducedQuasiStaticSolver,
    HyperReducedSolverStats,
    HyperReducedDynamicRelaxationStats,
)

__all__ = [
    "ReducedIntegrationDomain",
    "ECSWTrainingMatrix",
    "ECSWOptimizer",
    "ContactAwareReducedIntegrationDomain",
    "ContactAwareECSWTrainingMatrix",
    "ContactAwareECSWOptimizer",
    "ContactAwareHyperReducedModel",
    "greedy_deim",
    "q_deim",
    "DEIMForceBasis",
    "ForceSnapshotManager",
    "HyperReducedECSWModel",
    "HyperReducedDEIMModel",
    "HyperReducedStaticSolver",
    "HyperReducedDynamicSolver",
    "HyperReducedDynamicRelaxationSolver",
    "HyperReducedQuasiStaticSolver",
    "HyperReducedSolverStats",
    "HyperReducedDynamicRelaxationStats",
    "HyperReducedRigidContact",
    "ContactTelemetry",
]
