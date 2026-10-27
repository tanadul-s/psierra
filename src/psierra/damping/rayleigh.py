from __future__ import annotations
import scipy.sparse as spsp


class RayleighDamping:
    """
    Rayleigh Damping Model: C = alpha * M + beta * K.

    Parameters:
    -----------
    alpha : float, default=0.0
        Mass-proportional damping coefficient (damps low-frequency rigid/bending modes).
    beta : float, default=0.0
        Stiffness-proportional damping coefficient (damps high-frequency numerical oscillations).
    """

    def __init__(self, alpha: float = 0.0, beta: float = 0.0) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)

    def assemble(
        self, mass_matrix: spsp.csr_matrix, stiffness_matrix: spsp.csr_matrix
    ) -> spsp.csr_matrix:
        """Assembles the global damping matrix C."""
        C = self.alpha * mass_matrix + self.beta * stiffness_matrix
        return C.tocsr()
