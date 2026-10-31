from __future__ import annotations
import numpy as np


def get_tetrahedral_quadrature(points: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns Gaussian / Keast numerical quadrature points and integration weights
    for the master tetrahedron:
        Omega_ref = {(xi, eta, zeta) : xi >= 0, eta >= 0, zeta >= 0, xi + eta + zeta <= 1}

    Weights are scaled to integrate directly over the master tetrahedron whose
    reference volume is V_ref = 1/6:
        int_{Omega_ref} f(xi) dV_xi = sum_{q=1}^{N_qp} w_q * f(xi_q)
    such that sum_{q=1}^{N_qp} w_q = 1/6.

    Parameters:
    -----------
    points : int, default=4
        Number of quadrature integration points. Supported options:
        - 1: 1-point centroid rule (degree 1 algebraic precision)
        - 4: 4-point Hammer-Marlowe-Stroud rule (degree 2 algebraic precision, exact for Tet10 linear elasticity)
        - 5: 5-point Hammer-Marlowe-Stroud rule (degree 3 algebraic precision, exact for cubics)

    Returns:
    --------
    qp : np.ndarray of shape (points, 3)
        Quadrature point coordinates (xi, eta, zeta).
    qw : np.ndarray of shape (points,)
        Quadrature weights summing to 1/6.
    """
    if points == 1:
        qp = np.array([[0.25, 0.25, 0.25]], dtype=np.float64)
        qw = np.array([1.0 / 6.0], dtype=np.float64)
        return qp, qw

    if points == 4:
        # Standard symmetric 4-point rule (Degree 2 exact)
        a = (5.0 - np.sqrt(5.0)) / 20.0
        b = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
        qp = np.array(
            [
                [a, a, a],
                [b, a, a],
                [a, b, a],
                [a, a, b],
            ],
            dtype=np.float64,
        )
        qw = np.full(4, 1.0 / 24.0, dtype=np.float64)
        return qp, qw

    if points == 5:
        # 5-point rule (Degree 3 exact)
        qp = np.array(
            [
                [0.25, 0.25, 0.25],
                [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
                [0.5, 1.0 / 6.0, 1.0 / 6.0],
                [1.0 / 6.0, 0.5, 1.0 / 6.0],
                [1.0 / 6.0, 1.0 / 6.0, 0.5],
            ],
            dtype=np.float64,
        )
        qw = np.array(
            [-2.0 / 15.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0],
            dtype=np.float64,
        )
        return qp, qw

    raise ValueError(
        f"Unsupported tetrahedral quadrature points: {points}. Supported: 1, 4, 5."
    )

