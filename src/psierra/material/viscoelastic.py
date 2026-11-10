"""
Viscoelastic Constitutive Laws (Generalized Maxwell / Prony Series Formulation).

Formulates finite-strain and small-strain viscoelasticity via a discrete spectrum
of Maxwell relaxation branches wrapping an arbitrary hyperelastic or elastic base model:
    S(t) = S_inf(t) + sum_{m=1}^{Np} Q_m(t)
    g(t) = g_inf + sum_{m=1}^{Np} g_m * exp(-t / tau_m)
where g_inf + sum_{m=1}^{Np} g_m = 1.0.

Provides recursive exponential time integration (Simo 1987, Holzapfel 2000)
with exact algorithmic tangent modulus C_{n+1} = eta(dt) * C0_{n+1}.
"""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np

from psierra.material.base import Material
from psierra.material.parameters import IsotropicElasticParameters


@dataclass(frozen=True)
class PronyTerm:
    """
    A single Maxwell branch in the discrete Prony series relaxation spectrum.

    Parameters:
    -----------
    relative_modulus : float
        Normalized relaxation factor g_m = G_m / G_0 in (0, 1).
    relaxation_time : float
        Characteristic relaxation time constant tau_m in seconds (> 0).
    """

    relative_modulus: float
    relaxation_time: float

    def __post_init__(self) -> None:
        g = float(self.relative_modulus)
        tau = float(self.relaxation_time)
        if not (0.0 < g < 1.0):
            raise ValueError(
                f"Prony relative modulus g_m must satisfy 0 < g_m < 1, got {g}"
            )
        if tau <= 0.0:
            raise ValueError(
                f"Prony relaxation time tau_m must be strictly positive, got {tau}"
            )


class Viscoelastic(Material):
    """
    Generalized Maxwell / Prony Series Finite-Strain Viscoelastic Material.

    Wraps an instantaneous elastic or hyperelastic base material (e.g. NeoHookean,
    Hookean, Yeoh, Mooney-Rivlin) and models time-dependent stress relaxation,
    creep, and rate-dependent hysteretic energy dissipation.

    Governing Equations:
    --------------------
    Instantaneous 2nd Piola-Kirchhoff stress:
        S^0(F) = 2 * dPsi_0 / dC
    Viscoelastic stress convolution:
        S(t) = g_inf * S^0(t) + sum_{m=1}^{Np} Q_m(t)
    where Q_m satisfies:
        d/dt(Q_m) + Q_m / tau_m = g_m * d/dt(S^0)

    Algorithmic Update over [t_n, t_{n+1}] with time step dt:
    ----------------------------------------------------------
        gamma_m = exp(-dt / tau_m)
        beta_m  = (tau_m / dt) * (1 - gamma_m)   [lim_{dt -> 0} beta_m = 1.0]
        Q_{m, n+1} = gamma_m * Q_{m, n} + g_m * beta_m * (S^0_{n+1} - S^0_n)
        S_{n+1} = eta(dt) * S^0_{n+1} + H_n
        where eta(dt) = g_inf + sum_{m=1}^{Np} g_m * beta_m
              H_n = sum_{m=1}^{Np} (gamma_m * Q_{m, n} - g_m * beta_m * S^0_n)
        C_{n+1} = eta(dt) * C0_{n+1}
    """

    supports_element_indexing: bool = True

    def __init__(
        self,
        base_material: Material,
        prony_series: Sequence[PronyTerm | tuple[float, float]],
        deviatoric_only: bool = False,
    ) -> None:
        """
        Parameters:
        -----------
        base_material : Material
            Underlying instantaneous elastic or hyperelastic material model.
        prony_series : Sequence of PronyTerm or (g_m, tau_m) tuples
            Prony relaxation spectrum components.
        deviatoric_only : bool, default=False
            If True, viscoelastic relaxation acts only on the isochoric stress,
            preserving an undamped elastic bulk response.
        """
        if not isinstance(base_material, Material):
            raise TypeError(
                f"base_material must be an instance of Material, got {type(base_material)}"
            )
        self._base_material = base_material
        self._deviatoric_only = bool(deviatoric_only)

        # Parse and validate Prony series terms
        parsed_terms: list[PronyTerm] = []
        for term in prony_series:
            if isinstance(term, PronyTerm):
                parsed_terms.append(term)
            elif isinstance(term, (tuple, list)) and len(term) == 2:
                parsed_terms.append(PronyTerm(float(term[0]), float(term[1])))
            else:
                raise TypeError(
                    f"Expected PronyTerm or (g_m, tau_m) tuple, got {term}"
                )

        if not parsed_terms:
            raise ValueError("prony_series must contain at least one relaxation term.")

        sum_g = sum(t.relative_modulus for t in parsed_terms)
        if sum_g >= 1.0:
            raise ValueError(
                f"Sum of Prony relative moduli must be strictly less than 1.0 (sum = {sum_g:.4f}). "
                "g_infinity = 1 - sum(g_m) must be positive."
            )

        self._prony_terms: tuple[PronyTerm, ...] = tuple(parsed_terms)
        self._n_terms = len(self._prony_terms)
        self._g_infinity = 1.0 - sum_g

        # Active time step integration coefficients
        self._dt: float | None = None
        self._gamma: np.ndarray = np.ones(self._n_terms, dtype=np.float64)
        self._beta: np.ndarray = np.ones(self._n_terms, dtype=np.float64)
        self._eta: float = 1.0

        # State history variables per element:
        # _Q: internal 2nd PK stress for each Maxwell branch, shape (n_terms, n_elements, 3, 3)
        # _S0_prev: previous converged elastic 2nd PK stress, shape (n_elements, 3, 3)
        self._n_elements: int = 1
        self._Q = np.zeros((self._n_terms, self._n_elements, 3, 3), dtype=np.float64)
        self._S0_prev = np.zeros((self._n_elements, 3, 3), dtype=np.float64)

        # Trial state buffers during Newton iterations of active time step
        self._Q_trial = np.zeros((self._n_terms, self._n_elements, 3, 3), dtype=np.float64)
        self._S0_trial = np.zeros((self._n_elements, 3, 3), dtype=np.float64)

    # =========================================================================
    # Material Properties & Delegations
    # =========================================================================

    @property
    def base_material(self) -> Material:
        """Underlying instantaneous elastic or hyperelastic material model."""
        return self._base_material

    @property
    def parameters(self) -> IsotropicElasticParameters:
        """Elastic parameters of the underlying base material."""
        if hasattr(self._base_material, "parameters"):
            return self._base_material.parameters
        raise AttributeError("Base material does not expose parameters attribute.")

    @property
    def mass_density(self) -> float:
        """Mass density delegated to base material."""
        return self._base_material.mass_density

    @property
    def prony_terms(self) -> tuple[PronyTerm, ...]:
        """Sequence of Prony series terms (g_m, tau_m)."""
        return self._prony_terms

    @property
    def g_infinity(self) -> float:
        """Normalized long-term equilibrium modulus g_inf = 1 - sum(g_m)."""
        return self._g_infinity

    @property
    def deviatoric_only(self) -> bool:
        """Whether viscoelasticity acts solely on the isochoric stress component."""
        return self._deviatoric_only

    @property
    def dt(self) -> float | None:
        """Currently active time increment dt in seconds."""
        return self._dt

    @property
    def algorithmic_factor(self) -> float:
        """Active algorithmic tangent scaling factor eta(dt) in [g_inf, 1.0]."""
        return self._eta

    def relaxation_factor(self, t: float) -> float:
        """
        Computes continuous analytical relaxation factor g(t):
            g(t) = g_inf + sum_{m=1}^{Np} g_m * exp(-t / tau_m)

        Parameters:
        -----------
        t : float
            Elapsed physical time in seconds (t >= 0).

        Returns:
        --------
        g : float
            Normalized instantaneous/relaxed shear modulus ratio in [g_inf, 1.0].
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        val = self._g_infinity
        for term in self._prony_terms:
            val += term.relative_modulus * np.exp(-t / term.relaxation_time)
        return float(val)

    # =========================================================================
    # History & State Allocation / Lifecycle
    # =========================================================================

    def initialize_history(self, n_elements: int = 1) -> None:
        """
        Allocates internal history tensors for the specified number of elements.

        Parameters:
        -----------
        n_elements : int
            Total continuum elements in the domain.
        """
        n_elem = max(1, int(n_elements))
        self._n_elements = n_elem
        self._Q = np.zeros((self._n_terms, n_elem, 3, 3), dtype=np.float64)
        self._S0_prev = np.zeros((n_elem, 3, 3), dtype=np.float64)
        self._Q_trial = np.zeros((self._n_terms, n_elem, 3, 3), dtype=np.float64)
        self._S0_trial = np.zeros((n_elem, 3, 3), dtype=np.float64)

    def set_time_step(self, dt: float) -> None:
        """
        Configures algorithmic integration coefficients for the active time step dt.

        Parameters:
        -----------
        dt : float
            Time increment in seconds (> 0).
        """
        if dt <= 0.0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        self._dt = float(dt)

        # Precompute exponential relaxation coefficients
        eta = self._g_infinity
        for m, term in enumerate(self._prony_terms):
            tau = term.relaxation_time
            gm = term.relative_modulus
            ratio = self._dt / tau
            gamma_m = float(np.exp(-ratio))
            # Limit as dt -> 0: beta_m -> 1.0; for finite dt: (1 - gamma_m) / ratio
            if ratio < 1e-12:
                beta_m = 1.0 - 0.5 * ratio
            else:
                beta_m = (1.0 - gamma_m) / ratio

            self._gamma[m] = gamma_m
            self._beta[m] = beta_m
            eta += gm * beta_m

        self._eta = float(eta)

    def commit_state(self) -> None:
        """Commits converged trial state into the accepted history state at t_{n+1}."""
        self._Q[:] = self._Q_trial[:]
        self._S0_prev[:] = self._S0_trial[:]

    def revert_state(self) -> None:
        """Reverts trial state buffers upon Newton rejection or time step cutback."""
        self._Q_trial[:] = self._Q[:]
        self._S0_trial[:] = self._S0_prev[:]

    def reset_state(self) -> None:
        """Resets all internal history variables to initial undeformed reference state."""
        self._Q.fill(0.0)
        self._S0_prev.fill(0.0)
        self._Q_trial.fill(0.0)
        self._S0_trial.fill(0.0)

    # =========================================================================
    # Constitutive Stress & Tangent Evaluation
    # =========================================================================

    def compute_constitutive_matrix(self) -> np.ndarray:
        """
        Computes the small-strain 6x6 isotropic elasticity matrix C.
        Scaled by the active algorithmic factor eta(dt).
        """
        C0 = self._base_material.compute_constitutive_matrix()
        return self._eta * C0

    def compute_stress_and_tangent(
        self, deformation_gradient: np.ndarray, elem_index: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the 6x6 tangent constitutive matrix C and 3x3 Second Piola-Kirchhoff
        stress S with viscoelastic history accumulation.

        Parameters:
        -----------
        deformation_gradient : np.ndarray of shape (3, 3)
            Current spatial deformation gradient tensor F.
        elem_index : int, optional
            Index of the element being evaluated for history tracking.

        Returns:
        --------
        C_tan : np.ndarray of shape (6, 6)
            Algorithmic material tangent stiffness matrix.
        S_total : np.ndarray of shape (3, 3)
            Total 2nd Piola-Kirchhoff stress tensor.
        """
        # Query instantaneous base material
        C0, S0 = self._base_material.compute_stress_and_tangent(deformation_gradient)

        # If time step is unconfigured, return instantaneous elastic response
        if self._dt is None:
            return C0, S0

        e = elem_index if elem_index is not None else 0
        if e >= self._n_elements:
            # Dynamically resize buffer if evaluated beyond current allocation
            new_size = max(e + 1, self._n_elements * 2)
            self._resize_history(new_size)

        S0_n = self._S0_prev[e]
        delta_S0 = S0 - S0_n

        # Update Maxwell branch history tensors:
        # Q_{m, n+1} = gamma_m * Q_{m, n} + g_m * beta_m * (S0_{n+1} - S0_n)
        S_non_eq = np.zeros((3, 3), dtype=np.float64)
        for m in range(self._n_terms):
            gm = self._prony_terms[m].relative_modulus
            gamma_m = self._gamma[m]
            beta_m = self._beta[m]

            Q_m_new = gamma_m * self._Q[m, e] + gm * beta_m * delta_S0
            self._Q_trial[m, e] = Q_m_new
            S_non_eq += Q_m_new

        self._S0_trial[e] = S0

        # Total 2nd PK Stress: S_{n+1} = g_inf * S^0_{n+1} + sum_{m} Q_{m, n+1}
        S_total = self._g_infinity * S0 + S_non_eq

        # Algorithmic Tangent: C_{n+1} = eta(dt) * C0_{n+1}
        C_tan = self._eta * C0

        return C_tan, S_total

    def _resize_history(self, new_size: int) -> None:
        """Internal helper to resize element state history arrays."""
        old_size = self._n_elements
        self._n_elements = new_size

        new_Q = np.zeros((self._n_terms, new_size, 3, 3), dtype=np.float64)
        new_Q[:, :old_size] = self._Q
        self._Q = new_Q

        new_S0 = np.zeros((new_size, 3, 3), dtype=np.float64)
        new_S0[:old_size] = self._S0_prev
        self._S0_prev = new_S0

        new_Q_tr = np.zeros((self._n_terms, new_size, 3, 3), dtype=np.float64)
        new_Q_tr[:, :old_size] = self._Q_trial
        self._Q_trial = new_Q_tr

        new_S0_tr = np.zeros((new_size, 3, 3), dtype=np.float64)
        new_S0_tr[:old_size] = self._S0_trial
        self._S0_trial = new_S0_tr
