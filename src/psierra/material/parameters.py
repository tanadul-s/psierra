from __future__ import annotations
from collections.abc import Sequence
import numpy as np


class IsotropicElasticParameters:
    """
    Isotropic Elastic Material Parameters.

    Manages isotropic elastic moduli and mass density with direct, exact calculations.
    Each setter directly assigns the user-provided constants (without round-trip conversions)
    and computes all remaining moduli natively.

    Usage:
    ------
    >>> params = IsotropicElasticParameters()
    >>> params.set_from_bulk_shear(
    ...     bulk_modulus=5000.0,
    ...     shear_modulus=1071.4,
    ...     mass_density=1040.0,
    ... )
    >>> print(params.bulk_modulus)      # Exactly 5000.0
    >>> print(params.shear_modulus)     # Exactly 1071.4
    >>> print(params.youngs_modulus)    # 2999.92...
    >>> print(params.poissons_ratio)    # 0.39999...
    """

    def __init__(self) -> None:
        """Initialize an unpopulated isotropic elastic parameter container."""
        self._youngs_modulus: float | None = None
        self._poissons_ratio: float | None = None
        self._shear_modulus: float | None = None
        self._lame_parameter_1: float | None = None
        self._lame_parameter_2: float | None = None
        self._bulk_modulus: float | None = None
        self._p_wave_modulus: float | None = None
        self._mass_density: float | None = None

    # =========================================================================
    # Direct Setters (Exact assignments + native derivations)
    # =========================================================================

    def set_from_youngs_poisson(
        self,
        youngs_modulus: float,
        poissons_ratio: float,
        mass_density: float,
    ) -> None:
        """Set state directly from Young's modulus (E), Poisson's ratio (nu), and mass density (rho)."""
        E = float(youngs_modulus)
        nu = float(poissons_ratio)
        rho = float(mass_density)

        if E <= 0.0:
            raise ValueError(f"Young's modulus must be positive. Got {E}")
        if not (-1.0 < nu < 0.5):
            raise ValueError(
                f"Poisson's ratio must satisfy -1.0 < poissons_ratio < 0.5. Got {nu}"
            )
        self._validate_density(rho)

        # Directly assign input values
        self._youngs_modulus = E
        self._poissons_ratio = nu
        self._mass_density = rho

        # Derive other moduli natively from (E, nu)
        mu = E / (2.0 * (1.0 + nu))
        lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        kappa = E / (3.0 * (1.0 - 2.0 * nu))

        self._shear_modulus = mu
        self._lame_parameter_2 = mu
        self._lame_parameter_1 = lam
        self._bulk_modulus = kappa
        self._p_wave_modulus = lam + 2.0 * mu

    def set_from_bulk_shear(
        self,
        bulk_modulus: float,
        shear_modulus: float,
        mass_density: float,
    ) -> None:
        """Set state directly from bulk modulus (kappa), shear modulus (mu), and mass density (rho)."""
        kappa = float(bulk_modulus)
        mu = float(shear_modulus)
        rho = float(mass_density)

        if kappa <= 0.0:
            raise ValueError(f"Bulk modulus must be positive. Got {kappa}")
        if mu <= 0.0:
            raise ValueError(f"Shear modulus must be positive. Got {mu}")
        self._validate_density(rho)

        # Directly assign input values
        self._bulk_modulus = kappa
        self._shear_modulus = mu
        self._lame_parameter_2 = mu
        self._mass_density = rho

        # Derive other moduli natively from (kappa, mu)
        self._lame_parameter_1 = kappa - (2.0 / 3.0) * mu
        self._youngs_modulus = (9.0 * kappa * mu) / (3.0 * kappa + mu)
        self._poissons_ratio = (3.0 * kappa - 2.0 * mu) / (2.0 * (3.0 * kappa + mu))
        self._p_wave_modulus = kappa + (4.0 / 3.0) * mu

    def set_from_lame_parameters(
        self,
        lame_parameter_1: float,
        lame_parameter_2: float,
        mass_density: float,
    ) -> None:
        """Set state directly from Lamé parameter 1 (lambda), Lamé parameter 2 (mu), and mass density (rho)."""
        lam = float(lame_parameter_1)
        mu = float(lame_parameter_2)
        rho = float(mass_density)

        if mu <= 0.0:
            raise ValueError(
                f"Lamé parameter 2 (shear modulus) must be positive. Got {mu}"
            )
        if (lam + mu) <= 0.0:
            raise ValueError(
                f"Constraint (lame_parameter_1 + lame_parameter_2 > 0) violated. Got lambda={lam}, mu={mu}"
            )
        self._validate_density(rho)

        # Directly assign input values
        self._lame_parameter_1 = lam
        self._lame_parameter_2 = mu
        self._shear_modulus = mu
        self._mass_density = rho

        # Derive other moduli natively from (lambda, mu)
        self._bulk_modulus = lam + (2.0 / 3.0) * mu
        self._youngs_modulus = (mu * (3.0 * lam + 2.0 * mu)) / (lam + mu)
        self._poissons_ratio = lam / (2.0 * (lam + mu))
        self._p_wave_modulus = lam + 2.0 * mu

    def set_from_youngs_shear(
        self,
        youngs_modulus: float,
        shear_modulus: float,
        mass_density: float,
    ) -> None:
        """Set state directly from Young's modulus (E), shear modulus (mu), and mass density (rho)."""
        E = float(youngs_modulus)
        mu = float(shear_modulus)
        rho = float(mass_density)

        if E <= 0.0 or mu <= 0.0:
            raise ValueError("Moduli must be positive.")
        if E >= 3.0 * mu:
            raise ValueError(
                f"Constraint E < 3*mu violated for physical materials (requires nu > -1). Got E={E}, mu={mu}"
            )
        self._validate_density(rho)

        # Directly assign input values
        self._youngs_modulus = E
        self._shear_modulus = mu
        self._lame_parameter_2 = mu
        self._mass_density = rho

        # Derive other moduli natively from (E, mu)
        self._poissons_ratio = (E / (2.0 * mu)) - 1.0
        self._lame_parameter_1 = (mu * (E - 2.0 * mu)) / (3.0 * mu - E)
        self._bulk_modulus = (E * mu) / (3.0 * (3.0 * mu - E))
        self._p_wave_modulus = (mu * (4.0 * mu - E)) / (3.0 * mu - E)

    def set_from_youngs_bulk(
        self,
        youngs_modulus: float,
        bulk_modulus: float,
        mass_density: float,
    ) -> None:
        """Set state directly from Young's modulus (E), bulk modulus (kappa), and mass density (rho)."""
        E = float(youngs_modulus)
        kappa = float(bulk_modulus)
        rho = float(mass_density)

        if E <= 0.0 or kappa <= 0.0:
            raise ValueError("Moduli must be positive.")
        if E >= 9.0 * kappa:
            raise ValueError(
                f"Constraint E < 9*kappa violated for physical materials (requires nu < 0.5). Got E={E}, kappa={kappa}"
            )
        self._validate_density(rho)

        # Directly assign input values
        self._youngs_modulus = E
        self._bulk_modulus = kappa
        self._mass_density = rho

        # Derive other moduli natively from (E, kappa)
        self._poissons_ratio = (3.0 * kappa - E) / (6.0 * kappa)
        self._shear_modulus = (3.0 * kappa * E) / (9.0 * kappa - E)
        self._lame_parameter_2 = self._shear_modulus
        self._lame_parameter_1 = (3.0 * kappa * (3.0 * kappa - E)) / (9.0 * kappa - E)
        self._p_wave_modulus = (3.0 * kappa * (3.0 * kappa + E)) / (9.0 * kappa - E)

    # =========================================================================
    # Direct Property Access
    # =========================================================================

    def _check_initialized(self) -> None:
        if self._youngs_modulus is None:
            raise RuntimeError(
                "IsotropicElasticParameters is not initialized. "
                "Call a set_from_* method before accessing properties."
            )

    @property
    def youngs_modulus(self) -> float:
        """Young's modulus (Pa)."""
        self._check_initialized()
        return self._youngs_modulus  # type: ignore[return-value]

    @property
    def poissons_ratio(self) -> float:
        """Poisson's ratio (dimensionless)."""
        self._check_initialized()
        return self._poissons_ratio  # type: ignore[return-value]

    @property
    def shear_modulus(self) -> float:
        """Shear modulus (Pa)."""
        self._check_initialized()
        return self._shear_modulus  # type: ignore[return-value]

    @property
    def lame_parameter_2(self) -> float:
        """Lamé parameter 2 (identically the shear modulus) (Pa)."""
        self._check_initialized()
        return self._lame_parameter_2  # type: ignore[return-value]

    @property
    def lame_parameter_1(self) -> float:
        """Lamé parameter 1 (first Lamé parameter lambda) (Pa)."""
        self._check_initialized()
        return self._lame_parameter_1  # type: ignore[return-value]

    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus (Pa)."""
        self._check_initialized()
        return self._bulk_modulus  # type: ignore[return-value]

    @property
    def p_wave_modulus(self) -> float:
        """P-wave / Longitudinal modulus (Pa)."""
        self._check_initialized()
        return self._p_wave_modulus  # type: ignore[return-value]

    @property
    def mass_density(self) -> float:
        """Mass density (kg/m^3)."""
        self._check_initialized()
        return self._mass_density  # type: ignore[return-value]

    # =========================================================================
    # Internal Validation & Representation
    # =========================================================================

    def _validate_density(self, density: float) -> None:
        if density <= 0.0:
            raise ValueError(f"Mass density must be positive. Got {density}")

    def __repr__(self) -> str:
        if self._youngs_modulus is None:
            return "IsotropicElasticParameters(uninitialized)"
        return (
            f"IsotropicElasticParameters"
            f"\n  youngs_modulus   = {self.youngs_modulus:.4e} Pa"
            f"\n  poissons_ratio   = {self.poissons_ratio:.4f}"
            f"\n  shear_modulus    = {self.shear_modulus:.4e} Pa"
            f"\n  lame_parameter_1 = {self.lame_parameter_1:.4e} Pa"
            f"\n  lame_parameter_2 = {self.lame_parameter_2:.4e} Pa"
            f"\n  bulk_modulus     = {self.bulk_modulus:.4e} Pa"
            f"\n  p_wave_modulus   = {self.p_wave_modulus:.4e} Pa"
            f"\n  mass_density     = {self.mass_density:.4e} kg/m^3"
        )


class MooneyRivlinParameters:
    """
    Mooney-Rivlin Hyperelastic Material Parameters.

    Governed by parameters c10, c01, bulk modulus (kappa), and mass density (rho).
    The initial shear modulus in the infinitesimal limit is mu = 2 * (c10 + c01).
    """

    def __init__(
        self,
        c10: float | None = None,
        c01: float | None = None,
        bulk_modulus: float | None = None,
        mass_density: float | None = None,
    ) -> None:
        self._c10: float | None = None
        self._c01: float | None = None
        self._bulk_modulus: float | None = None
        self._mass_density: float | None = None

        if c10 is not None or c01 is not None or bulk_modulus is not None or mass_density is not None:
            if c10 is None or c01 is None or bulk_modulus is None or mass_density is None:
                raise ValueError("All parameters (c10, c01, bulk_modulus, mass_density) must be provided.")
            self.set_from_constants(
                c10=c10,
                c01=c01,
                bulk_modulus=bulk_modulus,
                mass_density=mass_density,
            )

    def set_from_constants(
        self,
        c10: float,
        c01: float,
        bulk_modulus: float,
        mass_density: float,
    ) -> None:
        """Set state directly from Mooney-Rivlin constants."""
        c10_val = float(c10)
        c01_val = float(c01)
        k_val = float(bulk_modulus)
        rho_val = float(mass_density)

        if c10_val + c01_val <= 0.0:
            raise ValueError(f"Sum (c10 + c01) must be positive for positive shear stiffness. Got {c10_val + c01_val}")
        if k_val <= 0.0:
            raise ValueError(f"Bulk modulus must be positive. Got {k_val}")
        if rho_val <= 0.0:
            raise ValueError(f"Mass density must be positive. Got {rho_val}")

        self._c10 = c10_val
        self._c01 = c01_val
        self._bulk_modulus = k_val
        self._mass_density = rho_val

    def set_from_isotropic(
        self,
        params: IsotropicElasticParameters,
        ratio_c10: float = 0.5,
    ) -> None:
        """
        Derive Mooney-Rivlin constants from an IsotropicElasticParameters instance
        using a given partition ratio: c10 = ratio_c10 * (mu / 2), c01 = (1 - ratio_c10) * (mu / 2).
        """
        if not (0.0 <= ratio_c10 <= 1.0):
            raise ValueError(f"ratio_c10 must be in [0.0, 1.0]. Got {ratio_c10}")

        mu = params.shear_modulus
        c10_val = ratio_c10 * (0.5 * mu)
        c01_val = (1.0 - ratio_c10) * (0.5 * mu)
        self.set_from_constants(
            c10=c10_val,
            c01=c01_val,
            bulk_modulus=params.bulk_modulus,
            mass_density=params.mass_density,
        )

    @classmethod
    def from_constants(
        cls,
        c10: float,
        c01: float,
        bulk_modulus: float,
        mass_density: float,
    ) -> MooneyRivlinParameters:
        """Convenience factory method creating initialized MooneyRivlinParameters."""
        inst = cls()
        inst.set_from_constants(c10=c10, c01=c01, bulk_modulus=bulk_modulus, mass_density=mass_density)
        return inst

    @classmethod
    def from_isotropic(
        cls,
        params: IsotropicElasticParameters,
        ratio_c10: float = 0.5,
    ) -> MooneyRivlinParameters:
        """Convenience factory method deriving MooneyRivlinParameters from isotropic elasticity."""
        inst = cls()
        inst.set_from_isotropic(params=params, ratio_c10=ratio_c10)
        return inst

    def _check_initialized(self) -> None:
        if self._c10 is None:
            raise ValueError("MooneyRivlinParameters instance is uninitialized.")

    @property
    def c10(self) -> float:
        """Mooney-Rivlin parameter c10 (Pa)."""
        self._check_initialized()
        return self._c10  # type: ignore[return-value]

    @property
    def c01(self) -> float:
        """Mooney-Rivlin parameter c01 (Pa)."""
        self._check_initialized()
        return self._c01  # type: ignore[return-value]

    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus K (Pa)."""
        self._check_initialized()
        return self._bulk_modulus  # type: ignore[return-value]

    @property
    def shear_modulus(self) -> float:
        """Infinitesimal shear modulus mu = 2 * (c10 + c01) (Pa)."""
        return 2.0 * (self.c10 + self.c01)

    @property
    def youngs_modulus(self) -> float:
        """Equivalent infinitesimal Young's modulus E (Pa)."""
        k = self.bulk_modulus
        mu = self.shear_modulus
        return (9.0 * k * mu) / (3.0 * k + mu)

    @property
    def poissons_ratio(self) -> float:
        """Equivalent infinitesimal Poisson's ratio nu."""
        k = self.bulk_modulus
        mu = self.shear_modulus
        return (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))

    @property
    def lame_parameter_1(self) -> float:
        """First Lamé parameter lambda (Pa)."""
        return self.bulk_modulus - (2.0 / 3.0) * self.shear_modulus

    @property
    def lame_parameter_2(self) -> float:
        """Second Lamé parameter mu (Pa)."""
        return self.shear_modulus

    @property
    def mass_density(self) -> float:
        """Mass density (kg/m^3)."""
        self._check_initialized()
        return self._mass_density  # type: ignore[return-value]

    def __repr__(self) -> str:
        if self._c10 is None:
            return "MooneyRivlinParameters(uninitialized)"
        return (
            f"MooneyRivlinParameters"
            f"\n  c10              = {self.c10:.4e} Pa"
            f"\n  c01              = {self.c01:.4e} Pa"
            f"\n  shear_modulus    = {self.shear_modulus:.4e} Pa"
            f"\n  bulk_modulus     = {self.bulk_modulus:.4e} Pa"
            f"\n  youngs_modulus   = {self.youngs_modulus:.4e} Pa"
            f"\n  poissons_ratio   = {self.poissons_ratio:.4f}"
            f"\n  mass_density     = {self.mass_density:.4e} kg/m^3"
        )


class YeohParameters:
    """
    Yeoh (Reduced Polynomial) Hyperelastic Material Parameters.

    Governed by parameters c10, c20, c30, bulk modulus (kappa), and mass density (rho).
    The initial shear modulus in the infinitesimal limit is mu = 2 * c10.
    """

    def __init__(
        self,
        c10: float | None = None,
        c20: float | None = None,
        c30: float | None = None,
        bulk_modulus: float | None = None,
        mass_density: float | None = None,
    ) -> None:
        self._c10: float | None = None
        self._c20: float | None = None
        self._c30: float | None = None
        self._bulk_modulus: float | None = None
        self._mass_density: float | None = None

        if c10 is not None or c20 is not None or c30 is not None or bulk_modulus is not None or mass_density is not None:
            if c10 is None or c20 is None or c30 is None or bulk_modulus is None or mass_density is None:
                raise ValueError("All parameters (c10, c20, c30, bulk_modulus, mass_density) must be provided.")
            self.set_from_constants(
                c10=c10,
                c20=c20,
                c30=c30,
                bulk_modulus=bulk_modulus,
                mass_density=mass_density,
            )

    def set_from_constants(
        self,
        c10: float,
        c20: float,
        c30: float,
        bulk_modulus: float,
        mass_density: float,
    ) -> None:
        """Set state directly from Yeoh constants."""
        c10_val = float(c10)
        c20_val = float(c20)
        c30_val = float(c30)
        k_val = float(bulk_modulus)
        rho_val = float(mass_density)

        if c10_val <= 0.0:
            raise ValueError(f"c10 must be positive for positive initial shear stiffness. Got {c10_val}")
        if k_val <= 0.0:
            raise ValueError(f"Bulk modulus must be positive. Got {k_val}")
        if rho_val <= 0.0:
            raise ValueError(f"Mass density must be positive. Got {rho_val}")

        self._c10 = c10_val
        self._c20 = c20_val
        self._c30 = c30_val
        self._bulk_modulus = k_val
        self._mass_density = rho_val

    def set_from_isotropic(
        self,
        params: IsotropicElasticParameters,
        c20: float = 0.0,
        c30: float = 0.0,
    ) -> None:
        """
        Derive Yeoh constants from an IsotropicElasticParameters instance:
        c10 = mu / 2, with optional higher-order strain-stiffening parameters c20 and c30.
        """
        c10_val = 0.5 * params.shear_modulus
        self.set_from_constants(
            c10=c10_val,
            c20=c20,
            c30=c30,
            bulk_modulus=params.bulk_modulus,
            mass_density=params.mass_density,
        )

    @classmethod
    def from_constants(
        cls,
        c10: float,
        c20: float,
        c30: float,
        bulk_modulus: float,
        mass_density: float,
    ) -> YeohParameters:
        """Convenience factory method creating initialized YeohParameters."""
        inst = cls()
        inst.set_from_constants(c10=c10, c20=c20, c30=c30, bulk_modulus=bulk_modulus, mass_density=mass_density)
        return inst

    @classmethod
    def from_isotropic(
        cls,
        params: IsotropicElasticParameters,
        c20: float = 0.0,
        c30: float = 0.0,
    ) -> YeohParameters:
        """Convenience factory method deriving YeohParameters from isotropic elasticity."""
        inst = cls()
        inst.set_from_isotropic(params=params, c20=c20, c30=c30)
        return inst

    def _check_initialized(self) -> None:
        if self._c10 is None:
            raise ValueError("YeohParameters instance is uninitialized.")

    @property
    def c10(self) -> float:
        """Yeoh parameter c10 (Pa)."""
        self._check_initialized()
        return self._c10  # type: ignore[return-value]

    @property
    def c20(self) -> float:
        """Yeoh parameter c20 (Pa)."""
        self._check_initialized()
        return self._c20  # type: ignore[return-value]

    @property
    def c30(self) -> float:
        """Yeoh parameter c30 (Pa)."""
        self._check_initialized()
        return self._c30  # type: ignore[return-value]

    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus K (Pa)."""
        self._check_initialized()
        return self._bulk_modulus  # type: ignore[return-value]

    @property
    def shear_modulus(self) -> float:
        """Infinitesimal shear modulus mu = 2 * c10 (Pa)."""
        return 2.0 * self.c10

    @property
    def youngs_modulus(self) -> float:
        """Equivalent infinitesimal Young's modulus E (Pa)."""
        k = self.bulk_modulus
        mu = self.shear_modulus
        return (9.0 * k * mu) / (3.0 * k + mu)

    @property
    def poissons_ratio(self) -> float:
        """Equivalent infinitesimal Poisson's ratio nu."""
        k = self.bulk_modulus
        mu = self.shear_modulus
        return (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))

    @property
    def lame_parameter_1(self) -> float:
        """First Lamé parameter lambda (Pa)."""
        return self.bulk_modulus - (2.0 / 3.0) * self.shear_modulus

    @property
    def lame_parameter_2(self) -> float:
        """Second Lamé parameter mu (Pa)."""
        return self.shear_modulus

    @property
    def mass_density(self) -> float:
        """Mass density (kg/m^3)."""
        self._check_initialized()
        return self._mass_density  # type: ignore[return-value]

    def __repr__(self) -> str:
        if self._c10 is None:
            return "YeohParameters(uninitialized)"
        return (
            f"YeohParameters"
            f"\n  c10              = {self.c10:.4e} Pa"
            f"\n  c20              = {self.c20:.4e} Pa"
            f"\n  c30              = {self.c30:.4e} Pa"
            f"\n  shear_modulus    = {self.shear_modulus:.4e} Pa"
            f"\n  bulk_modulus     = {self.bulk_modulus:.4e} Pa"
            f"\n  youngs_modulus   = {self.youngs_modulus:.4e} Pa"
            f"\n  poissons_ratio   = {self.poissons_ratio:.4f}"
            f"\n  mass_density     = {self.mass_density:.4e} kg/m^3"
        )


class OgdenParameters:
    """
    Generalized Ogden Hyperelastic Material Parameters.

    Governed by pairs of coefficients (mu_p, alpha_p) for p = 1..N (N in [1, 3]),
    bulk modulus K, and mass density rho.
    The strain energy potential is:
        W_iso = sum_{p=1}^N (mu_p / alpha_p) * (lambda_bar_1^alpha_p + lambda_bar_2^alpha_p + lambda_bar_3^alpha_p - 3)

    In the infinitesimal limit (small strain), the initial shear modulus is:
        mu_0 = 0.5 * sum_{p=1}^N mu_p * alpha_p
    """

    def __init__(
        self,
        mu: Sequence[float] | None = None,
        alpha: Sequence[float] | None = None,
        bulk_modulus: float | None = None,
        mass_density: float | None = None,
    ) -> None:
        self._mu: tuple[float, ...] | None = None
        self._alpha: tuple[float, ...] | None = None
        self._bulk_modulus: float | None = None
        self._mass_density: float | None = None

        if mu is not None or alpha is not None or bulk_modulus is not None or mass_density is not None:
            if mu is None or alpha is None or bulk_modulus is None or mass_density is None:
                raise ValueError("All parameters (mu, alpha, bulk_modulus, mass_density) must be provided.")
            self.set_from_constants(
                mu=mu,
                alpha=alpha,
                bulk_modulus=bulk_modulus,
                mass_density=mass_density,
            )

    def set_from_constants(
        self,
        mu: Sequence[float],
        alpha: Sequence[float],
        bulk_modulus: float,
        mass_density: float,
    ) -> None:
        """Set state directly from Ogden parameters (mu_p, alpha_p), bulk modulus, and mass density."""
        if len(mu) != len(alpha):
            raise ValueError(f"Length of mu ({len(mu)}) and alpha ({len(alpha)}) must match.")
        if len(mu) == 0:
            raise ValueError("At least one term (mu, alpha) must be provided for Ogden model.")

        mu_tuple = tuple(float(m) for m in mu)
        alpha_tuple = tuple(float(a) for a in alpha)
        k_val = float(bulk_modulus)
        rho_val = float(mass_density)

        for i, a in enumerate(alpha_tuple):
            if abs(a) < 1e-12:
                raise ValueError(f"Ogden exponent alpha[{i}] cannot be zero.")

        # Compute initial shear modulus: 2 * mu_0 = sum(mu_p * alpha_p)
        two_mu0 = sum(m * a for m, a in zip(mu_tuple, alpha_tuple))
        if two_mu0 <= 0.0:
            raise ValueError(
                f"Sum sum(mu_p * alpha_p) must be positive to ensure positive initial shear stiffness. Got {two_mu0:.4e}"
            )
        if k_val <= 0.0:
            raise ValueError(f"Bulk modulus must be positive. Got {k_val}")
        if rho_val <= 0.0:
            raise ValueError(f"Mass density must be positive. Got {rho_val}")

        self._mu = mu_tuple
        self._alpha = alpha_tuple
        self._bulk_modulus = k_val
        self._mass_density = rho_val

    def set_from_isotropic(
        self,
        params: IsotropicElasticParameters,
        order: int = 1,
        alpha: Sequence[float] | None = None,
        weights: Sequence[float] | None = None,
    ) -> None:
        """
        Derive Ogden parameters from an IsotropicElasticParameters instance.
        Ensures the initial shear modulus matches mu_0 = params.shear_modulus:
            2 * mu_0 = sum_{p=1}^N mu_p * alpha_p
        """
        mu0 = params.shear_modulus
        if order < 1:
            raise ValueError(f"Order must be at least 1, got {order}")

        if alpha is not None:
            if len(alpha) != order:
                raise ValueError(f"Length of custom alpha ({len(alpha)}) must match order ({order}).")
            alphas = tuple(float(a) for a in alpha)
        elif order == 1:
            alphas = (2.0,)
        elif order == 2:
            alphas = (2.0, -2.0)
        elif order == 3:
            alphas = (1.3, 5.0, -2.0)
        else:
            alphas = tuple(2.0 * (i + 1) for i in range(order))

        if weights is not None:
            if len(weights) != order:
                raise ValueError(f"Length of custom weights ({len(weights)}) must match order ({order}).")
            w = np.array(weights, dtype=np.float64)
            w = w / np.sum(w)
        elif order == 1:
            w = np.array([1.0], dtype=np.float64)
        elif order == 2:
            w = np.array([0.5, 0.5], dtype=np.float64)
        elif order == 3:
            w = np.array([0.6, 0.1, 0.3], dtype=np.float64)
        else:
            w = np.ones(order, dtype=np.float64) / order

        mus = tuple(float(2.0 * w[p] * mu0 / alphas[p]) for p in range(order))
        self.set_from_constants(
            mu=mus,
            alpha=alphas,
            bulk_modulus=params.bulk_modulus,
            mass_density=params.mass_density,
        )

    @classmethod
    def from_constants(
        cls,
        mu: Sequence[float],
        alpha: Sequence[float],
        bulk_modulus: float,
        mass_density: float,
    ) -> OgdenParameters:
        """Convenience factory method creating initialized OgdenParameters."""
        inst = cls()
        inst.set_from_constants(mu=mu, alpha=alpha, bulk_modulus=bulk_modulus, mass_density=mass_density)
        return inst

    @classmethod
    def from_isotropic(
        cls,
        params: IsotropicElasticParameters,
        order: int = 1,
        alpha: Sequence[float] | None = None,
        weights: Sequence[float] | None = None,
    ) -> OgdenParameters:
        """Convenience factory method deriving OgdenParameters from isotropic elasticity."""
        inst = cls()
        inst.set_from_isotropic(params=params, order=order, alpha=alpha, weights=weights)
        return inst

    @classmethod
    def from_brain_parenchyma(
        cls,
        bulk_modulus: float,
        mass_density: float,
        order: int = 1,
    ) -> OgdenParameters:
        """
        Brain parenchyma soft tissue model (Miller & Chinzei 2002).
        Captures tension-compression asymmetry: mu1 = -842.0 Pa, alpha1 = -4.7 (mu1 * alpha1 > 0).
        """
        if order == 1:
            return cls.from_constants(
                mu=[-842.0],
                alpha=[-4.7],
                bulk_modulus=bulk_modulus,
                mass_density=mass_density,
            )
        elif order == 2:
            return cls.from_constants(
                mu=[-842.0, 420.0],
                alpha=[-4.7, 2.0],
                bulk_modulus=bulk_modulus,
                mass_density=mass_density,
            )
        else:
            return cls.from_constants(
                mu=[-842.0, 420.0, 100.0],
                alpha=[-4.7, 2.0, 5.0],
                bulk_modulus=bulk_modulus,
                mass_density=mass_density,
            )

    @classmethod
    def from_rubber(
        cls,
        bulk_modulus: float,
        mass_density: float,
    ) -> OgdenParameters:
        """Classical 3-term Ogden (1972) rubber parameters."""
        return cls.from_constants(
            mu=[6.3e5, 1.2e3, -1.0e4],
            alpha=[1.3, 5.0, -2.0],
            bulk_modulus=bulk_modulus,
            mass_density=mass_density,
        )

    def _check_initialized(self) -> None:
        if self._mu is None or self._alpha is None:
            raise ValueError("OgdenParameters instance is uninitialized.")

    @property
    def order(self) -> int:
        """Order N of the Ogden model."""
        self._check_initialized()
        return len(self._mu)  # type: ignore[arg-type]

    @property
    def mu(self) -> tuple[float, ...]:
        """Ogden shear coefficients mu_p (Pa or MPa)."""
        self._check_initialized()
        return self._mu  # type: ignore[return-value]

    @property
    def alpha(self) -> tuple[float, ...]:
        """Ogden dimensionless exponents alpha_p."""
        self._check_initialized()
        return self._alpha  # type: ignore[return-value]

    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus K (Pa or MPa)."""
        self._check_initialized()
        return self._bulk_modulus  # type: ignore[return-value]

    @property
    def shear_modulus(self) -> float:
        """Infinitesimal shear modulus mu_0 = 0.5 * sum(mu_p * alpha_p) (Pa or MPa)."""
        self._check_initialized()
        return 0.5 * sum(m * a for m, a in zip(self._mu, self._alpha))  # type: ignore[arg-type]

    @property
    def youngs_modulus(self) -> float:
        """Equivalent infinitesimal Young's modulus E (Pa or MPa)."""
        k = self.bulk_modulus
        mu = self.shear_modulus
        return (9.0 * k * mu) / (3.0 * k + mu)

    @property
    def poissons_ratio(self) -> float:
        """Equivalent infinitesimal Poisson's ratio nu."""
        k = self.bulk_modulus
        mu = self.shear_modulus
        return (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))

    @property
    def lame_parameter_1(self) -> float:
        """First Lamé parameter lambda (Pa or MPa)."""
        return self.bulk_modulus - (2.0 / 3.0) * self.shear_modulus

    @property
    def lame_parameter_2(self) -> float:
        """Second Lamé parameter mu (Pa or MPa)."""
        return self.shear_modulus

    @property
    def mass_density(self) -> float:
        """Mass density (kg/m^3 or tonne/mm^3)."""
        self._check_initialized()
        return self._mass_density  # type: ignore[return-value]

    def __repr__(self) -> str:
        if self._mu is None:
            return "OgdenParameters(uninitialized)"
        terms_str = ", ".join(f"(μ={m:.2e}, α={a:.2f})" for m, a in zip(self.mu, self.alpha))
        return (
            f"OgdenParameters(N={self.order})"
            f"\n  Terms            = [{terms_str}]"
            f"\n  shear_modulus    = {self.shear_modulus:.4e}"
            f"\n  bulk_modulus     = {self.bulk_modulus:.4e}"
            f"\n  youngs_modulus   = {self.youngs_modulus:.4e}"
            f"\n  poissons_ratio   = {self.poissons_ratio:.4f}"
            f"\n  mass_density     = {self.mass_density:.4e}"
        )

