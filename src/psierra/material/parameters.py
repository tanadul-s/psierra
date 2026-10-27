from __future__ import annotations


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
