"""
Example 0101: Constitutive Material Models & Parameter Formulation.

Showcases linear elastic, compressible hyperelastic, viscoelastic, and anisotropic
constitutive models on equal standing:
1. Hookean (Linear isotropic elasticity: E, nu, K, mu)
2. Neo-Hookean (Compressible hyperelasticity with logarithmic volume penalty)
3. Mooney-Rivlin (Two-parameter rubber elasticity)
4. Yeoh (Three-parameter soft tissue model with progressive strain-stiffening)
5. Ogden (Principal stretch-based generalized hyperelasticity)
6. Holzapfel-Gasser-Ogden (HGO: Matrix + collagen fiber bundle dispersion)
7. Viscoelastic (Generalized Maxwell / Prony series rate-dependent relaxation)
"""

from __future__ import annotations
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psierra.material import (
    IsotropicElasticParameters,
    MooneyRivlinParameters,
    YeohParameters,
    OgdenParameters,
    HGOParameters,
    PronyTerm,
    Hookean,
    NeoHookean,
    MooneyRivlin,
    Yeoh,
    Ogden,
    HGO,
    Viscoelastic,
)


def main() -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Example 0101: Constitutive Material Models & Mechanics[/bold cyan]\n"
            "[dim]Linear Elastic, Hyperelastic (Neo-Hookean, Mooney-Rivlin, Yeoh, Ogden), HGO, & Viscoelastic[/dim]",
            title="[bold blue]project-sierra — Basics[/bold blue]",
            border_style="cyan",
        )
    )

    # 1. Base Isotropic Parameters (Soft biological tissue benchmark: E = 15.0 kPa, nu = 0.45)
    console.print("\n[bold yellow][1/3] Instantiating Base Elastic Constants[/bold yellow]")
    base_iso = IsotropicElasticParameters()
    base_iso.set_from_youngs_poisson(
        youngs_modulus=15.0e3,  # 15 kPa
        poissons_ratio=0.45,
        mass_density=1040.0,   # kg/m^3
    )
    mu0 = base_iso.shear_modulus
    k0 = base_iso.bulk_modulus

    console.print(f"  • Young's Modulus (E)     : [green]{base_iso.youngs_modulus:.1f} Pa[/green]")
    console.print(f"  • Poisson's Ratio (nu)    : [green]{base_iso.poissons_ratio:.3f}[/green]")
    console.print(f"  • Initial Shear Modulus   : [green]{mu0:.2f} Pa[/green]")
    console.print(f"  • Initial Bulk Modulus    : [green]{k0:.2f} Pa[/green]")

    # 2. Instantiating Models
    console.print("\n[bold yellow][2/3] Configuring Diverse Constitutive Laws[/bold yellow]")
    
    # Model 1: Hookean
    mat_hooke = Hookean(base_iso)
    
    # Model 2: Neo-Hookean
    mat_neo = NeoHookean(base_iso)

    # Model 3: Mooney-Rivlin
    mr_params = MooneyRivlinParameters.from_constants(
        c10=0.4 * (0.5 * mu0),
        c01=0.6 * (0.5 * mu0),
        bulk_modulus=k0,
        mass_density=base_iso.mass_density,
    )
    mat_mr = MooneyRivlin(mr_params)

    # Model 4: Yeoh (Strain-stiffening)
    yeoh_params = YeohParameters.from_constants(
        c10=0.5 * mu0,
        c20=-120.0,
        c30=450.0,
        bulk_modulus=k0,
        mass_density=base_iso.mass_density,
    )
    mat_yeoh = Yeoh(yeoh_params)

    # Model 5: Ogden (Order 1: Neo-Hookean equivalent with alpha = 2.0)
    ogden_params = OgdenParameters(
        mu=[mu0],
        alpha=[2.0],
        bulk_modulus=k0,
        mass_density=base_iso.mass_density,
    )
    mat_ogden = Ogden(ogden_params)

    # Model 6: HGO (Matrix + 1 fiber family aligned with X)
    hgo_params = HGOParameters(
        c=0.5 * mu0,
        k1=25.0e3,
        k2=10.0,
        kappa=0.1,
        fiber_directions=np.array([[1.0, 0.0, 0.0]]),
        bulk_modulus=k0,
        mass_density=base_iso.mass_density,
    )
    mat_hgo = HGO(hgo_params)

    # Model 7: Viscoelastic (Prony series on Neo-Hookean base)
    prony_terms = [
        PronyTerm(relative_modulus=0.35, relaxation_time=0.05),  # Fast relaxation
        PronyTerm(relative_modulus=0.25, relaxation_time=0.50),  # Slow relaxation
    ]
    mat_visco = Viscoelastic(base_material=mat_neo, prony_series=prony_terms)

    console.print("  [green]✓ Instantiated 7 constitutive models successfully.[/green]")

    # 3. Uniaxial Stretch Tension Comparison
    console.print("\n[bold yellow][3/3] Benchmark: 2nd Piola-Kirchhoff Stress S_xx under Stretch[/bold yellow]")
    stretches = [1.0, 1.05, 1.10, 1.20, 1.35, 1.50]

    table = Table(
        title="Uniaxial Extension: 2nd Piola-Kirchhoff Stress S_xx (Pa)",
        border_style="cyan",
    )
    table.add_column("Stretch (λ)", justify="center", style="bold white")
    table.add_column("Green-Lagrange E_xx", justify="center", style="dim white")
    table.add_column("Hookean", justify="right", style="cyan")
    table.add_column("Neo-Hookean", justify="right", style="yellow")
    table.add_column("Mooney-Rivlin", justify="right", style="blue")
    table.add_column("Yeoh (Stiffening)", justify="right", style="magenta")
    table.add_column("Ogden (N=1)", justify="right", style="green")
    table.add_column("HGO (Fibers)", justify="right", style="bold green")

    for lam in stretches:
        lam_t = 1.0 / np.sqrt(lam)
        F = np.diag([lam, lam_t, lam_t])
        E_xx = 0.5 * (lam**2 - 1.0)

        # Hookean linear approximation (S = C : E)
        E_vec = np.array([E_xx, 0.5 * (lam_t**2 - 1.0), 0.5 * (lam_t**2 - 1.0), 0.0, 0.0, 0.0])
        S_hooke_xx = (mat_hooke.compute_constitutive_matrix() @ E_vec)[0]

        _, S_neo = mat_neo.compute_stress_and_tangent(F)
        _, S_mr = mat_mr.compute_stress_and_tangent(F)
        _, S_yeoh = mat_yeoh.compute_stress_and_tangent(F)
        _, S_ogden = mat_ogden.compute_stress_and_tangent(F)
        _, S_hgo = mat_hgo.compute_stress_and_tangent(F)

        table.add_row(
            f"{lam:.2f}",
            f"{E_xx:.4f}",
            f"{S_hooke_xx:.1f}",
            f"{S_neo[0, 0]:.1f}",
            f"{S_mr[0, 0]:.1f}",
            f"{S_yeoh[0, 0]:.1f}",
            f"{S_ogden[0, 0]:.1f}",
            f"{S_hgo[0, 0]:.1f}",
        )

    console.print(table)
    console.print(
        "\n[bold green]✓ Observation:[/bold green] All nonlinear models agree with Hookean elasticity at small strain (λ <= 1.05)."
    )
    console.print(
        "  At moderate-to-large strains, Yeoh and HGO exhibit rapid strain-stiffening characteristic of collagen fiber recruitment."
    )
    console.print(
        f"  Viscoelastic model retains instantaneous shear modulus G_0 = {mu0:.1f} Pa "
        f"and long-term factor g_infinity = {mat_visco.g_infinity:.2f} (G_inf = {mu0 * mat_visco.g_infinity:.1f} Pa).\n"
    )


if __name__ == "__main__":
    main()
