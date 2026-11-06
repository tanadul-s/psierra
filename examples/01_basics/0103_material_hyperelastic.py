from __future__ import annotations
import numpy as np
from rich.console import Console
from rich.table import Table

from psierra.material import (
    IsotropicElasticParameters,
    MooneyRivlinParameters,
    YeohParameters,
    NeoHookean,
    MooneyRivlin,
    Yeoh,
)


def main() -> None:
    console = Console()
    console.print(
        "[bold cyan]======================================================================[/bold cyan]"
    )
    console.print(
        "[bold cyan] Example 0103: Hyperelastic Material Models (Neo-Hookean, Mooney-Rivlin, Yeoh)[/bold cyan]"
    )
    console.print(
        "[bold cyan]======================================================================[/bold cyan]\n"
    )

    # 1. Base isotropic parameters for soft biological parenchyma (E = 15.0 kPa, nu = 0.45)
    base_iso = IsotropicElasticParameters()
    base_iso.set_from_youngs_poisson(
        youngs_modulus=15.0e3,
        poissons_ratio=0.45,
        mass_density=1040.0,
    )
    mu0 = base_iso.shear_modulus
    k0 = base_iso.bulk_modulus

    console.print(f"• Baseline Shear Modulus (mu_0) : [green]{mu0:.2f} Pa[/green]")
    console.print(f"• Baseline Bulk Modulus  (K_0)  : [green]{k0:.2f} Pa[/green]\n")

    # 2. Instantiate Model A: Neo-Hookean (Initialized directly from IsotropicElasticParameters)
    mat_neo = NeoHookean(base_iso)

    # 3. Instantiate Model B: Mooney-Rivlin (Two-parameter rubber/elastomer with c10 = 0.4*mu, c01 = 0.1*mu)
    mr_params = MooneyRivlinParameters.from_constants(
        c10=0.4 * (0.5 * mu0),
        c01=0.6 * (0.5 * mu0),
        bulk_modulus=k0,
        mass_density=base_iso.mass_density,
    )
    mat_mr = MooneyRivlin(mr_params)

    # 4. Instantiate Model C: Yeoh (Three-parameter soft tissue model with physiological strain-stiffening)
    yeoh_params = YeohParameters.from_constants(
        c10=0.5 * mu0,
        c20=-120.0,
        c30=450.0,
        bulk_modulus=k0,
        mass_density=base_iso.mass_density,
    )
    mat_yeoh = Yeoh(yeoh_params)

    # Verify that all three models match the exact same small-strain shear modulus
    console.print(
        f"• Neo-Hookean Initial Shear Modulus : [green]{mat_neo.compute_constitutive_matrix()[3, 3]:.2f} Pa[/green]"
    )
    console.print(
        f"• Mooney-Rivlin Initial Shear Modulus: [green]{mat_mr.compute_constitutive_matrix()[3, 3]:.2f} Pa[/green]"
    )
    console.print(
        f"• Yeoh Initial Shear Modulus         : [green]{mat_yeoh.compute_constitutive_matrix()[3, 3]:.2f} Pa[/green]\n"
    )

    # 5. Uniaxial Stretch Tension Comparison
    # Under unconstrained uniaxial stretch lambda along X, transverse stretches are lambda_t = 1 / sqrt(lambda)
    stretch_values = [1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0]

    table = Table(
        title="Uniaxial Tension: 2nd Piola-Kirchhoff Stress S_xx (Pa)",
        border_style="cyan",
    )
    table.add_column("Stretch (lambda)", style="bold white", justify="center")
    table.add_column("Strain (E_xx)", style="dim white", justify="center")
    table.add_column("Neo-Hookean", style="yellow", justify="right")
    table.add_column("Mooney-Rivlin", style="blue", justify="right")
    table.add_column("Yeoh (Strain-Stiffening)", style="bold green", justify="right")

    for lam in stretch_values:
        lam_t = 1.0 / np.sqrt(lam)
        F = np.diag([lam, lam_t, lam_t])
        E_xx = 0.5 * (lam**2 - 1.0)

        _, S_neo = mat_neo.compute_stress_and_tangent(F)
        _, S_mr = mat_mr.compute_stress_and_tangent(F)
        _, S_yeoh = mat_yeoh.compute_stress_and_tangent(F)

        table.add_row(
            f"{lam:.2f}",
            f"{E_xx:.4f}",
            f"{S_neo[0, 0]:.1f}",
            f"{S_mr[0, 0]:.1f}",
            f"{S_yeoh[0, 0]:.1f}",
        )

    console.print(table)
    console.print(
        "\n[bold green]✓ Observation:[/bold green] At small strain (lambda = 1.10), all models yield close agreement."
    )
    console.print(
        "[bold green]✓ Observation:[/bold green] At large strains (lambda >= 1.60), Yeoh exhibits dramatic physiological strain-stiffening"
    )
    console.print(
        "  characteristic of collagen fiber recruitment in biological organ tissue.\n"
    )


if __name__ == "__main__":
    main()

