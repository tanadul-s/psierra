from rich import print

from psierra.material import IsotropicElasticParameters, Hookean


def main():

    # 1. Create parameters
    params = IsotropicElasticParameters()
    params.set_from_youngs_poisson(
        youngs_modulus=1.0e6, poissons_ratio=0.3, mass_density=1000.0
    )

    # 2. Instantiate material
    mat = Hookean(params)

    # 3. Compute constitutive matrix
    C = mat.compute_constitutive_matrix()
    print("Constitutive Matrix C (6x6):\n", C)


if __name__ == "__main__":
    main()
