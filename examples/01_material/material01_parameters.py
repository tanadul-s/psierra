from rich import print

from psierra.material import IsotropicElasticParameters


def main():
    params_a6061 = IsotropicElasticParameters()
    params_a6061.set_from_youngs_poisson(youngs_modulus=68.0e+9, poissons_ratio=0.33, mass_density=2700)
    print(params_a6061)
    
    params_ss304 = IsotropicElasticParameters()
    params_ss304.set_from_bulk_shear(bulk_modulus=134e+9, shear_modulus=74e+9, mass_density=7930)
    print(params_ss304)
    
    params_ss316 = IsotropicElasticParameters()
    print(params_ss316)
    

if __name__ == "__main__":
    main()