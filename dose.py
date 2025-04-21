def get_mass_energy_absorption_coefficient(energy, material="tissue"):
    """
    Get mass energy absorption coefficient (μen/ρ) for a given energy and material

    Parameters:
        energy (float): Gamma-ray energy in MeV
        material (str): Material name (tissue, air, concrete, void)

    Returns:
        float: Mass energy absorption coefficient in cm²/g
    """
    # Reference data for μen/ρ (cm²/g) at different energies
    # Source: NIST XCOM database (https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html)

    # Tissue (ICRU-44)
    if material.lower() == "tissue":
        energies = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15,
                    0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
                    6.0, 8.0, 10.0, 15.0]
        coefficients = [4.742, 1.334, 0.5389, 0.1537, 0.0683, 0.0416, 0.0318, 0.0262,
                        0.0256, 0.0277, 0.0297, 0.0319, 0.0328, 0.0330, 0.0329, 0.0321,
                        0.0311, 0.0284, 0.0264, 0.0239, 0.0227, 0.0220, 0.0218, 0.0218,
                        0.0222, 0.0240]

    # Air
    elif material.lower() == "air":
        energies = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15,
                    0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
                    6.0, 8.0, 10.0, 15.0]
        coefficients = [4.742, 1.267, 0.5099, 0.1482, 0.0661, 0.0382, 0.0290, 0.0241,
                        0.0231, 0.0251, 0.0268, 0.0287, 0.0295, 0.0297, 0.0296, 0.0289,
                        0.0280, 0.0256, 0.0238, 0.0216, 0.0205, 0.0199, 0.0197, 0.0197,
                        0.0201, 0.0217]

    # Concrete (ANSI/ANS-6.4-2006)
    elif material.lower() == "concrete":
        energies = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15,
                    0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
                    6.0, 8.0, 10.0, 15.0]
        coefficients = [5.239, 1.448, 0.5775, 0.1539, 0.0678, 0.0412, 0.0321, 0.0273,
                        0.0268, 0.0289, 0.0308, 0.0328, 0.0335, 0.0336, 0.0333, 0.0323,
                        0.0312, 0.0284, 0.0264, 0.0240, 0.0229, 0.0224, 0.0223, 0.0225,
                        0.0230, 0.0252]

    # Void - effectively zero absorption
    elif material.lower() == "void":
        return 0.0

    else:
        raise ValueError(f"Material '{material}' not supported. Use 'tissue', 'air', 'concrete', or 'void'.")

    # Linear interpolation
    if energy <= energies[0]:
        return coefficients[0]
    elif energy >= energies[-1]:
        return coefficients[-1]
    else:
        for i in range(len(energies) - 1):
            if energies[i] <= energy <= energies[i + 1]:
                fraction = (energy - energies[i]) / (energies[i + 1] - energies[i])
                return coefficients[i] + fraction * (coefficients[i + 1] - coefficients[i])

    # Default return if interpolation fails
    return coefficients[np.argmin(np.abs(np.array(energies) - energy))]




# Material densities in g/cm³
material_densities = {
    "tissue": 1.0,  # From your materials definition
    "air": 0.001205,  # From your materials definition
    "concrete": 2.3,  # From your materials definition
    "void": 1e-10,  # From your materials definition
    "lead": 11.35,  # Keep for potential future use
    "iron": 7.87  # Keep for potential future use
}


def calculate_kerma(energy, flux, material="tissue"):
    """
    Calculate KERMA (Kinetic Energy Released per unit MAss)

    Parameters:
        energy (float): Gamma-ray energy in MeV
        flux (float): Particle flux in photons/cm²-s
        material (str): Material name (tissue, air, water)

    Returns:
        float: KERMA rate in Gy/s (J/kg-s)
    """
    # Get mass energy absorption coefficient (cm²/g)
    mu_en_rho = get_mass_energy_absorption_coefficient(energy, material)

    # Convert photon energy from MeV to joules
    energy_joules = energy * 1.602e-13  # 1 MeV = 1.602e-13 J

    # Calculate KERMA (Gy/s = J/kg-s)
    # K = Φ * E * (μen/ρ) * conversion factors
    kerma = flux * energy_joules * mu_en_rho * 0.1  # 0.1 converts from (J/g-s) to (J/kg-s)

    return kerma


def kerma_to_equivalent_dose(kerma, radiation_weighting_factor=1.0):
    """
    Convert KERMA to equivalent dose

    Parameters:
        kerma (float): KERMA in Gy/s
        radiation_weighting_factor (float): Radiation weighting factor (1.0 for photons)

    Returns:
        float: Equivalent dose rate in Sv/s
    """
    # For photons, the radiation weighting factor is 1.0
    # For other radiation types, different factors would apply
    return kerma * radiation_weighting_factor


def convert_heating_to_dose_rate(heating, density, energy=None, flux=None):
    """
    Convert heating rate (energy deposition) to dose rate

    Parameters:
        heating (float): Heating rate in W/cm³
        density (float): Material density in g/cm³
        energy (float, optional): Photon energy in MeV (for spectrum correction)
        flux (float, optional): Particle flux in photons/cm²-s (for spectrum correction)

    Returns:
        tuple: (dose_rate_gy_per_s, dose_rate_sv_per_s, dose_rate_rem_per_hr)
    """
    # Convert W/cm³ to Gy/s (J/kg-s)
    # 1 W/cm³ = 1 J/s/cm³
    # Divide by density (g/cm³) to get J/s/g
    # Multiply by 1000 to convert J/s/g to J/s/kg (Gy/s)
    dose_rate_gy_per_s = heating / density * 1000

    # Apply spectrum correction if energy and flux are provided
    if energy is not None and flux is not None:
# Calculate expected KERMA

expected_kerma = calculate_kerma(energy, flux)

# Calculate correction factor
if expected_kerma > 0:
    correction = expected_kerma / dose_rate_gy_per_s
    dose_rate_gy_per_s *= correction

    # Convert to Sv/s (for photons, 1 Gy = 1 Sv)
    dose_rate_sv_per_s = dose_rate_gy_per_s

    # Convert to rem/hr
    # 1 Sv = 100 rem, 1 hr = 3600 s
    dose_rate_rem_per_hr = dose_rate_sv_per_s * 100 * 3600

    return (dose_rate_gy_per_s, dose_rate_sv_per_s, dose_rate_rem_per_hr)


def estimate_dose_from_heating_tally(heating_result, material="tissue", energy=None):
    """
    Estimate dose from a heating tally (e.g., from MCNP F6 or F7 tally)

    Parameters:
        heating_result (float): Heating tally result in MeV/g per source particle
        material (str): Material name
        energy (float, optional): Source photon energy in MeV

    Returns:
        float: Dose rate in rem/hr per source particle
    """

    # Material densities in g/cm³
    material_densities = {
        "tissue": 1.04,
        "water": 1.0,
        "air": 0.001205,
        "concrete": 2.3,
        "lead": 11.35,
        "iron": 7.87
    }

    # Get density for the material
    if material.lower() in material_densities:
        density = material_densities[material.lower()]
    else:
        raise ValueError(f"Material '{material}' density not defined")

    # Convert MeV/g to J/kg (Gy)
    # 1 MeV = 1.602e-13 J
    dose_gy = heating_result * 1.602e-13 * 1000  # *1000 to convert from J/g to J/kg

    # For photons and electrons, 1 Gy = 1 Sv
    dose_sv = dose_gy

    # Convert to rem
    # 1 Sv = 100 rem
    dose_rem = dose_sv * 100

    # Return dose in rem per source particle
    return dose_rem

# Example: Calculate KERMA from flux
energy = 1.0  # MeV
flux = 1e8  # photons/cm²-s

# Calculate KERMA
kerma_gy_per_s = calculate_kerma(energy, flux, material="tissue")

# Convert to equivalent dose rate
dose_rate_sv_per_s = kerma_to_equivalent_dose(kerma_gy_per_s)

# Convert to rem/hr for comparison with other functions
dose_rate_rem_per_hr = dose_rate_sv_per_s * 100 * 3600  # 100 rem/Sv, 3600 s/hr

# Compare with the existing method
dose_from_flux = estimate_dose_from_flux(energy, flux)

print(f"KERMA-based dose rate: {dose_rate_rem_per_hr:.4e} rem/hr")
print(f"Flux-based dose rate:  {dose_from_flux:.4e} rem/hr")
print(f"Ratio: {dose_rate_rem_per_hr/dose_from_flux:.4f}")



# Example: Convert heating rate from CAD calculations
heating_rate = 2.5e-6  # W/cm³
material_density = 1.0  # g/cm³ (water)

# Convert to dose rates
gy_per_s, sv_per_s, rem_per_hr = convert_heating_to_dose_rate(heating_rate, material_density)

print(f"Dose rate: {rem_per_hr:.4e} rem/hr")

def analyze_results(self, results, method="flux"):
    """Analyze results using different dose calculation methods"""
    if method == "flux":
        # Use existing flux-to-dose conversion
        return self._analyze_flux_based_dose(results)
    elif method == "kerma":
        # Use KERMA-based calculation
        return self._analyze_kerma_based_dose(results)
    elif method == "heating":
        # Use heating-based calculation
        return self._analyze_heating_based_dose(results)
    else:
        raise ValueError(f"Unknown analysis method: {method}")
