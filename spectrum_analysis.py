import numpy as np
import matplotlib.pyplot as plt
from config import channel_diameters, gamma_energies, detector_distances, detector_angles


def plot_energy_spectrum_by_distance(results_dict, energy, channel_diameter, detector_angles=[0]):
    """
    Plot photon energy spectrum as a function of distance behind the wall.

    Parameters:
        results_dict (dict): Dictionary of simulation results
        energy (float): Energy level to visualize (MeV)
        channel_diameter (float): Channel diameter to visualize (cm)
        detector_angles (list): List of angles to include (default: [0] for direct line-of-sight)

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    # Create figure
    plt.figure(figsize=(12, 8))

    # Define colors for different distances
    colors = plt.cm.viridis(np.linspace(0, 1, len(detector_distances)))

    # Keep track of which distances have been plotted
    plotted_distances = []

    # Plot spectrum for each distance
    for i, distance in enumerate(detector_distances):
        for angle in detector_angles:
            key = f"E{energy}_D{channel_diameter}_dist{distance}_ang{angle}"
            if key in results_dict and 'spectrum' in results_dict[key]:
                spectrum_data = np.array(results_dict[key]['spectrum'])

                # Skip if spectrum is all zeros or too small
                if np.sum(spectrum_data) < 1e-10:
                    continue

                # Get energy bins from the first available result
                if 'energy_bins' in results_dict[key]:
                    energy_bins = np.array(results_dict[key]['energy_bins'])
                    energy_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:]) / 1e6  # Convert to MeV
                else:
                    # Approximate energy bins if not available
                    energy_centers = np.logspace(np.log10(0.01), np.log10(10), len(spectrum_data))

                # Plot spectrum with distance-specific color
                plt.loglog(energy_centers, spectrum_data,
                           color=colors[i],
                           label=f"{distance} cm",
                           linewidth=2)

                plotted_distances.append(distance)

    # Add labels and title
    plt.xlabel('Photon Energy (MeV)')
    plt.ylabel('Flux per Energy Bin (photons/cm²/s)')
    plt.title(
        f'Photon Energy Spectrum vs. Distance Behind Wall\nEnergy: {energy} MeV, Channel Diameter: {channel_diameter} cm, Angle: {detector_angles[0]}°')

    # Add grid
    plt.grid(True, which='both', linestyle='--', alpha=0.6)

    # Add legend if we've plotted any data
    if plotted_distances:
        # Sort legend entries by distance
        handles, labels = plt.gca().get_legend_handles_labels()
        sorted_indices = sorted(range(len(plotted_distances)), key=lambda i: plotted_distances[i])
        plt.legend([handles[i] for i in sorted_indices], [labels[i] for i in sorted_indices],
                   title='Distance Behind Wall', loc='best')

        # Save figure
        angle_str = '-'.join(str(a) for a in detector_angles)
        plt.savefig(f"results/energy_spectrum_E{energy}_D{channel_diameter}_ang{angle_str}.png",
                    dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.close()
    return fig


def create_comprehensive_spectrum_plots(results_dict):
    """
    Create comprehensive energy spectrum plots for all configurations.
    Shows how spectrum changes with distance, energy, and channel diameter.
    """
    # Create plots for each energy and channel diameter combination
    for energy in gamma_energies:
        for channel_diameter in channel_diameters:
            # Create spectrum plot for straight-line (0°) angle
            plot_energy_spectrum_by_distance(results_dict, energy, channel_diameter, [0])

            # Also create spectrum plots for 15° and 45° angles if available
            if any(f"E{energy}_D{channel_diameter}_dist{d}_ang15" in results_dict for d in detector_distances):
                plot_energy_spectrum_by_distance(results_dict, energy, channel_diameter, [15])

            if any(f"E{energy}_D{channel_diameter}_dist{d}_ang45" in results_dict for d in detector_distances):
                plot_energy_spectrum_by_distance(results_dict, energy, channel_diameter, [45])

    # Create a combined plot showing spectra for different energies at fixed distance and channel
    distance = detector_distances[0]  # Use first distance (30 cm)
    channel_diameter = channel_diameters[1]  # Use second diameter (0.5 cm)
    angle = 0  # Use straight-line angle

    plt.figure(figsize=(12, 8))

    for energy in gamma_energies:
        key = f"E{energy}_D{channel_diameter}_dist{distance}_ang{angle}"
        if key in results_dict and 'spectrum' in results_dict[key]:
            spectrum_data = np.array(results_dict[key]['spectrum'])

            # Skip if spectrum is all zeros or too small
            if np.sum(spectrum_data) < 1e-10:
                continue

            # Get energy bins from the first available result
            if 'energy_bins' in results_dict[key]:
                energy_bins = np.array(results_dict[key]['energy_bins'])
                energy_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:]) / 1e6  # Convert to MeV
            else:
                # Approximate energy bins if not available
                energy_centers = np.logspace(np.log10(0.01), np.log10(10), len(spectrum_data))

            # Plot spectrum
            plt.loglog(energy_centers, spectrum_data,
                       label=f"{energy} MeV",
                       linewidth=2)

    plt.xlabel('Photon Energy (MeV)')
    plt.ylabel('Flux per Energy Bin (photons/cm²/s)')
    plt.title(
        f'Photon Energy Spectra for Different Source Energies\nDistance: {distance} cm, Channel Diameter: {channel_diameter} cm')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(title='Source Energy', loc='best')
    plt.savefig(f"results/energy_spectrum_comparison_dist{distance}_D{channel_diameter}.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_spectrum_intensity_vs_distance(results_dict, energy, channel_diameter, angle=0):
    """
    Plot the falloff of spectrum intensity with distance for different energy ranges.

    Parameters:
        results_dict (dict): Dictionary of simulation results
        energy (float): Source energy to visualize (MeV)
        channel_diameter (float): Channel diameter to visualize (cm)
        angle (float): Detector angle (default: 0)

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    plt.figure(figsize=(10, 8))

    # Collect data for different distances
    distances = []
    low_energy_flux = []  # 0-20% of source energy
    mid_energy_flux = []  # 20-80% of source energy
    high_energy_flux = []  # 80-100% of source energy
    total_flux = []  # All energies

    for distance in detector_distances:
        key = f"E{energy}_D{channel_diameter}_dist{distance}_ang{angle}"
        if key in results_dict and 'spectrum' in results_dict[key]:
            spectrum_data = np.array(results_dict[key]['spectrum'])

            # Skip if spectrum is all zeros or too small
            if np.sum(spectrum_data) < 1e-10:
                continue

            # Get energy bins from the result
            if 'energy_bins' in results_dict[key]:
                energy_bins = np.array(results_dict[key]['energy_bins'])
                energy_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:]) / 1e6  # Convert to MeV
            else:
                # Approximate energy bins if not available
                energy_centers = np.logspace(np.log10(0.01), np.log10(10), len(spectrum_data))

            # Determine energy range indices
            low_indices = energy_centers <= 0.2 * energy
            mid_indices = (energy_centers > 0.2 * energy) & (energy_centers <= 0.8 * energy)
            high_indices = energy_centers > 0.8 * energy

            # Calculate flux in each energy range
            low_flux = np.sum(spectrum_data[low_indices]) if any(low_indices) else 0
            mid_flux = np.sum(spectrum_data[mid_indices]) if any(mid_indices) else 0
            high_flux = np.sum(spectrum_data[high_indices]) if any(high_indices) else 0
            total = np.sum(spectrum_data)

            # Add to lists
            distances.append(distance)
            low_energy_flux.append(low_flux)
            mid_energy_flux.append(mid_flux)
            high_energy_flux.append(high_flux)
            total_flux.append(total)

    # Plot if we have data
    if distances:
        # Sort all data by distance
        sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
        sorted_distances = [distances[i] for i in sorted_indices]
        sorted_low = [low_energy_flux[i] for i in sorted_indices]
        sorted_mid = [mid_energy_flux[i] for i in sorted_indices]
        sorted_high = [high_energy_flux[i] for i in sorted_indices]
        sorted_total = [total_flux[i] for i in sorted_indices]

        # Plot all data
        plt.semilogy(sorted_distances, sorted_total, 'k-', linewidth=2, label='Total Flux')
        plt.semilogy(sorted_distances, sorted_low, 'b-', linewidth=2, label=f'Low Energy (<{0.2 * energy:.2f} MeV)')
        plt.semilogy(sorted_distances, sorted_mid, 'g-', linewidth=2,
                     label=f'Mid Energy ({0.2 * energy:.2f}-{0.8 * energy:.2f} MeV)')
        plt.semilogy(sorted_distances, sorted_high, 'r-', linewidth=2, label=f'High Energy (>{0.8 * energy:.2f} MeV)')

        # Add labels and title
        plt.xlabel('Distance Behind Wall (cm)')
        plt.ylabel('Flux (photons/cm²/s)')
        plt.title(
            f'Photon Flux vs. Distance Behind Wall\nEnergy: {energy} MeV, Channel Diameter: {channel_diameter} cm, Angle: {angle}°')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend(loc='best')

        # Save figure
        plt.savefig(f"results/flux_vs_distance_E{energy}_D{channel_diameter}_ang{angle}.png",
                    dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.close()
    return fig 