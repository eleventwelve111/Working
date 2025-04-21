import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from config import wall_thickness, source_to_wall_distance, detector_diameter, ft_to_cm

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def plot_2d_mesh(results, title):
    """
    Plot 2D radiation field

    Parameters:
        results (dict): Results from the simulation
        title (str): Plot title

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    mesh_result = np.array(results['mesh_result'])

    fig, ax = plt.subplots(figsize=(15, 8))

    # Create the mesh grid
    x = np.linspace(-10, source_to_wall_distance + wall_thickness + 200, 101)
    y = np.linspace(-50, 50, 101)
    X, Y = np.meshgrid(x, y)

    # Plot the mesh with logarithmic colorscale
    im = ax.pcolormesh(X, Y, mesh_result.T,
                       norm=LogNorm(vmin=max(mesh_result.min(), 1e-10), vmax=mesh_result.max()),
                       cmap='viridis')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Photon Flux (particles/cm²)')

    # Add wall position
    ax.axvline(x=source_to_wall_distance, color='red', linestyle='-', linewidth=2, label='Wall Front')
    ax.axvline(x=source_to_wall_distance + wall_thickness, color='red', linestyle='-', linewidth=2, label='Wall Back')

    # Add source position
    ax.plot(0, 0, 'ro', markersize=10, label='Source')

    # Add detector position
    detector_x = results['detector_x']
    detector_y = results['detector_y']
    detector_circle = plt.Circle((detector_x, detector_y), detector_diameter / 2,
                                 fill=False, color='blue', linewidth=2, label='Detector')
    ax.add_patch(detector_circle)

    # Add channel
    channel_radius = results['channel_diameter'] / 2
    ax.plot([source_to_wall_distance, source_to_wall_distance + wall_thickness],
            [0, 0], 'y-', linewidth=max(channel_radius * 50, 1), label='Air Channel')

    # Set labels and title
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    # Save figure
    plt.savefig(f"results/mesh_E{results['energy']}_D{results['channel_diameter']}_" +
                f"dist{results['detector_distance']}_ang{results['detector_angle']}.png",
                dpi=300, bbox_inches='tight')

    plt.close(fig)
    return fig


# Add create_radiation_distribution_heatmap function since it's been removed from visualization_advanced.py
def create_radiation_distribution_heatmap(results, title=None):
    """
    Create an enhanced Cartesian heatmap showing radiation distribution from source to detector
    with optimized visualization for this specific shielding problem

    Parameters:
        results (dict): Results from simulation
        title (str, optional): Plot title

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    # Extract mesh data
    mesh_result = np.array(results['mesh_result'])

    # Create figure with higher resolution
    fig, ax = plt.subplots(figsize=(15, 9), dpi=150)

    # Define the extent of the plot (x and y limits)
    x_min = -10  # Source area
    x_max = source_to_wall_distance + wall_thickness + 200  # Past detector
    y_min = -50
    y_max = 50

    # Create an enhanced custom colormap specifically for radiation visualization
    colors_list = [
        (0.0, 0.0, 0.3),  # Dark blue (background/low values)
        (0.0, 0.2, 0.6),  # Blue
        (0.0, 0.5, 0.8),  # Light blue
        (0.0, 0.8, 0.8),  # Cyan
        (0.0, 0.9, 0.3),  # Blue-green
        (0.5, 1.0, 0.0),  # Green
        (0.8, 1.0, 0.0),  # Yellow-green
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.8, 0.0),  # Yellow-orange
        (1.0, 0.6, 0.0),  # Orange
        (1.0, 0.0, 0.0)  # Red (highest intensity)
    ]

    cmap_name = 'EnhancedRadiation'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors_list, N=256)

    # Use contourf for smoother visualization
    # First, create coordinate meshes
    x = np.linspace(x_min, x_max, mesh_result.shape[0])
    y = np.linspace(y_min, y_max, mesh_result.shape[1])
    X, Y = np.meshgrid(x, y)

    # Apply smoothing if needed for better visualization
    smoothed_data = gaussian_filter(mesh_result.T, sigma=1)

    # Set zero values to NaN to make them transparent
    min_nonzero = np.min(smoothed_data[smoothed_data > 0]) / 10
    smoothed_data[smoothed_data < min_nonzero] = np.nan

    # Plot using contourf for a smoother representation with more levels
    levels = np.logspace(np.log10(min_nonzero), np.log10(np.nanmax(smoothed_data)), 20)
    contour = ax.contourf(X, Y, smoothed_data,
                          levels=levels,
                          norm=LogNorm(),
                          cmap=custom_cmap,
                          alpha=0.95,
                          extend='both')

    # Add contour lines for a better indication of dose levels
    contour_lines = ax.contour(X, Y, smoothed_data,
                               levels=levels[::4],  # Fewer contour lines
                               colors='black',
                               alpha=0.3,
                               linewidths=0.5)

    # Add colorbar with scientific notation
    cbar = fig.colorbar(contour, ax=ax, format='%.1e', pad=0.02)
    cbar.set_label('Radiation Flux (particles/cm²/s)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Add a semi-transparent shaded region for the wall
    wall_patch = plt.Rectangle((source_to_wall_distance, y_min),
                               wall_thickness, y_max - y_min,
                               facecolor='gray', alpha=0.5,
                               edgecolor='black', linewidth=1.5,
                               label='Concrete Wall')
    ax.add_patch(wall_patch)

    # Add source position with improved marker
    ax.plot(0, 0, 'ro', markersize=12, markeredgecolor='black', markeredgewidth=1.5, label='Source')

    # Add detector position with improved styling
    detector_x = results['detector_x']
    detector_y = results['detector_y']
    detector_circle = plt.Circle((detector_x, detector_y), detector_diameter / 2,
                                 fill=False, color='red', linewidth=2, label='Detector')
    ax.add_patch(detector_circle)

    # Add beam path line from source to detector with an arrow
    arrow_props = dict(arrowstyle='->', linewidth=2, color='yellow', alpha=0.9)
    beam_arrow = ax.annotate('', xy=(detector_x, detector_y), xytext=(0, 0),
                             arrowprops=arrow_props)

    # Add channel with improved styling
    channel_radius = results['channel_diameter'] / 2
    channel_rect = plt.Rectangle((source_to_wall_distance, -channel_radius),
                                 wall_thickness, 2 * channel_radius,
                                 facecolor='white', alpha=1.0, linewidth=1.5,
                                 edgecolor='black', label='Air Channel')
    ax.add_patch(channel_rect)

    # Add angle indicator if angle is not 0
    angle = results['detector_angle']
    if angle > 0:
        # Draw angle arc
        angle_radius = 50  # Size of the arc
        arc = plt.matplotlib.patches.Arc((source_to_wall_distance + wall_thickness, 0),
                                         angle_radius * 2, angle_radius * 2,
                                         theta1=0, theta2=angle,
                                         color='white', linewidth=2)
        ax.add_patch(arc)
        # Add angle text
        angle_text_x = (source_to_wall_distance + wall_thickness) + angle_radius * 0.7 * np.cos(np.radians(angle / 2))
        angle_text_y = angle_radius * 0.7 * np.sin(np.radians(angle / 2))
        ax.text(angle_text_x, angle_text_y, f"{angle}°", color='white',
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))

    # Set labels and title with improved styling
    ax.set_xlabel('Distance (cm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Lateral Distance (cm)', fontsize=14, fontweight='bold')

    if title is None:
        title = (f"Radiation Distribution: {results['energy']} MeV, Channel Diameter={results['channel_diameter']} cm\n"
                 f"Distance={results['detector_distance']} cm, Angle={results['detector_angle']}°")
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)

    # Add improved legend with better positioning
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(by_label.values(), by_label.keys(),
                       loc='upper right', framealpha=0.9, fontsize=11)
    legend.get_frame().set_edgecolor('black')

    # Add distance markers (concentric circles from wall exit)
    wall_exit_x = source_to_wall_distance + wall_thickness
    for dist in [50, 100, 150]:
        # Use dashed circle
        dist_circle = plt.Circle((wall_exit_x, 0), dist,
                                 fill=False, color='white', linestyle='--', linewidth=1, alpha=0.6)
        ax.add_patch(dist_circle)
        # Add text for distance
        ax.text(wall_exit_x + dist * np.cos(np.radians(45)), dist * np.sin(np.radians(45)),
                f"{dist} cm", color='white', fontsize=9, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

    # Add enhanced grid with better styling
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)  # Place grid below other elements

    # Add scale indicators - distance markers along x-axis
    x_ticks = np.append(np.arange(0, source_to_wall_distance, 50),
                        [source_to_wall_distance, source_to_wall_distance + wall_thickness])
    x_ticks = np.append(x_ticks, np.arange(source_to_wall_distance + wall_thickness, x_max, 50))
    ax.set_xticks(x_ticks)

    # Add detailed information box
    info_text = (f"Source: {results['energy']} MeV Gamma\n"
                 f"Wall: {wall_thickness / ft_to_cm:.1f} ft concrete\n"
                 f"Channel: {results['channel_diameter']} cm diam\n"
                 f"Detector: {results['detector_distance']} cm from wall\n"
                 f"Angle: {results['detector_angle']}°\n"
                 f"Max Dose: {results['dose_rem_per_hr']:.2e} rem/hr")

    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Ensure proper aspect ratio
    ax.set_aspect('equal')

    # Save high-resolution figure
    plt.savefig(f"results/radiation_dist_E{results['energy']}_D{results['channel_diameter']}_" +
                f"dist{results['detector_distance']}_ang{results['detector_angle']}.png",
                dpi=300, bbox_inches='tight')

    return fig


def plot_dose_vs_angle(results_dict, energy):
    """
    Plot dose vs angle for different distances and channel diameters

    Parameters:
        results_dict (dict): Dictionary of results from simulations
        energy (float): Energy in MeV to plot

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    from config import channel_diameters, detector_distances, detector_angles

    fig, ax = plt.subplots(figsize=(14, 10))

    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd', 'x', '*']
    colors = plt.cm.viridis(np.linspace(0, 1, len(channel_diameters) * len(detector_distances)))

    color_idx = 0
    for diameter in channel_diameters:
        for distance in detector_distances:
            angles = []
            doses = []

            for angle in detector_angles:
                key = f"E{energy}_D{diameter}_dist{distance}_ang{angle}"
                if key in results_dict:
                    angles.append(angle)
                    doses.append(results_dict[key]['dose_rem_per_hr'])

            if angles and doses:
                label = f"Diam={diameter} cm, Dist={distance} cm"
                ax.semilogy(angles, doses,
                            marker=markers[color_idx % len(markers)],
                            linestyle=linestyles[color_idx % len(linestyles)],
                            color=colors[color_idx],
                            label=label)
                color_idx += 1

    ax.set_xlabel('Detector Angle (degrees)')
    ax.set_ylabel('Dose Rate (rem/hr)')
    ax.set_title(f'Dose vs Angle - {energy} MeV Gamma Source')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f'results/dose_vs_angle_E{energy}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return fig


def create_polar_dose_heatmap(results_dict, energy, channel_diameter=None):
    """
    Create an enhanced polar heat map visualization of dose distribution

    Parameters:
        results_dict (dict): Dictionary of results from simulations
        energy (float): Energy in MeV to plot
        channel_diameter (float, optional): If specified, show results only for this channel diameter

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    from config import detector_distances, detector_angles, channel_diameters
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, Rbf

    # Set up figure with higher resolution
    fig, ax = plt.subplots(figsize=(10, 10), dpi=120, subplot_kw={'projection': 'polar'})

    # Define grid for interpolation
    r_grid = np.linspace(0, 200, 100)  # Distance from wall: 0 to 200 cm
    theta_grid = np.linspace(0, np.pi / 2, 100)  # Angles: 0 to 90 degrees

    # Create meshgrid for polar coordinates
    r_mesh, theta_mesh = np.meshgrid(r_grid, theta_grid)

    # Initialize dose array with NaN values
    dose_values = np.full((100, 100), np.nan)

    # Collect data points for interpolation
    r_points = []
    theta_points = []
    dose_points = []

    # Track actual data points for marking
    actual_r = []
    actual_theta = []
    actual_dose = []
    actual_diameter = []

    for key, result in results_dict.items():
        parts = key.split('_')
        result_energy = float(parts[0][1:])
        result_diam = float(parts[1][1:])

        # Filter by energy and optionally channel diameter
        if result_energy == energy:
            if channel_diameter is None or result_diam == channel_diameter:
                distance = float(parts[2][4:])
                angle = float(parts[3][3:])

                if 'dose_rem_per_hr' in result:
                    # Convert to polar coordinates
                    r = distance  # Distance from wall
                    theta = np.radians(angle)  # Convert degrees to radians
                    dose = result['dose_rem_per_hr']

                    # Add to points list for interpolation
                    r_points.append(r)
                    theta_points.append(theta)
                    dose_points.append(dose)

                    # Save actual data points
                    actual_r.append(r)
                    actual_theta.append(theta)
                    actual_dose.append(dose)
                    actual_diameter.append(result_diam)

    # If we have data points, perform interpolation
    if len(r_points) > 0:
        # Create combined coordinates
        points = np.vstack((r_points, theta_points)).T

        # Flatten meshgrid for interpolation
        mesh_points = np.vstack((r_mesh.flatten(), theta_mesh.flatten())).T

        # Interpolate using appropriate method based on number of points
        if len(r_points) >= 15:
            # For many points, linear interpolation works well
            interpolator = LinearNDInterpolator(points, dose_points, fill_value=np.min(dose_points) / 10)
        elif len(r_points) >= 4:
            # For moderate number of points, use Radial Basis Function
            rbf = Rbf(r_points, theta_points, dose_points, function='multiquadric', epsilon=5)
            interpolated_doses = rbf(r_mesh.flatten(), theta_mesh.flatten())
            dose_values = interpolated_doses.reshape(r_mesh.shape)
        else:
            # For very few points, use nearest neighbor
            interpolator = NearestNDInterpolator(points, dose_points)

        # Get interpolated values if not already done by RBF
        if 'rbf' not in locals():
            interpolated_doses = interpolator(mesh_points)
            dose_values = interpolated_doses.reshape(r_mesh.shape)

    # Create enhanced colormap (yellow -> green -> blue)
    colors_list = [
        (1.0, 1.0, 0.0),  # Yellow (high dose)
        (0.5, 1.0, 0.0),  # Yellow-green
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.7, 0.7),  # Teal
        (0.0, 0.4, 0.8),  # Blue
        (0.0, 0.0, 0.5),  # Dark blue (low dose)
    ]
    cmap_name = 'EnhancedDoseMap'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors_list, N=256)

    # Plot the heatmap with logarithmic color scale and improved colormap
    vmin = max(np.nanmin(dose_values), 1e-8)  # Avoid negative or zero values
    vmax = max(np.nanmax(dose_values), vmin * 100)

    pcm = ax.pcolormesh(theta_mesh, r_mesh, dose_values,
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                        cmap=custom_cmap, shading='auto')

    # Add enhanced colorbar
    cbar = fig.colorbar(pcm, ax=ax, pad=0.1, format='%.1e')
    cbar.set_label('Dose [rem/hr]', fontsize=12, fontweight='bold')

    # Set up the polar axis
    ax.set_theta_zero_location('N')  # 0 degrees at the top
    ax.set_theta_direction(-1)  # Clockwise

    # Add angle labels (degrees)
    ax.set_xticks(np.radians([0, 15, 30, 45, 60, 75, 90]))
    ax.set_xticklabels(['0°', '15°', '30°', '45°', '60°', '75°', '90°'], fontsize=10)

    # Customize radial ticks and labels
    radii = [50, 100, 150, 200]
    ax.set_rticks(radii)
    ax.set_rgrids(radii, labels=[f"{r} cm" for r in radii], fontsize=10)

    # Add title with styling
    if channel_diameter is None:
        title = f"Dose Distribution: {energy} MeV (All Channel Diameters)"
    else:
        title = f"Dose Distribution: {energy} MeV, Channel Diameter: {channel_diameter} cm"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add actual data points as markers
    if len(actual_r) > 0:
        # Define colors for different diameters if showing all diameters
        if channel_diameter is None:
            unique_diameters = sorted(set(actual_diameter))
            diameter_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_diameters)))
            diameter_color_map = dict(zip(unique_diameters, diameter_colors))

            # Plot with different colors for different diameters
            for r, theta, diam in zip(actual_r, actual_theta, actual_diameter):
                ax.plot(theta, r, 'o', color=diameter_color_map[diam], markersize=6,
                        markeredgecolor='white', markeredgewidth=1)

            # Add legend for diameters
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=diameter_color_map[d],
                                      markeredgecolor='white', markersize=6, label=f'Ø: {d} cm')
                               for d in unique_diameters]
            ax.legend(handles=legend_elements, loc='lower right', title='Channel Diameters')
        else:
            # Just plot points with same color
            ax.plot(actual_theta, actual_r, 'o', color='red', markersize=6,
                    markeredgecolor='white', markeredgewidth=1)

    # Add intensity contours
    contour_levels = np.logspace(np.log10(vmin), np.log10(vmax), 5)
    contours = ax.contour(theta_mesh, r_mesh, dose_values, levels=contour_levels,
                          colors='white', linewidths=0.8, alpha=0.6)

    # Add wall location indicator
    ax.plot(np.linspace(0, np.pi / 2, 100), np.zeros(100), 'k-', linewidth=3)
    ax.text(np.radians(45), 0, 'Wall', color='black', ha='center', va='bottom',
            fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

    # Add dose gradient indicators
    if len(r_points) > 3:
        gradient_text = "Dose decreases with distance and angle"
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.5, 0.92, gradient_text, transform=ax.transAxes, fontsize=10,
                ha='center', va='top', bbox=props)

    # Save high-resolution figure
    if channel_diameter is None:
        filename = f"results/polar_dose_E{energy}.png"
    else:
        filename = f"results/polar_dose_E{energy}_D{channel_diameter}.png"

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return fig


# Functions moved from visualization_advanced.py

def create_radiation_outside_wall_heatmap(results, title=None):
    """
    Create an enhanced close-up Cartesian heatmap showing radiation distribution outside the wall
    with optimized visualization for this specific shielding problem

    Parameters:
        results (dict): Results from simulation
        title (str, optional): Plot title

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    # Extract mesh data
    mesh_result = np.array(results['mesh_result'])

    # Create figure with higher resolution
    fig, ax = plt.subplots(figsize=(14, 11), dpi=150)

    # Define the extent of the plot focused specifically on the area outside the wall
    x_min = source_to_wall_distance + wall_thickness - 5  # Slightly before wall exit
    x_max = source_to_wall_distance + wall_thickness + 150  # 150 cm outside wall
    y_min = -75
    y_max = 75

    # Calculate indices in the mesh corresponding to these limits
    mesh_x_coords = np.linspace(-10, source_to_wall_distance + wall_thickness + 200, mesh_result.shape[0])
    mesh_y_coords = np.linspace(-50, 50, mesh_result.shape[1])

    x_indices = np.logical_and(mesh_x_coords >= x_min, mesh_x_coords <= x_max)
    y_indices = np.logical_and(mesh_y_coords >= y_min, mesh_y_coords <= y_max)

    # Extract the section of the mesh for the region of interest
    x_subset = mesh_x_coords[x_indices]
    y_subset = mesh_y_coords[y_indices]
    outside_wall_data = mesh_result[np.ix_(x_indices, y_indices)]

    # Create coordinate meshes for the plot
    X, Y = np.meshgrid(x_subset, y_subset)

    # Apply adaptive smoothing for better visualization
    sigma = max(1, min(3, 5 / (results['channel_diameter'] + 0.1)))  # Smaller channels need more smoothing
    smoothed_data = gaussian_filter(outside_wall_data.T, sigma=sigma)

    # Set zero or very small values to NaN to make them transparent
    min_nonzero = np.max([np.min(smoothed_data[smoothed_data > 0]) / 10, 1e-12])
    smoothed_data[smoothed_data < min_nonzero] = np.nan

    # Create an enhanced custom colormap specifically for radiation visualization
    colors_list = [
        (0.0, 0.0, 0.3),  # Dark blue (background/low values)
        (0.0, 0.2, 0.6),  # Blue
        (0.0, 0.5, 0.8),  # Light blue
        (0.0, 0.8, 0.8),  # Cyan
        (0.0, 0.9, 0.3),  # Blue-green
        (0.5, 1.0, 0.0),  # Green
        (0.8, 1.0, 0.0),  # Yellow-green
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.8, 0.0),  # Yellow-orange
        (1.0, 0.6, 0.0),  # Orange
        (1.0, 0.0, 0.0)  # Red (highest intensity)
    ]

    cmap_name = 'EnhancedRadiation'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors_list, N=256)

    # Use contourf for smoother visualization with more levels
    levels = np.logspace(np.log10(min_nonzero), np.log10(np.nanmax(smoothed_data)), 20)
    contour = ax.contourf(X, Y, smoothed_data,
                          levels=levels,
                          norm=LogNorm(),
                          cmap=custom_cmap,
                          alpha=0.95,
                          extend='both')

    # Add contour lines for a better indication of dose levels
    contour_lines = ax.contour(X, Y, smoothed_data,
                               levels=levels[::4],  # Fewer contour lines
                               colors='black',
                               alpha=0.3,
                               linewidths=0.5)

    # Add colorbar with scientific notation
    cbar = fig.colorbar(contour, ax=ax, format='%.1e', pad=0.01)
    cbar.set_label('Radiation Flux (particles/cm²/s)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Add wall back position with improved styling
    wall_exit_x = source_to_wall_distance + wall_thickness
    ax.axvline(x=wall_exit_x, color='black', linestyle='-', linewidth=2.5, label='Wall Back')

    # Draw a small section of the wall for context
    wall_section = plt.Rectangle((x_min, y_min), wall_exit_x - x_min, y_max - y_min,
                                 facecolor='gray', alpha=0.5, edgecolor='black')
    ax.add_patch(wall_section)

    # Add detector position with improved styling
    detector_x = results['detector_x']
    detector_y = results['detector_y']

    # Only show detector if it's in the displayed area
    if x_min <= detector_x <= x_max and y_min <= detector_y <= y_max:
        detector_circle = plt.Circle((detector_x, detector_y), detector_diameter / 2,
                                     fill=False, edgecolor='red', linewidth=2, label='Detector')
        ax.add_patch(detector_circle)

        # Add beam path from channel to detector with an arrow
        arrow_props = dict(arrowstyle='->', linewidth=2, color='yellow', alpha=0.9)
        beam_arrow = ax.annotate('', xy=(detector_x, detector_y), xytext=(wall_exit_x, 0),
                                 arrowprops=arrow_props)

    # Add channel exit with improved styling
    channel_radius = results['channel_diameter'] / 2
    channel_exit = plt.Circle((wall_exit_x, 0), channel_radius,
                              facecolor='white', alpha=1.0, edgecolor='black', linewidth=1.5,
                              label='Channel Exit')
    ax.add_patch(channel_exit)

    # Add concentric circles to show distance from channel exit
    for radius in [25, 50, 75, 100]:
        # Draw dashed circle
        distance_circle = plt.Circle((wall_exit_x, 0), radius,
                                     fill=False, edgecolor='white', linestyle='--', linewidth=1, alpha=0.6)
        ax.add_patch(distance_circle)

        # Add distance label along 45° angle
        angle = 45
        label_x = wall_exit_x + radius * np.cos(np.radians(angle))
        label_y = radius * np.sin(np.radians(angle))
        ax.text(label_x, label_y, f"{radius} cm", color='white', fontsize=9,
                ha='center', va='center', rotation=angle,
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

    # Add detector angle indication if not at 0°
    angle = results['detector_angle']
    if angle > 0:
        # Draw angle arc
        angle_radius = 30
        arc = plt.matplotlib.patches.Arc((wall_exit_x, 0),
                                         angle_radius * 2, angle_radius * 2,
                                         theta1=0, theta2=angle,
                                         color='white', linewidth=2)
        ax.add_patch(arc)

        # Add angle text at arc midpoint
        angle_text_x = wall_exit_x + angle_radius * 0.7 * np.cos(np.radians(angle / 2))
        angle_text_y = angle_radius * 0.7 * np.sin(np.radians(angle / 2))
        ax.text(angle_text_x, angle_text_y, f"{angle}°", color='white',
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))

    # Set labels and title with improved styling
    ax.set_xlabel('Distance (cm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Lateral Distance (cm)', fontsize=14, fontweight='bold')

    if title is None:
        title = (f"Radiation Distribution Outside Wall\n"
                 f"{results['energy']} MeV Gamma, Channel Diameter: {results['channel_diameter']} cm")
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)

    # Add improved legend with better positioning
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(by_label.values(), by_label.keys(),
                       loc='upper right', framealpha=0.9, fontsize=11)
    legend.get_frame().set_edgecolor('black')

    # Add enhanced grid with better styling
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Add detailed information box
    info_text = (f"Source: {results['energy']} MeV Gamma\n"
                 f"Wall: {wall_thickness / ft_to_cm:.1f} ft concrete\n"
                 f"Channel: {results['channel_diameter']} cm diam\n"
                 f"Detector: {results['detector_distance']} cm from wall\n"
                 f"Angle: {results['detector_angle']}°\n"
                 f"Dose Rate: {results['dose_rem_per_hr']:.2e} rem/hr")

    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Highlight the region of 10% or greater of the maximum dose
    if not np.isnan(np.max(smoothed_data)):
        high_dose_level = np.max(smoothed_data) * 0.1
        high_dose_contour = ax.contour(X, Y, smoothed_data,
                                       levels=[high_dose_level],
                                       colors=['red'],
                                       linewidths=2)

        # Add label for high dose region
        plt.clabel(high_dose_contour, inline=True, fontsize=9,
                   fmt=lambda x: "10% of Max Dose")

    # Ensure proper aspect ratio
    ax.set_aspect('equal')

    # Save high-resolution figure
    plt.savefig(f"results/outside_wall_E{results['energy']}_D{results['channel_diameter']}_" +
                f"dist{results['detector_distance']}_ang{results['detector_angle']}.png",
                dpi=300, bbox_inches='tight')

    return fig


def create_comprehensive_angle_plot(results_dict, energy):
    """
    Create an enhanced comprehensive plot with:
    - Angles on the x-axis
    - Dose on the y-axis (log scale)
    - Different curves for each channel diameter
    - Points on each curve representing different distances

    Parameters:
        results_dict (dict): Dictionary of results from simulations
        energy (float): Energy in MeV to plot

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    from config import channel_diameters, detector_distances, detector_angles

    plt.figure(figsize=(14, 10), dpi=120)

    # Create enhanced color palette for different diameters
    diameter_colors = plt.cm.viridis_r(np.linspace(0, 0.9, len(channel_diameters)))

    # Define markers for different distances
    distance_markers = ['o', 's', '^', 'd', 'p', '*']
    marker_sizes = [10, 9, 9, 8, 8, 8]  # Slightly different sizes for visual distinction

    # Track plotted data for legend
    diameter_handles = []
    distance_handles = []

    # For each diameter, create a curve
    for d_idx, diameter in enumerate(sorted(channel_diameters)):
        color = diameter_colors[d_idx]

        # For each distance, collect angle and dose data
        for dist_idx, distance in enumerate(detector_distances):
            marker = distance_markers[dist_idx % len(distance_markers)]
            marker_size = marker_sizes[dist_idx % len(marker_sizes)]

            angles = []
            doses = []

            # Collect data for all angles at this distance and diameter
            for angle in detector_angles:
                key = f"E{energy}_D{diameter}_dist{distance}_ang{angle}"
                if key in results_dict and 'dose_rem_per_hr' in results_dict[key]:
                    angles.append(angle)
                    doses.append(results_dict[key]['dose_rem_per_hr'])

            if angles and doses:
                # Sort by angle
                sorted_idx = np.argsort(angles)
                sorted_angles = [angles[i] for i in sorted_idx]
                sorted_doses = [doses[i] for i in sorted_idx]

                # Plot data points
                if dist_idx == 0:  # First distance for this diameter
                    # Plot line with label for diameter
                    line, = plt.semilogy(sorted_angles, sorted_doses, '-',
                                         color=color, linewidth=2.5,
                                         label=f'Diameter: {diameter} cm')
                    diameter_handles.append(line)
                else:
                    # Plot line without label (to avoid duplicates)
                    plt.semilogy(sorted_angles, sorted_doses, '-',
                                 color=color, linewidth=2.5, alpha=0.9)

                # Plot markers for each distance
                if d_idx == 0:  # First diameter for this distance
                    # Plot markers with label for distance
                    point, = plt.semilogy(sorted_angles, sorted_doses, marker,
                                          color=color, markersize=marker_size,
                                          markeredgecolor='black', markeredgewidth=0.8,
                                          label=f'Distance: {distance} cm')
                    distance_handles.append(point)
                else:
                    # Plot markers without label
                    plt.semilogy(sorted_angles, sorted_doses, marker,
                                 color=color, markersize=marker_size,
                                 markeredgecolor='black', markeredgewidth=0.8)

    # Add labels and title with enhanced styling
    plt.xlabel('Detector Angle (degrees)', fontsize=12, fontweight='bold')
    plt.ylabel('Dose Rate (rem/hr)', fontsize=12, fontweight='bold')
    plt.title(f'Dose Rate vs. Angle for {energy} MeV Source\nEffect of Channel Diameter and Distance',
              fontsize=14, fontweight='bold', pad=10)

    # Add enhanced grid
    plt.grid(True, which='both', linestyle='--', alpha=0.6)

    # Set x-axis ticks with all angles
    plt.xticks(detector_angles)

    # Add minor grid lines
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)

    # Create enhanced two-part legend
    if diameter_handles and distance_handles:
        # First legend for diameters (lines)
        legend1 = plt.legend(handles=diameter_handles, loc='upper right',
                             title='Channel Diameter', title_fontsize=12,
                             fontsize=10, framealpha=0.9)
        legend1.get_frame().set_edgecolor('black')

        # Add the first legend manually so we can create a second one
        plt.gca().add_artist(legend1)

        # Second legend for distances (markers)
        legend2 = plt.legend(handles=distance_handles, loc='lower left',
                             title='Distance from Wall', title_fontsize=12,
                             fontsize=10, framealpha=0.9)
        legend2.get_frame().set_edgecolor('black')

    # Add annotations explaining the data
    plt.annotate('Dose decreases with increasing angle',
                 xy=(30, plt.gca().get_ylim()[0] * 10),
                 xytext=(30, plt.gca().get_ylim()[0] * 3),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add plot explanations
    info_text = (f"Energy: {energy} MeV\n"
                 f"• Each curve represents a channel diameter\n"
                 f"• Each point represents a measurement distance\n"
                 f"• Y-axis is logarithmic scale")
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
    plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)

    # Save high-resolution figure
    plt.savefig(f"results/comprehensive_angle_plot_E{energy}.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    return plt.figure()  # Return a new figure 