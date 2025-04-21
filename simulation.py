#!/usr/bin/env python3

import openmc
import time
import os
import pickle
import uuid
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import json


class SimulationRunner:
    """
    A class to run radiation streaming simulations with OpenMC.
    Provides comprehensive simulation setup and results processing.
    """

    def __init__(self, config):
        """
        Initialize the simulation runner with configuration.

        Args:
            config: Configuration object or dictionary with simulation parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Create required directories
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories for simulation outputs."""
        dirs = ['output', 'results', 'tallies', 'plots']
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)

    def generate_uuid(self):
        """Generate a unique ID for the simulation."""
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    def run_simulation(self,
                       energy: float,
                       channel_diameter: float,
                       detector_distance: Optional[float] = None,
                       detector_angle: float = 0,
                       particles: int = 1000000,
                       batches: int = 10,
                       plot: bool = False,
                       plot_resolution: Tuple[int, int] = (1000, 500),
                       channel_offset: Tuple[float, float] = (0, 0),
                       stepped_channel: bool = False,
                       stepped_offset: float = 0,
                       verbose: bool = True) -> Dict:
        """
        Run a single simulation with the given parameters.

        Args:
            energy: Energy of the source photons in MeV
            channel_diameter: Diameter of the channel in cm
            detector_distance: Distance from wall exit to detector in cm
            detector_angle: Angle from centerline in degrees
            particles: Number of particles per batch
            batches: Number of batches
            plot: Whether to generate plots
            plot_resolution: Resolution for plots (width, height)
            channel_offset: (y,z) offset of channel from center in cm
            stepped_channel: Whether to create a stepped channel
            stepped_offset: Lateral offset for stepped channel in cm
            verbose: Whether to print progress messages

        Returns:
            dict: Simulation results
        """
        # Generate a unique ID for this simulation
        sim_id = self.generate_uuid()

        if verbose:
            self.logger.info(f"Running simulation {sim_id}:")
            self.logger.info(f"  Energy: {energy} MeV")
            self.logger.info(f"  Channel diameter: {channel_diameter} cm")
            self.logger.info(f"  Detector distance: {detector_distance} cm")
            self.logger.info(f"  Detector angle: {detector_angle}°")
            if channel_offset != (0, 0):
                self.logger.info(f"  Channel offset: {channel_offset} cm")
            if stepped_channel:
                self.logger.info(f"  Stepped channel with offset: {stepped_offset} cm")

        try:
            # Import modules
            from src.core.materials import MaterialBuilder
            from src.core.geometry import GeometryBuilder
            from src.core.source import SourceBuilder
            from src.core.tallies import TallyBuilder

            # Create materials
            material_builder = MaterialBuilder()
            materials = material_builder.create_materials()

            # Create geometry
            geometry_builder = GeometryBuilder(
                wall_thickness=self.config.wall_thickness,
                source_wall_distance=self.config.source_to_wall_distance,
                world_x=self.config.world_width,
                world_y=self.config.world_height,
                world_z=self.config.world_depth,
                phantom_diameter=self.config.phantom_diameter
            )

            geom_info = geometry_builder.create_geometry(
                materials,
                channel_diameter=channel_diameter,
                detector_distance=detector_distance,
                detector_angle=detector_angle,
                channel_offset=channel_offset,
                stepped_channel=stepped_channel,
                stepped_offset=stepped_offset
            )

            phantom_cell = geom_info['phantom_cell']

            # Create source
            source_builder = SourceBuilder(self.config.source_to_wall_distance)
            source = source_builder.create_source(energy)

            # Create tallies
            tally_builder = TallyBuilder(
                world_x=self.config.world_width,
                world_y=self.config.world_height,
                world_z=self.config.world_depth
            )

            # Create energy bins based on source energy
            energy_max = energy * 1.2
            energy_bins = np.logspace(np.log10(0.01), np.log10(energy_max), 20)

            tallies, tally_info = tally_builder.create_tallies(
                phantom_cell=phantom_cell,
                energy_bins=energy_bins
            )

            # Create settings
            settings = openmc.Settings()
            settings.batches = batches
            settings.particles = particles
            settings.photon_transport = True
            settings.electron_treatment = 'ttb'  # Thick target bremsstrahlung
            settings.source = source
            settings.run_mode = 'fixed source'
            settings.output = {'tallies': False, 'summary': False}

            # Set cutoffs for efficient simulation
            settings.cutoff = {
                'energy_photon': 0.001,  # 1 keV cutoff for photons
                'energy_electron': 0.1  # 100 keV cutoff for electrons
            }

            # Export to XML
            settings.export_to_xml()

            # Create plots if requested
            if plot:
                self._create_plots(sim_id, plot_resolution)

            # Run OpenMC
            start_time = time.time()
            openmc.run()
            end_time = time.time()
            run_time = end_time - start_time

            if verbose:
                self.logger.info(f"  Simulation completed in {run_time:.1f} seconds")

            # Process and collect results
            results = self._process_results(
                sim_id,
                energy,
                channel_diameter,
                detector_distance,
                detector_angle,
                particles,
                batches,
                run_time,
                phantom_cell,
                channel_offset,
                stepped_channel,
                stepped_offset
            )

            # Save tally results
            self._save_tally_results(sim_id)

            # Generate additional plots if requested
            if plot:
                self._generate_result_plots(sim_id, results, geom_info)

            return results

        except Exception as e:
            self.logger.error(f"Error in simulation {sim_id}: {str(e)}", exc_info=True)

            # Return partial results with error information
            return {
                'simulation_id': sim_id,
                'energy': energy,
                'channel_diameter': channel_diameter,
                'distance': detector_distance,
                'angle': detector_angle,
                'error': str(e),
                'status': 'failed'
            }

    def run_batch_simulations(self, scenarios, verbose=True):
        """
        Run a batch of simulations from a list of scenarios.

        Args:
            scenarios: List of dictionaries with simulation parameters
            verbose: Whether to print progress messages

        Returns:
            list: Results from all simulations
        """
        results = []

        total = len(scenarios)
        for i, scenario in enumerate(scenarios, 1):
            if verbose:
                self.logger.info(f"Running scenario {i}/{total}")

            # Extract parameters with defaults
            params = {
                'energy': scenario.get('energy'),
                'channel_diameter': scenario.get('channel_diameter'),
                'detector_distance': scenario.get('detector_distance'),
                'detector_angle': scenario.get('detector_angle', 0),
                'particles': scenario.get('particles', self.config.default_particles),
                'batches': scenario.get('batches', self.config.default_batches),
                'plot': scenario.get('plot', False),
                'channel_offset': scenario.get('channel_offset', (0, 0)),
                'stepped_channel': scenario.get('stepped_channel', False),
                'stepped_offset': scenario.get('stepped_offset', 0),
                'verbose': verbose
            }

            # Run simulation
            result = self.run_simulation(**params)
            results.append(result)

            # Save intermediate results
            self._save_results_database(results)

        return results

    def _process_results(self, sim_id, energy, channel_diameter, detector_distance,
                         detector_angle, particles, batches, run_time, phantom_cell,
                         channel_offset, stepped_channel, stepped_offset):
        """Process simulation results and calculate doses."""
        results = {
            'simulation_id': sim_id,
            'energy': energy,
            'channel_diameter': channel_diameter,
            'distance': detector_distance,
            'angle': detector_angle,
            'channel_offset_y': channel_offset[0],
            'channel_offset_z': channel_offset[1],
            'stepped_channel': stepped_channel,
            'stepped_offset': stepped_offset if stepped_channel else 0,
            'particles': particles,
            'batches': batches,
            'runtime': run_time,
            'total_dose': None,
            'normalized_dose': None,
            'phantom_flux': None,
            'kerma': None,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }

        # Get results from tallies
        try:
            sp = openmc.StatePoint(f'statepoint.{batches}.h5')

            # Get dose tallies if detector was present
            if phantom_cell is not None:
                # Extract tallies
                flux_tally = sp.get_tally(name='flux')
                energy_tally = sp.get_tally(name='energy')
                kerma_tally = sp.get_tally(name='kerma')

                # Extract mean values
                phantom_flux = flux_tally.mean.flatten()[0]
                energy_deposition = energy_tally.mean.flatten()[0]
                kerma_value = kerma_tally.mean.flatten()[0]

                # Calculate relative errors
                flux_rel_error = flux_tally.std_dev.flatten()[0] / phantom_flux if phantom_flux > 0 else 0
                energy_rel_error = energy_tally.std_dev.flatten()[0] / energy_deposition if energy_deposition > 0 else 0
                kerma_rel_error = kerma_tally.std_dev.flatten()[0] / kerma_value if kerma_value > 0 else 0

                # Calculate dose rates
                source_strength = 3.7e10  # 1 Ci in Bq
                mev_to_rem = 1.6022e-10 * 100  #


continue

Context
— none
fetched
mev_to_rem = 1.6022e-10 * 100  # MeV/g to rem

# Calculate dose from energy deposition
dose_rate = energy_deposition * mev_to_rem * source_strength  # rem/hr

# Calculate dose from kerma (more accurate for photons)
kerma_dose_rate = kerma_value * mev_to_rem * source_strength  # rem/hr

# Store results
results['phantom_flux'] = phantom_flux
results['phantom_flux_rel_error'] = flux_rel_error
results['total_dose'] = dose_rate
results['total_dose_rel_error'] = energy_rel_error
results['kerma'] = kerma_dose_rate
results['kerma_rel_error'] = kerma_rel_error

# Calculate normalized dose
# Normalize by channel area (πr²)
if channel_diameter > 0:
    channel_area = np.pi * (channel_diameter / 2) ** 2
    results['normalized_dose'] = kerma_dose_rate / channel_area
else:
    results['normalized_dose'] = 0.0

except Exception as e:
self.logger.error(f"Error processing results: {str(e)}", exc_info=True)
results['error'] = f"Result processing error: {str(e)}"
results['status'] = 'completed_with_errors'

return results


def _save_tally_results(self, sim_id):
    """Save tally results for later analysis."""
    try:
        # Get statepoint file
        sp_file = f'statepoint.{openmc.settings.Settings().batches}.h5'
        if not os.path.exists(sp_file):
            self.logger.warning(f"Statepoint file {sp_file} not found")
            return

        # Save a copy of the statepoint file with simulation ID
        os.makedirs('tallies', exist_ok=True)
        dest_path = f'tallies/{sim_id}_statepoint.h5'
        import shutil
        shutil.copy(sp_file, dest_path)

        self.logger.info(f"Saved tallies to {dest_path}")

    except Exception as e:
        self.logger.error(f"Error saving tallies: {str(e)}", exc_info=True)


def _save_results_database(self, results):
    """Save the results database to a JSON file."""
    try:
        os.makedirs('results', exist_ok=True)
        with open('results/simulation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        self.logger.error(f"Error saving results database: {str(e)}", exc_info=True)


def _create_plots(self, sim_id, resolution=(1000, 500)):
    """Create geometry plots for visualization."""
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)

    # XY plot
    plot_xy = openmc.Plot()
    plot_xy.filename = f'plots/{sim_id}_xy'
    plot_xy.width = [self.config.world_width, self.config.world_height]
    plot_xy.pixels = resolution
    plot_xy.color_by = 'material'
    plot_xy.basis = 'xy'
    plot_xy.origin = [self.config.world_width / 2 - 50, 0, 0]

    # XZ plot
    plot_xz = openmc.Plot()
    plot_xz.filename = f'plots/{sim_id}_xz'
    plot_xz.width = [self.config.world_width, self.config.world_depth]
    plot_xz.pixels = resolution
    plot_xz.color_by = 'material'
    plot_xz.basis = 'xz'
    plot_xz.origin = [self.config.world_width / 2 - 50, 0, 0]

    # Create and save the plots
    plots = openmc.Plots([plot_xy, plot_xz])
    plots.export_to_xml()
    openmc.plot_geometry()


def _generate_result_plots(self, sim_id, results, geom_info):
    """Generate additional plots from simulation results."""
    try:
        # Get statepoint file
        sp_file = f'statepoint.{openmc.settings.Settings().batches}.h5'
        if not os.path.exists(sp_file):
            return

        sp = openmc.StatePoint(sp_file)

        # Get mesh tally
        mesh_tally = sp.get_tally(name='flux_mesh')
        if mesh_tally is None:
            return

        # Extract flux values and reshape to mesh dimensions
        flux = mesh_tally.get_values(scores=['flux']).flatten()
        mesh_filter = mesh_tally.find_filter(openmc.MeshFilter)
        mesh = mesh_filter.mesh
        shape = mesh.dimension
        flux_reshaped = np.reshape(flux, shape)

        # Create directory for plots
        os.makedirs('plots', exist_ok=True)

        # Create flux heatmap
        plt.figure(figsize=(12, 8))

        # Convert to log scale for better visualization
        # Add small value to avoid log(0)
        flux_log = np.log10(flux_reshaped[:, :, 0] + 1e-10)

        # Create heatmap
        plt.imshow(flux_log.T, cmap='jet', origin='lower', aspect='auto',
                   extent=[0, self.config.world_width,
                           -self.config.world_height / 2, self.config.world_height / 2])

        # Plot wall outline
        wall_min_x = geom_info['wall_min_x']
        wall_max_x = geom_info['wall_max_x']
        wall_height = geom_info['wall_height']

        # Create wall rectangle
        wall_rect = plt.Rectangle(
            (wall_min_x, -wall_height / 2),
            wall_max_x - wall_min_x,
            wall_height,
            fill=False,
            edgecolor='black',
            linestyle='--',
            linewidth=2
        )
        plt.gca().add_patch(wall_rect)

        # Add colorbar and labels
        cbar = plt.colorbar(label='Log10(Flux)')
        plt.xlabel('X Position [cm]')
        plt.ylabel('Y Position [cm]')
        plt.title(
            f'Neutron Flux Map (log scale)\nEnergy: {results["energy"]} MeV, Channel D: {results["channel_diameter"]} cm')

        # Save plot
        plt.savefig(f'plots/{sim_id}_flux_map.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Generate spectral plot if energy tally exists
        if 'phantom_flux' in results and results['phantom_flux'] is not None:
            try:
                energy_flux_tally = sp.get_tally(name='energy_flux')
                if energy_flux_tally is not None:
                    # Extract energy bins and flux values
                    energy_filter = energy_flux_tally.find_filter(openmc.EnergyFilter)
                    energy_bins = energy_filter.bins

                    # Get midpoints of energy bins
                    energy_midpoints = [(energy_bins[i] + energy_bins[i + 1]) / 2 for i in range(len(energy_bins) - 1)]

                    # Get flux values
                    energy_flux = energy_flux_tally.get_values(scores=['flux']).flatten()

                    # Plot spectrum
                    plt.figure(figsize=(10, 6))
                    plt.step(energy_bins[:-1], energy_flux, where='post')
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlabel('Energy [MeV]')
                    plt.ylabel('Flux per unit energy')
                    plt.title(
                        f'Energy Spectrum at Detector\nEnergy: {results["energy"]} MeV, Channel D: {results["channel_diameter"]} cm')
                    plt.grid(True, which='both', linestyle='--', alpha=0.5)

                    # Save plot
                    plt.savefig(f'plots/{sim_id}_energy_spectrum.png', dpi=300, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                self.logger.warning(f"Could not generate energy spectrum plot: {str(e)}")

    except Exception as e:
        self.logger.error(f"Error generating result plots: {str(e)}", exc_info=True)