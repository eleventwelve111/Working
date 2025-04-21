#!/usr/bin/env python3

import openmc
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
import logging


class TallyBuilder:
    """
    A builder class for creating OpenMC tallies for radiation streaming simulations.
    Provides flexibility in defining various tallies for comprehensive analysis.
    """

    def __init__(self, world_x, world_y, world_z):
        """
        Initialize the tally builder with world dimensions.

        Args:
            world_x, world_y, world_z: Dimensions of the world in cm
        """
        self.world_x = world_x
        self.world_y = world_y
        self.world_z = world_z
        self.logger = logging.getLogger(__name__)

    def create_tallies(self,
                       phantom_cell: Optional[openmc.Cell] = None,
                       mesh_resolution: Tuple[int, int, int] = (100, 50, 1),
                       energy_bins: Optional[List[float]] = None,
                       include_particle_tallies: bool = False) -> Tuple[openmc.Tallies, Dict]:
        """
        Create tallies for the radiation streaming simulation.

        Args:
            phantom_cell: ICRU phantom cell for dose tallies
            mesh_resolution: Resolution for the mesh tally (x, y, z)
            energy_bins: Energy bin boundaries for spectral tallies in MeV
            include_particle_tallies: Whether to include particle-specific tallies

        Returns:
            tuple: (tallies object, dictionary of tally info)
        """
        self.logger.info("Creating tallies...")
        tallies = openmc.Tallies()
        tally_info = {}

        # Create default energy bins if not provided
        if energy_bins is None:
            energy_bins = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

        # Create standard mesh tally
        mesh, mesh_tally = self._create_mesh_tally(mesh_resolution)
        tallies.append(mesh_tally)
        tally_info['flux_mesh'] = {'mesh': mesh, 'tally': mesh_tally}

        # Create energy-dependent mesh tally if energy bins provided
        if len(energy_bins) > 1:
            energy_filter = openmc.EnergyFilter(energy_bins)
            energy_mesh_tally = openmc.Tally(name='energy_flux_mesh')
            energy_mesh_tally.filters = [openmc.MeshFilter(mesh), energy_filter]
            energy_mesh_tally.scores = ['flux']
            tallies.append(energy_mesh_tally)
            tally_info['energy_flux_mesh'] = {'mesh': mesh, 'tally': energy_mesh_tally, 'bins': energy_bins}

        # Add specialized particle tallies if requested
        if include_particle_tallies:
            particle_tallies = self._create_particle_tallies(mesh)
            for tally in particle_tallies:
                tallies.append(


continue

Context
— none
fetched
tallies.append(tally)
tally_info['particle_tallies'] = particle_tallies

# Add dose tallies if phantom is present
if phantom_cell is not None:
    dose_tallies = self._create_dose_tallies(phantom_cell, energy_bins)
    for tally in dose_tallies:
        tallies.append(tally)
    tally_info['dose_tallies'] = dose_tallies

# Export to XML
try:
    tallies.export_to_xml()
except Exception as e:
    self.logger.error(f"Failed to export tallies: {str(e)}")
    raise

return tallies, tally_info


def _create_mesh_tally(self, mesh_resolution):
    """Create a regular mesh tally for visualization."""
    # Create mesh
    mesh = openmc.RegularMesh()
    mesh.dimension = mesh_resolution
    mesh.lower_left = [0, -self.world_y / 2, -0.5]
    mesh.upper_right = [self.world_x, self.world_y / 2, 0.5]

    # Create mesh filter
    mesh_filter = openmc.MeshFilter(mesh)

    # Create flux tally on mesh
    flux_mesh_tally = openmc.Tally(name='flux_mesh')
    flux_mesh_tally.filters = [mesh_filter]
    flux_mesh_tally.scores = ['flux']

    return mesh, flux_mesh_tally


def _create_dose_tallies(self, phantom_cell, energy_bins):
    """Create dose tallies for the phantom."""
    dose_tallies = []

    # Cell filter for phantom
    cell_filter = openmc.CellFilter(phantom_cell)

    # Energy-integrated flux tally
    flux_tally = openmc.Tally(name='flux')
    flux_tally.filters = [cell_filter]
    flux_tally.scores = ['flux']
    dose_tallies.append(flux_tally)

    # Energy deposition tally
    energy_tally = openmc.Tally(name='energy')
    energy_tally.filters = [cell_filter]
    energy_tally.scores = ['heating']
    dose_tallies.append(energy_tally)

    # Add energy-dependent flux tally
    if len(energy_bins) > 1:
        energy_filter = openmc.EnergyFilter(energy_bins)
        spec_tally = openmc.Tally(name='energy_flux')
        spec_tally.filters = [cell_filter, energy_filter]
        spec_tally.scores = ['flux']
        dose_tallies.append(spec_tally)

    # Add kerma tally for more accurate dose calculation
    kerma_tally = openmc.Tally(name='kerma')
    kerma_tally.filters = [cell_filter]
    kerma_tally.scores = ['kerma-photon']
    dose_tallies.append(kerma_tally)

    return dose_tallies


def _create_particle_tallies(self, mesh):
    """Create particle-specific tallies."""
    particle_tallies = []

    # Create particle filter
    particles = ['photon', 'electron', 'positron']
    particle_filter = openmc.ParticleFilter(particles)

    # Create mesh filter
    mesh_filter = openmc.MeshFilter(mesh)

    # Particle flux tally
    particle_flux_tally = openmc.Tally(name='particle_flux')
    particle_flux_tally.filters = [mesh_filter, particle_filter]
    particle_flux_tally.scores = ['flux']
    particle_tallies.append(particle_flux_tally)

    return particle_tallies