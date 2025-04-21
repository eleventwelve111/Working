import numpy as np
import openmc


def create_source(energy, cone_angle):
    """
    Create a source with particles directed through the channel
    Ensures all particles go through the channel without hitting the wall

    Parameters:
        energy (float): Energy of the source particles in MeV
        cone_angle (float): Half-angle of the cone that encompasses the channel in radians

    Returns:
        openmc.Source: The OpenMC source
    """
    # Create a point source at the origin
    source = openmc.Source()
    source.space = openmc.stats.Point((0, 0, 0))

    # Set energy (monoenergetic in MeV)
    source.energy = openmc.stats.Discrete([energy * 1e6], [1.0])  # Convert MeV to eV

    # Direct particles in a cone toward the channel
    # Using the calculated cone angle to ensure all particles go through
    # mu limits from cos(cone_angle) to 1.0 to restrict to the forward cone
    source.angle = openmc.stats.PolarAzimuthal(
        mu=openmc.stats.Uniform(np.cos(cone_angle), 1.0),
        phi=openmc.stats.Uniform(0, 2 * np.pi),
        reference_uvw=(1, 0, 0)  # Direction along x-axis
    )

    # Set particle type to photon
    source.particle = 'photon'

    return source 