#!/usr/bin/env python3

import openmc
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Union


class GeometryBuilder:
    """
    A builder class for creating OpenMC geometry for radiation streaming simulations.
    This provides a more flexible and reusable approach to geometry creation.
    """

    def __init__(self, wall_thickness, source_wall_distance,
                 world_x, world_y, world_z, phantom_diameter):
        """
        Initialize the geometry builder with configuration parameters.

        Args:
            wall_thickness: Thickness of the shielding wall in cm
            source_wall_distance: Distance from origin to wall in cm
            world_x, world_y, world_z: Dimensions of the world boundaries in cm
            phantom_diameter: Diameter of the ICRU sphere phantom in cm
        """
        self.wall_thickness = wall_thickness
        self.source_wall_distance = source_wall_distance
        self.world_x = world_x
        self.world_y = world_y
        self.world_z = world_z
        self.phantom_diameter = phantom_diameter
        self.logger = logging.getLogger(__name__)

    def create_geometry(self,
                        materials: Dict[str, openmc.Material],
                        channel_diameter: float = 0.5,
                        detector_distance: Optional[float] = None,
                        detector_angle: float = 0,
                        channel_offset: Tuple[float, float] = (0, 0),
                        stepped_channel: bool = False,
                        stepped_offset: float = 0) -> Dict:
        """
        Create the geometry for the radiation streaming simulation.

        Args:
            materials: Dictionary of materials with keys 'air', 'void', 'concrete', 'tissue'
            channel_diameter: Diameter of the streaming channel in cm
            detector_distance: Distance from wall exit to detector in cm
            detector_angle: Angle from centerline in degrees
            channel_offset: (y,z) offset of channel from center in cm
            stepped_channel: Whether to create a stepped channel for reduced streaming
            stepped_offset: Lateral offset for stepped channel design in cm

        Returns:
            dict: Dictionary with geometry information
        """
        # Input validation
        self._validate_inputs(materials, channel_diameter, detector_distance, detector_angle)

        # Log configuration
        self.logger.info(f"Creating geometry with channel diameter: {channel_diameter} cm, "
                         f"detector distance: {detector_distance} cm, angle: {detector_angle}°")

        # Extract materials
        try:
            air = materials['air']
            void = materials['void']
            concrete = materials['concrete']
            tissue = materials['tissue']
        except KeyError as e:
            raise ValueError(f"Missing required material: {e}")

        # Define universe
        universe = openmc.Universe()

        # Wall parameters
        wall_min_x = self.source_wall_distance
        wall_max_x = wall_min_x + self.wall_thickness

        # Create wall region
        wall_region = self._create_wall_region(wall_min_x, wall_max_x)

        # Create channel region if needed
        channel_region = None
        if channel_diameter > 0:
            offset_y, offset_z = channel_offset
            channel_region = self._create_channel_region(
                channel_diameter, wall_min_x, wall_max_x,
                offset_y, offset_z, stepped_channel, stepped_offset
            )

        # Create world boundaries
        world_box = self._create_world_boundaries()

        # Create wall cell
        wall_cell = self._create_wall_cell(wall_region, channel_region, concrete)

        # Create detector (phantom) if needed
        phantom_cell = None
        if detector_distance is not None:
            phantom_cell = self._create_phantom(
                wall_max_x, detector_distance, detector_angle, tissue
            )

        # Create air/void cells
        cells = [wall_cell]

        # Create channel cell if needed
        channel_cell = None
        if channel_region is not None:
            channel_cell = self._create_channel_cell(channel_region, air)
            cells.append(channel_cell)

        # Create surrounding air cell
        void_cell = self._create_surrounding_air_cell(
            world_box, wall_region, phantom_cell, air
        )
        cells.append(void_cell)

        # Add phantom to cells if it exists
        if phantom_cell is not None:
            cells.append(phantom_cell)

        # Create universe and geometry
        universe.add_cells(cells)
        geometry = openmc.Geometry(universe)

        # Export to XML and return geometry info
        try:
            geometry.export_to_xml()
        except Exception as e:
            self.logger.error(f"Failed to export geometry: {str(e)}")
            raise

        # Return information needed for other modules
        geom_info = {
            'wall_min_x': wall_min_x,
            'wall_max_x': wall_max_x,
            'wall_height': self.world_y,
            'wall_width': self.world_z,
            'channel_radius': channel_diameter / 2.0 if channel_diameter > 0 else 0,
            'channel_offset': channel_offset,
            'phantom_cell': phantom_cell,
            'geometry': geometry
        }

        return geom_info

    def _validate_inputs(self, materials, channel_diameter, detector_distance, detector_angle):
        """Validate input parameters."""
        # Check materials dictionary
        required_materials = ['air', 'void', 'concrete', 'tissue']
        for mat in required_materials:
            if mat not in materials:
                raise ValueError(f"Required material '{mat}' not found in materials dictionary")

        # Check numerical parameters
        if channel_diameter < 0:
            raise ValueError(f"Channel diameter must be non-negative, got {channel_diameter}")

        if detector_distance is not None and detector_distance < 0:
            raise ValueError(f"Detector distance must be non-negative, got {detector_distance}")

    def _create_wall_region(self, wall_min_x, wall_max_x):
        """Create the concrete wall region."""
        return (
                -openmc.ZPlane(self.world_y / 2) &
                +openmc.ZPlane(-self.world_y / 2) &
                -openmc.YPlane(self.world_z / 2) &
                +openmc.YPlane(-self.world_z / 2) &
                -openmc.XPlane(wall_max_x) &
                +openmc.XPlane(wall_min_x)
        )

    def _create_channel_region(self, channel_diameter, wall_min_x, wall_max_x,
                               offset_y=0, offset_z=0, stepped=False, step_offset=0):
        """Create the channel region through the wall."""
        channel_radius = channel_diameter / 2.0

        if not stepped:
            # Simple straight-through channel
            return (
                    -openmc.ZCylinder(
                        r=channel_radius,
                        x0=0,
                        y0=offset_y,
                        z0=offset_z,
                        axis='x'
                    ) &
                    -openmc.XPlane(wall_max_x) &
                    +openmc.XPlane(wall_min_x)
            )
        else:
            # Stepped channel (for additional shielding)
            midpoint_x = wall_min_x + self.wall_thickness / 2.0

            # First half of channel
            first_half = (
                    -openmc.ZCylinder(
                        r=channel_radius,
                        x0=0,
                        y0=offset_y,
                        z0=offset_z,
                        axis='x'
                    ) &
                    -openmc.XPlane(midpoint_x) &
                    +openmc.XPlane(wall_min_x)
            )

            # Second half of channel (offset for streaming reduction)
            second_half = (
                    -openmc.ZCylinder(
                        r=channel_radius,
                        x0=0,
                        y0=offset_y + step_offset,
                        z0=offset_z,
                        axis='x'
                    ) &
                    -openmc.XPlane(wall_max_x) &
                    +openmc.XPlane(midpoint_x)
            )

            return first_half | second_half

    def _create_world_boundaries(self):
        """Create the world boundary surfaces with vacuum conditions."""
        return (
                -openmc.XPlane(x0=-50, boundary_type='vacuum') &
                +openmc.XPlane(x0=self.world_x, boundary_type='vacuum') &
                -openmc.YPlane(y0=-self.world_y / 2, boundary_type='vacuum') &
                +openmc.YPlane(y0=self.world_y / 2, boundary_type='vacuum') &
                -openmc.ZPlane(z0=-self.world_z / 2, boundary_type='vacuum') &
                +openmc.ZPlane(z0=self.world_z / 2, boundary_type='vacuum')
        )

    def _create_wall_cell(self, wall_region, channel_region, concrete):
        """Create the concrete wall cell."""
        wall_cell = openmc.Cell(name='concrete_wall')

        if channel_region is not None:
            wall_cell.region = wall_region & ~channel_region
        else:
            wall_cell.region = wall_region

        wall_cell.fill = concrete
        return wall_cell

    def _create_phantom(self, wall_exit_x, detector_distance, detector_angle, tissue):
        """Create the tissue phantom (ICRU sphere) for dose calculations."""
        # Calculate detector position
        angle_rad = np.radians(detector_angle)
        detector_x = wall_exit_x + detector_distance * np.cos(angle_rad)
        detector_y = detector_distance * np.sin(angle_rad)

        # Create sphere region
        phantom_region = openmc.Sphere(
            x0=detector_x,
            y0=detector_y,
            z0=0,
            r=self.phantom_diameter / 2,
            boundary_type='transmission'
        )

        # Create phantom cell
        phantom_cell = openmc.Cell(name='phantom')
        phantom_cell.fill = tissue
        phantom_cell.region = phantom_region

        return phantom_cell

    def _create_channel_cell(self, channel_region, air):
        """Create the air-filled channel cell."""
        channel_cell = openmc.Cell(name='air_channel')
        channel_cell.region = channel_region
        channel_cell.fill = air
        return channel_cell

    def _create_surrounding_air_cell(self, world_box, wall_region, phantom_cell, air):
        """Create the surrounding air cell."""
        void_cell = openmc.Cell(name='surrounding_air')
        region = world_box & ~wall_region

        if phantom_cell is not None:
            region = region & ~phantom_cell.region

        void_cell.region = region
        void_cell.fill = air
        return void_cell