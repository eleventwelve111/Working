#!/usr/bin/env python3

import os
import yaml
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any


@dataclass
class SimulationConfig:
    """Configuration for radiation streaming simulations."""
    # Physical parameters
    wall_thickness: float  # cm
    source_to_wall_distance: float  # cm
    world_width: float  # cm
    world_height: float  # cm
    world_depth: float  # cm
    phantom_diameter: float  # cm

    # Conversion factors
    foot_to_cm: float = 30.48

    # Default simulation parameters
    default_particles: int = 1000000
    default_batches: int = 10

    # Output directories
    output_dir: str = "output"
    plot_dir: str = "plots"
    results_dir: str = "results"
    tallies_dir: str = "tallies"

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Validate physical dimensions
        for param_name in ['wall_thickness', 'source_to_wall_distance',
                           'world_width', 'world_height', 'world_depth',
                           'phantom_diameter', 'foot_to_cm']:
            value = getattr(self, param_name)
            if value <= 0:
                raise ValueError(f"Parameter {param_name} must be positive, got {value}")

        # Validate simulation parameters
        if self.default_particles <= 0:
            raise ValueError(f"default_particles must be positive, got {self.default_particles}")
        if self.default_batches <= 0:
            raise ValueError(f"default_batches must be positive, got {self.default_batches}")

    @classmethod
    def from_file(cls, filepath: str) -> 'SimulationConfig':
        """Load configuration from a file (YAML or JSON)."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        try:
            # Determine file type and load
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                with open(filepath, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ValueError("Configuration file must be YAML or JSON")

            # Filter out unknown parameters
            valid_fields = cls.__annotations__.keys()
            filtered_data = {k: v for k, v in config_data.items() if k in valid_fields}

            # Check if required fields are present
            required_fields = {'wall_thickness', 'source_to_wall_distance',
                               'world_width', 'world_height', 'world_depth',
                               'phantom_diameter'}
            missing = required_fields - set(filtered_data.keys())
            if missing:
                raise ValueError(f"Missing required configuration parameters: {missing}")

            return cls(**filtered_data)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON config: {str(e)}")
        except TypeError as e:
            raise ValueError(f"Configuration error: {str(e)}")

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to a file (YAML or JSON)."""
        config_dict = asdict(self)

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        try:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                with open(filepath, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            elif filepath.endswith('.json'):
                with open(filepath, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ValueError("Output file must have .yaml, .yml, or .json extension")
        except Exception as e:
            raise IOError(f"Failed to save configuration to {filepath}: {str(e)}")

    def create_directories(self) -> None:
        """Create necessary output directories."""
        for directory in [self.output_dir, self.plot_dir, self.results_dir, self.tallies_dir]:
            os.makedirs(directory, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def default_config(cls) -> 'SimulationConfig':
        """Create a default configuration."""
        return cls(
            wall_thickness=30.0,  # 30 cm thick wall
            source_to_wall_distance=50.0,  # 50 cm from source to wall
            world_width=200.0,  # 2 meter wide world
            world_height=100.0,  # 1 meter tall world
            world_depth=50.0,  # 50 cm deep world
            phantom_diameter=20.0,  # 20 cm phantom diameter
        )
