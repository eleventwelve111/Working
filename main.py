#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import yaml
import json
from typing import List, Dict, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from config import SimulationConfig
from simulation import SimulationRunner
from src.core.analysis import ResultAnalyzer


def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Gamma-ray shielding simulation tool')

    # Configuration options
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--create-config', action='store_true',
                        help='Create a default configuration file and exit')

    # Simulation parameters
    parser.add_argument('--energy', type=float,
                        help='Energy of the source photons in MeV')
    parser.add_argument('--channel-diameter', type=float,
                        help='Diameter of the channel in cm')
    parser.add_argument('--detector-distance', type=float,
                        help='Distance from wall exit to detector in cm')
    parser.add_argument('--detector-angle', type=float, default=0,
                        help='Angle from centerline in degrees')
    parser.add_argument('--particles', type=int,
                        help='Number of particles per batch')
    parser.add_argument('--batches', type=int,
                        help='Number of batches')

    # Advanced options
    parser.add_argument('--channel-offset-y', type=float, default=0,
                        help='Y-offset of channel from center in cm')
    parser.add_argument('--channel-offset-z', type=float, default=0,
                        help='Z-offset of channel from center in cm')
    parser.add_argument('--stepped-channel', action='store_true',
                        help='Create a stepped channel for better shielding')
    parser.add_argument('--stepped-offset', type=float, default=0,
                        help='Lateral offset for stepped channel in cm')

    # Batch simulation options
    parser.add_argument('--batch-file', type=str,
                        help='Path to batch scenarios file (YAML or JSON)')

    # Analysis options
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze existing results instead of running simulations')
    parser.add_argument('--analysis-type', type=str, choices=['channel', 'distance', 'energy'],
                        help='Type of analysis to perform')

    # Visualization options
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots for the simulation')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plot generation (for batch runs)')

    # Utility options
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    return parser.parse_args()


def load_batch_scenarios(filepath):
    """Load batch simulation scenarios from file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Batch scenarios file not found: {filepath}")

    if filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'r') as f:
            scenarios = yaml.safe_load(f)
    elif filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            scenarios = json.load(f)
    else:
        raise ValueError("Batch file must be YAML or JSON")

    # Validate scenarios
    if not isinstance(scenarios, list):
        raise ValueError("Batch scenarios must be a list")

    return scenarios


def run_single_simulation(config, args, logger):
    """Run a single simulation with the provided parameters."""
    # Create simulation runner
    runner = SimulationRunner(config)

    # Prepare simulation parameters
    params = {
        'energy': args.energy,
        'channel_diameter': args.channel_diameter,
        'detector_distance': args.detector_distance,
        'detector_angle': args.detector_angle,
        'particles': args.particles or config.default_particles,
        'batches': args.batches or config.default_batches,
        'plot': args.plot,
        'channel_offset': (args.channel_offset_y, args.channel_offset_z),
        'stepped_channel': args.stepped_channel,
        'stepped_offset': args.stepped_offset,
        'verbose': args.verbose
    }

    # Run simulation
    logger.info("Starting single simulation with parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")

    result = runner.run_simulation(**params)

    # Output results summary
    logger.info("Simulation completed. Results:")
    if 'error' in result:
        logger.error(f"  Error: {result['error']}")
    else:
        logger.info(f"  Simulation ID: {result['simulation_id']}")
        logger.info(f"  Total dose: {result.get('total_dose', 'N/A')} rem/hr")
        logger.info(f"  Kerma: {result.get('kerma', 'N/A')} rem/hr")
        logger.info(f"  Runtime: {result.get('runtime', 'N/A')} seconds")

    return result


def analyze_results(args, logger):
    """Analyze existing simulation results."""
    analyzer = ResultAnalyzer()

    # Load results from database
    if not analyzer.load_results():
        logger.error("Failed to load simulation results")
        return

    # Prepare fixed parameters for analysis
    fixed_params = {}
    if args.energy:
        fixed_params['energy'] = args.energy
    if args.channel_diameter:
        fixed_params['diameter'] = args.channel_diameter
    if args.detector_distance:
        fixed_params['distance'] = args.detector_distance

    # Generate visualization based on analysis type
    if not args.analysis_type:
        logger.error("Analysis type must be specified with --analysis-type")
        return

    logger.info(f"Performing {args.analysis_type} analysis with fixed parameters: {fixed_params}")
    plot_path = analyzer.generate_visualization(
        analysis_type=args.analysis_type,
        fixed_params=fixed_params,
        output_dir='plots/analysis',
        show_fits=True
    )

    if plot_path:
        logger.info(f"Analysis plot saved to {plot_path}")
    else:
        logger.error("Failed to generate analysis plot")


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(log_level)

    logger.info(f"Starting gamma-ray shielding simulation tool")

    # Handle config creation request
    if args.create_config:
        config_file = args.config
        logger.info(f"Creating default configuration file: {config_file}")
        config = SimulationConfig.default_config()
        try:
            config.save_to_file(config_file)
            logger.info(f"Default configuration saved to {config_file}")
            return 0
        except Exception as e:
            logger.error(f"Failed to create configuration file: {str(e)}")
            return 1

    # Load configuration
    try:
        config = SimulationConfig.from_file(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        logger.info("Creating directories with default configuration")
        config = SimulationConfig.default_config()

    # Create output directories
    config.create_directories()

    # If analysis mode is enabled, analyze results instead of running simulations
    if args.analyze:
        analyze_results(args, logger)
        return 0

    # If batch file is provided, run batch simulations
    if args.batch_file:
        try:
            scenarios = load_batch_scenarios(args.batch_file)
            logger.info(f"Loaded {len(scenarios)} scenarios from {args.batch_file}")

            runner = SimulationRunner(config)
            results = runner.run_batch_simulations(scenarios, verbose=args.verbose)

            logger.info(f"Completed {len(results)} simulations")
            return 0

        except Exception as e:
            logger.error(f"Batch simulation failed: {str(e)}")
            return 1

    # Otherwise, run a single simulation
    if args.energy is None or args.channel_diameter is None:
        logger.error("Energy and channel diameter are required for single simulation")
        return 1

    try:
        result = run_single_simulation(config, args, logger)
        return 0 if result.get('status') != 'failed' else 1
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())