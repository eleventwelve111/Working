# Working
# Gamma Ray Shielding Simulation

This project simulates gamma ray penetration through a concrete wall with a cylindrical air channel. It evaluates dose rates at various distances and angles behind the wall for different gamma-ray energies and channel diameters.

## Project Structure

- `setup.py` - Main script to run the simulations
- `config.py` - Configuration parameters
- `materials.py` - Material definitions
- `geometry.py` - Geometry creation
- `sources.py` - Source definitions
- `dose.py` - Dose calculation functions
- `tallies.py` - Tally creation
- `simulation.py` - Simulation runner
- `visualization.py` - Basic visualization functions
- `visualization_advanced.py` - Advanced visualization functions
- `spectrum_analysis.py` - Spectrum analysis functions
- `report.py` - Report generation functions

## Requirements

- Python 3.7+
- OpenMC
- NumPy
- Matplotlib
- SciPy
- Pandas

## Installation

1. Make sure you have OpenMC installed with all dependencies. It's recommended to use a Conda environment:

```bash
conda create -n openmc_env python=3.8
conda activate openmc_env
conda install -c conda-forge openmc
conda install numpy matplotlib scipy pandas
```

2. You'll need cross-section data for OpenMC. Update the path in `config.py`:

```python
cross_sections_path = '/path/to/your/cross_sections.xml'
```

## Usage

1. Activate the OpenMC environment:

```bash
conda activate openmc_env
```

2. Run the simulation:

```bash
python setup.py
```

3. Results will be stored in the `results/` directory, including:
   - Visualization plots for different configurations
   - A comprehensive PDF report
   - JSON files with simulation data

## Configuration

You can modify simulation parameters in `config.py`:

- Wall thickness
- Source distance
- Channel diameters
- Gamma-ray energies
- Detector positions and angles

For quick testing, enable `test_mode` in `config.py` with a reduced set of parameters.

## Output

The simulation produces:
- Dose rate data for all configurations
- 2D radiation field visualizations
- Energy spectrum plots
- Dose vs. angle/distance/diameter plots
- Polar heatmaps of dose distribution
- A comprehensive PDF report summarizing findings

## Physics

The simulation models:
- Gamma ray source with specified energy
- Concrete wall with cylindrical air channel
- ICRU tissue phantom detector
- Radiation transport through materials
- Flux-to-dose conversion based on NCRP-38/ANS-6.1.1-1977 factors 
