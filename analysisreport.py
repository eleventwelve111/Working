#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import pickle
from scipy.optimize import curve_fit
from typing import List, Dict, Optional, Tuple, Union
import logging
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.platypus import PageBreak, ListFlowable, ListItem
from reportlab.lib.units import inch
import io


class ResultAnalyzer:
    """
    A class for analyzing radiation streaming simulation results.
    Provides tools for trend analysis, visualization, and reporting.
    """

    def __init__(self, results_db=None):
        """
        Initialize the analyzer.

        Args:
            results_db: Optional list of simulation results to analyze
        """
        self.results_db = results_db
        self.logger = logging.getLogger(__name__)

    def load_results(self, filepath='results/simulation_results.json'):
        """Load results from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                self.results_db = json.load(f)

            self.logger.info(f"Loaded {len(self.results_db)} simulation results from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading results: {str(e)}")
            return False

    def to_dataframe(self):
        """Convert results to pandas DataFrame for easier analysis."""
        if self.results_db is None or len(self.results_db) == 0:
            self.logger.warning("No results available for conversion to DataFrame")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(self.results_db)

        # Remove rows with errors
        if 'status' in df.columns:
            df = df[df['status'] == 'completed']

        return df

    def analyze_channel_diameter_trend(self, fixed_energy=None, fixed_distance=None):
        """
        Analyze the effect of channel diameter on dose/flux.

        Args:
            fixed_energy: Optional fixed energy to filter by
            fixed_distance: Optional fixed distance to filter by

        Returns:
            dict: Analysis results
        """
        df = self.to_dataframe()
        if df is None or len(df) == 0:
            return None

        # Filter data if needed
        if fixed_energy is not None:
            df = df[df['energy'] == fixed_energy]

        if fixed_distance is not None:
            df = df[df['distance'] == fixed_distance]

        if len(df) < 2:
            self.logger.warning("Not enough data points for trend analysis")
            return None

        # Group by channel diameter and compute statistics
        channel_data = []

        # For each unique channel diameter
        for diameter in sorted(df['channel_diameter'].unique()):
            subset = df[df['channel_diameter'] == diameter]

            # Collect statistics
            stats = {
                'diameter': diameter,
                'area': np.pi * (diameter / 2) ** 2 if diameter > 0 else 0,
                'dose_mean': subset['kerma'].mean() if 'kerma' in subset.columns else subset['total_dose'].mean(),
                'dose_std': subset['kerma'].std() if 'kerma' in subset.columns else subset['total_dose'].std(),
                'norm_dose_mean': subset['normalized_dose'].mean() if 'normalized_dose' in subset.columns else None,
                'norm_dose_std': subset['normalized_dose'].std() if 'normalized_dose' in subset.columns else None,
                'flux_mean': subset['phantom_flux'].mean() if 'phantom_flux' in subset.columns else None,
                'flux_std': subset['phantom_flux'].std() if 'phantom_flux' in subset.columns else None,
                'count': len(subset)
            }

            channel_data.append(stats)

        # Convert to DataFrame
        channel_df = pd.DataFrame(channel_data)

        # Fit trends if enough data points
        trends = {}

        if len(channel_df) >= 3:
            # Filter out zero values for fitting
            fit_df = channel_df[channel_df['diameter'] > 0].copy()

            if len(fit_df) >= 3:
                # Fit area vs. dose with power law: D = a * A^b
                try:
                    area = fit_df['area'].values
                    dose = fit_df['dose_mean'].values

                    popt, pcov = curve_fit(self._power_law, area, dose)
                    a, b = popt

                    trends['area_vs_dose'] = {
                        'model': 'power_law',
                        'formula': f'dose = {a:.4e} * area^{b:.4f}',
                        'parameters': {'a': a, 'b': b},
                        'r_squared': self._calculate_r_squared(dose, self._power_law(area, *popt))
                    }
                except Exception as e:
                    self.logger.warning(f"Could not fit area vs. dose: {str(e)}")

                # Fit diameter vs. dose with power law: D = a * d^b
                try:
                    diameter = fit_df['diameter'].values
                    dose = fit_df['dose_mean'].values

                    popt, pcov = curve_fit(self._power_law, diameter, dose)
                    a, b = popt

                    trends['diameter_vs_dose'] = {
                        'model': 'power_law',
                        'formula': f'dose = {a:.4e} * diameter^{b:.4f}',
                        'parameters': {'a': a, 'b': b},
                        'r_squared': self._calculate_r_squared(dose, self._power_law(diameter, *popt))
                    }
                except Exception as e:
                    self.logger.warning(f"Could not fit diameter vs. dose: {str(e)}")

        return {
            'data': channel_df.to_dict(orient='records'),
            'trends': trends,
            'filter': {
                'energy': fixed_energy,
                'distance': fixed_distance
            }
        }

    def analyze_distance_trend(self, fixed_energy=None, fixed_diameter=None):
        """
        Analyze the effect of distance on dose/flux.

        Args:
            fixed_energy: Optional fixed energy to filter by
            fixed_diameter: Optional fixed channel diameter to filter by

        Returns:
            dict: Analysis results
        """
        df = self.to_dataframe()
        if df is None or len(df) == 0:
            return None

        # Filter data if needed
        if fixed_energy is not None:
            df = df[df['energy'] == fixed_energy]

        if fixed_diameter is not None:
            df = df[df['channel_diameter'] == fixed_diameter]

        if len(df) < 2:
            self.logger.warning("Not enough data points for trend analysis")
            return None

        # Group by distance and compute statistics
        distance_data = []

        # For each unique distance value
        for distance in sorted(df['distance'].unique()):
            if distance is None:
                continue

            subset = df[df['distance'] == distance]

            # Collect statistics
            stats = {
                'distance': distance,
                'dose_mean': subset['kerma'].mean() if 'kerma' in subset.columns else subset['total_dose'].mean(),
                'dose_std': subset['kerma'].std() if 'kerma' in subset.columns else subset['total_dose'].std(),
                'flux_mean': subset['phantom_flux'].mean() if 'phantom_flux' in subset.columns else None,
                'flux_std': subset['phantom_flux'].std() if 'phantom_flux' in subset.columns else None,
                'count': len(subset)
            }

            distance_data.append(stats)

        # Convert to DataFrame
        distance_df = pd.DataFrame(distance_data)

        # Fit trends if enough data points
        trends = {}

        if len(distance_df) >= 3:
            # Fit inverse square law: dose = a / distance^2
            try:
                distances = distance_df['distance'].values
                doses = distance_df['dose_mean'].values

                popt, pcov = curve_fit(self._inverse_square, distances, doses)
                a = popt[0]

                trends['inverse_square'] = {
                    'model': 'inverse_square',
                    'formula': f'dose = {a:.4e} / distance^2',
                    'parameters': {'a': a},
                    'r_squared': self._calculate_r_squared(doses, self._inverse_square(distances, a))
                }
            except Exception as e:
                self.logger.warning(f"Could not fit inverse square law: {str(e)}")

            # Fit general power law: dose = a / distance^b
            try:
                popt, pcov = curve_fit(lambda x, a, b: a * (x ** -b), distances, doses)
                a, b = popt

                trends['power_law'] = {
                    'model': 'power_law',
                    'formula': f'dose = {a:.4e} / distance^{b:.4f}',
                    'parameters': {'a': a, 'b': b},
                    'r_squared': self._calculate_r_squared(doses, a * (distances ** -b))
                }
            except Exception as e:
                self.logger.warning(f"Could not fit power law: {str(e)}")

        return {
            'data': distance_df.to_dict(orient='records'),
            'trends': trends,
            'filter': {
                'energy': fixed_energy,
                'diameter': fixed_diameter
            }
        }

    def analyze_energy_trend(self, fixed_diameter=None, fixed_distance=None):
        """
        Analyze the effect of energy on dose/flux.

        Args:
            fixed_diameter: Optional fixed channel diameter to filter by
            fixed_distance: Optional fixed distance to filter by

        Returns:
            dict: Analysis results
        """
        df = self.to_dataframe()
        if df is None or len(df) == 0:
            return None

        # Filter data if needed
        if fixed_diameter is not None:
            df = df[df['channel_diameter'] == fixed_diameter]

        if fixed_distance is not None:
            df = df[df['distance'] == fixed_distance]

        if len(df) < 2:
            self.logger.warning("Not enough data points for trend analysis")
            return None

        # Group by energy and compute statistics
        energy_data = []

        # For each unique energy value
        for energy in sorted(df['energy'].unique()):
            subset = df[df['energy'] == energy]

            # Collect statistics
            stats = {
                'energy': energy,
                'dose_mean': subset['kerma'].mean() if 'kerma' in subset.columns else subset['total_dose'].mean(),
                'dose_std': subset['kerma'].std() if 'kerma' in subset.columns else subset['total_dose'].std(),
                'flux_mean': subset['phantom_flux'].mean() if 'phantom_flux' in subset.columns else None,
                'flux_std': subset['phantom_flux'].std() if 'phantom_flux' in subset.columns else None,
                'count': len(subset)
            }

            energy_data.append(stats)

        # Convert to DataFrame
        energy_df = pd.DataFrame(energy_data)

        # Fit trends if enough data points
        trends = {}

        if len(energy_df) >= 3:
            # Fit linear relationship: dose = a*energy + b
            try:
                energies = energy_df['energy'].values
                doses = energy_df['dose_mean'].values

                popt, pcov = curve_fit(lambda x, a, b: a * x + b, energies, doses)
                a, b = popt

                trends['linear'] = {
                    'model': 'linear',
                    'formula': f'dose = {a:.4e} * energy + {b:.4e}',
                    'parameters': {'a': a, 'b': b},
                    'r_squared': self._calculate_r_squared(doses, a * energies + b)
                }
            except Exception as e:
                self.logger.warning(f"Could not fit linear trend: {str(e)}")

            # Fit power law: dose = a * energy^b
            try:
                popt, pcov = curve_fit(self._power_law, energies, doses)
                a, b = popt

                trends['power_law'] = {
                    'model': 'power_law',
                    'formula': f'dose = {a:.4e} * energy^{b:.4f}',
                    'parameters': {'a': a, 'b': b},
                    'r_squared': self._calculate_r_squared(doses, self._power_law(energies, a, b))
                }
            except Exception as e:
                self.logger.warning(f"Could not fit power law: {str(e)}")

        return {
            'data': energy_df.to_dict(orient='records'),
            'trends': trends,
            'filter': {
                'diameter': fixed_diameter,
                'distance': fixed_distance
            }
        }

    def generate_visualization(self, analysis_type='channel', fixed_params=None,
                               output_dir='plots/analysis', show_fits=True):
        """
        Generate visualizations from analysis results.

        Args:
            analysis_type: Type of analysis ('channel', 'distance', or 'energy')
            fixed_params: Dictionary of fixed parameters for filtering
            output_dir: Directory to save output plots
            show_fits: Whether to include fitted trends

        Returns:
            str: Path to saved plot file
        """
        os.makedirs(output_dir, exist_ok=True)

        fixed_params = fixed_params or {}

        # Perform analysis based on type
        if analysis_type == 'channel':
            results = self.analyze_channel_diameter_trend(
                fixed_energy=fixed_params.get('energy'),
                fixed_distance=fixed_params.get('distance')
            )
            x_param = 'diameter'
            x_label = 'Channel Diameter (cm)'
            title_prefix = 'Effect of Channel Diameter on'

        elif analysis_type == 'distance':
            results = self.analyze_distance_trend(
                fixed_energy=fixed_params.get('energy'),
                fixed_diameter=fixed_


continue

Context
— none
fetched
fixed_energy = fixed_params.get('energy'),
fixed_diameter = fixed_params.get('diameter')
)
x_param = 'distance'
x_label = 'Distance from Source (cm)'
title_prefix = 'Effect of Distance on'

elif analysis_type == 'energy':
results = self.analyze_energy_trend(
fixed_diameter = fixed_params.get('diameter'),
fixed_distance = fixed_params.get('distance')
)
x_param = 'energy'
x_label = 'Photon Energy (MeV)'
title_prefix = 'Effect of Energy on'

else:
self.logger.error(f"Unknown analysis type: {analysis_type}")
return None

if results is None or 'data' not in results:
    self.logger.warning("No analysis results available for visualization")
    return None

# Create plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Get data
data_df = pd.DataFrame(results['data'])
x_values = data_df[x_param].values

# Plot dose
ax1.errorbar(
    x_values,
    data_df['dose_mean'],
    yerr=data_df['dose_std'],
    fmt='o-',
    color='blue',
    label='Dose Rate'
)

ax1.set_xlabel(x_label)
ax1.set_ylabel('Dose Rate (rem/hr)')
ax1.set_title(f'{title_prefix} Dose Rate')
ax1.grid(True, alpha=0.3)

# Add trend lines if available and requested
if show_fits and 'trends' in results and results['trends']:
    x_fit = np.linspace(min(x_values), max(x_values), 100)

    for trend_name, trend_info in results['trends'].items():
        if x_param in trend_name or analysis_type == 'distance':  # Handle distance trends specially
            if trend_name == 'inverse_square':
                a = trend_info['parameters']['a']
                y_fit = self._inverse_square(x_fit, a)
                ax1.plot(x_fit, y_fit, '--', label=f"Fit: {trend_info['formula']}")

            elif trend_name == 'power_law':
                a = trend_info['parameters']['a']
                b = trend_info['parameters']['b']
                if 'distance' in trend_name or analysis_type == 'distance':
                    y_fit = a * (x_fit ** -b)
                else:
                    y_fit = a * (x_fit ** b)
                ax1.plot(x_fit, y_fit, '--', label=f"Fit: {trend_info['formula']}")

            elif trend_name == 'linear':
                a = trend_info['parameters']['a']
                b = trend_info['parameters']['b']
                y_fit = a * x_fit + b
                ax1.plot(x_fit, y_fit, '--', label=f"Fit: {trend_info['formula']}")

            # Add R² value to the plot
            if 'r_squared' in trend_info:
                ax1.text(
                    0.05, 0.95 - 0.05 * list(results['trends'].keys()).index(trend_name),
                    f"R² = {trend_info['r_squared']:.4f}",
                    transform=ax1.transAxes,
                    fontsize=9,
                    verticalalignment='top'
                )

ax1.legend(loc='best')

# Plot flux if available
if 'flux_mean' in data_df and data_df['flux_mean'].notna().any():
    ax2.errorbar(
        x_values,
        data_df['flux_mean'],
        yerr=data_df['flux_std'],
        fmt='o-',
        color='green',
        label='Flux'
    )

    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Flux (particles/cm²)')
    ax2.set_title(f'{title_prefix} Particle Flux')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
else:
    ax2.text(0.5, 0.5, 'No flux data available', ha='center', va='center')
    ax2.set_xlabel(x_label)
    ax2.set_title(f'{title_prefix} Particle Flux')

# Set plot parameters
filter_text = ", ".join([f"{k}={v}" for k, v in results['filter'].items() if v is not None])
if filter_text:
    plt.suptitle(f'{title_prefix} Radiation Parameters\n({filter_text})', fontsize=14)
else:
    plt.suptitle(f'{title_prefix} Radiation Parameters', fontsize=14)

plt.tight_layout()

# Save plot
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
filename = f"{output_dir}/{analysis_type}_analysis_{timestamp}.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')

# Save the plot to a buffer for report generation
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
img_buffer.seek(0)

plt.close()

# Return the path to the saved file and the image buffer
return filename, img_buffer, results


def generate_report(self, analyses=None, output_dir='reports', filename=None):
    """
    Generate a comprehensive PDF report of simulation analyses.

    Args:
        analyses: List of analyses to include, each being a tuple of
                 (analysis_type, fixed_params, title)
        output_dir: Directory to save the report
        filename: Optional custom filename for the report

    Returns:
        str: Path to the generated PDF report
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"radiation_analysis_report_{timestamp}.pdf"

    filepath = os.path.join(output_dir, filename)

    # Create the PDF document
    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )

    # Styles for the document
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='Heading1',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='Heading2',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        name='Normal',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=8
    ))

    # Document elements
    elements = []

    # Title
    elements.append(Paragraph("Radiation Streaming Analysis Report", styles['Heading1']))
    elements.append(
        Paragraph(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 0.25 * inch))

    # Executive Summary
    elements.append(Paragraph("Executive Summary", styles['Heading2']))
    summary_text = """
        This report presents an analysis of radiation streaming through penetrations in shielding structures.
        The analyses examine the relationships between channel diameter, distance from source, 
        and photon energy on dose rates and particle flux.
        """
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 0.25 * inch))

    # Run analyses if provided
    if analyses:
        for analysis_idx, (analysis_type, fixed_params, title) in enumerate(analyses):
            elements.append(Paragraph(f"{title}", styles['Heading2']))

            # Generate visualization
            try:
                plot_path, img_buffer, results = self.generate_visualization(
                    analysis_type=analysis_type,
                    fixed_params=fixed_params,
                    output_dir='plots/analysis',
                    show_fits=True
                )

                if plot_path and img_buffer and results:
                    # Add plot to the document
                    img = Image(img_buffer, width=6.5 * inch, height=3 * inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.1 * inch))

                    # Add analysis interpretation
                    elements.append(Paragraph("Analysis Results:", styles['Heading2']))

                    # Different interpretations based on analysis type
                    if analysis_type == 'channel':
                        interpretation = self._interpret_channel_analysis(results)
                    elif analysis_type == 'distance':
                        interpretation = self._interpret_distance_analysis(results)
                    elif analysis_type == 'energy':
                        interpretation = self._interpret_energy_analysis(results)
                    else:
                        interpretation = "Analysis type not recognized."

                    elements.append(Paragraph(interpretation, styles['Normal']))

                    # Add data table
                    elements.append(Paragraph("Data Table:", styles['Heading2']))
                    data_df = pd.DataFrame(results['data'])

                    # Format the table data
                    table_data = [list(data_df.columns)]
                    for _, row in data_df.iterrows():
                        formatted_row = []
                        for col in data_df.columns:
                            value = row[col]
                            if pd.isna(value):
                                formatted_row.append("N/A")
                            elif isinstance(value, (int, float)):
                                if abs(value) < 0.01 or abs(value) > 1000:
                                    formatted_row.append(f"{value:.2e}")
                                else:
                                    formatted_row.append(f"{value:.4f}")
                            else:
                                formatted_row.append(str(value))
                        table_data.append(formatted_row)

                    # Create and style the table
                    if len(table_data) > 1:
                        table = Table(table_data)
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        elements.append(table)
                    else:
                        elements.append(Paragraph("No data available for table.", styles['Normal']))

                    # Add trend analysis if available
                    if 'trends' in results and results['trends']:
                        elements.append(Paragraph("Trend Analysis:", styles['Heading2']))

                        trend_items = []
                        for trend_name, trend_info in results['trends'].items():
                            trend_text = f"{trend_name.replace('_', ' ').title()}: {trend_info['formula']}"
                            if 'r_squared' in trend_info:
                                trend_text += f" (R² = {trend_info['r_squared']:.4f})"
                            trend_items.append(ListItem(Paragraph(trend_text, styles['Normal'])))

                        if trend_items:
                            elements.append(ListFlowable(trend_items, bulletType='bullet'))

                    # Add interpretation guidance
                    elements.append(Paragraph("Interpretation Guidance:", styles['Heading2']))
                    elements.append(Paragraph(self._get_interpretation_guidance(analysis_type), styles['Normal']))

                else:
                    elements.append(Paragraph(f"No results available for {analysis_type} analysis.", styles['Normal']))

            except Exception as e:
                self.logger.error(f"Error generating analysis for report: {str(e)}")
                elements.append(Paragraph(f"Error generating analysis: {str(e)}", styles['Normal']))

            # Add a page break between analyses, except for the last one
            if analysis_idx < len(analyses) - 1:
                elements.append(PageBreak())

    else:
        # Run default analyses if none provided
        self.logger.info("No specific analyses provided, running default set")

        # Channel diameter analysis
        elements.append(Paragraph("Channel Diameter Analysis", styles['Heading2']))
        try:
            plot_path, img_buffer, results = self.generate_visualization(
                analysis_type='channel',
                fixed_params={'energy': 1.0},
                output_dir='plots/analysis'
            )

            if plot_path and img_buffer:
                img = Image(img_buffer, width=6.5 * inch, height=3 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.25 * inch))
                elements.append(Paragraph(self._interpret_channel_analysis(results), styles['Normal']))
            else:
                elements.append(Paragraph("No results available for channel analysis.", styles['Normal']))
        except Exception as e:
            self.logger.error(f"Error generating channel analysis for report: {str(e)}")
            elements.append(Paragraph(f"Error generating channel analysis: {str(e)}", styles['Normal']))

        elements.append(PageBreak())

        # Distance analysis
        elements.append(Paragraph("Distance Analysis", styles['Heading2']))
        try:
            plot_path, img_buffer, results = self.generate_visualization(
                analysis_type='distance',
                fixed_params={'energy': 1.0},
                output_dir='plots/analysis'
            )

            if plot_path and img_buffer:
                img = Image(img_buffer, width=6.5 * inch, height=3 * inch)
                elements.append(img)
                elements.append(Spacer(1, 0.25 * inch))
                elements.append(Paragraph(self._interpret_distance_analysis(results), styles['Normal']))
            else:
                elements.append(Paragraph("No results available for distance analysis.", styles['Normal']))
        except Exception as e:


continue

Context
— none
fetched
elements.append(Paragraph("No results available for distance analysis.", styles['Normal']))
except Exception as e:
self.logger.error(f"Error generating distance analysis for report: {str(e)}")
elements.append(Paragraph(f"Error generating distance analysis: {str(e)}", styles['Normal']))

elements.append(PageBreak())

# Energy analysis
elements.append(Paragraph("Energy Analysis", styles['Heading2']))
try:
    plot_path, img_buffer, results = self.generate_visualization(
        analysis_type='energy',
        fixed_params={},
        output_dir='plots/analysis'
    )

    if plot_path and img_buffer:
        img = Image(img_buffer, width=6.5 * inch, height=3 * inch)
        elements.append(img)
        elements.append(Spacer(1, 0.25 * inch))
        elements.append(Paragraph(self._interpret_energy_analysis(results), styles['Normal']))
    else:
        elements.append(Paragraph("No results available for energy analysis.", styles['Normal']))
except Exception as e:
    self.logger.error(f"Error generating energy analysis for report: {str(e)}")
    elements.append(Paragraph(f"Error generating energy analysis: {str(e)}", styles['Normal']))

# Conclusions section
elements.append(PageBreak())
elements.append(Paragraph("Conclusions and Recommendations", styles['Heading2']))
conclusions_text = """
        Based on the analyses presented in this report, several key findings emerge:

        1. Radiation dose rates through penetrations increase significantly with channel diameter, 
           following a power law relationship. This highlights the importance of minimizing 
           penetration sizes in radiation shields.

        2. Dose rates decrease with distance from the penetration exit, generally following an 
           inverse square law, though deviations can occur due to scattering effects.

        3. Higher energy photons generally result in higher dose rates, with the relationship 
           dependent on the specific shielding material and geometry.

        Recommendations:

        1. Implement stepped or offset penetrations where possible to reduce direct streaming.
        2. Maintain maximum possible distance between radiation sources and sensitive equipment/personnel.
        3. Consider additional local shielding at penetration exits for critical areas.
        4. Design penetrations with the minimum diameter necessary for functional requirements.
        """
elements.append(Paragraph(conclusions_text, styles['Normal']))

# References
elements.append(Paragraph("References", styles['Heading2']))
references_text = """
        1. NCRP Report No. 151, "Structural Shielding Design and Evaluation for Megavoltage X- and Gamma-Ray Radiotherapy Facilities"
        2. ANSI/ANS-6.4.3-1991, "Gamma-Ray Attenuation Coefficients and Buildup Factors for Engineering Materials"
        3. ICRP Publication 103, "The 2007 Recommendations of the International Commission on Radiological Protection"
        """
elements.append(Paragraph(references_text, styles['Normal']))

# Build the document
try:
    doc.build(elements)
    self.logger.info(f"Report successfully generated at {filepath}")
    return filepath
except Exception as e:
    self.logger.error(f"Error building report document: {str(e)}")
    return None


def _interpret_channel_analysis(self, results):
    """Generate interpretation text for channel diameter analysis."""
    if not results or 'data' not in results:
        return "Insufficient data for interpretation."

    data_df = pd.DataFrame(results['data'])
    if len(data_df) < 2:
        return "Insufficient data points for trend interpretation."

    interpretation = """
        The analysis of channel diameter effects shows the relationship between penetration size and radiation streaming.
        """

    # Add relationship description
    if 'trends' in results and results['trends']:
        if 'diameter_vs_dose' in results['trends']:
            trend = results['trends']['diameter_vs_dose']
            b_value = trend['parameters']['b']
            r_squared = trend['r_squared']

            interpretation += f"\n\nThe data follows a power law relationship with an exponent of {b_value:.2f} (R² = {r_squared:.4f})."

            if 1.9 <= b_value <= 2.1:
                interpretation += "\nThis is consistent with theoretical predictions that dose rate is proportional to the cross-sectional area of the penetration."
            elif b_value > 2.1:
                interpretation += "\nThe exponent is higher than 2, suggesting additional effects beyond simple area scaling, possibly due to increased scattering within larger penetrations."
            elif b_value < 1.9:
                interpretation += "\nThe exponent is lower than 2, indicating that other factors may be mitigating the full impact of increased penetration size."

    # Add observations about the data range
    min_diameter = data_df['diameter'].min()
    max_diameter = data_df['diameter'].max()
    min_dose = data_df['dose_mean'].min()
    max_dose = data_df['dose_mean'].max()
    dose_ratio = max_dose / min_dose if min_dose > 0 else float('inf')

    interpretation += f"\n\nAcross the tested diameter range ({min_diameter:.1f} cm to {max_diameter:.1f} cm), the dose rate varied by a factor of {dose_ratio:.1f} ({min_dose:.2e} to {max_dose:.2e} rem/hr)."

    if 'filter' in results and results['filter']:
        filter_info = ", ".join([f"{k}={v}" for k, v in results['filter'].items() if v is not None])
        if filter_info:
            interpretation += f"\n\nThis analysis was performed with fixed parameters: {filter_info}."

    return interpretation


def _interpret_distance_analysis(self, results):
    """Generate interpretation text for distance analysis."""
    if not results or 'data' not in results:
        return "Insufficient data for interpretation."

    data_df = pd.DataFrame(results['data'])
    if len(data_df) < 2:
        return "Insufficient data points for trend interpretation."

    interpretation = """
        The analysis of distance effects shows how radiation levels change with distance from the penetration exit.
        """

    # Add relationship description
    if 'trends' in results and results['trends']:
        if 'inverse_square' in results['trends']:
            trend = results['trends']['inverse_square']
            r_squared = trend['r_squared']

            interpretation += f"\n\nThe data approximates an inverse square relationship (R² = {r_squared:.4f})."

            if r_squared > 0.95:
                interpretation += "\nThis strong agreement with the inverse square law indicates that the radiation is behaving as a point source beyond the penetration exit."
            elif 0.8 <= r_squared <= 0.95:
                interpretation += "\nThe moderate agreement with the inverse square law suggests some influence of scattered radiation or non-point source geometry."
            else:
                interpretation += "\nThe deviation from the inverse square law indicates significant effects from scattered radiation, geometric factors, or complex source distribution."

        if 'power_law' in results['trends']:
            trend = results['trends']['power_law']
            b_value = trend['parameters']['b']
            r_squared = trend['r_squared']

            interpretation += f"\n\nThe general power law model with an exponent of {b_value:.2f} (R² = {r_squared:.4f}) provides a mathematical description of the distance-dose relationship."

            if 1.9 <= b_value <= 2.1:
                interpretation += "\nThis is consistent with an inverse square law relationship."
            elif b_value > 2.1:
                interpretation += "\nThe exponent is higher than 2, suggesting additional attenuation effects like air attenuation or geometric factors."
            elif b_value < 1.9:
                interpretation += "\nThe exponent is lower than 2, indicating that scattered radiation or non-point source effects are significant."

    # Add observations about the data range
    min_distance = data_df['distance'].min()
    max_distance = data_df['distance'].max()
    min_dose = data_df['dose_mean'].min()
    max_dose = data_df['dose_mean'].max()
    dose_ratio = max_dose / min_dose if min_dose > 0 else float('inf')

    interpretation += f"\n\nAcross the tested distance range ({min_distance:.1f} cm to {max_distance:.1f} cm), the dose rate decreased by a factor of {dose_ratio:.1f} ({max_dose:.2e} to {min_dose:.2e} rem/hr)."

    if 'filter' in results and results['filter']:
        filter_info = ", ".join([f"{k}={v}" for k, v in results['filter'].items() if v is not None])
        if filter_info:
            interpretation += f"\n\nThis analysis was performed with fixed parameters: {filter_info}."

    return interpretation


def _interpret_energy_analysis(self, results):
    """Generate interpretation text for energy analysis."""
    if not results or 'data' not in results:
        return "Insufficient data for interpretation."

    data_df = pd.DataFrame(results['data'])
    if len(data_df) < 2:
        return "Insufficient data points for trend interpretation."

    interpretation = """
        The analysis of energy effects shows how radiation levels vary with the energy of the source photons.
        """

    # Add relationship description
    if 'trends' in results and results['trends']:
        if 'linear' in results['trends']:
            trend = results['trends']['linear']
            a_value = trend['parameters']['a']
            r_squared = trend['r_squared']

            interpretation += f"\n\nThe data shows a linear relationship with a slope of {a_value:.4e} (R² = {r_squared:.4f})."

            if r_squared > 0.95:
                interpretation += "\nThis strong linear relationship indicates that dose rate increases proportionally with energy in this range."
            else:
                interpretation += "\nThe moderate linear correlation suggests additional factors affecting the energy-dose relationship."

        if 'power_law' in results['trends']:
            trend = results['trends']['power_law']
            b_value = trend['parameters']['b']
            r_squared = trend['r_squared']

            interpretation += f"\n\nThe power law model with an exponent of {b_value:.2f} (R² = {r_squared:.4f}) provides an alternative description of the energy-dose relationship."

            if b_value > 1:
                interpretation += "\nThe exponent is greater than 1, indicating that dose increases more than linearly with energy."
            elif 0.8 <= b_value <= 1.2:
                interpretation += "\nThe exponent is approximately 1, consistent with a near-linear relationship between energy and dose."
            else:
                interpretation += "\nThe exponent is less than 1, suggesting diminishing returns in dose as energy increases."

    # Add observations about the data range
    min_energy = data_df['energy'].min()
    max_energy = data_df['energy'].max()
    min_dose = data_df['dose_mean'].min()
    max_dose = data_df['dose_mean'].max()
    dose_ratio = max_dose / min_dose if min_dose > 0 else float('inf')

    interpretation += f"\n\nAcross the tested energy range ({min_energy:.1f} MeV to {max_energy:.1f} MeV), the dose rate varied by a factor of {dose_ratio:.1f} ({min_dose:.2e} to {max_dose:.2e} rem/hr)."

    if 'filter' in results and results['filter']:
        filter_info = ", ".join([f"{k}={v}" for k, v in results['filter'].items() if v is not None])
        if filter_info:
            interpretation += f"\n\nThis analysis was performed with fixed parameters: {filter_info}."

    return interpretation


def _get_interpretation_guidance(self, analysis_type):
    """Provide guidance on interpreting the analysis."""
    if analysis_type == 'channel':
        return """
            Channel diameter analysis should be interpreted considering:

            1. Power law exponents close to 2 indicate that dose scales with penetration area
            2. Deviations from the square law may indicate complex scattering or streaming effects
            3. Optimal penetration designs should minimize diameter while meeting functional requirements
            4. Consider stepped or offset designs for larger required penetrations
            """
    elif analysis_type == 'distance':
        return """
            Distance analysis should be interpreted considering:

            1. Inverse square law behavior indicates point-source-like radiation properties
            2. Deviations from inverse square may indicate:
               - Significant scattered radiation contributions
               - Extended source geometry effects
               - Air attenuation at larger distances
            3. Distance is an effective and simple method for reducing radiation exposure
            """
    elif analysis_type == 'energy':
        return """
            Energy analysis should be interpreted considering:

            1. Higher energy photons typically produce higher dose rates due to:
               - Greater penetration through shielding
               - Different interaction mechanisms at different energies
            2. Material-dependent effects can cause non-linear relationships
            3. For multi-energy sources, focus shielding design on the highest energy components
            4. Consider energy-specific shielding materials for optimal performance
            """
    else:
        return """
            Analysis interpretation should consider:

            1. Statistical significance of the trends identified
            2. Physical models that explain the observed relationships
            3. Practical implications for radiation protection
            4. Potential sources of uncertainty in measurements and simulations
            """

    # Helper methods for curve fitting


def _inverse_square(self, x, a):
    """Inverse square law: f(x) = a / x²"""
    return a / (x ** 2)


def _power_law(self, x, a, b):
    """Power law: f(x) = a * x^b"""
    return a * (x ** b)


def _exponential(self, x, a, b):
    """Exponential function: f(x) = a * exp(b*x)"""
    return a * np.exp(b * x)


def _calculate_r_squared(self, y_true, y_pred):
    """Calculate coefficient of determination (R²)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot!= 0 else 0