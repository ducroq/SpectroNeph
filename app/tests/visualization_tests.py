#!/usr/bin/env python3
"""
Visualization Test Script for SpectroNeph

This script demonstrates the visualization functionality by generating example plots
using both artificial data and optionally real data if available.

Usage:
    python visualization_test.py [--datafile DATAFILE] [--output OUTPUT_DIR]
"""

import sys
import os
import argparse
import json
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the visualization module
from visualization.plots import (
    create_spectral_profile,
    create_time_series,
    create_ratio_analysis,
    create_comparison_plot,
    create_heatmap,
    create_agglutination_overview,
    CHANNEL_WAVELENGTHS
)

# Constants for artificial data generation
CHANNEL_NAMES = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "Clear", "NIR"]
SPECTRAL_SHAPE = {
    # Artificial spectral profile shape (relative values)
    "F1": 0.6,  # 415nm (Violet)
    "F2": 0.8,  # 445nm (Indigo)
    "F3": 1.0,  # 480nm (Blue)
    "F4": 0.9,  # 515nm (Cyan/Green)
    "F5": 0.7,  # 555nm (Green)
    "F6": 0.5,  # 590nm (Yellow)
    "F7": 0.3,  # 630nm (Orange)
    "F8": 0.2,  # 680nm (Red)
    "Clear": 3.0,  # Clear channel
    "NIR": 0.1  # Near IR
}

def generate_artificial_spectral_data(
    base_intensity: float = 1000.0,
    noise_level: float = 0.05
) -> Dict[str, int]:
    """
    Generate artificial spectral data.
    
    Args:
        base_intensity: Base intensity level
        noise_level: Relative noise level (0.0-1.0)
        
    Returns:
        Dictionary of channel values
    """
    data = {}
    for channel, relative_value in SPECTRAL_SHAPE.items():
        # Base value from spectral shape
        value = base_intensity * relative_value
        
        # Add some random noise
        noise = np.random.normal(0, noise_level * value)
        
        # Ensure value is positive and convert to int
        data[channel] = max(0, int(value + noise))
    
    return data

def generate_time_series_data(
    num_points: int = 30,
    duration_seconds: float = 300.0,
    base_intensity: float = 1000.0,
    trend_factor: float = 0.3,
    noise_level: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Generate artificial time series data.
    
    Args:
        num_points: Number of data points
        duration_seconds: Total duration in seconds
        base_intensity: Base intensity level
        trend_factor: Factor for trend change (0.0-1.0)
        noise_level: Relative noise level (0.0-1.0)
        
    Returns:
        List of measurement dictionaries
    """
    measurements = []
    start_time = time.time()
    
    # Create time points with some randomness
    time_points = np.linspace(0, duration_seconds, num_points)
    time_points += np.random.normal(0, duration_seconds / num_points * 0.1, num_points)
    time_points = np.sort(time_points)
    
    # Generate trend (e.g., violet increases while red decreases)
    trend = np.linspace(0, 1, num_points) * trend_factor
    
    for i, t in enumerate(time_points):
        # Generate spectral data with changing intensities
        data = generate_artificial_spectral_data(
            base_intensity=base_intensity,
            noise_level=noise_level
        )
        
        # Apply trend: increase violet, decrease red
        data["F1"] = int(data["F1"] * (1 + trend[i]))  # Violet increases
        data["F8"] = int(data["F8"] * (1 - trend[i] * 0.5))  # Red decreases
        
        # Create measurement with timestamp and raw data
        measurement = {
            "timestamp": start_time + t,
            "elapsed_seconds": t,
            "raw": data,
            # Add ratios
            "ratios": {
                "violet_red": data["F1"] / max(1, data["F8"]),
                "violet_green": data["F1"] / max(1, data["F4"]),
                "green_red": data["F4"] / max(1, data["F8"])
            }
        }
        
        measurements.append(measurement)
    
    return measurements

def generate_parameter_sweep_data(
    parameter: str = "gain",
    values: List[Any] = None,
    base_intensity: float = 1000.0,
    noise_level: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Generate data for a parameter sweep.
    
    Args:
        parameter: Parameter name
        values: List of parameter values
        base_intensity: Base intensity level
        noise_level: Relative noise level (0.0-1.0)
        
    Returns:
        List of measurement dictionaries
    """
    if values is None:
        if parameter == "gain":
            values = [1, 2, 4, 8, 16, 32]
        elif parameter == "integration_time":
            values = [10, 25, 50, 100, 200, 400]
        else:
            values = [1, 2, 3, 4, 5]
    
    measurements = []
    
    for value in values:
        # Scale intensity based on parameter
        if parameter == "gain":
            intensity_scale = value / 8.0  # Normalize to gain of 8x
        elif parameter == "integration_time":
            intensity_scale = value / 100.0  # Normalize to 100ms
        else:
            intensity_scale = 1.0
        
        # Generate spectral data
        data = generate_artificial_spectral_data(
            base_intensity=base_intensity * intensity_scale,
            noise_level=noise_level
        )
        
        # Create measurement
        measurement = {
            parameter: value,
            "raw": data
        }
        
        measurements.append(measurement)
    
    return measurements

def generate_heatmap_data(
    x_values: List[str] = None,
    y_values: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Generate data for a heatmap plot.
    
    Args:
        x_values: List of x-axis values
        y_values: List of y-axis values
        
    Returns:
        Nested dictionary with heatmap data
    """
    if x_values is None:
        x_values = ["1.0μm", "0.5μm", "0.1μm"]
    
    if y_values is None:
        y_values = ["100%", "70%", "40%", "10%", "0%"]
    
    data = {}
    
    # Create a peak in the middle
    peak_x = len(x_values) // 2
    peak_y = len(y_values) // 2
    
    for i, y in enumerate(y_values):
        data[y] = {}
        for j, x in enumerate(x_values):
            # Calculate distance from peak
            distance = np.sqrt(((i - peak_y) / len(y_values))**2 + 
                             ((j - peak_x) / len(x_values))**2)
            
            # Create a peak with some random noise
            value = 1.0 - distance + np.random.normal(0, 0.1)
            data[y][x] = max(0, value)
    
    return data

def load_real_data(filepath: str) -> Dict[str, Any]:
    """
    Load real measurement data from a file.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        Dictionary containing the loaded data
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None

def save_figure(fig, filename: str, output_dir: str):
    """
    Save a figure to a file.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        output_dir: Output directory
    """
    output_path = Path(output_dir) / filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SpectroNeph Visualization Test')
    parser.add_argument('--datafile', help='Path to real data file (optional)')
    parser.add_argument('--output', default='./test_output', help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load real data if specified
    real_data = None
    if args.datafile:
        print(f"Loading real data from {args.datafile}")
        real_data = load_real_data(args.datafile)
    
    # =========================================================
    # Generate plots with artificial data if no real data available
    # =========================================================
    
    # 1. Spectral Profile Plot
    print("Generating spectral profile plot...")
    
    if real_data and 'measurements' in real_data and real_data['measurements']:
        # Use the first measurement from real data
        measurement = real_data['measurements'][0]
        spectral_data = measurement.get('raw', {})
        
        # Fall back to artificial data if no raw data in real measurement
        if not spectral_data:
            spectral_data = generate_artificial_spectral_data()
    else:
        # Generate artificial spectral data
        spectral_data = generate_artificial_spectral_data()
    
    fig_spectral = create_spectral_profile(
        spectral_data,
        title="Spectral Profile"
    )
    save_figure(fig_spectral, "spectral_profile.png", output_dir)
    
    # 2. Time Series Plot
    print("Generating time series plot...")
    
    if real_data and 'measurements' in real_data and len(real_data['measurements']) > 1:
        # Use real measurements
        time_series_data = real_data['measurements']
    else:
        # Generate artificial time series data
        time_series_data = generate_time_series_data()
    
    fig_time_series = create_time_series(
        time_series_data,
        title="Signal Intensity Over Time"
    )
    save_figure(fig_time_series, "time_series.png", output_dir)
    
    # 3. Ratio Analysis Plot
    print("Generating ratio analysis plot...")
    
    # Use the same data as time series
    fig_ratio = create_ratio_analysis(
        time_series_data,
        title="Spectral Ratio Analysis"
    )
    save_figure(fig_ratio, "ratio_analysis.png", output_dir)
    
    # 4. Parameter Comparison Plot - Gain
    print("Generating gain comparison plot...")
    
    gain_data = generate_parameter_sweep_data(
        parameter="gain",
        values=[1, 2, 4, 8, 16, 32, 64]
    )
    
    fig_gain = create_comparison_plot(
        gain_data,
        param_key="gain",
        channel="F4",
        title="Effect of Gain on Signal Intensity"
    )
    save_figure(fig_gain, "gain_comparison.png", output_dir)
    
    # 5. Parameter Comparison Plot - Integration Time
    print("Generating integration time comparison plot...")
    
    int_time_data = generate_parameter_sweep_data(
        parameter="integration_time",
        values=[10, 25, 50, 100, 200, 400, 800]
    )
    
    fig_int_time = create_comparison_plot(
        int_time_data,
        param_key="integration_time",
        channel="F4",
        title="Effect of Integration Time on Signal Intensity"
    )
    save_figure(fig_int_time, "integration_time_comparison.png", output_dir)
    
    # 6. Heatmap Plot
    print("Generating heatmap plot...")
    
    heatmap_data = generate_heatmap_data(
        x_values=["1.0μm", "0.5μm", "0.1μm"],  # Bead sizes
        y_values=["100%", "70%", "40%", "10%", "0%"]  # Salt concentrations
    )
    
    fig_heatmap = create_heatmap(
        heatmap_data,
        title="Agglutination Response by Bead Size and Salt Concentration"
    )
    save_figure(fig_heatmap, "heatmap.png", output_dir)
    
    # 7. Agglutination Overview Plot
    print("Generating agglutination overview plot...")
    
    fig_overview = create_agglutination_overview(
        time_series_data,
        title="Agglutination Analysis Overview"
    )
    save_figure(fig_overview, "agglutination_overview.png", output_dir)
    
    print(f"\nAll plots generated successfully in {output_dir}")
    print("Open these files to view the visualizations.")

if __name__ == "__main__":
    main()