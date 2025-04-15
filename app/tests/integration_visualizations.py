#!/usr/bin/env python3
"""
Nephelometer Visualization Script

This script generates plots, graphs, and tables to visualize the performance
of the AS7341-based nephelometer system using data from test measurements.

Usage:
    python nephelometer_visualization.py [--port COM_PORT] [--output OUTPUT_DIR]
"""

import sys
import time
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.colors as mcolors

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Project root added to sys.path: {project_root}")

# Import modules from the SpectroNeph project directly
sys.path.insert(0, str(project_root / "app"))
from core.device import DeviceManager
from hardware.nephelometer import Nephelometer, MeasurementMode, AgglutinationState
from utils.logging import setup_logging, get_logger
from utils.helpers import detect_serial_port

# Initialize logger
setup_logging()
logger = get_logger("nephelometer_viz")

# Wavelength mapping for AS7341 channels
WAVELENGTH_MAP = {
    "F1": 415,
    "F2": 445,
    "F3": 480,
    "F4": 515,
    "F5": 555,
    "F6": 590,
    "F7": 630,
    "F8": 680,
    "Clear": 0,  # Not a specific wavelength
    "NIR": 0     # Not a specific wavelength
}

# Colors for plotting
CHANNEL_COLORS = {
    "F1": "#8B00FF",  # Violet
    "F2": "#0000FF",  # Blue
    "F3": "#0080FF",  # Light Blue
    "F4": "#00FF00",  # Green
    "F5": "#80FF00",  # Lime
    "F6": "#FFFF00",  # Yellow
    "F7": "#FF8000",  # Orange
    "F8": "#FF0000",  # Red
    "Clear": "#808080",  # Gray
    "NIR": "#404040"   # Dark Gray
}

def collect_measurement_data(nephelometer, config, num_measurements=10, include_kinetic=True):
    """
    Collect measurement data from the nephelometer for visualization.
    
    Args:
        nephelometer: Initialized nephelometer instance
        config: Configuration parameters for the nephelometer
        num_measurements: Number of standard measurements to take
        include_kinetic: Whether to include kinetic measurements
        
    Returns:
        dict: Collected measurement data
    """
    data = {
        "single": [],
        "kinetic": [],
        "gain_sweep": [],
        "integration_sweep": [],
        "metadata": {
            "timestamp": time.time(),
            "config": config.copy()
        }
    }
    
    # Apply configuration
    nephelometer.configure(config)
    
    # 1. Take single measurements
    print("Taking single measurements...")
    nephelometer.set_led(True)
    time.sleep(0.2)  # Allow LED to stabilize
    
    for i in range(num_measurements):
        measurement = nephelometer.take_single_measurement(subtract_background=False)
        data["single"].append(measurement)
        print(f"  Measurement {i+1}/{num_measurements} completed")
    
    # 2. Take measurements with different gain settings
    print("Taking gain sweep measurements...")
    gain_values = [3, 5, 7, 9, 10]  # Different gain values to try
    for gain in gain_values:
        nephelometer.configure({"gain": gain})
        time.sleep(0.1)  # Allow settings to take effect
        measurement = nephelometer.take_single_measurement(subtract_background=False)
        measurement["sweep_parameter"] = gain
        data["gain_sweep"].append(measurement)
        print(f"  Gain {gain} measurement completed")
    
    # 3. Take measurements with different integration times
    print("Taking integration time sweep measurements...")
    integration_times = [50, 100, 200, 300, 400]  # Different integration times to try
    nephelometer.configure({"gain": config["gain"]})  # Reset gain
    for integration_time in integration_times:
        nephelometer.configure({"integration_time": integration_time})
        time.sleep(0.1)  # Allow settings to take effect
        measurement = nephelometer.take_single_measurement(subtract_background=False)
        measurement["sweep_parameter"] = integration_time
        data["integration_sweep"].append(measurement)
        print(f"  Integration time {integration_time}ms measurement completed")
    
    # 4. Take kinetic measurements
    if include_kinetic:
        kinetic_data = []
        
        def kinetic_callback(data):
            kinetic_data.append(data.copy())
        
        print("Taking kinetic measurements...")
        nephelometer.configure(config)  # Reset configuration
        nephelometer.start_kinetic_measurement(
            duration_seconds=3.0,      # 3 second measurement
            samples_per_second=3.0,    # 3Hz sampling
            callback=kinetic_callback,
            subtract_background=False
        )
        
        # Wait for completion
        time.sleep(4.0)
        data["kinetic"] = kinetic_data
        print(f"  Kinetic measurement completed with {len(kinetic_data)} samples")
    
    # Turn off LED
    nephelometer.set_led(False)
    
    return data

def save_measurement_data(data, output_dir):
    """
    Save measurement data to a file.
    
    Args:
        data: Measurement data to save
        output_dir: Directory to save the data
    
    Returns:
        Path: Path to the saved file
    """
    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a timestamped filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = output_dir / f"nephelometer_data_{timestamp}.json"
    
    # Save the data to a JSON file
    with open(filename, 'w') as f:
        # We need to convert AgglutinationState objects to strings
        json.dump(data, f, default=lambda obj: obj.name if isinstance(obj, AgglutinationState) else str(obj), indent=2)
    
    print(f"Data saved to {filename}")
    return filename

def load_measurement_data(filename):
    """
    Load measurement data from a file.
    
    Args:
        filename: Path to the data file
        
    Returns:
        dict: Loaded measurement data
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data

def generate_spectral_profile_plot(data, output_dir):
    """
    Generate a spectral profile plot for the nephelometer.
    
    Args:
        data: Measurement data
        output_dir: Directory to save the plot
    """
    # Extract the first single measurement
    if not data["single"]:
        print("Warning: No single measurement data available")
        return
    
    measurement = data["single"][0]
    raw_data = measurement["raw"]
    
    # Filter to only specific channels (F1-F8)
    channels = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
    wavelengths = [WAVELENGTH_MAP[ch] for ch in channels]
    values = [raw_data.get(ch, 0) for ch in channels]
    colors = [CHANNEL_COLORS[ch] for ch in channels]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot as both bars and a connected line
    plt.bar(wavelengths, values, width=20, alpha=0.7, color=colors)
    plt.plot(wavelengths, values, 'o-', color='black', alpha=0.7, linewidth=2)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Raw Signal Value')
    plt.title('AS7341 Nephelometer Spectral Profile')
    plt.grid(True, alpha=0.3)
    
    # Add labels above each bar
    for i, v in enumerate(values):
        plt.text(wavelengths[i], v + max(values)*0.03, f"{channels[i]}", 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
                
    # Add configuration info
    config = data["metadata"]["config"]
    plt.figtext(0.02, 0.02, 
               f"Settings: Gain={config.get('gain', 'N/A')}, Integration={config.get('integration_time', 'N/A')}ms, LED={config.get('led_current', 'N/A')}mA",
               fontsize=9)
    
    # Save the plot
    output_path = output_dir / "spectral_profile.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Spectral profile plot saved to {output_path}")

def generate_gain_response_plot(data, output_dir):
    """
    Generate a gain response plot.
    
    Args:
        data: Measurement data
        output_dir: Directory to save the plot
    """
    # Extract gain sweep data
    if not data["gain_sweep"]:
        print("Warning: No gain sweep data available")
        return
    
    # Extract data
    gain_values = [m["sweep_parameter"] for m in data["gain_sweep"]]
    
    # Calculate the gain ratio (2^(gain-1))
    gain_ratios = [2**(g-1) for g in gain_values]
    
    # Get selected channels to plot
    channels = ["F1", "F4", "F8"]
    channel_values = {ch: [] for ch in channels}
    
    for measurement in data["gain_sweep"]:
        raw_data = measurement["raw"]
        for ch in channels:
            channel_values[ch].append(raw_data.get(ch, 0))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    for ch in channels:
        plt.plot(gain_values, channel_values[ch], 'o-', 
                 label=f"{ch} ({WAVELENGTH_MAP[ch]}nm)", color=CHANNEL_COLORS[ch])
    
    # Add theoretical linear response
    max_idx = 0
    max_val = 0
    for i, vals in enumerate(zip(*channel_values.values())):
        avg_val = sum(vals) / len(vals)
        if avg_val > max_val:
            max_val = avg_val
            max_idx = i
    
    if len(gain_values) > 1:
        ref_gain = gain_values[max_idx]
        ref_value = max_val
        
        theoretical_values = [ref_value * (2**(g-ref_gain)) for g in gain_values]
        plt.plot(gain_values, theoretical_values, '--', color='black', 
                label='Theoretical (2x per step)', alpha=0.7)
    
    plt.yscale('log')
    plt.xlabel('Gain Setting')
    plt.ylabel('Signal Value (log scale)')
    plt.title('AS7341 Nephelometer Gain Response')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add secondary axis with actual gain values
    ax2 = plt.gca().twiny()
    ax2.set_xticks(gain_values)
    ax2.set_xticklabels([f"{r}x" for r in gain_ratios])
    ax2.set_xlabel("Actual Gain")
    
    # Save the plot
    output_path = output_dir / "gain_response.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gain response plot saved to {output_path}")

def generate_integration_time_plot(data, output_dir):
    """
    Generate an integration time response plot.
    
    Args:
        data: Measurement data
        output_dir: Directory to save the plot
    """
    # Extract integration time sweep data
    if not data["integration_sweep"]:
        print("Warning: No integration time sweep data available")
        return
    
    # Extract data
    integration_times = [m["sweep_parameter"] for m in data["integration_sweep"]]
    
    # Get selected channels to plot
    channels = ["F1", "F4", "F8"]
    channel_values = {ch: [] for ch in channels}
    
    for measurement in data["integration_sweep"]:
        raw_data = measurement["raw"]
        for ch in channels:
            channel_values[ch].append(raw_data.get(ch, 0))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    for ch in channels:
        plt.plot(integration_times, channel_values[ch], 'o-', 
                 label=f"{ch} ({WAVELENGTH_MAP[ch]}nm)", color=CHANNEL_COLORS[ch])
    
    # Add theoretical linear response
    if len(integration_times) > 1:
        ref_idx = 0
        ref_time = integration_times[ref_idx]
        ref_values = {ch: channel_values[ch][ref_idx] for ch in channels}
        
        # Use F4 (green) as reference for theoretical line
        ref_ch = "F4"
        theoretical_values = [ref_values[ref_ch] * (t / ref_time) for t in integration_times]
        plt.plot(integration_times, theoretical_values, '--', color='black', 
                label='Theoretical (linear)', alpha=0.7)
    
    plt.xlabel('Integration Time (ms)')
    plt.ylabel('Signal Value')
    plt.title('AS7341 Nephelometer Integration Time Response')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = output_dir / "integration_time_response.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Integration time response plot saved to {output_path}")

def generate_repeatability_plot(data, output_dir):
    """
    Generate a measurement repeatability plot.
    
    Args:
        data: Measurement data
        output_dir: Directory to save the plot
    """
    # Extract single measurement data
    if not data["single"] or len(data["single"]) < 2:
        print("Warning: Not enough single measurements for repeatability analysis")
        return
    
    # Get selected channels to analyze
    channels = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
    
    # Extract values for each channel
    channel_values = {ch: [] for ch in channels}
    for measurement in data["single"]:
        raw_data = measurement["raw"]
        for ch in channels:
            channel_values[ch].append(raw_data.get(ch, 0))
    
    # Calculate statistics
    stats = {}
    for ch in channels:
        values = channel_values[ch]
        mean = np.mean(values)
        std = np.std(values)
        cv = (std / mean) * 100 if mean > 0 else 0  # Coefficient of variation (%)
        stats[ch] = {
            "mean": mean,
            "std": std,
            "cv": cv,
            "min": np.min(values),
            "max": np.max(values)
        }
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Mean and standard deviation by channel
    plt.subplot(2, 1, 1)
    x = np.arange(len(channels))
    means = [stats[ch]["mean"] for ch in channels]
    stds = [stats[ch]["std"] for ch in channels]
    colors = [CHANNEL_COLORS[ch] for ch in channels]
    
    plt.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    plt.xticks(x, channels)
    plt.ylabel('Signal Value')
    plt.title('Mean Signal Values with Standard Deviation')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Coefficient of variation by channel
    plt.subplot(2, 1, 2)
    cvs = [stats[ch]["cv"] for ch in channels]
    
    plt.bar(x, cvs, color=colors, alpha=0.7)
    plt.xticks(x, channels)
    plt.ylabel('Coefficient of Variation (%)')
    plt.title('Measurement Repeatability (lower is better)')
    plt.grid(True, alpha=0.3)
    
    # Add thresholds for visual reference
    plt.axhline(y=1, linestyle='--', color='green', alpha=0.7, label='CV = 1%')
    plt.axhline(y=5, linestyle='--', color='orange', alpha=0.7, label='CV = 5%')
    plt.axhline(y=10, linestyle='--', color='red', alpha=0.7, label='CV = 10%')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / "repeatability.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Repeatability plot saved to {output_path}")
    
    # Create a table of statistics
    stats_table = pd.DataFrame(stats).T
    stats_table.index.name = "Channel"
    stats_table.columns = ["Mean", "Std Dev", "CV (%)", "Min", "Max"]
    stats_table = stats_table.round(2)
    
    # Save the table to CSV
    table_path = output_dir / "repeatability_stats.csv"
    stats_table.to_csv(table_path)
    print(f"Repeatability statistics saved to {table_path}")

def generate_channel_correlation_plot(data, output_dir):
    """
    Generate a channel correlation heatmap.
    
    Args:
        data: Measurement data
        output_dir: Directory to save the plot
    """
    # Extract single measurement data
    if not data["single"] or len(data["single"]) < 3:
        print("Warning: Not enough single measurements for correlation analysis")
        return
    
    # Get channels to analyze
    channels = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
    
    # Extract values for each channel
    channel_values = {ch: [] for ch in channels}
    for measurement in data["single"]:
        raw_data = measurement["raw"]
        for ch in channels:
            channel_values[ch].append(raw_data.get(ch, 0))
    
    # Create a DataFrame for correlation analysis
    df = pd.DataFrame(channel_values)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add text annotations
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                     ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white")
    
    plt.colorbar(label='Correlation Coefficient')
    plt.xticks(np.arange(len(channels)), channels, rotation=45)
    plt.yticks(np.arange(len(channels)), channels)
    plt.title('Channel Correlation Matrix')
    
    # Save the plot
    output_path = output_dir / "channel_correlation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Channel correlation plot saved to {output_path}")

def generate_kinetic_plot(data, output_dir):
    """
    Generate a kinetic measurement plot.
    
    Args:
        data: Measurement data
        output_dir: Directory to save the plot
    """
    # Extract kinetic data
    if not data["kinetic"]:
        print("Warning: No kinetic data available")
        return
    
    # Get channels to analyze
    channels = ["F1", "F4", "F8"]
    
    # Extract values for each channel over time
    channel_values = {ch: [] for ch in channels}
    timestamps = []
    
    for i, measurement in enumerate(data["kinetic"]):
        raw_data = measurement["raw"]
        for ch in channels:
            channel_values[ch].append(raw_data.get(ch, 0))
        timestamps.append(measurement.get("elapsed_seconds", i/3.0))  # Default to estimated time
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    for ch in channels:
        plt.plot(timestamps, channel_values[ch], 'o-', 
                label=f"{ch} ({WAVELENGTH_MAP[ch]}nm)", color=CHANNEL_COLORS[ch])
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal Value')
    plt.title('AS7341 Nephelometer Kinetic Measurement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = output_dir / "kinetic_measurement.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Kinetic measurement plot saved to {output_path}")
    
    # Also plot the spectral ratios over time if available
    ratio_types = ["violet_red", "violet_green", "green_red"]
    ratio_values = {ratio: [] for ratio in ratio_types}
    ratio_names = {
        "violet_red": "Violet/Red (F1/F8)",
        "violet_green": "Violet/Green (F1/F4)",
        "green_red": "Green/Red (F4/F8)"
    }
    
    has_ratios = False
    for measurement in data["kinetic"]:
        if "ratios" in measurement:
            has_ratios = True
            for ratio in ratio_types:
                ratio_values[ratio].append(measurement["ratios"].get(ratio, 0))
    
    if has_ratios:
        plt.figure(figsize=(10, 6))
        
        for ratio in ratio_types:
            plt.plot(timestamps, ratio_values[ratio], 'o-', 
                    label=ratio_names[ratio])
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Ratio Value')
        plt.title('Spectral Ratios During Kinetic Measurement')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        output_path = output_dir / "kinetic_ratios.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Kinetic ratios plot saved to {output_path}")

def generate_spectral_ratio_plot(data, output_dir):
    """
    Generate a spectral ratio comparison plot.
    
    Args:
        data: Measurement data
        output_dir: Directory to save the plot
    """
    # Extract single measurement data
    if not data["single"]:
        print("Warning: No single measurement data available")
        return
    
    # Extract the average ratio values
    ratio_values = {"violet_red": [], "violet_green": [], "green_red": []}
    
    for measurement in data["single"]:
        if "ratios" in measurement:
            for ratio in ratio_values.keys():
                ratio_values[ratio].append(measurement["ratios"].get(ratio, 0))
    
    if not all(ratio_values.values()):
        print("Warning: Not enough ratio data available")
        return
    
    # Calculate statistics for each ratio
    ratio_stats = {}
    for ratio, values in ratio_values.items():
        ratio_stats[ratio] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }
    
    # Create bar plot with error bars
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(ratio_values))
    ratio_means = [ratio_stats[r]["mean"] for r in ratio_values]
    ratio_stds = [ratio_stats[r]["std"] for r in ratio_values]
    
    # Use meaningful names for x-axis labels
    labels = ["Violet/Red (F1/F8)", "Violet/Green (F1/F4)", "Green/Red (F4/F8)"]
    
    plt.bar(x, ratio_means, yerr=ratio_stds, alpha=0.7, capsize=5,
            color=['purple', 'teal', 'orange'])
    plt.xticks(x, labels)
    plt.ylabel('Ratio Value')
    plt.title('AS7341 Nephelometer Spectral Ratios')
    plt.grid(True, alpha=0.3)
    
    # Add the values above the bars
    for i, v in enumerate(ratio_means):
        plt.text(i, v + ratio_stds[i] + 0.03, f"{v:.2f}", ha='center')
    
    # Save the plot
    output_path = output_dir / "spectral_ratios.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Spectral ratio plot saved to {output_path}")

def generate_agglutination_simulation_plot(data, output_dir):
    """
    Generate a plot simulating different agglutination states based on ratio manipulation.
    
    Args:
        data: Measurement data
        output_dir: Directory to save the plot
    """
    # Extract a single measurement as baseline
    if not data["single"]:
        print("Warning: No single measurement data available")
        return
    
    baseline = data["single"][0]["raw"]
    
    # Create simulated agglutination states by adjusting spectral ratios
    # This is a simplified model - in reality, agglutination has complex effects
    agglutination_levels = [0.0, 0.5, 1.0, 2.0, 3.0]  # Represents none to complete
    simulated_data = []
    
    for level in agglutination_levels:
        # Make a copy of the baseline
        sim_data = baseline.copy()
        
        # Apply the agglutination model:
        # - Increase violet channel (F1) relative to red (F8)
        # - Moderate increase in blue (F2, F3)
        # - Small decrease in red (F8)
        # This is based on typical spectral shifts during agglutination
        if level > 0:
            # Increase F1 (violet)
            sim_data["F1"] = int(baseline["F1"] * (1 + 0.2 * level))
            # Increase F2, F3 (blue)
            sim_data["F2"] = int(baseline["F2"] * (1 + 0.15 * level))
            sim_data["F3"] = int(baseline["F3"] * (1 + 0.1 * level))
            # Decrease F8 (red)
            sim_data["F8"] = int(baseline["F8"] * (1 - 0.05 * level))
        
        simulated_data.append(sim_data)
    
    # Plot the simulated spectral profiles
    channels = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
    wavelengths = [WAVELENGTH_MAP[ch] for ch in channels]
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Spectral profiles
    plt.subplot(2, 1, 1)
    
    for i, (level, sim_data) in enumerate(zip(agglutination_levels, simulated_data)):
        values = [sim_data.get(ch, 0) for ch in channels]
        plt.plot(wavelengths, values, 'o-', 
                label=f"Agglutination Level {level}", 
                alpha=0.8, linewidth=2)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Signal Value')
    plt.title('Simulated Spectral Profiles for Different Agglutination Levels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Spectral ratios
    plt.subplot(2, 1, 2)
    
    # Calculate ratios for each simulated state
    ratio_types = ["violet_red", "violet_green", "green_red"]
    ratio_values = {ratio: [] for ratio in ratio_types}
    
    for sim_data in simulated_data:
        f1 = sim_data.get("F1", 1)
        f4 = sim_data.get("F4", 1)
        f8 = sim_data.get("F8", 1)
        
        ratio_values["violet_red"].append(f1 / max(1, f8))
        ratio_values["violet_green"].append(f1 / max(1, f4))
        ratio_values["green_red"].append(f4 / max(1, f8))
    
    # Plot ratios
    width = 0.25
    x = np.arange(len(agglutination_levels))
    
    plt.bar(x - width, ratio_values["violet_red"], width, label='Violet/Red', color='purple')
    plt.bar(x, ratio_values["violet_green"], width, label='Violet/Green', color='teal')
    plt.bar(x + width, ratio_values["green_red"], width, label='Green/Red', color='orange')
    
    plt.xlabel('Agglutination Level')
    plt.ylabel('Ratio Value')
    plt.title('Spectral Ratios for Simulated Agglutination States')
    plt.xticks(x, [str(level) for level in agglutination_levels])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / "agglutination_simulation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Agglutination simulation plot saved to {output_path}")

def generate_system_performance_dashboard(data, output_dir):
    """
    Generate a summary dashboard of system performance.
    
    Args:
        data: Measurement data
        output_dir: Directory to save the plot
    """
    # Create a dashboard with key performance metrics
    if not data["single"]:
        print("Warning: No single measurement data available")
        return
    
    # Extract data for dashboard
    # 1. Signal levels per channel
    channels = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
    avg_values = {ch: 0 for ch in channels}
    for measurement in data["single"]:
        raw_data = measurement["raw"]
        for ch in channels:
            avg_values[ch] += raw_data.get(ch, 0) / len(data["single"])
    
    # 2. Signal-to-noise ratio (estimated from repeatability)
    snr_estimates = {}
    if len(data["single"]) > 1:
        for ch in channels:
            values = [m["raw"].get(ch, 0) for m in data["single"]]
            mean = np.mean(values)
            std = np.std(values) if len(values) > 1 else 1
            snr = mean / std if std > 0 else 0
            snr_estimates[ch] = snr
    
    # 3. Key spectral ratios
    avg_ratios = {"violet_red": 0, "violet_green": 0, "green_red": 0}
    ratio_count = 0
    for measurement in data["single"]:
        if "ratios" in measurement:
            for ratio in avg_ratios:
                avg_ratios[ratio] += measurement["ratios"].get(ratio, 0)
            ratio_count += 1
    
    if ratio_count > 0:
        for ratio in avg_ratios:
            avg_ratios[ratio] /= ratio_count
    
    # Create the dashboard
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Signal levels
    plt.subplot(2, 2, 1)
    x = np.arange(len(channels))
    values = [avg_values[ch] for ch in channels]
    colors = [CHANNEL_COLORS[ch] for ch in channels]
    
    plt.bar(x, values, color=colors, alpha=0.7)
    plt.xticks(x, channels)
    plt.ylabel('Average Signal')
    plt.title('Channel Signal Levels')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: SNR estimates
    plt.subplot(2, 2, 2)
    if snr_estimates:
        x = np.arange(len(channels))
        values = [snr_estimates[ch] for ch in channels]
        
        plt.bar(x, values, color=colors, alpha=0.7)
        plt.xticks(x, channels)
        plt.ylabel('Signal-to-Noise Ratio')
        plt.title('Estimated Channel SNR')
        plt.grid(True, alpha=0.3)
        
        # Log scale to better see differences
        plt.yscale('log')
        
        # Add threshold lines
        plt.axhline(y=10, linestyle='--', color='orange', alpha=0.7, label='SNR = 10')
        plt.axhline(y=100, linestyle='--', color='green', alpha=0.7, label='SNR = 100')
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Insufficient data for SNR calculation", 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    # Plot 3: Spectral Ratios
    plt.subplot(2, 2, 3)
    if ratio_count > 0:
        x = np.arange(len(avg_ratios))
        values = list(avg_ratios.values())
        
        # Use meaningful names for x-axis labels
        labels = ["Violet/Red\n(F1/F8)", "Violet/Green\n(F1/F4)", "Green/Red\n(F4/F8)"]
        
        plt.bar(x, values, alpha=0.7, color=['purple', 'teal', 'orange'])
        plt.xticks(x, labels)
        plt.ylabel('Ratio Value')
        plt.title('Key Spectral Ratios')
        plt.grid(True, alpha=0.3)
        
        # Add the values above the bars
        for i, v in enumerate(values):
            plt.text(i, v + 0.03, f"{v:.2f}", ha='center')
    else:
        plt.text(0.5, 0.5, "No ratio data available", 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    # Plot 4: System specifications
    plt.subplot(2, 2, 4)
    config = data["metadata"]["config"]
    
    specs = [
        f"Sample: 0.02% polystyrene beads",
        f"Gain: {config.get('gain', 'N/A')}",
        f"Integration time: {config.get('integration_time', 'N/A')} ms",
        f"LED current: {config.get('led_current', 'N/A')} mA",
        f"Sensor: AS7341 (8-channel spectral)",
        f"Measurement angle: 90°",
        f"Sample tests: {len(data['single'])}",
        f"Average SNR: {np.mean(list(snr_estimates.values())):.1f}" if snr_estimates else "SNR: N/A",
    ]
    
    plt.axis('off')
    plt.title('System Specifications')
    
    y_pos = 0.8
    for spec in specs:
        plt.text(0.1, y_pos, spec, fontsize=10, transform=plt.gca().transAxes)
        y_pos -= 0.1
    
    plt.tight_layout()
    
    # Save the dashboard
    output_path = output_dir / "system_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"System performance dashboard saved to {output_path}")

def main():
    """Main entry point for the visualization script."""
    parser = argparse.ArgumentParser(description='Generate visualizations for nephelometer data')
    parser.add_argument('--port', help='Serial port to connect to')
    parser.add_argument('--input', help='Input JSON file with measurement data (if not collecting new data)')
    parser.add_argument('--output', default='./test_data', help='Output directory for visualizations')
    parser.add_argument('--no-collect', action='store_true', help='Skip data collection and use input file')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load or collect data
    if args.no_collect and args.input:
        # Load data from file
        print(f"Loading data from {args.input}...")
        data = load_measurement_data(args.input)
    else:
        # Collect new data
        # Connect to device
        device_manager = DeviceManager()
        
        # Use provided port or detect
        port = args.port if args.port else detect_serial_port()
        if not device_manager.connect(port=port):
            print("❌ Could not connect to device. Make sure it's connected and the port is correct.")
            return 1
        
        print(f"✓ Connected to device on {device_manager._comm._port}")
        
        # Create and initialize nephelometer
        nephelometer = Nephelometer(device_manager)
        nephelometer.initialize()
        
        # Configuration for optimal signal with 0.02% beads
        config = {
            "gain": 10,                # Maximum gain
            "integration_time": 300,   # Long integration time
            "led_current": 20          # Maximum LED current
        }
        
        # Collect measurements
        data = collect_measurement_data(nephelometer, config, num_measurements=20)
        
        # Save the data
        data_file = save_measurement_data(data, output_dir)
        
        # Disconnect
        device_manager.disconnect()
        print("✓ Disconnected from device")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_spectral_profile_plot(data, output_dir)
    generate_gain_response_plot(data, output_dir)
    generate_integration_time_plot(data, output_dir)
    generate_repeatability_plot(data, output_dir)
    generate_channel_correlation_plot(data, output_dir)
    generate_kinetic_plot(data, output_dir)
    generate_spectral_ratio_plot(data, output_dir)
    generate_agglutination_simulation_plot(data, output_dir)
    generate_system_performance_dashboard(data, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())