"""
Core plotting functionality for SpectroNeph.

This module provides visualization tools for spectral data from the AS7341
nephelometer, including spectral profiles, time series, and ratio analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import datetime

# Constants
# AS7341 channel information (wavelength in nm)
CHANNEL_WAVELENGTHS = {
    "F1": 415,  # Violet
    "F2": 445,  # Indigo
    "F3": 480,  # Blue
    "F4": 515,  # Cyan/Green
    "F5": 555,  # Green
    "F6": 590,  # Yellow
    "F7": 630,  # Orange
    "F8": 680,  # Red
    "Clear": 0,  # Clear channel (not at a specific wavelength)
    "NIR": 0,    # Near IR (not at a specific wavelength)
}

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

def create_spectral_profile(
    data: Dict[str, int], 
    title: Optional[str] = None, 
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Create a spectral profile figure from channel data.
    
    Args:
        data: Dictionary of channel values
        title: Optional title for the plot
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter to spectral channels (F1-F8)
    channels = [ch for ch in data.keys() if ch.startswith('F') and ch[1:].isdigit()]
    channels.sort()  # Ensure channels are in order
    
    if not channels:
        ax.text(0.5, 0.5, "No spectral data available", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Extract wavelengths and values
    wavelengths = [CHANNEL_WAVELENGTHS[ch] for ch in channels]
    values = [data[ch] for ch in channels]
    colors = [CHANNEL_COLORS[ch] for ch in channels]
    
    # Create bar chart
    bars = ax.bar(wavelengths, values, width=20, color=colors, alpha=0.7)
    
    # Add connecting line
    ax.plot(wavelengths, values, 'o-', color='black', alpha=0.7, linewidth=1.5)
    
    # Add labels and grid
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Signal Intensity')
    ax.set_title(title or 'Spectral Profile')
    ax.grid(True, alpha=0.3)
    
    # Add channel labels above bars
    for i, channel in enumerate(channels):
        ax.text(wavelengths[i], values[i] + (max(values) * 0.03), 
                channel, ha='center', va='bottom', fontsize=9)
    
    # Ensure X-axis covers all wavelengths with some padding
    ax.set_xlim(min(wavelengths) - 30, max(wavelengths) + 30)
    
    # Ensure Y-axis starts at 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig

def create_time_series(
    measurements: List[Dict[str, Any]], 
    channels: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Create a time series figure from multiple measurements.
    
    Args:
        measurements: List of measurement dictionaries
        channels: Optional list of channels to plot (defaults to F1, F4, F8)
        title: Optional title for the plot
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    if not measurements:
        ax.text(0.5, 0.5, "No measurement data available", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Default channels if not specified (violet, green, red)
    if channels is None:
        channels = ["F1", "F4", "F8"]
    
    # Filter to available channels
    available_channels = []
    for ch in channels:
        # Check if this channel exists in the first measurement
        if ch in measurements[0].get('raw', {}) or ch in measurements[0]:
            available_channels.append(ch)
    
    if not available_channels:
        ax.text(0.5, 0.5, "No channel data available", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Extract timestamps and values
    timestamps = []
    channel_values = {ch: [] for ch in available_channels}
    
    for m in measurements:
        # Extract timestamp
        if 'timestamp' in m:
            timestamps.append(m['timestamp'])
        elif 'elapsed_seconds' in m:
            timestamps.append(m['elapsed_seconds'])
        else:
            # Use index as timestamp if none available
            timestamps.append(len(timestamps))
        
        # Extract channel values
        for ch in available_channels:
            if 'raw' in m and ch in m['raw']:
                channel_values[ch].append(m['raw'][ch])
            elif ch in m:
                channel_values[ch].append(m[ch])
            else:
                channel_values[ch].append(0)  # Default if not found
    
    # Convert timestamps to elapsed time if they're absolute
    if timestamps and timestamps[0] > 1000000000:  # Unix timestamp check
        start_time = timestamps[0]
        timestamps = [t - start_time for t in timestamps]
    
    # Plot each channel
    for ch in available_channels:
        ax.plot(timestamps, channel_values[ch], 'o-', 
                label=f"{ch} ({CHANNEL_WAVELENGTHS[ch]}nm)", 
                color=CHANNEL_COLORS.get(ch, 'black'),
                alpha=0.8, linewidth=2)
    
    # Add labels and grid
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Signal Intensity')
    ax.set_title(title or 'Time Series')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_ratio_analysis(
    measurements: List[Dict[str, Any]],
    ratio_types: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Figure:
    """
    Create a ratio analysis figure from multiple measurements.
    
    Args:
        measurements: List of measurement dictionaries
        ratio_types: Optional list of ratio types to plot
        title: Optional title for the plot
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure with two subplots (time series and distribution)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    if not measurements:
        ax1.text(0.5, 0.5, "No measurement data available", 
                 ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "No measurement data available", 
                 ha='center', va='center', transform=ax2.transAxes)
        return fig
    
    # Default ratio types if not specified
    if ratio_types is None:
        ratio_types = ["violet_red", "violet_green", "green_red"]
    
    # Check if ratios are available
    has_ratios = False
    for m in measurements:
        if 'ratios' in m and any(r in m['ratios'] for r in ratio_types):
            has_ratios = True
            break
    
    if not has_ratios:
        # Calculate ratios if not available
        for m in measurements:
            if 'ratios' not in m:
                m['ratios'] = {}
            
            # Get raw data (either directly or from 'raw' key)
            raw_data = m.get('raw', m)
            
            # Calculate common ratios if the necessary channels exist
            if 'F1' in raw_data and 'F8' in raw_data:
                m['ratios']['violet_red'] = raw_data['F1'] / max(1, raw_data['F8'])
            
            if 'F1' in raw_data and 'F4' in raw_data:
                m['ratios']['violet_green'] = raw_data['F1'] / max(1, raw_data['F4'])
            
            if 'F4' in raw_data and 'F8' in raw_data:
                m['ratios']['green_red'] = raw_data['F4'] / max(1, raw_data['F8'])
    
    # Extract timestamps and ratio values
    timestamps = []
    ratio_values = {r: [] for r in ratio_types}
    
    for m in measurements:
        # Extract timestamp
        if 'timestamp' in m:
            timestamps.append(m['timestamp'])
        elif 'elapsed_seconds' in m:
            timestamps.append(m['elapsed_seconds'])
        else:
            # Use index as timestamp if none available
            timestamps.append(len(timestamps))
        
        # Extract ratio values
        for ratio in ratio_types:
            if 'ratios' in m and ratio in m['ratios']:
                ratio_values[ratio].append(m['ratios'][ratio])
            else:
                ratio_values[ratio].append(0)  # Default if not found
    
    # Convert timestamps to elapsed time if they're absolute
    if timestamps and timestamps[0] > 1000000000:  # Unix timestamp check
        start_time = timestamps[0]
        timestamps = [t - start_time for t in timestamps]
    
    # Plot 1: Time series of ratios
    for ratio in ratio_types:
        if any(ratio_values[ratio]):  # Only plot if there are non-zero values
            ax1.plot(timestamps, ratio_values[ratio], 'o-', 
                    label=f"{ratio.replace('_', '/')}", 
                    alpha=0.8, linewidth=2)
    
    # Add labels and grid for plot 1
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Ratio Value')
    ax1.set_title(title or 'Spectral Ratios Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Distribution of ratio values
    for ratio in ratio_types:
        values = [v for v in ratio_values[ratio] if v > 0]  # Filter out zeros
        if values:
            ax2.hist(values, bins=20, alpha=0.5, label=f"{ratio.replace('_', '/')}")
    
    # Add labels and grid for plot 2
    ax2.set_xlabel('Ratio Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Ratio Values')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def create_comparison_plot(
    measurements: List[Dict[str, Any]],
    param_key: str,
    channel: str = "F4",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Create a comparison plot showing how a channel value changes with a parameter.
    
    Args:
        measurements: List of measurement dictionaries
        param_key: Key for the parameter to compare (e.g., 'gain', 'integration_time')
        channel: Channel to plot values for
        title: Optional title for the plot
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    if not measurements:
        ax.text(0.5, 0.5, "No measurement data available", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Extract parameter values and channel values
    param_values = []
    channel_values = []
    
    for m in measurements:
        # Get parameter value
        if param_key in m:
            param_val = m[param_key]
        elif 'metadata' in m and param_key in m['metadata']:
            param_val = m['metadata'][param_key]
        elif 'config' in m and param_key in m['config']:
            param_val = m['config'][param_key]
        else:
            continue  # Skip if parameter not found
        
        # Get channel value
        if 'raw' in m and channel in m['raw']:
            channel_val = m['raw'][channel]
        elif channel in m:
            channel_val = m[channel]
        else:
            continue  # Skip if channel not found
        
        param_values.append(param_val)
        channel_values.append(channel_val)
    
    if not param_values:
        ax.text(0.5, 0.5, f"No data found for parameter '{param_key}' and channel '{channel}'", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Create scatter plot with connecting line
    ax.plot(param_values, channel_values, 'o-', 
            color=CHANNEL_COLORS.get(channel, 'blue'),
            alpha=0.8, linewidth=2, markersize=8)
    
    # Add labels and grid
    ax.set_xlabel(param_key.replace('_', ' ').title())
    ax.set_ylabel(f'{channel} Signal Intensity')
    ax.set_title(title or f'Effect of {param_key.replace("_", " ").title()} on {channel} Channel')
    ax.grid(True, alpha=0.3)
    
    # Add points to make it easier to see
    for x, y in zip(param_values, channel_values):
        ax.text(x, y, f"{y}", ha='left', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_heatmap(
    data: Dict[str, Dict[str, float]],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Figure:
    """
    Create a heatmap visualization.
    
    Args:
        data: Nested dictionary where first key is y-axis, second key is x-axis,
              and values are the intensities
        title: Optional title for the plot
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    if not data:
        ax.text(0.5, 0.5, "No data available for heatmap", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Extract x and y axis labels
    y_labels = list(data.keys())
    x_labels = []
    for y in y_labels:
        x_labels.extend(data[y].keys())
    x_labels = sorted(list(set(x_labels)))
    
    # Create data array
    array = np.zeros((len(y_labels), len(x_labels)))
    for i, y in enumerate(y_labels):
        for j, x in enumerate(x_labels):
            if x in data[y]:
                array[i, j] = data[y][x]
    
    # Create heatmap
    im = ax.imshow(array, cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value')
    
    # Add labels and grid
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    
    # Rotate x labels if there are many
    if len(x_labels) > 5:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title
    ax.set_title(title or 'Heatmap Visualization')
    
    # Add values in each cell
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            ax.text(j, i, f"{array[i, j]:.1f}", 
                    ha="center", va="center", 
                    color="white" if array[i, j] > np.mean(array) else "black")
    
    plt.tight_layout()
    return fig

def create_agglutination_overview(
    measurements: List[Dict[str, Any]],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> Figure:
    """
    Create a comprehensive overview of agglutination analysis.
    
    Args:
        measurements: List of measurement dictionaries
        title: Optional title for the plot
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)
    
    # Define grid layout
    gs = plt.GridSpec(3, 2, figure=fig)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Spectral profile
    ax2 = fig.add_subplot(gs[0, 1])  # Key ratios
    ax3 = fig.add_subplot(gs[1, :])  # Time series
    ax4 = fig.add_subplot(gs[2, 0])  # Ratio histogram
    ax5 = fig.add_subplot(gs[2, 1])  # Agglutination score
    
    if not measurements:
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.text(0.5, 0.5, "No measurement data available", 
                    ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # 1. Spectral profile from the latest measurement
    latest = measurements[-1]
    raw_data = latest.get('raw', latest)
    
    # Filter to spectral channels (F1-F8)
    channels = [ch for ch in raw_data.keys() if ch.startswith('F') and ch[1:].isdigit()]
    channels.sort()  # Ensure channels are in order
    
    if channels:
        wavelengths = [CHANNEL_WAVELENGTHS[ch] for ch in channels]
        values = [raw_data[ch] for ch in channels]
        colors = [CHANNEL_COLORS[ch] for ch in channels]
        
        ax1.bar(wavelengths, values, width=20, color=colors, alpha=0.7)
        ax1.plot(wavelengths, values, 'o-', color='black', alpha=0.7, linewidth=1.5)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Signal Intensity')
        ax1.set_title('Latest Spectral Profile')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No spectral data available", 
                ha='center', va='center', transform=ax1.transAxes)
    
    # 2. Key ratios
    ratio_names = ["violet_red", "violet_green", "green_red"]
    has_ratios = False
    
    for m in measurements:
        if 'ratios' in m and any(r in m['ratios'] for r in ratio_names):
            has_ratios = True
            break
    
    if not has_ratios:
        # Calculate ratios if not available
        for m in measurements:
            if 'ratios' not in m:
                m['ratios'] = {}
            
            # Get raw data (either directly or from 'raw' key)
            raw_data = m.get('raw', m)
            
            # Calculate common ratios if the necessary channels exist
            if 'F1' in raw_data and 'F8' in raw_data:
                m['ratios']['violet_red'] = raw_data['F1'] / max(1, raw_data['F8'])
            
            if 'F1' in raw_data and 'F4' in raw_data:
                m['ratios']['violet_green'] = raw_data['F1'] / max(1, raw_data['F4'])
            
            if 'F4' in raw_data and 'F8' in raw_data:
                m['ratios']['green_red'] = raw_data['F4'] / max(1, raw_data['F8'])
    
    # Get average ratios
    avg_ratios = {r: 0 for r in ratio_names}
    count = 0
    
    for m in measurements:
        if 'ratios' in m:
            count += 1
            for r in ratio_names:
                if r in m['ratios']:
                    avg_ratios[r] += m['ratios'][r]
    
    if count > 0:
        for r in ratio_names:
            avg_ratios[r] /= count
        
        # Plot average ratios
        ratio_labels = [r.replace('_', '/') for r in ratio_names]
        ratio_values = [avg_ratios[r] for r in ratio_names]
        
        ax2.bar(ratio_labels, ratio_values, color=['purple', 'teal', 'orange'], alpha=0.7)
        ax2.set_ylabel('Ratio Value')
        ax2.set_title('Average Spectral Ratios')
        ax2.grid(True, alpha=0.3)
        
        # Add values above bars
        for i, v in enumerate(ratio_values):
            ax2.text(i, v + 0.05, f"{v:.2f}", ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, "No ratio data available", 
                ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Time series
    timestamps = []
    violet_red_values = []
    
    for m in measurements:
        # Extract timestamp
        if 'timestamp' in m:
            timestamps.append(m['timestamp'])
        elif 'elapsed_seconds' in m:
            timestamps.append(m['elapsed_seconds'])
        else:
            # Use index as timestamp if none available
            timestamps.append(len(timestamps))
        
        # Extract violet/red ratio
        if 'ratios' in m and 'violet_red' in m['ratios']:
            violet_red_values.append(m['ratios']['violet_red'])
        else:
            violet_red_values.append(0)
    
    # Convert timestamps to elapsed time if they're absolute
    if timestamps and timestamps[0] > 1000000000:  # Unix timestamp check
        start_time = timestamps[0]
        timestamps = [t - start_time for t in timestamps]
    
    if any(violet_red_values):
        ax3.plot(timestamps, violet_red_values, 'o-', 
                color='purple', alpha=0.8, linewidth=2)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Violet/Red Ratio')
        ax3.set_title('Agglutination Ratio Over Time')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No time series data available", 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Ratio histogram
    if any(violet_red_values):
        values = [v for v in violet_red_values if v > 0]  # Filter out zeros
        if values:
            ax4.hist(values, bins=15, color='purple', alpha=0.7)
            ax4.set_xlabel('Violet/Red Ratio')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Ratio Distribution')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No valid ratio values for histogram", 
                    ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, "No ratio data for histogram", 
                ha='center', va='center', transform=ax4.transAxes)
    
    # 5. Agglutination score (estimated from violet/red ratio)
    if any(violet_red_values):
        latest_ratio = violet_red_values[-1]
        
        # Simple agglutination classification
        if latest_ratio >= 2.0:
            score = 4.0
            state = "Complete"
        elif latest_ratio >= 1.5:
            score = 3.0
            state = "Strong"
        elif latest_ratio >= 1.2:
            score = 2.0
            state = "Moderate"
        elif latest_ratio >= 1.0:
            score = 1.0
            state = "Minimal"
        else:
            score = 0.0
            state = "None"
        
        # Create a gauge-like visualization
        max_score = 4.0
        theta = np.linspace(0, np.pi, 1000)
        radius = 0.8
        
        # Draw the gauge arc
        x_arc = radius * np.cos(theta) + 0.5
        y_arc = radius * np.sin(theta) + 0.1
        ax5.plot(x_arc, y_arc, 'k-', lw=3)
        
        # Draw tick marks
        for i in range(5):
            tick_theta = np.pi * i / 4
            x_tick_start = radius * np.cos(tick_theta) + 0.5
            y_tick_start = radius * np.sin(tick_theta) + 0.1
            x_tick_end = (radius - 0.1) * np.cos(tick_theta) + 0.5
            y_tick_end = (radius - 0.1) * np.sin(tick_theta) + 0.1
            ax5.plot([x_tick_start, x_tick_end], [y_tick_start, y_tick_end], 'k-', lw=2)
            
            # Add tick labels
            x_label = (radius + 0.1) * np.cos(tick_theta) + 0.5
            y_label = (radius + 0.1) * np.sin(tick_theta) + 0.1
            ax5.text(x_label, y_label, str(i), ha='center', va='center')
        
        # Draw the needle
        score_theta = np.pi * score / max_score
        x_needle = [0.5, radius * np.cos(score_theta) + 0.5]
        y_needle = [0.1, radius * np.sin(score_theta) + 0.1]
        ax5.plot(x_needle, y_needle, 'r-', lw=2)
        
        # Add text info
        ax5.text(0.5, 0.5, f"{state}\nScore: {score:.1f}\nRatio: {latest_ratio:.2f}", 
                 ha='center', va='center', fontsize=12)
        
        ax5.set_title('Agglutination Assessment')
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
    else:
        ax5.text(0.5, 0.5, "Insufficient data for agglutination assessment", 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Agglutination Assessment')
        ax5.axis('off')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95] if title else [0, 0, 1, 1])
    return fig