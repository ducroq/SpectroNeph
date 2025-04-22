"""
Specialized spectral data visualization for SpectroNeph.

This module provides specialized visualization tools focused on spectral data analysis,
including spectral profiles, comparative spectral analysis, and agglutination visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time
from datetime import datetime
from utils.logging import get_logger

# Initialize module logger
logger = get_logger(__name__)

# Import from local modules
from visualization.plots import CHANNEL_WAVELENGTHS, CHANNEL_COLORS

def create_spectral_fingerprint(
    data_series: List[Dict[str, Any]], 
    normalize: bool = True,
    reference_channel: str = "Clear",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Figure:
    """
    Create a spectral fingerprint visualization that shows spectral profile evolution over time.
    
    Args:
        data_series: List of spectral measurements over time
        normalize: Whether to normalize values by reference channel
        reference_channel: Channel to use for normalization
        title: Optional plot title
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Figure: Matplotlib figure
    """
    if not data_series:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
    
    # Create figure with two subplots (heatmap and selected profiles)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    
    # Get spectral channels
    channels = [ch for ch in CHANNEL_WAVELENGTHS if ch.startswith('F')]
    channels.sort()  # Sort channels (F1-F8)
    
    # Prepare data for heatmap
    wavelengths = [CHANNEL_WAVELENGTHS[ch] for ch in channels]
    times = []
    spectrum_matrix = []