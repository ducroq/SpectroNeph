"""
Real-time visualization functionality for SpectroNeph.

This module provides tools for visualizing spectral data in real-time,
including live updating plots and dashboard components.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import matplotlib
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time
import threading
import queue
from collections import deque
from utils.logging import get_logger

# Set the backend to a non-interactive one if running headless
# matplotlib.use('Agg')

# Initialize module logger
logger = get_logger(__name__)

# Import local modules
from visualization.plots import (
    CHANNEL_WAVELENGTHS,
    CHANNEL_COLORS,
    create_spectral_profile,
    create_time_series
)

class RealTimeSpectralPlot:
    """
    Real-time updating spectral plot.
    
    This class handles the real-time visualization of spectral data
    from the AS7341 nephelometer, with automatic updates as new data arrives.
    """
    
    def __init__(self, 
                buffer_size: int = 100,
                update_interval: int = 100,  # ms
                figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize the real-time spectral plot.
        
        Args:
            buffer_size: Number of measurements to keep in buffer
            update_interval: Interval between updates in milliseconds
            figsize: Figure size as (width, height) in inches
        """
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.figsize = figsize
        
        # Data buffer for time series
        self.data_buffer = deque(maxlen=buffer_size)
        
        # Latest spectral data
        self.latest_spectral_data = {}
        
        # Setup plot
        self.fig = plt.figure(figsize=figsize)
        
        # Create grid for subplots
        gs = self.fig.add_gridspec(2, 2)
        
        # Create subplots
        self.ax_spectral = self.fig.add_subplot(gs[0, 0])  # Spectral profile
        self.ax_time_series = self.fig.add_subplot(gs[0, 1])  # Time series for key channels
        self.ax_ratios = self.fig.add_subplot(gs[1, 0])  # Spectral ratios
        self.ax_status = self.fig.add_subplot(gs[1, 1])  # Status and info
        
        # Initialize axes
        self._init_spectral_axis()
        self._init_time_series_axis()
        self._init_ratios_axis()
        self._init_status_axis()
        
        # Animation control
        self.animation = None
        self.running = False
        self.paused = False
        
        # Thread-safe queue for data updates
        self.data_queue = queue.Queue()
        self.lock = threading.Lock()

    def _init_spectral_axis(self):
        """Initialize the spectral profile axis."""
        self.ax_spectral.set_title('Spectral Profile')
        self.ax_spectral.set_xlabel('Wavelength (nm)')
        self.ax_spectral.set_ylabel('Signal Intensity')
        self.ax_spectral.grid(True, alpha=0.3)
        
        # Initialize empty bars for each channel
        self.channels = [f"F{i}" for i in range(1, 9)]
        wavelengths = [CHANNEL_WAVELENGTHS[ch] for ch in self.channels]
        self.spectral_bars = self.ax_spectral.bar(
            wavelengths, [0] * len(self.channels),
            width=20, color=[CHANNEL_COLORS[ch] for ch in self.channels], alpha=0.7
        )
        
        # Add connecting line
        self.spectral_line, = self.ax_spectral.plot(
            wavelengths, [0] * len(self.channels), 'o-', color='black', alpha=0.7, linewidth=1.5
        )
        
        # Set reasonable y-axis limit
        self.ax_spectral.set_ylim(0, 1000)  # Will be adjusted automatically
        
        # Set x-axis range to cover all wavelengths with padding
        self.ax_spectral.set_xlim(400, 700)
        
        # Add channel labels
        for i, ch in enumerate(self.channels):
            self.ax_spectral.text(
                wavelengths[i], 0, ch,
                ha='center', va='bottom', fontsize=9,
                color='black'
            )
        
        # Store the channel labels for updating
        self.spectral_labels = self.ax_spectral.texts[-len(self.channels):]

    def _init_time_series_axis(self):
        """Initialize the time series axis."""
        self.ax_time_series.set_title('Channel Intensity Over Time')
        self.ax_time_series.set_xlabel('Time (s)')
        self.ax_time_series.set_ylabel('Signal Intensity')
        self.ax_time_series.grid(True, alpha=0.3)
        
        # Initialize empty time series for key channels
        self.key_channels = ['F1', 'F4', 'F8']  # Violet, Green, Red
        self.time_data = []
        self.channel_data = {ch: [] for ch in self.key_channels}
        
        # Create line objects for each channel
        self.time_series_lines = {}
        for ch in self.key_channels:
            line, = self.ax_time_series.plot(
                [], [], 'o-', 
                label=f"{ch} ({CHANNEL_WAVELENGTHS[ch]}nm)",
                color=CHANNEL_COLORS[ch],
                alpha=0.8, linewidth=2
            )
            self.time_series_lines[ch] = line
        
        # Add legend
        self.ax_time_series.legend(loc='upper left')
        
        # Set reasonable y-axis limit (will be adjusted automatically)
        self.ax_time_series.set_ylim(0, 1000)
        
        # Set initial x-axis limits
        self.ax_time_series.set_xlim(0, 10)  # 10 seconds initially

    def _init_ratios_axis(self):
        """Initialize the spectral ratios axis."""
        self.ax_ratios.set_title('Spectral Ratios')
        self.ax_ratios.set_xlabel('Time (s)')
        self.ax_ratios.set_ylabel('Ratio Value')
        self.ax_ratios.grid(True, alpha=0.3)
        
        # Initialize empty ratio time series
        self.ratio_data = {
            'violet_red': [],
            'violet_green': [],
            'green_red': []
        }
        
        # Create line objects for each ratio
        self.ratio_lines = {}
        colors = ['purple', 'teal', 'orange']
        labels = ['Violet/Red', 'Violet/Green', 'Green/Red']
        
        for i, (ratio, label) in enumerate(zip(self.ratio_data.keys(), labels)):
            line, = self.ax_ratios.plot(
                [], [], 'o-', 
                label=label,
                color=colors[i],
                alpha=0.8, linewidth=2
            )
            self.ratio_lines[ratio] = line
        
        # Add legend
        self.ax_ratios.legend(loc='upper left')
        
        # Set reasonable y-axis limit (will be adjusted)
        self.ax_ratios.set_ylim(0, 5)
        
        # Set initial x-axis limits
        self.ax_ratios.set_xlim(0, 10)  # 10 seconds initially

    def _init_status_axis(self):
        """Initialize the status and info axis."""
        self.ax_status.set_title('Status and Information')
        self.ax_status.axis('off')  # No axes for status panel
        
        # Add status text elements
        self.status_text = self.ax_status.text(
            0.5, 0.9, "Awaiting data...",
            ha='center', va='top', fontsize=12
        )
        
        # Add statistics panel
        stats_text = (
            "Statistics:\n"
            "Max Value: --\n"
            "Min Value: --\n"
            "Violet/Red Ratio: --\n"
            "Measurements: 0"
        )
        self.stats_text = self.ax_status.text(
            0.05, 0.7, stats_text,
            ha='left', va='top', fontsize=10
        )
        
        # Add configuration info
        config_text = (
            "Configuration:\n"
            "Gain: --\n"
            "Integration Time: --\n"
            "LED Current: --"
        )
        self.config_text = self.ax_status.text(
            0.05, 0.3, config_text,
            ha='left', va='top', fontsize=10
        )

    def start(self, plt_show: bool = True):
        """
        Start the real-time visualization.
        
        Args:
            plt_show: Whether to call plt.show() to display the plot
        """
        if self.running:
            return
        
        self.running = True
        
        # Create animation
        self.animation = FuncAnimation(
            self.fig, self._update_plot,
            interval=self.update_interval,
            blit=False
        )
        
        # Show the plot
        if plt_show:
            plt.tight_layout()
            plt.show()

    def stop(self):
        """Stop the real-time visualization."""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()

    def pause(self):
        """Pause the real-time visualization."""
        if self.running and not self.paused:
            self.paused = True
            if self.animation:
                self.animation.event_source.stop()

    def resume(self):
        """Resume the real-time visualization."""
        if self.running and self.paused:
            self.paused = False
            if self.animation:
                self.animation.event_source.start()

    def _update_plot(self, frame):
        """
        Update the plot with new data.
        
        Args:
            frame: Animation frame (not used)
            
        Returns:
            List of updated artists
        """
        with self.lock:
            # Process any new data in the queue
            self._process_queue()
            
            # Update spectral profile
            self._update_spectral_plot()
            
            # Update time series
            self._update_time_series_plot()
            
            # Update ratios
            self._update_ratios_plot()
            
            # Update status
            self._update_status_plot()
        
        # Return a list of all updated artists
        artists = (
            list(self.spectral_bars) + 
            [self.spectral_line] + 
            list(self.spectral_labels) + 
            list(self.time_series_lines.values()) + 
            list(self.ratio_lines.values()) + 
            [self.status_text, self.stats_text, self.config_text]
        )
        
        return artists

    def _process_queue(self):
        """Process any new data in the queue."""
        while not self.data_queue.empty():
            try:
                # Get new data
                data = self.data_queue.get_nowait()
                
                # Process the data
                self._process_data(data)
                
                # Mark as done
                self.data_queue.task_done()
            except queue.Empty:
                break

    def _process_data(self, data):
        """
        Process new data and update internal buffers.
        
        Args:
            data: New measurement data
        """
        # Extract timestamp
        timestamp = data.get('timestamp', time.time())
        
        # Convert to elapsed time if this is the first measurement
        if not self.time_data:
            self.start_time = timestamp
            elapsed = 0
        else:
            elapsed = timestamp - self.start_time
        
        # Store the elapsed time
        self.time_data.append(elapsed)
        
        # Extract raw data
        raw_data = data.get('raw', data)
        
        # Update latest spectral data
        self.latest_spectral_data = raw_data.copy()
        
        # Update channel data
        for ch in self.key_channels:
            if ch in raw_data:
                self.channel_data[ch].append(raw_data[ch])
            else:
                self.channel_data[ch].append(0)
        
        # Calculate ratios if not present
        if 'ratios' not in data:
            ratios = {}
            if 'F1' in raw_data and 'F8' in raw_data:
                ratios['violet_red'] = raw_data['F1'] / max(1, raw_data['F8'])
            
            if 'F1' in raw_data and 'F4' in raw_data:
                ratios['violet_green'] = raw_data['F1'] / max(1, raw_data['F4'])
            
            if 'F4' in raw_data and 'F8' in raw_data:
                ratios['green_red'] = raw_data['F4'] / max(1, raw_data['F8'])
        else:
            ratios = data['ratios']
        
        # Update ratio data
        for ratio in self.ratio_data:
            if ratio in ratios:
                self.ratio_data[ratio].append(ratios[ratio])
            else:
                self.ratio_data[ratio].append(0)
        
        # Add to buffer
        self.data_buffer.append(data)

    def _update_spectral_plot(self):
        """Update the spectral profile plot."""
        # Get values for each channel
        values = []
        for i, ch in enumerate(self.channels):
            value = self.latest_spectral_data.get(ch, 0)
            values.append(value)
            
            # Update bar height
            self.spectral_bars[i].set_height(value)
            
            # Update label position
            self.spectral_labels[i].set_y(value + max(values) * 0.03)
        
        # Update line
        wavelengths = [CHANNEL_WAVELENGTHS[ch] for ch in self.channels]
        self.spectral_line.set_ydata(values)
        
        # Adjust y-axis limits if needed
        max_value = max(values) if values else 1000
        current_ylim = self.ax_spectral.get_ylim()
        if max_value > current_ylim[1] * 0.9 or max_value < current_ylim[1] * 0.5:
            self.ax_spectral.set_ylim(0, max_value * 1.1)

    def _update_time_series_plot(self):
        """Update the time series plot."""
        # Update each channel's line
        for ch in self.key_channels:
            if self.time_data and self.channel_data[ch]:
                self.time_series_lines[ch].set_data(self.time_data, self.channel_data[ch])
        
        # Adjust x-axis limits to show the most recent data
        if self.time_data:
            max_time = max(self.time_data)
            current_xlim = self.ax_time_series.get_xlim()
            
            # If we've gone beyond the current view or have a smaller window than needed
            if max_time > current_xlim[1] or current_xlim[1] > max_time * 2:
                # Show the most recent data with a window of appropriate size
                window_size = min(30, max(10, max_time * 0.5))  # Window of 10-30 seconds
                self.ax_time_series.set_xlim(max(0, max_time - window_size), max_time + 1)
            
            # Adjust y-axis limits if needed
            all_values = []
            for ch in self.key_channels:
                if self.channel_data[ch]:
                    all_values.extend(self.channel_data[ch])
            
            if all_values:
                max_value = max(all_values)
                current_ylim = self.ax_time_series.get_ylim()
                if max_value > current_ylim[1] * 0.9 or max_value < current_ylim[1] * 0.5:
                    self.ax_time_series.set_ylim(0, max_value * 1.1)

    def _update_ratios_plot(self):
        """Update the ratios plot."""
        # Update each ratio's line
        for ratio, line in self.ratio_lines.items():
            if self.time_data and self.ratio_data[ratio]:
                line.set_data(self.time_data, self.ratio_data[ratio])
        
        # Adjust x-axis limits to match the time series plot
        if self.time_data:
            self.ax_ratios.set_xlim(self.ax_time_series.get_xlim())
            
            # Adjust y-axis limits if needed
            all_values = []
            for ratio in self.ratio_data:
                if self.ratio_data[ratio]:
                    all_values.extend(self.ratio_data[ratio])
            
            if all_values:
                max_value = max(all_values)
                current_ylim = self.ax_ratios.get_ylim()
                if max_value > current_ylim[1] * 0.9 or max_value < current_ylim[1] * 0.5:
                    self.ax_ratios.set_ylim(0, max(5, max_value * 1.1))

    def _update_status_plot(self):
        """Update the status plot."""
        # Update status text
        if self.data_buffer:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            status_text = f"Last Update: {timestamp}\nStatus: Running"
            self.status_text.set_text(status_text)
            
            # Update statistics
            latest_data = self.data_buffer[-1]
            raw_data = latest_data.get('raw', latest_data)
            
            # Find max and min values across channels
            spectral_values = [raw_data.get(ch, 0) for ch in self.channels if ch in raw_data]
            max_value = max(spectral_values) if spectral_values else 0
            min_value = min([v for v in spectral_values if v > 0]) if spectral_values else 0
            
            # Get violet/red ratio
            violet_red = 0
            if "ratios" in latest_data and "violet_red" in latest_data["ratios"]:
                violet_red = latest_data["ratios"]["violet_red"]
            elif "F1" in raw_data and "F8" in raw_data and raw_data["F8"] > 0:
                violet_red = raw_data["F1"] / raw_data["F8"]
            
            # Update statistics text
            stats_text = (
                f"Statistics:\n"
                f"Max Value: {max_value}\n"
                f"Min Value: {min_value if min_value > 0 else 'N/A'}\n"
                f"Violet/Red Ratio: {violet_red:.2f}\n"
                f"Measurements: {len(self.data_buffer)}"
            )
            self.stats_text.set_text(stats_text)
            
            # Update configuration info if available
            config_text = "Configuration:\n"
            if "config" in latest_data:
                config = latest_data["config"]
                if "gain" in config:
                    config_text += f"Gain: {config['gain']}x\n"
                else:
                    config_text += "Gain: --\n"
                
                if "integration_time" in config:
                    config_text += f"Integration Time: {config['integration_time']}ms\n"
                else:
                    config_text += "Integration Time: --\n"
                
                if "led_current" in config:
                    config_text += f"LED Current: {config['led_current']}mA"
                else:
                    config_text += "LED Current: --"
            else:
                # Default text if no config available
                config_text += "Gain: --\nIntegration Time: --\nLED Current: --"
            
            self.config_text.set_text(config_text)

    def add_data(self, data: Dict[str, Any]):
        """
        Add new data to the visualization.
        
        Args:
            data: New measurement data
        """
        # Add to queue for thread-safe processing
        self.data_queue.put(data)

    def get_figure(self) -> Figure:
        """
        Get the matplotlib figure.
        
        Returns:
            The matplotlib figure object
        """
        return self.fig

    def clear(self):
        """Clear all data and reset the visualization."""
        with self.lock:
            # Clear data buffers
            self.data_buffer.clear()
            self.latest_spectral_data = {}
            self.time_data = []
            self.channel_data = {ch: [] for ch in self.key_channels}
            self.ratio_data = {ratio: [] for ratio in self.ratio_data}
            
            # Reset plots
            for i in range(len(self.spectral_bars)):
                self.spectral_bars[i].set_height(0)
            
            self.spectral_line.set_ydata([0] * len(self.channels))
            
            for ch, line in self.time_series_lines.items():
                line.set_data([], [])
            
            for ratio, line in self.ratio_lines.items():
                line.set_data([], [])
            
            # Reset status
            self.status_text.set_text("Awaiting data...")
            self.stats_text.set_text(
                "Statistics:\n"
                "Max Value: --\n"
                "Min Value: --\n"
                "Violet/Red Ratio: --\n"
                "Measurements: 0"
            )
            self.config_text.set_text(
                "Configuration:\n"
                "Gain: --\n"
                "Integration Time: --\n"
                "LED Current: --"
            )


class RealTimeDataMonitor:
    """
    Real-time data monitor for the nephelometer.
    
    This class provides a high-level interface for real-time data
    visualization and monitoring, with support for data recording
    and event detection.
    """
    
    def __init__(self, 
                 buffer_size: int = 1000,
                 update_interval: int = 100):
        """
        Initialize the real-time data monitor.
        
        Args:
            buffer_size: Size of the data buffer
            update_interval: Update interval in milliseconds
        """
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        
        # Data buffer
        self.data_buffer = deque(maxlen=buffer_size)
        
        # Create the plot
        self.plot = RealTimeSpectralPlot(
            buffer_size=buffer_size,
            update_interval=update_interval
        )
        
        # Recording state
        self.recording = False
        self.recording_data = []
        self.recording_start_time = 0
        
        # Event detection
        self.detection_enabled = False
        self.detection_threshold = 1.2  # Default violet/red ratio threshold
        self.detection_callback = None
        
        # Event state
        self.event_detected = False
        self.event_start_time = 0
        self.event_data = []
        
        # Logger
        self.logger = get_logger("real_time_monitor")

    def start(self, plt_show: bool = True):
        """
        Start the real-time monitor.
        
        Args:
            plt_show: Whether to call plt.show() to display the plot
        """
        self.plot.start(plt_show=plt_show)

    def stop(self):
        """Stop the real-time monitor."""
        self.plot.stop()
        self.stop_recording()

    def add_data(self, data: Dict[str, Any]):
        """
        Add new data to the monitor.
        
        Args:
            data: New measurement data
        """
        # Add to the plot
        self.plot.add_data(data)
        
        # Add to the buffer
        self.data_buffer.append(data)
        
        # Handle recording
        if self.recording:
            self.recording_data.append(data)
        
        # Handle event detection
        if self.detection_enabled:
            self._check_for_events(data)

    def start_recording(self):
        """Start recording data."""
        if self.recording:
            return
        
        self.recording = True
        self.recording_data = []
        self.recording_start_time = time.time()
        
        self.logger.info("Started recording data")

    def stop_recording(self) -> List[Dict[str, Any]]:
        """
        Stop recording data.
        
        Returns:
            List of recorded data
        """
        if not self.recording:
            return []
        
        self.recording = False
        recorded_data = self.recording_data.copy()
        self.recording_data = []
        
        self.logger.info("Stopped recording data. Recorded %d measurements.", len(recorded_data))
        
        return recorded_data

    def enable_event_detection(self, 
                             threshold: float = 1.2,
                             callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Enable event detection.
        
        Args:
            threshold: Detection threshold (violet/red ratio)
            callback: Function to call when an event is detected
        """
        self.detection_enabled = True
        self.detection_threshold = threshold
        self.detection_callback = callback
        
        self.logger.info("Enabled event detection with threshold %.2f", threshold)

    def disable_event_detection(self):
        """Disable event detection."""
        self.detection_enabled = False
        self.event_detected = False
        
        self.logger.info("Disabled event detection")

    def _check_for_events(self, data: Dict[str, Any]):
        """
        Check for events in the data.
        
        Args:
            data: Measurement data to check
        """
        # Check for agglutination event based on violet/red ratio
        violet_red = 0
        
        # Get the ratio either from the ratios dict or calculate it
        if "ratios" in data and "violet_red" in data["ratios"]:
            violet_red = data["ratios"]["violet_red"]
        elif "raw" in data and "F1" in data["raw"] and "F8" in data["raw"] and data["raw"]["F8"] > 0:
            violet_red = data["raw"]["F1"] / data["raw"]["F8"]
        elif "F1" in data and "F8" in data and data["F8"] > 0:
            violet_red = data["F1"] / data["F8"]
        
        # Check if we've detected an event
        if not self.event_detected and violet_red >= self.detection_threshold:
            # Event start
            self.event_detected = True
            self.event_start_time = time.time()
            self.event_data = [data]
            
            # Call callback if provided
            if self.detection_callback:
                self.detection_callback({
                    "type": "event_start",
                    "time": self.event_start_time,
                    "threshold": self.detection_threshold,
                    "value": violet_red,
                    "data": data
                })
            
            self.logger.info("Event detected! Violet/Red ratio: %.2f", violet_red)
        
        elif self.event_detected:
            # Ongoing event
            self.event_data.append(data)
            
            # Check if event has ended (ratio below threshold)
            if violet_red < self.detection_threshold:
                # Event end
                self.event_detected = False
                event_duration = time.time() - self.event_start_time
                
                # Call callback if provided
                if self.detection_callback:
                    self.detection_callback({
                        "type": "event_end",
                        "start_time": self.event_start_time,
                        "end_time": time.time(),
                        "duration": event_duration,
                        "threshold": self.detection_threshold,
                        "value": violet_red,
                        "data": self.event_data
                    })
                
                self.logger.info("Event ended. Duration: %.2f seconds", event_duration)
    
    def clear_data(self):
        """Clear all data buffers."""
        self.data_buffer.clear()
        self.recording_data = []
        self.event_data = []
        self.plot.clear()
        
    def save_data(self, filename: str) -> bool:
        """
        Save recorded data to a file.
        
        Args:
            filename: Path to save the data
            
        Returns:
            bool: True if save was successful
        """
        try:
            # Get data to save
            data_to_save = self.recording_data if self.recording else list(self.data_buffer)
            
            if not data_to_save:
                self.logger.warning("No data to save")
                return False
            
            # Create a data structure with metadata
            export_data = {
                "metadata": {
                    "timestamp": time.time(),
                    "count": len(data_to_save),
                    "recording": self.recording,
                    "detection_enabled": self.detection_enabled,
                    "detection_threshold": self.detection_threshold
                },
                "measurements": data_to_save
            }
            
            # Determine the file format from the extension
            if filename.endswith(".json"):
                from data.storage import data_storage
                data_storage.save_session(export_data, "json", filename)
            elif filename.endswith(".csv"):
                from data.export import data_exporter
                data_exporter.export_to_csv(export_data, filename)
            else:
                # Default to JSON
                from data.storage import data_storage
                if not filename.endswith(".json"):
                    filename += ".json"
                data_storage.save_session(export_data, "json", filename)
            
            self.logger.info("Saved %d measurements to %s", len(data_to_save), filename)
            return True
            
        except Exception as e:
            self.logger.error("Error saving data: %s", str(e))
            return False
    
    def load_data(self, filename: str) -> bool:
        """
        Load data from a file.
        
        Args:
            filename: Path to the data file
            
        Returns:
            bool: True if load was successful
        """
        try:
            # Load the data
            from data.storage import data_storage
            loaded_data = data_storage.load_session(filename)
            
            if not loaded_data or "measurements" not in loaded_data:
                self.logger.warning("No valid data found in %s", filename)
                return False
            
            # Clear existing data
            self.clear_data()
            
            # Add each measurement to the buffer
            measurements = loaded_data["measurements"]
            for measurement in measurements:
                self.add_data(measurement)
            
            self.logger.info("Loaded %d measurements from %s", len(measurements), filename)
            return True
            
        except Exception as e:
            self.logger.error("Error loading data: %s", str(e))
            return False
            
    def get_figure(self) -> plt.Figure:
        """
        Get the matplotlib figure.
        
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        return self.plot.get_figure()
    
    def set_plot_options(self, **kwargs):
        """
        Configure plot options.
        
        Args:
            **kwargs: Options to set on the plot
        """
        # This method can be expanded with specific plot configuration options
        # For now, it's a placeholder for future customization
        pass
    
    def export_plot(self, filename: str, dpi: int = 300) -> bool:
        """
        Export the current plot to an image file.
        
        Args:
            filename: Output filename
            dpi: Resolution in dots per inch
            
        Returns:
            bool: True if export was successful
        """
        try:
            fig = self.get_figure()
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            self.logger.info("Exported plot to %s", filename)
            return True
        except Exception as e:
            self.logger.error("Error exporting plot: %s", str(e))
            return False

    def show_statistics(self) -> Dict[str, Any]:
        """
        Calculate and return statistics on the current data.
        
        Returns:
            Dict: Statistics dictionary
        """
        if not self.data_buffer:
            return {}
        
        # Calculate key statistics from the data buffer
        stats = {
            "count": len(self.data_buffer),
            "channels": {},
            "ratios": {}
        }
        
        # Extract data for key channels
        channels = ["F1", "F4", "F8"]
        channel_data = {ch: [] for ch in channels}
        
        # Extract data for key ratios
        ratio_types = ["violet_red", "violet_green", "green_red"]
        ratio_data = {r: [] for r in ratio_types}
        
        # Process all measurements
        for measurement in self.data_buffer:
            # Extract channel data
            raw_data = measurement.get("raw", {})
            for ch in channels:
                if ch in raw_data:
                    channel_data[ch].append(raw_data[ch])
            
            # Extract ratio data
            ratios = measurement.get("ratios", {})
            for r in ratio_types:
                if r in ratios:
                    ratio_data[r].append(ratios[r])
        
        # Calculate statistics for each channel
        for ch in channels:
            if channel_data[ch]:
                values = channel_data[ch]
                stats["channels"][ch] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
        
        # Calculate statistics for each ratio
        for r in ratio_types:
            if ratio_data[r]:
                values = ratio_data[r]
                stats["ratios"][r] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
        
        return stats