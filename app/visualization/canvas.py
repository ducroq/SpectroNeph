from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import Dict, Optional
import time
from visualization.plots import (
    CHANNEL_WAVELENGTHS,
    CHANNEL_COLORS
)

class MplCanvas(FigureCanvas):
    """Base canvas for matplotlib plots."""
    
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.tight_layout()


class SpectralPlot(MplCanvas):
    """Canvas for spectral profile visualization."""
    
    def __init__(self, width=5, height=4, dpi=100):
        super().__init__(width, height, dpi)
        self.axes.set_title('Spectral Profile')
        self.axes.set_xlabel('Wavelength (nm)')
        self.axes.set_ylabel('Signal Intensity')
        self.axes.grid(True, alpha=0.3)
        
        # Initialize empty data
        self.wavelengths = [CHANNEL_WAVELENGTHS[ch] for ch in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]]
        self.values = [0] * len(self.wavelengths)
        self.colors = [CHANNEL_COLORS[ch] for ch in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]]
        
        # Create initial empty plot
        self.bars = self.axes.bar(self.wavelengths, self.values, width=20, color=self.colors, alpha=0.7)
        self.line, = self.axes.plot(self.wavelengths, self.values, 'o-', color='black', alpha=0.7, linewidth=1.5)
        
        # Set reasonable axis limits
        self.axes.set_xlim(400, 700)
        self.axes.set_ylim(0, 1000)
        
        # Add channel labels
        self.labels = []
        for i, wl in enumerate(self.wavelengths):
            label = self.axes.text(wl, 0, ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"][i],
                                  ha='center', va='bottom', fontsize=8)
            self.labels.append(label)
            
        self.fig.tight_layout()
    
    def update_plot(self, data: Dict[str, int]):
        """Update the plot with new spectral data."""
        channels = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
        
        # Update data values
        self.values = [data.get(ch, 0) for ch in channels]
        
        # Update bar heights
        for bar, val in zip(self.bars, self.values):
            bar.set_height(val)
            
        # Update line
        self.line.set_ydata(self.values)
        
        # Update labels
        max_value = max(self.values) if self.values else 1000
        for label, val in zip(self.labels, self.values):
            label.set_y(val + max_value * 0.03)
            
        # Adjust y-axis limits if needed
        if max_value > 0:
            current_ylim = self.axes.get_ylim()
            if max_value > current_ylim[1] * 0.9 or max_value < current_ylim[1] * 0.5:
                self.axes.set_ylim(0, max_value * 1.1)
        
        # Redraw the canvas
        self.fig.canvas.draw_idle()


class TimeSeriesPlot(MplCanvas):
    """Canvas for time series visualization."""
    
    def __init__(self, width=5, height=4, dpi=100):
        super().__init__(width, height, dpi)
        self.axes.set_title('Time Series')
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Signal Intensity')
        self.axes.grid(True, alpha=0.3)
        
        # Initialize data containers
        self.time_data = []
        self.channel_data = {ch: [] for ch in ["F1", "F4", "F8"]}  # Violet, Green, Red
        
        # Create empty lines
        self.lines = {}
        for ch, color in zip(["F1", "F4", "F8"], ['indigo', 'green', 'red']):
            line, = self.axes.plot([], [], 'o-', label=f"{ch} ({CHANNEL_WAVELENGTHS[ch]}nm)", 
                                  color=color, alpha=0.8, linewidth=1.5)
            self.lines[ch] = line
            
        # Add legend
        self.axes.legend(loc='upper left')
        
        # Set reasonable initial limits
        self.axes.set_xlim(0, 10)
        self.axes.set_ylim(0, 1000)
        
        self.fig.tight_layout()
        
    def update_plot(self, new_data: Dict[str, int], timestamp: Optional[float] = None):
        """Add a new data point and update the plot."""
        # Use current time if timestamp not provided
        if timestamp is None:
            if not self.time_data:
                self.start_time = time.time()
                timestamp = 0
            else:
                timestamp = time.time() - self.start_time
                
        # Add timestamp to time data
        self.time_data.append(timestamp)
        
        # Add channel values
        for ch in self.channel_data.keys():
            self.channel_data[ch].append(new_data.get(ch, 0))
            
        # Update the lines
        for ch, line in self.lines.items():
            line.set_data(self.time_data, self.channel_data[ch])
            
        # Adjust x-axis limits to show all data
        if self.time_data:
            x_min, x_max = min(self.time_data), max(self.time_data)
            x_range = max(10, x_max - x_min)  # At least 10 seconds
            self.axes.set_xlim(max(0, x_max - x_range), x_max + 1)
            
        # Adjust y-axis limits if needed
        all_values = []
        for values in self.channel_data.values():
            all_values.extend(values)
            
        if all_values:
            max_value = max(all_values)
            current_ylim = self.axes.get_ylim()
            if max_value > current_ylim[1] * 0.9 or max_value < current_ylim[1] * 0.5:
                self.axes.set_ylim(0, max_value * 1.1)
                
        # Redraw the canvas
        self.fig.canvas.draw_idle()
        
    def clear_data(self):
        """Clear all data from the plot."""
        self.time_data = []
        for ch in self.channel_data:
            self.channel_data[ch] = []
            
        # Reset the lines
        for ch, line in self.lines.items():
            line.set_data([], [])
            
        # Reset axes limits
        self.axes.set_xlim(0, 10)
        self.axes.set_ylim(0, 1000)
        
        # Redraw the canvas
        self.fig.canvas.draw_idle()


class RatioPlot(MplCanvas):
    """Canvas for spectral ratio visualization."""
    
    def __init__(self, width=5, height=4, dpi=100):
        super().__init__(width, height, dpi)
        self.axes.set_title('Spectral Ratios')
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Ratio Value')
        self.axes.grid(True, alpha=0.3)
        
        # Initialize data containers
        self.time_data = []
        self.ratio_data = {
            'violet_red': [],
            'violet_green': [],
            'green_red': []
        }
        
        # Create empty lines
        self.lines = {}
        colors = ['purple', 'teal', 'orange']
        labels = ['Violet/Red', 'Violet/Green', 'Green/Red']
        
        for (ratio, label, color) in zip(self.ratio_data.keys(), labels, colors):
            line, = self.axes.plot([], [], 'o-', label=label, color=color, alpha=0.8, linewidth=1.5)
            self.lines[ratio] = line
            
        # Add legend
        self.axes.legend(loc='upper left')
        
        # Set reasonable initial limits
        self.axes.set_xlim(0, 10)
        self.axes.set_ylim(0, 5)
        
        self.fig.tight_layout()
        
    def update_plot(self, ratios: Dict[str, float], timestamp: Optional[float] = None):
        """Add a new ratio data point and update the plot."""
        # Use current time if timestamp not provided
        if timestamp is None:
            if not self.time_data:
                self.start_time = time.time()
                timestamp = 0
            else:
                timestamp = time.time() - self.start_time
                
        # Add timestamp to time data
        self.time_data.append(timestamp)
        
        # Add ratio values
        for ratio in self.ratio_data.keys():
            self.ratio_data[ratio].append(ratios.get(ratio, 0))
            
        # Update the lines
        for ratio, line in self.lines.items():
            line.set_data(self.time_data, self.ratio_data[ratio])
            
        # Adjust x-axis limits to show all data
        if self.time_data:
            x_min, x_max = min(self.time_data), max(self.time_data)
            x_range = max(10, x_max - x_min)  # At least 10 seconds
            self.axes.set_xlim(max(0, x_max - x_range), x_max + 1)
            
        # Adjust y-axis limits if needed
        all_values = []
        for values in self.ratio_data.values():
            all_values.extend(values)
            
        if all_values:
            max_value = max(all_values)
            current_ylim = self.axes.get_ylim()
            if max_value > current_ylim[1] * 0.9 or max_value < current_ylim[1] * 0.5:
                new_max = max(5, max_value * 1.1)
                self.axes.set_ylim(0, new_max)
                
                # # Update threshold line and label position
                # self.threshold_line.set_ydata([self.threshold_value, self.threshold_value])
                # self.threshold_label.set_y(self.threshold_value + 0.05)
                
        # Redraw the canvas
        self.fig.canvas.draw_idle()
        
    def clear_data(self):
        """Clear all data from the plot."""
        self.time_data = []
        for ratio in self.ratio_data:
            self.ratio_data[ratio] = []
            
        # Reset the lines
        for ratio, line in self.lines.items():
            line.set_data([], [])
            
        # Reset axes limits
        self.axes.set_xlim(0, 10)
        self.axes.set_ylim(0, 5)
        
        # Redraw the canvas
        self.fig.canvas.draw_idle()
