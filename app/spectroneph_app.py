#!/usr/bin/env python3
"""
SpectroNeph - AS7341-based Nephelometer Application

A PyQt-based GUI application for controlling and visualizing data from 
the AS7341 spectral sensor nephelometer.
"""

import sys
import os
import time
import threading
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QFormLayout, QLineEdit, QFileDialog, QMessageBox, QSplitter,
    QProgressBar, QStatusBar, QAction, QToolBar, QDockWidget, QFrame, QRadioButton
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSettings, QThread
from PyQt5.QtGui import QIcon, QFont, QPixmap, QColor

# Import matplotlib for plotting
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Add project root to path to allow importing project modules
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project modules
from core.device import DeviceManager
from hardware.nephelometer import Nephelometer, MeasurementMode, AgglutinationState
from utils.logging import setup_logging, get_logger
from utils.helpers import detect_serial_port
from data.acquisition import acquisition_manager, AcquisitionSession
from data.processing import signal_processor
from data.storage import data_storage
from data.export import data_exporter
from dataAnalysisWidget import DataAnalysisWidget
from visualization.canvas import SpectralPlot, TimeSeriesPlot, RatioPlot


# Version information
APP_NAME = "SpectroNeph"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Your Name"
APP_DESCRIPTION = "AS7341 Nephelometer Control and Analysis Software"

# Initialize logging
setup_logging()
logger = get_logger(__name__)


class DeviceConnectionWidget(QWidget):
    """Widget for device connection and configuration."""
    
    device_connected = pyqtSignal(bool)
    connection_error = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.device_manager = DeviceManager()
        self.nephelometer = None
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface."""
        # Create layout
        layout = QVBoxLayout()
        
        # Port selection
        # port_layout = QHBoxLayout()
        # self.port_combo = QComboBox()
        self.port = None
        self.refresh_ports_button = QPushButton("Refresh")
        self.refresh_ports_button.clicked.connect(self.refresh_ports)
        # port_layout.addWidget(QLabel("Serial Port:"))
        # port_layout.addWidget(self.port_combo, 1)
        # port_layout.addWidget(self.refresh_ports_button)
        # layout.addLayout(port_layout)
        
        # Connect button
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_connection)
        layout.addWidget(self.connect_button)
        
        # Status
        self.status_label = QLabel("Not connected")
        layout.addWidget(self.status_label)
        
        # Sensor configuration group
        config_group = QGroupBox("Sensor Configuration")
        config_layout = QFormLayout()
        
        # Gain setting
        self.gain_combo = QComboBox()
        gain_values = [
            "0.5x (0)", "1x (1)", "2x (2)", "4x (3)", "8x (4)", 
            "16x (5)", "32x (6)", "64x (7)", "128x (8)", 
            "256x (9)", "512x (10)"
        ]
        self.gain_combo.addItems(gain_values)
        self.gain_combo.setCurrentIndex(5)  # Default to 16x gain
        config_layout.addRow("Gain:", self.gain_combo)
        
        # Integration time
        self.integration_spin = QSpinBox()
        self.integration_spin.setRange(1, 1000)
        self.integration_spin.setValue(100)
        self.integration_spin.setSuffix(" ms")
        config_layout.addRow("Integration Time:", self.integration_spin)
        
        # LED current
        self.led_current_spin = QSpinBox()
        self.led_current_spin.setRange(0, 20)
        self.led_current_spin.setValue(10)
        self.led_current_spin.setSuffix(" mA")
        config_layout.addRow("LED Current:", self.led_current_spin)
        
        # LED control
        self.led_check = QCheckBox("LED Enabled")
        config_layout.addRow("", self.led_check)
        
        # Apply button
        self.apply_button = QPushButton("Apply Configuration")
        self.apply_button.clicked.connect(self.apply_configuration)
        config_layout.addRow("", self.apply_button)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Set initial state
        self.setEnabled_config(False)
        
        # Add stretch to push everything to the top
        layout.addStretch(1)
        
        self.setLayout(layout)
        
        # Refresh ports initially
        self.refresh_ports()
        
    def refresh_ports(self):
        """Refresh the available serial ports."""
        # self.port_combo.clear()
        
        # Try to detect available ports
        # try:
            # available_ports = self.device_manager.list_available_ports()
            
            # if not available_ports:
            #     logger.warning("No serial ports found")
            #     self.port_combo.addItem("No ports found")
            #     return
                
            # for port in available_ports:
            #     self.port_combo.addItem(port)
                
            # Try to auto-detect device port
        self.port = detect_serial_port()
        if self.port:
            # and device_port in available_ports:
            # index = available_ports.index(device_port)
            # self.port_combo.setCurrentIndex(index)
            logger.info(f"Auto-detected device on port {self.port}")
        else:
            logger.warning("No valid device port detected")
            # self.port_combo.addItem("No ports found")
            return
                
        # except Exception as e:
        #     logger.error(f"Error refreshing ports: {str(e)}")
        #     self.connection_error.emit(f"Error refreshing ports: {str(e)}")
        #     self.port_combo.addItem("Error listing ports")
    
    def toggle_connection(self):
        """Connect to or disconnect from the device."""
        if self.device_manager.is_connected():
            # Disconnect
            try:
                self.device_manager.disconnect()
                self.status_label.setText("Not connected")
                self.connect_button.setText("Connect")
                self.setEnabled_config(False)
                self.device_connected.emit(False)
                logger.info("Disconnected from device")
            except Exception as e:
                logger.error(f"Error disconnecting: {str(e)}")
                self.connection_error.emit(f"Error disconnecting: {str(e)}")
        else:
            # Connect
            try:
                if self.device_manager.connect(port=self.port):
                    # Create nephelometer
                    self.nephelometer = Nephelometer(self.device_manager)
                    if self.nephelometer.initialize():
                        self.status_label.setText(f"Connected to {self.port}")
                        self.connect_button.setText("Disconnect")
                        self.setEnabled_config(True)
                        self.device_connected.emit(True)
                        
                        # Apply initial configuration
                        self.apply_configuration()
                        
                        logger.info(f"Connected to device on {self.port}")
                    else:
                        self.device_manager.disconnect()
                        self.connection_error.emit("Failed to initialize nephelometer")
                        logger.error("Failed to initialize nephelometer")
                else:
                    self.connection_error.emit(f"Failed to connect to {self.port}")
                    logger.error(f"Failed to connect to {self.port}")
            except Exception as e:
                logger.error(f"Error connecting: {str(e)}")
                self.connection_error.emit(f"Error connecting: {str(e)}")
    
    def apply_configuration(self):
        """Apply the current configuration to the device."""
        if not self.device_manager.is_connected() or not self.nephelometer:
            self.connection_error.emit("Device not connected")
            return
            
        try:
            # Get gain value from the combo box (parse the number in parentheses)
            gain_text = self.gain_combo.currentText()
            gain_value = int(gain_text.split('(')[1].split(')')[0])
            
            # Get other values
            integration_time = self.integration_spin.value()
            led_current = self.led_current_spin.value()
            led_enabled = self.led_check.isChecked()
            
            # Apply configuration
            config = {
                "gain": gain_value,
                "integration_time": integration_time,
                "led_current": led_current
            }
            
            if not self.nephelometer.configure(config):
                self.connection_error.emit("Failed to configure nephelometer")
                return
                
            # Set LED state
            if not self.nephelometer.set_led(led_enabled, led_current):
                self.connection_error.emit("Failed to control LED")
                return
                
            logger.info(f"Applied configuration: gain={gain_value}, " 
                       f"integration_time={integration_time}ms, "
                       f"led_current={led_current}mA, led_enabled={led_enabled}")
                
        except Exception as e:
            logger.error(f"Error applying configuration: {str(e)}")
            self.connection_error.emit(f"Error applying configuration: {str(e)}")
    
    def setEnabled_config(self, enabled: bool):
        """Enable or disable configuration controls."""
        self.gain_combo.setEnabled(enabled)
        self.integration_spin.setEnabled(enabled)
        self.led_current_spin.setEnabled(enabled)
        self.led_check.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)
    
    def get_nephelometer(self) -> Optional[Nephelometer]:
        """Get the nephelometer instance."""
        return self.nephelometer


class MeasurementControlWidget(QWidget):
    """Widget for measurement control."""
    
    measurement_started = pyqtSignal()
    measurement_stopped = pyqtSignal()
    measurement_error = pyqtSignal(str)
    data_acquired = pyqtSignal(dict)
    session_completed = pyqtSignal(list)  # Signal emitted when a session is completed with all measurements
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.nephelometer = None
        self.measurement_thread = None
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface."""
        # Create layout
        layout = QVBoxLayout()
        
        # Measurement type selection
        self.measurement_group = QGroupBox("Measurement Type")
        measurement_layout = QVBoxLayout()
        
        self.single_radio = QRadioButton("Single Measurement")
        self.continuous_radio = QRadioButton("Continuous Measurement")
        self.kinetic_radio = QRadioButton("Kinetic Measurement")
        
        self.single_radio.setChecked(True)
        measurement_layout.addWidget(self.single_radio)
        measurement_layout.addWidget(self.continuous_radio)
        measurement_layout.addWidget(self.kinetic_radio)
        
        self.measurement_group.setLayout(measurement_layout)
        layout.addWidget(self.measurement_group)
        
        # Measurement parameters
        self.params_group = QGroupBox("Measurement Parameters")
        params_layout = QFormLayout()
        
        # Interval for continuous measurement
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 60.0)
        self.interval_spin.setValue(1.0)
        self.interval_spin.setSuffix(" s")
        self.interval_spin.setDecimals(1)
        params_layout.addRow("Interval:", self.interval_spin)
        
        # Duration for kinetic measurement
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 3600.0)
        self.duration_spin.setValue(30.0)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setDecimals(1)
        params_layout.addRow("Duration:", self.duration_spin)
        
        # Background subtraction
        self.background_check = QCheckBox("Subtract Background")
        self.background_check.setChecked(True)
        params_layout.addRow("", self.background_check)
        
        self.params_group.setLayout(params_layout)
        layout.addWidget(self.params_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Measurement")
        self.start_button.clicked.connect(self.start_measurement)
        
        self.stop_button = QPushButton("Stop Measurement")
        self.stop_button.clicked.connect(self.stop_measurement)
        self.stop_button.setEnabled(False)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
        # Progress bar (for kinetic measurement)
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar, 1)
        
        layout.addLayout(progress_layout)
        
        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Add stretch to push everything to the top
        layout.addStretch(1)
        
        self.setLayout(layout)
        
        # Connect radio button changes
        self.single_radio.toggled.connect(self.update_controls)
        self.continuous_radio.toggled.connect(self.update_controls)
        self.kinetic_radio.toggled.connect(self.update_controls)
        
        # Update controls initial state
        self.update_controls()
        
        # Timer for kinetic progress
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.start_time = 0
        
    def update_controls(self):
        """Update controls based on selected measurement type."""
        # Enable/disable parameters based on measurement type
        if self.single_radio.isChecked():
            self.interval_spin.setEnabled(False)
            self.duration_spin.setEnabled(False)
        elif self.continuous_radio.isChecked():
            self.interval_spin.setEnabled(True)
            self.duration_spin.setEnabled(False)
        elif self.kinetic_radio.isChecked():
            self.interval_spin.setEnabled(True)
            self.duration_spin.setEnabled(True)
    
    def set_nephelometer(self, nephelometer: Optional[Nephelometer]):
        """Set the nephelometer instance."""
        self.nephelometer = nephelometer
        self.start_button.setEnabled(nephelometer is not None)
        
    def start_measurement(self):
        """Start a measurement based on the selected type."""
        if not self.nephelometer:
            self.measurement_error.emit("Nephelometer not connected")
            return
            
        try:
            # Update UI state
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.measurement_group.setEnabled(False)
            self.params_group.setEnabled(False)
            
            # Get parameters
            subtract_background = self.background_check.isChecked()
            
            if self.single_radio.isChecked():
                # Single measurement
                self.status_label.setText("Taking single measurement...")
                
                # Take measurement directly
                measurement = self.nephelometer.take_single_measurement(subtract_background=subtract_background)
                
                # Process the measurement to add ratios
                processed = signal_processor.calculate_ratios(measurement)
                
                # Emit the data
                self.data_acquired.emit(processed)
                
                # Update UI
                self.status_label.setText("Measurement complete")
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.measurement_group.setEnabled(True)
                self.params_group.setEnabled(True)
                
                # Emit signal
                self.measurement_started.emit()
                self.measurement_stopped.emit()
                
            else:
                # Continuous or kinetic measurement
                interval = self.interval_spin.value()
                
                if self.kinetic_radio.isChecked():
                    # Kinetic measurement
                    duration = self.duration_spin.value()
                    self.status_label.setText(f"Running kinetic measurement for {duration}s...")
                    
                    # Start progress tracking
                    self.progress_bar.setValue(0)
                    self.start_time = time.time()
                    self.progress_timer.start(100)  # Update every 100ms
                    
                    # Create and start measurement thread
                    self.measurement_thread = ContinuousMeasurementThread(
                        self.nephelometer, interval, duration, subtract_background
                    )
                else:
                    # Continuous measurement
                    self.status_label.setText(f"Running continuous measurement...")
                    
                    # Create and start measurement thread
                    self.measurement_thread = ContinuousMeasurementThread(
                        self.nephelometer, interval, subtract_background=subtract_background
                    )
                
                # Connect signals
                self.measurement_thread.measurement_acquired.connect(self.handle_measurement)
                self.measurement_thread.measurement_error.connect(self.handle_error)
                self.measurement_thread.measurement_completed.connect(self.measurement_completed)
                
                # Start the thread
                self.measurement_thread.start()
                
                # Emit signal
                self.measurement_started.emit()
                
        except Exception as e:
            logger.error(f"Error starting measurement: {str(e)}")
            self.measurement_error.emit(f"Error starting measurement: {str(e)}")
            self.reset_ui_state()
    
    def stop_measurement(self):
        """Stop the ongoing measurement."""
        try:
            if self.measurement_thread and self.measurement_thread.isRunning():
                self.status_label.setText("Stopping measurement...")
                self.measurement_thread.stop()
            else:
                self.reset_ui_state()
                
        except Exception as e:
            logger.error(f"Error stopping measurement: {str(e)}")
            self.measurement_error.emit(f"Error stopping measurement: {str(e)}")
            self.reset_ui_state()
    
    def handle_measurement(self, data: Dict[str, Any]):
        """Handle a new measurement."""
        try:
            # Process the measurement to add ratios if not already present
            if 'ratios' not in data:
                processed = signal_processor.calculate_ratios(data)
            else:
                processed = data
                
            # Emit the data
            self.data_acquired.emit(processed)
            
        except Exception as e:
            logger.error(f"Error processing measurement: {str(e)}")
            self.measurement_error.emit(f"Error processing measurement: {str(e)}")
    
    def handle_error(self, error_message: str):
        """Handle an error from the measurement thread."""
        self.measurement_error.emit(error_message)
        self.reset_ui_state()
    
    def measurement_completed(self):
        """Handle measurement completion."""
        # Stop progress timer
        self.progress_timer.stop()
        
        # Get all measurements if available
        if self.measurement_thread and hasattr(self.measurement_thread, 'get_measurements'):
            measurements = self.measurement_thread.get_measurements()
            if measurements:
                # Emit signal with all measurements
                self.session_completed.emit(measurements)
                self.status_label.setText(f"Measurement complete - {len(measurements)} samples acquired")
            else:
                self.status_label.setText("Measurement complete")
        else:
            self.status_label.setText("Measurement complete")
        
        # Reset UI
        self.reset_ui_state()
        
        # Emit signal
        self.measurement_stopped.emit()
    
    def reset_ui_state(self):
        """Reset the UI state after measurement."""
        self.start_button.setEnabled(self.nephelometer is not None)
        self.stop_button.setEnabled(False)
        self.measurement_group.setEnabled(True)
        self.params_group.setEnabled(True)
        self.progress_timer.stop()
        self.update_controls()
    
    def update_progress(self):
        """Update the progress bar for kinetic measurement."""
        if self.kinetic_radio.isChecked() and self.start_time > 0:
            duration = self.duration_spin.value()
            elapsed = time.time() - self.start_time
            progress = min(100, int(elapsed / duration * 100))
            self.progress_bar.setValue(progress)
            
            # Update status with remaining time
            remaining = max(0, duration - elapsed)
            self.status_label.setText(f"Running kinetic measurement... {remaining:.1f}s remaining")

class ContinuousMeasurementThread(QThread):
    """Thread for continuous measurement acquisition."""
    measurement_acquired = pyqtSignal(dict)
    measurement_error = pyqtSignal(str)
    measurement_completed = pyqtSignal()
    
    def __init__(self, nephelometer: Nephelometer, interval: float, duration: Optional[float] = None, 
                subtract_background: bool = True):
        super().__init__()
        self.nephelometer = nephelometer
        self.interval = interval
        self.duration = duration
        self.subtract_background = subtract_background
        self.stop_flag = False
        self.measurements = []  # Store all measurements
        
    def run(self):
        """Run the measurement thread."""
        start_time = time.time()
        
        try:
            # Define callback for measurements
            def measurement_callback(data):
                # Store the measurement
                self.measurements.append(data.copy())
                
                # Emit the measurement
                self.measurement_acquired.emit(data.copy())
            
            # Start measurement based on mode
            if self.duration is not None:
                # Kinetic measurement
                self.nephelometer.start_kinetic_measurement(
                    duration_seconds=self.duration,
                    samples_per_second=1.0/self.interval,
                    callback=measurement_callback,
                    subtract_background=self.subtract_background
                )
                
                # Wait for measurement to complete
                while time.time() - start_time < self.duration and not self.stop_flag:
                    time.sleep(0.1)
            else:
                # Continuous measurement
                self.nephelometer.start_continuous_measurement(
                    interval_seconds=self.interval,
                    callback=measurement_callback,
                    subtract_background=self.subtract_background
                )
                
                # Run until stopped
                while not self.stop_flag:
                    time.sleep(0.1)
            
        except Exception as e:
            self.measurement_error.emit(str(e))
        finally:
            # Stop measurement
            try:
                self.nephelometer.stop_measurement()
            except Exception as e:
                self.measurement_error.emit(f"Error stopping measurement: {str(e)}")
                
            self.measurement_completed.emit()
            
    def stop(self):
        """Stop the measurement thread."""
        self.stop_flag = True
        self.wait()  # Wait for thread to finish
        
    def get_measurements(self) -> List[Dict[str, Any]]:
        """Get all measurements acquired during the run."""
        return self.measurements


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")

        # Current session data
        self.current_session_data = []
        
        # Create UI
        self.initUI()
        
        # Connect signals
        self.connectSignals()
        
        # Initialize status
        self.statusBar().showMessage("Ready")
        self.showMaximized()
    
    def initUI(self):
        """Initialize the user interface."""
        # Create main central widget
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Create left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Device connection widget
        self.device_widget = DeviceConnectionWidget()
        left_layout.addWidget(self.device_widget)
        
        # Measurement control widget
        self.measurement_widget = MeasurementControlWidget()
        left_layout.addWidget(self.measurement_widget)
        
        # Data analysis widget
        self.analysis_widget = DataAnalysisWidget()
        left_layout.addWidget(self.analysis_widget)
        
        left_panel.setLayout(left_layout)
        
        # Create right panel for visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Spectral profile plot
        self.spectral_plot = SpectralPlot(width=5, height=3)
        spectral_toolbar = NavigationToolbar(self.spectral_plot, self)
        
        spectral_container = QWidget()
        spectral_layout = QVBoxLayout()
        spectral_layout.addWidget(self.spectral_plot)
        spectral_layout.addWidget(spectral_toolbar)
        spectral_container.setLayout(spectral_layout)
        
        # Time series plot
        self.time_series_plot = TimeSeriesPlot(width=5, height=3)
        time_series_toolbar = NavigationToolbar(self.time_series_plot, self)
        
        time_series_container = QWidget()
        time_series_layout = QVBoxLayout()
        time_series_layout.addWidget(self.time_series_plot)
        time_series_layout.addWidget(time_series_toolbar)
        time_series_container.setLayout(time_series_layout)
        
        # Ratio plot
        self.ratio_plot = RatioPlot(width=5, height=3)
        ratio_toolbar = NavigationToolbar(self.ratio_plot, self)
        
        ratio_container = QWidget()
        ratio_layout = QVBoxLayout()
        ratio_layout.addWidget(self.ratio_plot)
        ratio_layout.addWidget(ratio_toolbar)
        ratio_container.setLayout(ratio_layout)
        
        # Add all plots to right panel
        right_layout.addWidget(spectral_container)
        right_layout.addWidget(time_series_container)
        right_layout.addWidget(ratio_container)
        
        right_panel.setLayout(right_layout)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 1)  # 1/3 of width
        main_layout.addWidget(right_panel, 2)  # 2/3 of width
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Create menu bar
        self.createMenuBar()
        
        # Create status bar
        self.statusBar()
    
    def createMenuBar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        save_action = QAction("&Save Session", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_session)
        
        load_action = QAction("&Load Session", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_session)
        
        export_menu = file_menu.addMenu("&Export")
        
        export_csv_action = QAction("Export to &CSV", self)
        export_csv_action.triggered.connect(self.analysis_widget.export_to_csv)
        
        export_excel_action = QAction("Export to &Excel", self)
        export_excel_action.triggered.connect(self.analysis_widget.export_to_excel)
        
        export_report_action = QAction("Generate &Report", self)
        export_report_action.triggered.connect(self.analysis_widget.generate_report)
        
        export_menu.addAction(export_csv_action)
        export_menu.addAction(export_excel_action)
        export_menu.addAction(export_report_action)
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        file_menu.addAction(save_action)
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        file_menu.addMenu(export_menu)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        clear_action = QAction("&Clear Data", self)
        clear_action.triggered.connect(self.clear_data)
        
        settings_action = QAction("&Settings", self)
        settings_action.triggered.connect(self.show_settings)
        
        edit_menu.addAction(clear_action)
        edit_menu.addSeparator()
        edit_menu.addAction(settings_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        dark_mode_action = QAction("&Dark Mode", self)
        dark_mode_action.setCheckable(True)
        dark_mode_action.triggered.connect(self.toggle_dark_mode)
        
        view_menu.addAction(dark_mode_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        
        help_menu.addAction(about_action)
    
    def connectSignals(self):
        """Connect signals between widgets."""
        # Connect device widget signals
        self.device_widget.device_connected.connect(self.on_device_connection_changed)
        self.device_widget.connection_error.connect(self.show_error)
        
        # Connect measurement widget signals
        self.measurement_widget.measurement_started.connect(self.on_measurement_started)
        self.measurement_widget.measurement_stopped.connect(self.on_measurement_stopped)
        self.measurement_widget.measurement_error.connect(self.show_error)
        self.measurement_widget.data_acquired.connect(self.on_data_acquired)
        self.measurement_widget.session_completed.connect(self.on_session_completed)
        
        # Connect analysis widget signals
        self.analysis_widget.export_requested.connect(self.handle_export)
        
        # # Connect ratio plot signals
        # self.ratio_plot.agglutination_detected.connect(self.on_agglutination_detected)
    
    def on_device_connection_changed(self, connected):
        """Handle device connection status change."""
        if connected:
            self.statusBar().showMessage("Connected to device")
            # Pass nephelometer to measurement widget
            self.measurement_widget.set_nephelometer(self.device_widget.get_nephelometer())
        else:
            self.statusBar().showMessage("Device disconnected")
            self.measurement_widget.set_nephelometer(None)
    
    def on_measurement_started(self):
        """Handle measurement start."""
        self.statusBar().showMessage("Measurement in progress...")
        
        # Clear plots if it's a new measurement session
        self.clear_plots_only()
        
        # Clear current session data
        self.current_session_data = []
    
    def on_measurement_stopped(self):
        """Handle measurement stop."""
        self.statusBar().showMessage("Measurement stopped")
    
    def on_data_acquired(self, data):
        """Handle new data acquisition."""
        # Update plots with new data
        if "raw" in data:
            # Update spectral plot
            self.spectral_plot.update_plot(data["raw"])
            
            # Update time series plot
            self.time_series_plot.update_plot(data["raw"], data.get("timestamp"))
        
        # Update ratio plot if ratios are available
        if "ratios" in data:
            self.ratio_plot.update_plot(data["ratios"], data.get("timestamp"))
        
        # Add to current session data
        self.current_session_data.append(data)
        
        # Update analysis widget
        self.analysis_widget.set_current_data(self.current_session_data)
        
        # Update status bar
        self.statusBar().showMessage(f"Data acquired - {len(self.current_session_data)} samples")
    
    def on_session_completed(self, measurements):
        """Handle completion of a measurement session."""
        if not measurements:
            return
            
        # Store all measurements
        self.current_session_data = measurements
        
        # Update analysis widget
        self.analysis_widget.set_current_data(self.current_session_data)
        
        # Update status bar
        self.statusBar().showMessage(f"Session completed - {len(measurements)} samples acquired")
    
    # def on_agglutination_detected(self, ratio_value):
    #     """Handle agglutination detection."""
    #     self.statusBar().showMessage(f"AGGLUTINATION DETECTED! Ratio: {ratio_value:.2f}")
        
    #     # Show a message box
    #     QMessageBox.information(
    #         self, 
    #         "Agglutination Detected", 
    #         f"Agglutination has been detected with a violet/red ratio of {ratio_value:.2f}.\n\n"
    #         f"This exceeds the threshold of {self.analysis_widget.threshold_spin.value():.1f}."
    #     )
    
    def clear_data(self):
        """Clear all data and plots."""
        # Clear plots
        self.clear_plots_only()
        
        # Clear stored data
        self.current_session_data = []
        
        # Update analysis widget
        self.analysis_widget.set_current_data([])
        
        # Update status bar
        self.statusBar().showMessage("Data cleared")
    
    def clear_plots_only(self):
        """Clear only the plots, not the stored data."""
        self.time_series_plot.clear_data()
        self.ratio_plot.clear_data()
        
        # Reset spectral plot with zeros
        empty_data = {ch: 0 for ch in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]}
        self.spectral_plot.update_plot(empty_data)
    
    def save_session(self):
        """Save the current session to a file."""
        if not self.current_session_data:
            self.show_error("No data to save")
            return
            
        # Ask for file name
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if not filename:
            return
            
        # Ensure .json extension
        if not filename.lower().endswith('.json'):
            filename += '.json'
            
        try:
            # Create session data structure
            session_data = {
                "metadata": {
                    "timestamp": time.time(),
                    "count": len(self.current_session_data),
                    "app_version": APP_VERSION
                },
                "measurements": self.current_session_data
            }
            
            # Save to file (use Path for cross-platform compatibility)
            path = Path(filename)
            json_path = data_storage.save_session(session_data, "json", path.name)
            
            self.statusBar().showMessage(f"Session saved to {json_path}")
            
        except Exception as e:
            self.show_error(f"Error saving session: {str(e)}")
    
    def load_session(self):
        """Load a session from a file."""
        # Ask for file name
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if not filename:
            return
            
        try:
            # Load from file
            session_data = data_storage.load_session(filename)
            
            if "measurements" not in session_data or not session_data["measurements"]:
                self.show_error("No measurements found in file")
                return
                
            # Clear current data and plots
            self.clear_data()
            
            # Load all measurements
            measurements = session_data["measurements"]
            self.current_session_data = measurements
            
            # Update analysis widget
            self.analysis_widget.set_current_data(measurements)
            
            # Update plots with the most recent measurement
            latest = measurements[-1]
            if "raw" in latest:
                self.spectral_plot.update_plot(latest["raw"])
            
            # Replay the time series and ratio data
            for m in measurements:
                if "raw" in m:
                    self.time_series_plot.update_plot(m["raw"], m.get("timestamp"))
                
                if "ratios" in m:
                    self.ratio_plot.update_plot(m["ratios"], m.get("timestamp"))
            
            # Update status bar
            self.statusBar().showMessage(f"Loaded {len(measurements)} measurements from {filename}")
            
        except Exception as e:
            self.show_error(f"Error loading session: {str(e)}")
    
    def handle_export(self, format_type, data):
        """Handle export requests from analysis widget."""
        if not data:
            self.show_error("No data to export")
            return
            
        # Ask for file name based on format
        file_filter = ""
        extension = ""
        if format_type == "csv":
            file_filter = "CSV Files (*.csv);;All Files (*)"
            extension = ".csv"
        elif format_type == "excel":
            file_filter = "Excel Files (*.xlsx);;All Files (*)"
            extension = ".xlsx"
        elif format_type == "report":
            file_filter = "HTML Files (*.html);;All Files (*)"
            extension = ".html"
        else:
            self.show_error(f"Unsupported export format: {format_type}")
            return
            
        # Ask for file name
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Export to {format_type.upper()}", "", file_filter
        )
        
        if not filename:
            return
            
        # Ensure correct extension
        if not filename.lower().endswith(extension):
            filename += extension
            
        try:
            # Create session data structure
            session_data = {
                "metadata": {
                    "timestamp": time.time(),
                    "count": len(data),
                    "app_version": APP_VERSION
                },
                "measurements": data
            }
            
            # Export based on format
            if format_type == "csv":
                csv_path = data_storage.export_to_csv(session_data, filename)
                self.statusBar().showMessage(f"Data exported to {csv_path}")
                
            elif format_type == "excel":
                excel_path = data_exporter.export_session(session_data, "excel", filename)
                self.statusBar().showMessage(f"Data exported to {excel_path}")
                
            elif format_type == "report":
                html_path = data_exporter.export_session(session_data, "report", filename)
                self.statusBar().showMessage(f"Report generated at {html_path}")
                
        except Exception as e:
            self.show_error(f"Error exporting data: {str(e)}")
    
    def show_error(self, message):
        """Show an error message."""
        QMessageBox.warning(self, "Error", message)
        self.statusBar().showMessage(f"Error: {message}")
    
    def show_settings(self):
        """Show settings dialog."""
        # This would be implemented with a settings dialog
        QMessageBox.information(self, "Settings", "Settings dialog not implemented yet")
    
    def toggle_dark_mode(self, enabled):
        """Toggle dark mode."""
        # This would implement dark mode styling
        if enabled:
            # Apply dark mode style here
            self.statusBar().showMessage("Dark mode enabled")
        else:
            # Remove dark mode style here
            self.statusBar().showMessage("Dark mode disabled")
    
    def show_about(self):
        """Show about dialog."""
        about_text = f"""
        <h1>{APP_NAME} v{APP_VERSION}</h1>
        <p>AS7341 Nephelometer Control and Analysis Software</p>
        <p>Author: {APP_AUTHOR}</p>
        <p>{APP_DESCRIPTION}</p>
        <p>Built with PyQt5 and matplotlib</p>
        """
        
        QMessageBox.about(self, f"About {APP_NAME}", about_text)


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # For a consistent look across platforms
    
    main_window = MainWindow()
    main_window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
