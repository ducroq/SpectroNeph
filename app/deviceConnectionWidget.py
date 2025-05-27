import sys
from pathlib import Path
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QLabel, QMessageBox, QHBoxLayout
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from typing import Dict, Optional

# Add project root to path to allow importing project modules
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project modules
from core.device import DeviceManager


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
        
        # # Connect button
        # self.connect_button = QPushButton("Connect")
        # self.connect_button.clicked.connect(self.toggle_connection)
        # layout.addWidget(self.connect_button)
        
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

