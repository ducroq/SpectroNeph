#!/usr/bin/env python3
"""
Data Acquisition and Processing Test for SpectroNeph

This script tests the data acquisition, processing, storage, and export
functionality of the SpectroNeph system.

Usage:
    python data_acquisition_and_processing_test.py [--port COM_PORT] [--no-plots]
"""

import sys
import time
import argparse
import json
import threading
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Project root added to sys.path: {project_root}")

# Import modules from the SpectroNeph project
sys.path.insert(0, str(project_root / "app"))
from core.device import DeviceManager
from hardware.nephelometer import Nephelometer, MeasurementMode, AgglutinationState
from utils.logging import setup_logging, get_logger
from utils.helpers import detect_serial_port
from data.acquisition import acquisition_manager
from data.processing import signal_processor
from data.storage import data_storage
from data.export import data_exporter

# Initialize logger
setup_logging()
logger = get_logger("data_test")

class DataAcquisitionTest:
    """
    Test class for data acquisition and processing.
    """
    
    def __init__(self, port=None, show_plots=True):
        """
        Initialize the test.
        
        Args:
            port: Serial port to connect to
            show_plots: Whether to display plots
        """
        self.port = port
        self.show_plots = show_plots
        self.device_manager = DeviceManager()
        self.nephelometer = None
        self.data_dir = project_root / "test_data"
        self.data_dir.mkdir(exist_ok=True)
        self.session_id = None
        
    def run_all_tests(self):
        """Run all tests."""
        print("\n=== SpectroNeph Data Acquisition and Processing Tests ===\n")
        
        # Connect to device
        if not self.connect_to_device():
            print("❌ Could not connect to device")
            return False
        
        # Run acquisition tests
        if not self.test_single_acquisition():
            print("❌ Single acquisition test failed")
        
        if not self.test_continuous_acquisition():
            print("❌ Continuous acquisition test failed")
        
        if not self.test_kinetic_acquisition():
            print("❌ Kinetic acquisition test failed")
            
        # Run processing tests
        if not self.test_signal_processing():
            print("❌ Signal processing test failed")
        
        if not self.test_ratio_calculation():
            print("❌ Ratio calculation test failed")
        
        # Run storage and export tests
        if not self.test_data_storage():
            print("❌ Data storage test failed")
        
        if not self.test_data_export():
            print("❌ Data export test failed")
        
        # Disconnect from device
        self.device_manager.disconnect()
        print("✓ Disconnected from device")
        
        print("\n=== All tests completed ===\n")
        return True
    
    def connect_to_device(self):
        """Connect to the device."""
        print("Connecting to device...")
        
        # Use provided port or detect
        port = self.port if self.port else detect_serial_port()
        if not port:
            print("❌ No serial port detected")
            return False
        
        # Connect using acquisition manager
        if not acquisition_manager.connect(port=port):
            print(f"❌ Failed to connect to device on {port}")
            return False
        
        print(f"✓ Connected to device on {port}")
        return True
        
    def test_single_acquisition(self):
        """Test single acquisition with direct nephelometer access."""
        print("\nTest: Single Acquisition")
        print("-" * 30)
        
        try:
            # Create direct access to nephelometer
            if not acquisition_manager.nephelometer:
                print("❌ Nephelometer not initialized")
                return False
                
            nephelometer = acquisition_manager.nephelometer
            
            # Configure nephelometer
            config = {
                "gain": 8,  # 128x gain for maximum sensitivity
                "integration_time": 200,  # 200ms
                "led_current": 15  # 15mA
            }
            
            result = nephelometer.configure(config)
            print(f"✓ Configured nephelometer: {result}")
            
            # Take measurement directly
            print("  Taking measurement directly from nephelometer...")
            measurement = nephelometer.take_single_measurement(subtract_background=True)
            
            if not measurement:
                print("❌ No measurement acquired")
                return False
            
            print("✓ Successfully acquired measurement")
            self._print_measurement_summary(measurement)
            
            # Create a session manually to store this measurement
            self.session_id = acquisition_manager.create_session()
            session = acquisition_manager.get_session(self.session_id)
            
            # Manually add the measurement without callbacks
            with session._lock:
                measurement["session_id"] = session.session_id
                if "timestamp" not in measurement:
                    measurement["timestamp"] = time.time()
                session.measurements.append(measurement)
                session.data_buffer.append(measurement)
            
            print(f"✓ Manually added measurement to session {self.session_id}")
            
            # Generate spectra plot
            self._plot_spectral_data(measurement, "single_acquisition_spectrum.png")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in single acquisition test: {str(e)}")
            return False
            
    def test_continuous_acquisition(self):
        """Test continuous acquisition."""
        print("\nTest: Continuous Acquisition")
        print("-" * 30)
        
        measurements = []
        event = threading.Event()
        
        # Callback function for measurements
        def measurement_callback(data):
            measurements.append(data)
            if len(measurements) >= 5:
                event.set()
        
        try:
            # Create a session
            config = {
                "mode": "continuous",
                "interval": 0.5,  # 0.5 seconds between measurements
                "subtract_background": True,
                "sensor_config": {
                    "gain": 8,  # 128x gain
                    "integration_time": 100,  # 100ms
                    "led_current": 15  # 15mA
                }
            }
            
            # Create and configure the session
            session_id = acquisition_manager.create_session(config)
            print(f"✓ Created continuous acquisition session: {session_id}")
            
            # Register callback
            acquisition_manager.register_data_callback(
                measurement_callback, session_id
            )
            
            # Start the session
            result = acquisition_manager.start_session(session_id)
            
            if not result:
                print("❌ Failed to start continuous acquisition session")
                return False
            
            print("✓ Started continuous acquisition")
            print("  Waiting for data collection...")
            
            # Wait for measurements with timeout
            success = event.wait(timeout=10.0)
            
            # Stop the session
            acquisition_manager.stop_session(session_id)
            
            if not success:
                print("❌ Did not receive enough measurements within timeout")
                return False
            
            print(f"✓ Collected {len(measurements)} measurements")
            
            # Store session ID for later tests
            self.session_id = session_id
            
            # Generate time series plot
            self._plot_time_series(measurements, "continuous_acquisition_timeseries.png")
            
            return True
            
        except Exception as e:
            # Make sure to stop the session
            try:
                acquisition_manager.stop_session(session_id)
            except:
                pass
            
            print(f"❌ Error in continuous acquisition test: {str(e)}")
            return False
    
    def test_kinetic_acquisition(self):
        """Test kinetic acquisition."""
        print("\nTest: Kinetic Acquisition")
        print("-" * 30)
        
        measurements = []
        event = threading.Event()
        
        # Callback function for measurements
        def measurement_callback(data):
            measurements.append(data)
            if len(measurements) >= 10:
                event.set()
        
        try:
            # Create a session
            config = {
                "mode": "kinetic",
                "duration": 3.0,  # 3 seconds
                "samples_per_second": 5.0,  # 5Hz sampling
                "subtract_background": True,
                "sensor_config": {
                    "gain": 8,  # 128x gain
                    "integration_time": 100,  # 100ms
                    "led_current": 15  # 15mA
                }
            }
            
            # Create and configure the session
            session_id = acquisition_manager.create_session(config)
            print(f"✓ Created kinetic acquisition session: {session_id}")
            
            # Register callback
            acquisition_manager.register_data_callback(
                measurement_callback, session_id
            )
            
            # Start the session
            result = acquisition_manager.start_session(session_id)
            
            if not result:
                print("❌ Failed to start kinetic acquisition session")
                return False
            
            print("✓ Started kinetic acquisition")
            print("  Waiting for data collection (up to 10 seconds)...")
            
            # Wait for measurements with timeout
            success = event.wait(timeout=10.0)
            
            # No need to stop the session, kinetic mode stops automatically
            
            if not success:
                print("❌ Did not receive enough measurements within timeout")
                return False
            
            print(f"✓ Collected {len(measurements)} measurements")
            
            # Store the measurements for later tests
            if self.session_id:
                # Process measurements to add ratios
                processed_measurements = []
                for m in measurements:
                    # Calculate ratios
                    processed = signal_processor.calculate_ratios(m)
                    processed_measurements.append(processed)
                
                # Update the session with processed measurements
                session = acquisition_manager.get_session(self.session_id)
                if session:
                    session.measurements = processed_measurements
            
            # Generate time series plot
            self._plot_time_series(measurements, "kinetic_acquisition_timeseries.png")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in kinetic acquisition test: {str(e)}")
            return False
    
    def test_signal_processing(self):
        """Test signal processing functionality."""
        print("\nTest: Signal Processing")
        print("-" * 30)
        
        try:
            # Get measurements from previous tests
            session = acquisition_manager.get_session(self.session_id)
            
            if not session or not session.measurements:
                print("❌ No measurements available for processing")
                return False
            
            measurement = session.measurements[0]
            
            # Test filtering
            print("  Testing signal filtering...")
            filtered = signal_processor.filter_measurement(measurement, method="moving_average", window_size=3)
            
            if "processed" not in filtered or "filtered" not in filtered["processed"]:
                print("❌ Filtering did not produce expected output")
                return False
            
            print("✓ Signal filtering successful")
            
            # Test normalization
            print("  Testing signal normalization...")
            normalized = signal_processor.normalize_measurement(filtered, reference_channel="Clear")
            
            if "processed" not in normalized or "normalized" not in normalized["processed"]:
                print("❌ Normalization did not produce expected output")
                return False
            
            print("✓ Signal normalization successful")
            
            # Test feature extraction
            print("  Testing feature extraction...")
            features = signal_processor.extract_features(normalized)
            
            if not features:
                print("❌ Feature extraction did not produce expected output")
                return False
            
            print("✓ Feature extraction successful")
            print(f"  Extracted features: {', '.join(features.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in signal processing test: {str(e)}")
            return False
    
    def test_ratio_calculation(self):
        """Test ratio calculation and agglutination detection."""
        print("\nTest: Ratio Calculation and Agglutination Detection")
        print("-" * 30)
        
        try:
            # Get measurements from previous tests
            session = acquisition_manager.get_session(self.session_id)
            
            if not session or not session.measurements:
                print("❌ No measurements available for ratio calculation")
                return False
            
            # Apply ratio calculation to all measurements
            print("  Calculating ratios for all measurements...")
            for i, measurement in enumerate(session.measurements):
                # Calculate ratios if not already present
                if "ratios" not in measurement:
                    session.measurements[i] = signal_processor.calculate_ratios(measurement)
            
            # Check if ratios were calculated
            if "ratios" not in session.measurements[0]:
                print("❌ Ratio calculation did not produce expected output")
                return False
            
            print("✓ Ratio calculation successful")
            self._print_ratio_summary(session.measurements[0])
            
            # Test agglutination classification
            print("  Testing agglutination classification...")
            for i, measurement in enumerate(session.measurements):
                # Classify agglutination
                classification = signal_processor.classify_agglutination(measurement)
                session.measurements[i]["agglutination"] = classification
            
            # Check if classification was successful
            if "agglutination" not in session.measurements[0]:
                print("❌ Agglutination classification did not produce expected output")
                return False
            
            print("✓ Agglutination classification successful")
            self._print_agglutination_summary(session.measurements[0]["agglutination"])
            
            # Analyze kinetics if we have multiple measurements
            if len(session.measurements) > 5:
                print("  Testing kinetic analysis...")
                kinetics = signal_processor.analyze_kinetics(
                    session.measurements, parameter="violet_red"
                )
                
                if not kinetics:
                    print("❌ Kinetic analysis did not produce expected output")
                    return False
                
                print("✓ Kinetic analysis successful")
                print(f"  Rate: {kinetics.get('rate', 0):.4f}/sec, "
                     f"Change: {kinetics.get('change', 0):.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in ratio calculation test: {str(e)}")
            return False
    
    def test_data_storage(self):
        """Test data storage functionality."""
        print("\nTest: Data Storage")
        print("-" * 30)
        
        try:
            # Get session data
            session = acquisition_manager.get_session(self.session_id)
            
            if not session:
                print("❌ No session available for storage test")
                return False
            
            # Prepare session data for storage
            session_data = {
                "metadata": session.get_metadata(),
                "measurements": session.get_data()
            }

            data_storage.data_dir = self.data_dir  # lame fix
            
            # Test JSON storage
            print("  Testing JSON storage...")
            json_path = data_storage.save_session(
                session_data, format="json", 
                filename=f"test_session_{int(time.time())}.json"
            )
            
            if not json_path or not Path(json_path).exists():
                print("❌ JSON storage failed")
                return False
            
            print(f"✓ Session saved to {json_path}")
            
            # Test CSV storage
            print("  Testing CSV storage...")
            csv_path = data_storage.save_session(
                session_data, format="csv", 
                filename=f"test_session_{int(time.time())}.csv"
            )
            
            if not csv_path or not Path(csv_path).exists():
                print("❌ CSV storage failed")
                return False
            
            print(f"✓ Session saved to {csv_path}")
            
            # Test loading data
            print("  Testing data loading...")
            loaded_data = data_storage.load_session(Path(json_path).name)
            
            if not loaded_data or "measurements" not in loaded_data:
                print("❌ Data loading failed")
                return False
            
            print(f"✓ Loaded {len(loaded_data.get('measurements', []))} measurements from {json_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in data storage test: {str(e)}")
            return False
    
    def test_data_export(self):
        """Test data export functionality."""
        print("\nTest: Data Export")
        print("-" * 30)
        
        try:
            # Get session data
            session = acquisition_manager.get_session(self.session_id)
            
            if not session:
                print("❌ No session available for export test")
                return False
            
            # Prepare session data for export
            session_data = {
                "metadata": session.get_metadata(),
                "measurements": session.get_data()
            }

            data_exporter.export_dir = self.data_dir  # lame fix
            
            # Test Excel export
            print("  Testing Excel export...")
            excel_path = data_exporter.export_session(
                session_data, format="excel", 
                filename=f"test_export_{int(time.time())}.xlsx"
            )
            
            if not excel_path or not Path(excel_path).exists():
                print("❌ Excel export failed")
                return False
            
            print(f"✓ Session exported to {excel_path}")
            
            # Test HTML report
            print("  Testing HTML report generation...")
            html_path = data_exporter.export_session(
                session_data, format="report", 
                filename=f"test_report_{int(time.time())}.html"
            )
            
            if not html_path or not Path(html_path).exists():
                print("❌ HTML report generation failed")
                return False
            
            print(f"✓ HTML report generated at {html_path}")
            
            # Test figure generation
            print("  Testing figure generation...")
            figures = data_exporter.generate_figures(session_data)
            
            if not figures:
                print("❌ Figure generation failed")
                return False
            
            # Export one figure as a test
            if "spectral_profile" in figures:
                fig_path = self.data_dir / "test_spectral_profile.png"
                data_exporter.export_figure(figures["spectral_profile"], fig_path)
                print(f"✓ Exported figure to {fig_path}")
            
            print(f"✓ Generated {len(figures)} figures")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in data export test: {str(e)}")
            return False
    
    def _print_measurement_summary(self, measurement):
        """Print a summary of a measurement."""
        if not measurement:
            return
        
        print("\n  Measurement Summary:")
        print("  " + "-" * 28)
        
        # Print raw channels
        raw_data = measurement.get("raw", {})
        if raw_data:
            spectral_channels = [ch for ch in raw_data.keys() if ch.startswith("F")]
            if spectral_channels:
                print("  Spectral Channels:")
                for channel in sorted(spectral_channels):
                    print(f"    {channel}: {raw_data[channel]}")
        
        # Print timestamp
        if "timestamp" in measurement:
            print(f"  Timestamp: {measurement['timestamp']}")
    
    def _print_ratio_summary(self, measurement):
        """Print a summary of ratio calculations."""
        if not measurement or "ratios" not in measurement:
            return
        
        print("\n  Ratio Summary:")
        print("  " + "-" * 28)
        
        ratios = measurement["ratios"]
        for ratio, value in ratios.items():
            print(f"    {ratio}: {value:.4f}")
    
    def _print_agglutination_summary(self, classification):
        """Print a summary of agglutination classification."""
        if not classification:
            return
        
        print("\n  Agglutination Classification:")
        print("  " + "-" * 28)
        
        print(f"    State: {classification.get('state', 'unknown')}")
        print(f"    Score: {classification.get('score', 0):.1f}")
        print(f"    Confidence: {classification.get('confidence', 0):.2f}")
        print(f"    Primary Ratio: {classification.get('primary_ratio', 0):.4f}")
    
    def _plot_spectral_data(self, measurement, filename=None):
        """Plot spectral data from a measurement."""
        if not measurement or "raw" not in measurement:
            return
        
        # Extract data
        raw_data = measurement["raw"]
        
        # Get channels and values
        channels = []
        values = []
        colors = []
        
        # Map wavelengths to channels
        wavelengths = {
            "F1": 415, "F2": 445, "F3": 480, "F4": 515,
            "F5": 555, "F6": 590, "F7": 630, "F8": 680
        }
        
        # Channel colors
        channel_colors = {
            "F1": "indigo", "F2": "blue", "F3": "royalblue", "F4": "green",
            "F5": "yellowgreen", "F6": "gold", "F7": "orange", "F8": "red"
        }
        
        # Collect data for spectral channels
        for channel, wavelength in wavelengths.items():
            if channel in raw_data:
                channels.append(channel)
                values.append(raw_data[channel])
                colors.append(channel_colors.get(channel, "gray"))
        
        if not channels:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get wavelengths for x-axis
        x = [wavelengths[ch] for ch in channels]
        
        # Plot as bars with colors
        ax.bar(x, values, width=20, color=colors, alpha=0.7)
        
        # Add labels
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Signal Value")
        ax.set_title("Spectral Profile")
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add channel labels
        for i, (wavelength, value, channel) in enumerate(zip(x, values, channels)):
            ax.text(wavelength, value * 1.05, channel, ha="center", va="bottom")
        
        # Save figure if filename provided
        if filename:
            filepath = self.data_dir / filename
            fig.savefig(filepath, dpi=100, bbox_inches="tight")
            print(f"  Saved plot to {filepath}")
        
        # Display if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    def _plot_time_series(self, measurements, filename=None):
        """Plot time series data from measurements."""
        if not measurements:
            return
        
        # Extract timestamps
        timestamps = []
        for m in measurements:
            if "timestamp" in m:
                timestamps.append(m["timestamp"])
            elif "elapsed_seconds" in m:
                timestamps.append(m["elapsed_seconds"])
            else:
                timestamps.append(len(timestamps))
        
        # Normalize timestamps to start from 0
        if timestamps and timestamps[0] != 0:
            offset = timestamps[0]
            timestamps = [t - offset for t in timestamps]
        
        # Extract values for key channels
        channels = ["F1", "F4", "F8"]  # Violet, Green, Red
        channel_data = {ch: [] for ch in channels}
        
        for m in measurements:
            raw_data = m.get("raw", {})
            for ch in channels:
                if ch in raw_data:
                    channel_data[ch].append(raw_data[ch])
                else:
                    # Use 0 if channel not found
                    channel_data[ch].append(0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each channel
        colors = {"F1": "indigo", "F4": "green", "F8": "red"}
        
        for ch in channels:
            if len(channel_data[ch]) == len(timestamps):
                ax.plot(timestamps, channel_data[ch], 
                       "-o", label=ch, color=colors.get(ch, "gray"), alpha=0.7)
        
        # Add labels
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Signal Value")
        ax.set_title("Time Series")
        
        # Add legend and grid
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure if filename provided
        if filename:
            filepath = self.data_dir / filename
            fig.savefig(filepath, dpi=100, bbox_inches="tight")
            print(f"  Saved plot to {filepath}")
        
        # Display if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

def main():
    """Main entry point for the test script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="SpectroNeph Data Acquisition and Processing Tests"
    )
    parser.add_argument("--port", help="Serial port to connect to")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot display")
    args = parser.parse_args()
    
    # Create and run the test
    test = DataAcquisitionTest(port=args.port, show_plots=not args.no_plots)
    success = test.run_all_tests()
    
    # Return appropriate exit code
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())