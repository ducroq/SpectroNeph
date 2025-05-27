from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, \
    QFormLayout, QDoubleSpinBox, QSpinBox, \
    QCheckBox, QPushButton, QLabel, QMessageBox, \
        QHBoxLayout
from PyQt5.QtCore import pyqtSignal


class DataAnalysisWidget(QWidget):
    """Widget for data analysis."""
    
    export_requested = pyqtSignal(str, list)  # format, data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_data = []
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface."""
        # Create layout
        layout = QVBoxLayout()
        
        # # Moving average
        # self.moving_avg_check = QCheckBox("Enable Moving Average")
        # self.moving_avg_check.setChecked(False)
        # options_layout.addRow("", self.moving_avg_check)
        
        # # Window size
        # self.window_spin = QSpinBox()
        # self.window_spin.setRange(2, 20)
        # self.window_spin.setValue(5)
        # self.window_spin.setEnabled(False)
        # options_layout.addRow("Window Size:", self.window_spin)
        
        # # Connect checkbox to window size
        # self.moving_avg_check.toggled.connect(self.window_spin.setEnabled)
        
        # options_group.setLayout(options_layout)
        # layout.addWidget(options_group)
        
        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout()
        
        # Export buttons
        export_btn_layout = QHBoxLayout()
        
        self.export_csv_btn = QPushButton("Export to CSV")
        self.export_excel_btn = QPushButton("Export to Excel")
        self.export_report_btn = QPushButton("Generate Report")
        
        self.export_csv_btn.clicked.connect(self.export_to_csv)
        self.export_excel_btn.clicked.connect(self.export_to_excel)
        self.export_report_btn.clicked.connect(self.generate_report)
        
        export_btn_layout.addWidget(self.export_csv_btn)
        export_btn_layout.addWidget(self.export_excel_btn)
        export_btn_layout.addWidget(self.export_report_btn)
        
        export_layout.addLayout(export_btn_layout)
        
        # Statistics
        self.stats_label = QLabel("No data available for analysis")
        export_layout.addWidget(self.stats_label)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Add stretch to push everything to the top
        layout.addStretch(1)
        
        self.setLayout(layout)
    
    def set_current_data(self, data):
        """Set the current data for analysis."""
        if isinstance(data, list):
            self.current_data = data
            self.update_statistics()
        else:
            # Single measurement
            self.current_data = [data] if data else []
            self.update_statistics()
    
    def update_statistics(self):
        """Update statistics from current data."""
        if not self.current_data:
            self.stats_label.setText("No data available for analysis")
            return
            
        # Calculate key statistics
        num_samples = len(self.current_data)
        
        # Get the most recent violet/red ratio
        last_ratio = 0
        if self.current_data[-1].get("ratios", {}).get("violet_red"):
            last_ratio = self.current_data[-1]["ratios"]["violet_red"]
        
        # Calculate all violet/red ratios if available
        vr_ratios = []
        for m in self.current_data:
            if "ratios" in m and "violet_red" in m["ratios"]:
                vr_ratios.append(m["ratios"]["violet_red"])
        
        # Calculate statistics if we have ratio data
        if vr_ratios:
            mean_ratio = sum(vr_ratios) / len(vr_ratios)
            max_ratio = max(vr_ratios)
            min_ratio = min(vr_ratios)
            
            # Create statistics text
            stats_text = (
                f"Data Summary:\n"
                f"- Samples: {num_samples}\n"
                f"- Current V/R Ratio: {last_ratio:.2f}\n"
                f"- Mean V/R Ratio: {mean_ratio:.2f}\n"
                f"- Min/Max V/R: {min_ratio:.2f} / {max_ratio:.2f}\n"
            )
        else:
            stats_text = f"Data Summary:\n- Samples: {num_samples}\n- No ratio data available"
            
        self.stats_label.setText(stats_text)
    
    def export_to_csv(self):
        """Export current data to CSV."""
        if not self.current_data:
            self.show_error("No data available to export")
            return

        self.export_requested.emit("csv", self.current_data)
        
    def export_to_excel(self):
        """Export current data to Excel."""
        if not self.current_data:
            self.show_error("No data available to export")
            return
            
        self.export_requested.emit("excel", self.current_data)
        
    def generate_report(self):
        """Generate a report from current data."""
        if not self.current_data:
            self.show_error("No data available to generate report")
            return
            
        self.export_requested.emit("report", self.current_data)
    
    def show_error(self, message):
        """Show an error message."""
        QMessageBox.warning(self, "Export Error", message)


