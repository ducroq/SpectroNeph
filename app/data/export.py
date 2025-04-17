"""
Data export functionality for the SpectroNeph system.

This module handles exporting SpectroNeph data to various formats:
- CSV: For spreadsheet analysis
- Excel: For detailed workbooks
- JSON: For software interoperability
- Raw data formats: For other analysis tools
- Report formats: PDF, HTML, etc.
"""

import os
import time
import json
import csv
import datetime
import io
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, BinaryIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from config import settings
from utils.logging import get_logger
from data.storage import data_storage

# Initialize logger
logger = get_logger(__name__)

class DataExporter:
    """
    Data exporter class for SpectroNeph data.
    
    This class provides methods for exporting data to various formats
    and generating reports and visualizations.
    """
    
    def __init__(self, export_dir: Optional[str] = None):
        """
        Initialize the data exporter.
        
        Args:
            export_dir: Directory for exports, defaults to settings.DATA_DIR / 'exports'
        """
        self.export_dir = Path(export_dir) if export_dir else Path(settings.get("DATA_DIR", "data")) / "exports"
        
        # Create export directory if it doesn't exist
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized data exporter with export directory: {self.export_dir}")
    
    def export_session(self, session_data: Dict[str, Any], 
                      format: str = "excel",
                      filename: Optional[str] = None) -> str:
        """
        Export a session to a specific format.
        
        Args:
            session_data: Session data to export
            format: Export format ('excel', 'csv', 'json', 'report')
            filename: Optional filename, will be generated if not provided
            
        Returns:
            str: Path to the exported file
        """
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = session_data.get("metadata", {}).get("session_id", "session")
            filename = f"{session_id}_{timestamp}"
        
        # Ensure filename has the correct extension
        if format == "excel" and not filename.endswith(".xlsx"):
            filename = f"{filename}.xlsx"
        elif format == "csv" and not filename.endswith(".csv"):
            filename = f"{filename}.csv"
        elif format == "json" and not filename.endswith(".json"):
            filename = f"{filename}.json"
        elif format == "report" and not filename.endswith(".html"):
            filename = f"{filename}.html"
        
        # Create the full path
        filepath = self.export_dir / filename
        
        # Export in the specified format
        if format == "excel":
            return self._export_excel(session_data, filepath)
        elif format == "csv":
            return self._export_csv(session_data, filepath)
        elif format == "json":
            return self._export_json(session_data, filepath)
        elif format == "report":
            return self._export_report(session_data, filepath)
        else:
            logger.error(f"Unsupported export format: {format}")
            raise ValueError(f"Unsupported export format: {format}")
    
    def export_dataframe(self, measurements: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert measurements to a pandas DataFrame.
        
        Args:
            measurements: List of measurements to convert
            
        Returns:
            pd.DataFrame: DataFrame containing the measurements
        """
        # Create a flattened structure for DataFrame
        flattened_data = []
        
        for m in measurements:
            row = {
                "timestamp": m.get("timestamp", 0),
                "session_id": m.get("session_id", "")
            }
            
            # Add raw data
            if "raw" in m:
                for channel, value in m["raw"].items():
                    row[f"raw_{channel}"] = value
            
            # Add processed data
            if "processed" in m:
                for channel, value in m["processed"].items():
                    row[f"processed_{channel}"] = value
            
            # Add ratios
            if "ratios" in m:
                for ratio, value in m["ratios"].items():
                    row[f"ratio_{ratio}"] = value
            
            flattened_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(flattened_data)
        
        # Convert timestamp to datetime if it looks like a Unix timestamp
        if "timestamp" in df.columns and df["timestamp"].dtype == np.float64:
            try:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit='s')
            except:
                pass
        
        return df
    
    def generate_figures(self, session_data: Dict[str, Any], 
                        figure_types: Optional[List[str]] = None) -> Dict[str, Figure]:
        """
        Generate matplotlib figures from session data.
        
        Args:
            session_data: Session data to visualize
            figure_types: Types of figures to generate
            
        Returns:
            Dict[str, Figure]: Dictionary of figure types to matplotlib figures
        """
        if not figure_types:
            figure_types = ["spectral_profile", "time_series", "ratio_analysis"]
        
        figures = {}
        measurements = session_data.get("measurements", [])
        
        if not measurements:
            logger.warning("No measurements found for figure generation")
            return figures
        
        # Convert to DataFrame for easier manipulation
        df = self.export_dataframe(measurements)
        
        # Generate requested figures
        for figure_type in figure_types:
            if figure_type == "spectral_profile":
                figures[figure_type] = self._generate_spectral_profile(df, session_data)
            elif figure_type == "time_series":
                figures[figure_type] = self._generate_time_series(df, session_data)
            elif figure_type == "ratio_analysis":
                figures[figure_type] = self._generate_ratio_analysis(df, session_data)
        
        return figures
    
    def export_figure(self, figure: Figure, filepath: Union[str, Path], dpi: int = 300) -> str:
        """
        Export a matplotlib figure to a file.
        
        Args:
            figure: Figure to export
            filepath: Path to save the figure
            dpi: DPI for the exported figure
            
        Returns:
            str: Path to the exported figure
        """
        filepath = Path(filepath)
        
        # Create parent directories if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from extension
        format = filepath.suffix.lower()[1:]  # Remove the leading dot
        
        # Save the figure
        figure.savefig(filepath, dpi=dpi, bbox_inches='tight', format=format)
        
        return str(filepath)
    
    def export_excel_workbook(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                            filepath: Union[str, Path],
                            include_plots: bool = True) -> str:
        """
        Export data to an Excel workbook.
        
        Args:
            data: Data to export (session or list of measurements)
            filepath: Path to save the Excel file
            include_plots: Whether to include plots in the workbook
            
        Returns:
            str: Path to the exported file
        """
        filepath = Path(filepath)
        
        # Create parent directories if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract measurements and metadata
        measurements = []
        metadata = {}
        
        if isinstance(data, dict) and "measurements" in data:
            # This is a session structure
            measurements = data["measurements"]
            metadata = data.get("metadata", {})
        elif isinstance(data, list):
            # This is a list of measurements
            measurements = data
        else:
            logger.error("Invalid data format for Excel export")
            raise ValueError("Invalid data format for Excel export")
        
        if not measurements:
            logger.warning("No measurements to export")
            return str(filepath)
        
        # Convert measurements to DataFrame
        df = self.export_dataframe(measurements)
        
        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            # Write main data sheet
            df.to_excel(writer, sheet_name='Data', index=False)
            
            # Write metadata sheet
            if metadata:
                # Convert metadata to DataFrame for Excel output
                meta_df = pd.DataFrame([metadata])
                meta_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Add plots if requested
            if include_plots:
                workbook = writer.book
                
                # Add spectral profile sheet
                self._add_spectral_profile_sheet(workbook, writer, df, metadata)
                
                # Add time series sheet if we have timestamps
                if "timestamp" in df.columns or "datetime" in df.columns:
                    self._add_time_series_sheet(workbook, writer, df, metadata)
                
                # Add ratio analysis sheet if we have ratio data
                ratio_columns = [col for col in df.columns if col.startswith("ratio_")]
                if ratio_columns:
                    self._add_ratio_analysis_sheet(workbook, writer, df, metadata)
        
        logger.info(f"Exported data to Excel workbook: {filepath}")
        return str(filepath)
    
    def _export_excel(self, session_data: Dict[str, Any], filepath: Path) -> str:
        """
        Export session data to Excel format.
        
        Args:
            session_data: Session data to export
            filepath: Path to save the file
            
        Returns:
            str: Path to the exported file
        """
        return self.export_excel_workbook(session_data, filepath, include_plots=True)
    
    def _export_csv(self, session_data: Dict[str, Any], filepath: Path) -> str:
        """
        Export session data to CSV format.
        
        Args:
            session_data: Session data to export
            filepath: Path to save the file
            
        Returns:
            str: Path to the exported file
        """
        # We'll delegate to the storage module for CSV export
        return data_storage.export_to_csv(session_data, filepath, format_type="measurements")
    
    def _export_json(self, session_data: Dict[str, Any], filepath: Path) -> str:
        """
        Export session data to JSON format.
        
        Args:
            session_data: Session data to export
            filepath: Path to save the file
            
        Returns:
            str: Path to the exported file
        """
        # Delegate to the storage module for JSON export
        return data_storage._save_json(session_data, filepath)
    
    def _export_report(self, session_data: Dict[str, Any], filepath: Path) -> str:
        """
        Export session data to an HTML report.
        
        Args:
            session_data: Session data to export
            filepath: Path to save the file
            
        Returns:
            str: Path to the exported file
        """
        # Extract metadata and measurements
        metadata = session_data.get("metadata", {})
        measurements = session_data.get("measurements", [])
        
        if not measurements:
            logger.warning("No measurements found for report generation")
            return str(filepath)
        
        # Convert to DataFrame
        df = self.export_dataframe(measurements)
        
        # Generate figures
        figures = self.generate_figures(session_data)
        
        # Convert figures to base64 for embedding in HTML
        figure_images = {}
        for name, fig in figures.items():
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            figure_images[name] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
        
        # Build HTML content
        html_content = self._build_html_report(session_data, df, figure_images)
        
        # Save the HTML file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Exported HTML report to: {filepath}")
        return str(filepath)
    
    def _build_html_report(self, session_data: Dict[str, Any], 
                          df: pd.DataFrame, 
                          figure_images: Dict[str, str]) -> str:
        """
        Build HTML content for a report.
        
        Args:
            session_data: Session data for the report
            df: DataFrame of measurements
            figure_images: Dictionary of base64-encoded figure images
            
        Returns:
            str: HTML content
        """
        metadata = session_data.get("metadata", {})
        
        # Get basic session info
        session_id = metadata.get("session_id", "Unknown")
        start_time = metadata.get("start_time", 0)
        if start_time:
            start_time_str = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_time_str = "Unknown"
        
        # Get configuration info
        config = metadata.get("config", {})
        mode = config.get("mode", "Unknown")
        
        # Start building HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SpectroNeph Report - {session_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .figure-container {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metadata-table {{ width: 100%; margin-bottom: 20px; }}
                .metadata-table td {{ padding: 5px; border: none; }}
                .metadata-table td:first-child {{ font-weight: bold; width: 200px; }}
                .chart {{ max-width: 100%; height: auto; }}
                .footer {{ text-align: center; margin-top: 50px; font-size: 0.8em; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>SpectroNeph Measurement Report</h1>
                    <p>Session ID: {session_id}</p>
                    <p>Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                
                <div class="section">
                    <h2>Session Information</h2>
                    <table class="metadata-table">
                        <tr><td>Session ID</td><td>{session_id}</td></tr>
                        <tr><td>Start Time</td><td>{start_time_str}</td></tr>
                        <tr><td>Acquisition Mode</td><td>{mode}</td></tr>
                        <tr><td>Measurements</td><td>{len(session_data.get("measurements", []))}</td></tr>
        """
        
        # Add more configuration details
        for key, value in config.items():
            if key != "mode" and not isinstance(value, dict):
                html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>\n"
        
        html += """
                    </table>
                </div>
        """
        
        # Add figures
        if figure_images:
            html += """
                <div class="section">
                    <h2>Measurement Analysis</h2>
            """
            
            for name, img_data in figure_images.items():
                title = name.replace('_', ' ').title()
                html += f"""
                    <div class="figure-container">
                        <h3>{title}</h3>
                        <img src="data:image/png;base64,{img_data}" alt="{title}" class="chart">
                    </div>
                """
            
            html += """
                </div>
            """
        
        # Add data summary
        html += """
                <div class="section">
                    <h2>Data Summary</h2>
        """
        
        # Add spectral channels summary if available
        spectral_columns = [col for col in df.columns if col.startswith("raw_F")]
        if spectral_columns:
            stats_df = df[spectral_columns].describe().transpose()
            stats_df.index = [col.replace("raw_", "") for col in stats_df.index]
            
            html += """
                    <h3>Spectral Channels</h3>
                    <table>
                        <tr>
                            <th>Channel</th>
                            <th>Mean</th>
                            <th>Min</th>
                            <th>Max</th>
                            <th>Std Dev</th>
                        </tr>
            """
            
            for idx, row in stats_df.iterrows():
                html += f"""
                        <tr>
                            <td>{idx}</td>
                            <td>{row['mean']:.2f}</td>
                            <td>{row['min']:.2f}</td>
                            <td>{row['max']:.2f}</td>
                            <td>{row['std']:.2f}</td>
                        </tr>
                """
            
            html += """
                    </table>
            """
        
        # Add ratio summary if available
        ratio_columns = [col for col in df.columns if col.startswith("ratio_")]
        if ratio_columns:
            stats_df = df[ratio_columns].describe().transpose()
            stats_df.index = [col.replace("ratio_", "") for col in stats_df.index]
            
            html += """
                    <h3>Spectral Ratios</h3>
                    <table>
                        <tr>
                            <th>Ratio</th>
                            <th>Mean</th>
                            <th>Min</th>
                            <th>Max</th>
                            <th>Std Dev</th>
                        </tr>
            """
            
            for idx, row in stats_df.iterrows():
                html += f"""
                        <tr>
                            <td>{idx}</td>
                            <td>{row['mean']:.2f}</td>
                            <td>{row['min']:.2f}</td>
                            <td>{row['max']:.2f}</td>
                            <td>{row['std']:.2f}</td>
                        </tr>
                """
            
            html += """
                    </table>
            """
        
        html += """
                </div>
                
                <div class="section">
                    <h2>Raw Data Preview</h2>
                    <p>First 10 measurements:</p>
                    <table>
                        <tr>
        """
        
        # Add table headers
        preview_columns = ["timestamp"]
        if "datetime" in df.columns:
            preview_columns.append("datetime")
        
        # Add spectral channels
        for col in [c for c in df.columns if c.startswith("raw_F")]:
            preview_columns.append(col)
        
        # Add ratios
        for col in [c for c in df.columns if c.startswith("ratio_")]:
            preview_columns.append(col)
        
        # Limit columns for readability
        preview_columns = preview_columns[:10]
        
        # Add headers
        for col in preview_columns:
            display_col = col.replace("raw_", "").replace("ratio_", "Ratio: ")
            html += f"<th>{display_col}</th>\n"
        
        html += """
                        </tr>
        """
        
        # Add data rows (up to 10)
        for _, row in df.head(10).iterrows():
            html += "<tr>\n"
            for col in preview_columns:
                value = row.get(col, "")
                if isinstance(value, (float, np.float64)):
                    html += f"<td>{value:.4f}</td>\n"
                elif isinstance(value, pd.Timestamp):
                    html += f"<td>{value.strftime('%Y-%m-%d %H:%M:%S')}</td>\n"
                else:
                    html += f"<td>{value}</td>\n"
            html += "</tr>\n"
        
        html += """
                    </table>
                </div>
                
                <div class="footer">
                    <p>Generated by SpectroNeph Data Exporter</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_spectral_profile(self, df: pd.DataFrame, session_data: Dict[str, Any]) -> Figure:
        """
        Generate a spectral profile figure.
        
        Args:
            df: DataFrame containing the measurements
            session_data: Session data for context
            
        Returns:
            Figure: Matplotlib figure
        """
        # Find raw spectral channels
        spectral_columns = [col for col in df.columns if col.startswith("raw_F")]
        if not spectral_columns:
            logger.warning("No spectral channels found for profile generation")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No spectral data available", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Extract channel names and wavelengths
        channel_wavelengths = {
            "F1": 415, "F2": 445, "F3": 480, "F4": 515,
            "F5": 555, "F6": 590, "F7": 630, "F8": 680
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate mean values across all measurements
        mean_values = df[spectral_columns].mean()
        
        # Prepare data for plotting
        channels = [col.replace("raw_", "") for col in spectral_columns]
        wavelengths = [channel_wavelengths.get(ch, 0) for ch in channels]
        values = [mean_values[col] for col in spectral_columns]
        
        # Sort by wavelength
        data = sorted(zip(wavelengths, values, channels))
        wavelengths = [d[0] for d in data]
        values = [d[1] for d in data]
        channels = [d[2] for d in data]
        
        # Define colors for each channel
        channel_colors = {
            "F1": "indigo", "F2": "blue", "F3": "royalblue", "F4": "green",
            "F5": "yellowgreen", "F6": "gold", "F7": "orange", "F8": "red"
        }
        colors = [channel_colors.get(ch, "gray") for ch in channels]
        
        # Plot the spectrum
        ax.bar(wavelengths, values, width=20, color=colors, alpha=0.7)
        ax.plot(wavelengths, values, 'o-', color='black', alpha=0.7)
        
        # Add channel labels
        for i, (wl, val, ch) in enumerate(zip(wavelengths, values, channels)):
            ax.text(wl, val * 1.05, ch, ha='center', va='bottom')
        
        # Add labels and title
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Signal Intensity")
        
        # Get session info for title
        session_id = session_data.get("metadata", {}).get("session_id", "Unknown")
        title = f"Spectral Profile - Session {session_id}"
        ax.set_title(title)
        
        # Add grid and tight layout
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        return fig
    
    def _generate_time_series(self, df: pd.DataFrame, session_data: Dict[str, Any]) -> Figure:
        """
        Generate a time series figure.
        
        Args:
            df: DataFrame containing the measurements
            session_data: Session data for context
            
        Returns:
            Figure: Matplotlib figure
        """
        # Check if we have time information and spectral data
        has_time = "timestamp" in df.columns or "datetime" in df.columns
        spectral_columns = [col for col in df.columns if col.startswith("raw_F")]
        
        if not has_time or not spectral_columns:
            logger.warning("Missing time or spectral data for time series generation")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Insufficient data for time series", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Determine time column
        time_col = "datetime" if "datetime" in df.columns else "timestamp"
        
        # Sort by time
        df_sorted = df.sort_values(by=time_col)
        
        # Select key channels (subset for clarity)
        key_channels = ["raw_F1", "raw_F4", "raw_F8"]  # Violet, Green, Red
        available_channels = [ch for ch in key_channels if ch in df.columns]
        
        if not available_channels:
            available_channels = spectral_columns[:3]  # Take first 3 channels
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define colors for each channel
        channel_colors = {
            "raw_F1": "indigo", "raw_F2": "blue", "raw_F3": "royalblue", 
            "raw_F4": "green", "raw_F5": "yellowgreen", "raw_F6": "gold", 
            "raw_F7": "orange", "raw_F8": "red"
        }
        
        # Plot each channel
        for channel in available_channels:
            color = channel_colors.get(channel, "gray")
            label = channel.replace("raw_", "")
            ax.plot(df_sorted[time_col], df_sorted[channel], 
                   '-', color=color, label=label, alpha=0.7)
        
        # Add labels and title
        if time_col == "datetime":
            ax.set_xlabel("Time")
        else:
            ax.set_xlabel("Time (seconds)")
        
        ax.set_ylabel("Signal Intensity")
        
        # Get session info for title
        session_id = session_data.get("metadata", {}).get("session_id", "Unknown")
        mode = session_data.get("metadata", {}).get("config", {}).get("mode", "")
        title = f"Time Series - {mode.title()} Mode - Session {session_id}"
        ax.set_title(title)
        
        # Add legend and grid
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add tight layout
        fig.tight_layout()
        
        return fig
    
    def _generate_ratio_analysis(self, df: pd.DataFrame, session_data: Dict[str, Any]) -> Figure:
        """
        Generate a ratio analysis figure.
        
        Args:
            df: DataFrame containing the measurements
            session_data: Session data for context
            
        Returns:
            Figure: Matplotlib figure
        """
        # Check if we have ratio data
        ratio_columns = [col for col in df.columns if col.startswith("ratio_")]
        if not ratio_columns:
            logger.warning("No ratio data found for ratio analysis")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No ratio data available", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Check if we have time data
        has_time = "timestamp" in df.columns or "datetime" in df.columns
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[2, 1])
        
        # Plot 1: Time series of ratios (if time available)
        if has_time:
            time_col = "datetime" if "datetime" in df.columns else "timestamp"
            df_sorted = df.sort_values(by=time_col)
            
            for ratio in ratio_columns:
                label = ratio.replace("ratio_", "")
                ax1.plot(df_sorted[time_col], df_sorted[ratio], 
                        '-o', label=label, alpha=0.7)
            
            if time_col == "datetime":
                ax1.set_xlabel("Time")
            else:
                ax1.set_xlabel("Time (seconds)")
            
            ax1.set_ylabel("Ratio Value")
            ax1.set_title("Spectral Ratios Over Time")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No time data available for time series", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot 2: Distribution of ratio values
        for ratio in ratio_columns:
            label = ratio.replace("ratio_", "")
            ax2.hist(df[ratio].dropna(), bins=20, alpha=0.5, label=label)
        
        ax2.set_xlabel("Ratio Value")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Ratio Values")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Get session info for overall title
        session_id = session_data.get("metadata", {}).get("session_id", "Unknown")
        fig.suptitle(f"Ratio Analysis - Session {session_id}", fontsize=14)
        
        # Add tight layout with room for suptitle
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig
    
    def _add_spectral_profile_sheet(self, workbook, writer, df, metadata):
        """
        Add a spectral profile chart to an Excel workbook.
        
        Args:
            workbook: xlsxwriter workbook
            writer: pandas ExcelWriter
            df: DataFrame containing the measurements
            metadata: Session metadata
        """
        # Create a new worksheet
        worksheet = workbook.add_worksheet('Spectral Profile')
        
        # Find raw spectral channels
        spectral_columns = [col for col in df.columns if col.startswith("raw_F")]
        if not spectral_columns:
            worksheet.write(0, 0, "No spectral data available")
            return
        
        # Extract channel names and wavelengths
        channel_wavelengths = {
            "F1": 415, "F2": 445, "F3": 480, "F4": 515,
            "F5": 555, "F6": 590, "F7": 630, "F8": 680
        }
        
        # Calculate mean values across all measurements
        mean_values = df[spectral_columns].mean()
        
        # Prepare data for the chart
        channels = [col.replace("raw_", "") for col in spectral_columns]
        wavelengths = [channel_wavelengths.get(ch, 0) for ch in channels]
        values = [mean_values[col] for col in spectral_columns]
        
        # Sort by wavelength
        data = sorted(zip(wavelengths, values, channels))
        
        # Write headers
        worksheet.write(0, 0, "Channel")
        worksheet.write(0, 1, "Wavelength (nm)")
        worksheet.write(0, 2, "Value")
        
        # Write data
        for i, (wl, val, ch) in enumerate(data):
            row = i + 1
            worksheet.write(row, 0, ch)
            worksheet.write(row, 1, wl)
            worksheet.write(row, 2, val)
        
        # Create a chart
        chart = workbook.add_chart({'type': 'column'})
        
        # Add a series with wavelength as x-axis
        chart.add_series({
            'name': 'Spectral Profile',
            'categories': ['Spectral Profile', 1, 1, len(data), 1],
            'values': ['Spectral Profile', 1, 2, len(data), 2],
            'data_labels': {'value': True}
        })
        
        # Add chart labels
        chart.set_title({'name': 'Spectral Profile'})
        chart.set_x_axis({'name': 'Wavelength (nm)'})
        chart.set_y_axis({'name': 'Signal Value'})
        
        # Insert the chart into the worksheet
        worksheet.insert_chart('E2', chart, {'x_scale': 1.5, 'y_scale': 1.5})
    
    def _add_time_series_sheet(self, workbook, writer, df, metadata):
        """
        Add a time series chart to an Excel workbook.
        
        Args:
            workbook: xlsxwriter workbook
            writer: pandas ExcelWriter
            df: DataFrame containing the measurements
            metadata: Session metadata
        """
        # Create a new worksheet
        worksheet = workbook.add_worksheet('Time Series')
        
        # Check if we have time information and spectral data
        has_time = "timestamp" in df.columns or "datetime" in df.columns
        spectral_columns = [col for col in df.columns if col.startswith("raw_F")]
        
        if not has_time or not spectral_columns:
            worksheet.write(0, 0, "Insufficient data for time series")
            return
        
        # Determine time column
        time_col = "datetime" if "datetime" in df.columns else "timestamp"
        
        # Sort by time
        df_sorted = df.sort_values(by=time_col).copy()
        
        # Convert timestamps to elapsed seconds from start for better display
        if time_col == "timestamp":
            first_time = df_sorted[time_col].iloc[0]
            df_sorted["elapsed"] = df_sorted[time_col] - first_time
        else:
            df_sorted["elapsed"] = range(len(df_sorted))
        
        # Select key channels (subset for clarity)
        key_channels = ["raw_F1", "raw_F4", "raw_F8"]  # Violet, Green, Red
        available_channels = [ch for ch in key_channels if ch in df.columns]
        
        if not available_channels:
            available_channels = spectral_columns[:3]  # Take first 3 channels
        
        # Write headers
        worksheet.write(0, 0, "Time")
        for i, channel in enumerate(available_channels):
            worksheet.write(0, i + 1, channel.replace("raw_", ""))
        
        # Write data
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            worksheet_row = i + 1
            worksheet.write(worksheet_row, 0, row["elapsed"])
            for j, channel in enumerate(available_channels):
                worksheet.write(worksheet_row, j + 1, row[channel])
        
        # Create a chart
        chart = workbook.add_chart({'type': 'line'})
        
        # Add a series for each channel
        for i, channel in enumerate(available_channels):
            chart.add_series({
                'name': channel.replace("raw_", ""),
                'categories': ['Time Series', 1, 0, len(df_sorted), 0],
                'values': ['Time Series', 1, i + 1, len(df_sorted), i + 1],
            })
        
        # Add chart labels
        chart.set_title({'name': 'Signal Intensity Over Time'})
        chart.set_x_axis({'name': 'Time (elapsed seconds)'})
        chart.set_y_axis({'name': 'Signal Value'})
        
        # Insert the chart into the worksheet
        worksheet.insert_chart('E2', chart, {'x_scale': 1.5, 'y_scale': 1.5})
    
    def _add_ratio_analysis_sheet(self, workbook, writer, df, metadata):
        """
        Add a ratio analysis chart to an Excel workbook.
        
        Args:
            workbook: xlsxwriter workbook
            writer: pandas ExcelWriter
            df: DataFrame containing the measurements
            metadata: Session metadata
        """
        # Create a new worksheet
        worksheet = workbook.add_worksheet('Ratio Analysis')
        
        # Check if we have ratio data
        ratio_columns = [col for col in df.columns if col.startswith("ratio_")]
        if not ratio_columns:
            worksheet.write(0, 0, "No ratio data available")
            return
        
        # Write headers
        worksheet.write(0, 0, "Ratio")
        worksheet.write(0, 1, "Mean")
        worksheet.write(0, 2, "Min")
        worksheet.write(0, 3, "Max")
        worksheet.write(0, 4, "Std Dev")
        
        # Calculate statistics for each ratio
        ratio_stats = df[ratio_columns].describe().transpose()
        
        # Write statistics
        for i, (ratio, row) in enumerate(ratio_stats.iterrows()):
            worksheet_row = i + 1
            worksheet.write(worksheet_row, 0, ratio.replace("ratio_", ""))
            worksheet.write(worksheet_row, 1, row["mean"])
            worksheet.write(worksheet_row, 2, row["min"])
            worksheet.write(worksheet_row, 3, row["max"])
            worksheet.write(worksheet_row, 4, row["std"])
        
        # Create a chart for ratio means
        chart = workbook.add_chart({'type': 'column'})
        
        # Add the series
        chart.add_series({
            'name': 'Mean Ratio Values',
            'categories': ['Ratio Analysis', 1, 0, len(ratio_stats), 0],
            'values': ['Ratio Analysis', 1, 1, len(ratio_stats), 1],
            'data_labels': {'value': True}
        })
        
        # Add chart labels
        chart.set_title({'name': 'Mean Ratio Values'})
        chart.set_x_axis({'name': 'Ratio'})
        chart.set_y_axis({'name': 'Value'})
        
        # Insert the chart into the worksheet
        worksheet.insert_chart('G2', chart, {'x_scale': 1.2, 'y_scale': 1.2})


# Create a global instance for convenience
data_exporter = DataExporter()