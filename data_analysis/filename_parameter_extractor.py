import re
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import argparse


class FilenameParser:
    """
    Parser class for extracting parameters from measurement filenames.
    """
    
    def __init__(self):
        # Define regex patterns for different parameters
        self.patterns = {
            'date': r'(\d{6})_',
            'current': r'I(\d+)mA',
            'gain': r'G(\d+)_',
            'integration_time': r'IT(\d+)ms',
            'background_sub_yes': r'(?<!no_)(bckgnd_sub)',  # positive lookbehind to exclude "no_bckgnd_sub"
            'background_sub_no': r'(no_bckgnd_sub)',
            'bead_size': r'(\d+\.?\d*)um_beads',
            'solids_concentration': r'(\d+\.?\d*)_soldids',
            'salt_concentration': r'(\d+|no)_salt',
            'time_point': r'(\d+)min\.csv'
        }
    
    def parse_filename(self, filename: str) -> Dict[str, Union[str, float, int, bool]]:
        """
        Extract all parameters from a single filename.
        
        Args:
            filename (str): The filename to parse
            
        Returns:
            dict: Dictionary containing extracted parameters
        """
        params = {'filename': filename}
        
        # Extract date
        date_match = re.search(self.patterns['date'], filename)
        if date_match:
            date_str = date_match.group(1)
            params['date'] = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
            params['date_raw'] = date_str
        else:
            params['date'] = None
            params['date_raw'] = None
        
        # Extract LED current (mA)
        current_match = re.search(self.patterns['current'], filename)
        params['led_current_mA'] = int(current_match.group(1)) if current_match else None
        
        # Extract gain
        gain_match = re.search(self.patterns['gain'], filename)
        params['gain'] = int(gain_match.group(1)) if gain_match else None
        
        # Extract integration time (ms)
        it_match = re.search(self.patterns['integration_time'], filename)
        params['integration_time_ms'] = int(it_match.group(1)) if it_match else None
        
        # Check background subtraction status
        bg_yes_match = re.search(self.patterns['background_sub_yes'], filename)
        bg_no_match = re.search(self.patterns['background_sub_no'], filename)
        
        if bg_no_match:
            params['background_subtracted'] = False
        elif bg_yes_match:
            params['background_subtracted'] = True
        else:
            params['background_subtracted'] = None  # Cannot determine from filename
        
        # Extract bead size (μm)
        bead_match = re.search(self.patterns['bead_size'], filename)
        params['bead_size_um'] = float(bead_match.group(1)) if bead_match else None
        
        # Extract solids concentration (%)
        solids_match = re.search(self.patterns['solids_concentration'], filename)
        params['solids_concentration_percent'] = float(solids_match.group(1)) if solids_match else None
        
        # Extract salt concentration (%)
        salt_match = re.search(self.patterns['salt_concentration'], filename)
        if salt_match:
            salt_val = salt_match.group(1)
            if salt_val.lower() == 'no':
                params['salt_concentration_percent'] = 0.0
            else:
                params['salt_concentration_percent'] = float(salt_val)
        else:
            params['salt_concentration_percent'] = None
        
        # Extract time point (minutes)
        time_match = re.search(self.patterns['time_point'], filename)
        params['time_point_min'] = int(time_match.group(1)) if time_match else None
        
        # Create a condition identifier for grouping
        if all(x is not None for x in [params['solids_concentration_percent'], 
                                      params['salt_concentration_percent']]):
            params['condition_id'] = f"{params['solids_concentration_percent']}%_solids_{params['salt_concentration_percent']}%_salt"
        else:
            params['condition_id'] = None
        
        return params
    
    def parse_multiple_filenames(self, filenames: List[str]) -> pd.DataFrame:
        """
        Parse multiple filenames and return as DataFrame.
        
        Args:
            filenames (list): List of filenames to parse
            
        Returns:
            pd.DataFrame: DataFrame with extracted parameters
        """
        all_params = []
        for filename in filenames:
            params = self.parse_filename(filename)
            all_params.append(params)
        
        return pd.DataFrame(all_params)
    
    def parse_directory(self, directory_path: str, pattern: str = "*.csv") -> pd.DataFrame:
        """
        Parse all matching files in a directory.
        
        Args:
            directory_path (str): Path to directory containing files
            pattern (str): File pattern to match (default: "*.csv")
            
        Returns:
            pd.DataFrame: DataFrame with extracted parameters
        """
        path = Path(directory_path)
        filenames = [f.name for f in path.glob(pattern)]
        return self.parse_multiple_filenames(filenames)
    
    def get_experiment_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate a summary of experimental conditions from parsed data.
        
        Args:
            df (pd.DataFrame): DataFrame from parse_multiple_filenames
            
        Returns:
            dict: Summary statistics
        """
        summary = {}
        
        # Count unique values for each parameter
        for col in df.columns:
            if col not in ['filename', 'date', 'date_raw', 'condition_id']:
                unique_vals = df[col].dropna().unique()
                summary[col] = {
                    'unique_values': sorted(unique_vals.tolist()) if len(unique_vals) > 0 else [],
                    'count': len(unique_vals)
                }
        
        # Group by experimental conditions
        if 'condition_id' in df.columns:
            condition_groups = df.groupby('condition_id').agg({
                'time_point_min': list,
                'filename': 'count'
            }).to_dict('index')
            summary['experimental_conditions'] = condition_groups
        
        return summary


if __name__ == "__main__":

    """Example: Parse multiple filenames"""
    parser = FilenameParser()

    filenames = [
        r"test_data/250601_I20mA_G512_IT300ms_no_bckgnd_sub_1um_beads_0.02_soldids_no_salt_5min.csv",
        r"test_data/250601_I20mA_G512_IT300ms_no_bckgnd_sub_1um_beads_0.02_soldids_no_salt_0min.csv",
        r"test_data/250601_I20mA_G512_IT300ms_no_bckgnd_sub_1um_beads_0.01_soldids_50_salt_5min.csv"
    ]
    
    df = parser.parse_multiple_filenames(filenames)

    print(parser.get_experiment_summary(df))
    
    print("Multiple Files Parsing Example:")
    print("=" * 40)
    print(df[['filename', 'bead_size_um', 'solids_concentration_percent', 
              'salt_concentration_percent', 'time_point_min']].to_string(index=False))
