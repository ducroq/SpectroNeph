import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

# AS7341 approximate center wavelengths (nm)
wavelengths = {
    'F1': 415,   # Violet
    'F2': 445,   # Blue  
    'F3': 480,   # Cyan
    'F4': 515,   # Green
    'F5': 555,   # Yellow-green
    'F6': 590,   # Yellow
    'F7': 630,   # Orange
    'F8': 680    # Far-red
}

combined_correction_factors = {
    'F1': 61.2,   # Highest correction (weak LED + weak AS7341)
    'F2': 3.8,    # Moderate correction  
    'F3': 6.9,    # Moderate correction
    'F4': 2.1,    # Low correction (good LED + best AS7341)
    'F5': 1.0,    # Reference (normalized to F5)
    'F6': 2.2,    # Low correction
    'F7': 7.1,    # Moderate correction (weak LED in red)
    'F8': 10.8    # High correction (very weak LED + moderate AS7341)
}

def parse_side_scatter_filename(filename):
    """
    Parse side_scatter filename format with bracketed conditions.
    
    Expected format: side_scatter_{bead_type}_{buffer_type}_[condition]_{test_nr}.csv
    Examples: 
    - side_scatter_0.45micron_0.3%_solids_0.05%_tween_[blanco]_0.csv
    - side_scatter_0.45micron_0.3%_solids_0.05%_tween_[salt_3.6_mg_per_mL]_0.csv
    
    :param filename: The filename to parse
    :return: Dictionary with parsed components
    """
    # Remove file extension and 'side_scatter_' prefix
    base_name = filename.replace('.csv', '').replace('side_scatter_', '')
    
    # Initialize result dictionary
    result = {
        'bead_type': None,
        'buffer_type': None,
        'condition': None,
        'test_nr': None,
        'raw_filename': filename
    }
    
    try:
        # Extract condition from brackets first
        bracket_match = re.search(r'\[([^\]]+)\]', base_name)
        if bracket_match:
            result['condition'] = bracket_match.group(1)
            # Remove the bracketed part from the string for further parsing
            base_name_no_condition = re.sub(r'_\[[^\]]+\]', '', base_name)
        else:
            print(f"Warning: No bracketed condition found in {filename}")
            result['condition'] = 'unknown'
            base_name_no_condition = base_name
        
        # Split remaining parts by underscores
        parts = base_name_no_condition.split('_')
        
        # Last part should be test number
        if len(parts) >= 1:
            try:
                result['test_nr'] = int(parts[-1])
                parts = parts[:-1]  # Remove test_nr from parts
            except ValueError:
                print(f"Warning: Could not parse test number from {parts[-1]} in {filename}")
                result['test_nr'] = 0
        
        # Now we need to split the remaining parts into bead_type and buffer_type
        # Bead descriptions typically contain "micron" and end with "solids"
        bead_end_index = None
        for i, part in enumerate(parts):
            if 'solids' in part.lower():
                bead_end_index = i + 1
                break
        
        if bead_end_index is None:
            # Fallback: look for micron as indicator of bead description
            for i, part in enumerate(parts):
                if 'micron' in part.lower():
                    # Assume bead description continues for next few parts
                    # Look ahead for % and solids
                    search_end = min(i + 4, len(parts))
                    for j in range(i + 1, search_end):
                        if j < len(parts) and 'solids' in parts[j].lower():
                            bead_end_index = j + 1
                            break
                    if bead_end_index is None:
                        # Default assumption: micron + 2 more parts
                        bead_end_index = min(i + 3, len(parts))
                    break
        
        if bead_end_index is None:
            # Last fallback: assume first half is bead, second half is buffer
            bead_end_index = len(parts) // 2
        
        # Extract bead_type and buffer_type
        if bead_end_index > 0:
            result['bead_type'] = '_'.join(parts[:bead_end_index])
        
        if bead_end_index < len(parts):
            result['buffer_type'] = '_'.join(parts[bead_end_index:])
        
        # Clean up any empty strings
        for key in ['bead_type', 'buffer_type', 'condition']:
            if result[key] == '':
                result[key] = None
            
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        # Set reasonable defaults
        if result['condition'] is None:
            result['condition'] = 'unknown'
        if result['test_nr'] is None:
            result['test_nr'] = 0
    
    return result

def extract_salt_concentration(condition_str):
    """
    Extract salt concentration from condition string.
    
    :param condition_str: The condition string from brackets
    :return: Tuple of (has_salt, concentration_value, concentration_unit)
    """
    if not condition_str:
        return False, 0, None
    
    condition_lower = condition_str.lower()
    
    # Look for patterns like "salt_3.6_mg_per_ml" or "salt_0_mg_per_ml"
    salt_pattern = r'salt[_\s]*(\d+(?:\.\d+)?)[_\s]*(mg|g)[_\s]*per[_\s]*(ml|l)'
    match = re.search(salt_pattern, condition_lower)
    
    if match:
        concentration = float(match.group(1))
        unit = f"{match.group(2)}_per_{match.group(3)}"
        return True, concentration, unit
    
    # Also check for just "salt" without specific concentration
    if 'salt' in condition_lower and 'nacl' not in condition_lower:
        return True, None, None
    
    return False, 0, None

def assign_structured_label(parsed_info):
    """
    Assign a structured label based on parsed filename information.
    Uses the exact text within brackets as the primary label.
    
    :param parsed_info: Dictionary from parse_side_scatter_filename
    :return: Tuple of (primary_label, detailed_label)
    """
    condition = parsed_info.get('condition', 'unknown')
    if not condition:
        condition = 'unknown'
        
    bead_type = parsed_info.get('bead_type', '') or ''
    buffer_type = parsed_info.get('buffer_type', '') or ''
    
    # Use the exact bracketed condition as the primary label
    primary_label = condition
    
    # Detailed label includes all components
    detailed_label = f"{primary_label}_{bead_type}_{buffer_type}".replace('__', '_')
    
    return primary_label, detailed_label

def safe_divide(a, b, default=0):
    """Safe division avoiding divide by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        return np.where(np.isfinite(result), result, default)
    
def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the data with label columns added.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"  Loaded {len(df)} measurements from {Path(file_path).name}")
    except Exception as e:
        print(f"  Error loading file {file_path}: {e}")
        return None
    
    if df.empty:
        print("DataFrame is empty, no data to load.")
        return pd.DataFrame()
    
    df2 = df[['timestamp']].copy()  
    
    # Parse filename and extract structured information
    filename = Path(file_path).name
    parsed_info = parse_side_scatter_filename(filename)
    primary_label, detailed_label = assign_structured_label(parsed_info)
    
    # Add all extracted information as columns
    df2['filename'] = filename
    df2['label'] = primary_label
    df2['detailed_label'] = detailed_label
    df2['bead_type'] = parsed_info['bead_type']
    df2['buffer_type'] = parsed_info['buffer_type']
    df2['condition'] = parsed_info['condition']
    df2['test_nr'] = parsed_info['test_nr']
    
    # Extract salt concentration info
    has_salt, salt_conc, salt_unit = extract_salt_concentration(parsed_info['condition'])
    df2['has_salt'] = has_salt
    df2['salt_concentration'] = salt_conc
    df2['salt_unit'] = salt_unit
    
    print(f"  Parsed info: bead_type={parsed_info['bead_type']}, buffer_type={parsed_info['buffer_type']}")
    print(f"  Condition: {parsed_info['condition']}, Test nr: {parsed_info['test_nr']}")
    print(f"  Salt info: has_salt={has_salt}, concentration={salt_conc}, unit={salt_unit}")
    print(f"  Assigned label: {primary_label}")

    # Load raw measurements   
    for ch in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']:
        if f'raw_{ch}' in df.columns:
            df2[f'{ch}'] = df[f'raw_{ch}']  # * combined_correction_factors[ch]
        else:
            print(f"Warning: Raw channel {ch} not found in data")
    for ch in ['Clear', 'NIR']:
        if f'raw_{ch}' in df.columns:
            df2[f'{ch}'] = df[f'raw_{ch}']

    return df2

def load_multiple_files(file_paths):
    """
    Load and combine multiple CSV files, filtering for side_scatter files only.
    
    :param file_paths: List of file paths to load
    :return: Combined DataFrame with all data
    """
    all_data = []
    
    # Filter for side_scatter files only
    side_scatter_files = [fp for fp in file_paths if 'side_scatter' in Path(fp).name.lower()]
    
    print(f"Found {len(side_scatter_files)} side_scatter files out of {len(file_paths)} total files")
    
    for file_path in side_scatter_files:
        print(f"Processing: {file_path}")
        df = load_data(file_path)
        if df is not None and not df.empty:
            all_data.append(df)
    
    if not all_data:
        print("No valid side_scatter data files found!")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Print label distribution
    print("\nPrimary label distribution:")
    print(combined_df['label'].value_counts())
    
    print("\nCondition distribution:")
    print(combined_df['condition'].value_counts())
    
    print("\nSalt concentration distribution:")
    salt_summary = combined_df.groupby(['has_salt', 'salt_concentration', 'salt_unit']).size().reset_index(name='count')
    print(salt_summary.to_string(index=False))
    
    print("\nBead type distribution:")
    print(combined_df['bead_type'].value_counts())
    
    print("\nBuffer type distribution:")
    print(combined_df['buffer_type'].value_counts())
    
    print("\nTest conditions summary:")
    test_summary = combined_df.groupby(['bead_type', 'buffer_type', 'condition'])['test_nr'].agg(['count', 'min', 'max']).reset_index()
    print(test_summary.to_string(index=False))
    
    return combined_df

def compute_features(df):
    """
    Compute features from the DataFrame.
    
    :param df: DataFrame containing the data.
    :return: DataFrame with computed features.
    """
    if df.empty:
        print("DataFrame is empty, no features to compute.")
        return pd.DataFrame()

    print(f"Computing features for {len(df)} measurements...")
    
    # Fix performance warning by creating new columns as a dictionary first
    new_columns = {}
    
    # compute pairwise parameters of channels
    for ch in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Clear']:
        for ch1 in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Clear']:
            if ch != ch1:
                # pairwise ratios
                ratios = safe_divide(df[ch], df[ch1], default=0)
                new_columns[f'{ch}_to_{ch1}_ratio'] = ratios
                # log ratios
                new_columns[f'log_{ch}_to_{ch1}_ratio'] = np.log(ratios + 1e-10)
                # squared ratios
                new_columns[f'{ch}_to_{ch1}_ratio_squared'] = ratios ** 2
    
    # Add the new columns to the dataframe
    for col_name, col_data in new_columns.items():
        df[col_name] = col_data
    
    # Compute spectral sums
    df['total_sum'] = df['F1'] + df['F2'] + df['F3'] + df['F4'] + df['F5'] + df['F6'] + df['F7'] + df['F8']
    df['short_sum'] = df['F1'] + df['F2'] + df['F3'] # Blue-violet region
    df['mid_sum'] = df['F4'] + df['F5']              # Green region  
    df['long_sum'] = df['F6'] + df['F7'] + df['F8']  # Yellow-red region
    df['edges_sum'] = df['F1'] + df['F8']            # Spectral edges
    df['center_sum'] = df['F4'] + df['F5']           # Spectral center    

    # Sum ratios
    df['short_fraction'] = safe_divide(df['short_sum'], df['total_sum'])
    df['mid_fraction'] = safe_divide(df['mid_sum'], df['total_sum'])
    df['long_fraction'] = safe_divide(df['long_sum'], df['total_sum'])
    df['edges_fraction'] = safe_divide(df['edges_sum'], df['total_sum'])
    
    # Cross-region ratios
    df['short_long_sum_ratio'] = safe_divide(df['short_sum'], df['long_sum'])
    df['center_edges_sum_ratio'] = safe_divide(df['center_sum'], df['edges_sum'])


    # Spectral centroid (weighted average wavelength)
    _wavelengths = np.array([wavelengths[f'F{i}'] for i in range(1, 9)])
    intensities = np.array([df[f'F{i}'] for i in range(1, 9)])    
   
    # Ensure no negative values
    intensities = np.maximum(intensities, 0)
    
    weighted_sum = np.sum(_wavelengths[:, np.newaxis] * intensities, axis=0)
    total_intensity = np.sum(intensities, axis=0)

    # Avoid division by zero
    df['spectral_centroid'] = safe_divide(weighted_sum, total_intensity, default=np.mean(_wavelengths))
    
    # Spectral variance (measure of spectral width)
    centroid_expanded = df['spectral_centroid'].values[np.newaxis, :]
    variance_numerator = np.sum(intensities * (_wavelengths[:, np.newaxis] - centroid_expanded)**2, axis=0)
    df['spectral_variance'] = safe_divide(variance_numerator, total_intensity, default=0)
    
    # Avoid sqrt of negative numbers
    variance_clean = np.maximum(df['spectral_variance'], 0)
    df['spectral_width'] = np.sqrt(variance_clean)

    # Spectral asymmetry (skewness indicator)
    centroid_values = df['spectral_centroid'].values
    variance_values = df['spectral_variance'].values

    skew_numerator = np.sum(intensities * (_wavelengths[:, np.newaxis] - centroid_values)**3, axis=0)
    skew_denominator = total_intensity * (variance_values + 1e-10)**1.5
    df['spectral_skewness'] = safe_divide(skew_numerator, skew_denominator, default=0)

    print(f"Feature computation complete. Total columns: {len(df.columns)}")
    return df

def generate_summary_report(df):
    """
    Generate a comprehensive summary report of the experimental data.
    
    :param df: DataFrame with features and metadata
    :return: Dictionary containing summary statistics
    """
    print("\nGenerating summary report...")
    
    summary = {}
    
    # Basic statistics
    summary['total_measurements'] = len(df)
    summary['unique_conditions'] = df['condition'].nunique()
    summary['unique_bead_types'] = df['bead_type'].nunique()
    summary['unique_buffer_types'] = df['buffer_type'].nunique()
    
    # Condition breakdown
    summary['condition_counts'] = df['condition'].value_counts().to_dict()
    summary['label_counts'] = df['label'].value_counts().to_dict()
    
    # Salt concentration analysis
    summary['salt_concentration_counts'] = df['salt_concentration'].value_counts().to_dict()
    summary['has_salt_counts'] = df['has_salt'].value_counts().to_dict()
    
    # Test replication analysis
    replication_summary = df.groupby(['bead_type', 'buffer_type', 'condition']).agg({
        'test_nr': ['count', 'min', 'max'],
        'total_sum': ['mean', 'std'],
        'spectral_centroid': ['mean', 'std']
    }).round(3)
    
    summary['replication_summary'] = replication_summary
    
    # Feature statistics
    feature_cols = [col for col in df.columns if col not in 
                   ['timestamp', 'filename', 'label', 'detailed_label', 'bead_type', 
                    'buffer_type', 'condition', 'test_nr', 'has_salt', 'salt_concentration', 'salt_unit']]
    
    summary['total_features'] = len(feature_cols)
    summary['spectral_features'] = len([col for col in feature_cols if any(f'F{i}' in col for i in range(1, 9))])
    summary['ratio_features'] = len([col for col in feature_cols if '_ratio' in col])
    
    return summary

def save_results(df, summary, output_prefix='side_scatter_analysis'):
    """
    Save analysis results to multiple files.
    
    :param df: DataFrame with features and metadata
    :param summary: Summary dictionary from generate_summary_report
    :param output_prefix: Prefix for output filenames
    """
    print(f"\nSaving results with prefix: {output_prefix}")
    
    # 1. Save full feature dataset
    features_file = f"{output_prefix}_features.csv"
    df.to_csv(features_file, index=False)
    print(f"✓ Features saved to: {features_file}")
    
    # 2. Save experimental conditions summary
    conditions_file = f"{output_prefix}_conditions.csv"
    conditions_df = df.groupby(['bead_type', 'buffer_type', 'condition', 'salt_concentration']).agg({
        'test_nr': ['count', 'min', 'max'],
        'label': 'first',
        'total_sum': ['mean', 'std'],
        'spectral_centroid': ['mean', 'std'],
        'short_long_sum_ratio': ['mean', 'std']
    }).round(3)
    conditions_df.to_csv(conditions_file)
    print(f"✓ Conditions summary saved to: {conditions_file}")
    
    # 3. Save metadata only (for quick reference)
    metadata_file = f"{output_prefix}_metadata.csv"
    metadata_cols = ['filename', 'label', 'detailed_label', 'bead_type', 'buffer_type', 
                    'condition', 'test_nr', 'has_salt', 'salt_concentration', 'salt_unit']
    df[metadata_cols].to_csv(metadata_file, index=False)
    print(f"✓ Metadata saved to: {metadata_file}")
    
    # 4. Save summary statistics
    summary_file = f"{output_prefix}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=== Side Scatter Analysis Summary ===\n\n")
        f.write(f"Total measurements: {summary['total_measurements']}\n")
        f.write(f"Unique conditions: {summary['unique_conditions']}\n")
        f.write(f"Unique bead types: {summary['unique_bead_types']}\n")
        f.write(f"Unique buffer types: {summary['unique_buffer_types']}\n")
        f.write(f"Total features extracted: {summary['total_features']}\n")
        f.write(f"Spectral features: {summary['spectral_features']}\n")
        f.write(f"Ratio features: {summary['ratio_features']}\n\n")
        
        f.write("=== Condition Distribution ===\n")
        for condition, count in summary['condition_counts'].items():
            f.write(f"{condition}: {count}\n")
        
        f.write("\n=== Label Distribution ===\n")
        for label, count in summary['label_counts'].items():
            f.write(f"{label}: {count}\n")
        
        f.write("\n=== Salt Concentration Distribution ===\n")
        for conc, count in summary['salt_concentration_counts'].items():
            f.write(f"{conc} mg/mL: {count}\n")
        
        f.write("\n=== Has Salt Distribution ===\n")
        for has_salt, count in summary['has_salt_counts'].items():
            f.write(f"{has_salt}: {count}\n")
    
    print(f"✓ Summary saved to: {summary_file}")
    
    return {
        'features_file': features_file,
        'conditions_file': conditions_file,
        'metadata_file': metadata_file,
        'summary_file': summary_file
    }

# Testing and example usage
if __name__ == "__main__":
    # # Test the parsing function with bracketed format
    # test_files = [
    #     "side_scatter_0.45micron_0.3%_solids_0.05%_tween_[salt_0_mg_per_mL]_0.csv",
    #     "side_scatter_0.45micron_0.3%_solids_0.05%_tween_[blanco]_0.csv",
    #     "side_scatter_0.45micron_0.3%_solids_0.05%_tween_[salt_3.6_mg_per_mL]_0.csv",
    #     "side_scatter_1.0micron_0.2%_solids_0.1%_tween_[salt_agglutination]_1.csv",
    #     "side_scatter_0.5micron_running_buffer_[CRP_detection]_2.csv",
    #     "side_scatter_0.1micron_0.5%_solids_PBS_[negative_control]_3.csv"
    # ]
    
    # print("Testing improved parsing of side_scatter filenames:")
    # for filename in test_files:
    #     parsed_info = parse_side_scatter_filename(filename)
    #     primary_label, detailed_label = assign_structured_label(parsed_info)
    #     has_salt, salt_conc, salt_unit = extract_salt_concentration(parsed_info['condition'])
        
    #     print(f"\nFilename: {filename}")
    #     print(f"  Bead type: {parsed_info['bead_type']}")
    #     print(f"  Buffer type: {parsed_info['buffer_type']}")
    #     print(f"  Condition: {parsed_info['condition']}")
    #     print(f"  Test nr: {parsed_info['test_nr']}")
    #     print(f"  Has salt: {has_salt}, Concentration: {salt_conc}, Unit: {salt_unit}")
    #     print(f"  Primary label: {primary_label}")
    
    # Example of how to use with your data directory
    data_directory = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250702 WFBR experiments"

    if os.path.exists(data_directory):
        # Get all CSV files
        csv_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) 
                    if f.endswith('.csv') and not 'summary' in f.lower()]
        
        print(f"\nFound {len(csv_files)} CSV files")
        
        # Load and process all data (will automatically filter for side_scatter files)
        combined_df = load_multiple_files(csv_files)
        
        # Full analysis:
        if not combined_df.empty:
            # Compute features
            features_df = compute_features(combined_df)
            
            # Generate summary report
            summary_report = generate_summary_report(features_df)
            
            # Save all results
            output_files = save_results(features_df, summary_report)
            
            print(f"\n=== Analysis Complete ===")
            print(f"Processed {len(features_df)} measurements")
            print(f"Extracted {summary_report['total_features']} features")
            print(f"Found {summary_report['unique_conditions']} unique conditions")
            print("\nOutput files created:")
            for file_type, filename in output_files.items():
                print(f"  {file_type}: {filename}")
                
    else:
        print(f"Data directory not found: {data_directory}")
        print("Please update the data_directory variable with the correct path.")