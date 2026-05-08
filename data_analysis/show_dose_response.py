import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kinetics_analyser import analyze_all_channels, plot_kinetics_summary
from collections import defaultdict

def process_data_files(path, filenames, wavelengths):
    """
    Process CSV files and extract features
    """
    data = []
    
    for csv_file in filenames:
        print(f"{csv_file}")
        df = pd.read_csv(path / csv_file)
        
        # Your existing processing
        start_time = df['timestamp'].iloc[0]
        df['time_seconds'] = df['timestamp'] - start_time
        df['total_sum'] = df['raw_F1'] + df['raw_F2'] + df['raw_F3'] + df['raw_F4'] + df['raw_F5'] + df['raw_F6'] + df['raw_F7'] + df['raw_F8']
        df['short_sum'] = df['raw_F1'] + df['raw_F2'] + df['raw_F3'] # Blue-violet region
        df['long_sum'] = df['raw_F6'] + df['raw_F7'] + df['raw_F8'] # Yellow-red region
        df['short_long_sum_ratio'] = df['short_sum'] / df['long_sum']
        df['F3_to_F7_ratio'] = df['raw_F3'] / df['raw_F7']
        df['F5_to_F7_ratio'] = df['raw_F5'] / df['raw_F7']
       
        # Calculate spectral centroid and variance
        intensities = np.array([df[f'raw_F{i}'] for i in range(1, 9)])
        _wavelengths = np.array([wavelengths[f'F{i}'] for i in range(1, 9)])
        total_intensity = np.sum(intensities, axis=0)    
        weighted_sum = np.sum(_wavelengths[:, np.newaxis] * intensities, axis=0)
        df['spectral_centroid'] = weighted_sum / total_intensity
        centroid_expanded = df['spectral_centroid'].values[np.newaxis, :]
        df['spectral_variance'] = np.sum(intensities * (_wavelengths[:, np.newaxis] - centroid_expanded)**2, axis=0) / total_intensity
        df['spectral_width'] = np.sqrt(np.maximum(df['spectral_variance'], 0))  # Avoid sqrt of negative numbers
        
        # Extract content between square brackets
        match = re.search(r'\[(.*?)\]', csv_file)    
        label = match.group(1) if match else 'Unknown'
        
        # Feature extraction
        features = [df['spectral_centroid'].iloc[-2], df['spectral_width'].iloc[-2],
                    df['spectral_centroid'].iloc[0] - df['spectral_centroid'].iloc[-2],
                    df['F3_to_F7_ratio'].iloc[-2], df['F5_to_F7_ratio'].iloc[-2],
                    df['short_long_sum_ratio'].iloc[-2]]
        
        data.append((label, features))
    
    return data

def create_boxplot_visualization(data):
    """
    Create box plots for feature comparison
    """
    feature_names = ['Spectral Centroid (nm)', 'Spectral Width (nm)', 'Spectral Shift (nm)', 
                    'F3/F7 Ratio', 'F5/F7 Ratio', 'Short/Long Ratio']
    
    # Organize data by concentration
    conc_data = defaultdict(lambda: defaultdict(list))
    for label, features in data:
        for i, feature_val in enumerate(features):
            conc_data[label][i].append(float(feature_val))
    
    # Create subplots for each feature
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Sort concentrations by numerical value
    concentrations = sorted(conc_data.keys(), key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
    
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        
        # Prepare data for box plot
        box_data = []
        labels = []
        
        for conc in concentrations:
            box_data.append(conc_data[conc][i])
            # Clean up label for display
            clean_label = conc.replace('CRP_', '').replace('_ng_per_mL', '\nng/mL')
            labels.append(clean_label)
        
        # Create box plot
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        
        # Color the boxes with a gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(concentrations)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(feature_name, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, alpha=0.3)
        
        # Add mean values as red dots
        for j, conc in enumerate(concentrations):
            mean_val = np.mean(conc_data[conc][i])
            ax.plot(j+1, mean_val, 'ro', markersize=6)
    
    plt.tight_layout()
    plt.suptitle('Feature Analysis Across Concentrations', fontsize=16, y=1.02)
    return fig

def create_feature_trends(data, log_scale=False):
    """
    Create a line plot showing feature trends across concentrations
    """
    feature_names = ['Spectral Centroid [nm]', 'Spectral Width [nm]', 'Spectral Shift [nm]', 
                    'F3/F7 Ratio', 'F5/F7 Ratio', 'Short/Long Ratio']
    
    # Calculate means for each concentration
    conc_means = defaultdict(lambda: defaultdict(list))
    for label, features in data:
        for i, feature_val in enumerate(features):
            conc_means[label][i].append(float(feature_val))
    
    # Function to extract concentration from label
    def extract_concentration(label):
        import re
        # Try mg_per_mL first
        match = re.search(r'(\w+)_([0-9]+\.?[0-9]*)_mg_per_mL', label)
        if match:
            return float(match.group(2))
        # Try ng_per_mL
        match = re.search(r'(\w+)_([0-9]+\.?[0-9]*)_ng_per_mL', label)
        if match:
            return float(match.group(2))
        # Fallback: try to find any number in the label
        numbers = re.findall(r'[0-9]+\.?[0-9]*', label)
        return float(numbers[0]) if numbers else 0
    
    # Sort concentrations by numerical value
    concentrations = sorted(conc_means.keys(), key=extract_concentration)
    conc_values = [extract_concentration(c) for c in concentrations]
    
    # Determine units from the first concentration label
    sample_label = concentrations[0] if concentrations else ""
    if 'mg_per_mL' in sample_label:
        unit_label = 'Concentration (mg/mL)'
    elif 'ng_per_mL' in sample_label:
        unit_label = 'Concentration (ng/mL)'
    else:
        unit_label = 'Concentration'

    print("Concentrations:", conc_values)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        
        means = []
        stds = []
        
        for conc in concentrations:
            feature_values = conc_means[conc][i]
            means.append(np.mean(feature_values))
            stds.append(np.std(feature_values))
        
        # Plot with error bars
        ax.errorbar(conc_values, means, yerr=stds, marker='o', linewidth=2, 
                   markersize=8, capsize=5, capthick=2, alpha=0.7, label=feature_name)
        
        ax.set_title(feature_name, fontsize=12, fontweight='bold')
        ax.set_xlabel(unit_label)

        # Use log scale you have very wide ranges
        if log_scale:
            ax.set_xscale('symlog')  # Uncomment if needed for very wide concentration ranges

        ax.grid(True, alpha=0.3)
        
        # Add individual data points
        for j, conc in enumerate(concentrations):
            feature_values = conc_means[conc][i]
            x_vals = [conc_values[j]] * len(feature_values)
            ax.scatter(x_vals, feature_values, alpha=0.6, s=30)
    
    plt.tight_layout()
    plt.suptitle('Feature Trends vs Concentration', fontsize=16, y=1.02)
    return fig

def print_summary_statistics(data):
    """
    Print summary statistics for each feature
    """
    feature_names = ['Spectral Centroid [nm]', 'Spectral Width [nm]', 'Spectral Shift [nm]', 
                    'F3/F7 Ratio', 'F5/F7 Ratio', 'Short/Long Ratio']
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Organize data by concentration
    conc_data = defaultdict(lambda: defaultdict(list))
    for label, features in data:
        for i, feature_val in enumerate(features):
            conc_data[label][i].append(float(feature_val))
    
    concentrations = sorted(conc_data.keys(), key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
    
    for i, feature_name in enumerate(feature_names):
        print(f"\n{feature_name}:")
        print("-" * len(feature_name))
        
        for conc in concentrations:
            values = conc_data[conc][i]
            mean_val = np.mean(values)
            std_val = np.std(values)
            clean_conc = conc.replace('CRP_', '').replace('_ng_per_mL', ' ng/mL')
            print(f"  {clean_conc:12}: {mean_val:8.3f} ± {std_val:.3f}")

# directory_path = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250604 WFBR experiments\perpendicular_spectrometer"
directory_path = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250702 WFBR experiments"
path = Path(directory_path)

pattern = '*CRP*.csv'
pattern = '*salt*.csv'
filenames = [f.name for f in path.glob(pattern) if 'side_scatter' in f.name and 'micro_cuvette' not in f.name]

print("Files matching pattern:", pattern)

# AS7341 approximate center wavelengths (nm)
channels = ['Clear', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'NIR']

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

# Process the data
data = process_data_files(path, filenames, wavelengths)

# # Create visualizations
# fig1 = create_boxplot_visualization(data)
# plt.show()

fig2 = create_feature_trends(data, log_scale=False)
plt.show()

# # Print summary statistics
# print_summary_statistics(data)