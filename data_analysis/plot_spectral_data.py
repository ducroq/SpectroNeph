import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kinetics_analyser import analyze_all_channels, plot_kinetics_summary

def plot_spectral_data(csv_file):
    """
    Plot spectral data relative to start and normalized to first measurement and clear channel.
    
    Parameters:
    csv_file (str): Path to the CSV file
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
     # Convert timestamp to relative time in seconds
    start_time = df['timestamp'].iloc[0]
    df['time_seconds'] = df['timestamp'] - start_time
    
    # Define the channels to plot
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
    
    # Extract filename without path and extension for title
    filename = os.path.basename(csv_file)
    title = os.path.splitext(filename)[0]
    
    # Create the plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    # Plot each channel normalized to first measurement
    # for channel in channels:
    #     column_name = f'raw_{channel}'
    #     if column_name in df.columns:
    #         normalized_on_clear_values = df[column_name] / clear_values
    #         # Normalize each measurement by dividing by the first measurement
    #         first_value = normalized_on_clear_values.iloc[0]
    #         if first_value != 0:  # Avoid division by zero
    #             normalized_values = normalized_on_clear_values / first_value
    #             plt.plot(df['time_seconds'], normalized_values, 
    #                     marker='o', markersize=3, linewidth=1, label=channel)

    # total_sum, F1_to_F7_ratio, spectral_centroid, F5_to_F7_ratio, short_long_sum_ratio

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
    
    plot_columns = ['short_long_sum_ratio', 'F3_to_F7_ratio', 'F5_to_F7_ratio']

    for x in plot_columns:
        normalized_values = df[x] / df[x].iloc[0]
        ax[0].plot(df['time_seconds'], normalized_values, 
                marker='o', markersize=3, linewidth=1, label=x)
    
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('Intensity ratios')
    ax[0].set_title(title)
    ax[0].legend(loc='upper left')
    ax[0].grid(True, alpha=0.3)
    ax[0].set_xlim(df['time_seconds'].min(), df['time_seconds'].max())
    ax[0].set_ylim(0.99, 1.10)
    # ax[0].set_ylim(0, 2)

    ax[1].plot(df['time_seconds'], df['spectral_centroid'], 
                marker='o', markersize=3, linewidth=1, label='Spectral Centroid')
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('Spectral Centroid (nm)')
    ax[1].grid(True, alpha=0.3)
    ax[1].set_xlim(df['time_seconds'].min(), df['time_seconds'].max())
    ax[1].set_ylim(df['spectral_centroid'].min() - 10, df['spectral_centroid'].max() + 10)

    ax[1].legend().set_visible(False)

    # Add a text box with endpoint values
    textstr = f'Final Spectral Centroid: {df['spectral_centroid'].iloc[-2]:.2f} nm\n' \
                f'Spectral Width: {df['spectral_width'].iloc[-2]:.2f} nm\n' \
                f'Spectral shift: {df['spectral_centroid'].iloc[0] - df['spectral_centroid'].iloc[-2]:.2f} nm\n' \
                f'F3/F7 ratio: {df['F3_to_F7_ratio'].iloc[-2]:.2f}\n' \
                f'F5/F7 ratio: {df['F5_to_F7_ratio'].iloc[-2]:.2f}\n' \
                f'Short/Long ratio: {df['short_long_sum_ratio'].iloc[-2]:.2f}'
                
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax[1].text(0.98, 0.98, textstr, transform=ax[1].transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
        
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    return fig
        
 

# directory_path = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250604 WFBR experiments\perpendicular_spectrometer"
directory_path = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250702 WFBR experiments"
pattern = '*.csv'

path = Path(directory_path)
filenames = [f.name for f in path.glob(pattern) if not 'summary' in f.name]

print("Files matching pattern:", pattern)
for filename in filenames:

    # Plot each file
    fig = plot_spectral_data(path / filename)
    
    # fig, results = plot_kinetics_summary(path / filename, 
    #                                      time_window =[0,60])
    # # Maximize the figure
    # fig.set_size_inches(12, 8)
    
    # Save the figure
    fig.savefig(path / 'tmp' / f"{filename}_features.png", dpi=300, bbox_inches='tight')
                                         

# Show the plot
plt.show()
