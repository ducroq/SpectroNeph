#!/usr/bin/env python3
"""
Simple Function-Based Nephelometer Analysis

No classes, just simple functions to analyze your nephelometer data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re


def parse_filename(filename):
    """Extract parameters from filename."""
    params = {'filename': filename}
    
    # Extract date
    date_match = re.search(r'(\d{6})_', filename)
    if date_match:
        params['date'] = date_match.group(1)
    
    # Extract current
    current_match = re.search(r'I[=]?(\d+)mA', filename)
    if current_match:
        params['current_mA'] = int(current_match.group(1))
    
    # Extract gain
    gain_match = re.search(r'G[=]?(\d+)', filename)
    if gain_match:
        params['gain'] = int(gain_match.group(1))
    
    # Extract integration time
    it_match = re.search(r'IT[=]?(\d+)ms', filename)
    if it_match:
        params['integration_time_ms'] = int(it_match.group(1))
    
    # Background subtraction
    if 'no_bckgnd_sub' in filename:
        params['background_subtracted'] = False
    elif 'bckgnd_sub' in filename:
        params['background_subtracted'] = True
    
    # Extract bead size
    bead_match = re.search(r'(\d+\.?\d*)um_beads', filename)
    if bead_match:
        params['bead_size_um'] = float(bead_match.group(1))
    
    # Extract solids concentration
    solids_match = re.search(r'(\d+\.?\d*)%_solids', filename)
    if solids_match:
        params['solids_percent'] = float(solids_match.group(1))
    
    # Extract salt concentration
    salt_match = re.search(r'(\d+\.?\d*)%_salt', filename)
    if salt_match:
        params['salt_percent'] = float(salt_match.group(1))
    
    # Extract time point
    time_match = re.search(r'(\d+)min\.csv', filename)
    if time_match:
        params['time_min'] = int(time_match.group(1))
    
    return params


def load_data(file_paths):
    """Load all CSV files and combine with parameters."""
    all_data = []
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: {file_path} not found")
            continue
        
        try:
            # Load CSV
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} rows from {path.name}")
            
            # Parse filename and add parameters
            params = parse_filename(path.name)
            for key, value in params.items():
                df[key] = value
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_data:
        raise ValueError("No data loaded!")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal dataset: {len(combined_data)} measurements from {len(all_data)} files")
    return combined_data


def calculate_ratios(data):
    """Calculate key spectral ratios."""
    result = data.copy()
    
    # Define ratios to calculate
    ratios = {
        '515nm_630nm': ('processed_F4', 'processed_F7'),    # Green/Red - PRIMARY
        '445nm_630nm': ('processed_F2', 'processed_F7'),    # Blue/Red - Secondary  
        '415nm_515nm': ('processed_F1', 'processed_F4'),    # Violet/Green - Research
        '415nm_630nm': ('processed_F1', 'processed_F7'),    # Violet/Red - Reference
        '480nm_630nm': ('processed_F3', 'processed_F7'),    # Cyan/Red - Additional
    }
    
    for ratio_name, (num_col, den_col) in ratios.items():
        if num_col in result.columns and den_col in result.columns:
            # Calculate ratio, avoiding division by zero
            mask = (result[den_col] != 0) & (result[den_col].notna()) & (result[num_col].notna())
            result[ratio_name] = np.where(mask, result[num_col] / result[den_col], np.nan)
            print(f"Calculated {ratio_name}")
        else:
            print(f"Skipping {ratio_name} - missing columns ({num_col}, {den_col})")
    
    return result


def summarize_conditions(data):
    """Create summary statistics by condition."""
    # Group by experimental conditions
    summary_data = []
    
    for (salt, time), group in data.groupby(['salt_percent', 'time_min']):
        condition = {
            'salt_percent': salt,
            'time_min': time,
            'n_measurements': len(group)
        }
        
        # Calculate stats for key ratios
        key_ratios = ['515nm_630nm', '445nm_630nm', '415nm_515nm']
        for ratio in key_ratios:
            if ratio in group.columns:
                values = group[ratio].dropna()
                if len(values) > 0:
                    condition[f'{ratio}_mean'] = values.mean()
                    condition[f'{ratio}_std'] = values.std()
                    condition[f'{ratio}_cv'] = (values.std() / values.mean() * 100) if values.mean() != 0 else np.nan
        
        summary_data.append(condition)
    
    return pd.DataFrame(summary_data)


def plot_analysis(data, save_path=None):
    """Create analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Nephelometer Analysis', fontsize=16, fontweight='bold')
    
    # 1. Spectral channels by condition
    ax1 = axes[0, 0]
    plot_spectral_channels(data, ax1)
    
    # 2. Key ratio comparison
    ax2 = axes[0, 1]
    plot_ratio_comparison(data, ax2)
    
    # 3. Time evolution
    ax3 = axes[1, 0]
    plot_time_evolution(data, ax3)
    
    # 4. Salt effect
    ax4 = axes[1, 1]
    plot_salt_effect(data, ax4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {save_path}")
    
    plt.show()

def plot_spectral_analysis(data, save_path=None):
    """Create spectral channels analysis plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Spectral Profile Analysis', fontsize=14, fontweight='bold')
    
    plot_spectral_channels(data, ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spectral analysis saved to {save_path}")
    
    plt.show()


def plot_ratio_analysis(data, save_path=None):
    """Create ratio comparison plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('515nm/630nm Ratio Distribution', fontsize=14, fontweight='bold')
    
    plot_ratio_comparison(data, ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ratio analysis saved to {save_path}")
    
    plt.show()


def plot_time_analysis(data, save_path=None):
    """Create time evolution plot."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle('Ratio Changes Over Time', fontsize=14, fontweight='bold')
    
    plot_time_evolution(data, ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time analysis saved to {save_path}")
    
    plt.show()


def plot_salt_analysis(data, save_path=None):
    """Create salt effect plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Immediate Salt Effect on Spectral Ratios', fontsize=14, fontweight='bold')
    
    plot_salt_effect(data, ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Salt analysis saved to {save_path}")
    
    plt.show()
    

def plot_spectral_channels(data, ax):
    """Plot spectral channels with individual points and connected mean lines."""
    channels = ['processed_F1', 'processed_F2', 'processed_F3', 'processed_F4', 
                'processed_F5', 'processed_F6', 'processed_F7', 'processed_F8']
    wavelengths = [415, 445, 480, 515, 555, 590, 630, 680]
    
    # Get unique conditions and assign colors
    conditions = data.groupby(['salt_percent', 'time_min']).size().index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    
    # Plot for each condition
    for i, (salt_pct, time_min) in enumerate(conditions):
        group = data[(data['salt_percent'] == salt_pct) & (data['time_min'] == time_min)]
        color = colors[i]
        label = f'{salt_pct}% salt, {time_min} min'
        
        # First, plot all individual measurements as scatter points
        for _, row in group.iterrows():
            intensities = []
            wl_used = []
            
            for channel, wl in zip(channels, wavelengths):
                if channel in row and not pd.isna(row[channel]):
                    intensities.append(row[channel])
                    wl_used.append(wl)
            
            if intensities:
                # Plot individual points with transparency
                ax.scatter(wl_used, intensities, color=color, alpha=0.4, s=30, zorder=1)
        
        # Then, plot the mean line on top
        mean_intensities = []
        wl_used = []
        
        for channel, wl in zip(channels, wavelengths):
            if channel in group.columns:
                mean_intensity = group[channel].mean()
                if not np.isnan(mean_intensity):
                    mean_intensities.append(mean_intensity)
                    wl_used.append(wl)
        
        if mean_intensities:
            # Plot connected mean line with higher z-order to appear on top
            ax.plot(wl_used, mean_intensities, 'o-', color=color, 
                   linewidth=3, markersize=8, label=label, zorder=2)
    
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity (AU)')
    ax.set_title('Spectral Profile: Individual Measurements + Mean')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_ratio_comparison(data, ax):
    """Plot key ratios as box plots."""
    # Prepare data for plotting
    plot_data = []
    
    for _, row in data.iterrows():
        condition = f"{row['salt_percent']}% salt\n{row['time_min']} min"
        if '515nm_630nm' in row and not pd.isna(row['515nm_630nm']):
            plot_data.append({
                'Condition': condition,
                '515nm/630nm Ratio': row['515nm_630nm']
            })
    
    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(data=plot_df, x='Condition', y='515nm/630nm Ratio', ax=ax)
        ax.set_title('515nm/630nm Ratio by Condition')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No ratio data available', ha='center', va='center', transform=ax.transAxes)


def plot_time_evolution(data, ax):
    """Plot how ratios change over time."""
    key_ratios = ['515nm_630nm', '445nm_630nm', '415nm_515nm']
    colors = ['red', 'blue', 'green']
    
    for ratio, color in zip(key_ratios, colors):
        if ratio in data.columns:
            # For each salt condition
            for salt_pct in data['salt_percent'].unique():
                salt_data = data[data['salt_percent'] == salt_pct]
                
                # Calculate mean and std for each time point
                time_stats = salt_data.groupby('time_min')[ratio].agg(['mean', 'std']).reset_index()
                
                if len(time_stats) > 0:
                    linestyle = '-' if salt_pct == 0 else '--'
                    label = f'{ratio.replace("_", "/")} ({salt_pct}% salt)'
                    
                    ax.errorbar(time_stats['time_min'], time_stats['mean'], 
                              yerr=time_stats['std'], marker='o', linewidth=2,
                              label=label, color=color, linestyle=linestyle, capsize=5)
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Spectral Ratio')
    ax.set_title('Ratio Changes Over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)


def plot_salt_effect(data, ax):
    """Plot immediate salt effect on ratios."""
    # Get t=0 data only
    t0_data = data[data['time_min'] == 0]
    
    if len(t0_data) == 0:
        ax.text(0.5, 0.5, 'No t=0 data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Calculate salt effects for key ratios
    key_ratios = ['515nm_630nm', '445nm_630nm', '415nm_515nm']
    effects = {}
    
    for ratio in key_ratios:
        if ratio in t0_data.columns:
            no_salt = t0_data[t0_data['salt_percent'] == 0][ratio].mean()
            with_salt = t0_data[t0_data['salt_percent'] == 50][ratio].mean()
            
            if not pd.isna(no_salt) and not pd.isna(with_salt) and no_salt != 0:
                effect = ((with_salt - no_salt) / no_salt) * 100
                effects[ratio] = effect
    
    if effects:
        names = list(effects.keys())
        values = list(effects.values())
        
        colors = ['red' if abs(v) > 10 else 'orange' if abs(v) > 5 else 'green' for v in values]
        bars = ax.barh(names, values, color=colors, alpha=0.7)
        
        ax.set_xlabel('Percent Change (%)')
        ax.set_title('Immediate Salt Effect on Ratios\n(t=0 minutes)')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(value + (0.5 if value > 0 else -0.5), bar.get_y() + bar.get_height()/2,
                   f'{value:.1f}%', ha='left' if value > 0 else 'right', va='center')
    else:
        ax.text(0.5, 0.5, 'Cannot calculate salt effects', ha='center', va='center', transform=ax.transAxes)


def analyze_agglutination(data):
    """Analyze agglutination by comparing time points."""
    print("\n" + "="*60)
    print("AGGLUTINATION ANALYSIS")
    print("="*60)
    
    # Compare 0 min vs 5 min for each salt condition
    for salt_pct in data['salt_percent'].unique():
        salt_data = data[data['salt_percent'] == salt_pct]
        
        t0_data = salt_data[salt_data['time_min'] == 0]
        t5_data = salt_data[salt_data['time_min'] == 5]
        
        if len(t0_data) > 0 and len(t5_data) > 0:
            print(f"\n{salt_pct}% Salt Condition:")
            
            for ratio in ['515nm_630nm', '445nm_630nm', '415nm_515nm']:
                if ratio in salt_data.columns:
                    t0_mean = t0_data[ratio].mean()
                    t5_mean = t5_data[ratio].mean()
                    
                    if not pd.isna(t0_mean) and not pd.isna(t5_mean) and t0_mean != 0:
                        change = ((t5_mean - t0_mean) / t0_mean) * 100
                        
                        # Assess significance
                        if abs(change) > 15:
                            significance = "STRONG"
                        elif abs(change) > 5:
                            significance = "MODERATE"
                        else:
                            significance = "MINIMAL"
                        
                        print(f"  {ratio}: {t0_mean:.4f} → {t5_mean:.4f} "
                            f"({change:+.1f}% change, {significance})")

def analyze_salt_effect(data):
    """Analyze immediate salt effect."""
    print("\n" + "="*60)
    print("SALT EFFECT ANALYSIS")
    print("="*60)
    
    # Compare no salt vs 50% salt at t=0
    t0_data = data[data['time_min'] == 0]
    
    if len(t0_data) > 0:
        no_salt = t0_data[t0_data['salt_percent'] == 0]
        with_salt = t0_data[t0_data['salt_percent'] == 50]
        
        if len(no_salt) > 0 and len(with_salt) > 0:
            print(f"\nImmediate Salt Effect (t=0):")
            
            for ratio in ['515nm_630nm', '445nm_630nm', '415nm_515nm']:
                if ratio in t0_data.columns:
                    no_salt_mean = no_salt[ratio].mean()
                    with_salt_mean = with_salt[ratio].mean()
                    
                    if not pd.isna(no_salt_mean) and not pd.isna(with_salt_mean) and no_salt_mean != 0:
                        effect = ((with_salt_mean - no_salt_mean) / no_salt_mean) * 100
                        
                        # Assess significance
                        if abs(effect) > 10:
                            significance = "STRONG"
                        elif abs(effect) > 5:
                            significance = "MODERATE"
                        else:
                            significance = "MINIMAL"
                        
                        print(f"  {ratio.replace('_', '/')}: {no_salt_mean:.4f} → {with_salt_mean:.4f} "
                              f"({effect:+.1f}% change, {significance})")


def print_summary(data):
    """Print overall summary."""
    summary = summarize_conditions(data)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print(f"Total measurements: {len(data)}")
    print(f"Conditions analyzed: {len(summary)}")
    
    print("\nConditions:")
    for _, row in summary.iterrows():
        print(f"  {row['salt_percent']}% salt, {row['time_min']} min: {row['n_measurements']} measurements")
    
    print("\nKey Findings:")
    if '515nm_630nm_mean' in summary.columns:
        for _, row in summary.iterrows():
            condition = f"{row['salt_percent']}% salt, {row['time_min']} min"
            ratio_mean = row['515nm_630nm_mean']
            ratio_cv = row.get('515nm_630nm_cv', np.nan)
            
            if not pd.isna(ratio_mean):
                cv_str = f"{ratio_cv:.1f}%" if not pd.isna(ratio_cv) else "N/A"
                print(f"  {condition}: 515nm/630nm = {ratio_mean:.4f} (CV: {cv_str})")

if __name__ == "__main__":

    files = [
        r"test_data/250601_I=20mA_G=512_IT=300ms_no_bckgnd_sub_1um_beads_0.01%_solids_0%_salt_5min.csv",
        r"test_data/250601_I=20mA_G=512_IT=300ms_no_bckgnd_sub_1um_beads_0.01%_solids_0%_salt_0min.csv",
        r"test_data/250601_I=20mA_G=512_IT=300ms_no_bckgnd_sub_1um_beads_0.01%_solids_50%_salt_0min.csv",
        r"test_data/250601_I=20mA_G=512_IT=300ms_no_bckgnd_sub_1um_beads_0.01%_solids_50%_salt_5min.csv",
        # r"test_data/250601_I=20mA_G=512_IT=300ms_no_bckgnd_sub_1um_beads_0.02%_solids_0%_salt_0min.csv",
        # r"test_data/250601_I=20mA_G=512_IT=300ms_no_bckgnd_sub_1um_beads_0.02%_solids_0%_salt_5min.csv"                
    ]
    
    # Load and process data
    print("Loading nephelometer data...")
    data = load_data(files)
    
    print("\nCalculating spectral ratios...")
    data = calculate_ratios(data)
    
    print("\nCreating analysis plots...")
    plot_analysis(data, "nephelometer_simple_analysis.png")

    # Create individual plots
    plot_spectral_analysis(data, "spectral_analysis.png")
    plot_ratio_analysis(data, "ratio_analysis.png") 
    plot_time_analysis(data, "time_analysis.png")
    plot_salt_analysis(data, "salt_analysis.png")
    
    # Print analyses
    print_summary(data)
    analyze_agglutination(data)
    analyze_salt_effect(data)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


