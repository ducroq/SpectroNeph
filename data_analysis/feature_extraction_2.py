"""
Nephelometric Feature Extraction Script
This script analyzes nephelometric data from AS7341 spectrometer measurements.
It computes various spectral ratios, scattering indicators, and aggregation detection scores.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class AS7341FeatureExtractor:
    """
    Comprehensive data analysis for AS7341 spectrometer.
    
    Computes:
    - Size-sensitive spectral ratios
    - Scattering mechanism indicators  
    - Aggregation detection scores
    - Temporal stability metrics
    """
    
    def __init__(self):
        # AS7341 approximate center wavelengths (nm)
        self.wavelengths = {
            'F1': 415,   # Violet
            'F2': 445,   # Blue  
            'F3': 480,   # Cyan
            'F4': 515,   # Green
            'F5': 555,   # Yellow-green
            'F6': 590,   # Yellow
            'F7': 630,   # Orange
            'F8': 680    # Far-red
        }

        self.combined_correction_factors = {
            'F1': 61.2,   # Highest correction (weak LED + weak AS7341)
            'F2': 3.8,    # Moderate correction  
            'F3': 6.9,    # Moderate correction
            'F4': 2.1,    # Low correction (good LED + best AS7341)
            'F5': 1.0,    # Reference (normalized to F5)
            'F6': 2.2,    # Low correction
            'F7': 7.1,    # Moderate correction (weak LED in red)
            'F8': 10.8    # High correction (very weak LED + moderate AS7341)
        }

    def safe_divide(self, a, b, default=0):
        """Safe division avoiding divide by zero."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(a, b)
            return np.where(np.isfinite(result), result, default)
    
    def compute_basic_ratios(self, df):
        """Compute fundamental spectral ratios."""
        
        # Extract processed channels (use raw if processed not available)
        channels = {}
        for ch in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Clear']:
            if f'processed_{ch}' in df.columns:
                channels[ch] = df[f'processed_{ch}'].values
            elif f'raw_{ch}' in df.columns:
                channels[ch] = df[f'raw_{ch}'].values
            else:
                print(f"Warning: Channel {ch} not found in data")
                channels[ch] = np.zeros(len(df))
        
        results = {}
        
        # Size-sensitive ratios (Rayleigh vs Mie scattering)
        results['violet_farred_ratio'] = self.safe_divide(channels['F1'], channels['F8'])
        results['blue_red_ratio'] = self.safe_divide(channels['F1'] + channels['F2'], 
                                                   channels['F6'] + channels['F7'])
        results['short_long_ratio'] = self.safe_divide(channels['F1'] + channels['F2'] + channels['F3'],
                                                     channels['F6'] + channels['F7'] + channels['F8'])
        results['f3_f7_ratio'] = self.safe_divide(channels['F3'], channels['F7'])
        results['f4_f8_ratio'] = self.safe_divide(channels['F4'], channels['F8'])
        results['f1_f8_ratio'] = self.safe_divide(channels['F1'], channels['F8'])
        results['f4_f7_ratio'] = self.safe_divide(channels['F4'], channels['F7'])
        
        # Adjacent wavelength ratios (spectral slope analysis)
        results['f1_f2_ratio'] = self.safe_divide(channels['F1'], channels['F2'])
        results['f2_f3_ratio'] = self.safe_divide(channels['F2'], channels['F3'])
        results['f3_f4_ratio'] = self.safe_divide(channels['F3'], channels['F4'])
        results['f4_f5_ratio'] = self.safe_divide(channels['F4'], channels['F5'])
        results['f5_f6_ratio'] = self.safe_divide(channels['F5'], channels['F6'])
        results['f6_f7_ratio'] = self.safe_divide(channels['F6'], channels['F7'])
        results['f7_f8_ratio'] = self.safe_divide(channels['F7'], channels['F8'])
        
        # Scattering mechanism indicators
        results['rayleigh_component'] = channels['F1'] + channels['F2'] + channels['F3']
        results['mie_component'] = channels['F6'] + channels['F7'] + channels['F8']
        results['rayleigh_mie_ratio'] = self.safe_divide(results['rayleigh_component'], 
                                                       results['mie_component'])
        
        # Peak ratios for spectral shape analysis
        results['peak_ratio_f4_f5'] = self.safe_divide(channels['F4'], channels['F5'])
        results['peak_ratio_f3_f6'] = self.safe_divide(channels['F3'], channels['F6'])
        
        # Normalization ratios (remove intensity variations)
        for ch in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']:
            results[f'{ch.lower()}_normalized'] = self.safe_divide(channels[ch], channels['Clear'])
        
        # Total signal metrics
        results['total_scatter'] = sum(channels[f'F{i}'] for i in range(1, 9))
        
        # Spectral fractions
        for i in range(1, 9):
            results[f'f{i}_fraction'] = self.safe_divide(channels[f'F{i}'], results['total_scatter'])
        
        results['green_dominance'] = self.safe_divide(
            channels['F3'] + channels['F4'] + channels['F5'],
            channels['F1'] + channels['F2'] + channels['F6'] + channels['F7'] + channels['F8']
        )

        # Spectral flatness (how "peaked" vs "flat" the spectrum is)
        channel_intensities = np.array([np.nanmean(channels[f'F{i}']) for i in range(1, 9)])
        results['spectral_flatness'] = np.nanstd(channel_intensities) / np.nanmean(channel_intensities)

        return results, channels
    
    def compute_advanced_metrics(self, channels):
        """Compute advanced spectral analysis metrics."""
        
        results = {}
        
        # Spectral centroid (weighted average wavelength)
        wavelengths = np.array([self.wavelengths[f'F{i}'] for i in range(1, 9)])
        intensities = np.array([channels[f'F{i}'] for i in range(1, 9)])
        
        # Ensure no negative values
        intensities = np.maximum(intensities, 0)
        
        weighted_sum = np.sum(wavelengths[:, np.newaxis] * intensities, axis=0)
        total_intensity = np.sum(intensities, axis=0)
        
        # Avoid division by zero
        results['spectral_centroid'] = self.safe_divide(weighted_sum, total_intensity, 
                                                       default=np.mean(wavelengths))
        
        # Spectral variance (measure of spectral width)
        centroid_expanded = results['spectral_centroid'][np.newaxis, :]
        variance_numerator = np.sum(intensities * (wavelengths[:, np.newaxis] - centroid_expanded)**2, axis=0)
        results['spectral_variance'] = self.safe_divide(variance_numerator, total_intensity, default=0)
        
        results['spectral_sum'] = sum(channels[ch] for ch in ['F1', 'F2', 'F3', 'F6', 'F7', 'F8'])

        # Avoid sqrt of negative numbers
        variance_clean = np.maximum(results['spectral_variance'], 0)
        results['spectral_width'] = np.sqrt(variance_clean)
        
        # Polydispersity proxy (normalized spectral width)
        results['polydispersity_proxy'] = self.safe_divide(results['spectral_width'], 
                                                         results['spectral_centroid'], default=0)
        
        # Spectral asymmetry (skewness indicator)
        centroid_expanded = results['spectral_centroid'][np.newaxis, :]
        skew_numerator = np.sum(intensities * (wavelengths[:, np.newaxis] - centroid_expanded)**3, axis=0)
        skew_denominator = total_intensity * (results['spectral_variance'] + 1e-10)**1.5  # Small epsilon to avoid div by zero
        results['spectral_skewness'] = self.safe_divide(skew_numerator, skew_denominator, default=0)
        
        # # Turbidity indicators at different wavelengths
        # results['turbidity_445_680'] = self.safe_divide(channels['F2'], channels['F8'])
        # results['turbidity_480_630'] = self.safe_divide(channels['F3'], channels['F7'])
        
        return results
    
    def compute_temporal_features(self, df, basic_ratios):
        """Compute temporal stability and change metrics."""
        
        results = {}
        
        # Time-based analysis (if timestamp available)
        if 'timestamp' in df.columns:
            time = df['timestamp'].values
            try:
                # Unix timestamps are already numeric, no conversion needed
                if len(time) > 1:
                    # Filter out any NaN values
                    time_clean = time[~pd.isna(time)]
                    if len(time_clean) > 1:
                        duration = float(time_clean[-1] - time_clean[0])
                        results['measurement_duration'] = duration
                        results['sampling_rate'] = len(time_clean) / duration if duration > 0 else 0
                        print(f"  Time analysis: {len(time_clean)} measurements over {duration:.2f} seconds ({results['sampling_rate']:.1f} Hz)")
                    else:
                        results['measurement_duration'] = 0
                        results['sampling_rate'] = 0
                        print("  Warning: Less than 2 valid timestamps found")
                else:
                    results['measurement_duration'] = 0
                    results['sampling_rate'] = 0
                    print("  Warning: Only one timestamp available")
            except Exception as e:
                print(f"  Warning: Timestamp processing failed: {e}")
                print(f"  First few timestamps: {time[:5]}")
                results['measurement_duration'] = 0
                results['sampling_rate'] = 0
        else:
            results['measurement_duration'] = 0
            results['sampling_rate'] = 0
            print("  No timestamp column found")
        
        # Rolling statistics for key ratios
        window_size = min(10, len(df) // 4)  # Adaptive window size
        window_size = max(3, window_size)  # Minimum window size
        
        key_ratios = ['violet_farred_ratio', 'blue_red_ratio', 'short_long_ratio', 'total_scatter']
        
        for ratio_name in key_ratios:
            if ratio_name in basic_ratios:
                values = basic_ratios[ratio_name]
                
                # Remove any infinite or NaN values
                clean_values = values[np.isfinite(values)]
                
                if len(clean_values) > 0:
                    # Coefficient of variation (stability indicator)
                    mean_val = np.mean(clean_values)
                    std_val = np.std(clean_values)
                    
                    if mean_val != 0 and not np.isnan(mean_val) and not np.isnan(std_val):
                        results[f'{ratio_name}_cv'] = std_val / mean_val
                    else:
                        results[f'{ratio_name}_cv'] = 0.0
                    
                    # Trend analysis (linear slope)
                    if len(clean_values) > 2:
                        try:
                            x = np.arange(len(clean_values))
                            # Use finite values only
                            finite_mask = np.isfinite(clean_values)
                            if np.sum(finite_mask) > 2:
                                slope, _ = np.polyfit(x[finite_mask], clean_values[finite_mask], 1)
                                results[f'{ratio_name}_trend'] = slope if np.isfinite(slope) else 0.0
                            else:
                                results[f'{ratio_name}_trend'] = 0.0
                        except:
                            results[f'{ratio_name}_trend'] = 0.0
                    else:
                        results[f'{ratio_name}_trend'] = 0.0
                    
                    # Rolling mean and std
                    if len(values) >= window_size:
                        try:
                            rolling_mean = pd.Series(values).rolling(window=window_size, center=True, min_periods=1).mean()
                            rolling_std = pd.Series(values).rolling(window=window_size, center=True, min_periods=1).std()
                            
                            results[f'{ratio_name}_rolling_mean'] = rolling_mean.ffill().bfill().values
                            results[f'{ratio_name}_rolling_std'] = rolling_std.fillna(0.0).values
                            
                            # Stability score (inverse of relative variation)
                            rel_variation = rolling_std / rolling_mean
                            rel_variation = rel_variation.fillna(0.0)  # Handle NaN (pandas method)
                            rel_variation_array = np.where(np.isfinite(rel_variation), rel_variation, 0.0)  # Convert to numpy, handle inf
                            stability_array = 1.0 / (1.0 + rel_variation_array)
                            stability_array = np.where(np.isfinite(stability_array), stability_array, 1.0)  # Handle any remaining NaN/inf
                            results[f'{ratio_name}_stability'] = stability_array
                        except Exception as e:
                            print(f"Warning: Rolling calculation failed for {ratio_name}: {e}")
                            # Fallback to simple arrays
                            results[f'{ratio_name}_rolling_mean'] = np.full_like(values, np.mean(clean_values))
                            results[f'{ratio_name}_rolling_std'] = np.full_like(values, np.std(clean_values))
                            results[f'{ratio_name}_stability'] = np.ones_like(values)
                    else:
                        # Not enough data for rolling window
                        mean_val = np.mean(clean_values) if len(clean_values) > 0 else 0
                        std_val = np.std(clean_values) if len(clean_values) > 0 else 0
                        results[f'{ratio_name}_rolling_mean'] = np.full_like(values, mean_val)
                        results[f'{ratio_name}_rolling_std'] = np.full_like(values, std_val)
                        results[f'{ratio_name}_stability'] = np.ones_like(values)
                else:
                    # No valid data
                    results[f'{ratio_name}_cv'] = 0.0
                    results[f'{ratio_name}_trend'] = 0.0
                    results[f'{ratio_name}_rolling_mean'] = np.zeros_like(values)
                    results[f'{ratio_name}_rolling_std'] = np.zeros_like(values)
                    results[f'{ratio_name}_stability'] = np.ones_like(values)
        
        return results
    
    def compute_SAS(self, channels, buffer_system='running'):
        """
        Calculate Spectral Agglutination Score
        
        Parameters:
        F1-F8: Channel intensities
        buffer_system: 'running' or 'tween'
        
        Returns:
        SAS score (positive = agglutination)
        """
        F1 = np.nanmean(channels['F1'])
        F2 = np.nanmean(channels['F2'])
        F3 = np.nanmean(channels['F3'])
        F4 = np.nanmean(channels['F4'])
        F5 = np.nanmean(channels['F5'])
        F6 = np.nanmean(channels['F6'])
        F7 = np.nanmean(channels['F7'])
        F8 = np.nanmean(channels['F8'])
      
        # Calculate features
        f1f8_ratio = self.safe_divide(F1, F8)
        f4f7_ratio = self.safe_divide(F4, F7)
        
        # Calculate centroid (weighted average wavelength)
        wavelengths = [415, 445, 480, 515, 555, 590, 630, 680]
        channels = [F1, F2, F3, F4, F5, F6, F7, F8]
        centroid = sum(w * c for w, c in zip(wavelengths, channels)) / sum(channels)
        
        # Calculate total scatter  
        spectral_sum  = F1 + F2 + F3 + F6 + F7 + F8
        
        if buffer_system == 'running':
            print("     Using running buffer constants")
            
            # Running buffer baseline constants
            MU_F1F8, SIGMA_F1F8 = 2.855, 0.094
            MU_F4F7, SIGMA_F4F7 = 0.272, 0.005
            MU_CENTROID, SIGMA_CENTROID = 515.298, 0.531
            MU_SPECTRAL_SUM, SIGMA_SPECTRAL_SUM = 91824, 7924 
            
            # Calculate z-scores
            z_f1f8 = (f1f8_ratio - MU_F1F8) / SIGMA_F1F8
            z_f4f7 = (f4f7_ratio - MU_F4F7) / SIGMA_F4F7
            z_centroid = (centroid - MU_CENTROID) / SIGMA_CENTROID
            z_spectral_sum = (spectral_sum - MU_SPECTRAL_SUM) / SIGMA_SPECTRAL_SUM
            
            # Running buffer SAS
            SAS = 0.4*z_f1f8 + 0.3*z_f4f7 + 0.2*z_centroid + 0.1*z_spectral_sum
            
        elif buffer_system == 'tween':
            print("     Using Tween buffer constants")
            # Tween buffer baseline constants (ACTUAL DATA from CSV)
            MU_F1F8, SIGMA_F1F8 = 3.023, 0.007
            MU_F4F7, SIGMA_F4F7 = 0.284, 0.0003
            MU_CENTROID, SIGMA_CENTROID = 513.323, 0.114
            MU_SPECTRAL_SUM, SIGMA_SPECTRAL_SUM = 173427, 8414
            
            # Calculate z-scores
            z_f1f8 = (f1f8_ratio - MU_F1F8) / SIGMA_F1F8
            z_f4f7 = (f4f7_ratio - MU_F4F7) / SIGMA_F4F7
            z_centroid = (centroid - MU_CENTROID) / SIGMA_CENTROID
            z_spectral_sum = (spectral_sum - MU_SPECTRAL_SUM) / SIGMA_SPECTRAL_SUM
            
            # Tween buffer SAS
            SAS = 0.5*z_f1f8 + 0.3*z_spectral_sum + 0.1*z_centroid + 0.1*z_f4f7
        
        return SAS    
    
    
    def remove_nan_measurements(self, features):
        # Find measurements with any NaN values
        valid_mask = True
        for key, values in features.items():
            if isinstance(values, np.ndarray):
                valid_mask = valid_mask & ~np.isnan(values)
        
        # Keep only valid measurements
        cleaned_features = {}
        for key, values in features.items():
            if isinstance(values, np.ndarray):
                cleaned_features[key] = values[valid_mask]
            else:
                cleaned_features[key] = values
        
        return cleaned_features    
    
    def analyze_file(self, filepath):
        """Complete analysis of a single CSV file."""
        
        print(f"Analyzing file: {filepath}")
        
        # Load data
        try:
            df = pd.read_csv(filepath)
            # df = df.dropna()  # Remove rows with any NaN values

            print(f"  Loaded {len(df)} measurements")
        except Exception as e:
            print(f"  Error loading file: {e}")
            return None
        
        # Use only raw columns
        raw_columns = [col for col in df.columns if col.startswith('raw_')]
        if not raw_columns:
            print("  No raw spectral data found in file")
            return None
        df = df['timestamp'].to_frame().join(df[raw_columns]) if 'timestamp' in df.columns else df

        # Apply LED correction to raw spectral features
        for ch in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']:
            if f'raw_{ch}' in df.columns:
                df[f'processed_{ch}'] = df[f'raw_{ch}'] * self.combined_correction_factors[ch]
            else:
                print(f"Warning: Raw channel {ch} not found in data")
        for ch in ['Clear', 'NIR']:
            if f'raw_{ch}' in df.columns:
                df[f'processed_{ch}'] = df[f'raw_{ch}']

        # Compute all features
        basic_ratios, channels = self.compute_basic_ratios(df)
        advanced_metrics = self.compute_advanced_metrics(channels)
        temporal_features = self.compute_temporal_features(df, basic_ratios)
        buffer_system = 'running' if 'running' in os.path.basename(filepath).lower() else 'tween'
        SAS = self.compute_SAS(channels, buffer_system=buffer_system)
    
        # Combine all results
        all_features = {}
        all_features.update(channels)  # Include raw and processed channels
        all_features.update(basic_ratios)
        all_features.update(advanced_metrics)
        all_features.update(temporal_features)
        all_features.update({'SAS': SAS})

        # Remove NaN measurements
        all_features = self.remove_nan_measurements(all_features)
        
        # Create output DataFrame
        # Handle different length arrays by using the length of the input data
        n_measurements = len(df)
        
        output_data = {}
        for key, values in all_features.items():
            if isinstance(values, np.ndarray) and len(values) == n_measurements:
                output_data[key] = values
            elif isinstance(values, (int, float)):
                output_data[key] = [values] * n_measurements
            else:
                # Skip features that don't match measurement count
                continue
        
        # Add original columns
        for col in df.columns:
            if col not in output_data:
                output_data[col] = df[col].values
        
        result_df = pd.DataFrame(output_data)
        
        # Summary statistics
        summary = self.compute_summary_stats(all_features)
        
        print(f"  Computed {len(all_features)} features")
        
        return result_df, summary, all_features
    
    def compute_summary_stats(self, features):
        """Compute summary statistics for key features."""
        
        summary = {}
        
        key_features = [
            'violet_farred_ratio', 'blue_red_ratio', 'short_long_ratio',
            'total_scatter', 'spectral_centroid', 'detection_score'
        ]
        
        for feature in key_features:
            if feature in features:
                values = features[feature]
                if isinstance(values, np.ndarray) and len(values) > 0:
                    summary[f'{feature}_mean'] = np.mean(values)
                    summary[f'{feature}_std'] = np.std(values)
                    summary[f'{feature}_min'] = np.min(values)
                    summary[f'{feature}_max'] = np.max(values)
                    summary[f'{feature}_cv'] = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        
        return summary
    
    def plot_analysis(self, features, title="Nephelometric Analysis"):
        """
        Create comprehensive analysis plots focused on key nephelometric features.
        
        Parameters:
        -----------
        features : dict
            Features dictionary (LED-corrected data)
        title : str
            Plot title
        """
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Colors for consistency
        colors = {
            'total': 'black',
            'clear': 'gray',
            'nir': 'darkred',
            'f1': '#8B00FF',  # Violet
            'f2': '#0000FF',  # Blue
            'f3': '#00FFFF',  # Cyan
            'f4': '#00FF00',  # Green
            'f5': '#FFFF00',  # Yellow
            'f6': '#FFA500',  # Orange
            'f7': '#FF4500',  # Red-Orange
            'f8': '#FF0000'   # Red
        }

        # Plot 1: Total Scatter Intensity, Clear, and NIR over time
        ax1 = axes[0, 0]
        if 'total_scatter' in features:
            ax1.plot(features['total_scatter'], color=colors['total'], linewidth=2, label='Total F1-F8', alpha=0.8)
        if 'Clear' in features:
            ax1.plot(features['Clear'], color=colors['clear'], linewidth=1.5, label='Clear', alpha=0.8)
        if 'processed_NIR' in features:
            ax1.plot(features['processed_NIR'], color=colors['nir'], linewidth=1.5, label='NIR', alpha=0.8)
        
        # ax1.set_title('Intensity Channels vs Time', fontweight='bold')
        ax1.set_xlabel('Measurement #')
        ax1.set_ylabel('Intensity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average Spectral Distribution (like your attached plot)
        ax2 = axes[0, 1]
        
        # AS7341 wavelengths
        wavelengths = [415, 445, 480, 515, 555, 590, 630, 680]
        
        # Calculate average intensities for each channel
        channel_keys = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']
        
        avg_intensities = []
        for key in channel_keys:
            if key in features:
                avg_intensities.append(np.nanmean(features[key]))
            else:
                avg_intensities.append(0)
        
        # Bar plot of spectral distribution
        bars = ax2.bar(wavelengths, avg_intensities, width=25, alpha=0.7, 
                    color=[colors[f'f{i}'] for i in range(1, 9)], edgecolor='black', linewidth=0.5)
        
        # Add LED-corrected intensities (now the main data)
        # Line plot for the corrected spectral distribution
        ax2.plot(wavelengths, avg_intensities, 'ko-', linewidth=2, markersize=6, 
                label='LED-Corrected', alpha=0.8)
        
        # ax2.set_title('Average Spectral Distribution', fontweight='bold')
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Average Intensity')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Key Spectral Ratios over time
        ax3 = axes[0, 2]
        
        if 'violet_farred_ratio' in features:
            ax3.plot(features['violet_farred_ratio'], label='V/FR (F1/F8)', color='purple', alpha=0.8)
        if 'blue_red_ratio' in features:
            ax3.plot(features['blue_red_ratio'], label='B/R ((F1+F2)/(F6+F7))', color='blue', alpha=0.8)
        if 'rayleigh_mie_ratio' in features:
            ax3.plot(features['rayleigh_mie_ratio'], label='R/M (Short/Long)', color='green', alpha=0.8)
        
        # ax3.set_title('Key Spectral Ratios', fontweight='bold')
        ax3.set_xlabel('Measurement #')
        ax3.set_ylabel('Ratio')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary Statistics (instead of scattering components)
        ax4 = axes[1, 0]
        ax4.axis('off')  # Turn off axes

        summary_text = f"""MEASUREMENT SUMMARY

KEY FEATURES:
- Total Scatter: {np.mean(features['total_scatter']):.3f} ± {np.std(features['total_scatter']):.3f}
- Violet/Far-Red Ratio: {np.mean(features['f1_f8_ratio']):.3f} ± {np.std(features['f1_f8_ratio']):.3f} (CV: {np.std(features['f1_f8_ratio'])/np.mean(features['f1_f8_ratio'])*100:.1f}%)
- Green/Orange Ratio: {np.mean(features['f4_f7_ratio']):.3f} ± {np.std(features['f4_f7_ratio']):.3f} (CV: {np.std(features['f4_f7_ratio'])/np.mean(features['f4_f7_ratio'])*100:.1f}%)
- Spectral centroid: {np.mean(features['spectral_centroid']):.3f} ± {np.std(features['spectral_centroid']):.3f} nm (CV: {np.std(features['spectral_centroid'])/np.mean(features['spectral_centroid'])*100:.1f}%)

STABILITY:
- Total scatter CV: {np.std(features['total_scatter']) / np.mean(features['total_scatter']):.3f}%

DETECTION:
- SAS:{features['SAS']:.3f}
        """
# - SAS:{np.mean(features['SAS']):.3f} ± {np.std(features['SAS']):.3f} (CV: {np.std(features['SAS'])/np.mean(features['SAS'])*100:.1f}%)

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Plot 5: Spectral Properties
        ax5 = axes[1, 1]
        
        # Spectral centroid (LED-corrected data)
        if 'spectral_centroid' in features:
            ax5_twin = ax5.twinx()
            line1 = ax5.plot(features['spectral_centroid'], color='orange', label='Centroid', alpha=0.8)
            
            ax5.set_ylabel('Spectral Centroid (nm)', color='orange')
            ax5.tick_params(axis='y', labelcolor='orange')
            
            # Polydispersity on second y-axis
            if 'polydispersity_proxy' in features:
                line3 = ax5_twin.plot(features['polydispersity_proxy'], color='brown', label='Polydispersity', alpha=0.8)
                
                ax5_twin.set_ylabel('Polydispersity', color='brown')
                ax5_twin.tick_params(axis='y', labelcolor='brown')
            
            # Combine legends
            lines1, labels1 = ax5.get_legend_handles_labels()
            lines2, labels2 = ax5_twin.get_legend_handles_labels() if 'polydispersity_proxy' in features else ([], [])
            ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        
        # ax5.set_title('Spectral Properties', fontweight='bold')
        ax5.set_xlabel('Measurement #')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: SAS Analysis
        ax6 = axes[1, 2]
        
        if 'SAS' in features:
            ax6.plot(features['SAS'], color='purple', linewidth=2, label='Detection Score', alpha=0.8)
            
            # # Add detection thresholds
            # ax6.axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, label='Detection Threshold')
            # ax6.axhline(y=0.30, color='red', linestyle='--', alpha=0.7, label='Strong Aggregation')
            
            # # Highlight detection events
            # if 'aggregation_detected' in features:
            #     detection_events = np.where(features['aggregation_detected'])[0]
            #     if len(detection_events) > 0:
            #         ax6.scatter(detection_events, 
            #                 [features['detection_score'][i] for i in detection_events],
            #                 color='red', s=30, alpha=0.8, label='Detection Events', zorder=5)
        
        # Add ratio changes if available
        if 'violet_farred_change' in features:
            ax6_twin = ax6.twinx()
            ax6_twin.plot(features['violet_farred_change'], color='green', alpha=0.6, 
                        label='V/FR Change', linewidth=1)
            ax6_twin.set_ylabel('V/FR Change', color='green')
            ax6_twin.tick_params(axis='y', labelcolor='green')
            ax6_twin.axhline(y=0, color='green', linestyle='-', alpha=0.3)
        
        # ax6.set_title('Aggregation Detection', fontweight='bold')
        ax6.set_xlabel('Measurement #')
        ax6.set_ylabel('Detection Score')
        ax6.legend(loc='upper left', fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
       
        return fig

# Example usage and testing
if __name__ == "__main__":
    import os

    analyzer = AS7341FeatureExtractor()
    
    # Test with sample data
    sample_file = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250604 WFBR experiments\perpendicular_spectrometer\0.45micron_running_buffer_just_sonified_(no CRP)_0.csv"
    
    try:
        result_df, summary, features = analyzer.analyze_file(sample_file)

        fig = analyzer.plot_analysis(features, title=f"Analysis: {os.path.basename(sample_file)}")
        
#         plot_path = Path(args.input_file).parent / f"{Path(args.input_file).stem}_analysis.png"
#         fig.savefig(plot_path, dpi=300, bbox_inches='tight')
#         print(f"Plots saved to: {plot_path}")
        
        print("Key Features Computed:")
        for key in sorted(features.keys()):
            if isinstance(features[key], np.ndarray):
                print(f"  {key}: array({len(features[key])})")
            else:
                print(f"  {key}: {features[key]}")
        
        print(f"\nOutput DataFrame shape: {result_df.shape}")
        print(f"Output columns: {list(result_df.columns)}")

        plt.show()
        
    except FileNotFoundError:
        print("Sample file not found. Use command line interface instead.")