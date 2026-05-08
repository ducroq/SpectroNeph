"""
Simplified Nephelometric Measurement Summary
============================================

Clean, simple summary of key nephelometric features.
"""

import numpy as np
import pandas as pd


def export_summary_to_csv(features, name, output_filename=None):
    """
    Export spectral measurement summary to CSV file.
    
    Parameters:
    features: dict containing measurement data arrays
    name: str, sample name
    output_filename: str, optional filename (defaults to name + '_summary.csv')
    """
    
    if output_filename is None:
        output_filename = f"{name.replace('.csv', '')}_summary.csv"
    
    # Create summary data structure
    summary_data = []
    
    # Basic info
    summary_data.append({
        'Category': 'Basic Info',
        'Parameter': 'Sample Name',
        'Value': name,
        'Std Dev': '',
        'CV (%)': '',
        'Unit': ''
    })
    
    summary_data.append({
        'Category': 'Basic Info',
        'Parameter': 'Measurements',
        'Value': len(features['total_scatter']),
        'Std Dev': '',
        'CV (%)': '',
        'Unit': 'count'
    })
    
    summary_data.append({
        'Category': 'Basic Info',
        'Parameter': 'Sampling Rate',
        'Value': f"{features['sampling_rate']:.3f}",
        'Std Dev': '',
        'CV (%)': '',
        'Unit': 'Hz'
    })
    
    # Channel data
    for ch in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Clear']:
        if ch in features:
            mean_val = np.mean(features[ch])
            std_val = np.std(features[ch])
            cv_val = (std_val / mean_val) * 100
            
            summary_data.append({
                'Category': 'Channel Data',
                'Parameter': ch,
                'Value': f"{mean_val:.3f}",
                'Std Dev': f"{std_val:.3f}",
                'CV (%)': f"{cv_val:.1f}",
                'Unit': 'counts'
            })
    
    # Key ratios for agglutination detection
    key_ratios = [
        # PRIMARY SIZE-SENSITIVE RATIOS (for SAS calculation)
        ('F1', 'F8', 'Violet/Far-Red (Primary)'),
        ('F4', 'F7', 'Green/Orange (Primary)'),
        
        # SECONDARY SIZE-SENSITIVE RATIOS
        ('F1', 'F7', 'Violet/Orange'),
        ('F2', 'F8', 'Blue/Far-Red'),
        ('F3', 'F8', 'Cyan/Far-Red'),
        ('F4', 'F8', 'Green/Far-Red'),
        
        # WAVELENGTH BAND RATIOS
        ('F1', 'F2', 'Violet/Blue'),
        ('F2', 'F3', 'Blue/Cyan'),
        ('F3', 'F4', 'Cyan/Green'),
        ('F6', 'F7', 'Yellow/Orange'),
        ('F7', 'F8', 'Orange/Far-Red'),
        
        # SCATTERING REGIME RATIOS
        ('F1', 'F6', 'Short/Long-1'),
        ('F2', 'F7', 'Short/Long-2'),
        ('F3', 'F7', 'Short/Long-3'),
        
        # COMPOSITE RATIOS
        # Short wavelengths / Long wavelengths
        # Note: These will be calculated separately as they involve sums
    ]
    
    for ch1, ch2, description in key_ratios:
        if ch1 in features and ch2 in features:
            ratio_values = features[ch1] / features[ch2]
            mean_val = np.mean(ratio_values)
            std_val = np.std(ratio_values)
            cv_val = (std_val / mean_val) * 100
            
            summary_data.append({
                'Category': 'Key Ratios',
                'Parameter': f"{ch1}/{ch2} ({description})",
                'Value': f"{mean_val:.3f}",
                'Std Dev': f"{std_val:.3f}",
                'CV (%)': f"{cv_val:.1f}",
                'Unit': 'ratio'
            })
    
    # Spectral properties
    spectral_params = [
        ('spectral_centroid', 'Centroid', 'nm'),
        ('spectral_width', 'Width', 'nm'),
        ('polydispersity_proxy', 'Polydispersity', 'ratio'),
        ('spectral_sum', 'Spectral Sum', 'counts')
    ]
    
    for param_key, param_name, unit in spectral_params:
        if param_key in features:
            mean_val = np.mean(features[param_key])
            std_val = np.std(features[param_key])
            cv_val = (std_val / mean_val) * 100
            
            summary_data.append({
                'Category': 'Spectral Properties',
                'Parameter': param_name,
                'Value': f"{mean_val:.3f}",
                'Std Dev': f"{std_val:.3f}",
                'CV (%)': f"{cv_val:.1f}",
                'Unit': unit
            })
    
    # Composite ratios (sums of channels)
    composite_ratios = [
        (['F1', 'F2'], ['F7', 'F8'], 'Short/Long (Blue-Violet/Orange-Red)'),
        (['F1', 'F2', 'F3'], ['F6', 'F7', 'F8'], 'Rayleigh/Mie'),
        (['F1', 'F2', 'F3'], ['F7', 'F8'], 'Short/Long (Extended)'),
        (['F1', 'F2'], ['F6', 'F7'], 'Blue-Violet/Yellow-Orange'),
    ]
    
    for numerator_chs, denominator_chs, description in composite_ratios:
        if all(ch in features for ch in numerator_chs + denominator_chs):
            numerator = sum(features[ch] for ch in numerator_chs)
            denominator = sum(features[ch] for ch in denominator_chs)
            ratio_values = numerator / denominator
            mean_val = np.mean(ratio_values)
            std_val = np.std(ratio_values)
            cv_val = (std_val / mean_val) * 100
            
            summary_data.append({
                'Category': 'Composite Ratios',
                'Parameter': description,
                'Value': f"{mean_val:.3f}",
                'Std Dev': f"{std_val:.3f}",
                'CV (%)': f"{cv_val:.1f}",
                'Unit': 'ratio'
            })
    if 'total_scatter' in features:
        total_scatter = features['total_scatter']
        mean_val = np.mean(total_scatter)
        std_val = np.std(total_scatter)
        cv_val = (std_val / mean_val) * 100
        
        summary_data.append({
            'Category': 'Scattering Analysis',
            'Parameter': 'Total Scatter',
            'Value': f"{mean_val:.0f}",
            'Std Dev': f"{std_val:.0f}",
            'CV (%)': f"{cv_val:.1f}",
            'Unit': 'counts'
        })
    
    # Rayleigh and Mie scattering
    if all(ch in features for ch in ['F1', 'F2', 'F3', 'F6', 'F7', 'F8']):
        rayleigh = features['F1'] + features['F2'] + features['F3']
        mie = features['F6'] + features['F7'] + features['F8']
        rayleigh_mie_ratio = rayleigh / mie
        
        # Rayleigh
        mean_val = np.mean(rayleigh)
        summary_data.append({
            'Category': 'Scattering Analysis',
            'Parameter': 'Rayleigh Scatter (F1+F2+F3)',
            'Value': f"{mean_val:.0f}",
            'Std Dev': f"{np.std(rayleigh):.0f}",
            'CV (%)': f"{(np.std(rayleigh)/mean_val)*100:.1f}",
            'Unit': 'counts'
        })
        
        # Mie
        mean_val = np.mean(mie)
        summary_data.append({
            'Category': 'Scattering Analysis',
            'Parameter': 'Mie Scatter (F6+F7+F8)',
            'Value': f"{mean_val:.0f}",
            'Std Dev': f"{np.std(mie):.0f}",
            'CV (%)': f"{(np.std(mie)/mean_val)*100:.1f}",
            'Unit': 'counts'
        })
        
        # Rayleigh/Mie ratio
        mean_val = np.mean(rayleigh_mie_ratio)
        summary_data.append({
            'Category': 'Scattering Analysis',
            'Parameter': 'Rayleigh/Mie Ratio',
            'Value': f"{mean_val:.3f}",
            'Std Dev': f"{np.std(rayleigh_mie_ratio):.3f}",
            'CV (%)': f"{(np.std(rayleigh_mie_ratio)/mean_val)*100:.1f}",
            'Unit': 'ratio'
        })
    
    # Stability analysis
    stability_params = [
        ('total_scatter', 'Total Scatter Stability'),
        ('short_long_ratio', 'Short/Long Ratio Stability')
    ]
    
    for param_key, param_name in stability_params:
        if param_key in features:
            stability = np.std(features[param_key]) / np.mean(features[param_key])
            summary_data.append({
                'Category': 'Stability Analysis',
                'Parameter': param_name,
                'Value': f"{stability:.3f}",
                'Std Dev': '',
                'CV (%)': '',
                'Unit': 'CV'
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(summary_data)
    df.to_csv(output_filename, index=False)
    
    print(f"Summary exported to: {output_filename}")
    print(f"Total parameters: {len(summary_data)}")
    
    return df

def print_summary(features, name):
    """Print formatted summary."""
    
    print(f"\n{'='*50}")
    print(f"MEASUREMENT SUMMARY: {name}")
    print(f"{'='*50}")
    print(f"Measurements: {len(features['total_scatter'])}")
    print(f"Sampling Rate: {features['sampling_rate']} Hz")
    
    print(f"\n1. DATA")
    for ch in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Clear']:
        if ch in features:
            print(f"   {ch}: {np.mean(features[ch]):.3f} ± {np.std(features[ch]):.3f} (CV: {np.std(features[ch])/np.mean(features[ch])*100:.1f}%)")
        else:
            print(f"   {ch}: Not available")

    # print(f"\n2. RATIOS")
    # for ch in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Clear']:
    #     if ch in features:
    #         for ch2 in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Clear']:
    #             if ch != ch2 and ch2 in features:
    #                 ratio_name = f"{ch} / {ch2}"
    #                 ratio_value = features[ch] / features[ch2]
    #                 print(f"   {ratio_name}: {np.mean(ratio_value):.3f} ± {np.std(ratio_value):.3f} (CV: {np.std(ratio_value)/np.mean(ratio_value)*100:.1f}%)")

    # print(f"\n3. SPECTRAL PROPERTIES")
    # print(f"   Centroid:                                    {np.mean(features['spectral_centroid']):.3f} ± {np.std(features['spectral_centroid']):.3f} nm (CV: {np.std(features['spectral_centroid'])/np.mean(features['spectral_centroid'])*100:.1f}%)")
    # print(f"   Width:                                       {np.mean(features['spectral_width']):.3f} ± {np.std(features['spectral_width']):.3f} nm (CV: {np.std(features['spectral_width'])/np.mean(features['spectral_width'])*100:.1f}%)")
    # print(f"   Polydispersity (Width / Centroid):           {np.mean(features['polydispersity_proxy']):.3f} ± {np.std(features['polydispersity_proxy']):.3f}")

    print(f"\n4. KEY FEATURES")
    print(f"total scatter: {np.mean(features['total_scatter']):.3f} ± {np.std(features['total_scatter']):.3f}")
    print(f"Violet/Far-Red Ratio: {np.mean(features['f1_f8_ratio']):.3f} ± {np.std(features['f1_f8_ratio']):.3f} (CV: {np.std(features['f1_f8_ratio'])/np.mean(features['f1_f8_ratio'])*100:.1f}%)")
    print(f"Green/Orange Ratio: {np.mean(features['f4_f7_ratio']):.3f} ± {np.std(features['f4_f7_ratio']):.3f} (CV: {np.std(features['f4_f7_ratio'])/np.mean(features['f4_f7_ratio'])*100:.1f}%)")
    print(f"Spectral centroid: {np.mean(features['spectral_centroid']):.3f} ± {np.std(features['spectral_centroid']):.3f} nm (CV: {np.std(features['spectral_centroid'])/np.mean(features['spectral_centroid'])*100:.1f}%)")

    print(f"\n5. STABILITY ANALYSIS")
    print(f"   Total Scatter Stability:                     {np.std(features['total_scatter']) / np.mean(features['total_scatter']):.3f}")
    print(f"   Short/Long Ratio Stability:                  {np.std(features['short_long_ratio']) / np.mean(features['short_long_ratio']):.3f}")

    print(f"\n6. SPECTRAL AGGLUTINATION SCORE")
    print(f"   Spectral Agglutination Score:                {np.mean(features['SAS']):.3f} ± {np.std(features['SAS']):.3f} (CV: {np.std(features['SAS'])/np.mean(features['SAS'])*100:.1f}%)")

    print(f"{'='*50}\n")
    

# Example usage and testing
if __name__ == "__main__":   
    import os
    import glob
    import matplotlib.pyplot as plt
    from feature_extraction_2 import AS7341FeatureExtractor

    analyzer = AS7341FeatureExtractor()

    # Define the folder path containing CSV files
    folder_path = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250604 WFBR experiments\perpendicular_spectrometer"

    # Get all CSV files in the folder
    csv_files = [f for f in glob.glob(os.path.join(folder_path, "*.csv") ) if "summary" not in os.path.basename(f).lower()]

    
    # for i, csv_file in enumerate(csv_files):
    #     print(f"Processing file {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
    #     result_df, summary, features = analyzer.analyze_file(csv_file)

    #     fig = analyzer.plot_analysis(features, title=f"Analysis: {os.path.basename(csv_file)}")
        
    #     print_summary(features, os.path.basename(csv_file))
    #     df = export_summary_to_csv(features, os.path.basename(csv_file), output_filename=os.path.join(folder_path, f"{os.path.basename(csv_file).replace('.csv', '')}_summary.csv"))



    # Test with sample data
    sample_file = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250604 WFBR experiments\perpendicular_spectrometer\0.45micron_running_buffer_just_sonified_(with CRP)_0.csv"
    # sample_file = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250604 WFBR experiments\perpendicular_spectrometer\0.45micron_running_buffer_just_sonified_(no CRP)_0.csv"
    # sample_file = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250604 WFBR experiments\perpendicular_spectrometer\0.45micron_running_buffer_just_sonified_0.csv"
    # sample_file = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250604 WFBR experiments\perpendicular_spectrometer\0.45micron_running_buffer_just_sonified_1.csv"
    # sample_file = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250604 WFBR experiments\perpendicular_spectrometer\0.45micron_running_buffer_unsonified_(possibly clustered).csv"
    # sample_file = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250604 WFBR experiments\perpendicular_spectrometer\0.45micron_tween_buffer_no_salt_0.csv"
    # sample_file = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250604 WFBR experiments\perpendicular_spectrometer\0.45micron_tween_buffer_salt_0.csv"
    sample_file = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250604 WFBR experiments\perpendicular_spectrometer\running_buffer_blanco_0.csv"

    result_df, summary, features = analyzer.analyze_file(sample_file)

    fig = analyzer.plot_analysis(features, title=f"Analysis: {os.path.basename(sample_file)}")

    print_summary(features, os.path.basename(sample_file))

    plt.show()