import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Channel specifications from datasheets
as7341_channels = {
    'F1': {'peak': 415, 'fwhm': 20},  # Estimated FWHM based on typical values
    'F2': {'peak': 445, 'fwhm': 20},
    'F3': {'peak': 480, 'fwhm': 20},
    'F4': {'peak': 515, 'fwhm': 20},
    'F5': {'peak': 555, 'fwhm': 20},
    'F6': {'peak': 590, 'fwhm': 20},
    'F7': {'peak': 630, 'fwhm': 20},
    'F8': {'peak': 680, 'fwhm': 20},
    'NIR': {'peak': 910, 'fwhm': 40}
}

as7343_channels = {
    'F1': {'peak': 405, 'fwhm': 30},   # From datasheet
    'F2': {'peak': 425, 'fwhm': 22},
    'FZ': {'peak': 450, 'fwhm': 55},
    'F3': {'peak': 475, 'fwhm': 30},
    'F4': {'peak': 515, 'fwhm': 40},
    'F5': {'peak': 550, 'fwhm': 35},
    'FY': {'peak': 555, 'fwhm': 100},
    'FXL': {'peak': 600, 'fwhm': 80},
    'F6': {'peak': 640, 'fwhm': 50},
    'F7': {'peak': 690, 'fwhm': 55},
    'F8': {'peak': 745, 'fwhm': 60},
    'NIR': {'peak': 855, 'fwhm': 54}
}

def gaussian_response(wavelengths, peak, fwhm):
    """
    Generate Gaussian response curve for a spectral channel
    
    Args:
        wavelengths: Array of wavelength values
        peak: Peak wavelength (nm)
        fwhm: Full Width Half Maximum (nm)
    
    Returns:
        Normalized response values
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    response = np.exp(-0.5 * ((wavelengths - peak) / sigma) ** 2)
    return response

def plot_spectral_channels():
    """
    Create plots showing spectral response channels for AS7341 and AS7343
    """
    # Wavelength range
    wavelengths = np.arange(350, 1000, 1)
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    # fig.suptitle('Spectral Response Comparison: AS7341 vs AS7343', 
    #              fontsize=16, fontweight='bold')
    
    # Colors for different channels
    colors_7341 = plt.cm.tab10(np.linspace(0, 1, len(as7341_channels)))
    colors_7343 = plt.cm.Set3(np.linspace(0, 1, len(as7343_channels)))
    
    # Plot AS7341 channels
    ax1.set_title('Normalized spectral response of AS7341', fontsize=14, fontweight='bold')
    for i, (channel, specs) in enumerate(as7341_channels.items()):
        response = gaussian_response(wavelengths, specs['peak'], specs['fwhm'])
        ax1.plot(wavelengths, response, 
                label=f"{channel} ({specs['peak']}nm)", 
                color=colors_7341[i], 
                linewidth=2.5)
        
        # Add peak markers
        ax1.axvline(x=specs['peak'], color=colors_7341[i], 
                   linestyle='--', alpha=0.7, linewidth=1)
    
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Normalized Response', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xlim(350, 1000)
    ax1.set_ylim(0, 1.1)
    
    # Add visible spectrum background
    vis_range = np.arange(400, 700, 1)
    ax1.fill_between(vis_range, 0, 0.1, alpha=0.2, color='yellow', 
                    label='Visible Spectrum (~400-700nm)')
    
    # Plot AS7343 channels
    ax2.set_title('Normalized spectral response of AS7343', fontsize=14, fontweight='bold')
    for i, (channel, specs) in enumerate(as7343_channels.items()):
        response = gaussian_response(wavelengths, specs['peak'], specs['fwhm'])
        ax2.plot(wavelengths, response, 
                label=f"{channel} ({specs['peak']}nm)", 
                color=colors_7343[i], 
                linewidth=2.5)
        
        # Add peak markers
        ax2.axvline(x=specs['peak'], color=colors_7343[i], 
                   linestyle='--', alpha=0.7, linewidth=1)
    
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Normalized Response', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xlim(350, 1000)
    ax2.set_ylim(0, 1.1)
    
    # Add visible spectrum background
    ax2.fill_between(vis_range, 0, 0.1, alpha=0.2, color='yellow', 
                    label='Visible Spectrum (~400-700nm)')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    return fig

def print_channel_summary():
    """
    Print a summary of channel specifications
    """
    print("=" * 60)
    print("SPECTRAL SENSOR CHANNEL COMPARISON")
    print("=" * 60)
    
    print("\nAS7341 (11 Channels):")
    print("-" * 30)
    for channel, specs in as7341_channels.items():
        print(f"{channel:>4}: {specs['peak']:>3}nm (FWHM: {specs['fwhm']:>2}nm)")
    
    print(f"\nTotal channels: {len(as7341_channels)}")
    print(f"Visible channels: {len([c for c in as7341_channels.values() if c['peak'] < 750])}")
    
    print("\nAS7343 (14 Channels):")
    print("-" * 30)
    for channel, specs in as7343_channels.items():
        print(f"{channel:>4}: {specs['peak']:>3}nm (FWHM: {specs['fwhm']:>2}nm)")
    
    print(f"\nTotal channels: {len(as7343_channels)}")
    print(f"Visible channels: {len([c for c in as7343_channels.values() if c['peak'] < 750])}")
    
    print("\n" + "=" * 60)
    print("KEY DIFFERENCES:")
    print("• AS7343 has 3 additional spectral channels")
    print("• Better coverage in violet/blue region (405-475nm)")
    print("• Higher resolution across visible spectrum")
    print("• Extended near-IR coverage (745nm + 855nm)")
    print("=" * 60)

def main():
    """
    Main function to run the spectral analysis
    """
    print("Generating spectral response plots...")
    
    # Print channel summary
    print_channel_summary()
    
    # Create and display plots
    fig = plot_spectral_channels()
    
    # Save the plot
    try:
        plt.savefig('as7341_vs_as7343_spectral_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print("\nPlot saved as 'as7341_vs_as7343_spectral_comparison.png'")
    except Exception as e:
        print(f"\nCould not save plot: {e}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()