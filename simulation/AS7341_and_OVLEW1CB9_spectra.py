import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt

def channel_response(x, peak, fwhm, max_val=1.0):
    """
    Generate a gaussian-like response curve for a sensor channel,
    where the width parameter is the Full Width at Half Maximum (FWHM).
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return max_val * np.exp(-0.5 * ((x - peak) / sigma) ** 2)

# Create wavelength range (350-1050 nm)
wavelength = np.linspace(350, 1050, 1000)

# Create response curves for each channel
# Parameters from AS7341data sheet (peak wavelength, width)
f1 = channel_response(wavelength, 415, 26)  # Purple
f2 = channel_response(wavelength, 445, 30)  # Dark blue
f3 = channel_response(wavelength, 480, 36)  # Blue
f4 = channel_response(wavelength, 515, 39)  # Cyan
f5 = channel_response(wavelength, 555, 39)  # Green
f6 = channel_response(wavelength, 590, 40)  # Yellow
f7 = channel_response(wavelength, 630, 50)  # Orange
f8 = channel_response(wavelength, 680, 52)  # Red
nir = channel_response(wavelength, 910, 50)  # NIR - wider band, not really Gaussian though

# Define the data points that can be roughly estimated from the graph in the AS7341 datasheet
wavelength_data = np.array([
    350, 450, 550, 650, 750, 850, 950, 1050
])
flicker_data = np.array([
    0.0, 0.18, 0.57, 0.86, 1.0, 0.85, 0.4, 0.05
])
clear_data = np.array([
    0.39, 0.6, 0.8, 0.96, 0.99, 0.8, 0.38, 0.05
])

# Create interpolation functions
flicker_interp = CubicSpline(wavelength_data, flicker_data) 
clear_interp = CubicSpline(wavelength_data, clear_data)

# Interpolate the relative sensitivity values
flicker = flicker_interp(wavelength)
clear = clear_interp(wavelength)

# Define the data points that can be roughly estimated from the graph in the OVLEW1CB9 datasheet
wavelength_data = np.array([
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000
])

intensity_data = np.array([
    0.0, 0.0, 0.02, 1.0, 0.18, 0.38, 0.25, 0.10, 0.05, 0.01, 0.0, 0.0, 0.0
])

# Create interpolation functions
intensity_interp = CubicSpline(wavelength_data, intensity_data) 

# Interpolate the relative sensitivity values
intensity = intensity_interp(wavelength)

# Plot the graph
plt.figure(figsize=(12, 8))
plt.plot(wavelength, f1, '-', color='indigo', linewidth=2, label='F1')
plt.plot(wavelength, f2, '-', color='navy', linewidth=2, label='F2')
plt.plot(wavelength, f3, '-', color='dodgerblue', linewidth=2, label='F3')
plt.plot(wavelength, f4, '-', color='aqua', linewidth=2, label='F4')
plt.plot(wavelength, f5, '-', color='green', linewidth=2, label='F5')
plt.plot(wavelength, f6, '-', color='gold', linewidth=2, label='F6')
plt.plot(wavelength, f7, '-', color='orange', linewidth=2, label='F7')
plt.plot(wavelength, f8, '-', color='red', linewidth=2, label='F8')
plt.plot(wavelength, clear, '--', color='gray', linewidth=2, label='Clear')
plt.plot(wavelength, nir, '-', color='darkred', linewidth=2, label='NIR')
plt.plot(wavelength, flicker, '-', color='darkblue', linewidth=2, label='Flicker')
plt.plot(wavelength, intensity, label='Luminous Intensity', color='blue', linewidth=2)
plt.fill_between(wavelength, intensity, color='blue', alpha=0.1)
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')

# Set labels and title
plt.xlabel('wavelength [nm]', fontsize=12)
plt.ylabel('relative sensitivity', fontsize=12)
plt.title('Normalized Spectral Responsivity (estimated from datasheets)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(350, 1050)
plt.ylim(0, 1.1)

# Add legend
plt.legend(ncol=3)

# Show the plot
plt.tight_layout()
plt.show()