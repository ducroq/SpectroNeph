import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt

# Create wavelength range (350-1050 nm)
wavelength = np.linspace(350, 1050, 1000)

# Define the data points that can be roughly estimated from the graph in the datasheet
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
plt.plot(wavelength, intensity, label='Luminous Intensity', color='blue', linewidth=2)
plt.fill_between(wavelength, intensity, color='blue', alpha=0.1)
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')

# Set labels and title
plt.xlabel('wavelength [nm]', fontsize=12)
plt.ylabel('relative sensitivity', fontsize=12)
plt.title('Normalized Luminous Intensity (estimated from OVLEW1CB9 datasheet)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(350, 1050)
plt.ylim(0, 1.1)

# Add legend
plt.legend(ncol=3)

# Show the plot
plt.tight_layout()
plt.show()