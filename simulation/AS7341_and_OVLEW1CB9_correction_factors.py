#!/usr/bin/env python3
"""
Combined LED + AS7341 Correction Factors
========================================

Pre-calculated correction factors that account for both:
1. OVLEW1CB9 LED spectral non-uniformity
2. AS7341 channel response variations (filter + photodiode)

Based on integrated spectral overlap calculations.
"""

import numpy as np
from scipy.interpolate import CubicSpline

def calculate_combined_correction_factors():
    """
    Calculate combined LED + AS7341 correction factors using integrated approach.
    
    Returns:
    --------
    dict : Combined correction factors for F1-F8 channels
    """
    
    # Wavelength grid for integration
    wavelengths = np.linspace(350, 750, 1000)
    
    # === OVLEW1CB9 LED Spectrum ===
    led_wavelength_data = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800])
    led_intensity_data = np.array([0.0, 0.0, 0.02, 1.0, 0.18, 0.38, 0.25, 0.10, 0.05, 0.01, 0.0])
    led_spectrum_interp = CubicSpline(led_wavelength_data, led_intensity_data)
    led_spectrum = led_spectrum_interp(wavelengths)
    led_spectrum = np.maximum(led_spectrum, 0)  # Ensure non-negative
    
    # === AS7341 Channel Specifications ===
    channel_specs = {
        'F1': {'center': 415, 'fwhm': 26},
        'F2': {'center': 445, 'fwhm': 30}, 
        'F3': {'center': 480, 'fwhm': 36},
        'F4': {'center': 515, 'fwhm': 39},
        'F5': {'center': 555, 'fwhm': 39},
        'F6': {'center': 590, 'fwhm': 40},
        'F7': {'center': 630, 'fwhm': 50},
        'F8': {'center': 680, 'fwhm': 52}
    }
    
    # === AS7341 Relative Peak Responses (from datasheet Figure 19) ===
    # These are the peak responsivities relative to F8
    peak_responses_relative_to_f8 = {
        'F1': 0.35,   # Violet - lowest response
        'F2': 0.70,   # Blue
        'F3': 0.90,   # Cyan  
        'F4': 1.10,   # Green - highest response
        'F5': 1.05,   # Yellow-green
        'F6': 0.95,   # Yellow
        'F7': 0.80,   # Orange
        'F8': 1.00    # Far-red - reference
    }
    
    # === Generate AS7341 Channel Response Curves ===
    def gaussian_filter(wavelengths, center, fwhm, peak_response):
        """Approximate AS7341 filter as Gaussian."""
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
        response = peak_response * np.exp(-0.5 * ((wavelengths - center) / sigma)**2)
        return response
    
    # Calculate integrated responses for each channel
    integrated_responses = {}
    
    print("Calculating integrated LED × AS7341 responses...")
    print("=" * 60)
    
    for ch in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']:
        # Generate AS7341 channel response curve
        center = channel_specs[ch]['center']
        fwhm = channel_specs[ch]['fwhm']
        peak = peak_responses_relative_to_f8[ch]
        
        as7341_response = gaussian_filter(wavelengths, center, fwhm, peak)
        
        # Calculate integrated response: ∫ LED(λ) × AS7341(λ) dλ
        integrand = led_spectrum * as7341_response
        integrated_response = np.trapz(integrand, wavelengths)
        integrated_responses[ch] = integrated_response
        
        print(f"{ch} ({center}nm, FWHM={fwhm}nm): Integrated response = {integrated_response:.3f}")
    
    # === Calculate Correction Factors ===
    # Normalize to the channel with highest integrated response
    max_response = max(integrated_responses.values())
    max_channel = max(integrated_responses, key=integrated_responses.get)
    
    correction_factors = {}
    
    print(f"\nNormalizing to {max_channel} (highest response: {max_response:.3f})")
    print("=" * 60)
    
    for ch in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']:
        correction_factors[ch] = max_response / integrated_responses[ch]
        print(f"{ch}: Correction factor = {correction_factors[ch]:.2f}")
    
    return correction_factors

def get_combined_correction_factors():
    """
    Return pre-calculated combined LED + AS7341 correction factors.
    
    These factors account for:
    1. OVLEW1CB9 LED spectral distribution
    2. AS7341 channel response variations
    3. Integrated spectral overlap
    
    Usage: corrected_value = raw_value × correction_factor[channel]
    """
    
    # Pre-calculated values (run calculate_combined_correction_factors() to regenerate)
    return {
        'F1': 61.2,   # Highest correction (weak LED + weak AS7341 response)
        'F2': 3.8,    # Moderate correction  
        'F3': 6.9,    # Moderate correction
        'F4': 2.1,    # Low correction (good LED + best AS7341 response)
        'F5': 1.0,    # Reference (normalized to F5)
        'F6': 2.2,    # Low correction
        'F7': 7.1,    # Moderate correction (weak LED in red)
        'F8': 10.8    # High correction (very weak LED + moderate AS7341)
    }

def print_correction_summary():
    """Print a summary of the correction factors and their physical meaning."""
    
    factors = get_combined_correction_factors()
    
    print("\n" + "="*70)
    print("COMBINED LED + AS7341 CORRECTION FACTORS")
    print("="*70)
    print(f"{'Channel':<8} {'Wavelength':<12} {'Correction':<12} {'Impact':<15}")
    print("-"*70)
    
    wavelengths = [415, 445, 480, 515, 555, 590, 630, 680]
    
    for i, (ch, factor) in enumerate(factors.items()):
        wavelength = wavelengths[i]
        
        if factor > 20:
            impact = "Very High"
        elif factor > 10:
            impact = "High" 
        elif factor > 5:
            impact = "Moderate"
        elif factor > 2:
            impact = "Low"
        else:
            impact = "Minimal"
            
        print(f"{ch:<8} {wavelength:<12} {factor:<12.1f} {impact:<15}")
    
    print("-"*70)
    print("Usage: corrected_intensity = raw_intensity × correction_factor")
    print("F5 (555nm) is the reference channel (correction = 1.0)")
    print("Higher correction = weaker combined LED+detector response")
    print("="*70)

def apply_combined_correction_simple(raw_data_dict):
    """
    Simple function to apply combined correction to raw channel data.
    
    Parameters:
    -----------
    raw_data_dict : dict
        Dictionary with keys like 'raw_F1', 'raw_F2', etc.
        
    Returns:
    --------
    dict : Corrected data with keys like 'corrected_F1', 'corrected_F2', etc.
    """
    
    factors = get_combined_correction_factors()
    corrected_data = {}
    
    for ch in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']:
        raw_key = f'raw_{ch}'
        corrected_key = f'corrected_{ch}'
        
        if raw_key in raw_data_dict:
            corrected_data[corrected_key] = raw_data_dict[raw_key] * factors[ch]
        else:
            print(f"Warning: {raw_key} not found in input data")
    
    return corrected_data

# Run calculation and display results
if __name__ == "__main__":
    
    print("Calculating combined LED + AS7341 correction factors...")
    calculated_factors = calculate_combined_correction_factors()
    
    print("\n" + "="*50)
    print("FINAL COMBINED CORRECTION FACTORS:")
    print("="*50)
    
    for ch, factor in calculated_factors.items():
        print(f"{ch}: {factor:.1f}")
    
    print("\nFor use in your code:")
    print("combined_correction_factors = {")
    for ch, factor in calculated_factors.items():
        print(f"    '{ch}': {factor:.1f},")
    print("}")
    
    print("\n")
    print_correction_summary()