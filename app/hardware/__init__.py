# hardware/__init__.py
"""
Hardware package for the nephelometer application.

This package provides interfaces to hardware devices used in the nephelometer,
including the AS7341 spectral sensor.
"""

from hardware.as7341 import AS7341, AS7341Error
from hardware.nephelometer import (
    Nephelometer, NephelometerError, AgglutinationState, MeasurementMode
)

__all__ = [
    # AS7341 spectral sensor
    'AS7341', 'AS7341Error',
    
    # Nephelometer
    'Nephelometer', 'NephelometerError', 'AgglutinationState', 'MeasurementMode'
]