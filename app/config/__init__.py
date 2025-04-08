# config/__init__.py
"""
Configuration package for the nephelometer application.

This package provides configuration management and settings for the application.
"""

from config.settings import Settings, ConfigurationError, settings

__all__ = [
    'Settings',
    'ConfigurationError',
    'settings'
]