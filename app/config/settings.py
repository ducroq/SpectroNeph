import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import sys

class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""
    pass

class Settings:
    """
    Application settings management.
    
    This class handles loading and providing access to application settings
    from a combination of environment variables, configuration files, and defaults.
    """
    # Default settings
    DEFAULTS = {
        # Application settings
        "APP_NAME": "AS7341 Nephelometer",
        "APP_VERSION": "0.1.0",
        "DEBUG": False,
        
        # Logging settings
        "LOG_DIR": "logs",
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "LOG_TO_CONSOLE": True,
        "LOG_TO_FILE": True,
        
        # Hardware settings
        "DEFAULT_BAUDRATE": 115200,
        "SERIAL_TIMEOUT": 1.0,
        "AUTO_CONNECT": True,
        "CONNECTION_RETRIES": 3,
        
        # AS7341 sensor defaults
        "DEFAULT_INTEGRATION_TIME": 100,  # ms
        "DEFAULT_GAIN": 16,  # x16 gain
        "DEFAULT_LED_CURRENT": 10,  # mA
        
        # Data acquisition settings
        "DEFAULT_SAMPLING_RATE": 1.0,  # Hz
        "MAX_SAMPLING_RATE": 10.0,  # Hz
        "BUFFER_SIZE": 1000,  # samples
        
        # Data processing settings
        "ENABLE_BACKGROUND_SUBTRACTION": True,
        "ENABLE_DARK_CORRECTION": True,
        "DEFAULT_MOVING_AVERAGE_WINDOW": 5,
        
        # UI settings
        "PLOT_UPDATE_INTERVAL": 500,  # ms
        "THEME": "dark",
        "MAX_POINTS_DISPLAYED": 1000,
        
        # File paths
        "DATA_DIR": "data",
        "EXPERIMENT_TEMPLATES_DIR": "experiments/templates",
        "CONFIG_PROFILES_DIR": "config/profiles",
    }
    
    def __init__(self):
        """Initialize settings with defaults, then override from files and environment."""
        self._settings = self.DEFAULTS.copy()
        self._load_from_yaml()
        self._load_from_env()
        
        # Ensure critical directories exist
        self._ensure_directories()
    
    def _load_from_yaml(self):
        """Load settings from YAML configuration files."""
        config_paths = [
            Path(__file__).parent / "default_config.yaml",  # Default config
            Path.home() / ".nephelometer" / "config.yaml",  # User config
            Path("config.yaml")  # Project-level config
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        yaml_settings = yaml.safe_load(f)
                        if yaml_settings:
                            self._settings.update(yaml_settings)
                except Exception as e:
                    print(f"Warning: Could not load configuration from {config_path}: {str(e)}")
    
    def _load_from_env(self):
        """Override settings from environment variables."""
        for key in self._settings.keys():
            env_value = os.environ.get(f"NEPHELOMETER_{key}")
            if env_value is not None:
                # Try to convert to the same type as the default
                default_type = type(self._settings[key])
                if default_type == bool:
                    self._settings[key] = env_value.lower() in ('true', 'yes', '1', 'y')
                else:
                    try:
                        self._settings[key] = default_type(env_value)
                    except (ValueError, TypeError):
                        # If conversion fails, use string value
                        self._settings[key] = env_value
    
    def _ensure_directories(self):
        """Ensure that critical directories exist."""
        directories = [
            self._settings["LOG_DIR"],
            self._settings["DATA_DIR"],
            self._settings["CONFIG_PROFILES_DIR"]
        ]
        
        for dir_path in directories:
            path = Path(dir_path)
            try:
                path.mkdir(exist_ok=True, parents=True)
            except Exception as e:
                print(f"Warning: Could not create directory {dir_path}: {str(e)}")
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to settings."""
        if name in self._settings:
            return self._settings[name]
        raise AttributeError(f"Setting '{name}' not found")
    
    def get(self, name: str, default: Any = None) -> Any:
        """Dictionary-style access to settings with default value."""
        return self._settings.get(name, default)
    
    def as_dict(self) -> Dict[str, Any]:
        """Return all settings as a dictionary."""
        return self._settings.copy()
    
    def update(self, settings_dict: Dict[str, Any]) -> None:
        """Update settings from a dictionary."""
        self._settings.update(settings_dict)
    
    def load_profile(self, profile_name: str) -> bool:
        """
        Load a specific configuration profile.
        
        Args:
            profile_name: Name of the profile to load
            
        Returns:
            bool: True if profile was loaded successfully
        """
        profile_path = Path(self._settings["CONFIG_PROFILES_DIR"]) / f"{profile_name}.yaml"
        
        if not profile_path.exists():
            return False
            
        try:
            with open(profile_path, 'r') as f:
                profile_settings = yaml.safe_load(f)
                if profile_settings:
                    self._settings.update(profile_settings)
            return True
        except Exception:
            return False
    
    def save_profile(self, profile_name: str, settings: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save current or provided settings to a profile.
        
        Args:
            profile_name: Name to save the profile as
            settings: Specific settings to save, or None for all current settings
            
        Returns:
            bool: True if profile was saved successfully
        """
        profile_dir = Path(self._settings["CONFIG_PROFILES_DIR"])
        
        try:
            profile_dir.mkdir(exist_ok=True, parents=True)
            profile_path = profile_dir / f"{profile_name}.yaml"
            
            with open(profile_path, 'w') as f:
                yaml.dump(settings or self._settings, f, default_flow_style=False)
            return True
        except Exception:
            return False

# Create a singleton instance
settings = Settings()

# config/__init__.py
from .settings import settings, ConfigurationError

__all__ = ['settings', 'ConfigurationError']