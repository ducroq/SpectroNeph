nephelometer/
│
├── main.py                      # Application entry point
│
├── config/                                                                 ✓
│   ├── __init__.py              # Package initialization                   ✓
│   ├── settings.py              # Core settings and configuration          ✓
│   ├── default_config.yaml      # Default configuration values             ✓
│   └── profiles/                # Saved sensor configuration profiles      ✓
│       └── default.yaml         # Default sensor profile                   ✓
│
├── core/                                                                   ✓
│   ├── __init__.py                                                         ✓
│   ├── communication.py         # ESP32 serial communication               ✓
│   ├── device.py                # Device connection management             ✓
│   ├── protocol.py              # Communication protocol implementation    ✓
│   └── exceptions.py            # Custom exceptions                        ✓
│
├── hardware/                                                               ✓
│   ├── __init__.py                                                         ✓
│   ├── as7341.py                # AS7341 sensor interface                  ✓
│   └── nephelometer.py          # Nephelometer hardware abstractions       ✓
│
├── data/
│   ├── __init__.py                                                         ✓
│   ├── acquisition.py           # Data acquisition management              ✓
│   ├── processing.py            # Signal processing and algorithms         ✓
│   ├── storage.py               # Data persistence
│   └── export.py                # Data export functions
│
├── analysis/
│   ├── __init__.py
│   ├── statistics.py            # Statistical analysis tools
│   ├── agglutination.py         # Agglutination-specific analysis
│   ├── calibration.py           # Calibration procedures
│   └── particle_sizing.py       # Particle size estimation
│
├── visualization/
│   ├── __init__.py
│   ├── plots.py                 # Core plotting functionality
│   ├── real_time.py             # Real-time visualization
│   ├── spectral.py              # Spectral data visualization
│   └── export.py                # Plot export utilities
│
├── ui/
│   ├── __init__.py
│   ├── app.py                   # Main application window
│   ├── dashboard.py             # Main dashboard UI
│   ├── experiment_panel.py      # Experiment control panel
│   ├── visualization_panel.py   # Data visualization panel
│   ├── settings_panel.py        # Settings and configuration UI
│   └── dialogs/                 # Application dialogs
│       ├── __init__.py
│       ├── connection.py        # Device connection dialog
│       └── experiment_config.py # Experiment configuration dialog
│
├── utils/                                                              
│   ├── __init__.py                                                     
│   ├── logging.py               # Logging configuration                ✓
│   ├── validation.py            # Input validation utilities           ✓
│   └── helpers.py               # General helper functions             ✓
│
├── resources/
│   ├── ui/                      # UI resources (icons, etc.)
│   └── docs/                    # Documentation resources
│
└── logs/                        # Application logs directory


firmware/
├── src/
│   ├── main.cpp                 # Application entry point
│   ├── protocol.cpp             # Protocol implementation
│   ├── protocol.h               # Protocol declarations
│   ├── commands.cpp             # Command handler implementation
│   ├── commands.h               # Command handler declarations
│   ├── as7341.cpp               # AS7341 sensor driver
│   ├── as7341.h                 # AS7341 sensor declarations
│   ├── streaming.cpp            # Data streaming implementation
│   ├── streaming.h              # Data streaming declarations
│   └── config.h                 # Configuration parameters
│
├── include/                     # Empty directory for PlatformIO 
│
├── lib/                         # Third-party libraries
│   └── README.md                # Information about required libraries
│
├── test/                        # Test files
│   └── README.md                # Testing instructions
│
├── platformio.ini               # PlatformIO configuration
└── README.md                    # Setup and usage instructions