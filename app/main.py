import sys
import argparse
import time
from pathlib import Path

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import settings
from utils.logging import setup_logging, get_logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AS7341 Nephelometer Control Application')
    
    # Configuration options
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--profile', help='Configuration profile to load')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Connection options
    parser.add_argument('--port', help='Serial port to connect to')
    parser.add_argument('--baudrate', type=int, help='Serial baudrate')
    parser.add_argument('--no-connect', action='store_true', 
                        help='Don\'t automatically connect to device')
    
    # Application mode options
    parser.add_argument('--headless', action='store_true', 
                        help='Run in headless mode (no GUI)')
    parser.add_argument('--experiment', help='Path to experiment definition file to run')
    
    return parser.parse_args()

def apply_command_line_settings(args):
    """Apply command line arguments to settings."""
    if args.debug:
        settings.update({"DEBUG": True, "LOG_LEVEL": "DEBUG"})
    
    if args.port:
        settings.update({"DEFAULT_PORT": args.port})
        
    if args.baudrate:
        settings.update({"DEFAULT_BAUDRATE": args.baudrate})
        
    if args.no_connect:
        settings.update({"AUTO_CONNECT": False})
        
    # Load specified configuration profile
    if args.profile:
        if not settings.load_profile(args.profile):
            print(f"Warning: Could not load profile '{args.profile}'")

def initialize_application():
    """Initialize core application components."""
    # Set up logging first
    logger = setup_logging()
    app_logger = get_logger("main")
    
    # Log startup information
    app_logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    
    if settings.DEBUG:
        app_logger.debug("Debug mode enabled")
        app_logger.debug("Current settings: %s", settings.as_dict())
    
    # Import components only after logging is set up
    try:
        if not settings.get("HEADLESS_MODE", False):
            # Import and initialize UI (for GUI mode)
            from ui.app import NephelometerApp
            app_logger.info("Initializing UI...")
            return NephelometerApp()
        else:
            # Import headless controller (for script/automated mode)
            from core.headless import HeadlessController
            app_logger.info("Initializing headless mode...")
            return HeadlessController()
    except ImportError as e:
        app_logger.error("Failed to import required modules: %s", str(e))
        app_logger.error("Make sure all dependencies are installed")
        return None
    except Exception as e:
        app_logger.error("Initialization error: %s", str(e), exc_info=True)
        return None

def run_application(app):
    """Run the initialized application."""
    if app is None:
        return 1
        
    logger = get_logger("main")
    
    try:
        if isinstance(app, (str, int, float, bool)) or app is None:
            # If app is a primitive type, it's an error code or None
            return app
            
        # Check if the app has a run or start method
        if hasattr(app, 'run'):
            logger.debug("Starting application main loop...")
            return app.run()
        elif hasattr(app, 'start'):
            logger.debug("Starting application...")
            return app.start()
        else:
            logger.error("Application object has no run() or start() method")
            return 1
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        return 0
    except Exception as e:
        logger.error("Application error: %s", str(e), exc_info=True)
        return 1

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Apply command line settings
    apply_command_line_settings(args)
    
    # Special handling for headless mode
    if args.headless:
        settings.update({"HEADLESS_MODE": True})
    
    # Initialize and run the application
    app = initialize_application()
    exit_code = run_application(app)
    
    # Return exit code to the OS
    return exit_code

if __name__ == "__main__":
    sys.exit(main())