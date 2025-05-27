import logging
import sys
from pathlib import Path
from datetime import datetime
import logging.handlers
import traceback
import os
import threading
from typing import Optional, Dict, Any

from config import settings

class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds contextual information to log records.
    
    This adapter allows adding experiment ID, sample ID, and other
    contextual information to log messages.
    """
    def process(self, msg, kwargs):
        # Add any extra context to the message
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        extra = kwargs.get('extra', {})

        # Ensure experiment_id exists
        if 'experiment_id' not in extra:
            extra['experiment_id'] = 'none'        
        
        # Include thread name for async operations
        if 'thread_name' not in extra:
            extra['thread_name'] = threading.current_thread().name
            
        # Include process ID for multiprocessing
        if 'process_id' not in extra:
            extra['process_id'] = os.getpid()
            
        return msg, kwargs

class ExperimentLoggerAdapter(LoggerAdapter):
    """
    Specialized logger adapter for experiment-related logging.
    """
    def __init__(self, logger, experiment_id=None, sample_id=None):
        """
        Initialize with experiment context.
        
        Args:
            logger: Base logger to adapt
            experiment_id: Current experiment ID
            sample_id: Current sample ID
        """
        extra = {
            'experiment_id': experiment_id or 'none',
            'sample_id': sample_id or 'none'
        }
        super().__init__(logger, extra)
    
    def process(self, msg, kwargs):
        # Ensure we have the latest experiment context
        extra = kwargs.get('extra', {})
        if 'experiment_id' not in extra and self.extra.get('experiment_id'):
            extra['experiment_id'] = self.extra.get('experiment_id')
        if 'sample_id' not in extra and self.extra.get('sample_id'):
            extra['sample_id'] = self.extra.get('sample_id')
        
        kwargs['extra'] = extra
        return super().process(msg, kwargs)
    
    def set_experiment(self, experiment_id):
        """Update the experiment ID for this logger."""
        self.extra['experiment_id'] = experiment_id
    
    def set_sample(self, sample_id):
        """Update the sample ID for this logger."""
        self.extra['sample_id'] = sample_id

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to console output.
    """
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color if this is a console handler
        if hasattr(self, 'is_console') and self.is_console:
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)

def _get_detailed_formatter(for_console=False):
    """Create a detailed formatter for logs, optionally with colors."""
    fmt = settings.LOG_FORMAT
    
    # Add experiment info if available, but only when running experiments
    if 'experiment_id' in settings.get('LOG_FORMAT_EXTRAS', []) and 'tests' not in __name__:
        fmt = "%(asctime)s - [%(experiment_id)s] - %(name)s - %(levelname)s - %(message)s"
    
    formatter = ColoredFormatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    if for_console:
        formatter.is_console = True
    return formatter

def setup_logging() -> logging.Logger:
    """
    Set up structured logging for the application.
    
    Returns:
        logging.Logger: Configured root logger
    """
    # Get log level from settings
    level_name = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, level_name, logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.LOG_DIR)
    
    # Try to create the directory - but handle permission errors gracefully
    try:
        log_dir.mkdir(exist_ok=True, parents=True)
    except PermissionError:
        print(f"Warning: Cannot create log directory at {log_dir}. Using current directory.")
        log_dir = Path('.')
    except Exception as e:
        print(f"Warning: Error creating log directory: {str(e)}. Using current directory.")
        log_dir = Path('.')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    handlers = []
    
    # Create console handler if enabled
    if settings.LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = _get_detailed_formatter(for_console=True)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    # Try to set up file logging if enabled
    if settings.LOG_TO_FILE:
        try:
            # Create daily rotating file handler
            log_file = log_dir / f"nephelometer_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_file, when='midnight', backupCount=7, encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_formatter = _get_detailed_formatter(for_console=False)
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
            
            # Add error log handler for warnings and above
            error_log_file = log_dir / f"nephelometer_errors_{datetime.now().strftime('%Y%m%d')}.log"
            error_handler = logging.handlers.TimedRotatingFileHandler(
                error_log_file, when='midnight', backupCount=7, encoding='utf-8'
            )
            error_handler.setLevel(logging.WARNING)
            error_handler.setFormatter(file_formatter)
            handlers.append(error_handler)
            
        except Exception as e:
            # If file logging fails, log a warning
            print(f"Warning: Could not set up file logging: {str(e)}. Using console logging only.")
    
    # Add all handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Suppress excessive logging from libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Create and return application logger
    logger = logging.getLogger("nephelometer")
    logger.debug("Logging initialized. Log directory: %s", log_dir)
    
    # Handle uncaught exceptions
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupt as error
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
            extra={'experiment_id': 'uncaught_exception'}  # Or some other default value
        )
    
    sys.excepthook = exception_handler
    
    return logger

def get_logger(name: str, experiment_id: Optional[str] = None, 
               sample_id: Optional[str] = None) -> logging.LoggerAdapter:
    """
    Get a logger with the specified name and optional experiment context.
    
    Args:
        name: Logger name, typically module name
        experiment_id: Optional ID for the current experiment
        sample_id: Optional ID for the current sample
        
    Returns:
        LoggerAdapter: Configured logger adapter with contextual information
    """
    logger = logging.getLogger(name)
    
    if experiment_id or sample_id:
        return ExperimentLoggerAdapter(logger, experiment_id, sample_id)
    return LoggerAdapter(logger, {})