import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional


class SLMLogger:
    """
    Centralised logging system.

    This class provides a structured logging system with separate loggers
    for different components of the application.
    """

    def __init__(self, log_level: int = logging.DEBUG, console_level: int = logging.INFO):
        """Initialise the SLM logger."""
        self.main_logger: logging.Logger
        self.training_logger: logging.Logger
        self.gui_logger: logging.Logger
        self.testing_logger: logging.Logger
        self._log_level = log_level
        self._console_level = console_level
        self._setup_loggers()

    def _setup_loggers(self):
        """Setup the main logger and sub-loggers with proper error handling."""

        # Import here to avoid circular imports
        from .paths import ProjectPaths

        try:
            # Create logs directory if it doesn't exist
            logs_dir = ProjectPaths.get_logs_dir()
            os.makedirs(logs_dir, exist_ok=True)

            # Main logger configuration
            main_logger = logging.getLogger("slm")
            main_logger.setLevel(self._log_level)

            # Don't propagate to root logger to avoid duplicate messages
            main_logger.propagate = False

            # Clear any existing handlers to avoid duplicates
            main_logger.handlers.clear()

            # Create formatters
            detailed_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            simple_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self._console_level)
            console_handler.setFormatter(simple_formatter)
            main_logger.addHandler(console_handler)

            # File handler for all logs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(logs_dir, f"slm_{timestamp}.log")

            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
            )
            file_handler.setLevel(self._log_level)
            file_handler.setFormatter(detailed_formatter)
            main_logger.addHandler(file_handler)

            # Set instance attributes
            self.main_logger = main_logger

            # Create sub-loggers for different components
            self.training_logger = main_logger.getChild("training")
            self.gui_logger = main_logger.getChild("gui")
            self.testing_logger = main_logger.getChild("testing")

            # Set sub-logger levels (inherit from parent but can be overridden)
            self.training_logger.setLevel(self._log_level)
            self.gui_logger.setLevel(self._log_level)
            self.testing_logger.setLevel(self._log_level)

        except Exception as e:
            # Fallback to basic console logging if setup fails
            print(f"Warning: Failed to setup advanced logging: {e}")
            print("Falling back to basic console logging...")

            main_logger = logging.getLogger("slm")
            main_logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
            main_logger.addHandler(console_handler)

            self.main_logger = main_logger
            self.training_logger = main_logger.getChild("training")
            self.gui_logger = main_logger.getChild("gui")
            self.testing_logger = main_logger.getChild("testing")

    def get_training_logger(self) -> logging.Logger:
        """Get the training sub-logger."""
        return self.training_logger

    def get_gui_logger(self) -> logging.Logger:
        """Get the GUI sub-logger."""
        return self.gui_logger

    def get_testing_logger(self) -> logging.Logger:
        """Get the testing sub-logger."""
        return self.testing_logger

    def get_main_logger(self) -> logging.Logger:
        """Get the main logger."""
        return self.main_logger


# Global logger instance - kept for convenience functions
_logger_instance: Optional[SLMLogger] = None

def get_logger() -> SLMLogger:
    """Get the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = SLMLogger()
    return _logger_instance


def get_training_logger() -> logging.Logger:
    """Convenience function to get training logger."""
    return get_logger().get_training_logger()


def get_gui_logger() -> logging.Logger:
    """Convenience function to get GUI logger."""
    return get_logger().get_gui_logger()


def get_testing_logger() -> logging.Logger:
    """Convenience function to get testing logger."""
    return get_logger().get_testing_logger()


def get_main_logger() -> logging.Logger:
    """Convenience function to get main logger."""
    return get_logger().get_main_logger()


def create_logger(log_level: int = logging.DEBUG, console_level: int = logging.INFO) -> SLMLogger:
    """Create a new logger instance with custom settings."""
    return SLMLogger(log_level=log_level, console_level=console_level)


def reset_global_logger() -> None:
    """Reset the global logger instance."""
    global _logger_instance
    _logger_instance = None
