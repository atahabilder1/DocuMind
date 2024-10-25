"""
Logging and error handling utilities for DocuMind.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import traceback


class Logger:
    """
    Custom logger for DocuMind with structured logging.
    """

    def __init__(
        self,
        name: str = "DocuMind",
        log_dir: str = "logs",
        level: int = logging.INFO
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove existing handlers
        self.logger.handlers.clear()

        # Setup handlers
        self._setup_file_handler()
        self._setup_console_handler()

    def _setup_file_handler(self):
        """Setup file handler for logging to files."""
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def _setup_console_handler(self):
        """Setup console handler for logging to stdout."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message."""
        self.logger.error(message, exc_info=exc_info, extra=kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """Log critical message."""
        self.logger.critical(message, exc_info=exc_info, extra=kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)


class ErrorHandler:
    """
    Handle and log errors in DocuMind operations.
    """

    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize error handler.

        Args:
            logger: Logger instance
        """
        self.logger = logger or Logger()

    def handle_error(
        self,
        error: Exception,
        context: str = "",
        reraise: bool = False
    ) -> dict:
        """
        Handle an error with logging.

        Args:
            error: Exception that occurred
            context: Context where error occurred
            reraise: Whether to re-raise the exception

        Returns:
            Error details dictionary
        """
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }

        # Log the error
        self.logger.error(
            f"Error in {context}: {error_details['error_message']}",
            exc_info=True
        )

        if reraise:
            raise error

        return error_details

    def validate_input(
        self,
        value: any,
        value_type: type,
        name: str = "input"
    ) -> bool:
        """
        Validate input type and log errors.

        Args:
            value: Value to validate
            value_type: Expected type
            name: Name of the value

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(value, value_type):
            error_msg = f"Invalid {name}: expected {value_type.__name__}, got {type(value).__name__}"
            self.logger.error(error_msg)
            return False
        return True

    def log_operation(
        self,
        operation: str,
        status: str,
        details: dict = None
    ):
        """
        Log an operation with status.

        Args:
            operation: Operation name
            status: Operation status (success, failed, etc.)
            details: Additional details
        """
        message = f"Operation '{operation}' {status}"
        if details:
            message += f" - {details}"

        if status == "success":
            self.logger.info(message)
        elif status == "failed":
            self.logger.error(message)
        else:
            self.logger.warning(message)


# Global logger instance
_global_logger = None


def get_logger(name: str = "DocuMind") -> Logger:
    """
    Get or create global logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger(name)
    return _global_logger
