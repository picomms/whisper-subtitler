"""
Logging configuration for whisper-subtitler.

This module provides a consistent logging configuration across all
components of the application with configurable levels and outputs.
"""

import logging
import os
import sys


class ColoredFormatter(logging.Formatter):
    """Logging formatter with colored output for terminal."""

    COLORS = {
        "DEBUG": "\033[38;5;246m",  # gray
        "INFO": "\033[38;5;255m",  # white
        "WARNING": "\033[38;5;214m",  # orange
        "ERROR": "\033[38;5;196m",  # red
        "CRITICAL": "\033[48;5;196m\033[38;5;231m",  # white on red
        "RESET": "\033[0m",
    }

    def format(self, record):
        """Format the log record with colors for terminal output."""
        if record.levelname in self.COLORS:
            record.levelname_colored = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            record.msg_colored = f"{self.COLORS[record.levelname]}{record.msg}{self.COLORS['RESET']}"
        else:
            record.levelname_colored = record.levelname
            record.msg_colored = record.msg

        return super().format(record)


def setup_logging(config=None, level: str | None = None, log_file: str | None = None) -> logging.Logger:
    """Set up logging with the specified configuration.

    Args:
        config: Configuration object with logging settings
        level: Override log level
        log_file: Override log file path

    Returns:
        Configured logger instance
    """
    # Get log level from config or parameter
    log_level = level
    if config and not log_level:
        log_level = config.log_level
    if not log_level:
        log_level = "INFO"

    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Get log file from config or parameter
    log_file_path = log_file
    if config and not log_file_path and config.log_file:
        log_file_path = config.log_file

    # Create logger
    logger = logging.getLogger("whisper_subtitler")
    logger.setLevel(numeric_level)
    logger.handlers = []  # Clear any existing handlers

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    # Create formatters
    if sys.stdout.isatty():
        # Use colored output for terminal
        console_formatter = ColoredFormatter(
            "%(asctime)s [%(levelname_colored)s] %(msg_colored)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        # Use plain text for pipes and redirects
        console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Set formatters for handlers
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)

    # Add file handler if log file is specified
    if log_file_path:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(numeric_level)

        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    logger.debug("Logging initialized")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger that inherits the main configuration.

    Args:
        name: The name of the logger component

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"whisper_subtitler.{name}")
