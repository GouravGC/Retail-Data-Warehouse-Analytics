"""
Logging configuration for the project.

Sets up both file and console logging handlers with
a standardised format that includes timestamp, module, level, and message.
"""

import logging
from pathlib import Path

from src.constants.paths import LOGS_DIR, LOG_FILE_PATH


def setup_logging() -> None:
    """
    Configure the root logger with file and console handlers.

    The log file is created under ``LOGS_DIR`` with a timestamped name.
    Console output is set to INFO level, file output to DEBUG level.
    """
    # Ensure the logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_format = (
        "[%(asctime)s] %(name)s - %(levelname)s - "
        "%(module)s::%(funcName)s:%(lineno)d - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # ---- File handler ----
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # ---- Console handler ----
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # ---- Root logger ----
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Convenience function to get a named logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("This is an info message")
    """
    return logging.getLogger(name)

