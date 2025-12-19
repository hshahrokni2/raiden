"""
Raiden Logging Configuration.

Provides consistent logging setup across all modules with:
- Structured log format with timestamps
- File and console handlers
- Log level configuration via environment variable
- Context-aware logging (building ID, operation type)

Usage:
    from src.utils.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Processing building", extra={"building_id": "123"})
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Default log level from environment
DEFAULT_LOG_LEVEL = os.environ.get("RAIDEN_LOG_LEVEL", "INFO").upper()

# Log directory
LOG_DIR = Path(os.environ.get("RAIDEN_LOG_DIR", "logs"))


class RaidenFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors and sys.stdout.isatty()
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    def format(self, record: logging.LogRecord) -> str:
        # Add extra context if available
        extras = []
        for key in ["building_id", "address", "ecm_id", "archetype_id"]:
            if hasattr(record, key):
                extras.append(f"{key}={getattr(record, key)}")
        if extras:
            record.msg = f"{record.msg} [{', '.join(extras)}]"

        formatted = super().format(record)

        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            return f"{color}{formatted}{self.RESET}"
        return formatted


class FileFormatter(logging.Formatter):
    """JSON-like formatter for file output."""

    def __init__(self):
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra context
        for key in ["building_id", "address", "ecm_id", "archetype_id", "error_type"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return str(log_data)


def setup_logging(
    level: str = DEFAULT_LOG_LEVEL,
    log_to_file: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for the entire application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to also log to a file
        log_file: Custom log file path (default: logs/raiden_YYYYMMDD.log)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level, logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(RaidenFormatter(use_colors=True))
    console_handler.setLevel(getattr(logging, level, logging.INFO))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        LOG_DIR.mkdir(exist_ok=True)
        if log_file is None:
            log_file = LOG_DIR / f"raiden_{datetime.now().strftime('%Y%m%d')}.log"
        else:
            log_file = Path(log_file)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(FileFormatter())
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Convenience function for one-time setup
_initialized = False


def ensure_logging() -> None:
    """Ensure logging is set up (call once at application start)."""
    global _initialized
    if not _initialized:
        setup_logging()
        _initialized = True


# Auto-initialize when imported
ensure_logging()
