"""Logging setup: console + rotating file handler."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_dir: str | Path = "logs") -> logging.Logger:
    """Configure root logging once; safe to call repeatedly."""
    global _CONFIGURED
    root = logging.getLogger()
    if not _CONFIGURED:
        root.setLevel(logging.DEBUG)

        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root.addHandler(console)

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / "predictor.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)

        # Third-party noise reduction.
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("websocket").setLevel(logging.WARNING)
        _CONFIGURED = True

    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, RotatingFileHandler
        ):
            handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logging.getLogger("predictor")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"predictor.{name}")
