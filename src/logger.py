"""Centralized logging — AppLogger class wrapping Python's logging module."""
import logging
import sys


class AppLogger:
    """Configures and exposes a named application logger."""

    def __init__(self, level: str = "INFO"):
        self._logger = logging.getLogger("app")
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler = logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger for an internal module."""
    return logging.getLogger(f"app.{name}")
