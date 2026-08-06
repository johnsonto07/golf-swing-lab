"""Structured logging setup shared by the Streamlit pages and CLI tools.

Design notes:
- Logs go to both the console and a rotating file under data/logs/.
- Format includes timestamp, level, logger name, and message so issues in a
  specific pipeline stage (video, storage, pose, ...) are traceable.
- Never logs secret values (e.g. OPENAI_API_KEY). Callers must not pass
  secrets into log messages.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from golf_lab.config import LOGS_DIR, ensure_data_dirs

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotently configure root logging. Safe to call from every entry point."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    ensure_data_dirs()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOGS_DIR / "golf_swing_lab.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
