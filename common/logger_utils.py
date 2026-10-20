import logging
import os
from logging.handlers import RotatingFileHandler

from common.path_utils import get_file_path

LOG_DIR = get_file_path("logs")
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d | %(message)s"


def get_logger(name: str, log_file: str) -> logging.Logger:
    """Logger writing to the console and to logs/<log_file> (rotated at 10MB, 5 backups).

    Level comes from LOG_LEVEL (default INFO). Safe to call repeatedly, e.g. on every
    Streamlit rerun: handlers are only attached once per logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    if logger.handlers:
        return logger

    os.makedirs(LOG_DIR, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_file), maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Don't also bubble up to the root logger (uvicorn/streamlit configure their own)
    logger.propagate = False
    return logger


def truncate(value, max_len: int = 500) -> str:
    """Shorten long values (Cypher results, LLM answers) so a log line stays readable."""
    # Escape newlines so one value stays on one log line
    text = str(value).replace("\n", "\\n")
    return text if len(text) <= max_len else f"{text[:max_len]}... ({len(text)} chars)"
