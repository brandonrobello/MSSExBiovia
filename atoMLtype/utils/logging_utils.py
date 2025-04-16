import logging
import os
from typing import Optional

_LOGGING_CONFIGURED = False

def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    """
    Set up global logging configuration. Safe to call multiple times.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        handlers=handlers
    )

    _LOGGING_CONFIGURED = True

def get_logger(name: str) -> logging.Logger:
    setup_logging()  # This ensures logging is configured once across the repo
    return logging.getLogger(name)
