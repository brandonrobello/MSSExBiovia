University of California Berkeley x BIOVIA  
MSSE Capstone Project – Spring 2025  
Authors: Yara Khoury, Brandon Robello, Jeremy Millford  
Created: May 2025  
Last Updated: May 2025 

# Utilities Module

This module provides shared logging utilities used throughout the atom typing project. Its purpose is to standardize how logs are generated and recorded across all modeling components and scripts.

---

## `logging_utils.py`

This script defines a centralized logging system for the entire repository using Python’s built-in `logging` module.

### Key Functions

| Function | Description |
|----------|-------------|
| `setup_logging(log_file=None, level=logging.INFO)` | Sets up logging to console and optionally to a file. Ensures configuration is only applied once. |
| `get_logger(name)` | Returns a configured logger instance for the given module or script name. |

### Features

- Timestamped and formatted logging across all modules
- Supports both console output and persistent file logging
- Safe to import in multiple scripts without reinitializing logging

---

## Usage Example

from atoMLtype.utils.logging_utils import get_logger

logger = get_logger(__name__)
logger.info("Pipeline started successfully.")

## Dependencies
Python standard library only:
- logging
- os
- typing.Optional