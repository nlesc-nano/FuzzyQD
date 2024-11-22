import logging
import datetime

def setup_logger(level_name="INFO"):
    """
    Sets up the logger with the specified level.
    :param level_name: Logging level as a string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = getattr(logging, level_name.upper(), logging.INFO)  # Default to INFO if invalid
    log_filename = f"fuzzyqd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create or retrieve the logger
    logger = logging.getLogger("fuzzyqd_logger")
    logger.setLevel(level)  # Explicitly set the logging level

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add handlers
    file_handler = logging.FileHandler(log_filename)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

# Default logger instance (can be configured dynamically later)
logger = setup_logger()

