import logging
import os
from typing import Optional


def setup_logging(
    log_file: str = "trialmatcher.log",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console_level: int = logging.WARNING,
):
    """
    Set up logging for the application.

    :param log_file: Path to the log file.
    :param log_dir: Directory to save the log file. Optional.
    :param level: Logging level (e.g., logging.INFO, logging.DEBUG).
    :param console_level: Logging level for the console output.
    """
    # If log_dir is provided, prepend it to the log_file, else use current working directory.
    if log_dir:
        log_file = os.path.join(log_dir, log_file)
    else:
        log_file = os.path.join(os.getcwd(), log_file)

    # Create the directory for the log file if it doesn't exist.
    log_dirname = os.path.dirname(log_file)
    if log_dirname:
        os.makedirs(log_dirname, exist_ok=True)

    # Get the root logger
    logger = logging.getLogger("trialmatcher")

    # Check if handlers already exist to prevent duplicate handlers
    if not logger.handlers:
        # Configure logging
        logger.setLevel(level)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s|%(levelname)s|%(filename)s->%(funcName)s:%(lineno)s|%(message)s"
            )
        )
        logger.addHandler(file_handler)

        # Create stream handler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(console_level)
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s|%(levelname)s|%(filename)s->%(funcName)s:%(lineno)s|%(message)s"
            )
        )
        logger.addHandler(stream_handler)

        # Disable propagation to prevent logging from other libraries
        logger.propagate = False

    # set logging level for other modules to error, but keep trialmatcher logging level at DEBUG
    # trialmatcher logs will be sent to the handlers which will log at the levels specified above
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("trialmatcher").setLevel(logging.DEBUG)

    logger.info("#" * 80)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger
