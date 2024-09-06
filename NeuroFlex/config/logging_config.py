# logging_config.py

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir='logs', log_level=logging.INFO):
    """
    Set up logging configuration for the NeuroFlex project.
    
    Args:
    log_dir (str): Directory to store log files
    log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
    
    Returns:
    logger: Configured logger object
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create logger
    logger = logging.getLogger('NeuroFlex')
    logger.setLevel(log_level)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'neuroflex.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )

    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

    # Set formatters for handlers
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Example usage
if __name__ == "__main__":
    logger = setup_logging(log_level=logging.DEBUG)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

# TODO: Add support for different log formats based on the environment (development, production)
# TODO: Implement log rotation based on time (e.g., daily logs) in addition to size
# TODO: Add support for remote logging (e.g., sending logs to a centralized server)
# TODO: Implement log filtering for sensitive information
