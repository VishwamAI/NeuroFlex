import logging


def setup_logging(log_level=logging.INFO):
    """
    Set up basic logging configuration for the NeuroFlex module.

    Args:
        log_level (int): The logging level to use. Defaults to logging.INFO.

    Returns:
        logging.Logger: A configured logger object.
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("NeuroFlex")
    return logger
