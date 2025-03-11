import logging

LOG_LEVEL = logging.INFO

import logging

def getLogger(name: str) -> logging.Logger:
    """
    Returns a logger with a custom output format using our globally configured log level.
    Only our loggers (e.g. those with names starting with "manabot") will use this level.
    """
    logger = logging.getLogger(name)
    
    # Only configure once per logger.
    if getattr(logger, '_custom_configured', False):
        return logger

    # Define the custom format.
    custom_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(custom_format)
    
    # Set the logger's level from our global configuration.
    logger.setLevel(LOG_LEVEL)
    
    # If handlers exist, update them; otherwise, add a new StreamHandler.
    if logger.handlers:
        for handler in logger.handlers:
            handler.setFormatter(formatter)
            handler.setLevel(LOG_LEVEL)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(LOG_LEVEL)
        logger.addHandler(handler)
    
    # Mark the logger as custom configured.
    logger._custom_configured = True
    return logger
