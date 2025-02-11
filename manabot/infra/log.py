import logging


def getLogger(name: str) -> logging.Logger:
    """
    Returns a logger where the format has been configured to omit the filename,
    but we retain the log level.
    """
    logger = logging.getLogger(name)
    
    # Only configure once per logger.
    if getattr(logger, '_custom_configured', False):
        return logger

    # Define the custom format (notice no file name here)
    custom_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(custom_format)
    
    # Get the effective level (could be inherited) and set it explicitly.
    current_level = logger.getEffectiveLevel()
    logger.setLevel(current_level)
    
    # If handlers exist, update them; otherwise, add a new StreamHandler.
    if logger.handlers:
        for handler in logger.handlers:
            handler.setFormatter(formatter)
            handler.setLevel(current_level)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(current_level)
        logger.addHandler(handler)
    
    # Mark the logger as custom configured so future calls skip reconfiguration.
    logger._custom_configured = True
    return logger

