import logging
import sys
from typing import Optional

def setup_logger(verbose: bool = False) -> logging.Logger:
    """
    Configure and return the application logger.
    
    Args:
        verbose: If True, set log level to DEBUG. Otherwise, set to INFO.
    """
    logger = logging.getLogger("llm_benchmark")
    
    # Prevent double logging if setup is called multiple times
    if logger.hasHandlers():
        return logger

    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Format: [TIMESTAMP] [LEVEL] [MODULE] Message
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger

# Global logger instance to be imported by other modules
logger = setup_logger()