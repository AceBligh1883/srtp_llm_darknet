# src/common/logger.py
"""
日志模块
"""
import logging
from src.common import config

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

def setup_logger() -> logging.Logger:
    """配置并返回日志记录器"""
    logger = logging.getLogger("DarknetSystem")
    logger.propagate = False 

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(LOG_LEVELS.get(config.LOG_LEVEL, logging.INFO))

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(config.LOG_FORMAT, config.DATE_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()
