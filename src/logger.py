# -*- coding: utf-8 -*-
"""
日志模块
"""

import logging
from config import config

# 日志配置字典
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

def setup_logger() -> logging.Logger:
    """
    配置并返回日志记录器
    
    :return: 配置好的Logger对象
    """
    # 获取根日志记录器
    logger = logging.getLogger()
    
    # 设置日志级别
    logger.setLevel(LOG_LEVELS.get(config.LOG_LEVEL, logging.INFO))
    
    # 清除现有的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT, config.DATE_FORMAT))
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志记录器
logger = setup_logger()
