# -*- coding: utf-8 -*-
"""
爬虫协调器模块
"""

import os
import redis
import json

from src.logger import logger
from config import config
from urllib.parse import urlparse, urlunparse

class URLProcessor:
    """URL处理工具类"""
    @staticmethod
    def normalize(url: str) -> str:
        """标准化 URL，自动添加协议并转换为小写"""
        url = url.strip()
        if not url.lower().startswith("http"):
            url = "http://" + url
        return url.lower()
    
    @staticmethod
    def is_onion(url: str) -> bool:
        """判断是否为 .onion 域名"""
        return urlparse(url).netloc.endswith(".onion")
    
    @staticmethod
    def canonicalize(url: str) -> str:
        """规范化 URL，用于去重（忽略末尾斜杠、统一大小写）"""
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip('/')
        return urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))

class Coordinator:
    """任务协调器，负责初始化任务队列"""
    def __init__(self):
        self.redis = redis.Redis(
            host=config.REDIS_HOST, 
            port=config.REDIS_PORT, 
            db=config.REDIS_DB
        )
        
        # 确保目录存在
        self._setup_directories()
    
    def _setup_directories(self):
        """确保所有必要的目录存在"""
        directories = [
            config.TEXT_DIR, 
            config.IMAGE_DIR,
            config.OUTPUT_DIR,
            config.SCREENSHOTS_DIR,
            config.VIDEOS_DIR,
            config.FILES_DIR
        ]
        
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
    
    def initialize_queue(self, seed_urls=None, clear_existing=False):
        """初始化任务队列"""
        if seed_urls is None:
            seed_urls = config.SEED_URLS
        
        try:
            self.redis.ping()
            logger.info("Redis连接成功")
        except redis.ConnectionError as e:
            logger.error(f"Redis连接失败: {e}")
            return False
        
        # 清空现有队列（可选）
        if clear_existing:
            self.redis.delete(config.TASK_QUEUE)
            self.redis.delete(config.VISITED_SET)
            logger.info("已清空现有队列")
        
        # 添加种子URL
        for url in seed_urls:
            canon = URLProcessor.canonicalize(url)
            if canon:
                task = {"url": canon, "depth": 0}
                self.redis.rpush(config.TASK_QUEUE, json.dumps(task))
                logger.info(f"已添加种子URL到队列: {canon}")
            else:
                logger.warning(f"URL规范化失败: {url}")
        
        return True
    
    def get_queue_status(self):
        """获取队列状态"""
        queue_size = self.redis.llen(config.TASK_QUEUE)
        visited_size = self.redis.scard(config.VISITED_SET)
        return {
            "queue_size": queue_size,
            "visited_size": visited_size
        }
