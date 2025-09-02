# src/crawler/coordinator.py
"""
爬虫协调器，负责初始化和管理Redis任务队列
"""
import redis
import json
from typing import List, Dict, Optional

from src.common import config
from src.common.logger import logger
from src.crawler.url_processor import URLProcessor

class Coordinator:
    """任务协调器，负责初始化和管理任务队列"""
    def __init__(self):
        try:
            self.redis = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=True # 自动解码，操作更方便
            )
            self.redis.ping()
            logger.info("Redis连接成功")
        except redis.ConnectionError as e:
            logger.error(f"Redis连接失败: {e}")
            raise

    def initialize_queue(self, seed_urls: Optional[List[str]] = None, clear_existing: bool = False):
        """初始化任务队列"""
        if seed_urls is None:
            seed_urls = config.SEED_URLS

        if clear_existing:
            self.redis.delete(config.TASK_QUEUE)
            self.redis.delete(config.VISITED_SET)
            logger.info("已清空现有任务队列和已访问集合")

        added_count = 0
        for url in seed_urls:
            canon_url = URLProcessor.canonicalize(url)
            if canon_url and not self.redis.sismember(config.VISITED_SET, canon_url):
                task = {"url": canon_url, "depth": 0}
                self.redis.rpush(config.TASK_QUEUE, json.dumps(task))
                # 种子URL也应被视为已访问，防止重复添加
                self.redis.sadd(config.VISITED_SET, canon_url)
                added_count += 1
        
        if added_count > 0:
            logger.info(f"成功添加 {added_count} 个新的种子URL到队列")
        else:
            logger.warning("没有新的种子URL被添加（可能已存在或URL无效）")

    def get_queue_status(self) -> Dict[str, int]:
        """获取队列状态"""
        queue_size = self.redis.llen(config.TASK_QUEUE)
        visited_size = self.redis.scard(config.VISITED_SET)
        return {"queue_size": queue_size, "visited_size": visited_size}
