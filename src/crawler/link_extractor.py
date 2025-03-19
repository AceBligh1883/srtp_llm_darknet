# -*- coding: utf-8 -*-
"""
链接提取器模块
"""

import json
from bs4 import BeautifulSoup
from typing import Set, List

from src.logger import logger
from config import config
from src.utils.url_processor import URLProcessor

class LinkExtractor:
    """从HTML中提取链接"""
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def extract_links(self, soup: BeautifulSoup, base_url: str, depth: int) -> None:
        """从页面中提取链接并添加到队列"""
        if depth >= config.MAX_DEPTH:
            return
        
        links = self._find_links(soup, base_url)
        new_links = self._filter_new_links(links)
        self._add_to_queue(new_links, depth + 1)
    
    def _find_links(self, soup: BeautifulSoup, base_url: str) -> Set[str]:
        """查找页面中的所有链接"""
        links = set()
        for tag in soup.find_all(["a", "img", "video", "audio", "source", "iframe", "embed", "object"]):
            href = tag.get("href") or tag.get("src")
            if href:
                full_url = URLProcessor.join(base_url, href)
                norm_url = URLProcessor.normalize(full_url)
                if URLProcessor.is_onion(norm_url):
                    canon = URLProcessor.canonicalize(norm_url)
                    links.add(canon)
        return links
    
    def _filter_new_links(self, links: Set[str]) -> List[str]:
        """过滤出未访问过的链接"""
        return [link for link in links if not self.redis.sismember(config.VISITED_SET, link)]
    
    def _add_to_queue(self, links: List[str], depth: int) -> None:
        """将链接添加到任务队列"""
        for link in links:
            task = {"url": link, "depth": depth}
            self.redis.rpush(config.TASK_QUEUE, json.dumps(task))
            self.redis.sadd(config.VISITED_SET, link)
            logger.debug(f"添加新链接到队列: {link} [深度 {depth}]")
