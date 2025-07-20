# src/crawler/parser.py
"""
HTML解析与链接提取模块
"""
import json
from bs4 import BeautifulSoup
from typing import Set

from src.common import config
from src.common.logger import logger
from src.common.url_processor import URLProcessor

class Parser:
    """从HTML中提取链接"""

    @staticmethod
    def extract_links(html_content: str, base_url: str) -> Set[str]:
        """从页面中提取所有有效的.onion链接"""
        links = set()
        if not html_content:
            return links
            
        soup = BeautifulSoup(html_content, "lxml") # 使用lxml解析器，更快
        for tag in soup.find_all('a', href=True):
            href = tag['href']
            if href and not href.startswith(('mailto:', 'tel:')):
                full_url = URLProcessor.join(base_url, href)
                if URLProcessor.is_onion(full_url):
                    canon_url = URLProcessor.canonicalize(full_url)
                    if canon_url:
                        links.add(canon_url)
        return links
