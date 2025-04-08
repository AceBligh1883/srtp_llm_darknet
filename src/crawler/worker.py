# -*- coding: utf-8 -*-
"""
爬虫工作器模块
"""

import os
import redis
import json
from bs4 import BeautifulSoup

from src.logger import logger
from config import config
from src.utils.url_processor import URLProcessor
from src.utils.io_utils import save_content 
from src.crawler.screenshot_manager import ScreenshotManager
from src.crawler.http_client import HttpClient
from src.crawler.link_extractor import LinkExtractor

class Worker:
    """爬虫工作器，负责执行爬取任务"""
    def __init__(self):
        # 初始化Redis客户端
        self.redis = redis.Redis(
            host=config.REDIS_HOST, 
            port=config.REDIS_PORT, 
            db=config.REDIS_DB
        )
        
        # 初始化组件
        self.http_client = HttpClient()
        self.link_extractor = LinkExtractor(self.redis)
        self.screenshot_manager = ScreenshotManager()
        self._setup_directories()
    
    def _setup_directories(self):
        """确保所有必要的目录存在"""
        directories = [
            config.TEXT_DIR, 
            config.IMAGE_DIR,
            config.OUTPUT_DIR,
            config.SCREENSHOTS_DIR,
            config.VIDEOS_DIR,
            config.FILE_DIR
        ]
        
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
    
    async def process_page(self, url, depth):
        """处理单个页面"""
        canon_url = URLProcessor.canonicalize(url)
        # 标记为已访问
        self.redis.sadd(config.VISITED_SET, canon_url)
        logger.info(f"处理页面 [深度 {depth}]: {url}")
        
        # 获取页面内容
        html, success = await self.http_client.fetch(url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if text:
                await save_content(text, url, "text")
            self.link_extractor.extract_links(soup, url, depth)
        
        # 将截图任务添加到独立进程的队列
        #if success:
        #    self.screenshot_manager.add_task(url)
    
    async def run(self):
        """运行爬虫工作器"""
        #self.screenshot_manager.start()
        await self.http_client.init_session()
        
        try:
            while True:
                # 从队列获取任务
                task_data = self.redis.blpop(config.TASK_QUEUE, timeout=30)
                if not task_data:
                    logger.info("任务队列为空，爬虫退出...")
                    break
                
                _, task_json = task_data
                try:
                    task = json.loads(task_json)
                except Exception as e:
                    logger.error(f"任务数据解析错误: {e}")
                    continue
                
                url = task.get("url")
                depth = task.get("depth", 0)
                
                if url and URLProcessor.is_onion(url):
                    await self.process_page(url, depth)
        
        except Exception as e:
            logger.error(f"爬虫运行异常: {e}")
        
        finally:
            await self.http_client.close()
            self.screenshot_manager.stop()
            logger.info("爬虫已关闭")


