# src/crawler/worker.py
"""
爬虫工作器，负责执行爬取任务的核心逻辑
"""
from bs4 import BeautifulSoup
import redis
import json
from src.common import config
from src.common.logger import logger
from src.crawler.client import HttpClient
from src.crawler.parser import Parser
from src.crawler.saver import ContentSaver
from src.crawler.screenshot import ScreenshotManager

class Worker:
    """爬虫工作器，负责执行爬取任务"""
    def __init__(self, use_screenshot=False):
        self.redis = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)
        self.http_client = HttpClient()
        self.use_screenshot = use_screenshot
        if self.use_screenshot:
            self.screenshot_manager = ScreenshotManager()

    async def _process_task(self, task: dict):
        """处理单个爬取任务"""
        url = task.get("url")
        depth = task.get("depth", 0)
        if not url: return

        logger.info(f"开始处理 [深度 {depth}]: {url}")

        # 1. 获取页面内容
        html_content = await self.http_client.fetch(url)

        # 2. 如果是HTML，保存文本并提取链接
        if html_content:
            # 保存纯文本
            soup = BeautifulSoup(html_content, "lxml")
            text = soup.get_text(separator='\n', strip=True)
            if text:
                ContentSaver.save(url, 'text', text.encode('utf-8'))
            
            # 提取新链接并加入队列
            if depth < config.MAX_DEPTH:
                new_links = Parser.extract_links(html_content, url)
                for link in new_links:
                    if not self.redis.sismember(config.VISITED_SET, link):
                        new_task = {"url": link, "depth": depth + 1}
                        self.redis.rpush(config.TASK_QUEUE, json.dumps(new_task))
                        self.redis.sadd(config.VISITED_SET, link)
                        logger.debug(f"发现新链接: {link}")

        # 3. 截图 (如果启用)
        if self.use_screenshot:
            self.screenshot_manager.take_screenshot(url)

    async def run(self):
        """运行爬虫工作器主循环"""
        await self.http_client.start()
        if self.use_screenshot:
            self.screenshot_manager.start()

        try:
            while True:
                # 从Redis队列中阻塞式获取任务
                task_data = self.redis.blpop(config.TASK_QUEUE, timeout=60)
                if not task_data:
                    logger.info("任务队列在60秒内无新任务，爬虫即将退出...")
                    break
                
                _, task_json_str = task_data
                try:
                    task = json.loads(task_json_str)
                    await self._process_task(task)
                except json.JSONDecodeError:
                    logger.error(f"无法解析任务数据: {task_json_str}")
                except Exception as e:
                    logger.error(f"处理任务时发生未知错误: {e}")

        finally:
            await self.http_client.stop()
            if self.use_screenshot:
                self.screenshot_manager.stop()
            logger.info("爬虫工作器已停止。")

