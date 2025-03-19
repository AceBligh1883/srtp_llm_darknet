# -*- coding: utf-8 -*-
"""
HTTP客户端模块
"""

import asyncio
import aiohttp
import aiohttp_socks
from typing import Optional, Tuple

from src.logger import logger
from config import config
from src.utils.io_utils import save_content 

class HttpClient:
    """HTTP客户端，处理网络请求"""
    def __init__(self):
        self.session = None
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)
    
    async def init_session(self):
        """初始化HTTP会话"""
        connector = aiohttp_socks.ProxyConnector.from_url(config.TOR_PROXY)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)
        )
        logger.info("HTTP会话已初始化")
    
    async def close(self):
        """关闭HTTP会话"""
        if self.session:
            await self.session.close()
            logger.info("HTTP会话已关闭")
    
    async def fetch(self, url: str) -> Tuple[Optional[str], bool]:
        """获取URL内容"""
        async with self.semaphore:
            # 速率限制
            await asyncio.sleep(1 / config.RATE_LIMIT)
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content_type = response.headers.get("Content-Type", "").lower()
                        if "text/html" in content_type:
                            return await response.text(), True
                        else:
                            data = await response.read()
                            if "image" in content_type:
                                await save_content(data, url, "image")
                            elif "video" in content_type:
                                await save_content(data, url, "video")
                            elif any(x in content_type for x in ["application/pdf", "application/msword", "audio/", "application/octet-stream"]):
                                await save_content(data, url, "file")
                            return None, True
                    else:
                        logger.warning(f"请求失败: {url} 状态码: {response.status}")
                        return None, False
            except Exception as e:
                logger.warning(f"请求异常: {url} - {e}")
                return None, False
