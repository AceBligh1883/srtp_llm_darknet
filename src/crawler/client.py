# src/crawler/client.py
"""
HTTP客户端模块，负责通过Tor代理进行网络请求
"""
import asyncio
import aiohttp
import aiohttp_socks
from typing import Optional

from src.common import config
from src.common.logger import logger
from src.crawler.saver import ContentSaver

class HttpClient:
    """HTTP客户端，处理通过Tor的网络请求"""
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)

    async def start(self):
        """初始化AIOHTTP会话"""
        try:
            connector = aiohttp_socks.ProxyConnector.from_url(config.TOR_PROXY)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT),
                headers={'User-Agent': config.USER_AGENT}
            )
            logger.info("HTTP客户端会话已初始化 (Tor Proxy)")
        except Exception as e:
            logger.error(f"初始化HTTP客户端失败: {e}")
            raise

    async def stop(self):
        """关闭AIOHTTP会话"""
        if self.session:
            await self.session.close()
            logger.info("HTTP客户端会话已关闭")

    async def fetch(self, url: str) -> Optional[str]:
        """
        获取URL内容。
        如果是HTML，返回文本内容。
        如果是其他二进制文件（图片、视频等），直接保存并返回None。
        """
        async with self.semaphore:
            await asyncio.sleep(1.0 / config.RATE_LIMIT)
            try:
                async with self.session.get(url, allow_redirects=True) as response:
                    response.raise_for_status() # 如果状态码不是2xx，则抛出异常
                    
                    content_type = response.headers.get("Content-Type", "").lower()
                    content = await response.read()

                    if "text/html" in content_type:
                        # 尝试多种编码解码
                        try:
                            return content.decode('utf-8')
                        except UnicodeDecodeError:
                            return content.decode('latin-1')
                    else:
                        # 对于非HTML内容，直接调用saver保存
                        if "image" in content_type:
                            ContentSaver.save(url, 'image', content)
                        elif "video" in content_type:
                            ContentSaver.save(url, 'video', content)
                        else:
                            ContentSaver.save(url, 'file', content)
                        return None # 表示已处理，无需进一步解析
            except Exception as e:
                logger.warning(f"请求失败: {url} - {type(e).__name__}: {e}")
                return None
