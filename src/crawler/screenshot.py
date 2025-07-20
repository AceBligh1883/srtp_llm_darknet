# src/crawler/screenshot.py
"""
截图管理模块，使用Selenium通过Tor浏览器进行截图
"""
import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

from src.common import config
from src.common.logger import logger
from src.crawler.saver import ContentSaver

class ScreenshotManager:
    """管理单个浏览器实例并执行截图任务"""
    def __init__(self):
        self.browser: webdriver.Firefox | None = None

    def start(self):
        """初始化并启动浏览器实例"""
        try:
            logger.info("正在初始化Tor浏览器用于截图...")
            options = Options()
            options.add_argument("--headless")
            # Tor代理配置
            options.set_preference("network.proxy.type", 1)
            options.set_preference("network.proxy.socks", "127.0.0.1")
            options.set_preference("network.proxy.socks_port", 9150)
            options.set_preference("network.proxy.socks_remote_dns", True)
            options.set_preference("general.useragent.override", config.USER_AGENT)
            
            options.binary_location = config.TOR_BROWSER_BINARY
            service = Service(config.GECKODRIVER_PATH)
            
            self.browser = webdriver.Firefox(service=service, options=options)
            self.browser.set_page_load_timeout(config.REQUEST_TIMEOUT)
            logger.info("Tor浏览器截图实例初始化成功。")
        except Exception as e:
            logger.error(f"初始化Tor浏览器失败: {e}")
            self.browser = None

    def stop(self):
        """关闭浏览器实例"""
        if self.browser:
            try:
                self.browser.quit()
                logger.info("Tor浏览器截图实例已关闭。")
            except Exception as e:
                logger.error(f"关闭Tor浏览器时出错: {e}")
            self.browser = None

    def take_screenshot(self, url: str):
        """对给定的URL进行截图"""
        if not self.browser:
            logger.warning("截图功能不可用，浏览器未初始化。")
            return

        try:
            logger.info(f"正在截图: {url}")
            self.browser.get(url)
            # 等待几秒让页面动态内容加载
            time.sleep(5) 
            png_data = self.browser.get_screenshot_as_png()
            ContentSaver.save(url, 'screenshot', png_data)
        except Exception as e:
            logger.error(f"截图失败: {url} - {type(e).__name__}: {e}")
