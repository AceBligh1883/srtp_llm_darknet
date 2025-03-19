# -*- coding: utf-8 -*-
"""
截图管理模块
"""

import os
import time
import multiprocessing
import queue
from datetime import datetime
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
from concurrent.futures import ThreadPoolExecutor

from src.logger import logger
from config import config

class ScreenshotManager:
    """独立的截图工作进程管理器"""
    def __init__(self, task_queue_size=100):
        # 使用进程间队列
        self.task_queue = multiprocessing.Queue(maxsize=task_queue_size)
        self.stop_event = multiprocessing.Event()
        self.process = None
    
    def start(self):
        """启动截图工作进程"""
        # 确保目录存在
        os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)
            
        self.process = multiprocessing.Process(
            target=self._worker_process,
            args=(self.task_queue, self.stop_event)
        )
        self.process.daemon = True
        self.process.start()
        logger.info(f"截图工作进程已启动 (PID: {self.process.pid})")
        return self.process.pid
    
    def stop(self):
        """停止截图工作进程"""
        if self.process and self.process.is_alive():
            logger.info("正在停止截图工作进程...")
            self.stop_event.set()
            self.process.join(timeout=5)
            if self.process.is_alive():
                logger.warning("截图工作进程未能正常退出，强制终止")
                self.process.terminate()
            logger.info("截图工作进程已停止")
    
    def add_task(self, url):
        """添加截图任务到队列"""
        try:
            self.task_queue.put(url, block=False)
            return True
        except queue.Full:
            logger.warning(f"截图任务队列已满，跳过: {url}")
            return False
    
    @staticmethod
    def _worker_process(task_queue, stop_event):
        """截图工作进程的主函数"""
        # 初始化浏览器池
        browsers = []
        browser_creation_errors = 0
        max_browser_errors = 3
        
        logger.info("截图工作进程启动，开始初始化浏览器实例")
        
        try:
            # 创建浏览器实例
            for i in range(config.SCREENSHOT_WORKERS):
                try:
                    logger.info(f"初始化浏览器实例 {i+1}/{config.SCREENSHOT_WORKERS}")
                    browser = ScreenshotManager._init_browser(config)
                    browsers.append(browser)
                    logger.info(f"浏览器实例 {i+1} 初始化成功")
                except Exception as e:
                    logger.error(f"初始化浏览器实例 {i+1} 失败: {e}")
                    browser_creation_errors += 1
                    if browser_creation_errors >= max_browser_errors:
                        logger.error("浏览器创建失败次数过多，放弃初始化")
                        break
            
            if browsers:
                logger.info(f"截图工作进程已初始化 {len(browsers)} 个浏览器实例")
            else:
                logger.error("没有成功初始化任何浏览器实例，截图功能将不可用")
            
            # 创建线程池
            with ThreadPoolExecutor(max_workers=max(1, len(browsers))) as executor:
                # 主循环
                while not stop_event.is_set():
                    try:
                        # 非阻塞方式获取任务，超时后检查停止事件
                        try:
                            url = task_queue.get(timeout=1)
                        except queue.Empty:
                            continue
                        
                        # 从浏览器池中获取一个浏览器
                        if browsers:
                            browser = browsers.pop(0)
                            # 提交截图任务到线程池
                            future = executor.submit(
                                ScreenshotManager._take_screenshot,
                                browser, url, config, logger
                            )
                            # 设置回调，将浏览器放回池中
                            future.add_done_callback(
                                lambda f, b=browser: browsers.append(b) if not f.exception() else None
                            )
                        else:
                            # 如果没有可用的浏览器，将任务放回队列
                            logger.warning("没有可用的浏览器实例，稍后重试")
                            task_queue.put(url)
                            # 短暂等待
                            time.sleep(1)
                            
                            # 尝试重新初始化一个浏览器
                            if browser_creation_errors < max_browser_errors:
                                try:
                                    logger.info("尝试重新初始化一个浏览器实例")
                                    browser = ScreenshotManager._init_browser(config)
                                    browsers.append(browser)
                                    logger.info("浏览器实例重新初始化成功")
                                except Exception as e:
                                    logger.error(f"重新初始化浏览器实例失败: {e}")
                                    browser_creation_errors += 1
                    
                    except Exception as e:
                        logger.error(f"截图工作进程异常: {e}")
                        # 防止过快循环消耗CPU
                        time.sleep(1)
        
        finally:
            # 关闭所有浏览器
            for browser in browsers:
                try:
                    browser.quit()
                except Exception as e:
                    logger.error(f"关闭浏览器异常: {e}")
            
            logger.info("截图工作进程已退出")
    
    @staticmethod
    def _init_browser(config):
        """初始化浏览器实例"""
        firefox_options = Options()
        firefox_options.headless = True
        # 设置 Tor 代理
        firefox_options.set_preference("network.proxy.type", 1)
        firefox_options.set_preference("network.proxy.socks", "127.0.0.1")
        firefox_options.set_preference("network.proxy.socks_port", 9150)
        firefox_options.set_preference("network.proxy.socks_remote_dns", True)
        firefox_options.set_preference("general.useragent.override", config.USER_AGENT)
        
        # 添加更多稳定性设置
        firefox_options.set_preference("browser.tabs.remote.autostart", False)
        firefox_options.set_preference("browser.tabs.remote.autostart.2", False)
        firefox_options.set_preference("dom.ipc.processCount", 1)
        firefox_options.set_preference("browser.sessionstore.resume_from_crash", False)
        firefox_options.set_preference("toolkit.startup.max_resumed_crashes", -1)
        
        # 指定 Tor 浏览器的二进制路径
        firefox_options.binary_location = config.TOR_BROWSER_BINARY
        
        service = Service(config.GECKODRIVER_PATH)
        browser = webdriver.Firefox(service=service, options=firefox_options)
        
        # 设置页面加载超时
        browser.set_page_load_timeout(30)
        
        return browser
    
    @staticmethod
    def _take_screenshot(browser, url, config, logger):
        """执行截图任务"""
        try:
            # 将 https 替换为 http（根据需要调整）
            target_url = url.replace("https://", "http://")
            logger.info(f"开始截图: {target_url}")
            
            try:
                browser.get(target_url)
            except TimeoutException:
                logger.warning(f"页面加载超时: {target_url}")
                # 即使超时也尝试截图
            
            # 等待页面加载
            try:
                WebDriverWait(browser, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                logger.warning(f"等待页面元素超时: {target_url}")
            
            # 检查并关闭弹出提示
            try:
                alert = browser.switch_to.alert
                logger.info(f"检测到提示信息: {alert.text}，自动关闭。")
                alert.dismiss()
            except Exception:
                pass
            
            # 截图
            screenshot = browser.get_screenshot_as_png()
            
            # 保存截图
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parsed = urlparse(url)
            base = f"{timestamp}_{parsed.netloc}_{hash(url) % 10000}"
            filename = f"{base}.png"
            path = os.path.join(config.SCREENSHOTS_DIR, filename)
            
            with open(path, "wb") as f:
                f.write(screenshot)
            
            logger.info(f"[截图] 已保存至: {path}")
            return True
            
        except WebDriverException as e:
            logger.error(f"截图异常: {url} - {e}")
            return False
        except Exception as e:
            logger.error(f"截图过程中发生错误: {url} - {e}")
            return False
