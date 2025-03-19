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
    """独立的截图工作进程管理器，负责管理浏览器实例和截图任务"""
    
    def __init__(self, task_queue_size=100):
        """
        初始化截图管理器
        """
        self.task_queue = multiprocessing.Queue(maxsize=task_queue_size)
        self.stop_event = multiprocessing.Event()
        self.process = None
    
    def start(self):
        """
        启动截图工作进程
        """
        # 确保截图目录存在
        os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)
            
        self.process = multiprocessing.Process(
            target=self._worker_process,
            args=(self.task_queue, self.stop_event)
        )
        self.process.daemon = True
        self.process.start()
        logger.info("截图工作进程已启动，PID: %d", self.process.pid)
        return self.process.pid
    
    def stop(self):
        """停止截图工作进程"""
        if not self.process or not self.process.is_alive():
            return
            
        logger.debug("正在停止截图工作进程...")
        self.stop_event.set()
        self.process.join(timeout=5)
        
        if self.process.is_alive():
            logger.warning("截图工作进程未能正常退出，强制终止")
            self.process.terminate()
        
        logger.info("截图工作进程已停止")
    
    def add_task(self, url):
        """
        添加截图任务到队列
        """
        if not url.startswith("http"):
            logger.debug("无效的 URL: %s", url)
            return False
        
        try:
            self.task_queue.put(url, block=False)
            return True
        except queue.Full:
            logger.debug("截图任务队列已满，跳过: %s", url)
            return False
    
    @staticmethod
    def _worker_process(task_queue, stop_event):
        """
        截图工作进程的主函数
        """
        browsers = []
        browser_creation_errors = 0
        max_browser_errors = 3
        
        logger.debug("开始初始化浏览器实例")
        
        # 初始化浏览器池
        for i in range(config.SCREENSHOT_WORKERS):
            try:
                logger.debug("初始化浏览器实例 %d/%d", i+1, config.SCREENSHOT_WORKERS)
                browser = ScreenshotManager._init_browser(config)
                browsers.append(browser)
                logger.debug("浏览器实例 %d 初始化成功", i+1)
            except Exception as e:
                logger.error("初始化浏览器实例 %d 失败: %s", i+1, str(e))
                browser_creation_errors += 1
                if browser_creation_errors >= max_browser_errors:
                    logger.error("浏览器创建失败次数过多，停止创建更多实例")
                    break
        
        # 检查是否有可用的浏览器实例
        if not browsers:
            logger.error("没有成功初始化任何浏览器实例，截图功能将不可用")
        else:
            logger.info("已初始化 %d 个浏览器实例", len(browsers))
        
        # 使用线程池管理截图任务
        try:
            with ThreadPoolExecutor(max_workers=max(1, len(browsers))) as executor:
                while not stop_event.is_set():
                    # 获取任务
                    try:
                        url = task_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    
                    # 处理任务
                    if browsers:
                        browser = browsers.pop(0)
                        future = executor.submit(
                            ScreenshotManager._take_screenshot,
                            browser, url, config, logger
                        )
                        # 任务完成后将浏览器放回池中(如果没有发生异常)
                        future.add_done_callback(
                            lambda f, b=browser: browsers.append(b) if not f.exception() else None
                        )
                    else:
                        # 无可用浏览器时重新加入队列并尝试创建新浏览器
                        logger.warning("没有可用的浏览器实例，任务重新入队")
                        task_queue.put(url)
                        time.sleep(1)
                        
                        # 尝试重新创建浏览器实例
                        if browser_creation_errors < max_browser_errors:
                            try:
                                logger.debug("尝试重新初始化一个浏览器实例")
                                browser = ScreenshotManager._init_browser(config)
                                browsers.append(browser)
                                logger.debug("浏览器实例重新初始化成功")
                            except Exception as e:
                                logger.error("重新初始化浏览器实例失败: %s", str(e))
                                browser_creation_errors += 1
        except Exception as e:
            logger.error("截图工作进程异常: %s", str(e))
        finally:
            # 关闭所有浏览器实例
            for browser in browsers:
                try:
                    browser.quit()
                except Exception as e:
                    logger.error("关闭浏览器异常: %s", str(e))
            logger.info("截图工作进程已退出")
    
    @staticmethod
    def _init_browser(config):
        """
        初始化Firefox浏览器实例
        """
        firefox_options = Options()
        
        # 基本配置
        firefox_options.add_argument("--headless")
        firefox_options.add_argument("--disable-gpu") 
        firefox_options.add_argument("--no-sandbox")
        firefox_options.add_argument("--disable-dev-shm-usage")
        
        # Tor代理配置
        firefox_options.set_preference("network.proxy.type", 1)
        firefox_options.set_preference("network.proxy.socks", "127.0.0.1")
        firefox_options.set_preference("network.proxy.socks_port", 9150)
        firefox_options.set_preference("network.proxy.socks_remote_dns", True)
        firefox_options.set_preference("general.useragent.override", config.USER_AGENT)
        
        # 隐藏自动化特征
        firefox_options.set_preference("dom.webdriver.enabled", False)
        firefox_options.set_preference("useAutomationExtension", False)

        # 禁用提示和弹窗
        firefox_options.set_preference("dom.disable_beforeunload", True)
        firefox_options.set_preference("dom.disable_open_during_load", True)
        firefox_options.set_preference("dom.popup_maximum", 0)
        firefox_options.set_preference("prompts.contentPromptSubDialog", False)
        firefox_options.set_preference("prompts.tab_modal.enabled", False)
        
        # 语言设置
        firefox_options.set_preference("intl.multilingual.enabled", False)
        firefox_options.set_preference("intl.accept_languages", "en-US, en")
        firefox_options.set_preference("privacy.spoof_english", 2)
        firefox_options.set_preference("browser.translation.ui.show", False)
        firefox_options.set_preference("browser.translation.detectLanguage", False)
        
        # 创建浏览器实例
        firefox_options.binary_location = config.TOR_BROWSER_BINARY
        service = Service(config.GECKODRIVER_PATH)
        browser = webdriver.Firefox(service=service, options=firefox_options)
        browser.set_page_load_timeout(30)
        
        return browser
    
    @staticmethod
    def _take_screenshot(browser, url, config, logger):
        """
        执行截图任务
        """
        target_url = url.replace("https://", "http://")
        logger.info("开始截图: %s", target_url)
        
        try:
            # 加载页面并处理可能的弹窗
            try:
                browser.get(target_url)
            except Exception as e:
                logger.warning("页面加载异常: %s - %s", target_url, str(e))
                # 尝试处理弹窗
                ScreenshotManager._handle_alerts(browser, logger)
            
            # 等待页面加载完成
            try:
                WebDriverWait(browser, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                logger.warning("等待页面元素超时: %s", target_url)
            
            # 再次检查弹窗
            ScreenshotManager._handle_alerts(browser, logger)
            
            # 截图并保存
            screenshot = browser.get_screenshot_as_png()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parsed = urlparse(url)
            base = f"{timestamp}_{parsed.netloc}_{hash(url) % 10000}"
            filename = f"{base}.png"
            path = os.path.join(config.SCREENSHOTS_DIR, filename)
            
            with open(path, "wb") as f:
                f.write(screenshot)
            
            logger.info("截图已保存: %s", path)
            return True
            
        except WebDriverException as e:
            logger.error("浏览器异常: %s - %s", url, str(e))
            return False
        except Exception as e:
            logger.error("截图过程异常: %s - %s", url, str(e))
            return False
    
    @staticmethod
    def _handle_alerts(browser, logger):
        """
        处理页面中可能出现的警告框
        """
        try:
            alert = browser.switch_to.alert
            alert_text = alert.text
            logger.debug("检测到弹窗: %s，自动关闭", alert_text)
            alert.dismiss()
        except Exception:
            # 没有弹窗或处理失败，忽略
            pass
