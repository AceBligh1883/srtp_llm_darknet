# src/crawler/saver.py
"""
内容保存器，负责将抓取到的内容保存到文件系统
"""
import os
import hashlib
from datetime import datetime
from urllib.parse import urlparse

from src.common import config
from src.common.logger import logger
from src.common.url_processor import URLProcessor

class ContentSaver:
    """将抓取到的内容保存到统一的输入目录"""

    @staticmethod
    def get_save_path(url: str, content_type: str, content: bytes) -> str:
        """根据内容类型和URL生成确定性的文件保存路径"""
        # 1. 确定基础目录
        if content_type == 'text':
            base_dir = config.TEXT_DIR
            ext = '.txt'
        elif content_type == 'image':
            base_dir = config.IMAGE_DIR
            # 尝试从URL获取扩展名，否则默认为.jpg
            path = urlparse(url).path
            _, url_ext = os.path.splitext(path)
            ext = url_ext if url_ext.lower() in config.SUPPORTED_IMAGE_FORMATS else '.jpg'
        elif content_type == 'screenshot':
            base_dir = config.SCREENSHOTS_DIR
            ext = '.png'
        else: # video, file etc.
            base_dir = config.FILE_DIR
            path = urlparse(url).path
            _, ext = os.path.splitext(path)
            if not ext: ext = '.bin' # 默认二进制扩展名
        
        # 2. 生成基于哈希的文件名，避免重复和文件名过长问题
        content_hash = hashlib.sha256(content).hexdigest()[:16]
        domain = URLProcessor.get_domain(url)
        # 使用域名和哈希创建子目录，分散文件
        sub_dir = os.path.join(base_dir, domain, content_hash[:2])
        os.makedirs(sub_dir, exist_ok=True)

        # 文件名格式: {时间戳}_{内容哈希}.{扩展名}
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{content_hash}{ext}"
        
        return os.path.join(sub_dir, filename)

    @staticmethod
    def save(url: str, content_type: str, content: bytes):
        """保存内容到文件"""
        if not content:
            return
        
        try:
            save_path = ContentSaver.get_save_path(url, content_type, content)
            if os.path.exists(save_path):
                logger.debug(f"文件已存在，跳过保存: {save_path}")
                return

            with open(save_path, 'wb') as f:
                f.write(content)
            logger.info(f"[{content_type.upper()}] 已保存: {save_path}")
        except Exception as e:
            logger.error(f"保存内容失败 (URL: {url}): {e}")
