# src/etl/loader.py
import os
from typing import List
from src.common import config
from src.common.logger import logger
from src.storage.database import DatabaseManager

class DataLoader:
    """负责从文件系统发现和加载原始数据。"""
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def discover_new_files(self, directory: str, content_type: str) -> List[str]:
        """发现目录中尚未被处理的新文件。"""
        if not os.path.exists(directory):
            logger.error(f"目录不存在: {directory}")
            return []
        all_paths = []
        supported_formats = config.SUPPORTED_IMAGE_FORMATS if content_type == 'image' else ('.txt',)
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(supported_formats):
                    all_paths.append(os.path.join(root, filename))
        if not all_paths: return []
        
        logger.info(f"在 {os.path.basename(directory)} 目录发现 {len(all_paths)} 个 {content_type} 文件，正在检查是否为新文件...")
        existing_hashes = self.db_manager.get_all_processed_hashes()
        new_files = []
        for file_path in all_paths:
            try:
                with open(file_path, 'rb') as f:
                    file_hash = self.db_manager.create_metadata_from_file(file_path, content_type).file_hash
                if file_hash not in existing_hashes:
                    new_files.append(file_path)
            except Exception as e:
                logger.error(f"检查文件 {file_path} 时出错: {e}")
        logger.info(f"发现 {len(new_files)} 个新 {content_type} 文件需要处理。")
        return new_files
