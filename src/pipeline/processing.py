# src/pipeline/processing.py
"""
核心处理流水线，负责编排处理流程
"""
import os
from tqdm import tqdm

from src.common import config
from src.common.logger import logger
from src.storage.database import DatabaseManager
from src.storage.index import IndexManager
from src.features.embedding import EmbeddingGenerator
from src.features.text_processing import TextProcessor
from src.features.image_processing import ImageProcessor

class ProcessingPipeline:
    """编排一个文件的完整处理与索引流程"""

    def __init__(self):
        logger.info("初始化处理流水线...")
        self.db_manager = DatabaseManager(config.DB_PATH)
        self.index_manager = IndexManager(self.db_manager)
        self.embed_generator = EmbeddingGenerator()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        logger.info("处理流水线初始化完成。")

    def run_for_file(self, file_path: str, content_type: str) -> bool:
        """对单个文件执行完整的处理和索引流程"""
        try:
            # 1. 创建元数据
            metadata = self.db_manager.create_metadata_from_file(file_path, content_type)

            # 2. 保存元数据到SQLite，获取ID
            # 如果文件已处理过，会直接返回ID并跳过后续步骤
            doc_id = self.db_manager.save_metadata(metadata)
            if not doc_id:
                return False # 保存失败或已存在但获取ID失败

            # 3. 预处理和特征化（生成向量）
            if content_type == 'text':
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                processed_text = self.text_processor.process(raw_text)
                vector = self.embed_generator.get_text_embedding(processed_text)
            elif content_type == 'image':
                pil_image = self.image_processor.process(file_path)
                vector = self.embed_generator.get_image_embedding(pil_image)
            else:
                logger.warning(f"不支持的内容类型: {content_type}")
                return False

            if vector is None:
                logger.error(f"文件向量生成失败: {file_path}")
                return False

            # 4. 索引向量到Elasticsearch
            success = self.index_manager.index_document(doc_id, vector)
            return success

        except Exception as e:
            logger.error(f"处理文件 '{file_path}' 失败: {e}")
            return False

    def process_directory(self, directory: str, content_type: str):
        """处理指定目录下的所有文件"""
        if not os.path.exists(directory):
            logger.error(f"目录不存在: {directory}")
            return

        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if content_type == 'image' and not file.lower().endswith(tuple(config.SUPPORTED_IMAGE_FORMATS)):
                    continue
                if content_type == 'text' and not file.lower().endswith('.txt'):
                    continue
                file_paths.append(os.path.join(root, file))
        
        if not file_paths:
            logger.warning(f"在目录 {directory} 中未找到 {content_type} 类型的文件。")
            return

        logger.info(f"发现 {len(file_paths)} 个 {content_type} 文件，开始处理...")
        success_count = 0
        for file_path in tqdm(file_paths, desc=f"处理 {content_type} 文件"):
            if self.run_for_file(file_path, content_type):
                success_count += 1
        
        logger.info(f"{content_type} 文件处理完成。成功: {success_count}/{len(file_paths)}")
