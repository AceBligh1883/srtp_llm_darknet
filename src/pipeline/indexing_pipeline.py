# src/pipelines/indexing_pipeline.py
from src.common import config
from src.common.logger import logger
from src.storage.database import DatabaseManager
from src.storage.index import IndexManager
from src.etl.loader import DataLoader
from src.etl.indexer import Indexer
from src.services.embedding_service import EmbeddingService

class IndexingPipeline:
    """编排一个完整的数据索引流程。"""
    def __init__(self):
        db_manager = DatabaseManager(config.DB_PATH)
        index_manager = IndexManager(db_manager)
        embedding_service = EmbeddingService()
        self.loader = DataLoader(db_manager)
        self.indexer = Indexer(db_manager, index_manager, embedding_service)
        logger.info("索引流水线初始化完成。")

    def run(self):
        """执行完整的文本和图像索引流程。"""
        logger.info("--- 开始执行文本索引流程 ---")
        text_files = self.loader.discover_new_files(config.TEXT_DIR, 'text')
        if text_files:
            self.indexer.run(text_files, 'text')
        else:
            logger.info("没有新的文本文件需要索引。")

        logger.info("--- 开始执行图像索引流程 ---")
        all_image_files = []
        for image_dir in config.IMAGE_SOURCE_DIRS:
            all_image_files.extend(self.loader.discover_new_files(image_dir, 'image'))
        if all_image_files:
            self.indexer.run(all_image_files, 'image')
        else:
            logger.info("没有新的图像文件需要索引。")
