# src/search/engine.py
"""
统一的检索引擎
"""
from typing import List
from src.common import config
from src.common.logger import logger
from src.common.data_models import SearchResult
from src.storage.database import DatabaseManager
from src.storage.index import IndexManager
from src.features.embedding import EmbeddingGenerator
from src.features.text_processing import TextProcessor
from src.features.image_processing import ImageProcessor

class SearchEngine:
    """
    多模态检索引擎，负责将用户查询转换为向量并与索引交互。
    """
    def __init__(self):
        logger.info("初始化检索引擎...")
        db_manager = DatabaseManager(config.DB_PATH)
        self.index_manager = IndexManager(db_manager)
        self.embed_generator = EmbeddingGenerator()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        logger.info("检索引擎初始化完成。")

    def search_by_text(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """通过文本查询进行向量搜索"""
        logger.info(f"执行文本搜索: '{query[:50]}...'")
        processed_query = self.text_processor.process(query)
        query_vector = self.embed_generator.get_text_embedding(processed_query)

        if query_vector is None:
            logger.error("查询文本向量生成失败。")
            return []

        return self.index_manager.vector_search(query_vector, top_k)

    def search_by_image(self, image_path: str, top_k: int) -> List[SearchResult]:
        """
        使用图像文件路径进行k-NN搜索。
        """
        logger.info(f"执行图像搜索: '{image_path}'")
        try:
            # 1. 为查询图像生成向量
            query_vector = self.embedding_generator.generate_for_image(image_path)
            if query_vector is None:
                logger.error(f"为图像 {image_path} 生成向量失败。")
                return []
            
            # 2. 执行k-NN搜索
            return self._perform_knn_search(query_vector, top_k)
        except Exception as e:
            logger.error(f"图像搜索过程中发生错误: {e}")
            return []