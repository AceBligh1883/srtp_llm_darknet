# src/search/engine.py
"""
统一的检索引擎
"""
import os
from typing import List
from src.common import config
from src.common.logger import logger
from src.common.data_models import SearchResult
from src.storage.database import DatabaseManager
from src.storage.index import IndexManager
from src.services.embedding_service import EmbeddingService
from src.processing.text_processor import TextProcessor
from src.processing.image_processor import ImageProcessor
from src.processing.translator import translate_to_english

class SearchEngine:
    """
    多模态检索引擎，负责将用户查询转换为向量并与索引交互。
    """
    def __init__(self):
        logger.debug("初始化检索引擎...")
        db_manager = DatabaseManager(config.DB_PATH)
        self.index_manager = IndexManager(db_manager)
        self.embedding_service = EmbeddingService()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        logger.debug("检索引擎初始化完成。")

    def search_by_text(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """通过文本查询进行向量搜索"""
        logger.info(f"执行文本搜索: '{query[:50]}...'")
        translated_query = translate_to_english(query)
        processed_query = self.text_processor.process(translated_query)
        query_vectors = self.embedding_service.get_text_embeddings([processed_query])

        if not query_vectors:
            logger.error("查询文本向量生成失败。")
            return []
        query_vector = query_vectors[0]
        text_results = self.index_manager.hybrid_search(
            query, query_vector, top_k, content_type_filter='text'
        )
        image_results = self.index_manager.vector_search(
            query_vector, top_k, content_type_filter='image'
        )
        combined_results = text_results + image_results
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results

    def search_by_image(self, image_path: str, top_k: int) -> List[SearchResult]:
        """
        使用图像文件路径进行k-NN搜索。
        """
        logger.info(f"执行图像搜索: '{image_path}'")
        pil_image = self.image_processor.process(image_path)
        if not pil_image: return []
        query_vectors = self.embedding_service.get_image_embeddings([pil_image])
        
        if query_vectors is None:
            logger.error(f"为查询 '{image_path}' 生成向量失败。")
            return []
        query_vector = query_vectors[0]
        image_results = self.index_manager.vector_search(
            query_vector, top_k, content_type_filter='image'
        )
        text_results = self.index_manager.vector_search(
            query_vector, top_k, content_type_filter='text'
        )
        combined_results = text_results + image_results
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results
