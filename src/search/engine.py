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
        query_vectors = self.embed_generator.get_text_embeddings([processed_query], prompt_name="query")

        if not query_vectors:
            logger.error("查询文本向量生成失败。")
            return []
        query_vector = query_vectors[0]
        candidate_results = self.index_manager.vector_search(query_vector, top_k * 5)
        
        text_results = []
        other_results = []
        
        for res in candidate_results:
            if res.metadata and res.metadata.content_type == 'text':
                text_results.append(res)
            else:
                other_results.append(res)
        
        final_results = text_results + other_results
        
        return final_results[:top_k]

    def search_by_image(self, query_input: str, top_k: int) -> List[SearchResult]:
        """
        使用图像文件路径进行k-NN搜索。
        """
        query_vector = None
        if os.path.exists(query_input) and query_input.lower().endswith(config.SUPPORTED_IMAGE_FORMATS):
            logger.info(f"检测到图像文件路径，执行图搜图: '{query_input}'")
            try:
                pil_image = self.image_processor.process(query_input)
                if pil_image:
                    image_vectors = self.embed_generator.get_image_embeddings([pil_image])
                    if image_vectors:
                        query_vector = image_vectors[0]
            except Exception as e:
                logger.error(f"处理查询图像 '{query_input}' 时出错: {e}")
                return []
        else:
            logger.info(f"检测到文本查询，执行文搜图: '{query_input}'")
            processed_query = self.text_processor.process(query_input)
            text_vectors = self.embed_generator.get_text_embeddings([processed_query], prompt_name="query")
            if text_vectors:
                query_vector = text_vectors[0]
        
        if query_vector is None:
            logger.error(f"为查询 '{query_input}' 生成向量失败。")
            return []
        
        candidate_results = self.index_manager.vector_search(query_vector, top_k * 5)

        image_results = []
        for res in candidate_results:
            if res.metadata and res.metadata.content_type == 'image':
                image_results.append(res)
                
                if len(image_results) >= top_k:
                    break
        
        return image_results