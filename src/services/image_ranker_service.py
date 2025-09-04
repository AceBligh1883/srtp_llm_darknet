# src/services/image_ranker_service.py
from typing import List, Optional
from src.common.data_models import SearchResult
from src.common.logger import logger
from src.services.embedding_service import EmbeddingService
from src.processing.image_processor import ImageProcessor
from src.common import config
import numpy as np

class ImageRankerService:
    """图像的关联性融合排序。"""
    def __init__(self):
        logger.debug("初始化 ImageRankerService...")
        self.embedding_service = EmbeddingService()
        self.image_processor = ImageProcessor()
        logger.debug("ImageRankerService 初始化完成。")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个numpy向量的余弦相似度"""
        if vec1 is None or vec2 is None:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def rank_images(
        self,
        master_query: str,
        query_image_vector: Optional[List[float]],
        image_results: List[SearchResult],
        limit: int = config.RAG_TOP_K
    ) -> List[SearchResult]:
        
        if not image_results:
            return []

        logger.info(f"正在对 {len(image_results)} 张图片进行关联性融合排序...")

        pil_images = [self.image_processor.process(res.metadata.file_path) for res in image_results]
        valid_images_with_results = [(img, res) for img, res in zip(pil_images, image_results) if img]
        
        if not valid_images_with_results:
            return []
            
        candidate_image_vectors = self.embedding_service.get_image_embeddings([img for img, res in valid_images_with_results])
        if candidate_image_vectors is None:
            logger.error("无法为候选图片生成向量。")
            return image_results[:limit]

        master_query_vector = None
        if master_query:
            text_vectors = self.embedding_service.get_text_embeddings([master_query])
            if text_vectors:
                master_query_vector = np.array(text_vectors[0])

        query_image_np_vector = np.array(query_image_vector) if query_image_vector else None

        final_scores = []
        for i, (img, res) in enumerate(valid_images_with_results):
            candidate_vec = np.array(candidate_image_vectors[i])
            
            score_text = self._cosine_similarity(candidate_vec, master_query_vector)
            score_image = self._cosine_similarity(candidate_vec, query_image_np_vector)
            
            fusion_score = score_text + score_image
            final_scores.append(fusion_score)

        for i, (img, res) in enumerate(valid_images_with_results):
            res.score = final_scores[i]

        ranked_results = [res for img, res in valid_images_with_results]
        ranked_results.sort(key=lambda x: x.score, reverse=True)

        return ranked_results[:limit]
