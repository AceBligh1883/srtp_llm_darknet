# src/services/reranker_service.py
from sentence_transformers import CrossEncoder
from typing import Iterable, List, Tuple
import torch
from src.common.data_models import SearchResult
from src.common.logger import logger

class RerankerService:
    def __init__(self, model_name):
        logger.debug(f"正在加载Rerank模型: {model_name}...")
        try:
            device='cuda' if torch.cuda.is_available() else 'cpu'
            self.model = CrossEncoder(model_name, max_length=512, device=device)
            logger.debug("Rerank模型加载成功。")
        except Exception as e:
            logger.error(f"加载Rerank模型失败: {e}", exc_info=True)
            raise

    def rerank(self, query: str, results: Iterable[Tuple[SearchResult, str]]) -> List[SearchResult]:
        """
        对检索结果进行重排序。

        Args:
            query (str): 用户的原始查询。
            results (List[SearchResult]): 从搜索引擎初步检索到的结果列表。

        Returns:
            List[SearchResult]: 经过重排序并更新了分数的结果列表。
        """
        results = list(results)
        if not results or not query: return []

        original_results = [item[0] for item in results]
        contents_to_rank = [item[1] for item in results]

        sentence_pairs = [(query, content) for content in contents_to_rank]
        logger.info(f"正在对 {len(original_results)} 个结果进行重排序...")
        scores = self.model.predict(sentence_pairs, show_progress_bar=False, batch_size=32).tolist()
        for i, res in enumerate(original_results):
            res.score = float(scores[i])
        original_results.sort(key=lambda x: x.score, reverse=True)
        return original_results

