# src/storage/index.py
"""
只负责Elasticsearch操作的模块
"""
from typing import List
import numpy as np
from elasticsearch import Elasticsearch

from src.common import config
from src.common.logger import logger
from src.common.data_models import SearchResult
from src.storage.database import DatabaseManager

class IndexManager:
    """封装所有与Elasticsearch的交互"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        try:
            self.es = Elasticsearch(
                hosts=[{'host': config.ES_HOST, 'port': config.ES_PORT, 'scheme': 'http'}],
                request_timeout=config.ES_TIMEOUT
            )
            if not self.es.ping():
                raise ConnectionError("无法连接到Elasticsearch")
            self._init_index()
        except Exception as e:
            logger.error(f"初始化Elasticsearch连接失败: {e}")
            raise

    def _init_index(self):
        """如果索引不存在，则创建它"""
        if not self.es.indices.exists(index=config.ES_INDEX):
            mapping = {
                "properties": {
                    "doc_id": {"type": "integer"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": config.VECTOR_DIM,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
            try:
                self.es.indices.create(index=config.ES_INDEX, mappings=mapping)
                logger.info(f"成功创建Elasticsearch索引: {config.ES_INDEX}")
            except Exception as e:
                logger.error(f"创建Elasticsearch索引失败: {e}")
                raise

    def index_document(self, doc_id: int, vector: np.ndarray) -> bool:
        """
        索引单个文档的向量。
        Elasticsearch中的文档ID与SQLite中的主键ID保持一致。
        """
        try:
            doc = {"doc_id": doc_id, "vector": vector}
            self.es.index(index=config.ES_INDEX, id=str(doc_id), document=doc)
            logger.info(f"文档已索引到ES, ID: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"索引文档(ID: {doc_id})到ES失败: {e}")
            return False

    def vector_search(self, query_vector: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """执行纯向量检索"""
        try:
            knn_query = {
                "field": "vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 100
            }
            response = self.es.search(
                index=config.ES_INDEX,
                knn=knn_query,
                _source=["doc_id"]
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                doc_id = int(hit["_source"]["doc_id"])
                score = float(hit["_score"])
                metadata = self.db_manager.get_metadata_by_id(doc_id)
                if metadata:
                    results.append(SearchResult(doc_id=doc_id, score=score, metadata=metadata))
            return results
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
