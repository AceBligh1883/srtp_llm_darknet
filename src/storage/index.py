# src/storage/index.py
"""
只负责Elasticsearch操作的模块
"""
from typing import List, Dict
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from src.common import config
from src.common.logger import logger
from src.common.data_models import SearchResult, ContentMeta
from src.storage.database import DatabaseManager

class IndexManager:
    """封装所有与Elasticsearch的交互"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.index_name = config.ES_INDEX
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
        if not self.es.indices.exists(index=self.index_name):
            mapping = {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": config.VECTOR_DIM,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard" 
                    },
                    "metadata": {
                        "properties": {
                            "doc_id": {"type": "keyword"},
                            "content_type": {"type": "keyword"},
                            "file_path": {"type": "keyword"},
                            "created_at": {"type": "date"},
                            "file_hash": {"type": "keyword"}
                        }
                    }
                }
            }
            try:
                self.es.indices.create(index=self.index_name, mappings=mapping)
                logger.info(f"成功创建Elasticsearch索引 (新结构): {self.index_name}")
            except Exception as e:
                logger.error(f"创建Elasticsearch索引失败: {e}")
                raise
    
    def index_batch(self, items: List[dict], vectors: List[list[float]]) -> tuple[int, int]:
        """
        批量索引文档，并从 item['metadata'] 中提取完整信息。
        """
        actions = []
        for i, item in enumerate(items):
            metadata: ContentMeta = item.get('metadata')
            if not metadata or not metadata.id or i >= len(vectors) or vectors[i] is None:
                logger.warning(f"跳过索引，因为元数据或向量为空。Item keys: {item.keys()}")
                continue
            
            source = {
                "content": item.get('text_content', ''),
                "vector": vectors[i],
                "metadata": {
                    "doc_id": str(metadata.id),
                    "content_type": metadata.content_type,
                    "file_path": metadata.file_path,
                    "created_at": metadata.timestamp,
                    "file_hash": metadata.file_hash
                }
            }
            
            action = {
                "_index": self.index_name,
                "_id": str(metadata.id),
                "_source": source
            }
            actions.append(action)
        
        if not actions:
            return 0, 0

        try:
            success_count, failed_items = bulk(self.es, actions, raise_on_error=False)
            if failed_items:
                logger.warning(f"批量索引中有 {len(failed_items)} 个文档失败。")
            return success_count, len(failed_items)
        except Exception as e:
            logger.error(f"批量索引过程中发生严重错误: {e}")
            return 0, len(actions)

    def documents_exist(self, doc_ids: List[int]) -> List[int]:
        """
        检查具有给定ID的文档是否已存在于Elasticsearch中。
        """
        if not doc_ids:
            return []
        try:
            str_doc_ids = [str(doc_id) for doc_id in doc_ids]
            response = self.es.mget(
                index=self.index_name,
                body={'ids': str_doc_ids},
                _source=False
            )
            existing_ids = [
                int(doc['_id']) for doc in response['docs'] if doc.get('found', False)
            ]
            return existing_ids
        except Exception as e:
            logger.error(f"批量检查文档存在性失败: {e}")
            return []

    def hybrid_search(self, query_text: str, query_vector: np.ndarray, top_k: int = 10, content_type_filter: str = None) -> List[SearchResult]:
        """
        执行混合搜索，结合了k-NN向量搜索和BM25关键词搜索。
        """
        try:
            knn_query = { "field": "vector", "query_vector": query_vector, "k": top_k, "num_candidates": 100 }
            keyword_query = { "match": { "content": { "query": query_text, "boost": 0.2 } } }

            query_body = { "knn": knn_query, "query": keyword_query, "size": top_k }
            if content_type_filter:
                query_body["query"] = {
                    "bool": {
                        "must": [keyword_query],
                        "filter": [{"term": {"metadata.content_type": content_type_filter}}]
                    }
                }

            response = self.es.search(index=self.index_name, **query_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                doc_id = int(hit["_source"]["metadata"]["doc_id"])
                score = float(hit["_score"])
                metadata = self.db_manager.get_metadata_by_id(doc_id)
                if metadata:
                    results.append(SearchResult(doc_id=doc_id, score=score, metadata=metadata))
            return results
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def vector_search(self, query_vector: np.ndarray, top_k: int = 10, content_type_filter: str = None) -> List[SearchResult]:
        """执行纯向量k-NN检索"""
        try:
            knn_query = { "field": "vector", "query_vector": query_vector, "k": top_k, "num_candidates": 100 }
            if content_type_filter:
                knn_query["filter"] = { "term": { "metadata.content_type": content_type_filter } }
            query_body = { "knn": knn_query, "_source": ["metadata"] }
            response = self.es.search(index=self.index_name, **query_body)

            results = []
            for hit in response["hits"]["hits"]:
                doc_id = int(hit["_source"]["metadata"]["doc_id"])
                score = float(hit["_score"]) 
                metadata = self.db_manager.get_metadata_by_id(doc_id)
                if metadata:
                    results.append(SearchResult(doc_id=doc_id, score=score, metadata=metadata))
            return results
        except Exception as e:
            logger.error(f"向量检索失败: {e}", exc_info=True)
            return []
