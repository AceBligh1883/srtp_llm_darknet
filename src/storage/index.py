# src/storage/index.py
"""
只负责Elasticsearch操作的模块
"""
import os
from typing import List
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from src.common import config
from src.common.logger import logger
from src.common.data_models import SearchResult
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
                    "doc_id": {"type": "integer"},
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
                    "content_type": { "type": "keyword" } 
                }
            }
            try:
                self.es.indices.create(index=config.ES_INDEX, mappings=mapping)
                logger.info(f"成功创建Elasticsearch索引: {config.ES_INDEX}")
            except Exception as e:
                logger.error(f"创建Elasticsearch索引失败: {e}")
                raise
    
    def document_exists(self, doc_id: int) -> bool:
        """
        检查具有给定ID的文档是否已存在于Elasticsearch中。
        """
        try:
            return self.es.exists(index=self.index_name, id=str(doc_id))
        except Exception as e:
            logger.error(f"检查文档存在性失败 (ID: {doc_id}): {e}")
            return False
        
    def index_document(self, doc_id: int, vector: np.ndarray) -> bool:
        """
        索引单个文档的向量。
        Elasticsearch中的文档ID与SQLite中的主键ID保持一致。
        """
        try:
            doc = {"doc_id": doc_id, "vector": vector}
            self.es.index(index=self.index_name, id=str(doc_id), document=doc)
            return True
        except Exception as e:
            logger.error(f"索引文档(ID: {doc_id})到ES失败: {e}")
            return False
    
    def index_batch(self, items: List[dict], vectors: List[list[float]]) -> tuple[int, int]:
        """
        使用 bulk API 高效地批量索引文档。
        :param items: 包含 'doc_id' 的字典列表
        :param vectors: 向量列表
        :return: (成功数量, 失败数量)
        """
        actions = []
        for i, item in enumerate(items):
            doc_id = item['doc_id']
            if i >= len(vectors) or vectors[i] is None:
                logger.warning(f"跳过索引，因为文档ID {doc_id} 的向量为空。")
                continue
            vector = vectors[i]
            
            content_to_index = item.get('text_content', '')
            content_type_to_index = item.get('content_type', '')

            source = {
                "doc_id": doc_id,
                "vector": vector,
                "content": content_to_index,
                "content_type": content_type_to_index
            }
            
            action = {
                "_index": self.index_name,
                "_id": str(doc_id),
                "_source": source
            }
            actions.append(action)
        
        if not actions:
            return 0, 0

        try:
            success_count, failed_items = bulk(self.es, actions, raise_on_error=False)
            
            if failed_items:
                logger.warning(f"批量索引中有 {len(failed_items)} 个文档失败。")
                for i, item in enumerate(failed_items):
                    if i >= 5:
                        logger.warning("...更多失败日志已省略...")
                        break
                    error_reason = item.get('index', {}).get('error', '未知错误')
                    failed_doc_id = item.get('index', {}).get('_id', '未知ID')
                    logger.error(f"  - 文档ID: {failed_doc_id}, 失败原因: {error_reason}")
            return success_count, len(failed_items)
        except Exception as e:
            logger.error(f"批量索引过程中发生严重错误: {e}")
            return 0, len(actions)

    def hybrid_search(self, query_text: str, query_vector: np.ndarray, top_k: int = 10, content_type_filter: str = None) -> List[SearchResult]:
        """
        执行混合搜索，结合了k-NN向量搜索和BM25关键词搜索。
        """
        try:
            knn_query = {
                "field": "vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 100 
            }
            keyword_query = {
                "match": {
                    "content": {
                        "query": query_text,
                        "boost": 0.2
                    }
                }
            }

            query_body = {
                "knn": knn_query,
                "query": keyword_query,
                "size": top_k
            }

            if content_type_filter:
                query_body["query"] = {
                    "bool": {
                        "must": [keyword_query],
                        "filter": [
                            {"term": {"content_type": content_type_filter}}
                        ]
                    }
                }

            response = self.es.search(
                index=self.index_name,
                **query_body
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

    def vector_search(self, query_vector: np.ndarray, top_k: int = 10, content_type_filter: str = None) -> List[SearchResult]:
        """执行纯向量k-NN检索"""
        try:
            knn_query = {
                "field": "vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 100
            }
            
            query_body = {
                "knn": knn_query,
                "_source": ["doc_id"]
            }
            if content_type_filter:
                query_body["query"] = {
                    "bool": {
                        "filter": [
                            {"term": {"content_type": content_type_filter}}
                        ]
                    }
                }

            response = self.es.search(
                index=self.index_name,
                **query_body
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
            logger.error(f"向量检索失败: {e}", exc_info=True)
            return []
