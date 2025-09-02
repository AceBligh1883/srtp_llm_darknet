# src/analysis/graph_engine.py
import re
from typing import List, Dict
from neo4j import GraphDatabase, exceptions
from src.common import config, schema
from src.common.logger import logger

class KnowledgeGraphEngine:
    """负责从文本中提取知识三元组并构建知识图谱。"""
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI, 
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
            logger.info("成功连接到Neo4j数据库。")
        except exceptions.AuthError as e:
            logger.error(f"Neo4j认证失败: {e}")
            raise
        except Exception as e:
            logger.error(f"无法连接到Neo4j: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("已断开与Neo4j的连接。")

    def add_triples(self, triples: List[Dict], source_doc_id: str):
        """将提取的三元组批量写入Neo4j。"""
        if not triples:
            return
            
        with self.driver.session(database="neo4j") as session:
            session.execute_write(self._create_triples_tx, triples, source_doc_id)
        logger.info(f"已将 {len(triples)} 个三元组（来源: {source_doc_id}）写入图谱。")

    @staticmethod
    def _create_triples_tx(tx, triples: List[Dict], source_doc_id: str):
        for i, triple in enumerate(triples):
            head = triple.get("head")
            tail = triple.get("tail")
            relation = triple.get("relation")
            head_type = triple.get("head_type", "Thing")
            tail_type = triple.get("tail_type", "Thing")

            if not all([head, tail, relation]):
                logger.warning(f"跳过不完整的三元组 #{i} (来源: {source_doc_id}): {triple}")
                continue

            relation_clean = relation.upper().replace(" ", "_")
            if relation_clean not in schema.RELATION_TYPES:
                relation_clean = "ASSOCIATED_WITH"

            head_type_clean = re.sub(r'[^a-zA-Z0-9_]', '', head_type)
            tail_type_clean = re.sub(r'[^a-zA-Z0-9_]', '', tail_type)
            if not head_type_clean: head_type_clean = "Thing"
            if not tail_type_clean: tail_type_clean = "Thing"    

            query = (
                f"MERGE (h:`{head_type_clean}` {{name: $head_name}}) "
                f"MERGE (t:`{tail_type_clean}` {{name: $tail_name}}) "
                f"MERGE (h)-[r:`{relation_clean}`]->(t) "
                "ON CREATE SET r.source = [$source] "
                "ON MATCH SET r.source = r.source + $source"
            )
            
            params = {
                "head_name": head,
                "tail_name": tail,
                "source": source_doc_id
            }
            tx.run(query, **params)

