# src/analysis/graph_engine.py

import json
import re
from typing import List, Dict
from neo4j import GraphDatabase, exceptions

from src.common import config, schema
from src.common.logger import logger
from src.clients.gemini_client import GeminiClient

class KnowledgeGraphEngine:
    """
    负责从文本中提取知识三元组并构建知识图谱。
    """
    
    KNOWLEDGE_EXTRACTION_PROMPT = """
        你是一个高度精确的知识图谱构建机器人。你的任务是从提供的文本中提取信息，并将其结构化为知识图谱。
        请遵循以下严格的规则：
        1.  首先，识别出文本中所有符合以下【实体类型】的实体。
        2.  然后，找出这些实体之间符合以下【关系类型】的关系。
        3.  最后，将结果格式化为一个JSON列表，其中每个元素都是一个包含"head"（头实体）, "relation"（关系）, "tail"（尾实体）和 "head_type", "tail_type" 键的字典。
        4.  实体和关系必须严格来自下面定义的类型。如果实体或关系不符合定义，请不要提取。
        5.  如果文本中没有可提取的有效三元组，请返回一个空列表 `[]`。

        --- 【实体类型定义】 ---
        {entity_schema}
        
        --- 【关系类型定义】 ---
        {relation_schema}

        --- 【文本内容】 ---
        {text_content}

        --- 【输出示例】 ---
        [
        {{ 
            "head": "Conti", "head_type": "ORGANIZATION",
            "relation": "TARGETS",
            "tail": "Healthcare Sector", "tail_type": "ORGANIZATION"
        }},
        {{
            "head": "Conti", "head_type": "ORGANIZATION",
            "relation": "USES",
            "tail": "TrickBot", "tail_type": "MALWARE"
        }}
        ]

        CRITICAL: 你的回答必须是一个严格的、不含任何额外文本的JSON列表。

        提取的知识三元组 (JSON格式):
        """

    def __init__(self):
        self.llm_client = GeminiClient()
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
        """关闭与Neo4j的连接。"""
        if self.driver:
            self.driver.close()
            logger.info("已断开与Neo4j的连接。")

    def extract_triples_from_text(self, text_content: str, doc_id: str) -> List[Dict]:
        """从单个文本文档中提取知识三元组。"""
        content_snippet = text_content
        
        prompt = self.KNOWLEDGE_EXTRACTION_PROMPT.format(
            entity_schema=json.dumps(schema.ENTITY_TYPES, indent=2),
            relation_schema=", ".join(schema.RELATION_TYPES),
            text_content=content_snippet
        )
        
        response_str = self.llm_client.generate(prompt)
        
        try:
            match = re.search(r'\[.*\]', response_str, re.DOTALL)
            if not match:
                logger.warning(f"在文档 {doc_id} 的LLM响应中未找到列表格式。")
                return []
            
            json_str = match.group(0)
            triples = json.loads(json_str)
            
            if isinstance(triples, list) and all('head' in t and 'relation' in t and 'tail' in t for t in triples):
                logger.info(f"成功从文档 {doc_id} 中提取 {len(triples)} 个三元组。")
                return triples
            else:
                logger.warning(f"文档 {doc_id} 提取的三元组格式不正确。")
                return []
        except json.JSONDecodeError:
            logger.error(f"解析文档 {doc_id} 的三元组JSON失败。响应: {response_str[:200]}...")
            return []

    def add_triples_to_graph(self, triples: List[Dict], source_doc_id: str):
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

