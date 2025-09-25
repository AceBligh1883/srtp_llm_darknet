# src/analysis/graph_engine.py
import re
from typing import List, Dict, Set, Tuple
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

            relation_clean = relation.upper().replace(" ", "_").replace("-", "_")
            if relation_clean not in schema.RELATION_TYPES:
                relation_clean = "ASSOCIATED_WITH"

            head_type_clean = re.sub(r'[^a-zA-Z0-9_]', '', head_type)
            tail_type_clean = re.sub(r'[^a-zA-Z0-9_]', '', tail_type)
            if not head_type_clean or head_type_clean not in schema.ENTITY_TYPES: 
                head_type_clean = "Thing"
            if not tail_type_clean or tail_type_clean not in schema.ENTITY_TYPES: 
                tail_type_clean = "Thing"

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
        logger.info(f"成功处理 {len(triples)} 条三元组 (来源: {source_doc_id})")

    def _generate_variants(self, keyword: str) -> List[str]:
        """为单个关键词生成多种大小写、空格和连字符的标准化变体。"""
        base = re.sub(r'[^a-zA-Z0-9\s-]', '', keyword).strip()
        with_space = base.replace('-', ' ') 
        no_space = with_space.replace(' ', '')
        variants = set()
        for form in [with_space, no_space]:
            if form: 
                variants.add(form.lower()) 
                variants.add(form.upper())   
                variants.add(form.title())    
        variants.add(keyword)
        return list(variants)

    def expand_with_graph(self, keywords: List[str], limit_per_keyword: int = 5) -> Tuple[Set[str], List[Dict]]:
        if not keywords:
            return set(), []

        initial_keywords_set = set(keywords)
        expanded_keywords = set(keywords)
        graph_facts = []
        seen_relations = set()
        
        with self.driver.session() as session:
            all_variants = list(set(kw for kw in keywords for kw in self._generate_variants(kw)))
            exact_match_query = """
                UNWIND $variants AS variant
                MATCH (seed_node) WHERE seed_node.name = variant
                RETURN elementId(seed_node) AS node_id, seed_node.name AS node_name
            """
            result = session.run(exact_match_query, variants=all_variants)
            
            found_nodes = {record["node_id"]: record["node_name"] for record in result}
            found_node_ids = set(found_nodes.keys())
            
            if not found_node_ids:
                index_names = [f"{entity_type.lower()}_name_idx" for entity_type in schema.ENTITY_TYPES]
                fuzzy_match_query = """
                    CALL db.index.fulltext.queryNodes($index, $keyword) YIELD node
                    RETURN elementId(node) AS node_id, node.name AS node_name
                """
                for kw in keywords:
                    for index_name in index_names:
                        result = session.run(fuzzy_match_query, index=index_name, keyword=kw + '*')
                        for record in result:
                            if record["node_id"] not in found_node_ids:
                                found_nodes[record["node_id"]] = record["node_name"]
                                found_node_ids.add(record["node_id"])

            if not found_node_ids:
                return expanded_keywords, graph_facts
            
            expansion_query = """
                MATCH (seed_node) WHERE elementId(seed_node) IN $ids
                MATCH (seed_node)-[r]-(neighbor)
                RETURN 
                    seed_node.name AS head, labels(seed_node)[0] AS head_type,
                    type(r) AS relation,
                    neighbor.name AS tail, labels(neighbor)[0] AS tail_type
                LIMIT $limit
            """
            results = session.run(expansion_query, ids=list(found_node_ids), limit=len(found_node_ids) * limit_per_keyword)
            
            for record in results:
                relation_key = frozenset([record["head"], record["tail"], record["relation"]])
                if relation_key not in seen_relations:
                    graph_facts.append({
                        "head": record["head"], "head_type": record["head_type"],
                        "relation": record["relation"],
                        "tail": record["tail"], "tail_type": record["tail_type"]
                    })
                    seen_relations.add(relation_key)
                    
                    expanded_keywords.add(record["head"])
                    expanded_keywords.add(record["tail"])
        return expanded_keywords, graph_facts
    