# src/services/knowledge_extractor.py
import json
import re
from typing import List, Dict
from src.common import prompts, schema
from src.common.logger import logger
from src.clients import get_llm_client

class KnowledgeExtractor:
    """
    从文本内容中提取结构化的知识三元组。
    """
    def __init__(self):
        self.llm_client = get_llm_client()

    def extract(self, text_content: str, doc_id: str) -> List[Dict]:
        """
        从文本中提取三元组。
        """
        if not text_content or not text_content.strip():
            return []

        prompt = prompts.KNOWLEDGE_EXTRACTION_PROMPT.format(
            entity_schema=json.dumps(schema.ENTITY_TYPES, indent=2),
            relation_schema=", ".join(schema.RELATION_TYPES),
            text_content=text_content
        )
        
        response_str = self.llm_client.generate(prompt)
        
        try:
            match = re.search(r'\[.*\]', response_str, re.DOTALL)
            if not match:
                logger.warning(f"在文档 {doc_id} 的LLM响应中未找到列表格式。")
                return []
            
            triples = json.loads(match.group(0))
            
            if isinstance(triples, list) and all('head' in t and 'relation' in t and 'tail' in t for t in triples):
                logger.info(f"成功从文档 {doc_id} 中提取 {len(triples)} 个三元组。")
                return triples
            else:
                logger.warning(f"文档 {doc_id} 提取的三元组格式不正确。")
                return []
        except json.JSONDecodeError:
            logger.error(f"解析文档 {doc_id} 的三元组JSON失败。响应: {response_str[:200]}...")
            return []
