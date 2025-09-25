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
        self.llm_client = get_llm_client("deepseek")

    def extract_from_documents(self, documents: List[Dict[str, str]]) -> Dict[str, List[Dict]]:
        """
        LLM的响应中解析出三元组列表。
        """
        if not documents:
            return {}
        is_batch = len(documents) > 1
        doc_ids = [doc['doc_id'] for doc in documents]
        if is_batch:
            content_parts = []
            for doc in documents:
                part = f"--- DOCUMENT START: {doc['doc_id']} ---\n{doc['content']}\n--- DOCUMENT END: {doc['doc_id']} ---"
                content_parts.append(part)
            text_content = "\n\n".join(content_parts)
        else:
            text_content = documents[0]['content']
            
        prompt = prompts.KNOWLEDGE_EXTRACTION_PROMPT.format(
            entity_schema=json.dumps(schema.ENTITY_TYPES, indent=2, ensure_ascii=False),
            relation_schema=", ".join(schema.RELATION_TYPES),
            text_content=text_content
        )
        response_text = self.llm_client.generate(prompt)
        try:
            if is_batch:
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if not match:
                    logger.error(f"批处理响应中未找到JSON对象。Doc IDs: {doc_ids}")
                    return {doc_id: [] for doc_id in doc_ids}
                
                all_results = json.loads(match.group(0))
                if isinstance(all_results, dict):
                    for doc_id in doc_ids:
                        if doc_id not in all_results:
                            all_results[doc_id] = []
                    return all_results
                else:
                    logger.error(f"批处理响应不是一个字典。Doc IDs: {doc_ids}")
                    return {doc_id: [] for doc_id in doc_ids}
            else:
                match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if not match:
                    logger.warning(f"单文档响应中未找到列表格式。Doc ID: {doc_ids[0]}")
                    return {doc_ids[0]: []}
                
                triples = json.loads(match.group(0))
                if isinstance(triples, list):
                    return {doc_ids[0]: triples}
                else:
                    logger.warning(f"单文档响应不是一个列表。Doc ID: {doc_ids[0]}")
                    return {doc_ids[0]: []}
        
        except json.JSONDecodeError:
            logger.error(f"解析JSON响应失败。Doc IDs: {doc_ids}。响应预览: {response_text[:300]}...")
            return {doc_id: [] for doc_id in doc_ids}

        
    