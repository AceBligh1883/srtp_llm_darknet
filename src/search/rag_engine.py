# src/search/rag_engine.py
"""
检索增强生成 (RAG) 引擎
"""
import re
from typing import List
from src.processing.translator import translate_to_english
from src.common import config
from src.common.logger import logger
from src.common.data_models import RAGReport
from src.search.engine import SearchEngine
from src.services.reranker_service import RerankerService
from src.clients import get_llm_client
from src.processing.image_processor import ImageProcessor
from src.services.content_enhancer import ContentEnhancer
from src.pipeline.rag_pipeline import QueryUnderstandingStage, RetrievalStage, RankingStage, SynthesisStage
from src.analysis.graph_engine import KnowledgeGraphEngine

class RAGEngine:
    """负责协调检索和生成，以回答用户问题的引擎。"""
    def __init__(self):
        logger.debug("正在初始化RAG引擎...")
        search_engine = SearchEngine()
        llm_client = get_llm_client()
        image_processor = ImageProcessor()
        content_enhancer = ContentEnhancer(llm_client, image_processor)
        reranker = RerankerService(config.RERANKER_MODEL)
        self.kg_engine = KnowledgeGraphEngine()
        self.query_stage = QueryUnderstandingStage(content_enhancer)
        self.retrieval_stage = RetrievalStage(search_engine)
        self.ranking_stage = RankingStage(content_enhancer, reranker)
        self.synthesis_stage = SynthesisStage(llm_client)
        self.path_to_placeholder_map = {}
        logger.debug("RAG引擎初始化成功。")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取潜在的实体关键词"""
        logger.info(translate_to_english(text))
        keywords = re.findall(
            r'"([^"]+)"|'  # Quoted phrases
            r'\b([A-Z][a-zA-Z0-9-]{2,}(?:\s+[A-Z][a-zA-Z0-9-]{2,})*)\b|' 
            r'\b(CVE-\d{4}-\d{4,})\b',  # CVEs
            translate_to_english(text)
        )
        cleaned_keywords = {item for sublist in keywords for item in sublist if item}
        return list(cleaned_keywords)


    def ask(self, question: str, image_path: str = None) -> RAGReport:
        """执行完整的RAG流程。"""
        try:
            query_analysis = self.query_stage.run(question, image_path)
            all_initial_keywords = set()
            master_query_keywords = self._extract_keywords(query_analysis["master_query_text"])
            all_initial_keywords.update(master_query_keywords)

            for query_text in query_analysis["queries"]:
                keywords_from_query = self._extract_keywords(query_text)
                all_initial_keywords.update(keywords_from_query)

            all_initial_keywords = {kw for kw in all_initial_keywords}
            if all_initial_keywords:
                expanded_keywords, graph_facts = self.kg_engine.expand_with_graph(list(all_initial_keywords))
            else:
                expanded_keywords, graph_facts = set(), []
            original_query_count = len(query_analysis["queries"])
            query_analysis["queries"].update(expanded_keywords)
            logger.info(f"查询已通过KG增强: 从 {original_query_count} 个查询扩展到 {len(query_analysis['queries'])} 个。")
            initial_results = self.retrieval_stage.run(query_analysis)
            ranked_results, descriptions = self.ranking_stage.run(query_analysis, initial_results)
            report_body, path_map = self.synthesis_stage.run(
                question=question,
                evidence=ranked_results,
                descriptions=descriptions,
                query_analysis=query_analysis,
                kg_facts=graph_facts
            )
            return RAGReport(
                question=question,
                query_image_path=image_path,
                answer=report_body,
                evidence=ranked_results,
                image_references=path_map,
                graph_facts=graph_facts 
            )
        
        except RuntimeError as e:
            logger.error(f"RAG流程执行失败: {e}")
            return RAGReport(
                question=question,
                query_image_path=image_path,
                answer=f"抱歉，处理您的请求时发生错误: {e}"
            )
        except Exception as e:
            logger.critical(f"RAG流程发生未知严重错误: {e}", exc_info=True)
            return RAGReport(
                question=question,
                query_image_path=image_path,
                answer="抱歉，处理您的请求时发生了一个意外的内部错误。"
            )