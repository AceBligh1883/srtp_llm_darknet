# src/search/rag_engine.py
"""
检索增强生成 (RAG) 引擎
"""
from src.common import config
from src.common.logger import logger
from src.common.data_models import RAGReport
from src.search.engine import SearchEngine
from src.services.reranker_service import RerankerService
from src.clients import get_llm_client
from src.processing.image_processor import ImageProcessor
from src.services.content_enhancer import ContentEnhancer
from src.pipeline.rag_pipeline import QueryUnderstandingStage, RetrievalStage, RankingStage, SynthesisStage

class RAGEngine:
    """负责协调检索和生成，以回答用户问题的引擎。"""
    def __init__(self):
        logger.debug("正在初始化RAG引擎...")
        search_engine = SearchEngine()
        llm_client = get_llm_client()
        image_processor = ImageProcessor()
        content_enhancer = ContentEnhancer(llm_client, image_processor)
        reranker = RerankerService(config.RERANKER_MODEL)
        self.query_stage = QueryUnderstandingStage(content_enhancer)
        self.retrieval_stage = RetrievalStage(search_engine)
        self.ranking_stage = RankingStage(content_enhancer, reranker)
        self.synthesis_stage = SynthesisStage(llm_client)
        self.path_to_placeholder_map = {}
        logger.debug("RAG引擎初始化成功。")
    
    def ask(self, question: str, image_path: str = None) -> RAGReport:
        """执行完整的RAG流程。"""
        try:
            query_analysis = self.query_stage.run(question, image_path)
            initial_results = self.retrieval_stage.run(query_analysis)
            ranked_results, descriptions = self.ranking_stage.run(query_analysis, initial_results)
            
            report_body, path_map = self.synthesis_stage.run(question, ranked_results, descriptions, query_analysis)
            return RAGReport(
                question=question,
                query_image_path=image_path,
                answer=report_body,
                evidence=ranked_results,
                image_references=path_map
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