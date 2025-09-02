# src/pipeline/rag_pipeline.py
from typing import List, Set, Dict, Tuple
from src.common import prompts
from src.common.logger import logger
from src.common.data_models import SearchResult
from src.services.content_enhancer import ContentEnhancer
from src.search.engine import SearchEngine
from src.services.reranker_service import RerankerService

class QueryUnderstandingStage:
    """理解用户查询"""
    def __init__(self, content_enhancer: ContentEnhancer):
        self.enhancer = content_enhancer

    def run(self, question: str, image_path: str = None) -> Dict:
        queries = {question} if question else set()
        analysis_result = {"main_image_desc": "", "final_llm_images": []}
        master_query_text = question

        if image_path:
            logger.info(f"正在分析查询图片 '{image_path}'")
            try:
                analysis = self.enhancer.analyze_query_image(image_path)
                queries.update(analysis["keywords"])
                queries.add(analysis["description"])
                analysis_result["main_image_desc"] = analysis["description"]
                analysis_result["final_llm_images"].append(analysis["pil_image"])
                analysis_result["main_query_image_path"] = image_path

                if not master_query_text:
                    master_query_text = analysis["description"]
            except ValueError as e:
                raise RuntimeError(f"分析主查询图片失败: {e}")
        
        elif question:
            logger.info(f"正在重写文本查询 '{question[:50]}'")
            queries.update(self.enhancer.rewrite_query(question))
        
        return {"queries": queries, "master_query_text": master_query_text, **analysis_result}

class RetrievalStage:
    """从索引中召回候选文档"""
    def __init__(self, search_engine: SearchEngine):
        self.engine = search_engine

    def run(self, queries: Set[str]) -> List[SearchResult]:
        logger.info(f"正在从 {len(queries)} 个查询中检索文档...")
        all_results = [res for query in queries for res in self.engine.search_by_text(query)]
        unique_results = list({res.doc_id: res for res in all_results}.values())
        if not unique_results:
            raise RuntimeError("数据库中找不到与您查询相关的任何文档。")
        return unique_results

class RankingStage:
    """对召回的文档进行重排序"""
    def __init__(self, content_enhancer: ContentEnhancer, reranker: RerankerService):
        self.enhancer = content_enhancer
        self.reranker = reranker

    def run(self, queries: str, initial_results: List[SearchResult]) -> Tuple[List[SearchResult], Dict[str, str]]:
        logger.info("正在为重排序准备内容...")
        image_paths = [r.metadata.file_path for r in initial_results if r.metadata.content_type == 'image']
        descriptions = self.enhancer.describe_images_batch(image_paths)

        def get_content(result: SearchResult) -> str:
            if result.metadata.content_type == 'image':
                return descriptions.get(result.metadata.file_path, "")
            try:
                with open(result.metadata.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(8192)
            except Exception:
                return ""
        
        rerank_data = ((res, get_content(res)) for res in initial_results)
        ranked_results = self.reranker.rerank(queries, rerank_data)
        return ranked_results, descriptions

class SynthesisStage:
    """构建上下文并生成最终报告"""
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def run(self, question: str, evidence: List[SearchResult], descriptions: Dict[str, str], query_analysis: Dict) -> Tuple[str, Dict]:
        logger.info(f"正在使用 {len(evidence)} 份证据构建最终上下文...")
        path_map = {}
        contexts = []
        image_counter = 1

        if "main_query_image_path" in query_analysis:
            path_map["main_query_image"] = query_analysis["main_query_image_path"]

        for result in evidence:
            path = result.metadata.file_path
            if result.metadata.content_type == 'text':
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    contexts.append(f"Source: {path}\nContent:\n{f.read()}")
            elif result.metadata.content_type == 'image':
                placeholder = f"image_{image_counter}"
                image_counter += 1
                path_map[placeholder] = path
                contexts.append(f"Source: 图片 (标识符: {placeholder})\nContent:\n{descriptions.get(path, '[描述不可用]')}")
        
        if not contexts:
            raise RuntimeError("找到相关文档，但无法构建上下文生成答案。")

        prompt_template = prompts.MULTIMODAL_RAG_PROMPT if query_analysis["final_llm_images"] else prompts.RAG_PROMPT
        prompt = prompt_template.format(
            context="\n\n---\n\n".join(contexts),
            question=question or "基于提供的图片和资料进行综合分析。",
            main_image_description=query_analysis["main_image_desc"]
        )
        
        logger.info("正在生成最终报告...")
        report_body = self.llm_client.generate(prompt, pil_image=query_analysis["final_llm_images"])
        
        return report_body, path_map
