# src/pipeline/rag_pipeline.py
from typing import List, Dict, Tuple
from src.common import prompts
from src.common.logger import logger
from src.common.data_models import SearchResult
from src.services.content_enhancer import ContentEnhancer
from src.search.engine import SearchEngine
from src.processing.image_processor import ImageProcessor
from src.services.reranker_service import RerankerService
from src.services.embedding_service import EmbeddingService
from src.services.image_ranker_service import ImageRankerService

class QueryUnderstandingStage:
    """理解用户查询"""
    def __init__(self, content_enhancer: ContentEnhancer):
        self.enhancer = content_enhancer
        self.embedding_service = EmbeddingService()

    def run(self, question: str, image_path: str = None) -> Dict:
        queries = {question} if question else set()
        analysis_result = {"main_image_desc": "", "final_llm_images": [], "query_image_vector": None}
        master_query_text = question

        if image_path:
            logger.info(f"正在分析查询图片 '{image_path}'")
            try:
                analysis = self.enhancer.analyze_query_image(image_path)
                pil_image = analysis["pil_image"]

                queries.update(analysis["keywords"])
                queries.add(analysis["description"])
                analysis_result["main_image_desc"] = analysis["description"]
                analysis_result["final_llm_images"].append(analysis["pil_image"])
                analysis_result["main_query_image_path"] = image_path

                image_vectors = self.embedding_service.get_image_embeddings([pil_image])
                if image_vectors:
                    analysis_result["query_image_vector"] = image_vectors[0]

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
        
    def run(self, query_analysis: Dict) -> List[SearchResult]:
        text_queries = query_analysis["queries"]
        image_vector = query_analysis["query_image_vector"]

        all_results = []
        if text_queries:
            logger.info(f"正在执行 {len(text_queries)} 个文本查询...")
            for query in text_queries:
                all_results.extend(self.engine.search_by_text(query))
        
        if image_vector is not None:
            logger.info(f"正在执行原始图片向量查询...")
            all_results.extend(self.engine.search_by_vector(image_vector, top_k=20))
        
        if not all_results:
            raise RuntimeError("数据库中找不到与您查询相关的任何文档。")
        
        unique_results = list({res.doc_id: res for res in all_results}.values())
        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results

class RankingStage:
    """对召回的文档进行重排序"""
    def __init__(self, content_enhancer: ContentEnhancer, reranker: RerankerService):
        self.enhancer = content_enhancer
        self.text_reranker = reranker
        self.image_ranker = ImageRankerService()

    def run(self, query_analysis: dict, initial_results: List[SearchResult]) -> Tuple[List[SearchResult], Dict[str, str]]:
        master_query = query_analysis["master_query_text"]
        query_image_vector = query_analysis.get("query_image_vector")

        text_results = [r for r in initial_results if r.metadata.content_type == 'text']
        image_results = [r for r in initial_results if r.metadata.content_type == 'image']

        top_image_results = self.image_ranker.rank_images(
            master_query, query_image_vector, image_results
        )

        def get_text_content(result: SearchResult) -> str:
            try:
                with open(result.metadata.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(8192)
            except Exception:
                return ""
            
        text_rerank_data = ((res, get_text_content(res)) for res in text_results)
        top_text_results = self.text_reranker.rerank(master_query, text_rerank_data)

        image_paths_to_describe = [r.metadata.file_path for r in top_image_results]
        descriptions = self.enhancer.describe_images_batch(image_paths_to_describe)
        
        final_candidates = top_text_results + top_image_results
        final_candidates.sort(key=lambda x: x.score, reverse=True)
        return final_candidates, descriptions

class SynthesisStage:
    """构建上下文并生成最终报告"""
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.image_processor = ImageProcessor()

    def _format_kg_facts(self, kg_facts: List[Dict]) -> str:
        """将图谱事实格式化为自然语言。"""
        if not kg_facts:
            return "知识图谱中未发现与查询直接相关的结构化情报。\n"
        
        header = "--- 知识图谱情报摘要 ---\n根据结构化数据分析，发现以下关键关联：\n"
        lines = [
            f"- 实体 '{fact['head']}' ({fact['head_type']}) 与实体 '{fact['tail']}' ({fact['tail_type']}) 存在 '{fact['relation']}' 关系。"
            for fact in kg_facts
        ]
        return header + "\n".join(lines) + "\n"
    
    def run(self, question: str, evidence: List[SearchResult], descriptions: Dict[str, str], query_analysis: Dict, kg_facts: List[Dict] = None) -> Tuple[str, Dict]:
        logger.info(f"正在使用 {len(evidence)} 份证据和 {len(kg_facts or [])} 条图谱事实构建最终上下文...")
        final_context = self._format_kg_facts(kg_facts)
        final_context += "\n--- 相关非结构化情报 ---\n"
        
        path_map = {}
        contexts = []
        final_llm_images = []
        image_counter = 1
        
        has_main_query_image = "main_query_image_path" in query_analysis
        if has_main_query_image:
            path_map["main_query_image"] = query_analysis["main_query_image_path"]
            final_llm_images.extend(query_analysis["final_llm_images"])

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

                pil_image = self.image_processor.process(path)
                if pil_image:
                    final_llm_images.append(pil_image)
        
        if not contexts and not kg_facts:
            raise RuntimeError("找不到任何相关的结构化或非结构化信息。")

        final_context += "\n\n".join(contexts)

        if has_main_query_image:
            prompt_template = prompts.MULTIMODAL_RAG_PROMPT
            prompt = prompt_template.format(
                context=final_context,
                question=question or "基于提供的图文资料进行综合分析。",
                main_image_description=query_analysis.get("main_image_desc", "N/A")
            )
        else:
            prompt_template = prompts.RAG_PROMPT
            prompt = prompt_template.format(
                context=final_context,
                question=question
            )
        
        logger.info("正在生成最终报告...")
        report_body = self.llm_client.generate(prompt, pil_image=final_llm_images)
        
        return report_body, path_map
