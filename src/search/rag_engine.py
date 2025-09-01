# src/search/rag_engine.py
"""
检索增强生成 (RAG) 引擎
"""
import os
from typing import Iterable, List
from src.common import config, prompts
from src.common.logger import logger
from src.common.data_models import SearchResult
from src.search.engine import SearchEngine
from src.search.reranker import Reranker
from src.clients.gemini_client import GeminiClient
from src.features.image_processing import ImageProcessor
from src.features.content_processor import analyze_query_image, rewrite_query, describe_images_batch

class RAGEngine:
    """
    负责协调检索和生成，以回答用户问题的引擎。
    """
    def __init__(self):
        logger.debug("正在初始化RAG引擎...")
        self.search_engine = SearchEngine()
        self.llm_client = GeminiClient()
        self.image_processor = ImageProcessor()
        self.reranker = Reranker(config.RERANKER_MODEL)
        self.path_to_placeholder_map = {} 
        logger.debug("RAG引擎初始化成功。")

    def _retrieve(self, queries: set) -> List[SearchResult]:
        """执行检索和去重"""
        logger.info(f"混合检索查询集: {list(queries)}")
        all_results = [res for query in queries for res in self.search_engine.search_by_text(query, top_k=config.RAG_TOP_K)]
        return list({res.doc_id: res for res in all_results}.values())
    
    def _build_context(self, search_results: Iterable[SearchResult], descriptions: dict = None) -> List[str]:
        """构建最终提供给LLM的上下文"""
        self.path_to_placeholder_map.clear()
        contexts = []
        image_counter = 1
        descriptions = descriptions or {}
        for result in search_results:
            path = result.metadata.file_path
            if result.metadata.content_type == 'text' and os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        contexts.append(f"Source: {path}\nContent:\n{f.read()}")
                except Exception as e:
                    logger.error(f"读取上下文文件失败 {path}: {e}")
            elif result.metadata.content_type == 'image':
                placeholder_id = f"image_{image_counter}"
                image_counter += 1
                self.path_to_placeholder_map[placeholder_id] = path
                contexts.append(f"Source: 图片 (标识符: {placeholder_id})\nContent:\n图像内容描述: {descriptions.get(path, '[描述不可用]')}")
        return contexts
    
    def _create_appendix(self, final_results: List[SearchResult]) -> str:
        """生成报告的附录"""
        text_lines = [f"{i+1}. **文件**: `{r.metadata.file_path}` (相关性: {r.score:.4f})" for i, r in enumerate(r for r in final_results if r.metadata.content_type == 'text')]
        image_lines = [f"{i+1}. **文件**: `{r.metadata.file_path}` (相关性: {r.score:.4f})" for i, r in enumerate(r for r in final_results if r.metadata.content_type == 'image')]
        return "\n".join([
            "\n\n---\n### **证据与资料来源附录**",
            "\n#### **引用的文本证据**",
            "\n".join(text_lines) or "未找到相关的文本文档。",
            "\n\n#### **参考的视觉资料**",
            "\n".join(image_lines) or "未找到相关的视觉资料。"
        ])
    
    def ask(self, question: str, image_path: str = None) -> str:
        """执行完整的RAG流程。"""
        queries = {question} if question else set()
        final_prompt_template = prompts.RAG_PROMPT
        final_llm_images = []
        main_image_desc = ""

        if image_path:
            logger.info(f"检测到图像查询，路径: '{image_path}'")
            try:
                analysis = analyze_query_image(self.llm_client, self.image_processor, image_path)
                queries.update(analysis["keywords"])
                queries.add(analysis["description"])
                main_image_desc = analysis["description"]
                final_llm_images.append(analysis["pil_image"])
                self.path_to_placeholder_map["main_query_image"] = image_path
                final_prompt_template = prompts.MULTIMODAL_RAG_PROMPT
            except ValueError as e:
                return str(e)
        elif question:
            logger.info(f"收到文本查询: '{question[:50]}'")
            queries.update(rewrite_query(self.llm_client, question))
        else:
            return "错误：必须提供问题或图片路径。"
        
        results = self._retrieve(queries)
        if not results:
            return "抱歉，数据库中找不到与您查询相关的任何文档。"
        
        processed_results = results
        context_descriptions = {}

        if not image_path:
            image_paths = [r.metadata.file_path for r in results if r.metadata.content_type == 'image']
            descriptions = describe_images_batch(self.llm_client, self.image_processor, image_paths)
            context_descriptions = descriptions

            def get_content(result):
                if result.metadata.content_type == 'image': return descriptions.get(result.metadata.file_path, "")
                try:
                    with open(result.metadata.file_path, 'r', encoding='utf-8', errors='ignore') as f: return f.read(1024)
                except Exception: return ""
                
            processed_results = self.reranker.rerank(question, ((res, get_content(res)) for res in results))
        
        contexts = getattr(config, 'RAG_CONTEXT_TOP_K', 10)
        evidence = processed_results[:contexts]
        contexts = self._build_context(evidence, context_descriptions)
        
        if not contexts:
            return "找到相关文档，但无法构建上下文生成答案。"
            
        logger.info(f"上下文构建完成，使用 {len(contexts)} 份材料。正在生成最终报告...")

        prompt = final_prompt_template.format(
            context="\n\n---\n\n".join(contexts), 
            question=question or "基于提供的图片和资料进行综合分析。",
            main_image_description=main_image_desc
        )
        report_body = self.llm_client.generate(prompt, pil_image=final_llm_images)
        report_appendix = self._create_appendix(evidence)
        
        return report_body + report_appendix
    