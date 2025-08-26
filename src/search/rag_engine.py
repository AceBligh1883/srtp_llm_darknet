# src/search/rag_engine.py
"""
检索增强生成 (RAG) 引擎
"""
import json
import os
import ast
import re
import time
from typing import Iterable, List
from src.common import config
from src.common.logger import logger
from src.common.data_models import SearchResult
from src.search.engine import SearchEngine
from src.clients.gemini_client import GeminiClient
from src.features.image_processing import ImageProcessor

class RAGEngine:
    """
    负责协调检索和生成，以回答用户问题的引擎。
    """
    PROMPT_TEMPLATE = """
        你是一名专业的暗网情报分析师。你的任务是严格根据下面提供的“背景资料”来回答用户的问题。
        请遵循以下规则：
        1.  **只使用提供的背景资料**，绝对不要依赖任何你自己的知识或外部信息。
        2.  如果背景资料中没有足够的信息来回答问题，请明确指出：“根据提供的资料，无法回答该问题。”
        3.  你的回答应该简洁、准确，并直接针对用户的问题。
        4.  如果可能，可以在回答中引用资料来源（例如，根据文件 'path/to/file.txt' 的内容...）。

        --- 背景资料 ---
        {context}
        --- 背景资料结束 ---

        用户问题: {question}

        你的回答:
        """
    
    REWRITE_PROMPT_TEMPLATE = """
        你的任务是将复杂的用户问题分解成一个或多个简洁的、多语言的、适合向量数据库检索的关键词查询。以 Python 列表的格式返回。

        --- 示例 ---
        用户问题: '告诉我关于AR-15的规格，以及它和AK-47的比较'
        你的回答:
        ["AR-15 规格", "AR-15 与 AK-47 比较", "AR-15 Specs", "AR-15 vs. AK-47 Comparison"]
        --- 示例结束 ---

        CRITICAL: 你的整个回答必须只有 Python 列表，不能有任何周围的文字、解释或Markdown标记。

        用户问题: '{question}'

        分解后的查询列表:
        """
    
    IMAGE_ANALYSIS_PROMPT_TEMPLATE = """
        你是一个图像分析专家。请仔细观察这张图片，并以JSON格式提供以下信息：
        1.  一个详细的、客观的描述 (description)。
        2.  一个包含5到10个最关键的核心关键词的列表 (keywords)，这些关键词应该适合用于数据库检索。

        CRITICAL: 你的回答必须是一个严格的、不含任何额外文本的JSON对象。

        示例输出:
        {
        "description": "一张特写照片，展示了一把黑色的半自动手枪，型号类似格洛克19，放置在深色的木质桌面上。背景有些模糊。",
        "keywords": ["handgun", "pistol", "Glock 19", "black firearm", "gun on table", "手枪", "黑色武器"]
        }
        """

    def __init__(self):
        logger.debug("正在初始化RAG引擎...")
        self.search_engine = SearchEngine()
        self.llm_client = GeminiClient()
        self.image_processor = ImageProcessor() 
        logger.debug("RAG引擎初始化成功。")

    def _rewrite_query(self, question: str) -> List[str]:
        prompt = self.REWRITE_PROMPT_TEMPLATE.format(question=question)
        rewritten_queries_str = self.llm_client.generate(prompt)
        logger.debug(f"LLM为查询重写返回的原始字符串: '{rewritten_queries_str}'")
        
        try:
            match = re.search(r'\[(.*?)\]', rewritten_queries_str, re.DOTALL)
            if not match:
                raise ValueError("在LLM的响应中未找到格式为 [...] 的列表。")
            list_str = f"[{match.group(1)}]"
            
            queries = ast.literal_eval(list_str)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries
            else:
                raise ValueError("LLM返回的不是一个字符串列表")
        except (ValueError, SyntaxError, TypeError) as e:
            logger.warning(f"解析重写查询失败: {e}。将使用原始问题进行检索。")
            return [question]

    def _build_context_from_results(self, search_results: Iterable[SearchResult]) -> List[str]:
        """从搜索结果中读取文件内容，构建上下文列表。"""
        contexts = []
        source_files = []
        results_list = list(search_results)

        for result in results_list:
            meta = result.metadata
            if meta and meta.file_path:
                source_files.append(meta.file_path)
                if meta.content_type == 'text' and os.path.exists(meta.file_path):
                    try:
                        with open(meta.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            contexts.append(f"Source: {meta.file_path}\n\nContent:\n{content}")
                    except Exception as e:
                        logger.error(f"读取上下文文件失败 {meta.file_path}: {e}")
                elif meta.content_type == 'image':
                     contexts.append(f"相关图片路径: {meta.file_path}")
        
        return contexts, list(set(source_files))
     
    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        context_str = "\n\n---\n\n".join(contexts)
        if not context_str:
            context_str = "未找到任何相关资料。"
        return self.PROMPT_TEMPLATE.format(context=context_str, question=question)

    def ask(self, question: str) -> str:
        """
        执行完整的RAG流程来回答问题。
        """
        logger.debug(f"收到RAG问题:'{question[:50]}...'")
        
        sub_queries = self._rewrite_query(question)
        queries_to_search = set(sub_queries)
        queries_to_search.add(question)
        
        logger.info(f"混合检索查询集: {list(queries_to_search)}")
        all_results = []
        for query in queries_to_search:
            results = self.search_engine.search_by_text(query, top_k=config.RAG_TOP_K)
            all_results.extend(results)
        
        unique_results_dict = {res.doc_id: res for res in all_results}
        unique_results = sorted(unique_results_dict.values(), key=lambda r: r.score, reverse=True)
        
        if not unique_results:
            logger.warning("混合检索未能找到任何相关文档。")
            return "抱歉，我在数据库中找不到与您问题相关的任何文档来生成回答。"
        
        contexts, source_files = self._build_context_from_results(unique_results)
        if source_files:
            logger.info(f"检索到 {len(source_files)} 份相关材料来源: {source_files}")
        
        if not contexts:
            logger.warning("找到了相关文档条目，但无法读取其内容。")
            return "我找到了一些相关的文档条目，但无法成功读取它们的内容来组织答案。"

        final_prompt = self._build_prompt(question, contexts)
        logger.debug(f"发送给LLM的最终Prompt:\n{final_prompt}")

        logger.info("正在生成最终回答...")
        answer = self.llm_client.generate(final_prompt)

        report = []
        report.append("### **情报分析报告**")
        report.append("---")
        report.append("#### **核心结论**")
        report.append(answer)
        report.append("\n")
        report.append("#### **关键证据来源 (Top 5)**")
        for i, res in enumerate(unique_results[:5]):
            file_name = os.path.basename(res.metadata.file_path)
            report.append(f"{i+1}. **文件**: `{file_name}` (相似度: {res.score:.2f})")
        
        if len(unique_results) > 5:
            report.append(f"\n*（以及其他 {len(unique_results) - 5} 个相关来源）*")

        return "\n".join(report)

    def ask_with_image(self, image_path: str, question: str = None) -> str:
        """
        执行一个有意义的图像RAG流程。

        Args:
            image_path (str): 用户提供的查询图像路径。
            question (str, optional): 用户针对该图像提出的具体问题。

        Returns:
            str: AI生成的、基于视觉和检索信息的回答。
        """
        logger.info(f"收到RAG图像查询: '{image_path}', 问题: '{question}'")
        query_image = self.image_processor.process(image_path)
        if not query_image:
            logger.error(f"无法打开或处理查询图像: {image_path}")
            return "错误：无法打开或处理您提供的图像文件。"
        
        analysis_response_str = self.llm_client.generate(
            self.IMAGE_ANALYSIS_PROMPT_TEMPLATE, 
            pil_image=query_image
        )
        image_description = ""
        keywords = []

        try:
            match = re.search(r'```json\s*(\{.*?\})\s*```', analysis_response_str, re.DOTALL)
            json_str = ""
            if match:
                json_str = match.group(1)
            else:
                json_str = analysis_response_str
            analysis_data = json.loads(json_str)
            image_description = analysis_data.get("description", "")
            keywords = analysis_data.get("keywords", [])
            logger.info(f"初步图像描述: {image_description[:100]}...")
            logger.info(f"提取到关键词: {keywords}")
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"解析图像分析JSON失败: {e}。将使用完整响应作为描述。")
            image_description = analysis_response_str
        search_queries = set(keywords) 
        if question:
            search_queries.add(question)
        if image_description:
            search_queries.add(image_description)
    
        all_text_results = []
        for query in search_queries:
            results = self.search_engine.search_by_text(query, top_k=5) 
            all_text_results.extend(results)

        unique_results_dict = {res.doc_id: res for res in all_text_results}
        unique_text_results = sorted(unique_results_dict.values(), key=lambda r: r.score, reverse=True)
        final_question = question if question else "请基于你对这张图片的观察，并结合以下背景资料，给出一份详细的情报分析报告。"

        contexts, _ = self._build_context_from_results(unique_text_results)
        context_str = "\n\n---\n\n".join(contexts)
        if not context_str:
            context_str = "未找到相关的文本背景资料。"
        final_prompt = f"""
            你是一名顶尖的多模态情报分析师。你的任务是结合你对给定图像的直接观察和以下提供的文本背景资料，来回答用户的问题。

            ### 你的观察 (图像描述):
            {image_description}

            ### 相关的文本背景资料:
            {context_str}

            ---
            ### 用户最终问题:
            {final_question}

            请综合以上所有信息，生成你的最终分析报告。
            """
        time.sleep(10)
        final_answer = self.llm_client.generate(final_prompt, pil_image=query_image)
        report = []
        report.append("### **多模态情报分析报告**")
        report.append("---")
        report.append("#### **核心分析**")
        report.append(final_answer)
        report.append("\n")
        report.append("#### **引用的文本证据来源 (Top 5)**")
        if unique_text_results:
            for i, res in enumerate(unique_text_results[:5]):
                file_name = os.path.basename(res.metadata.file_path)
                report.append(f"{i+1}. **文件**: `{file_name}` (相似度: {res.score:.2f})")
        else:
            report.append("无。")

        return "\n".join(report)
