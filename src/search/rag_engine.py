# src/search/rag_engine.py
"""
检索增强生成 (RAG) 引擎
"""
import os
from typing import List

from src.common import config
from src.common.logger import logger
from src.search.engine import SearchEngine
from src.clients.gemini_client  import GeminiClient

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

    def __init__(self):
        logger.info("正在初始化RAG引擎...")
        self.search_engine = SearchEngine()
        self.llm_client = GeminiClient()
        logger.info("RAG引擎初始化成功。")

    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        """
        根据模板构建最终的Prompt。
        """
        context_str = "\n\n---\n\n".join(contexts)
        if not context_str:
            context_str = "未找到任何相关资料。"
            
        return self.PROMPT_TEMPLATE.format(context=context_str, question=question)

    def ask(self, question: str) -> str:
        """
        执行完整的RAG流程来回答问题。
        """
        logger.info(f"收到RAG问题:'{question[:50]}...'")

        logger.info("检索相关文档...")
        search_results = self.search_engine.search_by_text(
            query=question, 
            top_k=config.RAG_TOP_K
        )

        if not search_results:
            logger.warning("未能根据问题找到任何相关文档。")
            return "抱歉，我在数据库中找不到与您问题相关的任何文档来生成回答。"

        contexts = []
        logger.info(f"找到 {len(search_results)} 个潜在相关文档，正在读取内容...")
        for result in search_results:
            meta = result.metadata
            if meta and meta.content_type == 'text' and meta.file_path and os.path.exists(meta.file_path):
                try:
                    with open(meta.file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        contexts.append(f"Source: {meta.file_path}\n\nContent:\n{content}")
                except Exception as e:
                    logger.error(f"读取上下文文件失败 {meta.file_path}: {e}")
        
        if not contexts:
            logger.warning("找到了相关文档条目，但无法读取其内容。")
            return "我找到了一些相关的文档条目，但无法成功读取它们的内容来组织答案。"

        logger.info("使用检索到的上下文构建Prompt...")
        final_prompt = self._build_prompt(question, contexts)
        logger.debug(f"发送给LLM的最终Prompt:\n{final_prompt}")

        logger.info("使用LLM生成回答...")
        answer = self.llm_client.generate(final_prompt)
        
        logger.info("回答生成完毕。")
        return answer

    def ask_with_image(self, image_path: str) -> str:
            """
            使用图像作为输入，执行完整的RAG流程来回答问题。
            """
            logger.info(f"收到RAG图像查询: '{image_path}'")

            logger.info("基于图像检索相关文档...")
            search_results = self.search_engine.search_by_image(
                image_path=image_path,
                top_k=config.RAG_TOP_K
            )

            if not search_results:
                logger.warning("未能根据图像找到任何相关文档。")
                return "抱歉，我在数据库中找不到与您提供的图像相关的任何文档来生成回答。"

            contexts = []
            logger.info(f"找到 {len(search_results)} 个潜在相关文档，正在读取内容...")
            for result in search_results:
                meta = result.metadata
                if meta and meta.content_type == 'text' and meta.file_path and os.path.exists(meta.file_path):
                    try:
                        with open(meta.file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            contexts.append(f"来源文件: {meta.file_path}\n\n文件内容:\n{content}")
                    except Exception as e:
                        logger.error(f"读取上下文文件失败 {meta.file_path}: {e}")
                elif meta and meta.content_type == 'image':
                    contexts.append(f"相关图片资料: {meta.file_path}")

            if not contexts:
                logger.warning("找到了相关文档条目，但无法读取其内容。")
                return "我找到了一些相关的文档条目，但无法成功读取它们的内容来组织答案。"

            logger.info("使用检索到的上下文构建Prompt...")
            final_prompt = self._build_prompt(self.DEFAULT_IMAGE_QUESTION, contexts)
            logger.info(f"发送给大语言模型的最终Prompt:\n{final_prompt}")

            logger.info("使用LLM生成回答...")
            answer = self.qwen_client.generate(final_prompt)
            
            logger.info("回答生成完毕。")
            return answer