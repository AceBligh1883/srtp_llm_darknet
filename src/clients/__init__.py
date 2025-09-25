# src/clients/__init__.py
from src.common import config
from src.clients.llm_client import LLMClient
from src.clients.gemini_client import GeminiClient
from src.clients.qwen_client import QwenClient
from src.clients.deepseek_client import DeepSeekClient
from src.common.logger import logger

def get_llm_client(model_name: str = None) -> LLMClient:
    """
    根据配置或指定名称获取LLM客户端实例。
    """
    model_to_use = model_name or config.DEFAULT_MODEL
    
    if "gemini" in model_to_use.lower():
        return GeminiClient()
    elif "qwen" in model_to_use.lower():
        return QwenClient()
    elif "deepseek" in model_to_use.lower():
        return DeepSeekClient()
    else:
        logger.warning(f"未找到模型 '{model_to_use}' 的特定客户端，默认使用Gemini。")
        return GeminiClient()