# src/clients/qwen_client.py
"""
离线 Qwen 模型客户端
"""

import subprocess
from typing import Optional
from src.common.logger import logger
from src.common import config

class QwenClient:
    """
    离线 Qwen API 客户端
    通过调用命令行工具 ollama 调用本地模型，
    将输入的 prompt 传入模型，并返回模型生成的文本结果。
    """
    def __init__(self) -> None:
        self.default_model = config.DEFAULT_MODEL

    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        """
        调用本地命令行工具 ollama 生成响应。
        
        :param prompt: 模型调用的提示信息
        :param model: 使用的模型名称，默认为配置中指定的模型
        :return: 模型生成的文本结果；若出错则返回 "ERROR"
        """
        if model is None:
            model = self.default_model
            
        try:
            logger.debug(f"调用模型: {model}")
            result = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            output_text = result.stdout.decode("utf-8").strip()
            return output_text
        except subprocess.CalledProcessError as e:
            logger.error(f"命令调用错误: {e.stderr.decode('utf-8')}")
            return "ERROR"
