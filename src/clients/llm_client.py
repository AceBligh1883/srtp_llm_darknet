# src/clients/llm_client.py
from abc import ABC, abstractmethod
from typing import List, Union
from PIL.Image import Image

class LLMClient(ABC):
    """大语言模型客户端的抽象基类。"""
    
    @abstractmethod
    def generate(self, prompt: str, pil_image: Union[Image, List[Image]] = None) -> str:
        """
        生成内容的核心方法。

        Args:
            prompt (str): 文本提示。
            pil_image (Union[Image, List[Image]], optional): 单个或多个PIL图像。

        Returns:
            str: 模型生成的文本响应。
        """
        pass
