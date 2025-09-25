import json
import requests
from typing import List, Union

from src.common import config
from src.common.logger import logger
from src.clients.llm_client import LLMClient

class DeepSeekClient(LLMClient):
    """
    用于调用 DeepSeek 大语言模型的客户端。
    该客户端遵循 OpenAI API 格式，并集成了强大的重试机制。
    """
    def __init__(self):
        """
        初始化 DeepSeek 客户端。
        从配置文件中加载 API 的 URL、密钥和要使用的模型名称。
        """
        self.api_url = config.DEEPSEEK_API_URL
        self.api_key = config.DEEPSEEK_API_KEY
        self.model = config.DEEPSEEK_MODEL

        if not self.api_url or not self.api_key:
            raise ValueError("DEEPSEEK_API_URL and DEEPSEEK_API_KEY must be set in the config file.")

    def generate(self, prompt: str, pil_image: Union[object, List[object]] = None) -> str:
        """
        向 DeepSeek API 发送一个 prompt，并获取模型生成的文本内容。
        
        Args:
            prompt (str): 发送给语言模型的用户输入提示。
            pil_image: 此参数将被忽略，以保持与基类的兼容性。

        Returns:
            str: 模型生成的回答。如果发生错误或响应无效，则返回空字符串。
        """
        if pil_image:
            logger.warning("DeepSeekClient 不支持图像输入，图像参数将被忽略。")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # DeepSeek 使用与 OpenAI 兼容的消息格式
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1, # 对于知识抽取任务，更低的温度更稳定
            "stream": False
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=(10, 300)  # 连接超时10秒，读取超时300秒(5分钟)
            )
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                answer = response_data["choices"][0]["message"]["content"]
                logger.info(f"已成功从 DeepSeek API ({self.model}) 接收到响应。")
                return answer.strip()
            else:
                logger.error(f"从 DeepSeek API 收到的响应结构无效: {response_data}")
                return ""

        except requests.exceptions.RequestException as e:
            logger.warning(f"调用 DeepSeek API 时发生网络错误 (将进行重试): {e}")
            raise e # 重新抛出异常，以便 tenacity 进行重试
        except Exception as e:
            logger.error(f"处理 DeepSeek API 响应时发生未知错误: {type(e).__name__} - {e}", exc_info=True)
            return ""

