# src/clients/qwen_client.py
from typing import Optional
import requests
from src.common import config
from src.common.logger import logger

class QwenClientAPI:
    def __init__(self) -> None:
        self.api_url = "http://localhost:11434/api/generate"
        self.default_model = config.DEFAULT_MODEL
        self.system_message = (
            "你是一个高度精炼的助手。绝不要输出解释或多余文本。"
        )
    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        if model is None:
            model = self.default_model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "system": self.system_message,
            "temperature": 0.3,
            "top_p": 0.85,
            "stream": False
        }
        
        try:
            logger.debug(f"通过API调用模型: {model}")
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            if "message" in response_data and "content" in response_data["message"]:
                answer = response_data["message"]["content"]
                logger.info("已成功从Qwen API接收到响应。")
                return answer.strip()
            else:
                logger.error(f"从Qwen API收到的响应结构无效: {response_data}")
                return "ERROR: 从API收到的响应格式不正确。"
            return response.json().get("response", "").strip()
            
        except requests.RequestException as e:
            logger.error(f"API调用错误: {e}")
            return "ERROR"
