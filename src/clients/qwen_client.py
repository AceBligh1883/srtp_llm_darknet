# src/clients/qwen_client.py
from typing import Optional
import requests
from src.common import config
from src.common.logger import logger

class QwenClientAPI:
    def __init__(self) -> None:
        self.api_url = "http://localhost:11434/api/generate"
        self.default_model = config.DEFAULT_MODEL

    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        if model is None:
            model = self.default_model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False 
        }
        
        try:
            logger.debug(f"通过API调用模型: {model}")
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
            
        except requests.RequestException as e:
            logger.error(f"API调用错误: {e}")
            return "ERROR"
