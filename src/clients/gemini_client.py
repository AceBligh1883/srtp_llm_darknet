# src/clients/gemini_client.py

import base64
import io
import requests
import json
from src.common import config
from src.common.logger import logger
from PIL import Image

class GeminiClient:
    """
    通过兼容OpenAI格式的代理API调用Gemini模型的客户端。
    """
    def __init__(self):
        """
        初始化Gemini客户端。
        从配置文件中加载API的URL、密钥和要使用的模型名称。
        """
        self.api_url = config.GEMINI_API_URL
        self.api_key = config.GEMINI_API_KEY
        self.model = config.DEFAULT_MODEL 

        if not self.api_url or not self.api_key:
            raise ValueError("GEMINI_API_URL and GEMINI_API_KEY must be set in the config file.")

    def _image_to_base64(self, pil_image: Image.Image) -> str:
        """将PIL图像对象转换为Base64编码的字符串"""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate(self, prompt: str, pil_image: Image.Image = None) -> str:
        """
        向代理API发送一个prompt，并获取模型生成的文本内容。

        Args:
            prompt (str): 发送给语言模型的用户输入提示。

        Returns:
            str: 模型生成的回答。如果发生错误，则返回一条包含错误信息的提示性字符串。
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        messages = []
        if pil_image:
            base64_image = self._image_to_base64(pil_image)
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
            messages.append({"role": "user", "content": content})
        
        else:
            messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "stream": False 
        }

        logger.debug(f"正在向Gemini代理发送请求，使用模型: {self.model}")

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=180  
            )

            response.raise_for_status()
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                answer = response_data["choices"][0]["message"]["content"]
                logger.info("已成功从Gemini代理接收到响应。")
                return answer.strip()
            else:
                logger.error(f"从Gemini代理收到的响应结构无效: {response_data}")
                return "抱歉，从API收到的响应格式不正确。"

        except requests.exceptions.RequestException as e:
            logger.error(f"调用Gemini API时发生错误: {e}")
            return f"抱歉，调用API时发生网络错误: {e}"
        except Exception as e:
            logger.error(f"发生未知错误: {e}")
            return f"抱歉，处理API响应时发生未知错误。"

