# src/clients/gemini_client.py

import requests
import json
from src.common import config
from src.common.logger import logger

class GeminiClient:
    """
    通过兼容OpenAI格式的代理API调用Gemini模型的客户端。
    """
    def __init__(self):
        # 从配置文件中读取API URL和Key
        self.api_url = config.GEMINI_API_URL
        self.api_key = config.GEMINI_API_KEY
        self.model = config.DEFAULT_MODEL # 模型名称也从配置读取

        if not self.api_url or not self.api_key:
            raise ValueError("GEMINI_API_URL and GEMINI_API_KEY must be set in the config file.")

    def generate(self, prompt: str) -> str:
        """
        向代理API发送请求并获取模型生成的内容。
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构建符合OpenAI格式的请求体
        # 这里的 prompt 是我们从 RAGEngine 传过来的完整prompt
        payload = {
            "model": self.model,
            "messages": [
                # 我们可以简化，直接把整个prompt作为用户输入
                # 或者遵循system + user的角色划分，效果可能更好
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7, # 可以调整创新度
            "stream": False # 我们需要一次性返回结果，所以设置为False
        }

        logger.info(f"正在向Gemini代理发送请求，使用模型: {self.model}")

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=180  # 设置一个较长的超时时间，比如3分钟
            )

            # 检查HTTP状态码
            response.raise_for_status()

            response_data = response.json()
            
            # 解析返回结果，提取模型回答
            # 同样，这也是标准的OpenAI返回格式
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

