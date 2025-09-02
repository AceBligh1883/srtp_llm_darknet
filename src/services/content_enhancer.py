# src/services/content_enhancer.py
import re
import json
import ast
from typing import List, Dict
from src.common import prompts
from src.common.logger import logger
from src.clients.llm_client import LLMClient
from src.processing.image_processor import ImageProcessor

class ContentEnhancer:
    """使用LLM提供内容增强服务，如查询重写、图像描述等。"""
    def __init__(self, llm_client: LLMClient, image_processor: ImageProcessor):
        self.llm_client = llm_client
        self.image_processor = image_processor

    def rewrite_query(self, question: str) -> List[str]:
        """使用LLM重写和扩展用户查询。"""
        prompt = prompts.QUERY_REWRITER_PROMPT.format(question=question)
        response = self.llm_client.generate(prompt)
        try:
            match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if match:
                queries = ast.literal_eval(f"[{match.group(1)}]")
                if isinstance(queries, list):
                    return queries
            raise ValueError("未能从LLM响应中解析出列表")
        except (ValueError, SyntaxError, TypeError) as e:
            logger.warning(f"解析重写查询失败: {e}。返回原始问题。")
            return [question]

    def describe_images_batch(self, image_paths: List[str]) -> Dict[str, str]:
        """批量描述一组图像，返回 路径->描述 的字典。"""
        if not image_paths: return {}
        logger.info(f"正在批量描述 {len(image_paths)} 张图片...")
        try:
            path_map = {f"image_{i+1}": path for i, path in enumerate(image_paths)}
            valid_images = [img for img in (self.image_processor.process(p) for p in image_paths) if img]
            
            if not valid_images:
                return {path: "[图像无法处理]" for path in image_paths}

            response = self.llm_client.generate(prompts.BATCH_IMAGE_DESCRIPTION_PROMPT, pil_image=valid_images)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                raise ValueError("未能在LLM响应中找到有效的JSON对象。")
            
            json_data = json.loads(match.group(0))
            path_descriptions = {path_map.get(img_id): desc for img_id, desc in json_data.items() if path_map.get(img_id)}
            return {path: path_descriptions.get(path, "[描述在LLM响应中丢失]") for path in image_paths}
        
        except Exception as e:
            return {path: "[批量分析图像时出错]" for path in image_paths}

    def analyze_query_image(self, image_path: str) -> dict:
        """
        分析单个查询图片，返回其描述和关键词。
        返回一个字典: {"description": str, "keywords": List[str], "pil_image": Image}
        """
        logger.info(f"正在分析主查询图片: '{image_path}'")
        pil_image = self.image_processor.process(image_path)
        if not pil_image:
            raise ValueError(f"无法打开或处理查询图像: {image_path}")

        prompt = prompts.SINGLE_IMAGE_ANALYSIS_PROMPT
        response = self.llm_client.generate(prompt, pil_image=pil_image)
        
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
            return {
                "description": data.get("description", response),
                "keywords": data.get("keywords", []),
                "pil_image": pil_image
            }
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"解析主图片分析JSON失败。将使用完整响应作为描述。")
            return {"description": response, "keywords": [], "pil_image": pil_image}