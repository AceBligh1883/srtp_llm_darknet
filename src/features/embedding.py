# src/features/embedding.py
"""
向量生成器，封装模型推理
"""
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from typing import List, Union
from src.common import config
from src.common.logger import logger

class EmbeddingGenerator:
    """
    负责生成文本和图像的向量嵌入。
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"EmbeddingGenerator 将使用设备: {self.device}")
        model_id = config.MODEL_NAME
        try:
            self.model = AutoModel.from_pretrained(model_id).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            logger.debug(f"成功加载模型: {config.MODEL_NAME}")
        except Exception as e:
            logger.error(f"加载模型失败: {config.MODEL_NAME}. 错误: {e}")
            raise

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本的向量"""
        try:
            inputs = self.processor(text=texts, padding=True, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().tolist()
        except Exception as e:
            logger.error(f"生成文本向量批处理失败: {e}")
            return None

    def get_image_embeddings(self, images: List[Union[Image.Image, str]]) -> List[List[float]]:
        """为一批PIL图像或图像路径生成向量"""
        if not images:
            return []
        try:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().tolist()
        except Exception as e:
            logger.error(f"生成图像向量批处理失败: {e}", exc_info=True)
            return None