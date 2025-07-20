# src/features/embedding.py
"""
向量生成器，封装模型推理
"""
from typing import Optional
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from src.common import config
from src.common.logger import logger

class EmbeddingGenerator:
    """
    输入文本或图像，输出Numpy向量。
    这是一个无状态的计算模块。
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"EmbeddingGenerator 使用设备: {self.device}")
        try:
            self.model = CLIPModel.from_pretrained(config.MODEL_NAME).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(config.MODEL_NAME)
            self.model.eval()
        except Exception as e:
            logger.error(f"加载CLIP模型失败: {e}")
            raise

    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取文本的向量"""
        if not text:
            return None
        try:
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            with torch.no_grad():
                embedding = self.model.get_text_features(**inputs)
            # 归一化向量并转为numpy
            embedding /= embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().squeeze()
        except Exception as e:
            logger.error(f"生成文本向量失败: {e}")
            return None

    def get_image_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """获取图像的向量"""
        if not image:
            return None
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
            # 归一化向量并转为numpy
            embedding /= embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().squeeze()
        except Exception as e:
            logger.error(f"生成图像向量失败: {e}")
            return None
