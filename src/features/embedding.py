# src/features/embedding.py
"""
向量生成器，封装模型推理
"""
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
from typing import Optional
import numpy as np

from src.common import config
from src.common.logger import logger

class EmbeddingGenerator:
    """
    使用 sentence-transformers 加载多语言多模态模型，
    为文本和图像生成嵌入向量。
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"EmbeddingGenerator 使用设备: {self.device}")
        
        try:
            self.model = SentenceTransformer(config.MODEL_NAME, device=self.device)
            logger.info(f"成功加载多语言模型: {config.MODEL_NAME}")
        except Exception as e:
            logger.error(f"加载 sentence-transformers 模型失败: {e}")
            raise

    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取文本的向量"""
        if not text:
            return None
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding
        except Exception as e:
            logger.error(f"生成文本向量失败: {e}")
            return None

    def get_image_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """获取图像的向量"""
        if not image:
            return None
        try:
            embeddings = self.model.encode([image], normalize_embeddings=True)
            return embeddings[0]
        except Exception as e:
            logger.error(f"生成图像向量失败: {e}", exc_info=True)
            return None