# src/services/embedding_service.py
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from typing import List, Union
from src.common import config
from src.common.logger import logger

class EmbeddingService:
    """负责生成文本和图像的向量嵌入。"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"EmbeddingGenerator 将使用设备: {self.device}")
        model_id = config.MODEL_NAME

        model_kwargs = {'torch_dtype': torch.float16, 'attn_implementation': "flash_attention_2"} if self.device == "cuda" else {}
        try:
            self.model = AutoModel.from_pretrained(model_id, **model_kwargs).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            logger.debug(f"成功加载模型: {config.MODEL_NAME}")
        except Exception as e:
            logger.error(f"加载模型失败: {config.MODEL_NAME}. 错误: {e}")
            raise

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        为一批文本生成向量嵌入。

        Args:
            texts (List[str]): 待处理的原始文本字符串列表。

        Returns:
            List[List[float]]: 每个文本对应的向量嵌入列表。如果失败则返回None。
        """
        try:
            prompted_texts = [f"This is a photo of {text}" for text in texts]
            inputs = self.processor(
                text=prompted_texts, 
                padding="max_length", 
                return_tensors="pt", 
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            return features.cpu().tolist()
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
                features  = self.model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            return features.cpu().tolist()
        except Exception as e:
            logger.error(f"生成图像向量批处理失败: {e}", exc_info=True)
            return None