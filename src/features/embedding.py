# src/features/embedding.py
"""
向量生成器，封装模型推理
"""
import torch
from sentence_transformers import SentenceTransformer
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
        logger.info(f"EmbeddingGenerator 将使用设备: {self.device}")
        
        try:
            self.model = SentenceTransformer(
                config.MODEL_NAME, 
                device=self.device,
                trust_remote_code=True,
                model_kwargs={'torch_dtype': torch.float16}
            )
            logger.info(f"成功加载模型: {config.MODEL_NAME}")
        except Exception as e:
            logger.error(f"加载模型失败: {config.MODEL_NAME}. 错误: {e}")
            raise

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本的向量"""
        try:
            embeddings = self.model.encode(
                texts, 
                normalize_embeddings=True,
                task="retrieval",
                show_progress_bar=True,
                batch_size=config.PROCESSING_BATCH_SIZE
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"生成文本向量批处理失败: {e}")
            return None

    def get_image_embeddings(self, images: List[Union[Image.Image, str]]) -> List[List[float]]:
        """为一批PIL图像或图像路径生成向量"""
        try:
            embeddings = self.model.encode(
                images, 
                normalize_embeddings=True,
                task="retrieval",
                show_progress_bar=True,
                batch_size=config.PROCESSING_BATCH_SIZE
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"生成图像向量批处理失败: {e}", exc_info=True)
            return None