# src/features/image_processing.py
"""
图像预处理器
"""
from PIL import Image
from typing import Optional

from src.common.logger import logger

class ImageProcessor:
    """为CLIP模型准备图像"""

    def process(self, image_path: str) -> Optional[Image.Image]:
        """
        打开图像并转换为RGB格式。
        CLIP的processor会自动处理resize和normalize。
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            logger.error(f"打开或转换图像失败: {image_path} - {e}")
            return None
