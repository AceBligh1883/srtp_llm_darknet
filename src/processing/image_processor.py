# src/processing/image_processor.py
from PIL import Image
from typing import Optional
from src.common.logger import logger

class ImageProcessor:
    """为模型准备图像"""
    def process(self, image_path: str) -> Optional[Image.Image]:
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            logger.error(f"打开或转换图像失败: {image_path} - {e}")
            return None
