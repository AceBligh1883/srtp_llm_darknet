import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Optional
import torch
from torchvision import transforms
from src.logger import logger

class ImagePreprocessor:
    def __init__(self):
        self.target_size = (224, 224) 
        self.quality_threshold = 0.5
        
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ])

    def enhance_image(self, image: Image.Image) -> Image.Image:
        """基础图像增强"""
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        return image

    def denoise(self, image: Image.Image) -> Image.Image:
        """图像去噪"""
        img_array = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(img_array)
        return Image.fromarray(denoised)

    def standardize(self, image: Image.Image) -> Image.Image:
        """尺寸和格式标准化"""
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image

    def assess_quality(self, image: Image.Image) -> float:
        """评估图像质量"""
        gray = image.convert('L')
        array = np.array(gray)
        laplacian_var = cv2.Laplacian(array, cv2.CV_64F).var()
        quality_score = min(laplacian_var / 500.0, 1.0)
        return quality_score

    def preprocess(self, image_path: str) -> Optional[Image.Image]:
        """完整的图像预处理流程"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            quality_score = self.assess_quality(image)
            if quality_score < self.quality_threshold:
                logger.warning(f"图像质量较低 (分数: {quality_score:.2f}): {image_path}")
            
            # 应用预处理步骤
            image = self.denoise(image)
            image = self.enhance_image(image)
            image = self.standardize(image)
            
            return image
            
        except Exception as e:
            logger.error(f"图像预处理失败: {str(e)}")
            return None