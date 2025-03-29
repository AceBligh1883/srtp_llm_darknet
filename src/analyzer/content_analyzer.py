# -*- coding: utf-8 -*-
"""
暗网智能分析引擎
"""

from typing import Dict, Any, Optional
from src.logger import logger
from src.processor.text_preprocessor import TextPreprocessor
from src.processor.image_preprocessor import ImagePreprocessor
from src.utils.text_splitter import TextSplitter
from transformers import CLIPProcessor, CLIPModel
import torch

class DarknetAnalyzer:
    def __init__(self, max_text_length: int = 1024) -> None:
        # 初始化基础组件
        self.preprocessor = TextPreprocessor(max_text_length)
        self.text_chunker = TextSplitter()
        self.image_preprocessor = ImagePreprocessor()
        self._init_models()
        
    def _init_models(self):
        """初始化所有需要的模型"""
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.model.eval()

    def get_text_embedding(self, text: str) -> Dict[str, torch.Tensor]:
        """获取文本向量"""
        try:
            # 文本预处理
            processed_text = self.preprocessor.preprocess(text)
            if not processed_text:
                return None
                
            # 生成向量
            inputs = self.processor(text=processed_text, return_tensors="pt", 
                                padding=True, truncation=True)
            with torch.no_grad():
                embedding = self.model.get_text_features(**inputs)
                
            return {"embedding": embedding}
        
        except Exception as e:
            logger.error(f"文本处理失败: {str(e)}")
            return None

    def get_image_embedding(self, image_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """获取图像向量"""
        try:
            # 图像预处理
            image = self.image_preprocessor.preprocess(image_path)
            if image is None:
                return None
                
            # 生成向量
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
                
            return {"embedding": embedding}
            
        except Exception as e:
            logger.error(f"图像处理失败: {str(e)}")
            return None

    def analyze(self, text_path: str = None, image_path: str = None) -> Dict[str, Any]:
        """统一的分析接口"""
        result = {}
        
        if text_path:
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            embeddings = self.get_text_embedding(text_content)

        elif image_path:
            embeddings = self.get_image_embedding(image_path)
        
        if embeddings and "embedding" in embeddings:
            result["embedding"] = embeddings["embedding"].tolist()[0]
                
        return result
