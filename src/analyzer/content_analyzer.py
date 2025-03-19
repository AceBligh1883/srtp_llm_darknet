# -*- coding: utf-8 -*-
"""
暗网智能分析引擎
"""

from typing import Dict, Any, Optional
from src.logger import logger
from src.processor.text_preprocessor import TextPreprocessor
from src.utils.text_splitter import TextSplitter
from src.utils.vector_utils import VectorUtils
from transformers import BertTokenizer, BertModel
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

class DarknetAnalyzer:
    def __init__(self, max_text_length: int = 1024) -> None:
        # 初始化基础组件
        self.preprocessor = TextPreprocessor(max_text_length)
        self.text_chunker = TextSplitter()
        self.aligned_dim = 512
        self.vector_utils = VectorUtils()
        # 初始化模型
        self._init_models()
        
    def _init_models(self):
        """初始化所有需要的模型"""
        # 文本模型初始化
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
        # 图像模型初始化
        self.image_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.image_model.eval()
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def process_embedding(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """统一的向量处理流程"""
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
            
        # 使用 VectorUtils 进行向量对齐和归一化
        aligned = self.vector_utils.align_vectors([embedding], self.aligned_dim)[0]
            
        return {
            "original": embedding,
            "aligned": aligned
        }

    def get_text_embedding(self, text: str) -> Dict[str, torch.Tensor]:
        """获取文本向量"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        return self.process_embedding(embedding)

    def get_image_embedding(self, image_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """获取图像向量"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0)
            
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
                self.image_model = self.image_model.cuda()
                
            with torch.no_grad():
                embedding = self.image_model(image_tensor)
                
            return self.process_embedding(embedding.cpu().squeeze())
            
        except Exception as e:
            logger.error(f"图像处理失败: {str(e)}")
            return None

    def analyze(self, text: str = None, image_path: str = None) -> Dict[str, Any]:
        """统一的分析接口"""
        result = {}
        
        if text:
            processed_text = self.preprocessor.preprocess(text)
            embeddings = self.get_text_embedding(processed_text)
            result["text_embedding"] = {
                "original": embeddings["original"].tolist(),
                "aligned": embeddings["aligned"].tolist()
            }
        
        if image_path:
            embeddings = self.get_image_embedding(image_path)
            if embeddings:
                result["image_embedding"] = {
                    "original": embeddings["original"].tolist(),
                    "aligned": embeddings["aligned"].tolist()
                }
                
        return result