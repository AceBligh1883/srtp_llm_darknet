# utils/vector_utils.py
from typing import List
import torch

class VectorUtils:
    def __init__(self):
        # 存储投影层，确保一致性
        self.projections = {}
    
    @staticmethod
    def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(vector, p=2, dim=-1)
    
    def align_vectors(self, vectors: List[torch.Tensor], target_dim: int) -> List[torch.Tensor]:
        aligned_vectors = []
        for i, vec in enumerate(vectors):
            # 确保向量是张量
            if not isinstance(vec, torch.Tensor):
                vec = torch.tensor(vec, dtype=torch.float32)
                
            # 使用或创建投影层
            key = f"{vec.shape[-1]}_{target_dim}_{i}"
            if key not in self.projections:
                self.projections[key] = torch.nn.Linear(vec.shape[-1], target_dim)
                
            with torch.no_grad():
                aligned = self.projections[key](vec)
                aligned = self.normalize_vector(aligned)
                aligned_vectors.append(aligned)
        return aligned_vectors
