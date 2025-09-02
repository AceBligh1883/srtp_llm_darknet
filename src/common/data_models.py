# src/common/data_models.py
"""
项目中统一使用的数据模型
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ContentMeta:
    """
    内容元数据模型，代表存储在SQLite数据库中的一条记录。
    """
    id: Optional[int] = None
    url: str = "local"
    content_type: str = "unknown"
    file_path: str = ""
    file_hash: str = ""
    timestamp: str = ""
    kg_processed: int = 0
    
@dataclass
class SearchResult:
    """统一的搜索结果模型"""
    doc_id: int
    score: float
    metadata: ContentMeta

@dataclass
class RAGReport:
    """
    RAG流程的数据容器。
    """
    question: Optional[str]
    query_image_path: Optional[str]
    body: str  
    evidence: List[SearchResult] 
    image_references: Dict[str, str]