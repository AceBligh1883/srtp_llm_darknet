import os
import hashlib
import sqlite3
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse
from dataclasses import dataclass
from src.logger import logger
from config import config

@dataclass
class ContentMeta:
    """内容元数据"""
    url: str
    domain: str
    domain_hash: str
    content_type: str
    file_path: str
    timestamp: str
    file_hash: str
    
class StorageManager:
    """统一存储管理器"""
    def __init__(self, db_path: str = "data/darknet.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS content_meta (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    domain_hash TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    UNIQUE(url, content_type)
                )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
    
    def _get_domain_hash(self, url: str) -> str:
        """获取域名哈希值"""
        domain = urlparse(url).netloc
        return hashlib.sha256(domain.encode()).hexdigest()[:16]
    
    def _get_file_hash(self, content: bytes) -> str:
        """获取文件内容哈希值"""
        return hashlib.sha256(content).hexdigest()[:16]
    
    def _generate_filepath(self, url: str, content_type: str, 
                          content: bytes, ext: str) -> str:
        """生成文件存储路径"""
        domain_hash = self._get_domain_hash(url)
        file_hash = self._get_file_hash(content)
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # 基于内容类型选择存储目录
        base_dir = getattr(config, f"{content_type.upper()}_DIR")
        
        # 使用域名哈希创建子目录
        domain_dir = os.path.join(base_dir, domain_hash[:2], domain_hash[2:4])
        os.makedirs(domain_dir, exist_ok=True)
        
        # 生成文件名: {时间戳}_{文件哈希}.{扩展名}
        filename = f"{timestamp}_{file_hash}{ext}"
        return os.path.join(domain_dir, filename)
    
    async def save_content(self, url: str, content: bytes, 
                          content_type: str) -> Optional[str]:
        """保存内容并记录元数据"""
        try:
            # 确定文件扩展名
            ext_map = {
                "text": ".txt",
                "image": ".jpg",
                "screenshot": ".png",
                "video": ".mp4",
                "file": ".bin"
            }
            ext = ext_map.get(content_type, ".bin")
            
            # 生成存储路径
            filepath = self._generate_filepath(url, content_type, content, ext)
            
            # 保存文件
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb' if content_type != 'text' else 'w',
                     encoding='utf-8' if content_type == 'text' else None) as f:
                if content_type == 'text':
                    f.write(content.decode('utf-8'))
                else:
                    f.write(content)
            
            # 记录元数据
            meta = ContentMeta(
                url=url,
                domain=urlparse(url).netloc,
                domain_hash=self._get_domain_hash(url),
                content_type=content_type,
                file_path=filepath,
                timestamp=datetime.now().isoformat(),
                file_hash=self._get_file_hash(content)
            )
            
            self._save_metadata(meta)
            return filepath
            
        except Exception as e:
            logger.error(f"内容保存失败: {e}")
            return None
    
    def _save_metadata(self, meta: ContentMeta):
        """保存元数据到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                INSERT OR REPLACE INTO content_meta 
                (url, domain, domain_hash, content_type, file_path, timestamp, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (meta.url, meta.domain, meta.domain_hash, meta.content_type,
                     meta.file_path, meta.timestamp, meta.file_hash))
                conn.commit()
        except Exception as e:
            logger.error(f"元数据保存失败: {e}")