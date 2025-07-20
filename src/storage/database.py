# src/storage/database.py
"""
只负责SQLite元数据管理的模块
"""
import sqlite3
import hashlib
from datetime import datetime
from typing import Optional

from src.common.logger import logger
from src.common.data_models import ContentMeta

class DatabaseManager:
    """管理SQLite数据库中的元数据"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构，如果不存在"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS content_meta (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT,
                        content_type TEXT NOT NULL,
                        file_path TEXT UNIQUE NOT NULL,
                        file_hash TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON content_meta(file_hash)")
                conn.commit()
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise

    def save_metadata(self, metadata: ContentMeta) -> Optional[int]:
        """
        保存元数据到数据库，如果文件哈希已存在则跳过。
        返回数据库中的ID。
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM content_meta WHERE file_hash = ?", (metadata.file_hash,))
                existing = cursor.fetchone()
                if existing:
                    logger.warning(f"元数据已存在 (hash: {metadata.file_hash}), 跳过保存。ID: {existing[0]}")
                    return existing[0]

                cursor.execute("""
                    INSERT INTO content_meta (url, content_type, file_path, file_hash, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (metadata.url, metadata.content_type, metadata.file_path, metadata.file_hash, metadata.timestamp))
                content_id = cursor.lastrowid
                conn.commit()
                logger.info(f"元数据已保存, ID: {content_id}, Path: {metadata.file_path}")
                return content_id
        except sqlite3.IntegrityError:
             logger.warning(f"文件路径已存在，可能为并发冲突: {metadata.file_path}")
             return None
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
            return None

    def get_metadata_by_id(self, doc_id: int) -> Optional[ContentMeta]:
        """根据ID获取元数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM content_meta WHERE id = ?", (doc_id,))
                row = cursor.fetchone()
                if row:
                    return ContentMeta(**dict(row))
                return None
        except Exception as e:
            logger.error(f"获取元数据(ID: {doc_id})失败: {e}")
            return None

    @staticmethod
    def create_metadata_from_file(file_path: str, content_type: str) -> ContentMeta:
        """从文件创建元数据对象"""
        with open(file_path, 'rb') as f:
            content = f.read()
        file_hash = hashlib.sha256(content).hexdigest()

        return ContentMeta(
            content_type=content_type,
            file_path=file_path,
            file_hash=file_hash,
            timestamp=datetime.now().isoformat()
        )
