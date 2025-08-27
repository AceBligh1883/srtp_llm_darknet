# src/storage/database.py
"""
只负责SQLite元数据管理的模块
"""
import os
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Set

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
                cursor.execute("PRAGMA table_info(content_meta)")
                columns = [row[1] for row in cursor.fetchall()]

                if 'kg_processed' not in columns:
                    cursor.execute("ALTER TABLE content_meta ADD COLUMN kg_processed INTEGER DEFAULT 0")

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON content_meta(file_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_processed ON content_meta(kg_processed)")
                conn.commit()
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def get_all_processed_hashes(self) -> Set[str]:
        """获取数据库中所有元数据的哈希值集合，用于快速去重。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT file_hash FROM content_meta")
                return {row[0] for row in cursor.fetchall()}
        except sqlite3.Error as e:
            logger.error(f"获取所有哈希值失败: {e}")
            return set()

    def save_metadata_batch(self, metadatas: List[ContentMeta]) -> Dict[str, int]:
        """
        批量保存元数据到数据库。
        返回一个从文件哈希到新生成的ID的映射。
        """
        if not metadatas:
            return {}
        
        CHUNK_SIZE = 900
        total_hash_to_id_map = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for i in range(0, len(metadatas), CHUNK_SIZE):
                    chunk = metadatas[i:i + CHUNK_SIZE]
                    
                    rows_to_insert = [
                        (m.url, m.content_type, m.file_path, m.file_hash, m.timestamp)
                        for m in chunk
                    ]
                    
                    insert_sql = "INSERT INTO content_meta (url, content_type, file_path, file_hash, timestamp) VALUES (?, ?, ?, ?, ?)"
                    cursor.executemany(insert_sql, rows_to_insert)
                    
                    hashes_to_query = [m.file_hash for m in chunk]
                    placeholders = ', '.join(['?'] * len(hashes_to_query))
                    select_sql = f"SELECT id, file_hash FROM content_meta WHERE file_hash IN ({placeholders})"
                    
                    cursor.execute(select_sql, hashes_to_query)
                    chunk_map = {row[1]: row[0] for row in cursor.fetchall()}
                    total_hash_to_id_map.update(chunk_map)

                conn.commit()
                logger.info(f"成功分块批量保存 {len(total_hash_to_id_map)} 条元数据。")
                return total_hash_to_id_map

        except sqlite3.Error as e:
            if 'conn' in locals() and conn: conn.rollback()
            logger.error(f"批量保存元数据到数据库失败: {e}")
            return {}
            
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

    def get_unprocessed_kg_files(self) -> list[str]:
        """
        从数据库中获取所有尚未进行知识图谱处理的文本文件的路径。

        Returns:
            list[str]: 未处理文件的路径列表。
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_path FROM content_meta 
                    WHERE content_type = 'text' AND (kg_processed = 0 OR kg_processed IS NULL)
                """)
                files = [row[0] for row in cursor.fetchall()]
                return files
        except Exception as e:
            logger.error(f"查询未处理的KG文件失败: {e}")
            return []

    def mark_kg_processed(self, file_path: str):
        """
        将指定文件在数据库中的状态标记为已进行知识图谱处理。

        Args:
            file_path (str): 要标记的文件的路径。
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE content_meta SET kg_processed = 1 WHERE file_path = ?",
                    (file_path,)
                )
                conn.commit()
                if cursor.rowcount > 0:
                    logger.debug(f"已将文件 '{os.path.basename(file_path)}' 标记为KG已处理。")
        except Exception as e:
            logger.error(f"标记文件 '{file_path}' 为KG已处理时失败: {e}")