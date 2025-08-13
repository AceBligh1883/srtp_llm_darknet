# src/pipeline/processing.py
"""
核心处理流水线，负责编排处理流程
"""
import os
from typing import List, Tuple
from tqdm import tqdm
import multiprocessing
from src.common import config
from src.common.logger import logger
from src.storage.database import DatabaseManager
from src.storage.index import IndexManager
from src.features.embedding import EmbeddingGenerator
from src.features.text_processing import TextProcessor
from src.features.image_processing import ImageProcessor
    
class ProcessingPipeline:
    """编排一个文件的完整处理与索引流程"""

    def __init__(self):
        self.db_manager = DatabaseManager(config.DB_PATH)
        self.index_manager = IndexManager(self.db_manager)
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.embed_generator = EmbeddingGenerator()
        logger.info("处理流水线初始化完成。")

    def process_directory(self, directory: str, content_type: str):
        """处理指定目录下的所有文件"""
        if not os.path.exists(directory):
            logger.error(f"目录不存在: {directory}")
            return

        file_paths = []
        supported_formats = config.SUPPORTED_IMAGE_FORMATS if content_type == 'image' else ('.txt',)
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(supported_formats):
                    file_paths.append(os.path.join(root, filename))
        
        if not file_paths:
            logger.warning(f"在目录 {directory} 中未找到 {content_type} 类型的文件。")
            return
        logger.info(f"在 {os.path.basename(directory)} 目录发现 {len(file_paths)} 个 {content_type} 文件。")

        valid_items_for_embedding = []
        desc = f"预处理 {content_type} 并保存元数据"
        for file_path in tqdm(file_paths, desc=desc):
            try:
                data_to_embed = None
                if content_type == 'image':
                    data_to_embed = self.image_processor.process(file_path)
                elif content_type == 'text':
                    if os.path.getsize(file_path) < config.MIN_TEXT_LENGTH: continue
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw_text = f.read()
                    
                    if len(raw_text) > config.MAX_TEXT_CHARS:
                        raw_text = raw_text[:config.MAX_TEXT_CHARS]
                    
                    data_to_embed = self.text_processor.process(raw_text)
                    if len(data_to_embed) < config.MIN_TEXT_LENGTH: continue

                if data_to_embed:
                    metadata = self.db_manager.create_metadata_from_file(file_path, content_type)
                    doc_id = self.db_manager.save_metadata(metadata)
                    if doc_id:
                        valid_items_for_embedding.append({'doc_id': doc_id, 'data': data_to_embed})
            except Exception as e:
                logger.error(f"预处理文件 '{file_path}' 失败: {e}")

        if not valid_items_for_embedding:
            logger.warning("没有有效的文件可供处理。")
            return
        
        logger.info(f"开始为 {len(valid_items_for_embedding)} 个有效项目生成向量...")
        all_vectors = []

        if content_type == 'text':
            valid_items_for_embedding.sort(key=lambda item: len(item['data']), reverse=True)
            with tqdm(total=len(valid_items_for_embedding), desc=f"动态批处理 {content_type}") as pbar:
                i = 0
                while i < len(valid_items_for_embedding):
                    current_batch_items = []
                    current_batch_tokens = 0
                    while i < len(valid_items_for_embedding) and \
                          (current_batch_tokens + len(valid_items_for_embedding[i]['data'])) <= config.MAX_TOKENS_PER_BATCH:
                        item = valid_items_for_embedding[i]
                        current_batch_items.append(item)
                        current_batch_tokens += len(item['data'])
                        i += 1
                    
                    if not current_batch_items and i < len(valid_items_for_embedding):
                        current_batch_items.append(valid_items_for_embedding[i])
                        i += 1

                    if current_batch_items:
                        batch_data = [item['data'] for item in current_batch_items]
                        # 调用文本向量生成函数
                        batch_vectors = self.embed_generator.get_text_embeddings(batch_data, prompt_name="passage")
                        if batch_vectors: all_vectors.extend(batch_vectors)
                        else: all_vectors.extend([None] * len(current_batch_items))
                        pbar.update(len(current_batch_items))
        
        elif content_type == 'image':
            batch_size = config.PROCESSING_BATCH_SIZE
            with tqdm(total=len(valid_items_for_embedding), desc=f"批处理 {content_type}") as pbar:
                for i in range(0, len(valid_items_for_embedding), batch_size):
                    current_batch_items = valid_items_for_embedding[i : i + batch_size]
                    if not current_batch_items: continue

                    batch_data = [item['data'] for item in current_batch_items]
                    batch_vectors = self.embed_generator.get_image_embeddings(batch_data)
                    if batch_vectors: all_vectors.extend(batch_vectors)
                    else: all_vectors.extend([None] * len(current_batch_items))
                    pbar.update(len(current_batch_items))

        successful_items = [item for i, item in enumerate(valid_items_for_embedding) if all_vectors[i] is not None]
        vectors = [v for v in all_vectors if v is not None]
        
        if not vectors or len(vectors) != len(successful_items):
            logger.error("向量生成失败或数量不匹配，处理中止。")
            return
        
        success_count = 0
        desc_index = f"索引 {content_type} 向量到 Elasticsearch"
        for i, item in enumerate(tqdm(successful_items, desc=desc_index)):
            text_content_for_es = item.get('text_content') 
            if self.index_manager.index_document(item['doc_id'], vectors[i], text_content=text_content_for_es):
                success_count += 1
        
        logger.info(f"目录 {os.path.basename(directory)} 处理完成。成功索引: {success_count}/{len(successful_items)}")