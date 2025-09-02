# src/etl/indexer.py
import os
import sys
from typing import List
from tqdm import tqdm
from src.common import config
from src.common.logger import logger
from src.storage.database import DatabaseManager
from src.storage.index import IndexManager
from src.services.embedding_service import EmbeddingService
from src.processing.text_processor import TextProcessor
from src.processing.image_processor import ImageProcessor

class Indexer:
    """负责将内容处理、向量化并存入数据库和搜索引擎。"""
    def __init__(self, db_manager: DatabaseManager, index_manager: IndexManager, embedding_service: EmbeddingService):
        self.db_manager = db_manager
        self.index_manager = index_manager
        self.embedding_service = embedding_service
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()

    def _prepare_items(self, file_paths: List[str], content_type: str) -> List[dict]:
        items_to_process = []
        for file_path in tqdm(file_paths, desc=f"预处理 {content_type} 文件"):
            try:
                item_info = {}
                data_for_model = None
                if content_type == 'image':
                    pil_image = self.image_processor.process(file_path)
                    if pil_image and pil_image.width * pil_image.height > 0:
                        item_info['pixels'] = pil_image.width * pil_image.height
                        data_for_model = pil_image
                elif content_type == 'text':
                    if os.path.getsize(file_path) < config.MIN_TEXT_LENGTH: continue
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw_text = f.read()
                    processed_text = self.text_processor.process(raw_text)
                    if len(processed_text) >= config.MIN_TEXT_LENGTH:
                        item_info['text_content'] = processed_text
                        data_for_model = processed_text
                
                if data_for_model:
                    metadata = self.db_manager.create_metadata_from_file(file_path, content_type)
                    items_to_process.append({'metadata': metadata, 'data_for_model': data_for_model, **item_info})
            except Exception as e:
                logger.error(f"预处理文件 '{file_path}' 失败: {e}")
        return items_to_process
    
    def run(self, file_paths: List[str], content_type: str):
        if not file_paths: return
        
        items_to_process = self._prepare_items(file_paths, content_type)
        if not items_to_process:
            logger.warning(f"没有有效的 {content_type} 文件可供处理。")
            return
        
        metadata_to_save = [item['metadata'] for item in items_to_process]
        hash_to_id_map = self.db_manager.save_metadata_batch(metadata_to_save)
        if not hash_to_id_map:
            logger.error("元数据批量保存失败，处理中止。")
            return

        valid_items_for_embedding = []
        for item in items_to_process:
            file_hash = item['metadata'].file_hash
            if file_hash in hash_to_id_map:
                item['metadata'].id = hash_to_id_map[file_hash]
                valid_items_for_embedding.append(item)
        
        if content_type == 'text': valid_items_for_embedding.sort(key=lambda x: len(x['text_content']), reverse=True)
        elif content_type == 'image': valid_items_for_embedding.sort(key=lambda x: x.get('pixels', 0), reverse=True)
        
        total_items = len(valid_items_for_embedding)
        processed_count = 0
        with tqdm(total=total_items, desc=f"动态批处理与索引 {content_type}", file=sys.stdout) as pbar:
            i = 0
            while i < total_items:
                start_index = i
                if content_type == 'text':
                    current_batch_tokens = 0
                    while i < total_items and (current_batch_tokens + len(valid_items_for_embedding[i]['data_for_model'])) <= config.MAX_TOKENS_PER_BATCH and (i - start_index) < config.MAX_ITEMS_PER_BATCH:
                        current_batch_tokens += len(valid_items_for_embedding[i]['data_for_model'])
                        i += 1
                elif content_type == 'image':
                    current_batch_pixels = 0
                    while i < total_items and (current_batch_pixels + valid_items_for_embedding[i]['pixels']) <= config.MAX_IMAGE_PIXELS and (i - start_index) < config.MAX_ITEMS_PER_BATCH:
                        current_batch_pixels += valid_items_for_embedding[i]['pixels']
                        i += 1
                if i == start_index: i += 1
                
                current_batch_items = valid_items_for_embedding[start_index:i]
                if not current_batch_items: continue
                
                batch_data = [item['data_for_model'] for item in current_batch_items]
                batch_vectors = []
                if content_type == 'text': batch_vectors = self.embedding_service.get_text_embeddings(batch_data)
                elif content_type == 'image': batch_vectors = self.embedding_service.get_image_embeddings(batch_data)
                
                if batch_vectors and len(batch_vectors) == len(current_batch_items):
                    success, _ = self.index_manager.index_batch(current_batch_items, batch_vectors)
                    processed_count += success
                else:
                    logger.warning(f"批次向量生成失败或数量不匹配，跳过。")
                
                pbar.update(len(current_batch_items))
        logger.info(f"索引流程完成。总共成功索引: {processed_count}/{total_items} 个 {content_type} 项目。")
