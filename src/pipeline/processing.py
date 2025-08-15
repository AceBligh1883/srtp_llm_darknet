# src/pipeline/processing.py
"""
核心处理流水线，负责编排处理流程
"""
import os
import sys
from tqdm import tqdm
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
                data_for_model = None
                item_info = {}
                if content_type == 'image':
                    img = self.image_processor.process(file_path)
                    if img.size[0] < 32 or img.size[1] < 32: continue
                    item_info['pixels'] = img.width * img.height
                    data_for_model = file_path 

                elif content_type == 'text':
                    if os.path.getsize(file_path) < config.MIN_TEXT_LENGTH: continue
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw_text = f.read()
                    
                    if len(raw_text) > config.MAX_TEXT_CHARS:
                        raw_text = raw_text[:config.MAX_TEXT_CHARS]
                    
                    processed_text = self.text_processor.process(raw_text)
                    if len(processed_text) < config.MIN_TEXT_LENGTH: continue
                    data_for_model = processed_text
                    item_info['text_content'] = processed_text

                if data_for_model:
                    metadata = self.db_manager.create_metadata_from_file(file_path, content_type)
                    doc_id = self.db_manager.save_metadata(metadata)
                    
                    if doc_id and not self.index_manager.document_exists(doc_id):
                        item = {'doc_id': doc_id, 'data': data_for_model}
                        if 'pixels' in item_info:
                            item['pixels'] = item_info['pixels']
                        if 'text_content' in item_info:
                            item['text_content'] = item_info['text_content']
                        valid_items_for_embedding.append(item)
            except Exception as e:
                logger.error(f"预处理文件 '{file_path}' 失败: {e}")

        if not valid_items_for_embedding:
            logger.warning("没有有效的文件可供处理。")
            return
        
        if content_type == 'text':
            valid_items_for_embedding.sort(key=lambda x: len(x['text_content']), reverse=True)
        elif content_type == 'image':
            valid_items_for_embedding.sort(key=lambda x: x.get('pixels', 0), reverse=True)

        total_items = len(valid_items_for_embedding)
        processed_count = 0
        logger.info(f"开始为 {total_items} 个有效项目生成向量并索引...")

        with tqdm(total=total_items, desc=f"动态批处理与索引 {content_type}", file=sys.stdout) as pbar:
            i = 0
            while i < total_items:
                current_batch_items = []
                if content_type == 'text':
                    current_batch_tokens = 0
                    start_index = i
                    while i < total_items and \
                          (current_batch_tokens + len(valid_items_for_embedding[i]['data'])) <= config.MAX_TOKENS_PER_BATCH:
                        current_batch_tokens += len(valid_items_for_embedding[i]['data'])
                        i += 1
                    if i == start_index: i += 1 
                    current_batch_items = valid_items_for_embedding[start_index:i]
                
                if not current_batch_items: continue

                batch_data = [item['data'] for item in current_batch_items]
                batch_vectors = None
                if content_type == 'text':
                    batch_vectors = self.embed_generator.get_text_embeddings(batch_data, prompt_name="passage")
                elif content_type == 'image':
                    batch_vectors = self.embed_generator.get_image_embeddings(batch_data)

                if not batch_vectors or len(batch_vectors) != len(current_batch_items):
                    logger.warning(f"批次 {i//len(current_batch_items) if current_batch_items else 0} 向量生成失败或数量不匹配，跳过。")
                    pbar.update(len(current_batch_items))
                    continue

                success, _ = self.index_manager.index_batch(current_batch_items, batch_vectors)
                processed_count += success
                
                pbar.update(len(current_batch_items))

        logger.info(f"目录 {os.path.basename(directory)} 处理完成。总共成功索引: {processed_count}/{total_items} 个项目。")
